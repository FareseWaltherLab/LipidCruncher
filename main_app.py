import streamlit as st
import pandas as pd
import numpy as np
import lipidomics as lp

def main():
    """
    Main function for the Lipidomics Analysis Module Streamlit application.
    """
    st.header("Lipidomics Analysis Module")
    st.markdown("Process and clean lipidomics data from multiple sources.")
    
    initialize_session_state()
    
    # Always show format selection in sidebar
    data_format = display_format_selection()
    display_format_requirements(data_format)
    
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset', 
        type=['csv', 'txt']
    )
    
    if uploaded_file:
        df = load_and_validate_data(uploaded_file, data_format)
        if df is not None:
            format_type = 'lipidsearch' if data_format == 'LipidSearch 5.0' else 'generic'
            
            # Always process experiment setup regardless of confirmation
            confirmed, name_df, experiment, bqc_label, valid_samples, updated_df = process_experiment(df, format_type)
            
            # Update confirmation state
            st.session_state.confirmed = confirmed
            
            if valid_samples:
                # Only proceed with data processing if confirmed
                if confirmed:
                    # Use updated_df if available, otherwise use original df
                    df_to_clean = updated_df if updated_df is not None else df
                    cleaned_df, intsta_df = clean_data(df_to_clean, name_df, experiment, data_format)
                    
                    if cleaned_df is not None:
                        # Store essential data in session state
                        st.session_state.experiment = experiment
                        st.session_state.format_type = format_type
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.intsta_df = intsta_df
                        
                        # Display cleaned data and manage standards
                        display_cleaned_data(cleaned_df, intsta_df)
                        
                        # Handle normalization
                        normalized_df = handle_data_normalization(
                            cleaned_df, 
                            st.session_state.intsta_df,
                            experiment, 
                            format_type
                        )
                        
                        if normalized_df is not None:
                            st.session_state.normalized_df = normalized_df
                
                else:
                    # Clear processed data when unconfirmed
                    st.session_state.cleaned_df = None
                    st.session_state.intsta_df = None
                    st.session_state.normalized_df = None
                    st.session_state.experiment = None
                    st.session_state.format_type = None
                    st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
            
            else:
                st.error("Please ensure your samples are valid before proceeding.")
                
def initialize_session_state():
    """Initialize the Streamlit session state with default values."""
    # Data related states
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.session_state.intsta_df = None
    if 'normalized_df' not in st.session_state:
        st.session_state.normalized_df = None
    if 'original_column_order' not in st.session_state:
        st.session_state.original_column_order = None
    
    # Process control states
    if 'grouping_complete' not in st.session_state:
        st.session_state.grouping_complete = True
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    if 'confirmed' not in st.session_state:
        st.session_state.confirmed = False
    if 'experiment' not in st.session_state:
        st.session_state.experiment = None
    if 'format_type' not in st.session_state:
        st.session_state.format_type = None

def display_format_selection():
    """Display data format selection in sidebar."""
    return st.sidebar.selectbox(
        'Select Data Format',
        ['LipidSearch 5.0', 'Generic Format']
    )

def display_format_requirements(data_format):
    """Display format-specific requirements."""
    if data_format == 'LipidSearch 5.0':
        st.info("""
        **Dataset Requirements for LipidSearch 5.0 Module**
        
        Required columns:
        * `LipidMolec`: The molecule identifier for the lipid
        * `ClassKey`: The classification key for the lipid type
        * `CalcMass`: The calculated mass of the lipid molecule
        * `BaseRt`: The base retention time
        * `TotalGrade`: The overall quality grade of the lipid data
        * `TotalSmpIDRate(%)`: The total sample identification rate
        * `FAKey`: The fatty acid key associated with the lipid
        
        Additionally, each sample in your dataset must have a corresponding MeanArea column to represent intensity values. 
        For instance, if your dataset comprises 10 samples, you should have the following columns: 
        MeanArea[s1], MeanArea[s2], ..., MeanArea[s10] for each respective sample intensity.
        """)
    else:
        st.info("""
        **Dataset Requirements for Generic Format**
        
        Required columns:
        * `LipidMolec`: The molecule identifier for the lipid
        * `intensity[sample_name]`: One column per sample with intensity values
        """)

def load_and_validate_data(uploaded_file, data_format):
    """
    Load and validate uploaded data file.
    
    Returns:
        pd.DataFrame or None: Validated dataframe or None if validation fails
    """
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        df, success, message = lp.DataFormatHandler.validate_and_preprocess(
            df,
            'lipidsearch' if data_format == 'LipidSearch 5.0' else 'generic'
        )
        
        if not success:
            st.error(message)
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_group_samples(df, experiment, data_format):
    """
    Process and validate sample grouping.
    
    Args:
        df (pd.DataFrame): The dataset to process
        experiment (Experiment): The experiment setup
        data_format (str): Either 'lipidsearch' or 'generic'
    
    Returns:
        tuple: (name_df, group_df, valid_samples)
    """
    grouped_samples = lp.GroupSamples(experiment, data_format)
    
    if not grouped_samples.check_dataset_validity(df):
        st.sidebar.error("Invalid dataset format!")
        return None, None, None, False

    value_cols = grouped_samples.build_mean_area_col_list(df)
    if len(value_cols) != len(experiment.full_samples_list):
        st.sidebar.error("Number of samples in data doesn't match experiment setup!")
        return None, None, None, False

    st.sidebar.subheader('Group Samples')
    group_df = grouped_samples.build_group_df(df)
    st.sidebar.write(group_df)

    # Now handle_manual_grouping returns both group_df and updated df
    group_df, updated_df = handle_manual_grouping(group_df, experiment, grouped_samples, df)
    name_df = grouped_samples.update_sample_names(group_df)
    
    return name_df, group_df, updated_df, True

def handle_manual_grouping(group_df, experiment, grouped_samples, df):
    """
    Handle manual sample grouping if needed.
    
    Args:
        group_df (pd.DataFrame): DataFrame containing sample group information
        experiment (Experiment): Experiment object containing setup information
        grouped_samples (GroupSamples): GroupSamples object for handling sample grouping
        df (pd.DataFrame): Main data DataFrame
        
    Returns:
        tuple: (Updated group_df, Updated DataFrame)
    """
    st.sidebar.write('Are your samples properly grouped together?')
    ans = st.sidebar.radio('', ['Yes', 'No'])
    
    if ans == 'No':
        # Store original column order if not already stored
        if st.session_state.original_column_order is None:
            st.session_state.original_column_order = df.columns.tolist()
            
        st.session_state.grouping_complete = False
        selections = {}
        remaining_samples = group_df['sample name'].tolist()
        
        # Keep track of expected samples per condition
        expected_samples = dict(zip(experiment.conditions_list, 
                                  experiment.number_of_samples_list))
        
        # Process each condition
        for condition in experiment.conditions_list:
            st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")
            
            selected_samples = st.sidebar.multiselect(
                f'Pick the samples that belong to condition {condition}',
                remaining_samples
            )
            
            selections[condition] = selected_samples
            
            # Update remaining samples only if correct number selected
            if len(selected_samples) == expected_samples[condition]:
                remaining_samples = [s for s in remaining_samples if s not in selected_samples]
        
        # Verify all conditions have correct number of samples
        all_correct = all(len(selections[condition]) == expected_samples[condition] 
                         for condition in experiment.conditions_list)
        
        if all_correct:
            try:
                # Update the group_df and get column mapping
                group_df, old_to_new = grouped_samples.group_samples(group_df, selections)
                
                # Reorder and rename columns in the DataFrame
                df_reordered = grouped_samples.reorder_intensity_columns(df, old_to_new)
                
                st.session_state.grouping_complete = True
                
                # Generate and display name_df to show the new sample order
                name_df = grouped_samples.update_sample_names(group_df)
                st.sidebar.write("New sample order after regrouping:")
                st.sidebar.write(name_df)
                
                return group_df, df_reordered
                
            except ValueError as e:
                st.sidebar.error(f"Error updating groups: {str(e)}")
                st.session_state.grouping_complete = False
                return group_df, df
        else:
            st.session_state.grouping_complete = False
            return group_df, df
    else:
        st.session_state.grouping_complete = True
        if st.session_state.original_column_order is not None:
            df = df.reindex(columns=st.session_state.original_column_order)
    
    return group_df, df

def process_experiment(df, data_format='lipidsearch'):
    """
    Process the experiment setup and sample grouping based on user input.
    """
    st.sidebar.subheader("Define Experiment")
    n_conditions = st.sidebar.number_input('Enter the number of conditions', 
                                         min_value=1, max_value=20, value=1, step=1)
    
    conditions_list = [st.sidebar.text_input(f'Create a label for condition #{i + 1}') 
                      for i in range(n_conditions)]
    
    number_of_samples_list = [st.sidebar.number_input(f'Number of samples for condition #{i + 1}', 
                                                     min_value=1, max_value=1000, value=1, step=1) 
                             for i in range(n_conditions)]

    experiment = lp.Experiment()
    if not experiment.setup_experiment(n_conditions, conditions_list, number_of_samples_list):
        st.sidebar.error("All condition labels must be non-empty.")
        return False, None, None, None, False, None

    name_df, group_df, updated_df, valid_samples = process_group_samples(df, experiment, data_format)
    if not valid_samples:
        return False, None, None, None, False, None

    # Only show BQC selection and confirmation if grouping is complete
    if st.session_state.grouping_complete:
        bqc_label = specify_bqc_samples(experiment)
        confirmed = confirm_user_inputs(group_df, experiment)
    else:
        bqc_label = None
        confirmed = False
        st.sidebar.error("Please complete sample grouping before proceeding.")

    return confirmed, name_df, experiment, bqc_label, valid_samples, updated_df

def specify_bqc_samples(experiment):
    """
    Queries the user through the Streamlit sidebar to specify if Batch Quality Control (BQC) samples exist
    within their dataset and, if so, to identify the label associated with these samples.
    """
    st.sidebar.subheader("Specify Label of BQC Samples")
    bqc_ans = st.sidebar.radio('Do you have Batch Quality Control (BQC) samples?', ['Yes', 'No'], 1)
    bqc_label = None
    if bqc_ans == 'Yes':
        conditions_with_two_plus_samples = [
            condition for condition, number_of_samples in zip(experiment.conditions_list, experiment.number_of_samples_list)
            if number_of_samples > 1
        ]
        bqc_label = st.sidebar.radio('Which label corresponds to BQC samples?', conditions_with_two_plus_samples, 0)
    return bqc_label

def confirm_user_inputs(group_df, experiment):
    """
    Confirm user inputs before proceeding to the next step in the app.
    """
    try:
        st.sidebar.subheader("Confirm Inputs")

        # Display total number of samples
        st.sidebar.write("There are a total of " + str(sum(experiment.number_of_samples_list)) + " samples.")

        # Display information about sample and condition pairings
        for condition in experiment.conditions_list:
            build_replicate_condition_pair(condition, experiment)

        # Display a checkbox for the user to confirm the inputs
        return st.sidebar.checkbox("Confirm the inputs by checking this box")
    except Exception as e:
        raise Exception(f"Error in confirming user inputs: {e}")

def build_replicate_condition_pair(condition, experiment):
    """
    Display information about sample and condition pairings in the Streamlit sidebar.
    """
    index = experiment.conditions_list.index(condition)
    samples = experiment.individual_samples_list[index]

    display_text = f"- {' to '.join([samples[0], samples[-1]])} (total {len(samples)}) correspond to {condition}" if len(samples) > 5 else f"- {'-'.join(samples)} correspond to {condition}"
    st.sidebar.write(display_text)

def clean_data(df, name_df, experiment, data_format):
    """
    Clean data using appropriate cleaner based on the format.
    """
    if data_format == 'LipidSearch 5.0':
        cleaner = lp.CleanLipidSearchData()
    else:
        cleaner = lp.CleanGenericData()
        
    cleaned_df = cleaner.data_cleaner(df, name_df, experiment)
    cleaned_df, intsta_df = cleaner.extract_internal_standards(cleaned_df)
    
    return cleaned_df, intsta_df


def handle_standards_upload(normalizer):
    """Handle the upload and processing of custom standards file."""
    st.info("""
    You can upload your own standards file. Requirements:
    - Excel file (.xlsx)
    - Must contain column: 'LipidMolec'
    - The standards must exist in your dataset
    
    """)
    
    uploaded_file = st.file_uploader("Upload Excel file with standards", type=['xlsx'])
    if uploaded_file is not None:
        try:
            standards_df = pd.read_excel(uploaded_file)
            return normalizer.process_standards_file(standards_df, st.session_state.cleaned_df)
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error reading standards file: {str(e)}")
    return None

def manage_internal_standards(normalizer):
    """Handle internal standards management workflow."""
    standards_source = st.radio(
        "Select standards source:",
        ["Automatic Detection", "Upload Custom Standards"],
        index=0
    )
    
    if standards_source == "Automatic Detection":
        if not st.session_state.intsta_df.empty:
            st.success(f"✓ Found {len(st.session_state.intsta_df)} standards")
            st.write("Current standards:")
            display_data(st.session_state.intsta_df, "Current Standards", "current_standards.csv")
        else:
            st.info("No internal standards were automatically detected in the dataset")
    else:
        new_standards_df = handle_standards_upload(normalizer)
        if new_standards_df is not None:
            st.session_state.intsta_df = new_standards_df
            st.success(f"✓ Successfully loaded {len(new_standards_df)} custom standards")
            st.write("Loaded custom standards:")
            display_data(new_standards_df, "Current Standards", "current_standards.csv")

def display_cleaned_data(cleaned_df, intsta_df):
    """
    Display cleaned data and manage internal standards with simplified workflow.
    """
    # Update session state with new data if provided
    if cleaned_df is not None:
        st.session_state.cleaned_df = cleaned_df.copy()
        st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()

    # Create normalizer instance
    normalizer = lp.NormalizeData()

    # Display cleaned data
    with st.expander("View Cleaned Data"):
        display_data(st.session_state.cleaned_df, "Cleaned Data", "cleaned_data.csv")

    # Internal standards management
    with st.expander("Manage Internal Standards"):
        manage_internal_standards(normalizer)
            
def handle_data_normalization(cleaned_df, intsta_df, experiment, format_type):
    """
    Handle data normalization based on selected standards and normalization methods.
    """
    st.subheader("Data Normalization")
    
    # Validate required columns
    if 'ClassKey' not in cleaned_df.columns:
        st.error("ClassKey column is required for normalization. Please ensure your data includes lipid class information.")
        return None

    # Get the full list of classes
    all_class_lst = list(cleaned_df['ClassKey'].unique())
    
    # Class selection
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        all_class_lst,
        default=all_class_lst
    )

    if not selected_classes:
        st.warning("Please select at least one lipid class to proceed with normalization.")
        return None

    # Filter DataFrame based on selected classes
    filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()

    # Check if we have standards from earlier selection
    has_standards = not st.session_state.intsta_df.empty

    # Determine available normalization options based on standards availability
    if has_standards:
        normalization_options = ['None', 'Internal Standards', 'BCA Assay', 'Both']
    else:
        normalization_options = ['None', 'BCA Assay']
        st.warning("No internal standards available. Only BCA normalization is available.")

    normalization_method = st.radio(
        "Select normalization method:",
        options=normalization_options
    )

    normalized_df = filtered_df.copy()
    normalized_data_object = lp.NormalizeData()

    try:
        # Apply normalizations based on selection
        if normalization_method != 'None':
            # Track whether we need to do standards normalization
            do_standards = normalization_method in ['Internal Standards', 'Both'] and has_standards
            
            # Handle BCA Assay normalization first if selected
            if normalization_method in ['BCA Assay', 'Both']:
                with st.expander("Enter BCA Assay Data"):
                    protein_df = collect_protein_concentrations(experiment)
                    if protein_df is not None:
                        try:
                            # Apply BCA normalization but keep intensity column format
                            normalized_df = normalized_data_object.normalize_using_bca(normalized_df, protein_df, preserve_prefix=True)
                            st.success("BCA normalization applied successfully")
                        except Exception as e:
                            st.error(f"Error during BCA normalization: {str(e)}")
                            return None

            # Then handle internal standards normalization if selected
            if do_standards:
                with st.expander("Enter Inputs For Data Normalization Using Internal Standards"):
                    # Ensure intsta_df has standardized column names
                    intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
                    if not intensity_cols:
                        st.error("Internal standards data does not contain properly formatted intensity columns")
                        return None

                    # Group standards by their ClassKey
                    standards_by_class = {}
                    if 'ClassKey' in st.session_state.intsta_df.columns:
                        standards_by_class = st.session_state.intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
            
                    # Process each selected class with mapped standards
                    class_to_standard_map = process_class_standards(selected_classes, standards_by_class, st.session_state.intsta_df)
                    if not class_to_standard_map:
                        return None

                    # Get concentrations and apply normalization
                    normalized_df = apply_standards_normalization(
                        normalized_df, 
                        class_to_standard_map, 
                        selected_classes, 
                        st.session_state.intsta_df, 
                        normalized_data_object,
                        experiment
                    )
                    if normalized_df is None:
                        return None

            # Finally rename all intensity columns to concentration
            normalized_df = rename_intensity_to_concentration(normalized_df)

        # Display the normalized data
        if normalized_df is not None:
            st.subheader("View Normalized Dataset")
            st.write(normalized_df)
            csv = normalized_df.to_csv(index=False)
            st.download_button(
                label="Download Normalized Data",
                data=csv,
                file_name="normalized_data.csv",
                mime="text/csv"
            )

        return normalized_df

    except Exception as e:
        st.error(f"An unexpected error occurred during normalization: {str(e)}")
        return None

def rename_intensity_to_concentration(df):
    """Renames intensity columns to concentration columns at the end of normalization"""
    df = df.copy()
    rename_dict = {
        col: col.replace('intensity[', 'concentration[')
        for col in df.columns if col.startswith('intensity[')
    }
    return df.rename(columns=rename_dict)

def process_class_standards(selected_classes, standards_by_class, intsta_df):
    """
    Helper function to process class-standard mapping with flexible standard selection.
    
    Args:
        selected_classes (list): List of selected lipid classes
        standards_by_class (dict): Dictionary mapping classes to their standards
        intsta_df (pd.DataFrame): DataFrame containing internal standards
    
    Returns:
        dict: Mapping of lipid classes to selected standards
    """
    class_to_standard_map = {}
    all_available_standards = list(intsta_df['LipidMolec'].unique())
    
    for lipid_class in selected_classes:
        # Get class-specific standards if available
        class_specific_standards = standards_by_class.get(lipid_class, [])
        
        # Set default index based on class-specific standard
        default_idx = 0
        if class_specific_standards:
            # If we have a class-specific standard, find its index in all_available_standards
            try:
                default_idx = all_available_standards.index(class_specific_standards[0])
            except ValueError:
                default_idx = 0
        
        selected_standard = st.selectbox(
            f'Select internal standard for {lipid_class}' + 
            (' (default shown)' if class_specific_standards else ''),
            all_available_standards,
            index=default_idx
        )
        
        class_to_standard_map[lipid_class] = selected_standard

    if len(class_to_standard_map) != len(selected_classes):
        st.error("Please select standards for all lipid classes to proceed with normalization")
        return None
        
    return class_to_standard_map

def apply_standards_normalization(df, class_to_standard_map, selected_classes, intsta_df, normalizer, experiment):
    """Helper function to apply standards normalization"""
    # Create ordered list of standards matching selected_classes order
    added_intsta_species_lst = [class_to_standard_map[cls] for cls in selected_classes]
    
    # Get unique standards and their concentrations
    selected_standards = set(added_intsta_species_lst)
    intsta_concentration_dict = {}
    all_concentrations_entered = True
    
    st.write("Enter the concentration of each selected internal standard (µM):")
    for standard in selected_standards:
        concentration = st.number_input(
            f"Concentration (µM) for {standard}",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"conc_{standard}"
        )
        if concentration <= 0:
            st.error(f"Please enter a valid concentration for {standard}")
            all_concentrations_entered = False
        intsta_concentration_dict[standard] = concentration

    if not all_concentrations_entered:
        st.error("Please enter valid concentrations for all standards")
        return None

    try:
        normalized_df = normalizer.normalize_data(
            selected_classes,
            added_intsta_species_lst,
            intsta_concentration_dict,
            df,
            intsta_df,
            experiment
        )
        st.success("Internal standards normalization applied successfully")
        return normalized_df
    except Exception as e:
        st.error(f"Error during internal standards normalization: {str(e)}")
        return None

def collect_protein_concentrations(experiment):
    """
    Collects protein concentrations for each sample using Streamlit's UI.
    Args:
        experiment (Experiment): The experiment object containing the list of sample names.
    Returns:
        pd.DataFrame: A DataFrame with 'Sample' and 'Concentration' columns
    """
    method = st.radio(
        "Select the method for providing protein concentrations:",
        ["Manual Input", "Upload Excel File"],
        index=1
    )
    
    if method == "Manual Input":
        protein_concentrations = {}
        for sample in experiment.full_samples_list:
            concentration = st.number_input(
                f'Enter protein concentration for {sample} (mg/mL):',
                min_value=0.0, max_value=100000.0, 
                value=1.0, 
                step=0.1,
                key=sample
            )
            protein_concentrations[sample] = concentration

        protein_df = pd.DataFrame(list(protein_concentrations.items()), 
                                columns=['Sample', 'Concentration'])
        return protein_df
    
    elif method == "Upload Excel File":
        st.info("Upload an Excel file with a single column named 'Concentration'. Each row should correspond to the protein concentration for each sample in the experiment, in order.")
        
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                # Read the Excel file
                protein_df = pd.read_excel(uploaded_file, engine='openpyxl')
                
                # Basic validations
                if 'Concentration' not in protein_df.columns:
                    st.error("Excel file must contain a single column named 'Concentration'")
                    st.write("Found columns:", list(protein_df.columns))
                    return None
                    
                if len(protein_df) != len(experiment.full_samples_list):
                    st.error(f"Number of concentrations ({len(protein_df)}) does not match "
                            f"number of samples ({len(experiment.full_samples_list)})")
                    return None
                
                # Convert concentration values to numeric
                protein_df['Concentration'] = pd.to_numeric(protein_df['Concentration'], errors='coerce')
                if protein_df['Concentration'].isna().any():
                    st.error("Some concentration values could not be converted to numbers")
                    return None
                
                # Add sample names in the correct order
                protein_df['Sample'] = experiment.full_samples_list
                
                return protein_df
                
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                return None
        else:
            st.warning("Please upload an Excel file to proceed.")
            return None
        
def display_data(df, title, filename):
    """
    Display a DataFrame in an expander with download option.
    
    Parameters:
        df (pd.DataFrame): DataFrame to display
        title (str): Title for the data section
        filename (str): Name of file for download
    """
    st.write(df)
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {title}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

if __name__ == "__main__":
    main()