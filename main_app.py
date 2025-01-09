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
    
    data_format = display_format_selection()
    display_format_requirements(data_format)
    
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset', 
        type=['csv', 'txt']
    )
    
    if uploaded_file:
        # Process new data if we haven't already processed it
        if st.session_state.cleaned_df is None:
            df = load_and_validate_data(uploaded_file, data_format)
            if df is not None:
                format_type = 'lipidsearch' if data_format == 'LipidSearch 5.0' else 'generic'
                
                confirmed, name_df, experiment, bqc_label, valid_samples, updated_df = process_experiment(df, format_type)
                
                if confirmed and valid_samples:
                    # Use updated_df if available, otherwise use original df
                    df_to_clean = updated_df if updated_df is not None else df
                    cleaned_df, intsta_df = clean_data(df_to_clean, name_df, experiment, data_format)
                    
                    if cleaned_df is not None:
                        # Store essential data in session state
                        st.session_state.experiment = experiment
                        st.session_state.format_type = format_type
                        
                        # Display cleaned data and manage standards
                        display_cleaned_data(cleaned_df, intsta_df)
                        
                        # Handle normalization
                        normalized_df = handle_data_normalization(
                            cleaned_df, 
                            st.session_state.intsta_df,  # Use session state version for standards
                            experiment, 
                            format_type
                        )
                        
                        if normalized_df is not None:
                            st.session_state.normalized_df = normalized_df
                            
                elif not confirmed and valid_samples:
                    st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
        else:
            # Use existing data from session state
            display_cleaned_data(None, None)  # Will use session state values
            
            # Always show normalization options after cleaning if we have the required data
            if hasattr(st.session_state, 'experiment') and hasattr(st.session_state, 'format_type'):
                normalized_df = handle_data_normalization(
                    st.session_state.cleaned_df,
                    st.session_state.intsta_df,
                    st.session_state.experiment,
                    st.session_state.format_type
                )
                
                if normalized_df is not None:
                    st.session_state.normalized_df = normalized_df

def initialize_session_state():
    """Initialize the Streamlit session state."""
    # Data related states
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.session_state.intsta_df = None
    if 'normalized_df' not in st.session_state:
        st.session_state.normalized_df = None
        
    # Process control states
    if 'grouping_complete' not in st.session_state:
        st.session_state.grouping_complete = True
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

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
        * `MeanArea[sample_name]`: One column per sample with intensity values
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

def setup_experiment():
    """
    Set up experiment parameters through sidebar inputs.
    
    Returns:
        tuple: (experiment object, success status)
    """
    st.sidebar.subheader("Define Experiment")
    n_conditions = st.sidebar.number_input(
        'Enter the number of conditions', 
        min_value=1, 
        max_value=20, 
        value=1, 
        step=1
    )
    
    conditions_list = []
    number_of_samples_list = []
    
    for i in range(n_conditions):
        condition = st.sidebar.text_input(f'Create a label for condition #{i + 1}')
        n_samples = st.sidebar.number_input(
            f'Number of samples for condition #{i + 1}', 
            min_value=1, 
            max_value=1000, 
            value=1, 
            step=1
        )
        conditions_list.append(condition)
        number_of_samples_list.append(n_samples)

    experiment = lp.Experiment()
    success = experiment.setup_experiment(n_conditions, conditions_list, number_of_samples_list)
    
    if not success:
        st.sidebar.error("All condition labels must be non-empty.")
        return None, False
        
    return experiment, True

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
    """Handle manual sample grouping if needed."""    
    st.sidebar.write('Are your samples properly grouped together?')
    ans = st.sidebar.radio('', ['Yes', 'No'])
    
    if ans == 'No':
        st.session_state.grouping_complete = False
        selections = {}
        remaining_samples = group_df['sample name'].tolist()
        
        # Keep track of expected samples per condition
        expected_samples = dict(zip(experiment.conditions_list, 
                                  experiment.number_of_samples_list))
        
        for condition in experiment.conditions_list:
            st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")
            
            selected_samples = st.sidebar.multiselect(
                f'Pick the samples that belong to condition {condition}',
                remaining_samples
            )
            
            if len(selected_samples) > expected_samples[condition]:
                st.sidebar.error(f"Too many samples selected for {condition}. Please select exactly {expected_samples[condition]} samples.")
                return group_df, df
            
            selections[condition] = selected_samples
            
            if len(selected_samples) == expected_samples[condition]:
                remaining_samples = [s for s in remaining_samples if s not in selected_samples]
        
        if all(len(selections[condition]) == expected_samples[condition] 
               for condition in experiment.conditions_list):
            try:
                # Update the group_df with new sample ordering
                group_df = grouped_samples.group_samples(group_df, selections)
                
                # Create the new ordered sample list based on selections
                new_ordered_samples = []
                for condition in experiment.conditions_list:
                    new_ordered_samples.extend(selections[condition])
                
                # Create DataFrame copy with reordered intensity columns
                df_reordered = df.copy()
                
                # Add intensity columns in the new order
                prefix = 'MeanArea' if grouped_samples.data_format == 'lipidsearch' else 'Intensity'
                intensity_cols = [f"{prefix}[{sample}]" for sample in new_ordered_samples]
                
                # Get name_df for display
                name_df = grouped_samples.update_sample_names(group_df)
                
                st.sidebar.success('Sample grouping updated successfully!')
                
                # Display the new grouping information as a DataFrame
                st.sidebar.write("\nNew Sample Grouping:")
                st.sidebar.dataframe(name_df)
                
                st.session_state.grouping_complete = True
                
                return group_df, df_reordered
                
            except ValueError as e:
                st.sidebar.error(f"Error updating groups: {str(e)}")
                st.session_state.grouping_complete = False
        else:
            incorrect_conditions = [
                f"{cond} (selected {len(selections[cond])}/{expected_samples[cond]} samples)"
                for cond in experiment.conditions_list
                if len(selections[cond]) != expected_samples[cond]
            ]
            st.sidebar.error(
                f"Please select the correct number of samples for:\n"
                f"{', '.join(incorrect_conditions)}"
            )
            st.session_state.grouping_complete = False
    else:
        st.session_state.grouping_complete = True
    
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
    Clean data using appropriate cleaner based on format.
    
    Returns:
        tuple: (cleaned_df, intsta_df)
    """
    st.subheader("Clean, Filter and Normalize Data")
    
    if data_format == 'LipidSearch 5.0':
        cleaner = lp.CleanLipidSearchData()
    else:
        cleaner = lp.CleanGenericData()
        
    cleaned_df = cleaner.data_cleaner(df, name_df, experiment)
    cleaned_df, intsta_df = cleaner.extract_internal_standards(cleaned_df)
    
    return cleaned_df, intsta_df

def display_cleaned_data(cleaned_df, intsta_df):
    """
    Display cleaned data and manage internal standards with simplified workflow.
    """
    # Update session state with new data if provided
    if cleaned_df is not None:
        st.session_state.cleaned_df = cleaned_df.copy()
        st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()

    # Display cleaned data
    with st.expander("View Cleaned Data"):
        st.write(st.session_state.cleaned_df)
        csv = st.session_state.cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    # Simplified internal standards management
    with st.expander("Manage Internal Standards"):
        # First, check and display automatically detected standards
        if not st.session_state.intsta_df.empty:
            st.success(f"✓ Found {len(st.session_state.intsta_df)} automatically detected standards")
            st.write("Currently detected standards:")
            st.write(st.session_state.intsta_df)
            
            # Download detected standards button
            csv_intsta = st.session_state.intsta_df.to_csv(index=False)
            st.download_button(
                label="Download Current Standards",
                data=csv_intsta,
                file_name="current_standards.csv",
                mime="text/csv"
            )
        else:
            st.info("No internal standards were automatically detected in the dataset")

        # Always show option to upload custom standards
        st.markdown("---")  # Visual separator
        st.markdown("### Upload Custom Standards")
        st.info("""
        You can upload your own standards file. Requirements:
        - Excel file (.xlsx)
        - Must contain columns: 'LipidMolec' and 'ClassKey'
        - The standards must exist in your dataset
        
        Note: Uploading a standards file will replace any automatically detected standards.
        """)
        
        uploaded_file = st.file_uploader("Upload Excel file with standards", type=['xlsx'])
        if uploaded_file is not None:
            try:
                standards_df = pd.read_excel(uploaded_file)
                
                # Validate required columns
                required_cols = ['LipidMolec', 'ClassKey']
                if not all(col in standards_df.columns for col in required_cols):
                    st.error("Standards file must contain 'LipidMolec' and 'ClassKey' columns")
                    return

                # Validate standards exist in dataset
                valid_standards = standards_df['LipidMolec'].isin(st.session_state.cleaned_df['LipidMolec'])
                if not valid_standards.all():
                    invalid_standards = standards_df[~valid_standards]['LipidMolec'].tolist()
                    st.error(f"The following standards were not found in the dataset: {', '.join(invalid_standards)}")
                    return
                
                # Replace existing standards with uploaded ones
                st.session_state.intsta_df = standards_df.copy()
                st.success(f"✓ Successfully loaded {len(standards_df)} custom standards (previous standards were replaced)")
                
                # Display and allow download of standards
                st.write("Loaded custom standards:")
                st.write(standards_df)
                csv_standards = standards_df.to_csv(index=False)
                st.download_button(
                    label="Download Current Standards",
                    data=csv_standards,
                    file_name="current_standards.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error reading standards file: {str(e)}")
                return
            
def handle_data_normalization(cleaned_df, intsta_df, experiment, format_type):
    """
    Handle data normalization based on selected standards and normalization methods.
    
    Args:
        cleaned_df (pd.DataFrame): The cleaned dataset
        intsta_df (pd.DataFrame): DataFrame containing internal standards
        experiment (Experiment): Experiment object containing sample information
        format_type (str): The format type of the data
        
    Returns:
        pd.DataFrame: Normalized DataFrame or None if normalization fails
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
            # Handle BCA Assay normalization
            if normalization_method in ['BCA Assay', 'Both']:
                with st.expander("Enter BCA Assay Data"):
                    protein_df = collect_protein_concentrations(experiment)
                    if protein_df is not None:
                        try:
                            normalized_df = normalized_data_object.normalize_using_bca(normalized_df, protein_df)
                            st.success("BCA normalization applied successfully")
                        except Exception as e:
                            st.error(f"Error during BCA normalization: {str(e)}")
                            return None

            # Inside handle_data_normalization function, replace the internal standards section with:
            if has_standards and normalization_method in ['Internal Standards', 'Both']:
                with st.expander("Enter Inputs For Data Normalization Using Internal Standards"):
                    # Group standards by their ClassKey
                    standards_by_class = {}
                    if 'ClassKey' in st.session_state.intsta_df.columns:
                        standards_by_class = st.session_state.intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
            
                    # Dictionary to store class-standard mapping
                    class_to_standard_map = {}
            
                    # Process each selected class
                    for lipid_class in selected_classes:
                        # Get available standards for this class
                        available_standards = standards_by_class.get(
                            lipid_class,
                            list(st.session_state.intsta_df['LipidMolec'].unique())
                        )
                        
                        if not available_standards:
                            st.error(f"No standards available for class '{lipid_class}'")
                            continue
                            
                        # Default to first standard in list
                        default_idx = 0
                        selected_standard = st.selectbox(
                            f'Select internal standard for {lipid_class}',
                            available_standards,
                            index=default_idx
                        )
                        
                        class_to_standard_map[lipid_class] = selected_standard
            
                    # Only proceed if we have mapped all classes to standards
                    if len(class_to_standard_map) == len(selected_classes):
                        # Create ordered list of standards matching selected_classes order
                        added_intsta_species_lst = [class_to_standard_map[cls] for cls in selected_classes]
                        
                        # Get unique standards that were selected
                        selected_standards = set(added_intsta_species_lst)
                        
                        # Collect concentrations for selected standards
                        st.write("Enter the concentration of each selected internal standard (µM):")
                        intsta_concentration_dict = {}
                        all_concentrations_entered = True
                        
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
            
                        if all_concentrations_entered:
                            try:
                                # Apply the normalization
                                normalized_df = normalized_data_object.normalize_data(
                                    selected_classes,
                                    added_intsta_species_lst,
                                    intsta_concentration_dict,
                                    normalized_df,
                                    st.session_state.intsta_df,
                                    experiment
                                )
                                st.success("Internal standards normalization applied successfully")
                            except Exception as e:
                                st.error(f"Error during internal standards normalization: {str(e)}")
                                return None
                        else:
                            st.error("Please enter valid concentrations for all standards")
                            return None
                    else:
                        st.error("Please select standards for all lipid classes to proceed with normalization")
                        return None

        # Ensure column names are properly renamed
        normalized_df = normalized_data_object._rename_intensity_columns(normalized_df)

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