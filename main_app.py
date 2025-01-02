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
        # Only process new data if we haven't already or if the standards haven't been modified
        if (st.session_state.cleaned_df is None or 
            not st.session_state.standards_modified):
            
            df = load_and_validate_data(uploaded_file, data_format)
            if df is not None:
                format_type = 'lipidsearch' if data_format == 'LipidSearch 5.0' else 'generic'
                
                confirmed, name_df, experiment, bqc_label, valid_samples = process_experiment(df, format_type)
                
                if confirmed and valid_samples:
                    cleaned_df, intsta_df = clean_data(df, name_df, experiment, data_format)
                    if cleaned_df is not None:
                        display_cleaned_data(cleaned_df, intsta_df)
                elif not confirmed and valid_samples:
                    st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
        else:
            # Use existing data from session state
            display_cleaned_data(None, None)  # Will use session state values

def initialize_session_state():
    """Initialize the Streamlit session state."""
    # Only initialize if they don't exist at all
    if 'cleaned_df' not in st.session_state:
        st.write("DEBUG - Initializing cleaned_df in session state")
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.write("DEBUG - Initializing intsta_df in session state")
        st.session_state.intsta_df = None
    if 'standards_modified' not in st.session_state:
        st.write("DEBUG - Initializing standards_modified flag")
        st.session_state.standards_modified = False
    if 'initialized' not in st.session_state:
        st.write("DEBUG - Setting initial state flag")
        st.session_state.initialized = True
    if 'last_downloaded_state' not in st.session_state:
        st.write("DEBUG - Initializing download state tracker")
        st.session_state.last_downloaded_state = None

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
        tuple: (name_df, group_df, success status)
    """
    grouped_samples = lp.GroupSamples(experiment, data_format)
    
    if not grouped_samples.check_dataset_validity(df):
        st.sidebar.error("Invalid dataset format!")
        return None, None, False

    value_cols = grouped_samples.build_mean_area_col_list(df)
    if len(value_cols) != len(experiment.full_samples_list):
        st.sidebar.error("Number of samples in data doesn't match experiment setup!")
        return None, None, False

    st.sidebar.subheader('Group Samples')
    group_df = grouped_samples.build_group_df(df)
    st.sidebar.write(group_df)

    group_df = handle_manual_grouping(group_df, experiment, grouped_samples)
    name_df = grouped_samples.update_sample_names(group_df)
    
    return name_df, group_df, True

def handle_manual_grouping(group_df, experiment, grouped_samples):
    """Handle manual sample grouping if needed."""
    st.sidebar.write('Are your samples properly grouped together?')
    ans = st.sidebar.radio('', ['Yes', 'No'])
    
    if ans == 'No':
        selections = {}
        remaining_samples = group_df['sample name'].tolist()
        
        for condition in experiment.conditions_list:
            selected_samples = st.sidebar.multiselect(
                f'Pick the samples that belong to condition {condition}', 
                remaining_samples
            )
            selections[condition] = selected_samples
            remaining_samples = [s for s in remaining_samples if s not in selected_samples]
            
        group_df = grouped_samples.group_samples(group_df, selections)
        st.sidebar.write('Updated sample grouping:')
        st.sidebar.write(group_df)
    
    return group_df

def process_experiment(df, data_format='lipidsearch'):
    """
    Process the experiment setup and sample grouping based on user input.

    Args:
        df (pd.DataFrame): The dataset to be processed
        data_format (str): The data format ('lipidsearch' or 'generic')

    Returns:
        confirmed (bool): Whether the user has confirmed the inputs.
        name_df (DataFrame): DataFrame containing name mappings.
        experiment (Experiment): The configured Experiment object.
        bqc_label (str or None): Label used for BQC samples, if any.
        valid_samples (bool): Indicates whether sample validation was successful.
    """
    st.sidebar.subheader("Define Experiment")
    n_conditions = st.sidebar.number_input('Enter the number of conditions', min_value=1, max_value=20, value=1, step=1)
    conditions_list = [st.sidebar.text_input(f'Create a label for condition #{i + 1}') for i in range(n_conditions)]
    number_of_samples_list = [st.sidebar.number_input(f'Number of samples for condition #{i + 1}', min_value=1, max_value=1000, value=1, step=1) for i in range(n_conditions)]

    experiment = lp.Experiment()
    if not experiment.setup_experiment(n_conditions, conditions_list, number_of_samples_list):
        st.sidebar.error("All condition labels must be non-empty.")
        return False, None, None, None, False

    name_df, group_df, valid_samples = process_group_samples(df, experiment, data_format)
    if not valid_samples:
        return False, None, None, None, False

    bqc_label = specify_bqc_samples(experiment)
    confirmed = confirm_user_inputs(group_df, experiment)

    return confirmed, name_df, experiment, bqc_label, valid_samples

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
    Display cleaned data and internal standards data in expanders with download options.
    """
        
    # Initialize button click tracker in session state
    if 'remove_button_clicked' not in st.session_state:
        st.session_state.remove_button_clicked = False
        
    # Only update session state if we're passed new data
    if cleaned_df is not None and intsta_df is not None:
        st.session_state.cleaned_df = cleaned_df.copy()
        st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()
        st.session_state.standards_modified = False
        st.session_state.initialized = True
        
    # Debug input state
    st.write("DEBUG - Input State:")
    st.write("- cleaned_df is None:", cleaned_df is None)
    st.write("- intsta_df is None:", intsta_df is None)
    if cleaned_df is not None:
        st.write("- cleaned_df shape:", cleaned_df.shape)
    if intsta_df is not None:
        st.write("- intsta_df shape:", intsta_df.shape)

    # Handle session state updates
    if cleaned_df is not None:
        if (st.session_state.cleaned_df is None or 
            st.session_state.standards_modified or 
            'initialized' not in st.session_state):
            
            st.write("\nDEBUG - Updating session state with new data")
            st.session_state.cleaned_df = cleaned_df.copy()
            st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()
            st.session_state.standards_modified = False
            st.session_state.initialized = True
            
            st.write("- Updated cleaned_df shape:", st.session_state.cleaned_df.shape)
            st.write("- Updated intsta_df shape:", st.session_state.intsta_df.shape)

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

    # Internal standards management
    with st.expander("Manage Internal Standards"):
        if not st.session_state.intsta_df.empty:
            st.write("Current Internal Standards:")
            st.write(st.session_state.intsta_df)

            # Handle removals
            st.subheader("Remove Standards")
            current_standards = st.session_state.intsta_df['LipidMolec'].tolist()
            st.write("DEBUG - Available standards for removal:", current_standards)
            
            def on_remove_button_click():
                st.session_state.remove_button_clicked = True
            
            standards_to_remove = st.multiselect(
                "Select Standards to Remove",
                current_standards,
                key='remove_standards'
            )

            if standards_to_remove:
                if st.button("Remove Selected Standards", key="remove_button"):
                    st.write("\nDEBUG - Starting Removal Process:")
                    st.write("- Standards selected for removal:", standards_to_remove)
                    st.write("- Original cleaned_df shape:", st.session_state.cleaned_df.shape)
                    st.write("- Original intsta_df shape:", st.session_state.intsta_df.shape)
                    st.write("- Original standards list:", st.session_state.intsta_df['LipidMolec'].tolist())
                    
                    # Debug the inputs to remove_standards
                    st.write("\nDEBUG - Checking inputs to remove_standards:")
                    st.write("- cleaned_df is None:", st.session_state.cleaned_df is None)
                    st.write("- intsta_df is None:", st.session_state.intsta_df is None)
                    st.write("- standards_to_remove is empty:", len(standards_to_remove) == 0)
                    
                    # Store the current state of standards_to_remove in session state
                    st.session_state.pending_removals = standards_to_remove
                    st.session_state.removal_in_progress = True
                    st.experimental_rerun()

            # Check if we need to complete a removal operation
            if hasattr(st.session_state, 'removal_in_progress') and st.session_state.removal_in_progress:
                standards_to_remove = st.session_state.pending_removals
                
                # Perform the removal
                new_cleaned_df, new_intsta_df = lp.StandardsManager.remove_standards(
                    st.session_state.cleaned_df,
                    st.session_state.intsta_df,
                    standards_to_remove
                )
                
                st.write("\nDEBUG - After Standards Removal:")
                st.write("- New cleaned_df is None:", new_cleaned_df is None)
                st.write("- New intsta_df is None:", new_intsta_df is None)
                if new_cleaned_df is not None and new_intsta_df is not None:
                    st.write("- New cleaned_df shape:", new_cleaned_df.shape)
                    st.write("- New intsta_df shape:", new_intsta_df.shape)
                    st.write("- New standards list:", new_intsta_df['LipidMolec'].tolist())
                
                # Update session state with explicit checks
                if new_cleaned_df is not None and new_intsta_df is not None:
                    st.session_state.cleaned_df = new_cleaned_df.copy()
                    st.session_state.intsta_df = new_intsta_df.copy()
                    st.session_state.standards_modified = True
                    
                    st.write("\nDEBUG - Updated Session State:")
                    st.write("- Session cleaned_df shape:", st.session_state.cleaned_df.shape)
                    st.write("- Session intsta_df shape:", st.session_state.intsta_df.shape)
                    st.write("- Session standards list:", st.session_state.intsta_df['LipidMolec'].tolist())
                    
                    st.success(f"Removed {len(standards_to_remove)} standards")
                    
                    # Clear the removal flags
                    del st.session_state.removal_in_progress
                    del st.session_state.pending_removals
                else:
                    st.error("Failed to remove standards - got None as result")

            # Add new standards section
            st.subheader("Add New Standard")
            if 'ClassKey' in st.session_state.cleaned_df.columns:
                st.write("\nDEBUG - Adding New Standard:")
                st.write("- Current cleaned_df columns:", st.session_state.cleaned_df.columns.tolist())
                
                lipid_classes = sorted(st.session_state.cleaned_df['ClassKey'].unique())
                st.write("- Available lipid classes:", lipid_classes)
                
                selected_class = st.selectbox(
                    "Select Lipid Class",
                    lipid_classes,
                    key='add_standard_class'
                )

                if selected_class:
                    st.write(f"\nDEBUG - Selected class: {selected_class}")
                    # Check if class already has a standard
                    current_standards = dict(zip(
                        st.session_state.intsta_df['ClassKey'],
                        st.session_state.intsta_df['LipidMolec']
                    ))
                    
                    if selected_class in current_standards:
                        st.warning(f"Class {selected_class} already has a standard: {current_standards[selected_class]}")
                    else:
                        # Get potential standards for this class
                        potential_standards = st.session_state.cleaned_df[
                            st.session_state.cleaned_df['ClassKey'] == selected_class
                        ]['LipidMolec'].tolist()
                        
                        st.write("- Number of potential standards:", len(potential_standards))
                        st.write("- Available standards for class:", potential_standards[:5], "..." if len(potential_standards) > 5 else "")

                        if potential_standards:
                            selected_standard = st.selectbox(
                                "Select Species to Use as Standard",
                                sorted(potential_standards),
                                key='add_standard_species'
                            )

                            if st.button("Add as Standard", key="add_button"):
                                st.write("\nDEBUG - Starting Addition Process:")
                                st.write("- Class to add standard for:", selected_class)
                                st.write("- Standard to add:", selected_standard)
                                st.write("- Original cleaned_df shape:", st.session_state.cleaned_df.shape)
                                st.write("- Original intsta_df shape:", st.session_state.intsta_df.shape)
                                st.write("- Original standards list:", st.session_state.intsta_df['LipidMolec'].tolist())
                                
                                # Store pending addition in session state
                                st.session_state.pending_addition_class = selected_class
                                st.session_state.pending_addition_standard = selected_standard
                                st.session_state.addition_in_progress = True
                                st.experimental_rerun()

            # Check if we need to complete an addition operation
            if hasattr(st.session_state, 'addition_in_progress') and st.session_state.addition_in_progress:
                st.write("\nDEBUG - Completing Addition Process:")
                st.write("- Adding standard:", st.session_state.pending_addition_standard)
                st.write("- For class:", st.session_state.pending_addition_class)
                
                # Perform the addition
                new_cleaned_df, new_intsta_df = lp.StandardsManager.add_standard(
                    st.session_state.cleaned_df,
                    st.session_state.intsta_df,
                    st.session_state.pending_addition_class,
                    st.session_state.pending_addition_standard
                )
                
                st.write("\nDEBUG - After Standard Addition:")
                st.write("- New cleaned_df is None:", new_cleaned_df is None)
                st.write("- New intsta_df is None:", new_intsta_df is None)
                if new_cleaned_df is not None and new_intsta_df is not None:
                    st.write("- New cleaned_df shape:", new_cleaned_df.shape)
                    st.write("- New intsta_df shape:", new_intsta_df.shape)
                    st.write("- New standards list:", new_intsta_df['LipidMolec'].tolist())
                
                # Update session state with explicit checks
                if new_cleaned_df is not None and new_intsta_df is not None:
                    st.session_state.cleaned_df = new_cleaned_df.copy()
                    st.session_state.intsta_df = new_intsta_df.copy()
                    st.session_state.standards_modified = True
                    
                    st.write("\nDEBUG - Updated Session State:")
                    st.write("- Session cleaned_df shape:", st.session_state.cleaned_df.shape)
                    st.write("- Session intsta_df shape:", st.session_state.intsta_df.shape)
                    st.write("- Session standards list:", st.session_state.intsta_df['LipidMolec'].tolist())
                    
                    st.success(f"Added standard {st.session_state.pending_addition_standard} for class {st.session_state.pending_addition_class}")
                    
                    # Clear the addition flags
                    del st.session_state.addition_in_progress
                    del st.session_state.pending_addition_class
                    del st.session_state.pending_addition_standard
                else:
                    st.error("Failed to add standard - got None as result")

            # Within the display_cleaned_data function, replace the download section:

            # Download internal standards
            if not st.session_state.intsta_df.empty:
                st.write("\nDEBUG - Preparing download:")
                st.write(f"- Current intsta_df shape: {st.session_state.intsta_df.shape}")
                st.write("- Current standards:", st.session_state.intsta_df['LipidMolec'].tolist())
                
                # Create a copy of the data for download
                download_df = st.session_state.intsta_df.copy()
                csv_intsta = download_df.to_csv(index=False)
                
                # Add download button without state preservation
                st.download_button(
                    label="Download Internal Standards Data",
                    data=csv_intsta,
                    file_name="internal_standards.csv",
                    mime="text/csv",
                    key=f"download_standards_{st.session_state.intsta_df.shape[0]}"
                )

        st.write("\nDEBUG - Final State:")
        st.write("- Final cleaned_df shape:", st.session_state.cleaned_df.shape)
        st.write("- Final intsta_df shape:", st.session_state.intsta_df.shape)
        
        return st.session_state.cleaned_df, st.session_state.intsta_df

def process_and_display_data(df, data_format):
    """Process and display the data after validation."""
    experiment, exp_success = setup_experiment()
    if exp_success:
        name_df, group_success = process_group_samples(df, experiment)
        if group_success:
            cleaned_df, intsta_df = clean_data(df, name_df, experiment, data_format)
            display_cleaned_data(cleaned_df, intsta_df)

if __name__ == "__main__":
    main()