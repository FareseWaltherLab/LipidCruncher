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
        df = load_and_validate_data(uploaded_file, data_format)
        if df is not None:
            # Convert UI format selection to internal format string
            format_type = 'lipidsearch' if data_format == 'LipidSearch 5.0' else 'generic'
            
            confirmed, name_df, experiment, bqc_label, valid_samples = process_experiment(df, format_type)
            
            # Only proceed with cleaning and displaying data if user has confirmed inputs
            if confirmed and valid_samples:
                cleaned_df, intsta_df = clean_data(df, name_df, experiment, data_format)
                if cleaned_df is not None:  # Add check for successful cleaning
                    display_cleaned_data(cleaned_df, intsta_df)
            elif not confirmed and valid_samples:
                st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")

def initialize_session_state():
    """Initialize the Streamlit session state."""
    pass

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
    """Display cleaned data and internal standards data in expanders with download options."""
    if cleaned_df is not None:
        with st.expander("View Cleaned Data"):
            st.write(cleaned_df)
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        if not intsta_df.empty:
            with st.expander("View Internal Standards"):
                st.write(intsta_df)
                csv_intsta = intsta_df.to_csv(index=False)
                st.download_button(
                    label="Download Data",
                    data=csv_intsta,
                    file_name="internal_standards.csv",
                    mime="text/csv"
                )

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