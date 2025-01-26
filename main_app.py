# Existing imports
import base64
import copy
import hashlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from PIL import Image
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# Local imports
import lipidomics as lp

# Modify the initialize_session_state function to include all necessary state variables
def initialize_session_state():
    """Initialize the Streamlit session state with default values."""
    # Data related states
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.session_state.intsta_df = None
    if 'normalized_df' not in st.session_state:
        st.session_state.normalized_df = None
    if 'continuation_df' not in st.session_state:
        st.session_state.continuation_df = None
    if 'original_column_order' not in st.session_state:
        st.session_state.original_column_order = None
    if 'bqc_label' not in st.session_state:
        st.session_state.bqc_label = None
    
    # Process control states
    if 'module' not in st.session_state:
        st.session_state.module = "Data Cleaning, Filtering, & Normalization"
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
        
    # Normalization related states
    if 'normalization_inputs' not in st.session_state:
        st.session_state.normalization_inputs = {}
    if 'normalization_method' not in st.session_state:
        st.session_state.normalization_method = 'None'
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = []
    if 'create_norm_dataset' not in st.session_state:
        st.session_state.create_norm_dataset = False

# Modify the main function to preserve state when switching modules
def main():
    """Main function for the Lipidomics Analysis Module Streamlit application."""
    # Load and display the logo
    try:
        logo = Image.open('./images/logo.tif')  # Replace with your logo path
        # You can adjust the width to control the size of the logo
        st.image(logo, width=720)  # Adjust width as needed
    except Exception as e:
        # Fallback to text header if image fails to load
        st.error(f"Failed to load logo: {str(e)}")
        st.header("LipidCruncher")
    
    st.markdown("Process, analyze and visualize lipidomics data from multiple sources.")
    
    # Initialize session state for cache clearing
    if 'clear_cache' not in st.session_state:
        st.session_state.clear_cache = False
    
    initialize_session_state()
    
    # Always show format selection in sidebar
    data_format = display_format_selection()
    display_format_requirements(data_format)
    
    # Store the format type in session state for later use
    st.session_state.format_type = data_format
    
    # Update file type options based on format
    file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
    
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset', 
        type=file_types
    )
    
    if uploaded_file:
        df = load_and_validate_data(uploaded_file, data_format)
        if df is not None:
            # Display column mapping for generic format
            if data_format == 'Generic Format':
                display_column_mapping()
            
            # Process experiment setup
            confirmed, name_df, experiment, bqc_label, valid_samples, updated_df = process_experiment(df, data_format)
            
            # Update confirmation state
            st.session_state.confirmed = confirmed
            
            if valid_samples:
                if confirmed:
                    # Update session state with experiment information
                    update_session_state(name_df, experiment, bqc_label)
                    
                    # Use updated_df if available, otherwise use original df
                    df_to_clean = updated_df if updated_df is not None else df
                    
                    if st.session_state.module == "Data Cleaning, Filtering, & Normalization":
                        st.subheader("Data Standardization, Filtering, and Normalization Module")
                        cleaned_df, intsta_df = clean_data(df_to_clean, name_df, experiment, data_format)
                        
                        if cleaned_df is not None:
                            # Store essential data in session state
                            st.session_state.experiment = experiment
                            st.session_state.format_type = data_format
                            st.session_state.cleaned_df = cleaned_df
                            st.session_state.intsta_df = intsta_df
                            st.session_state.continuation_df = cleaned_df
                            
                            # Display cleaned data and manage standards
                            display_cleaned_data(cleaned_df, intsta_df)
                            
                            # Handle normalization
                            normalized_df = handle_data_normalization(
                                cleaned_df, 
                                st.session_state.intsta_df,
                                experiment, 
                                data_format
                            )
                            
                            if normalized_df is not None:
                                st.session_state.normalized_df = normalized_df
                                st.session_state.continuation_df = normalized_df
                                
                                # Add navigation button to next page
                                _, right_column = st.columns([3, 1])
                                with right_column:
                                    if st.button("Next: Quality Check & Analysis", key="next_to_qc_analysis"):
                                        st.session_state.module = "Quality Check & Analysis"
                                        st.experimental_rerun()
                                        
                    elif st.session_state.module == "Quality Check & Analysis":
                        if st.session_state.cleaned_df is not None:
                            quality_check_and_analysis_module(
                                st.session_state.continuation_df, 
                                st.session_state.intsta_df, 
                                st.session_state.experiment,
                                st.session_state.bqc_label,
                                st.session_state.format_type
                            )
                        
                        # Add back button with state preservation
                        if st.button("Back to Data Standardization, Filtering, and Normalization", key="back_to_cleaning"):
                            st.session_state.module = "Data Cleaning, Filtering, & Normalization"
                            # Preserve the current state
                            st.session_state.preserved_data = {
                                'cleaned_df': st.session_state.cleaned_df,
                                'intsta_df': st.session_state.intsta_df,
                                'normalized_df': st.session_state.normalized_df,
                                'continuation_df': st.session_state.continuation_df,
                                'normalization_method': st.session_state.normalization_method,
                                'selected_classes': st.session_state.selected_classes,
                                'create_norm_dataset': st.session_state.create_norm_dataset
                            }
                            st.experimental_rerun()
                
                else:
                    # Clear processed data when unconfirmed
                    clear_session_state()
                    st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
            else:
                st.error("Please ensure your samples are valid before proceeding.")

    # Restore preserved state if returning from Quality Check & Analysis
    if 'preserved_data' in st.session_state and st.session_state.module == "Data Cleaning, Filtering, & Normalization":
        for key, value in st.session_state.preserved_data.items():
            setattr(st.session_state, key, value)
        del st.session_state.preserved_data

    # Place the cache clearing button in the sidebar
    if st.sidebar.button("End Session and Clear Cache"):
        st.session_state.clear_cache = True
        st.experimental_rerun()

    # Check if cache should be cleared
    if st.session_state.clear_cache:
        clear_streamlit_cache()
        st.sidebar.success("Cache cleared successfully!")
        st.session_state.clear_cache = False
        
def clear_session_state():
    """Clear processed data from session state."""
    st.session_state.cleaned_df = None
    st.session_state.intsta_df = None
    st.session_state.normalized_df = None
    st.session_state.continuation_df = None
    st.session_state.experiment = None
    st.session_state.format_type = None
    
def update_session_state(name_df, experiment, bqc_label):
    """
    Update the Streamlit session state with experiment-related information.

    This function updates several session state variables with information
    derived from the experiment setup and naming dataframe.

    Args:
        name_df (pd.DataFrame): DataFrame containing naming information.
        experiment (Experiment): Experiment object containing experimental setup details.
        bqc_label (str): Label for Batch Quality Control samples.

    The following session state variables are updated:
    - name_df: DataFrame with naming information
    - experiment: Experiment object
    - bqc_label: Batch Quality Control label
    - full_samples_list: List of all samples
    - individual_samples_list: List of individual samples for each condition
    - conditions_list: List of experimental conditions
    - extensive_conditions_list: Detailed list of conditions
    - number_of_samples_list: Number of samples for each condition
    - aggregate_number_of_samples_list: Aggregated number of samples
    """
    st.session_state.name_df = name_df
    st.session_state.experiment = experiment
    st.session_state.bqc_label = bqc_label
    st.session_state.full_samples_list = experiment.full_samples_list
    st.session_state.individual_samples_list = experiment.individual_samples_list
    st.session_state.conditions_list = experiment.conditions_list
    st.session_state.extensive_conditions_list = experiment.extensive_conditions_list
    st.session_state.number_of_samples_list = experiment.number_of_samples_list
    st.session_state.aggregate_number_of_samples_list = experiment.aggregate_number_of_samples_list

def display_format_selection():
    return st.sidebar.selectbox(
        'Select Data Format',
        ['LipidSearch 5.0', 'Generic Format', 'Metabolomics Workbench']
    )

def display_format_requirements(data_format):
    """Display format-specific requirements."""
    if data_format == 'Metabolomics Workbench':
        st.info("""
        **Dataset Requirements for Metabolomics Workbench Format**
        
        The file must be a CSV containing:
        1. Required section markers:
           * MS_METABOLITE_DATA_START
           * MS_METABOLITE_DATA_END
           
        2. Three essential rows in the data section:
           * Row 1: Sample names
           * Row 2: Experimental conditions - one condition string per sample
           * Row 3+: Lipid measurements with lipid names in first column
           
        Example:
        ```
        MS_METABOLITE_DATA_START
        Samples,Sample1,Sample2,Sample3,Sample4
        Factors,WT,WT,KO,KO
        LPC(16:0),234.5,256.7,189.3,201.4
        PE(18:0_20:4),456.7,478.2,390.1,405.6
        MS_METABOLITE_DATA_END
        ```
        
        The data will be automatically processed to:
        * Extract the tabular data section
        * Standardize lipid names (e.g., "LPC 16:0" → "LPC(16:0)")
        * Create intensity columns (named as intensity[s1], intensity[s2], etc.)
        * Use the condition strings to suggest experimental setup
        """)
    elif data_format == 'LipidSearch 5.0':
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
        IMPORTANT: Your dataset must contain ONLY these columns in this order:
        
        1. **First Column - Lipid Names:**
           * Can have any column name
           * Must contain lipid molecule identifiers
           * Will be standardized to match these formats:
             - LPC O-17:4         -> LPC(O-17:4)
             - Cer d18:0/C24:0    -> Cer(d18:0_C24:0)
             - CE 14:0;0          -> CE(14:0)
             - CerG1(d13:0_25:2)  -> CerG1(d13:0_25:2)
             - PA 16:0/18:1+O     -> PA(16:0_18:1+O)
             
        2. **Remaining Columns - Intensity Values Only:**
           * Can have any column names
           * Each column should contain intensity values for one sample
           * The number of intensity columns must match the total number of samples in your experiment
           * These columns will be automatically standardized to: intensity[s1], intensity[s2], ..., intensity[sN]
        
        ⚠️ If your dataset contains any additional columns, please remove them before uploading.
        Only the lipid names column followed by intensity columns should be present.
        
        Note: The lipid class (e.g., LPC, Cer, CE) will be automatically extracted from the lipid names to create the ClassKey.
                """)

def load_and_validate_data(uploaded_file, data_format):
    """
    Load and validate uploaded data file.
    """
    try:
        if data_format == 'Metabolomics Workbench':
            # Read as text for Metabolomics Workbench format
            text_content = uploaded_file.getvalue().decode('utf-8')
            
            # Validate format
            df, success, message = lp.DataFormatHandler.validate_and_preprocess(
                text_content,
                data_format
            )
            
            if not success:
                st.error(message)
                return None
                
            st.success("File uploaded and processed successfully!")
            return df
            
        else:
            # Standard CSV processing for other formats
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
        data_format (str): The format of the data
    
    Returns:
        tuple: (name_df, group_df, updated_df, valid_samples)
    """
    grouped_samples = lp.GroupSamples(experiment, data_format)
    
    if not grouped_samples.check_dataset_validity(df):
        st.sidebar.error("Invalid dataset format!")
        return None, None, None, False

    value_cols = grouped_samples.build_mean_area_col_list(df)
    if not value_cols:
        st.sidebar.error("Error processing intensity columns!")
        return None, None, None, False

    st.sidebar.subheader('Group Samples')
    group_df = grouped_samples.build_group_df(df)
    if group_df.empty:
        st.sidebar.error("Error building sample groups!")
        return None, None, None, False
        
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
    
    if data_format == 'Metabolomics Workbench' and 'workbench_conditions' in st.session_state:
        # Parse conditions from the data (but don't display the parsed values)
        parsed_conditions = lp.DataFormatHandler._parse_workbench_conditions(
            set(st.session_state.workbench_conditions.values())
        )
        
        use_detected = st.sidebar.checkbox("Use detected experimental setup", value=True)
        
        if use_detected:
            # Use the detected conditions
            unique_conditions = set(st.session_state.workbench_conditions.values())
            n_conditions = len(unique_conditions)
            conditions_list = list(unique_conditions)
            
            # Count samples per condition
            sample_counts = {}
            for condition in conditions_list:
                count = sum(1 for x in st.session_state.workbench_conditions.values() if x == condition)
                sample_counts[condition] = count
                
            number_of_samples_list = [sample_counts[condition] for condition in conditions_list]
            
            # Display the detected setup
            st.sidebar.write("Using detected setup:")
            for cond, count in zip(conditions_list, number_of_samples_list):
                st.sidebar.write(f"* {cond}: {count} samples")
            
        else:
            # Allow manual setup
            n_conditions = st.sidebar.number_input('Enter the number of conditions', 
                                                 min_value=1, max_value=20, value=1, step=1)
            
            conditions_list = [st.sidebar.text_input(f'Create a label for condition #{i + 1}') 
                             for i in range(n_conditions)]
            
            number_of_samples_list = [st.sidebar.number_input(f'Number of samples for condition #{i + 1}', 
                                                            min_value=1, max_value=1000, value=1, step=1) 
                                    for i in range(n_conditions)]
    else:
        # Standard manual setup for other formats
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

def display_column_mapping():
    """Display the mapping between original and standardized column names in the sidebar."""
    if st.session_state.get('column_mapping') is not None:
        st.sidebar.subheader("Column Name Standardization")
        st.sidebar.dataframe(
            st.session_state.column_mapping.reset_index(drop=True),
            use_container_width=True
        )

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
            
def rename_intensity_to_concentration(df):
    """Renames intensity columns to concentration columns at the end of normalization"""
    df = df.copy()
    rename_dict = {
        col: col.replace('intensity[', 'concentration[')
        for col in df.columns if col.startswith('intensity[')
    }
    return df.rename(columns=rename_dict)

def handle_data_normalization(cleaned_df, intsta_df, experiment, format_type):
    """Handle data normalization with consistent session state usage."""
    
    # Store essential columns before normalization only for LipidSearch 5.0
    stored_columns = {}
    if format_type == 'LipidSearch 5.0':
        essential_columns = ['CalcMass', 'BaseRt']
        stored_columns = {col: cleaned_df[col] for col in essential_columns if col in cleaned_df.columns}
    
    # Validate required columns
    if 'ClassKey' not in cleaned_df.columns:
        st.error("ClassKey column is required for normalization.")
        return None

    # Get the full list of classes
    all_class_lst = list(cleaned_df['ClassKey'].unique())
    
    # Initialize or retrieve selected classes from session state
    # Only initialize with all classes if selected_classes is empty or doesn't exist
    if 'selected_classes' not in st.session_state or not st.session_state.selected_classes:
        st.session_state.selected_classes = all_class_lst.copy()
    
    # Class selection with session state persistence
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        options=all_class_lst,  # Available options are all classes
        default=st.session_state.selected_classes,  # Default to currently selected classes
        key='class_selection'
    )
    
    # Update session state with current selection
    if selected_classes:  # Only update if some classes are selected
        st.session_state.selected_classes = selected_classes

    if not selected_classes:
        st.warning("Please select at least one lipid class to proceed with normalization.")
        return None

    # Filter DataFrame based on selected classes
    filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()

    # Check if we have standards from earlier selection
    has_standards = not st.session_state.intsta_df.empty

    # Determine available normalization options
    normalization_options = ['None', 'Internal Standards', 'BCA Assay', 'Both'] if has_standards else ['None', 'BCA Assay']
    if not has_standards:
        st.warning("No internal standards available. Only BCA normalization is available.")

    # Initialize or retrieve normalization method from session state
    if 'normalization_method' not in st.session_state:
        st.session_state.normalization_method = 'None'
    
    # Select normalization method with session state persistence
    normalization_method = st.radio(
        "Select normalization method:",
        options=normalization_options,
        key='norm_method_selection',
        index=normalization_options.index(st.session_state.normalization_method)
    )
    
    # Update session state
    st.session_state.normalization_method = normalization_method

    normalized_df = filtered_df.copy()
    normalized_data_object = lp.NormalizeData()

    try:
        if normalization_method != 'None':
            # Store normalization settings in session state
            if 'normalization_settings' not in st.session_state:
                st.session_state.normalization_settings = {}
            
            # Track whether we need to do standards normalization
            do_standards = normalization_method in ['Internal Standards', 'Both'] and has_standards
            
            # Handle BCA Assay normalization
            if normalization_method in ['BCA Assay', 'Both']:
                with st.expander("Enter BCA Assay Data"):
                    protein_df = collect_protein_concentrations(experiment)
                    if protein_df is not None:
                        try:
                            normalized_df = normalized_data_object.normalize_using_bca(
                                normalized_df, 
                                protein_df, 
                                preserve_prefix=True
                            )
                            # Store BCA normalization settings
                            st.session_state.normalization_settings['bca'] = {
                                'protein_df': protein_df
                            }
                            st.success("BCA normalization applied successfully")
                        except Exception as e:
                            st.error(f"Error during BCA normalization: {str(e)}")
                            return None

            # Handle internal standards normalization
            if do_standards:
                with st.expander("Enter Inputs For Data Normalization Using Internal Standards"):
                    intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
                    if not intensity_cols:
                        st.error("Internal standards data does not contain properly formatted intensity columns")
                        return None

                    # Group standards by their ClassKey
                    standards_by_class = {}
                    if 'ClassKey' in st.session_state.intsta_df.columns:
                        standards_by_class = st.session_state.intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
            
                    # Process standards with session state preservation
                    class_to_standard_map = process_class_standards(
                        selected_classes, 
                        standards_by_class, 
                        st.session_state.intsta_df,
                        st.session_state.get('class_standard_map', {})
                    )
                    
                    if not class_to_standard_map:
                        return None

                    # Store the standards mapping
                    st.session_state.class_standard_map = class_to_standard_map

                    # Apply normalization
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

        # Rename intensity columns to concentration
        normalized_df = rename_intensity_to_concentration(normalized_df)
        
        # Add back LipidSearch essential columns if needed
        if normalized_df is not None and format_type == 'LipidSearch 5.0':
            for col, values in stored_columns.items():
                normalized_df[col] = values

        # Display normalized data (always show it)
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

def process_class_standards(selected_classes, standards_by_class, intsta_df, saved_mappings=None):
    """
    Process class-standard mapping with session state preservation.
    """
    class_to_standard_map = {}
    all_available_standards = list(intsta_df['LipidMolec'].unique())
    
    for lipid_class in selected_classes:
        # Get previously selected standard if available
        default_standard = None
        if saved_mappings and lipid_class in saved_mappings:
            default_standard = saved_mappings[lipid_class]
            try:
                default_idx = all_available_standards.index(default_standard)
            except ValueError:
                default_idx = 0
        else:
            # Try to use class-specific standard
            class_specific_standards = standards_by_class.get(lipid_class, [])
            default_idx = all_available_standards.index(class_specific_standards[0]) if class_specific_standards else 0
        
        selected_standard = st.selectbox(
            f'Select internal standard for {lipid_class}',
            all_available_standards,
            index=default_idx,
            key=f'standard_selection_{lipid_class}'
        )
        
        class_to_standard_map[lipid_class] = selected_standard

    if len(class_to_standard_map) != len(selected_classes):
        st.error("Please select standards for all lipid classes")
        return None
        
    return class_to_standard_map

def apply_standards_normalization(df, class_to_standard_map, selected_classes, intsta_df, normalizer, experiment):
    """Apply standards normalization with session state preservation."""
    # Create ordered list of standards
    added_intsta_species_lst = [class_to_standard_map[cls] for cls in selected_classes]
    
    # Get unique standards
    selected_standards = set(added_intsta_species_lst)
    
    # Initialize or retrieve concentration values from session state
    if 'standard_concentrations' not in st.session_state:
        st.session_state.standard_concentrations = {}
    
    intsta_concentration_dict = {}
    all_concentrations_entered = True
    
    st.write("Enter the concentration of each selected internal standard (µM):")
    for standard in selected_standards:
        # Use previously entered value as default if available
        default_value = st.session_state.standard_concentrations.get(standard, 1.0)
        
        concentration = st.number_input(
            f"Concentration (µM) for {standard}",
            min_value=0.0,
            value=default_value,
            step=0.1,
            key=f"conc_{standard}"
        )
        
        # Store current value in session state
        st.session_state.standard_concentrations[standard] = concentration
        
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
    
def quality_check_and_analysis_module(continuation_df, intsta_df, experiment, bqc_label, format_type):
    """
    Performs quality check and analysis on the data.
    
    Args:
        continuation_df (pd.DataFrame): The data to analyze
        intsta_df (pd.DataFrame): Internal standards data
        experiment (Experiment): Experiment setup information
        bqc_label (str): Label for batch quality control samples
        format_type (str): The format of the input data (e.g., 'LipidSearch 5.0')
    """
    st.subheader("2) Quality Check & Anomaly Detection")
    
    # Initialize variables
    box_plot_fig1 = None
    box_plot_fig2 = None
    bqc_plot = None
    retention_time_plot = None
    pca_plot = None

    # Initialize session state for plots if not already done
    if 'heatmap_generated' not in st.session_state:
        st.session_state.heatmap_generated = False
    if 'heatmap_fig' not in st.session_state:
        st.session_state.heatmap_fig = {}
    if 'correlation_plots' not in st.session_state:
        st.session_state.correlation_plots = {}
    if 'abundance_bar_charts' not in st.session_state:
        st.session_state.abundance_bar_charts = {'linear': None, 'log2': None}
    if 'abundance_pie_charts' not in st.session_state:
        st.session_state.abundance_pie_charts = {}
    if 'saturation_plots' not in st.session_state:
        st.session_state.saturation_plots = {}
    if 'volcano_plots' not in st.session_state:
        st.session_state.volcano_plots = {}
    if 'pathway_visualization' not in st.session_state:
        st.session_state.pathway_visualization = None

    # Quality Check
    box_plot_fig1, box_plot_fig2 = display_box_plots(continuation_df, experiment)
    continuation_df, bqc_plot = conduct_bqc_quality_assessment(bqc_label, continuation_df, experiment)
    
    # Only show retention time plots for LipidSearch 5.0
    if format_type == 'LipidSearch 5.0':
        retention_time_plot = display_retention_time_plots(continuation_df, format_type)
    
    # Pairwise Correlation Analysis
    selected_condition, corr_fig = analyze_pairwise_correlation(continuation_df, experiment)
    if selected_condition and corr_fig:
        st.session_state.correlation_plots[selected_condition] = corr_fig
    
    # PCA Analysis
    continuation_df, pca_plot = display_pca_analysis(continuation_df, experiment)
    
    st.subheader("3) Data Visualization, Interpretation, and Analysis ")
    # Analysis
    analysis_option = st.radio(
        "Select an analysis feature:",
        (
            "Class Level Breakdown - Bar Chart", 
            "Class Level Breakdown - Pie Charts", 
            "Class Level Breakdown - Saturation Plots", 
            "Class Level Breakdown - Pathway Visualization",
            "Species Level Breakdown - Volcano Plot",
            "Species Level Breakdown - Lipidomic Heatmap"
        )
    )

    if analysis_option == "Class Level Breakdown - Bar Chart":
        linear_chart, log2_chart = display_abundance_bar_charts(experiment, continuation_df)
        if linear_chart:
            st.session_state.abundance_bar_charts['linear'] = linear_chart
        if log2_chart:
            st.session_state.abundance_bar_charts['log2'] = log2_chart
    elif analysis_option == "Class Level Breakdown - Pie Charts":
        pie_charts = display_abundance_pie_charts(experiment, continuation_df)
        st.session_state.abundance_pie_charts.update(pie_charts)
    elif analysis_option == "Class Level Breakdown - Saturation Plots":
        saturation_plots = display_saturation_plots(experiment, continuation_df)
        st.session_state.saturation_plots.update(saturation_plots)
    elif analysis_option == "Class Level Breakdown - Pathway Visualization":
        pathway_fig = display_pathway_visualization(experiment, continuation_df)
        if pathway_fig:
            st.session_state.pathway_visualization = pathway_fig
    elif analysis_option == "Species Level Breakdown - Volcano Plot":
        volcano_plots = display_volcano_plot(experiment, continuation_df)
        st.session_state.volcano_plots.update(volcano_plots)
    elif analysis_option == "Species Level Breakdown - Lipidomic Heatmap":
        heatmap_result = display_lipidomic_heatmap(experiment, continuation_df)
        if isinstance(heatmap_result, tuple) and len(heatmap_result) == 2:
            regular_heatmap, clustered_heatmap = heatmap_result
        else:
            regular_heatmap = heatmap_result
            clustered_heatmap = None
        
        if regular_heatmap:
            st.session_state.heatmap_generated = True
            st.session_state.heatmap_fig['Regular Heatmap'] = regular_heatmap
        if clustered_heatmap:
            st.session_state.heatmap_fig['Clustered Heatmap'] = clustered_heatmap

    # PDF Generation Section
    st.subheader("Generate PDF Report")
    st.warning(
        "⚠️ Important: PDF Report Generation Guidelines\n\n"
        "1. Generate the PDF only after completing all desired analyses.\n"
        "2. Ensure all analyses you want in the report have been viewed at least once.\n"
        "3. Use this feature instead of downloading plots individually - it's more efficient for multiple downloads.\n"
        "4. Generate the PDF before clearing the cache.\n"
        "5. Avoid interacting with the app during PDF generation.\n"
        "6. Select 'No' if you're not ready to end your session."
    )

    generate_pdf = st.radio("Are you ready to generate the PDF report?", ('No', 'Yes'), index=0)

    if generate_pdf == 'Yes':
        st.info("Please confirm that you have completed all analyses and are ready to end your session.")
        confirm_generate = st.checkbox("I confirm that I'm ready to generate the PDF and end my session.")
        
        if confirm_generate:
            if box_plot_fig1 and box_plot_fig2:
                with st.spinner('Generating PDF report... Please do not interact with the app.'):
                    pdf_buffer = generate_pdf_report(
                        box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot, 
                        st.session_state.heatmap_fig, st.session_state.correlation_plots, 
                        st.session_state.abundance_bar_charts, st.session_state.abundance_pie_charts, 
                        st.session_state.saturation_plots, st.session_state.volcano_plots,
                        st.session_state.pathway_visualization
                    )
                if pdf_buffer:
                    st.success("PDF report generated successfully!")
                    st.download_button(
                        label="Download Quality Check & Analysis Report (PDF)",
                        data=pdf_buffer,
                        file_name="quality_check_and_analysis_report.pdf",
                        mime="application/pdf",
                    )
                    st.info(
                        "You can now download your PDF report. After downloading, please use the "
                        "'End Session and Clear Cache' button in the sidebar to conclude your session."
                    )
                    
                    # Close the figures to free up memory after PDF generation
                    plt.close(box_plot_fig1)
                    plt.close(box_plot_fig2)
                    plt.close('all')  # This closes all remaining matplotlib figures
                else:
                    st.error("Failed to generate PDF report. Please check the logs for details.")
            else:
                st.warning("Some plots are missing. Unable to generate PDF report.")
        else:
            st.info("Please confirm when you're ready to generate the PDF report.")
            
def display_box_plots(continuation_df, experiment):
    # Initialize a counter in session state if it doesn't exist
    if 'box_plot_counter' not in st.session_state:
        st.session_state.box_plot_counter = 0
    
    # Increment the counter
    st.session_state.box_plot_counter += 1
    
    # Generate a unique identifier based on the current data and counter
    unique_id = hashlib.md5(f"{str(continuation_df.index.tolist())}_{st.session_state.box_plot_counter}".encode()).hexdigest()
    
    expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
    with expand_box_plot:
        # Creating a deep copy for visualization to keep the original continuation_df intact
        visualization_df = continuation_df.copy(deep=True)
        
        # Ensure the columns reflect the current state of the DataFrame
        current_samples = [sample for sample in experiment.full_samples_list if f'concentration[{sample}]' in visualization_df.columns]
        
        mean_area_df = lp.BoxPlot.create_mean_area_df(visualization_df, current_samples)
        zero_values_percent_list = lp.BoxPlot.calculate_missing_values_percentage(mean_area_df)
        
        # Generate and display the first plot (Missing Values Distribution)
        fig1 = lp.BoxPlot.plot_missing_values(current_samples, zero_values_percent_list)
        st.pyplot(fig1)
        matplotlib_svg_download_button(fig1, "missing_values_distribution.svg")

        # Data for download (Missing Values)
        missing_values_data = np.vstack((current_samples, zero_values_percent_list)).T
        missing_values_csv = convert_df(pd.DataFrame(missing_values_data, columns=["Sample", "Percentage Missing"]))
        st.download_button(
            label="Download CSV",
            data=missing_values_csv,
            file_name="missing_values_data.csv",
            mime='text/csv',
            key=f"download_missing_values_data_{unique_id}"
        )
        
        st.write('--------------------------------------------------------------------------------')
        
        # Generate and display the second plot (Box Plot)
        fig2 = lp.BoxPlot.plot_box_plot(mean_area_df, current_samples)
        st.pyplot(fig2)
        matplotlib_svg_download_button(fig2, "box_plot.svg")
        
        # Provide option to download the raw data for Box Plot
        csv_data = convert_df(mean_area_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="box_plot_data.csv",
            mime='text/csv',
            key=f"download_box_plot_data_{unique_id}"
        )
    
    # Return the figure objects for later use in PDF generation
    return fig1, fig2

def matplotlib_svg_download_button(fig, filename):
    """
    Creates a Streamlit download button for the SVG format of a matplotlib figure.
    Args:
        fig (matplotlib.figure.Figure): Matplotlib figure object to be converted into SVG.
        filename (str): The desired name of the downloadable SVG file.
    """
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    
    # Read the SVG data and create a download button
    svg_string = buf.getvalue().decode('utf-8')
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml"
    )

def plotly_svg_download_button(fig, filename):
    """
    Creates a Streamlit download button for the SVG format of a plotly figure.
    Args:
        fig (plotly.graph_objs.Figure): Plotly figure object to be converted into SVG.
        filename (str): The desired name of the downloadable SVG file.
    """
    # Convert the figure to SVG format
    svg_bytes = fig.to_image(format="svg")
    
    # Decode the bytes to a string
    svg_string = svg_bytes.decode('utf-8')
    
    # Ensure the SVG string starts with the correct XML declaration
    if not svg_string.startswith('<?xml'):
        svg_string = '<?xml version="1.0" encoding="utf-8"?>\n' + svg_string
    
    # Create a download button
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml"
    )
    
@st.cache_data(ttl=3600)
def convert_df(df):
    """
    Convert a DataFrame to CSV format and encode it for downloading.

    Parameters:
    df (pd.DataFrame): The DataFrame to be converted.

    Returns:
    bytes: A CSV-formatted byte string of the DataFrame.

    Raises:
    Exception: If the DataFrame conversion to CSV fails.
    """
    if df is None or df.empty:
        raise ValueError("The DataFrame provided is empty or None.")
    try:
        return df.to_csv().encode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to convert DataFrame to CSV: {e}")
        
def conduct_bqc_quality_assessment(bqc_label, data_df, experiment):
    scatter_plot = None  # Initialize scatter_plot to None
    
    if bqc_label is not None:
        with st.expander("Quality Check Using BQC Samples"):
            bqc_sample_index = experiment.conditions_list.index(bqc_label)
            scatter_plot, prepared_df, reliable_data_percent = lp.BQCQualityCheck.generate_and_display_cov_plot(data_df, experiment, bqc_sample_index)
            
            st.plotly_chart(scatter_plot, use_container_width=True)
            plotly_svg_download_button(scatter_plot, "bqc_quality_check.svg")
            csv_data = convert_df(prepared_df[['LipidMolec', 'cov', 'mean']].dropna())
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="CoV_Plot_Data.csv",
                mime='text/csv'
            )
            
            if reliable_data_percent >= 80:
                st.info(f"{reliable_data_percent}% of the datapoints are confidently reliable (CoV < 30%).")
            elif reliable_data_percent >= 50:
                st.warning(f"{reliable_data_percent}% of the datapoints are confidently reliable.")
            else:
                st.error(f"Less than 50% of the datapoints are confidently reliable (CoV < 30%).")
            
            if prepared_df is not None and not prepared_df.empty:
                filter_option = st.radio("Would you like to filter your data using BQC samples?", ("No", "Yes"), index=0)
                if filter_option == "Yes":
                    cov_threshold = st.number_input('Enter the maximum acceptable CoV in %', min_value=10, max_value=1000, value=30, step=1)
                    filtered_df = lp.BQCQualityCheck.filter_dataframe_by_cov_threshold(cov_threshold, prepared_df)
                    
                    if filtered_df.empty:
                        st.error("The filtered dataset is empty. Please try a higher CoV threshold.")
                        st.warning("Returning the original dataset without filtering.")
                    else:
                        st.write('Filtered dataset:')
                        st.write(filtered_df)
                        csv_download = convert_df(filtered_df)
                        st.download_button(
                            label="Download Filtered Data",
                            data=csv_download,
                            file_name='Filtered_Data.csv',
                            mime='text/csv'
                        )
                        data_df = filtered_df  # Update data_df with the filtered dataset

    return data_df, scatter_plot  # Always return a tuple

def integrate_retention_time_plots(continuation_df):
    """
    Integrates retention time plots into the Streamlit app using Plotly.
    Based on the user's choice, this function either plots individual retention 
    times for each lipid class or allows for comparison across different classes. 
    It also provides options for downloading the plot data in CSV format.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data post-cleaning and normalization.

    Returns:
        plotly.graph_objs._figure.Figure or None: The multi-class retention time comparison plot if in Comparison Mode, else None.
    """
    mode = st.radio('Pick a mode', ['Comparison Mode', 'Individual Mode'])
    if mode == 'Individual Mode':
        # Handling individual retention time plots
        plots = lp.RetentionTime.plot_single_retention(continuation_df)
        for idx, (plot, retention_df) in enumerate(plots, 1):
            st.plotly_chart(plot, use_container_width=True)
            plotly_svg_download_button(plot, f"retention_time_plot_{idx}.svg")
            csv_download = convert_df(retention_df)
            st.download_button(label="Download CSV", data=csv_download, file_name=f'retention_plot_{idx}.csv', mime='text/csv')
        return None
    elif mode == 'Comparison Mode':
        # Handling comparison mode for retention time plots
        all_lipid_classes_lst = continuation_df['ClassKey'].value_counts().index.tolist()
        selected_classes_list = st.multiselect('Add/Remove classes:', all_lipid_classes_lst, all_lipid_classes_lst)
        if selected_classes_list:  # Ensuring that selected_classes_list is not empty
            plot, retention_df = lp.RetentionTime.plot_multi_retention(continuation_df, selected_classes_list)
            if plot:
                st.plotly_chart(plot, use_container_width=True)
                plotly_svg_download_button(plot, "retention_time_comparison.svg")
                csv_download = convert_df(retention_df)
                st.download_button(label="Download CSV", data=csv_download, file_name='Retention_Time_Comparison.csv', mime='text/csv')
                return plot
    return None

def display_retention_time_plots(continuation_df, format_type):
    """
    Displays retention time plots for lipid species within the Streamlit interface using Plotly.

    Args:
        continuation_df (pd.DataFrame): The DataFrame containing lipidomic data after any necessary transformations.
        format_type (str): The format of the input data (e.g., 'LipidSearch 5.0')

    Returns:
        plotly.graph_objs._figure.Figure or None: The multi-class retention time comparison plot if generated, else None.
    """
    if format_type == 'LipidSearch 5.0':
        expand_retention = st.expander('View Retention Time Plots: Check Sanity of Data')
        with expand_retention:
            return integrate_retention_time_plots(continuation_df)
    else:
        # For other formats, return None (retention time plots not applicable)
        return None
    
def analyze_pairwise_correlation(continuation_df, experiment):
    """
    Analyzes pairwise correlations for given conditions in the experiment data.
    This function creates a Streamlit expander for displaying pairwise correlation analysis.
    It allows the user to select a condition from those with multiple replicates and a sample type.
    The function then computes and displays a correlation heatmap for the selected condition.
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the selected condition and the matplotlib figure, or (None, None) if no plot was generated.
    """
    expand_corr = st.expander('Pairwise Correlation Analysis')
    with expand_corr:
        st.info("LipidCruncher removes the missing values before performing the correlation test.")
        # Filter out conditions with only one replicate
        multi_replicate_conditions = [condition for condition, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]
        # Ensure there are multi-replicate conditions before proceeding
        if multi_replicate_conditions:
            selected_condition = st.selectbox('Select a condition', multi_replicate_conditions)
            condition_index = experiment.conditions_list.index(selected_condition)
            sample_type = st.selectbox('Select the type of your samples', ['biological replicates', 'Technical replicates'])
            mean_area_df = lp.Correlation.prepare_data_for_correlation(continuation_df, experiment.individual_samples_list, condition_index)
            correlation_df, v_min, thresh = lp.Correlation.compute_correlation(mean_area_df, sample_type)
            fig = lp.Correlation.render_correlation_plot(correlation_df, v_min, thresh, experiment.conditions_list[condition_index])
            st.pyplot(fig)
            matplotlib_svg_download_button(fig, f"correlation_plot_{experiment.conditions_list[condition_index]}.svg")
            
            st.write('Find the exact correlation coefficients in the table below:')
            st.write(correlation_df)
            csv_download = convert_df(correlation_df)
            st.download_button(
                label="Download CSV",
                data=csv_download,
                file_name='Correlation_Matrix_' + experiment.conditions_list[condition_index] + '.csv',
                mime='text/csv'
            )
            return selected_condition, fig
        else:
            st.error("No conditions with multiple replicates found.")
            return None, None
        
def display_pca_analysis(continuation_df, experiment):
    """
    Displays the PCA analysis interface in the Streamlit app and generates a PCA plot using Plotly.
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the updated continuation_df and the PCA plot.
    """
    pca_plot = None
    with st.expander("Principal Component Analysis (PCA)"):
        samples_to_remove = st.multiselect('Select samples to remove from the analysis (optional):', experiment.full_samples_list)
        
        if samples_to_remove:
            if (len(experiment.full_samples_list) - len(samples_to_remove)) >= 2:
                continuation_df = experiment.remove_bad_samples(samples_to_remove, continuation_df)
            else:
                st.error('At least two samples are required for a meaningful analysis!')
        
        pca_plot, pca_df = lp.PCAAnalysis.plot_pca(continuation_df, experiment.full_samples_list, experiment.extensive_conditions_list)
        st.plotly_chart(pca_plot, use_container_width=True)
        plotly_svg_download_button(pca_plot, "pca_plot.svg")
        
        csv_data = convert_df(pca_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="PCA_data.csv",
            mime="text/csv"
        )
    return continuation_df, pca_plot

def display_abundance_bar_charts(experiment, continuation_df):
    with st.expander("Class Concentration Bar Chart"):
        experiment = st.session_state.experiment
        
        # Filter out conditions with only one sample
        valid_conditions = [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]
        
        selected_conditions_list = st.multiselect(
            'Add or remove conditions', 
            valid_conditions, 
            valid_conditions,
            key='conditions_select'
        )
        selected_classes_list = st.multiselect(
            'Add or remove classes:',
            list(continuation_df['ClassKey'].value_counts().index), 
            list(continuation_df['ClassKey'].value_counts().index),
            key='classes_select'
        )
        
        linear_fig, log2_fig = None, None
        if selected_conditions_list and selected_classes_list:
            # Statistical testing guidance
            if len(selected_conditions_list) > 2:
                st.info("""
                📊 **Statistical Testing Note:**
                - Multiple conditions detected: Using ANOVA + Tukey's test
                - This is the correct approach when interested in multiple comparisons
                - Do NOT run separate t-tests by unselecting conditions - this would inflate the false positive rate
                - The current analysis automatically adjusts significance thresholds to maintain a 5% false positive rate across all comparisons
                """)
            elif len(selected_conditions_list) == 2:
                st.info("""
                📊 **Statistical Testing Note:**
                - Two conditions detected: Using t-test
                - This is appropriate ONLY for a single pre-planned comparison
                - If you plan to compare multiple conditions, please select all relevant conditions to use ANOVA + Tukey's test
                """)
            
            # Perform statistical tests
            statistical_results = lp.AbundanceBarChart.perform_statistical_tests(
                continuation_df, 
                experiment, 
                selected_conditions_list, 
                selected_classes_list
            )
            
            # Display statistical testing information
            if statistical_results:
                if len(selected_conditions_list) == 2:
                    st.info("""
                    📊 **Statistical Testing Information**
                    - Method: Student t-test
                    - Significance levels:
                         * p < 0.05
                        - ** p < 0.01
                        - *** p < 0.001
                    """)
                else:
                    st.info("""
                    📊 **Statistical Testing Information**
                    - Method: ANOVA + Tukey's test
                    - Multiple comparison correction applied
                    - Significance levels:
                         * p < 0.05
                        - ** p < 0.01
                        - *** p < 0.001
                    """)
            
            # Optional detailed statistical results
            if st.checkbox("Show detailed statistical analysis"):
                display_statistical_details(
                    statistical_results, 
                    selected_conditions_list
                )

            # Generate linear scale chart
            linear_fig, abundance_df = lp.AbundanceBarChart.create_abundance_bar_chart(
                continuation_df, 
                experiment.full_samples_list, 
                experiment.individual_samples_list, 
                experiment.conditions_list, 
                selected_conditions_list, 
                selected_classes_list, 
                'linear scale',
                statistical_results
            )
            if linear_fig is not None and abundance_df is not None and not abundance_df.empty:
                st.pyplot(linear_fig)
                st.write("Linear Scale")
                
                matplotlib_svg_download_button(linear_fig, "abundance_bar_chart_linear.svg")
                
                try:
                    csv_data = convert_df(abundance_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name='abundance_bar_chart_linear.csv',
                        mime='text/csv',
                        key='abundance_chart_download_linear'
                    )
                except Exception as e:
                    st.error(f"Failed to convert DataFrame to CSV: {str(e)}")
            
            # Generate log2 scale chart
            log2_fig, abundance_df_log2 = lp.AbundanceBarChart.create_abundance_bar_chart(
                continuation_df, 
                experiment.full_samples_list, 
                experiment.individual_samples_list, 
                experiment.conditions_list, 
                selected_conditions_list, 
                selected_classes_list, 
                'log2 scale',
                statistical_results
            )
            if log2_fig is not None and abundance_df_log2 is not None and not abundance_df_log2.empty:
                st.pyplot(log2_fig)
                st.write("Log2 Scale")
                
                matplotlib_svg_download_button(log2_fig, "abundance_bar_chart_log2.svg")
                
                try:
                    csv_data_log2 = convert_df(abundance_df_log2)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data_log2,
                        file_name='abundance_bar_chart_log2.csv',
                        mime='text/csv',
                        key='abundance_chart_download_log2'
                    )
                except Exception as e:
                    st.error(f"Failed to convert DataFrame to CSV: {str(e)}")
            
            # Check if any conditions were removed due to having only one sample
            removed_conditions = set(experiment.conditions_list) - set(valid_conditions)
            if removed_conditions:
                st.warning(f"The following conditions were excluded due to having only one sample: {', '.join(removed_conditions)}")
                
        else:
            st.warning("Please select at least one condition and one class to create the charts.")
        
        return linear_fig, log2_fig

def display_statistical_details(statistical_results, selected_conditions):
    """Display detailed statistical analysis"""
    st.write("### Detailed Statistical Analysis")
    
    if len(selected_conditions) == 2:
        st.write(f"Comparing conditions: {selected_conditions[0]} vs {selected_conditions[1]}")
        st.write("Method: Independent t-test (Welch's t-test)")
    else:
        st.write(f"Comparing {len(selected_conditions)} conditions using ANOVA + Tukey's test")
        st.write("Note: P-values are adjusted for multiple comparisons")
    
    for lipid_class, results in statistical_results.items():
        st.write(f"\n#### {lipid_class}")
        p_value = results['p-value']
        
        if results['test'] == 't-test':
            st.write(f"t-statistic: {results['statistic']:.3f}")
            st.write(f"p-value: {p_value:.3f}")
        else:  # ANOVA
            st.write(f"F-statistic: {results['statistic']:.3f}")
            st.write(f"p-value: {p_value:.3f}")
            
            if results.get('tukey_results'):
                st.write("\nTukey's test pairwise comparisons:")
                tukey = results['tukey_results']
                for g1, g2, p in zip(tukey['group1'], tukey['group2'], tukey['p_values']):
                    st.write(f"- {g1} vs {g2}: p = {p:.3f}")
            else:
                st.write("No pairwise comparisons available (ANOVA p-value > 0.05)")
                
def display_abundance_pie_charts(experiment, continuation_df):
    pie_charts = {}
    with st.expander("Class Concentration Pie Chart"):
        full_samples_list = experiment.full_samples_list
        all_classes = lp.AbundancePieChart.get_all_classes(continuation_df, full_samples_list)
        selected_classes_list = st.multiselect('Select classes for the chart:', all_classes, all_classes)
        if selected_classes_list:
            filtered_df = lp.AbundancePieChart.filter_df_for_selected_classes(continuation_df, full_samples_list, selected_classes_list)
            color_mapping = lp.AbundancePieChart._generate_color_mapping(selected_classes_list)
            for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list):
                if len(samples) > 1:  # Skip conditions with only one sample
                    fig, df = lp.AbundancePieChart.create_pie_chart(filtered_df, full_samples_list, condition, samples, color_mapping)
                    st.subheader(f"Abundance Pie Chart for {condition}")
                    st.plotly_chart(fig)
                    
                    # Add SVG download button for the pie chart
                    plotly_svg_download_button(fig, f"abundance_pie_chart_{condition}.svg")
                    
                    # Add CSV download button for the pie chart data
                    csv_download = convert_df(df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name=f'abundance_pie_chart_{condition}.csv',
                        mime='text/csv'
                    )
                    
                    pie_charts[condition] = fig
                    
                    st.markdown("---")  # Add a separator between charts
    return pie_charts

def display_saturation_plots(experiment, continuation_df):
    saturation_plots = {}
    with st.expander("Saturation Plots"):
        full_samples_list = experiment.full_samples_list
        
        # Get all unique classes from the DataFrame
        all_classes = continuation_df['ClassKey'].unique().tolist()
        
        selected_classes_list = st.multiselect('Select classes for the saturation plot:', all_classes, all_classes)
        
        if selected_classes_list:
            # Filter the DataFrame for selected classes
            filtered_df = continuation_df[continuation_df['ClassKey'].isin(selected_classes_list)]
            
            selected_conditions = st.multiselect('Select conditions for the saturation plot:', experiment.conditions_list, experiment.conditions_list)
            
            if selected_conditions:
                # Add statistical testing guidance based on number of conditions
                if len(selected_conditions) > 2:
                    st.info("""
                    📊 **Statistical Testing for Fatty Acid Comparisons:**
                    - Multiple conditions detected: Using ANOVA + Tukey's test
                    - Each fatty acid type (SFA, MUFA, PUFA) is tested separately
                    - ANOVA determines if there are any differences between conditions
                    - If ANOVA is significant (p < 0.05), Tukey's test shows which specific conditions differ
                    - The analysis automatically adjusts significance thresholds to maintain a 5% false positive rate
                    - Stars (★) indicate significance levels:
                        * ★ p < 0.05
                        * ★★ p < 0.01
                        * ★★★ p < 0.001
                    """)
                elif len(selected_conditions) == 2:
                    st.info("""
                    📊 **Statistical Testing for Fatty Acid Comparisons:**
                    - Two conditions detected: Using Welch's t-test
                    - Each fatty acid type (SFA, MUFA, PUFA) is tested separately
                    - The t-test determines if there are significant differences between conditions
                    - Stars (★) indicate significance levels:
                        * ★ p < 0.05
                        * ★★ p < 0.01
                        * ★★★ p < 0.001
                    - Note: If you plan to compare multiple conditions, select all relevant conditions to use ANOVA
                    """)

                plots = lp.SaturationPlot.create_plots(filtered_df, experiment, selected_conditions)
                
                for lipid_class, (main_plot, percentage_plot, plot_data) in plots.items():
                    st.subheader(f"Saturation Plot for {lipid_class}")
                    
                    st.plotly_chart(main_plot)
                    plotly_svg_download_button(main_plot, f"saturation_plot_main_{lipid_class}.svg")
                    
                    main_csv_download = convert_df(plot_data)
                    st.download_button(
                        label="Download CSV",
                        data=main_csv_download,
                        file_name=f'saturation_plot_main_data_{lipid_class}.csv',
                        mime='text/csv'
                    )
                    
                    st.plotly_chart(percentage_plot)
                    plotly_svg_download_button(percentage_plot, f"saturation_plot_percentage_{lipid_class}.svg")
                    
                    percentage_data = lp.SaturationPlot._calculate_percentage_df(plot_data)
                    percentage_csv_download = convert_df(percentage_data)
                    st.download_button(
                        label="Download CSV",
                        data=percentage_csv_download,
                        file_name=f'saturation_plot_percentage_data_{lipid_class}.csv',
                        mime='text/csv'
                    )
                    
                    saturation_plots[lipid_class] = {'main': main_plot, 'percentage': percentage_plot}
                    
                    st.markdown("---")
    return saturation_plots

def display_pathway_visualization(experiment, continuation_df):
    """
    Displays an interactive lipid pathway visualization within a Streamlit application, 
    based on lipidomics data from a specified experiment. The function allows users 
    to select control and experimental conditions from the experiment setup. It then 
    calculates the saturation ratio and fold change for each lipid class and generates 
    a comprehensive pathway visualization, highlighting the abundance and saturation 
    levels of different lipid classes.

    The visualization is displayed within an expandable section in the Streamlit interface. 
    Users can also download the visualization as an SVG file for high-quality printing or 
    further use, as well as the underlying data as a CSV file.

    Args:
        experiment: An object containing detailed information about the experimental setup, 
                    including conditions and samples. This object is used to derive 
                    control and experimental conditions for the analysis.
        continuation_df (DataFrame): A DataFrame containing processed lipidomics data, 
                                    which includes information necessary for generating 
                                    the pathway visualization, such as lipid classes, 
                                    molecular structures, and abundances.

    Raises:
        ValueError: If there are not at least two conditions with more than one replicate 
                    in the experiment, which is a prerequisite for creating a meaningful 
                    pathway visualization.
    """

    with st.expander("Lipid Pathway Visualization"):
        # UI for selecting control and experimental conditions
        if len([x for x in experiment.number_of_samples_list if x > 1]) > 1:
            control_condition = st.selectbox('Select Control Condition',
                                             [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1], 
                                             index=0)
            experimental_condition = st.selectbox('Select Experimental Condition',
                                                  [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1 and cond != control_condition], 
                                                  index=1)
    
            # Check if both conditions are selected
            if control_condition and experimental_condition:
                class_saturation_ratio_df = lp.PathwayViz.calculate_class_saturation_ratio(continuation_df)
                class_fold_change_df = lp.PathwayViz.calculate_class_fold_change(continuation_df, experiment, control_condition, experimental_condition)
                
                fig, pathway_dict = lp.PathwayViz.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df, control_condition, experimental_condition)
                
                if fig is not None and pathway_dict:
                    st.pyplot(fig)
                    
                    # Add SVG download button for the pathway visualization
                    matplotlib_svg_download_button(fig, f"pathway_visualization_{control_condition}_vs_{experimental_condition}.svg")
                    
                    pathway_df = pd.DataFrame.from_dict(pathway_dict)
                    pathway_df.set_index('class', inplace=True)
                    st.write(pathway_df)
                    csv_download = convert_df(pathway_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name='pathway_df.csv',
                        mime='text/csv')
                    
                    return fig
                    
                else:
                    st.warning("Unable to generate pathway visualization due to insufficient data.")
        else:
            st.warning("At least two conditions with more than one replicate are required for pathway visualization.")
            
        return None
                
def display_volcano_plot(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with volcano plots in lipidomics data using Plotly.
    Returns a dictionary containing the generated plots.
    """
    volcano_plots = {}
    with st.expander("Volcano Plots - Test Hypothesis"):
        conditions_with_replicates = [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1]
        if len(conditions_with_replicates) <= 1:
            st.error('You need at least two conditions with more than one replicate to create a volcano plot.')
            return volcano_plots

        p_value_threshold = st.number_input('Enter the significance level for Volcano Plot', min_value=0.001, max_value=0.1, value=0.05, step=0.001, key="volcano_plot_p_value_threshold")
        q_value_threshold = -np.log10(p_value_threshold)
        control_condition = st.selectbox('Pick the control condition', conditions_with_replicates)
        default_experimental = conditions_with_replicates[1] if len(conditions_with_replicates) > 1 else conditions_with_replicates[0]
        experimental_condition = st.selectbox('Pick the experimental condition', conditions_with_replicates, index=conditions_with_replicates.index(default_experimental))
        selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), list(continuation_df['ClassKey'].value_counts().index))
        
        hide_non_significant = st.checkbox('Hide non-significant data points', value=False)

        plot, merged_df, removed_lipids_df = lp.VolcanoPlot.create_and_display_volcano_plot(experiment, continuation_df, control_condition, experimental_condition, selected_classes_list, q_value_threshold, hide_non_significant)
        st.plotly_chart(plot, use_container_width=True)
        volcano_plots['main'] = plot
        
        # Add SVG download button for the main volcano plot
        plotly_svg_download_button(plot, "volcano_plot.svg")

        # Download options
        csv_data = convert_df(merged_df[['LipidMolec', 'FoldChange', '-log10(pValue)', 'ClassKey']])
        st.download_button("Download CSV", csv_data, file_name="volcano_data.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')
        
        # Generate and display the concentration vs. fold change plot
        color_mapping = lp.VolcanoPlot._generate_color_mapping(merged_df)
        concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(merged_df, color_mapping, q_value_threshold, hide_non_significant)
        st.plotly_chart(concentration_vs_fold_change_plot, use_container_width=True)
        volcano_plots['concentration_vs_fold_change'] = concentration_vs_fold_change_plot
        
        # Add SVG download button for the concentration vs fold change plot
        plotly_svg_download_button(concentration_vs_fold_change_plot, "concentration_vs_fold_change_plot.svg")

        # CSV download option for concentration vs. fold change plot
        csv_data_for_concentration_plot = convert_df(download_df)
        st.download_button("Download CSV", csv_data_for_concentration_plot, file_name="concentration_vs_fold_change_data.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')
        
        # Additional functionality for lipid concentration distribution
        all_classes = list(merged_df['ClassKey'].unique())
        selected_class = st.selectbox('Select Lipid Class:', all_classes)
        if selected_class:
            class_lipids = merged_df[merged_df['ClassKey'] == selected_class]['LipidMolec'].unique().tolist()
            selected_lipids = st.multiselect('Select Lipids:', class_lipids, default=class_lipids[:1])
            if selected_lipids:
                selected_conditions = [control_condition, experimental_condition]
                plot_df = lp.VolcanoPlot.create_concentration_distribution_data(
                    merged_df, selected_lipids, selected_conditions, experiment
                )
                fig = lp.VolcanoPlot.create_concentration_distribution_plot(plot_df, selected_lipids, selected_conditions)
                st.pyplot(fig)
                volcano_plots['concentration_distribution'] = fig
                
                # Add SVG download button for the concentration distribution plot (matplotlib)
                matplotlib_svg_download_button(fig, "concentration_distribution_plot.svg")
                
                csv_data = convert_df(plot_df)
                st.download_button("Download CSV", csv_data, file_name=f"{'_'.join(selected_lipids)}_concentration.csv", mime="text/csv")
        st.write('------------------------------------------------------------------------------------')

        # Displaying the table of invalid lipids
        if not removed_lipids_df.empty:
            st.write("Lipids excluded from the plot (fold change zero or infinity):")
            st.dataframe(removed_lipids_df)
        else:
            st.write("No invalid lipids found.")

    return volcano_plots

def display_lipidomic_heatmap(experiment, continuation_df):
    """
    Displays a lipidomic heatmap in the Streamlit app, offering an interactive interface for users to 
    select specific conditions and lipid classes for visualization. This function facilitates the exploration 
    of lipidomic data by generating both clustered and regular heatmaps based on user input.
    Args:
        experiment (Experiment): An object containing detailed information about the experiment setup.
        continuation_df (pd.DataFrame): A DataFrame containing processed lipidomics data.
    Returns:
        tuple: A tuple containing the regular heatmap figure and the clustered heatmap figure (if generated).
    """
    regular_heatmap = None
    clustered_heatmap = None
    with st.expander("Lipidomic Heatmap"):
        # UI for selecting conditions and classes
        all_conditions = experiment.conditions_list
        selected_conditions = st.multiselect("Select conditions:", all_conditions, default=all_conditions)
        selected_conditions = [condition for condition in selected_conditions if len(experiment.individual_samples_list[experiment.conditions_list.index(condition)]) > 1]
        all_classes = continuation_df['ClassKey'].unique()
        selected_classes = st.multiselect("Select lipid classes:", all_classes, default=all_classes)
        
        # UI for selecting number of clusters
        n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=5)
        
        if selected_conditions and selected_classes:
            # Extract sample names based on selected conditions
            selected_samples = [sample for condition in selected_conditions 
                                for sample in experiment.individual_samples_list[experiment.conditions_list.index(condition)]]
            # Process data for heatmap generation
            filtered_df, _ = lp.LipidomicHeatmap.filter_data(continuation_df, selected_conditions, selected_classes, experiment.conditions_list, experiment.individual_samples_list)
            z_scores_df = lp.LipidomicHeatmap.compute_z_scores(filtered_df)
            
            # Generate both regular and clustered heatmaps
            regular_heatmap = lp.LipidomicHeatmap.generate_regular_heatmap(z_scores_df, selected_samples)
            clustered_heatmap, class_percentages = lp.LipidomicHeatmap.generate_clustered_heatmap(z_scores_df, selected_samples, n_clusters)
            
            # Apply layout updates to both heatmaps
            for heatmap, heatmap_type in [(regular_heatmap, "Regular"), (clustered_heatmap, "Clustered")]:
                heatmap.update_layout(
                    width=900,
                    height=600,
                    margin=dict(l=150, r=50, t=50, b=100),
                    xaxis_tickangle=-45,
                    yaxis_title="Lipid Molecules",
                    xaxis_title="Samples",
                    title=f"Lipidomic Heatmap ({heatmap_type})",
                    font=dict(size=10)
                )
                heatmap.update_traces(colorbar=dict(len=0.9, thickness=15))
            
            # Allow users to choose which heatmap to display
            heatmap_type = st.radio("Select Heatmap Type", ["Clustered", "Regular"], index=0)
            current_heatmap = clustered_heatmap if heatmap_type == "Clustered" else regular_heatmap
            
            # Display the selected heatmap
            st.plotly_chart(current_heatmap, use_container_width=True)
            
            # Display cluster information
            if heatmap_type == "Clustered":
                st.subheader(f"Cluster Information (Number of clusters: {n_clusters})")
                st.dataframe(class_percentages.round(2))
                
                # Create a downloadable CSV for cluster information
                csv_cluster_info = convert_df(class_percentages.reset_index())
                st.download_button("Download Cluster Information CSV", csv_cluster_info, 'cluster_information.csv', 'text/csv')
            
            # Download buttons
            plotly_svg_download_button(current_heatmap, f"Lipidomic_{heatmap_type}_Heatmap.svg")
            csv_download = convert_df(z_scores_df.reset_index())
            st.download_button("Download CSV", csv_download, f'z_scores_{heatmap_type}_heatmap.csv', 'text/csv')
    return regular_heatmap, clustered_heatmap

def generate_pdf_report(box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot, heatmap_figs, correlation_plots, abundance_bar_charts, abundance_pie_charts, saturation_plots, volcano_plots, pathway_visualization):
    pdf_buffer = io.BytesIO()
    
    try:
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Page 1: Box Plots
        img_buffer1 = io.BytesIO()
        box_plot_fig1.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
        img_buffer1.seek(0)
        img1 = ImageReader(img_buffer1)
        pdf.drawImage(img1, 50, 400, width=500, height=300, preserveAspectRatio=True)
        
        img_buffer2 = io.BytesIO()
        box_plot_fig2.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
        img_buffer2.seek(0)
        img2 = ImageReader(img_buffer2)
        pdf.drawImage(img2, 50, 50, width=500, height=300, preserveAspectRatio=True)
        
        # Page 2: BQC Plot
        pdf.showPage()
        if bqc_plot is not None:
            bqc_bytes = pio.to_image(bqc_plot, format='png', width=800, height=600, scale=2)
            bqc_img = ImageReader(io.BytesIO(bqc_bytes))
            pdf.drawImage(bqc_img, 50, 100, width=500, height=400, preserveAspectRatio=True)
        
        # Page 3: Retention Time Plot
        pdf.showPage()
        pdf.setPageSize(landscape(letter))
        if retention_time_plot is not None:
            svg_bytes = pio.to_image(retention_time_plot, format='svg')
            drawing = svg2rlg(io.BytesIO(svg_bytes))
            renderPDF.draw(drawing, pdf, 50, 50)
        
        # Page 4: PCA Plot
        pdf.showPage()
        pdf.setPageSize(landscape(letter))
        if pca_plot is not None:
            svg_bytes = pio.to_image(pca_plot, format='svg')
            drawing = svg2rlg(io.BytesIO(svg_bytes))
            renderPDF.draw(drawing, pdf, 50, 50)
        
        # Pages for Correlation Plots
        for condition, corr_fig in correlation_plots.items():
            pdf.showPage()
            pdf.setPageSize(letter)
            img_buffer = io.BytesIO()
            corr_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            pdf.drawImage(img, 50, 100, width=500, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 50, f"Correlation Plot for {condition}")
        
        # Add Abundance Bar Charts
        for scale, chart in abundance_bar_charts.items():
            if chart is not None:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                img_buffer = io.BytesIO()
                chart.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                img = ImageReader(img_buffer)
                pdf.drawImage(img, 50, 50, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 30, f"Abundance Bar Chart ({scale} scale)")
        
        # Add Abundance Pie Charts
        for condition, pie_chart in abundance_pie_charts.items():
            pdf.showPage()
            pdf.setPageSize(letter)
            pie_bytes = pio.to_image(pie_chart, format='png', width=800, height=600, scale=2)
            pie_img = ImageReader(io.BytesIO(pie_bytes))
            pdf.drawImage(pie_img, 50, 100, width=500, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 50, f"Abundance Pie Chart for {condition}")
        
        # Add Saturation Plots
        for lipid_class, plots in saturation_plots.items():
            if isinstance(plots, dict) and 'main' in plots and 'percentage' in plots:
                # Main plot
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                main_bytes = pio.to_image(plots['main'], format='png', width=1000, height=700, scale=2)
                main_img = ImageReader(io.BytesIO(main_bytes))
                pdf.drawImage(main_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, f"Saturation Plot (Main) for {lipid_class}")
                
                # Percentage plot
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                percentage_bytes = pio.to_image(plots['percentage'], format='png', width=1000, height=700, scale=2)
                percentage_img = ImageReader(io.BytesIO(percentage_bytes))
                pdf.drawImage(percentage_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, f"Saturation Plot (Percentage) for {lipid_class}")
        
        # Add Volcano Plots
        if volcano_plots:
            # Main Volcano Plot
            if 'main' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                main_bytes = pio.to_image(volcano_plots['main'], format='png', width=1000, height=700, scale=2)
                main_img = ImageReader(io.BytesIO(main_bytes))
                pdf.drawImage(main_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Volcano Plot")

            # Concentration vs Fold Change Plot
            if 'concentration_vs_fold_change' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                conc_bytes = pio.to_image(volcano_plots['concentration_vs_fold_change'], format='png', width=1000, height=700, scale=2)
                conc_img = ImageReader(io.BytesIO(conc_bytes))
                pdf.drawImage(conc_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Concentration vs Fold Change Plot")

            # Concentration Distribution Plot
            if 'concentration_distribution' in volcano_plots:
                pdf.showPage()
                pdf.setPageSize(landscape(letter))
                dist_buffer = io.BytesIO()
                volcano_plots['concentration_distribution'].savefig(dist_buffer, format='png', dpi=300, bbox_inches='tight')
                dist_buffer.seek(0)
                dist_img = ImageReader(dist_buffer)
                pdf.drawImage(dist_img, 50, 100, width=700, height=500, preserveAspectRatio=True)
                pdf.drawString(50, 80, "Concentration Distribution Plot")
        
        # Add Pathway Visualization
        if pathway_visualization is not None:
            pdf.showPage()
            pdf.setPageSize(landscape(letter))
            img_buffer = io.BytesIO()
            pathway_visualization.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img = ImageReader(img_buffer)
            pdf.drawImage(img, 50, 50, width=700, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 30, "Lipid Pathway Visualization")
        
        # Add Lipidomic Heatmaps
        if 'Regular Heatmap' in heatmap_figs:
            add_heatmap_to_pdf(pdf, heatmap_figs['Regular Heatmap'], "Regular Lipidomic Heatmap")
        
        if 'Clustered Heatmap' in heatmap_figs:
            add_heatmap_to_pdf(pdf, heatmap_figs['Clustered Heatmap'], "Clustered Lipidomic Heatmap")
        
        pdf.save()
        pdf_buffer.seek(0)
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return pdf_buffer

def add_heatmap_to_pdf(pdf, heatmap_fig, title):
    pdf.showPage()
    pdf.setPageSize(landscape(letter))
    heatmap_bytes = pio.to_image(heatmap_fig, format='png', width=900, height=600, scale=2)
    heatmap_img = ImageReader(io.BytesIO(heatmap_bytes))
    page_width, page_height = landscape(letter)
    img_width = 700
    img_height = (img_width / 900) * 600
    x_position = (page_width - img_width) / 2
    y_position = (page_height - img_height) / 2
    pdf.drawImage(heatmap_img, x_position, y_position, width=img_width, height=img_height, preserveAspectRatio=True)
    pdf.drawString(x_position, y_position - 20, title)

if __name__ == "__main__":
    main()