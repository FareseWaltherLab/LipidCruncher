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
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# Local imports
import lipidomics as lp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where main_app.py is
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')  # Path to the images directory

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
        
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

def display_landing_page():
    """Display the LipidCruncher landing page with enhanced module explanations."""
    # Load and display the logo
    try:
        logo_path = os.path.join(IMAGES_DIR, 'logo.tif')
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=720)
        else:
            st.error(f"Logo file not found at {logo_path}")
            st.header("LipidCruncher")
    except Exception as e:
        st.error(f"Failed to load logo: {str(e)}")
        st.header("LipidCruncher")

    st.markdown("""
    **LipidCruncher** is an open-source web platform developed by **[The Farese and Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther)** to transform lipidomic data analysis. 
    Designed to overcome traditional challenges like manual spreadsheet handling and insufficient quality assessment, LipidCruncher offers a comprehensive 
    solution that streamlines data standardization, normalization, and rigorous quality control while providing powerful visualization tools. 
    The platform significantly accelerates the research iteration process, allowing scientists to quickly identify anomalies and patterns through advanced 
    features including volcano plots, lipid saturation profiles, pathway mapping, and interactive heatmaps—all within an intuitive interface that requires 
    no bioinformatics expertise. LipidCruncher empowers researchers to efficiently convert complex lipid datasets into scientific insights, regardless 
    of their specific research focus.
    """, unsafe_allow_html=True)

    st.subheader("Key Features")
    st.markdown("""
    - **Versatile Data Input**: Import datasets from LipidSearch, Metabolomics Workbench, or generic CSV formats.
    - **Robust Processing**: Standardize, filter, and normalize data with options tailored to your experiment.
    - **Rigorous Quality Control**: Ensure data integrity with diagnostic tools like box plots, CoV analysis, and PCA.
    - **Advanced Visualizations**: Gain insights through interactive volcano plots, heatmaps, and pathway maps.
    - **Open Access**: Freely available online with source code on The Farese and Walther Lab [GitHub](https://github.com/FareseWaltherLab/LipidCruncher).
    """)

    st.subheader("How It Works")
    st.markdown("""
    LipidCruncher organizes the lipidomics analysis pipeline into three integrated modules, each designed to simplify and standardize critical steps from raw data to scientific insight:
    """)

    st.markdown("**Module 1: Data Input, Standardization, Filtering, and Normalization**")
    st.markdown("""
    - **Data Input**: Import data in flexible formats, including generic CSV, LipidSearch, and Metabolomics Workbench.
    - **Standardization**: Automatically align column naming for consistency across datasets.
    - **Filtering**: Clean data by removing empty rows, duplicates, and replacing null values with zeros.
    - **Normalization**: Choose from four options: no normalization, internal standard-based, protein-based, or combined internal standard and protein normalization.
    """)

    st.markdown("**Module 2: Quality Check and Anomaly Detection**")
    st.markdown("""
    - **Box Plots**: Visualize concentration distributions to confirm uniformity among replicates.
    - **Zero-Value Analysis**: Identify samples with excessive missing data.
    - **CoV Analysis**: Assess measurement precision using batch quality control (BQC) samples.
    - **Correlation Analysis**: Identify outliers through pairwise correlations between replicates.
    - **PCA**: Visualize sample clustering and flag potential outliers.
    """)

    st.markdown("**Module 3: Data Visualization, Interpretation, and Analysis**")
    st.markdown("""
    - **Bar and Pie Charts**: Display lipid class concentrations and proportional distributions.
    - **Metabolic Network Visualization**: Map lipid class changes in a metabolic context.
    - **Saturation Profiles**: Analyze fatty acid saturation levels (SFA, MUFA, PUFA).
    - **Volcano Plots**: Highlight significant lipid changes with customizable thresholds.
    - **Heatmaps**: Provide a high-resolution view of lipidomic alterations with interactive features.
    """)

    st.subheader("Pipeline Overview")
    try:
        pdf_path = os.path.join(IMAGES_DIR, 'figure1.pdf')
        images = convert_from_path(pdf_path, dpi=300)
        if images:
            fig1_image = images[0]
            img_byte_arr = io.BytesIO()
            fig1_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            st.image(img_byte_arr, caption="Overview of the LipidCruncher Analysis Pipeline", use_column_width=True)
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
        else:
            st.warning("No pages found in Figure 1 PDF.")
    except Exception as e:
        st.warning(f"Could not load Figure 1 PDF: {str(e)}. Please ensure 'figure1.pdf' is in the './images/' directory and Poppler is installed.")

    st.subheader("Learn More")
    
    st.markdown("""
    To explore LipidCruncher and see how we utilized it in a case study, 
    read our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1) published on bioRxiv.
    """, unsafe_allow_html=True)
    
    st.subheader("Get Started")
    st.markdown("""
    Upload your lipidomic dataset to begin analyzing it with LipidCruncher.
    """)
    
    # Center the "Start Crunching" button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Start Crunching"):
            st.session_state.page = 'app'
            st.experimental_rerun()
    
    st.subheader("Test Datasets")
    st.markdown("""
    Want to explore LipidCruncher but don't have your own dataset? 
    Try our sample datasets available on our lab's [GitHub](https://github.com/FareseWaltherLab/LipidCruncher/tree/main/sample_datasets). 
    These datasets are from the case study analyzed in our [paper](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1.article-metrics) 
    describing LipidCruncher's features.
    
    The case study experiment includes three conditions with four replicates each:
    - **WT** (Wild Type): samples s1-s4
    - **ADGAT-DKO** (Double Knockout): samples s5-s8
    - **BQC** (Batch Quality Control): samples s9-s12
    """)
                
    st.subheader("Support")
    st.markdown("""
    For reporting bugs, feature requests, or any questions or comments, please email abdih@mskcc.org.
    """)

def main():
    """Main function for the Lipidomics Analysis Module Streamlit application."""
    initialize_session_state()

    if st.session_state.page == 'landing':
        display_landing_page()
    elif st.session_state.page == 'app':
        try:
            logo_path = os.path.join(IMAGES_DIR, 'logo.tif')
            if os.path.exists(logo_path):
                logo = Image.open(logo_path)
                st.image(logo, width=720)
            else:
                st.error(f"Logo file not found at {logo_path}")
                st.header("LipidCruncher")
        except Exception as e:
            st.error(f"Failed to load logo: {str(e)}")
            st.header("LipidCruncher")
        
        st.markdown("Process, analyze and visualize lipidomics data from multiple sources.")
        
        data_format = display_format_selection()
        display_format_requirements(data_format)
        
        st.session_state.format_type = data_format
        
        file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
        
        uploaded_file = st.sidebar.file_uploader(
            f'Upload your {data_format} dataset', 
            type=file_types
        )
        
        if uploaded_file:
            df = load_and_validate_data(uploaded_file, data_format)
            if df is not None:
                if data_format == 'Generic Format':
                    display_column_mapping()
                
                confirmed, name_df, experiment, bqc_label, valid_samples, updated_df = process_experiment(df, data_format)
                
                st.session_state.confirmed = confirmed
                
                if valid_samples:
                    if confirmed:
                        update_session_state(name_df, experiment, bqc_label)
                        
                        df_to_clean = updated_df if updated_df is not None else df
                        
                        if st.session_state.module == "Data Cleaning, Filtering, & Normalization":
                            st.subheader("Data Standardization, Filtering, and Normalization Module")
                            cleaned_df, intsta_df = clean_data(df_to_clean, name_df, experiment, data_format)
                            
                            if cleaned_df is not None:
                                st.session_state.experiment = experiment
                                st.session_state.format_type = data_format
                                st.session_state.cleaned_df = cleaned_df
                                st.session_state.intsta_df = intsta_df
                                st.session_state.continuation_df = cleaned_df
                                
                                display_cleaned_data(cleaned_df, intsta_df)
                                
                                normalized_df = handle_data_normalization(
                                    cleaned_df, 
                                    st.session_state.intsta_df,
                                    experiment, 
                                    data_format
                                )
                                
                                if normalized_df is not None:
                                    st.session_state.normalized_df = normalized_df
                                    st.session_state.continuation_df = normalized_df
                                    
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
                            
                            if st.button("Back to Data Standardization, Filtering, and Normalization", key="back_to_cleaning"):
                                st.session_state.module = "Data Cleaning, Filtering, & Normalization"
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
                        clear_session_state()
                        st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
                else:
                    st.error("Please ensure your samples are valid before proceeding.")

        # Restore preserved state if returning from Quality Check & Analysis
        if 'preserved_data' in st.session_state and st.session_state.module == "Data Cleaning, Filtering, & Normalization":
            for key, value in st.session_state.preserved_data.items():
                setattr(st.session_state, key, value)
            del st.session_state.preserved_data
        
        # Add a button to return to the landing page
        if st.button("Back to Home"):
            st.session_state.page = 'landing'
            st.experimental_rerun()
        
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
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0']
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
           * Will be standardized to match te standard format. For example:
             - LPC O-18:1         -> LPC(O-18:1)
             - Cer d18:0/C24:0    -> Cer(d18:0_C24:0)
             - CE 14:0;0          -> CE(14:0)
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
    - CSV file (.csv)
    - Must contain column: 'LipidMolec'
    - The standards must exist in your dataset
    """)
    
    uploaded_file = st.file_uploader("Upload CSV file with standards", type=['csv'])
    if uploaded_file is not None:
        try:
            standards_df = pd.read_csv(uploaded_file)
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
        # Add checkbox for showing detailed information
        show_cleaning_info = st.checkbox("Show data cleaning process details", key="show_cleaning_info")
        
        if show_cleaning_info:
            st.markdown("### Data Cleaning and Standardization Process")
            st.markdown("""
            LipidCruncher performs a systematic cleaning and standardization process on your uploaded data 
            to ensure consistency, reliability, and compatibility with downstream analyses. The specific 
            procedures applied depend on your data format.
            """)
            
            # Tabs for different data formats
            data_format_tab = st.radio(
                "Select data format to view cleaning details:",
                ["LipidSearch Format", "Generic Format", "Metabolomics Workbench"],
                horizontal=True
            )
            
            if data_format_tab == "LipidSearch Format":
                st.markdown("#### Data Cleaning for LipidSearch Format")
                st.markdown("""
                For LipidSearch data, we perform the following standardization and cleaning steps:
                
                1. **Column Standardization**: We extract and standardize essential columns including LipidMolec, 
                   ClassKey, CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey, and all MeanArea 
                   columns for each sample.
                
                2. **Data Type Conversion**: MeanArea columns are converted to numeric format, with any 
                   non-numeric entries replaced by zeros to maintain data integrity.
                
                3. **Lipid Name Standardization**: Names of lipid molecules are standardized to ensure 
                   uniform formatting across the dataset.
                
                4. **Quality Filtering**: Only entries with grades 'A', 'B', or 'C' are retained. 
                   Grade 'C' is only accepted for specific lipid classes (LPC and SM).
                
                5. **Best Peak Selection**: For each unique lipid, the entry with the highest TotalSmpIDRate(%) 
                   is selected, as this indicates the most reliable measurement across samples.
                
                6. **Missing FA Keys Handling**: Rows with missing FA keys are removed, with exceptions 
                   for cholesterol (Ch) class molecules and deuterated standards.
                
                7. **Zero Value Handling**: Rows where all intensity values are zero or null are removed.
                
                8. **Duplicate Removal**: Duplicate entries based on LipidMolec are removed.
                """)
                
                st.info("This process ensures that only high-quality, consistent data points are retained for analysis.")
                
            elif data_format_tab == "Generic Format":
                st.markdown("#### Data Cleaning for Generic Format")
                st.markdown("""
                For Generic Format data, we perform the following standardization and cleaning steps:
                
                1. **Column Standardization**: The first column is standardized as 'LipidMolec', and remaining 
                   columns are formatted as 'intensity[s1]', 'intensity[s2]', etc.
                
                2. **Lipid Name Standardization**: Lipid names are standardized to follow a consistent format: 
                   Class(chain details). For example:
                   - LPC O-17:4 → LPC(O-17:4)
                   - Cer d18:0/C24:0 → Cer(d18:0_C24:0)
                   - CE 14:0;0 → CE(14:0)
                
                3. **Class Key Extraction**: A 'ClassKey' column is generated by extracting the lipid class 
                   from the standardized lipid name (e.g., 'PC' from 'PC(16:0_18:1)').
                
                4. **Data Type Conversion**: Intensity columns are converted to numeric format, with any 
                   non-numeric entries replaced by zeros.
                
                5. **Invalid Lipid Removal**: Rows with invalid lipid names (empty strings, single special 
                   characters, strings with only special characters) are removed.
                
                6. **Zero Value Handling**: Rows where all intensity values are zero or null are removed.
                
                7. **Duplicate Removal**: Duplicate entries based on LipidMolec are removed.
                """)
                
                st.info("This standardization process allows for consistent analysis regardless of the original format of your data.")
                
            else:  # Metabolomics Workbench
                st.markdown("#### Data Cleaning for Metabolomics Workbench Format")
                st.markdown("""
                For Metabolomics Workbench data, we perform the following standardization and cleaning steps:
                
                1. **Section Extraction**: Data is extracted from between the 'MS_METABOLITE_DATA_START' and 
                   'MS_METABOLITE_DATA_END' markers.
                
                2. **Header Processing**: The first row is processed as sample names, and the second row as 
                   experimental conditions.
                
                3. **Column Standardization**: The first column is standardized as 'LipidMolec', and remaining 
                   columns are formatted as 'intensity[s1]', 'intensity[s2]', etc.
                
                4. **Lipid Name Standardization**: Lipid names are standardized to follow a consistent format, 
                   similar to the Generic Format process.
                
                5. **Class Key Extraction**: A 'ClassKey' column is generated by extracting the lipid class 
                   from the standardized lipid name.
                
                6. **Data Type Conversion**: Intensity columns are converted to numeric format, with any 
                   non-numeric entries replaced by zeros.
                
                7. **Experimental Conditions Storage**: The experimental conditions from the second row are 
                   stored and used to suggest the experimental setup.
                """)
                
                st.info("This process ensures that your Metabolomics Workbench data is properly formatted for analysis in LipidCruncher.")
            
            # Add a divider
            st.markdown("---")
        
        # Display the cleaned data
        st.subheader("Cleaned Data")
        st.write("This table shows your data after cleaning and standardization:")
        display_data(st.session_state.cleaned_df, "Cleaned Data", "cleaned_data.csv")

    # Internal standards management
    with st.expander("Manage Internal Standards"):
        # Add checkbox for showing internal standards information
        show_standards_info = st.checkbox("Show internal standards detection details", key="show_standards_info")
        
        if show_standards_info:
            st.markdown("### Internal Standards Detection")
            st.markdown("""
            LipidCruncher automatically identifies internal standards from the SPLASH LIPIDOMIX® Mass Spec Standard (Avanti Polar Lipids, Cat# 330707-1) 
            by detecting patterns like "+D7" or ":(s)" notation in lipid names. If you use custom standards with different naming conventions, you can upload them using the 
            option below.
            """)
            
            if not st.session_state.intsta_df.empty:
                st.success(f"✓ {len(st.session_state.intsta_df)} internal standards were automatically detected in your dataset.")
            else:
                st.warning("No internal standards were automatically detected in your dataset.")
                
            # Add a divider
            st.markdown("---")
        
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
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = all_class_lst.copy()
    
    # Callback function to update session state
    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes
    
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        options=all_class_lst,
        default=all_class_lst,  # Set default to include all classes
        key='temp_selected_classes',
        on_change=update_selected_classes
    )
    
    # Use the direct selection result instead of session state
    if not selected_classes:
        st.warning("Please select at least one lipid class to proceed with normalization.")
        return None

    # Filter DataFrame based on selected classes
    filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()

    # Check if we have standards from earlier selection
    has_standards = not st.session_state.intsta_df.empty

    # Determine available normalization options
    normalization_options = ['None', 'Internal Standards', 'Protein-based', 'Both'] if has_standards else ['None', 'Protein-based']
    if not has_standards:
        st.warning("No internal standards available. Only protein-based normalization is available.")

    # Add normalization explanation 
    with st.expander("About Normalization Methods"):
        st.markdown("### Data Normalization Methods")
        st.markdown("""
        LipidCruncher offers four normalization methods to adjust your lipidomic data:
        
        **None**: Use raw intensity values without normalization. This is suitable if your data has already been normalized externally.
        
        **Internal Standards**: Normalize lipid measurements using spiked-in internal standards of known concentration. For each lipid class, you'll select an appropriate internal standard. The normalization formula is:
        ```
        Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard
        ```
        
        **Protein-based**: Normalize lipid intensities against protein concentration (e.g., determined by a BCA assay). This adjusts for differences in starting material. The normalization formula is:
        ```
        Concentration = Intensity_lipid / Protein_concentration
        ```
        
        **Both**: Apply both internal standards and protein normalization:
        ```
        Concentration = (Intensity_lipid / Intensity_standard) × (Concentration_standard / Protein_concentration)
        ```
        
        After normalization, intensity columns are renamed to concentration columns to reflect that values now represent absolute or relative lipid concentrations rather than raw intensities.
        """)

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
            
            # Handle Protein-based normalization
            if normalization_method in ['Protein-based', 'Both']:
                with st.expander("Enter Protein Concentration Data"):
                    protein_df = collect_protein_concentrations(experiment)
                    if protein_df is not None:
                        try:
                            normalized_df = normalized_data_object.normalize_using_bca(
                                normalized_df, 
                                protein_df, 
                                preserve_prefix=True
                            )
                            # Store protein-based normalization settings
                            st.session_state.normalization_settings['protein'] = {
                                'protein_df': protein_df
                            }
                            st.success("Protein-based normalization applied successfully")
                        except Exception as e:
                            st.error(f"Error during protein-based normalization: {str(e)}")
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
    st.subheader("Quality Check and Anomaly Detection Module")
    
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
    
    st.subheader("Data Visualization, Interpretation, and Analysis Module")
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
        "4. Avoid interacting with the app during PDF generation.\n"
    )

    generate_pdf = st.radio("Would you like to generate a PDF report?", ('No', 'Yes'), index=0)

    if generate_pdf == 'Yes':
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
                
                # Close only matplotlib figures, not Plotly figures
                plt.close('all')  # This closes all matplotlib figures
                
            else:
                st.error("Failed to generate PDF report. Please check the logs for details.")
        else:
            st.warning("Some plots are missing. Unable to generate PDF report.")
            
def display_box_plots(continuation_df, experiment):
    """
    Display box plots using Plotly for improved interactivity and readability.
    
    Args:
        continuation_df (pd.DataFrame): The dataset to visualize
        experiment (Experiment): Experiment object containing sample information
        
    Returns:
        tuple: Two Plotly figure objects (missing values plot, box plot)
    """
    # Initialize a counter in session state if it doesn't exist
    if 'box_plot_counter' not in st.session_state:
        st.session_state.box_plot_counter = 0
    
    # Increment the counter
    st.session_state.box_plot_counter += 1
    
    # Generate a unique identifier based on the current data and counter
    unique_id = hashlib.md5(
        f"{str(continuation_df.index.tolist())}_{st.session_state.box_plot_counter}".encode()
    ).hexdigest()
    
    expand_box_plot = st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns')
    with expand_box_plot:
        # Add explanation about box plots
        show_boxplot_info = st.checkbox("Show box plot analysis details", key="show_boxplot_info")
        if show_boxplot_info:
            st.markdown("### Box Plot Analysis")
            st.markdown("""
            Box plots are powerful diagnostic tools that help assess data quality and identify potential anomalies in your lipidomic dataset. LipidCruncher generates two complementary visualizations:
            
            **1. Missing Values Distribution:**
            This plot shows the percentage of missing (zero) values for each sample in your dataset. A higher percentage may indicate:
            - Lower sensitivity for that sample during analysis
            - Technical issues during sample preparation or data acquisition
            - Biological differences resulting in fewer detectable lipids
            
            Ideally, samples from the same experimental condition should show similar percentages. Substantial differences between replicates could signal potential quality issues.
            
            **2. Box Plot of Non-Zero Concentrations:**
            This visualization displays the distribution of non-zero concentration values for each sample using standard box plot elements:
            - The box shows the interquartile range (25th to 75th percentile)
            - The horizontal line inside the box represents the median
            - The whiskers extend to the most extreme data points within 1.5 times the interquartile range
            - Individual points beyond the whiskers represent potential outliers
            
            What to look for:
            - Similar median values and box sizes across replicates of the same condition indicate good reproducibility
            - Unusual distributions (very high/low median, wide/narrow box) may suggest technical issues
            - Consistent differences between experimental conditions may reflect genuine biological effects
            
            These visualizations help you make informed decisions about the quality and reliability of your lipidomic data before proceeding with further analysis.
            """)
            # Add a divider
            st.markdown("---")
        
        # Creating a deep copy for visualization
        visualization_df = continuation_df.copy(deep=True)
        
        # Ensure the columns reflect the current state of the DataFrame
        current_samples = [
            sample for sample in experiment.full_samples_list 
            if f'concentration[{sample}]' in visualization_df.columns
        ]
        
        mean_area_df = lp.BoxPlot.create_mean_area_df(visualization_df, current_samples)
        zero_values_percent_list = lp.BoxPlot.calculate_missing_values_percentage(mean_area_df)
        
        st.subheader("Missing Values Distribution")
        
        # Generate and display the first plot (Missing Values Distribution)
        fig1 = lp.BoxPlot.plot_missing_values(current_samples, zero_values_percent_list)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Add SVG download button for the missing values plot
        plotly_svg_download_button(fig1, "missing_values_distribution.svg")
        
        # Data for download (Missing Values)
        missing_values_data = np.vstack((current_samples, zero_values_percent_list)).T
        missing_values_df = pd.DataFrame(missing_values_data, columns=["Sample", "Percentage Missing"])
        missing_values_csv = convert_df(missing_values_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download CSV",
                data=missing_values_csv,
                file_name="missing_values_data.csv",
                mime='text/csv',
                key=f"download_missing_values_data_{unique_id}"
            )
        
        st.markdown("---")
        
        st.subheader("Box Plot of Non-Zero Concentrations")
        
        # Generate and display the second plot (Box Plot)
        fig2 = lp.BoxPlot.plot_box_plot(mean_area_df, current_samples)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Add SVG download button for the box plot
        plotly_svg_download_button(fig2, "box_plot.svg")
        
        # Provide option to download the raw data for Box Plot
        csv_data = convert_df(mean_area_df)
        
        col1, col2 = st.columns(2)
        with col1:
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
            # Add explanation about BQC analysis
            show_bqc_info = st.checkbox("Show BQC analysis details", key="show_bqc_info")
            if show_bqc_info:
                st.markdown("### Batch Quality Control (BQC) Analysis")
                st.markdown("""
                BQC analysis assesses the measurement reliability of your lipidomic dataset using Coefficient of Variation (CoV) calculations on BQC samples.
                
                **What is CoV?**  
                The Coefficient of Variation measures the relative variability of data by calculating:
                ```
                CoV = (Standard Deviation / Mean) × 100%
                ```
                
                It is calculated for each lipid species across all BQC samples (excluding zero values). Lower CoV values indicate higher measurement precision and reliability.
                
                **The CoV Scatter Plot:**
                - X-axis: Log10 of the mean concentration for each lipid species in BQC samples
                - Y-axis: CoV percentage for each lipid species
                - Each point represents a single lipid species
                - Hover over points to see the specific lipid, its mean concentration, and CoV value
                - Points above the threshold (red) indicate potentially less reliable measurements
                - The horizontal black line shows the selected CoV threshold
                
                **Interpreting CoV Values:**
                - CoV < 20%: Excellent measurement precision
                - CoV < 30%: Good measurement precision
                - CoV > 30%: Potential issues with measurement reliability
                
                **Filtering Option:**
                LipidCruncher allows you to filter the dataset to remove lipid species with high CoV values. This helps ensure that only reliable measurements are used in downstream analyses.
                
                **Reliability Metric:**
                The percentage of lipid species with CoV below the threshold is calculated to provide an overall data quality indicator:
                - ≥ 80%: Excellent data quality
                - 50-79%: Good data quality, but caution advised
                - < 50%: Potentially problematic dataset, careful interpretation required
                """)
                st.markdown("---")
            
            # Move threshold setting outside the filter option
            st.subheader("CoV Threshold Setting")
            cov_threshold = st.number_input(
                'Set CoV threshold (%)', 
                min_value=10, 
                max_value=1000, 
                value=30, 
                step=1,
                help="Data points above this threshold will be highlighted in red and can optionally be filtered out."
            )
            
            bqc_sample_index = experiment.conditions_list.index(bqc_label)
            
            # Generate and display the plot with the threshold
            scatter_plot, prepared_df, reliable_data_percent = lp.BQCQualityCheck.generate_and_display_cov_plot(
                data_df, 
                experiment, 
                bqc_sample_index,
                cov_threshold=cov_threshold
            )
            
            # Display the plot
            st.plotly_chart(scatter_plot, use_container_width=True)
            plotly_svg_download_button(scatter_plot, "bqc_quality_check.svg")
            csv_data = convert_df(prepared_df[['LipidMolec', 'cov', 'mean']].dropna())
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="CoV_Plot_Data.csv",
                mime='text/csv'
            )
            
            # Display reliability assessment with appropriate color coding
            if reliable_data_percent >= 80:
                st.success(f"{reliable_data_percent:.1f}% of the datapoints are confidently reliable (CoV < {cov_threshold}%).")
            elif reliable_data_percent >= 50:
                st.warning(f"{reliable_data_percent:.1f}% of the datapoints are confidently reliable (CoV < {cov_threshold}%).")
            else:
                st.error(f"Less than 50% of the datapoints are confidently reliable (CoV < {cov_threshold}%).")
            
            # Filtering section
            if prepared_df is not None and not prepared_df.empty:
                st.subheader("Data Filtering Options")
                filter_option = st.radio("Would you like to filter your data using BQC samples?", ("No", "Yes"), index=0)
                if filter_option == "Yes":
                    # Use the already selected threshold for filtering
                    st.info(f"Data will be filtered using the threshold of {cov_threshold}% CoV selected above.")
                    
                    if st.button("Apply Filter"):
                        # Get lipids that meet the threshold criteria (those with CoV < threshold)
                        reliable_lipids = prepared_df[prepared_df['cov'] < cov_threshold]['LipidMolec'].tolist()
                        
                        # Get all lipids that will be filtered out
                        all_lipids_in_original = data_df['LipidMolec'].tolist()
                        filtered_out_lipids = [lipid for lipid in all_lipids_in_original if lipid not in reliable_lipids]
                        
                        # Filter the original dataframe to keep only reliable lipids
                        filtered_df = data_df[data_df['LipidMolec'].isin(reliable_lipids)].reset_index(drop=True)
                        
                        if filtered_df.empty:
                            st.error("The filtered dataset is empty. Please try a higher CoV threshold.")
                            st.warning("Returning the original dataset without filtering.")
                        else:
                            # Calculate statistics
                            removed_count = len(filtered_out_lipids)
                            percentage_kept = round((len(filtered_df) / len(data_df)) * 100, 1)
                            
                            st.info(f"Filtering removed {removed_count} lipid species ({100 - percentage_kept:.1f}% of the dataset).")
                            
                            # Show list of filtered lipids
                            if filtered_out_lipids:
                                show_filtered_list = st.checkbox("Show list of filtered lipids", key="show_filtered_list")
                                
                                if show_filtered_list:
                                    # Create DataFrame of filtered lipids
                                    filtered_out_df = pd.DataFrame({'LipidMolec': filtered_out_lipids})
                                    st.write("**Lipids removed by filtering:**")
                                    st.dataframe(filtered_out_df)
                                    
                                    # Add download button for filtered lipids
                                    csv_filtered = convert_df(filtered_out_df)
                                    st.download_button(
                                        label="Download filtered lipids list",
                                        data=csv_filtered,
                                        file_name="filtered_out_lipids.csv",
                                        mime="text/csv"
                                    )
                            
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
        with st.expander('View Retention Time Plots: Check Sanity of Data'):
            # Add explanation about retention time analysis
            show_rt_info = st.checkbox("Show retention time analysis details", key="show_rt_info")
            if show_rt_info:
                st.markdown("### Retention Time Analysis")
                st.markdown("""
                Retention time analysis is a crucial quality check for LipidSearch data. This visualization plots the retention time of each lipid against its calculated mass, allowing you to verify the consistency and reliability of lipid identification.
                
                **What is Retention Time?**  
                Retention time is the duration a molecule takes to travel through a chromatography column. It directly correlates with a lipid's hydrophobicity—more hydrophobic lipids interact more strongly with the column and typically have longer retention times.
                
                **What to Look For:**
                
                1. **Class-specific Clustering**: Lipids from the same class should form distinct clusters in the plot. Each lipid class has characteristic hydrophobicity patterns, resulting in similar retention times for molecules within that class.
                
                2. **Mass-Retention Time Relationship**: Within a lipid class:
                   - Longer fatty acid chains (higher mass) generally show longer retention times
                   - More saturated lipids (fewer double bonds) typically elute later than their unsaturated counterparts
                
                3. **Outliers**: Points that deviate significantly from their class's typical pattern may indicate:
                   - Incorrect lipid identification
                   - Co-eluting compounds
                   - Unusual structural features
                
                **Two Viewing Modes:**
                
                - **Individual Mode**: Displays separate plots for each lipid class, allowing detailed examination of class-specific patterns
                - **Comparison Mode**: Shows multiple lipid classes in a single plot with color coding, enabling direct comparison between classes
                
                This analysis helps confirm the analytical integrity of your data and can reveal potential misidentifications or chromatographic issues.
                """)
                st.markdown("---")
            
            # Get the viewing mode selection
            mode = st.radio('Select a viewing mode', ['Comparison Mode', 'Individual Mode'])
            
            if mode == 'Individual Mode':
                # Handling individual retention time plots
                plots = lp.RetentionTime.plot_single_retention(continuation_df)
                for idx, (plot, retention_df) in enumerate(plots, 1):
                    st.plotly_chart(plot, use_container_width=True)
                    plotly_svg_download_button(plot, f"retention_time_plot_{idx}.svg")
                    csv_download = convert_df(retention_df)
                    st.download_button(
                        label="Download CSV", 
                        data=csv_download, 
                        file_name=f'retention_plot_{idx}.csv', 
                        mime='text/csv'
                    )
                return None
            elif mode == 'Comparison Mode':
                # Handling comparison mode for retention time plots
                all_lipid_classes_lst = continuation_df['ClassKey'].value_counts().index.tolist()
                # Use multiselect with instruction text
                st.markdown("**Select lipid classes to compare:**")
                selected_classes_list = st.multiselect(
                    'Add/Remove classes for comparison:',
                    all_lipid_classes_lst, 
                    default=all_lipid_classes_lst[:min(5, len(all_lipid_classes_lst))]  # Default to first 5 classes or fewer
                )
                
                if selected_classes_list:  # Ensuring that selected_classes_list is not empty
                    plot, retention_df = lp.RetentionTime.plot_multi_retention(continuation_df, selected_classes_list)
                    if plot:
                        st.plotly_chart(plot, use_container_width=True)
                        plotly_svg_download_button(plot, "retention_time_comparison.svg")
                        csv_download = convert_df(retention_df)
                        st.download_button(
                            label="Download CSV", 
                            data=csv_download, 
                            file_name='Retention_Time_Comparison.csv', 
                            mime='text/csv'
                        )
                        return plot
                else:
                    st.warning("Please select at least one lipid class to generate the comparison plot.")
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
        # Add explanation about correlation analysis
        show_corr_info = st.checkbox("Show correlation analysis details", key="show_corr_info")
        if show_corr_info:
            st.markdown("### Pairwise Correlation Analysis")
            st.markdown("""
            Pairwise correlation analysis helps assess the reproducibility of your lipidomic measurements by calculating how closely related the measurements are between sample replicates.
            
            **What is Correlation Analysis?**  
            Correlation analysis calculates the Pearson correlation coefficient between pairs of samples. This coefficient:
            - Ranges from -1 to 1
            - Values close to 1 indicate strong positive correlation (similar patterns)
            - Values close to 0 indicate no correlation
            - Values close to -1 indicate strong negative correlation (opposite patterns)
            
            **The Correlation Heatmap:**
            - The heatmap displays correlation coefficients between all pairs of replicates
            - Blue colors indicate higher correlation (more similar)
            - Red colors indicate lower correlation (less similar)
            - Only the lower triangle is shown to avoid redundancy
            
            **Interpreting Correlation Values:**
            - For biological replicates:
              - Correlation ≥ 0.7: Good reproducibility
              - Correlation < 0.7: Potential issues with reproducibility
            
            - For technical replicates:
              - Correlation ≥ 0.8: Good reproducibility
              - Correlation < 0.8: Potential technical issues
            
            **Data Preprocessing:**
            - Missing (zero) values are removed before calculating correlations
            - This ensures that correlations are based only on lipids detected in both samples
            
            Low correlation between replicates may indicate:
            1. Sample preparation inconsistencies
            2. Instrument performance issues
            3. True biological variation
            4. Potential outlier samples
            
            This analysis helps identify samples that might need to be excluded or further investigated before proceeding with biological interpretation.
            """)
            st.markdown("---")
        
        # Filter out conditions with only one replicate
        multi_replicate_conditions = [condition for condition, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1]
        
        # Ensure there are multi-replicate conditions before proceeding
        if multi_replicate_conditions:
            selected_condition = st.selectbox('Select a condition', multi_replicate_conditions)
            condition_index = experiment.conditions_list.index(selected_condition)
            
            sample_type = st.selectbox(
                'Select the type of your samples', 
                ['biological replicates', 'technical replicates'],
                help="Biological replicates: different samples from the same condition. Technical replicates: repeated measurements of the same sample."
            )
            
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
            
            # Highlight potentially problematic correlations
            low_correlations = []
            min_threshold = 0.7 if sample_type == 'biological replicates' else 0.8
            
            for i in range(len(correlation_df.columns)):
                for j in range(i+1, len(correlation_df.columns)):
                    if correlation_df.iloc[j, i] < min_threshold:
                        low_correlations.append(
                            f"{correlation_df.columns[i]} and {correlation_df.index[j]}: {correlation_df.iloc[j, i]:.3f}"
                        )
            
            if low_correlations:
                st.warning("**Potential correlation issues detected:**")
                for corr in low_correlations:
                    st.write(f"• {corr}")
                st.write(f"Consider investigating these sample pairs as they show correlations below the recommended threshold ({min_threshold}) for {sample_type}.")
            else:
                st.success(f"All correlations are above the recommended threshold ({min_threshold}) for {sample_type}.")
                
            return selected_condition, fig
        else:
            st.error("No conditions with multiple replicates found. Correlation analysis requires at least two replicates.")
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
    from scipy.stats import chi2  # Add import here
    
    pca_plot = None
    with st.expander("Principal Component Analysis (PCA)"):
        # Add explanation about PCA analysis
        show_pca_info = st.checkbox("Show PCA analysis details", key="show_pca_info")
        if show_pca_info:
            st.markdown("### Principal Component Analysis (PCA)")
            st.markdown("""
            PCA is a powerful dimensionality reduction technique that helps visualize complex lipidomic datasets by transforming high-dimensional data into a set of linearly uncorrelated variables called principal components.
            
            **What is PCA?**  
            PCA identifies the directions (principal components) in which your data varies the most. These components are ordered by the amount of variance they explain:
            - Principal Component 1 (PC1) explains the largest portion of variance
            - Principal Component 2 (PC2) explains the second largest portion
            
            The percentage shown next to each PC indicates how much of the total variance that component explains.
            
            **The PCA Plot:**
            - Each point represents one sample
            - Points that cluster together have similar lipid profiles
            - Greater distances between points indicate greater differences in lipid profiles
            - Confidence ellipses (95% confidence) are drawn around each experimental condition
            
            **Interpreting the Plot:**
            - **Well-separated clusters**: Different experimental conditions show distinct lipid profiles
            - **Overlapping clusters**: Conditions have similar lipid profiles
            - **Tight clusters**: Good reproducibility within conditions
            - **Spread-out clusters**: Higher variability within conditions
            - **Outliers**: Samples falling outside their group's confidence ellipse may indicate anomalies
            
            **Data Preprocessing:**
            - Data is standardized (centered and scaled) before PCA
            - This ensures that variables with larger values don't dominate the analysis
            
            **When to Remove Samples:**
            Consider removing samples that:
            1. Fall far outside their expected cluster
            2. Don't group with their biological replicates
            3. Show unusual patterns confirmed by other quality metrics
            
            **Note:** Sample removal should be done cautiously and documented. Biological outliers may represent real variation rather than technical issues.
            """)
            st.markdown("---")
        
        # Sample removal interface
        samples_to_remove = st.multiselect(
            'Select samples to remove from the analysis (optional):',
            experiment.full_samples_list,
            help="Select any samples that you want to exclude from the PCA analysis. This is useful if you suspect certain samples are outliers."
        )
        
        # Validate if we have enough samples after removal
        if samples_to_remove:
            remaining_count = len(experiment.full_samples_list) - len(samples_to_remove)
            if remaining_count < 2:
                st.error('At least two samples are required for a meaningful PCA analysis!')
            else:
                continuation_df = experiment.remove_bad_samples(samples_to_remove, continuation_df)
                st.success(f"Analysis will proceed with {remaining_count} samples.")
        
        # Generate and display PCA plot
        pca_plot, pca_df = lp.PCAAnalysis.plot_pca(continuation_df, experiment.full_samples_list, experiment.extensive_conditions_list)
        st.plotly_chart(pca_plot, use_container_width=True)
        plotly_svg_download_button(pca_plot, "pca_plot.svg")
        
        # Show the PCA data table
        csv_data = convert_df(pca_df)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="PCA_data.csv",
            mime="text/csv"
        )
        
        # Add interpretation suggestions based on the plot
        for condition in pca_df['Condition'].unique():
            condition_df = pca_df[pca_df['Condition'] == condition]
            if len(condition_df) >= 3:  # Only check if we have at least 3 samples
                # Calculate the Mahalanobis distance for each point
                center = condition_df[['PC1', 'PC2']].mean().values
                cov = np.cov(condition_df['PC1'], condition_df['PC2'])
                
                # If the covariance matrix is singular, we can't compute Mahalanobis distances
                try:
                    inv_cov = np.linalg.inv(cov)
                    
                    # Check for potential outliers
                    potential_outliers = []
                    for _, row in condition_df.iterrows():
                        point = np.array([row['PC1'], row['PC2']])
                        dist = np.sqrt(np.dot(np.dot((point - center), inv_cov), (point - center).T))
                        # Using chi-square distribution with 2 degrees of freedom (for 2D) and p=0.05
                        if dist > np.sqrt(chi2.ppf(0.95, 2)):
                            potential_outliers.append(row['Sample'])
                    
                    if potential_outliers:
                        st.warning(f"**Potential outliers detected in {condition}:** {', '.join(potential_outliers)}")
                        st.write("These samples fall outside the 95% confidence ellipse. Consider examining them more closely.")
                except np.linalg.LinAlgError:
                    # Covariance matrix is singular
                    pass
        
    return continuation_df, pca_plot

def display_abundance_bar_charts(experiment, continuation_df):
    """
    Display abundance bar charts with statistical analysis for selected conditions and lipid classes.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomics data
        
    Returns:
        tuple: (linear_fig, log2_fig) containing the Plotly figures for both scales
    """
    with st.expander("Class Concentration Bar Chart"):
        # Get valid conditions (more than one sample)
        valid_conditions = [
            cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) 
            if num_samples > 1
        ]
        
        # Select conditions and classes
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
        
        # Add explanation about bar chart analysis
        show_barchart_info = st.checkbox("Show bar chart analysis details", key="show_barchart_info")
        if show_barchart_info:
            st.markdown("### Class Concentration Bar Chart Analysis")
            st.markdown("""
            Bar charts provide a quantitative comparison of lipid class concentrations across different experimental conditions.
            
            **What the Chart Shows:**
            - Each bar represents the mean concentration of a specific lipid class in a condition
            - Error bars represent the standard error of the mean (SEM = standard deviation/√n), indicating the precision of the measurement
            - Statistical significance is indicated with asterisks (* p < 0.05, ** p < 0.01, *** p < 0.001)
            
            **Viewing Options:**
            - **Linear Scale:** Shows absolute concentration values, useful for comparing abundant lipid classes
            - **Log2 Scale:** Transforms data to a logarithmic scale, better for visualizing both low and high abundance lipids
            
            **Statistical Testing:**
            """)
            
            # Integrate the statistical testing guidance based on the number of conditions
            if len(selected_conditions_list) > 2:
                st.markdown("""
                **For multiple conditions (current selection):**
                - Multiple conditions are analyzed using ANOVA + Tukey's test
                - This is the correct approach when interested in multiple comparisons
                - Do NOT run separate t-tests by unselecting conditions - this would inflate the false discovery rate
                - One-way ANOVA determines if any group means differ significantly
                - If ANOVA is significant, Tukey's HSD test identifies which specific pairs of conditions differ
                - P-values are adjusted for multiple testing using the Benjamini-Hochberg method to control false discovery rate at 5%
                - Error bars shown are standard error of the mean (SEM = standard deviation/√n)
                """)
            elif len(selected_conditions_list) == 2:
                st.markdown("""
                **For two conditions (current selection):**
                - Two conditions are compared using Welch's t-test (does not assume equal variances)
                - This is appropriate ONLY for a single pre-planned comparison
                - If you plan to compare multiple conditions, please select all relevant conditions to use ANOVA + Tukey's test
                - P-values are adjusted for multiple testing using the Benjamini-Hochberg method
                - Error bars shown are standard error of the mean (SEM = standard deviation/√n)
                - Significance levels:
                  * p < 0.05, ** p < 0.01, *** p < 0.001
                """)
            else:
                st.markdown("""
                **Statistical testing requires selecting conditions:**
                - For two conditions: Welch's t-test will be used
                - For three or more conditions: ANOVA followed by Tukey's HSD test will be used
                - Please select conditions above to enable statistical testing
                """)
            
            st.markdown("""
            **Data Processing:**
            - For each condition, the total concentration of each lipid class is calculated by summing all species within that class
            - The mean and standard error are calculated across all replicates within each condition
            - Statistical tests are performed to identify significant differences between conditions
            """)
            st.markdown("---")
        
        linear_fig, log2_fig = None, None
        if selected_conditions_list and selected_classes_list:
            # Perform statistical tests
            with st.spinner("Performing statistical analysis..."):
                statistical_results = lp.AbundanceBarChart.perform_statistical_tests(
                    continuation_df, 
                    experiment, 
                    selected_conditions_list, 
                    selected_classes_list
                )
            
            # Optional detailed statistical results
            show_detailed_stats = st.checkbox("Show detailed statistical analysis", key="show_detailed_stats")
            if show_detailed_stats:
                display_statistical_details(
                    statistical_results, 
                    selected_conditions_list
                )

            # Generate linear scale chart
            with st.spinner("Generating linear scale chart..."):
                linear_fig, abundance_df = lp.AbundanceBarChart.create_abundance_bar_chart(
                    df=continuation_df,
                    full_samples_list=experiment.full_samples_list,
                    individual_samples_list=experiment.individual_samples_list,
                    conditions_list=experiment.conditions_list,
                    selected_conditions=selected_conditions_list,
                    selected_classes=selected_classes_list,
                    mode='linear scale',
                    anova_results=statistical_results
                )
            
            if linear_fig is not None and abundance_df is not None and not abundance_df.empty:
                st.write("Linear Scale")
                st.plotly_chart(linear_fig)
                
                # Add download options
                col1, col2 = st.columns(2)
                with col1:
                    # Add SVG download button for the linear scale plot
                    plotly_svg_download_button(linear_fig, "abundance_bar_chart_linear.svg")
                with col2:
                    # Add CSV download for the data
                    csv_data = convert_df(abundance_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name='abundance_bar_chart_linear.csv',
                        mime='text/csv',
                        key='abundance_chart_download_linear'
                    )
            
            # Generate log2 scale chart
            with st.spinner("Generating log2 scale chart..."):
                log2_fig, abundance_df_log2 = lp.AbundanceBarChart.create_abundance_bar_chart(
                    df=continuation_df,
                    full_samples_list=experiment.full_samples_list,
                    individual_samples_list=experiment.individual_samples_list,
                    conditions_list=experiment.conditions_list,
                    selected_conditions=selected_conditions_list,
                    selected_classes=selected_classes_list,
                    mode='log2 scale',
                    anova_results=statistical_results
                )
            
            if log2_fig is not None and abundance_df_log2 is not None and not abundance_df_log2.empty:
                st.write("Log2 Scale")
                st.plotly_chart(log2_fig)
                
                # Add download options
                col1, col2 = st.columns(2)
                with col1:
                    # Add SVG download button for the log2 scale plot
                    plotly_svg_download_button(log2_fig, "abundance_bar_chart_log2.svg")
                with col2:
                    # Add CSV download for the data
                    csv_data_log2 = convert_df(abundance_df_log2)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data_log2,
                        file_name='abundance_bar_chart_log2.csv',
                        mime='text/csv',
                        key='abundance_chart_download_log2'
                    )
            
            # Check if any conditions were removed due to having only one sample
            removed_conditions = set(experiment.conditions_list) - set(valid_conditions)
            if removed_conditions:
                st.warning(f"The following conditions were excluded due to having only one sample: {', '.join(removed_conditions)}")
                
        else:
            st.warning("Please select at least one condition and one class to create the charts.")
        
        return linear_fig, log2_fig

def display_statistical_details(statistical_results, selected_conditions):
    """Display detailed statistical analysis results."""
    st.write("### Detailed Statistical Analysis")
    
    if len(selected_conditions) == 2:
        st.write(f"Comparing conditions: {selected_conditions[0]} vs {selected_conditions[1]}")
        st.write("**Method:** Welch's t-test (does not assume equal variances between groups)")
        st.write("""
        The t-test determines whether the means of two groups are statistically different from each other. 
        Welch's t-test is used because it does not assume equal variances between the two groups, making it more robust.
        """)
    else:
        st.write(f"Comparing {len(selected_conditions)} conditions using ANOVA + Tukey's test")
        st.write("""
        **Method:** One-way Analysis of Variance (ANOVA) followed by Tukey's HSD (Honestly Significant Difference) test
        
        ANOVA tests whether any of the group means are statistically different from each other. It returns an F-statistic and p-value.
        If ANOVA indicates significant differences (p < 0.05), Tukey's HSD test is performed to determine which specific groups differ.
        """)
        st.write("**Multiple comparisons correction:** Benjamini-Hochberg procedure to control false discovery rate at 5%")
    
    # Create a table for statistical results
    results_data = []
    for lipid_class, results in statistical_results.items():
        p_value = results['p-value']
        adj_p_value = results.get('adjusted p-value', p_value)
        
        if results['test'] == 't-test':
            result_row = {
                'Lipid Class': lipid_class,
                'Test': 't-test',
                'Statistic': f"t = {results['statistic']:.3f}",
                'p-value': f"{p_value:.3f}",
                'Adjusted p-value': f"{adj_p_value:.3f}",
                'Significant': '✓' if adj_p_value < 0.05 else '✗'
            }
            results_data.append(result_row)
        else:  # ANOVA
            result_row = {
                'Lipid Class': lipid_class,
                'Test': 'ANOVA',
                'Statistic': f"F = {results['statistic']:.3f}",
                'p-value': f"{p_value:.3f}",
                'Adjusted p-value': f"{adj_p_value:.3f}",
                'Significant': '✓' if adj_p_value < 0.05 else '✗'
            }
            results_data.append(result_row)
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        st.write("#### Statistical Test Results Summary")
        st.dataframe(results_df)
    
    # Show detailed post-hoc results for ANOVA
    if len(selected_conditions) > 2:
        st.write("#### Post-hoc Test Results (Tukey's HSD)")
        st.write("The table below shows pairwise comparisons between conditions for each lipid class with significant ANOVA results.")
        
        has_posthoc_results = False
        for lipid_class, results in statistical_results.items():
            if results['test'] == 'ANOVA' and results.get('tukey_results'):
                adj_p_value = results.get('adjusted p-value', results['p-value'])
                if adj_p_value < 0.05:
                    has_posthoc_results = True
                    st.write(f"**{lipid_class}**")
                    
                    tukey = results['tukey_results']
                    tukey_data = []
                    for g1, g2, p in zip(tukey['group1'], tukey['group2'], tukey['p_values']):
                        significance = ''
                        if p < 0.001:
                            significance = '***'
                        elif p < 0.01:
                            significance = '**'
                        elif p < 0.05:
                            significance = '*'
                            
                        tukey_data.append({
                            'Group 1': g1,
                            'Group 2': g2,
                            'p-value': f"{p:.3f}",
                            'Significant': significance
                        })
                    
                    tukey_df = pd.DataFrame(tukey_data)
                    st.dataframe(tukey_df)
        
        if not has_posthoc_results:
            st.write("No significant ANOVA results to display post-hoc tests for.")
                
def display_abundance_pie_charts(experiment, continuation_df):
    """
    Display pie charts showing the proportional distribution of lipid classes for each condition.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomics data
        
    Returns:
        dict: Dictionary containing the generated pie chart figures by condition
    """
    pie_charts = {}
    with st.expander("Class Concentration Pie Chart"):
        # Get all lipid classes
        full_samples_list = experiment.full_samples_list
        all_classes = lp.AbundancePieChart.get_all_classes(continuation_df, full_samples_list)
        
        # Select lipid classes
        selected_classes_list = st.multiselect(
            'Select classes for the chart:', 
            all_classes, 
            all_classes,
            key='pie_chart_classes'
        )
        
        # Show explanation about pie chart analysis
        show_piechart_info = st.checkbox("Show pie chart analysis details", key="show_piechart_info")
        if show_piechart_info:
            st.markdown("### Class Concentration Pie Chart Analysis")
            st.markdown("""
            Pie charts visualize the relative proportions of different lipid classes within each experimental condition, providing a quick overview of lipid composition.
            
            **What the Chart Shows:**
            - Each segment represents a lipid class
            - The size of each segment corresponds to the percentage of that class in the total lipid pool
            - Hover over segments to see exact percentages
            - Consistent colors are used for the same lipid class across different conditions
            
            **Data Processing:**
            - For each condition, the total concentration of each lipid class is calculated by summing all species within that class
            - The sum across all replicates is used to represent each condition
            - Values are converted to percentages for the pie chart representation
            """)
            st.markdown("---")
        
        if selected_classes_list:
            # Filter dataframe for selected classes
            filtered_df = lp.AbundancePieChart.filter_df_for_selected_classes(continuation_df, full_samples_list, selected_classes_list)
            
            # Generate consistent colors for classes
            color_mapping = lp.AbundancePieChart._generate_color_mapping(selected_classes_list)
            
            # Create list of conditions with multiple samples
            conditions_with_samples = []
            for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list):
                if len(samples) > 1:  # Skip conditions with only one sample
                    conditions_with_samples.append((condition, samples))
            
            if not conditions_with_samples:
                st.warning("No conditions with multiple samples found.")
                return pie_charts
            
            # Create separate rows for each condition
            for condition, samples in conditions_with_samples:
                st.subheader(f"{condition}")
                fig, df = lp.AbundancePieChart.create_pie_chart(
                    filtered_df, full_samples_list, condition, samples, color_mapping
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    pie_charts[condition] = fig
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        plotly_svg_download_button(
                            fig, f"abundance_pie_chart_{condition}.svg"
                        )
                    with col2:
                        # Create a data frame with percentages for download
                        available_samples = [sample for sample in samples if f"concentration[{sample}]" in filtered_df.columns]
                        if available_samples:
                            total_values = filtered_df[[f"concentration[{sample}]" for sample in available_samples]].sum(axis=1)
                            percent_values = (total_values / total_values.sum() * 100).round(2)
                            download_df = pd.DataFrame({
                                'Class': filtered_df.index,
                                'Total Concentration': total_values.values,
                                'Percentage (%)': percent_values.values
                            })
                            csv_download = convert_df(download_df)
                            st.download_button(
                                label="Download CSV",
                                data=csv_download,
                                file_name=f'abundance_pie_chart_{condition}.csv',
                                mime='text/csv'
                            )
                # Add a divider between conditions
                st.markdown("---")
        else:
            st.warning("Please select at least one lipid class to create the pie charts.")
    
    return pie_charts

def display_saturation_plots(experiment, continuation_df):
    """
    Display saturation plots with statistical analysis for selected conditions and lipid classes.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomics data
        
    Returns:
        dict: Dictionary containing the generated plots by lipid class
    """
    saturation_plots = {}
    with st.expander("Class Level Breakdown - Saturation Plots"):
        # Check if we have any detailed FA compositions
        has_detailed_fa = any('_' in str(lipid) for lipid in continuation_df['LipidMolec'])
        
        if not has_detailed_fa:
            st.warning("""
            ⚠️ Note: Saturation plots work best with detailed fatty acid composition (e.g., PC(16:0_18:1)).
            Your data appears to use total composition (e.g., PC(34:1)) which only shows total 
            carbons and double bonds. This may affect the accuracy of the saturation analysis.
            """)

        # Show explanation about saturation plot analysis
        show_saturation_info = st.checkbox("Show saturation plot analysis details", key="show_saturation_info")
        if show_saturation_info:
            st.markdown("### Saturation Plot Analysis")
            st.markdown("""
            Saturation plots analyze the fatty acid composition within lipid classes, distinguishing between saturated (SFA), mono-unsaturated (MUFA), and poly-unsaturated fatty acids (PUFA).
            
            **What the Plots Show:**
            
            1. **Concentration Profile Plot**:
               - Groups of bars show SFA, MUFA, and PUFA concentrations for each condition
               - Error bars represent standard error of the mean (SEM = standard deviation/√n)
               - Stars (★) indicate statistical significance between conditions:
                 * ★ p < 0.05
                 * ★★ p < 0.01
                 * ★★★ p < 0.001
            
            2. **Percentage Distribution Plot**:
               - Stacked bars show the relative proportion of SFA, MUFA, and PUFA in each condition
               - Total height always equals 100%
               - Shows compositional shifts even when total lipid amounts differ
            
            **How SFA, MUFA, and PUFA Are Calculated:**
            
            1. For each lipid molecule (e.g., PC(16:0_18:1)), the number of double bonds in each fatty acid chain is determined:
               - 16:0 has 0 double bonds → SFA (saturated)
               - 18:1 has 1 double bond → MUFA (mono-unsaturated)
               - Chains with 2+ double bonds → PUFA (poly-unsaturated)
            
            2. The proportion of each fatty acid type is calculated per molecule:
               - For PC(16:0_18:1), which has 1 SFA and 1 MUFA chain, the proportions are:
                 * SFA: 1/2 = 0.5 (50%)
                 * MUFA: 1/2 = 0.5 (50%)
                 * PUFA: 0/2 = 0 (0%)
            
            3. These proportions are multiplied by the molecule's concentration:
               - If PC(16:0_18:1) has a concentration of 10 μM, then:
                 * SFA contribution = 10 μM × 0.5 = 5 μM
                 * MUFA contribution = 10 μM × 0.5 = 5 μM
                 * PUFA contribution = 10 μM × 0 = 0 μM
            
            4. All contributions are summed across all lipid molecules in each class to get total SFA, MUFA, and PUFA concentrations
            
            **Statistical Analysis Details:**
            
            - **For two conditions**: Welch's t-test is used, which does not assume equal variances
            
            - **For multiple conditions**: One-way ANOVA followed by Tukey's HSD post-hoc test for pairwise comparisons
            
            - **Multiple Testing Correction**: 
              * When testing multiple lipid classes, p-values are NOT automatically adjusted
              * For ANOVA with post-hoc tests, Tukey's HSD inherently controls for multiple comparisons within each lipid class
              * Users should interpret significance with caution when analyzing many lipid classes simultaneously
            """)
            st.markdown("---")

        # Get all lipid classes
        full_samples_list = experiment.full_samples_list
        all_classes = continuation_df['ClassKey'].unique().tolist()
        
        # Select classes and conditions
        selected_classes_list = st.multiselect(
            'Select classes for the saturation plot:', 
            all_classes, 
            all_classes
        )
        
        selected_conditions = st.multiselect(
            'Select conditions for the saturation plot:', 
            experiment.conditions_list, 
            experiment.conditions_list
        )
        
        if selected_classes_list and selected_conditions:
            # Filter the DataFrame for selected classes
            filtered_df = continuation_df[continuation_df['ClassKey'].isin(selected_classes_list)]
            
            # Generate the plots
            with st.spinner("Generating saturation plots..."):
                plots = lp.SaturationPlot.create_plots(filtered_df, experiment, selected_conditions)
                
                if plots:
                    for lipid_class, (main_plot, percentage_plot, plot_data) in plots.items():
                        st.subheader(f"Saturation Profile for {lipid_class}")
                        
                        # Display the main plot
                        st.markdown("#### Concentration Profile")
                        st.plotly_chart(main_plot)
                        plotly_svg_download_button(main_plot, f"saturation_plot_main_{lipid_class}.svg")
                        
                        # Provide download for the data
                        main_csv_download = convert_df(plot_data)
                        st.download_button(
                            label="Download CSV",
                            data=main_csv_download,
                            file_name=f'saturation_plot_main_data_{lipid_class}.csv',
                            mime='text/csv'
                        )
                        
                        # Display the percentage plot
                        st.markdown("#### Percentage Distribution")
                        st.plotly_chart(percentage_plot)
                        plotly_svg_download_button(percentage_plot, f"saturation_plot_percentage_{lipid_class}.svg")
                        
                        # Provide download for the percentage data
                        percentage_data = lp.SaturationPlot._calculate_percentage_df(plot_data)
                        percentage_csv_download = convert_df(percentage_data)
                        st.download_button(
                            label="Download CSV",
                            data=percentage_csv_download,
                            file_name=f'saturation_plot_percentage_data_{lipid_class}.csv',
                            mime='text/csv'
                        )
                        
                        # Add to the dictionary for return
                        saturation_plots[lipid_class] = {'main': main_plot, 'percentage': percentage_plot}
                        
                        # Add a separator between plots
                        st.markdown("---")
                else:
                    st.warning("No plots could be generated. This might be because there are insufficient samples or no data for the selected classes.")
        else:
            st.warning("Please select at least one class and one condition to generate saturation plots.")
    
    return saturation_plots

def display_pathway_visualization(experiment, continuation_df):
    """
    Displays an interactive lipid pathway visualization within a Streamlit application.
    
    Args:
        experiment: Experiment object containing sample information
        continuation_df: DataFrame containing the lipidomic data
        
    Returns:
        matplotlib.figure.Figure or None: The pathway visualization figure if generated, otherwise None
    """
    with st.expander("Class Level Breakdown - Pathway Visualization"):
        # Check if we have any detailed FA compositions
        has_detailed_fa = any('_' in str(lipid) for lipid in continuation_df['LipidMolec'])
        
        if not has_detailed_fa:
            st.warning("""
            ⚠️ Note: The pathway visualization works best with detailed fatty acid composition (e.g., PC(16:0_18:1)).
            Your data appears to use total composition (e.g., PC(34:1)), which may affect the accuracy of the 
            saturation ratio calculations (shown by the color scale). The circle sizes (showing abundance) 
            remain accurate.
            """)

        # Show explanation about the pathway visualization
        show_pathway_info = st.checkbox("Show pathway visualization details", key="show_pathway_info")
        if show_pathway_info:
            st.markdown("### Lipid Pathway Visualization")
            st.markdown("""
            The lipid pathway visualization provides a comprehensive view of how different lipid classes relate to each other in metabolic pathways and how they are affected by experimental conditions.
            
            **What the Visualization Shows:**
            
            - **Circle Positions**: Each circle represents a different lipid class positioned according to metabolic relationships
            - **Circle Size**: The size of each circle indicates the fold change in abundance between conditions 
              - Larger circles = increased abundance in experimental condition
              - Smaller circles = decreased abundance in experimental condition
            - **Circle Color**: Color intensity represents the saturation ratio (proportion of saturated fatty acids)
              - Warmer colors (red/yellow) = higher proportion of saturated fatty acids
              - Cooler colors (blue/purple) = lower proportion of saturated fatty acids
            
            **How the Data is Calculated:**
            
            1. **Fold Change Calculation**:
               - For each lipid class, the average concentration in the experimental condition is divided by the average concentration in the control condition
               - This ratio determines the circle size (squared for visual emphasis)
               - Values >1 indicate an increase in the experimental condition
               - Values <1 indicate a decrease in the experimental condition
            
            2. **Saturation Ratio Calculation**:
               - For each lipid molecule (e.g., PC(16:0_18:1)), the number of saturated and unsaturated fatty acid chains is counted:
                 * Chains with 0 double bonds (e.g., 16:0) are counted as saturated
                 * Chains with 1+ double bonds (e.g., 18:1) are counted as unsaturated
               - The saturation ratio is calculated as: number of saturated chains ÷ total number of chains
               - This ratio ranges from 0 (all unsaturated) to 1 (all saturated)
            """)
            st.markdown("---")

        # Allow selection of control and experimental conditions
        if len([x for x in experiment.number_of_samples_list if x > 1]) > 1:
            control_condition = st.selectbox('Select Control Condition',
                                             [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1], 
                                             index=0)
            experimental_condition = st.selectbox('Select Experimental Condition',
                                                  [cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) if num_samples > 1 and cond != control_condition], 
                                                  index=0)
    
            # Check if both conditions are selected
            if control_condition and experimental_condition:
                with st.spinner("Generating pathway visualization..."):
                    # Calculate saturation ratios
                    class_saturation_ratio_df = lp.PathwayViz.calculate_class_saturation_ratio(continuation_df)
                    
                    # Calculate fold changes
                    class_fold_change_df = lp.PathwayViz.calculate_class_fold_change(continuation_df, experiment, control_condition, experimental_condition)
                    
                    # Generate the visualization
                    fig, pathway_dict = lp.PathwayViz.create_pathway_viz(class_fold_change_df, class_saturation_ratio_df, control_condition, experimental_condition)
                    
                    if fig is not None and pathway_dict:
                        st.pyplot(fig)
                        
                        # Add SVG download button for the pathway visualization
                        matplotlib_svg_download_button(fig, f"pathway_visualization_{control_condition}_vs_{experimental_condition}.svg")
                        
                        # Create a summary table of the data
                        st.subheader("Pathway Data Summary")
                        st.markdown(f"**Comparing {experimental_condition} to {control_condition}**")
                        
                        # Create a more informative dataframe
                        pathway_df = pd.DataFrame({
                            'Lipid Class': pathway_dict['class'],
                            'Fold Change': pathway_dict['abundance ratio'],
                            'Saturation Ratio': pathway_dict['saturated fatty acids ratio']
                        })
                        
                        # Add a direction column for clarity
                        pathway_df['Change Direction'] = pathway_df['Fold Change'].apply(
                            lambda x: "↑ Increased" if x > 1 else 
                                     ("↓ Decreased" if x < 1 and x > 0 else "No Change")
                        )
                        
                        # Sort by absolute change
                        pathway_df['Absolute Change'] = abs(pathway_df['Fold Change'] - 1)
                        pathway_df = pathway_df.sort_values('Absolute Change', ascending=False)
                        
                        # Remove helper column and format others
                        pathway_df = pathway_df.drop('Absolute Change', axis=1)
                        pathway_df['Fold Change'] = pathway_df['Fold Change'].apply(lambda x: f"{x:.2f}")
                        pathway_df['Saturation Ratio'] = pathway_df['Saturation Ratio'].apply(lambda x: f"{x:.2f}")
                        
                        # Display the dataframe
                        st.dataframe(pathway_df)
                        
                        # Provide download button
                        csv_download = convert_df(pathway_df)
                        st.download_button(
                            label="Download CSV",
                            data=csv_download,
                            file_name='pathway_visualization_data.csv',
                            mime='text/csv')
                        
                        return fig
                    else:
                        st.warning("Unable to generate pathway visualization due to insufficient data.")
                        st.markdown("""
                        This could be due to:
                        1. Missing lipid classes needed for the visualization
                        2. Issues with fatty acid composition data
                        3. Insufficient data for calculating fold changes
                        
                        Try using a dataset with more comprehensive lipid coverage or check the format of your fatty acid data.
                        """)
        else:
            st.warning("At least two conditions with more than one replicate are required for pathway visualization.")
            
        return None
                
def display_volcano_plot(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with volcano plots in lipidomics data using Plotly.
    Returns a dictionary containing the generated plots.
    """
    volcano_plots = {}
    with st.expander("Species Level Breakdown - Volcano Plot"):
        # Check for conditions with multiple replicates
        conditions_with_replicates = [condition for index, condition in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[index] > 1]
        if len(conditions_with_replicates) <= 1:
            st.error('You need at least two conditions with more than one replicate to create a volcano plot.')
            return volcano_plots

        # Show explanation about volcano plot analysis
        show_volcano_info = st.checkbox("Show volcano plot analysis details", key="show_volcano_info")
        if show_volcano_info:
            st.markdown("### Volcano Plot Analysis")
            st.markdown("""
            Volcano plots are powerful visualizations for identifying statistically significant changes in lipid species between two experimental conditions.
            
            **What the Plot Shows:**
            
            - **X-axis (Log2 Fold Change)**: Shows the magnitude and direction of change between conditions
              - Positive values (right side): Increased in experimental condition
              - Negative values (left side): Decreased in experimental condition
              - Each unit represents a doubling (right) or halving (left) of concentration
            
            - **Y-axis (-log10 p-value)**: Shows the statistical significance of the change
              - Higher values (top): More statistically significant
              - Lower values (bottom): Less statistically significant
            
            - **Dashed Lines**: Reference thresholds for biological and statistical significance
              - Horizontal line: Statistical significance threshold (p-value)
              - Vertical lines: Biological significance thresholds (±2-fold change)
            
            - **Color Coding**: Points are colored by lipid class for easier pattern recognition
            
            **Data Processing:**
            
            - **Fold Change Calculation**: Log2(mean experimental / mean control)
              - Lipids with zero concentration in either condition are excluded and listed separately
            
            - **Statistical Testing**: Welch's t-test comparing control and experimental replicates
              - P-values are transformed to -log10 scale for visualization
              - No adjustment for multiple testing is applied
            
            - **Filtering Options**:
              - Set significance threshold (default p = 0.05)
              - Hide non-significant points
              - Select specific lipid classes
            
            **Companion Plots:**
            
            - **Concentration vs. Fold Change**: Shows relationship between abundance and magnitude of change
              - Reveals whether high or low abundance lipids are more affected
            
            - **Concentration Distribution**: Box plots of selected lipids across conditions
              - Provides detailed view of specific lipids of interest
            """)
            st.markdown("---")

        # Significance level selection
        p_value_threshold = st.number_input('Enter the significance level for Volcano Plot', min_value=0.001, max_value=0.1, value=0.05, step=0.001, key="volcano_plot_p_value_threshold")
        q_value_threshold = -np.log10(p_value_threshold)
        
        # Condition selection
        control_condition = st.selectbox('Pick the control condition', conditions_with_replicates)
        default_experimental = conditions_with_replicates[1] if len(conditions_with_replicates) > 1 else conditions_with_replicates[0]
        experimental_condition = st.selectbox('Pick the experimental condition', conditions_with_replicates, index=conditions_with_replicates.index(default_experimental))
        
        # Class selection
        selected_classes_list = st.multiselect('Add or remove classes:', list(continuation_df['ClassKey'].value_counts().index), list(continuation_df['ClassKey'].value_counts().index))
        
        # Display options
        hide_non_significant = st.checkbox('Hide non-significant data points', value=False)

        # Generate plots
        if selected_classes_list:
            with st.spinner("Generating volcano plot..."):
                plot, merged_df, removed_lipids_df = lp.VolcanoPlot.create_and_display_volcano_plot(
                    experiment, 
                    continuation_df, 
                    control_condition, 
                    experimental_condition, 
                    selected_classes_list, 
                    q_value_threshold, 
                    hide_non_significant
                )
                
                # Display main volcano plot
                st.plotly_chart(plot, use_container_width=True)
                volcano_plots['main'] = plot
                
                # Add download options
                plotly_svg_download_button(plot, "volcano_plot.svg")
                csv_data = convert_df(merged_df[['LipidMolec', 'FoldChange', '-log10(pValue)', 'ClassKey']])
                st.download_button("Download CSV", csv_data, file_name="volcano_data.csv", mime="text/csv")
            
            st.markdown("---")
            
            # Generate and display the concentration vs. fold change plot
            with st.spinner("Generating concentration vs. fold change plot..."):
                st.subheader("Concentration vs. Fold Change Plot")
                st.markdown("""
                This plot shows the relationship between a lipid's abundance (concentration) and its fold change. 
                It can reveal whether changes occur primarily in high- or low-abundance lipid species.
                """)
                
                color_mapping = lp.VolcanoPlot._generate_color_mapping(merged_df)
                concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(
                    merged_df, 
                    color_mapping, 
                    q_value_threshold, 
                    hide_non_significant
                )
                
                st.plotly_chart(concentration_vs_fold_change_plot, use_container_width=True)
                volcano_plots['concentration_vs_fold_change'] = concentration_vs_fold_change_plot
                
                # Add download options
                plotly_svg_download_button(concentration_vs_fold_change_plot, "concentration_vs_fold_change_plot.svg")
                csv_data_for_concentration_plot = convert_df(download_df)
                st.download_button("Download CSV", csv_data_for_concentration_plot, file_name="concentration_vs_fold_change_data.csv", mime="text/csv")
            
            st.markdown("---")
            
            # Additional functionality for lipid concentration distribution
            st.subheader("Lipid Concentration Distribution")
            st.markdown("""
            Select a lipid class and specific lipids to view their concentration distribution across conditions. 
            This provides a detailed look at individual lipid species that show interesting patterns in the volcano plot.
            """)
            
            all_classes = list(merged_df['ClassKey'].unique())
            selected_class = st.selectbox('Select Lipid Class:', all_classes)
            
            if selected_class:
                class_lipids = merged_df[merged_df['ClassKey'] == selected_class]['LipidMolec'].unique().tolist()
                selected_lipids = st.multiselect('Select Lipids:', class_lipids, default=class_lipids[:1])
                
                if selected_lipids:
                    selected_conditions = [control_condition, experimental_condition]
                    with st.spinner("Generating concentration distribution plot..."):
                        plot_df = lp.VolcanoPlot.create_concentration_distribution_data(
                            merged_df, 
                            selected_lipids, 
                            selected_conditions, 
                            experiment
                        )
                        fig = lp.VolcanoPlot.create_concentration_distribution_plot(plot_df, selected_lipids, selected_conditions)
                        st.pyplot(fig)
                        volcano_plots['concentration_distribution'] = fig
                        
                        # Add download options
                        matplotlib_svg_download_button(fig, "concentration_distribution_plot.svg")
                        csv_data = convert_df(plot_df)
                        st.download_button("Download CSV", csv_data, file_name=f"{'_'.join(selected_lipids)}_concentration.csv", mime="text/csv")
            
            st.markdown("---")

            # Displaying the table of invalid lipids
            st.subheader("Excluded Lipids")
            st.markdown("""
            The following lipids were excluded from the volcano plot because they had zero concentration in either 
            the control or experimental condition. This often occurs for lipids that are only present in one condition.
            """)
            
            if not removed_lipids_df.empty:
                st.dataframe(removed_lipids_df)
                csv_excluded = convert_df(removed_lipids_df)
                st.download_button("Download Excluded Lipids CSV", csv_excluded, file_name="excluded_lipids.csv", mime="text/csv")
            else:
                st.write("No lipids were excluded from the analysis.")
        else:
            st.warning("Please select at least one lipid class to generate the volcano plot.")

    return volcano_plots

def display_lipidomic_heatmap(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with lipidomic heatmaps.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomics data
        
    Returns:
        tuple: A tuple containing the regular heatmap figure and clustered heatmap figure
    """
    regular_heatmap = None
    clustered_heatmap = None
    
    with st.expander("Species Level Breakdown - Lipidomic Heatmap"):
        # Show explanation about heatmap analysis
        show_heatmap_info = st.checkbox("Show heatmap analysis details", key="show_heatmap_info")
        if show_heatmap_info:
            st.markdown("### Lipidomic Heatmap Analysis")
            st.markdown("""
            Heatmaps provide a comprehensive visualization of lipid abundance patterns across multiple samples and conditions.
            
            **What the Heatmap Shows:**
            
            - **Rows**: Individual lipid molecules (e.g., PC(16:0_18:1))
            - **Columns**: Individual samples grouped by condition
            - **Colors**: Represent standardized abundance values (Z-scores)
              - Red: Higher than average abundance
              - Blue: Lower than average abundance
              - White: Close to average abundance
            - **Z-scores**: Standardized values that show how many standard deviations a data point is from the mean
              - Z-score = (Value - Mean) / Standard Deviation
              - This normalization enables comparison across lipids with different absolute abundances
            
            **Heatmap Types:**
            
            1. **Regular Heatmap**: 
               - Shows lipids in their original order
               - Preserves the organization of your input data
               - Useful when you have a specific ordering in mind (e.g., by lipid class)
            
            2. **Clustered Heatmap**:
               - Groups similar lipids together based on their abundance patterns
               - Uses hierarchical clustering with Ward's method
               - Reveals co-regulated lipid groups that might have similar biological functions
               - Dashed lines separate distinct clusters
               - Number of clusters can be adjusted using the slider
            """)
            st.markdown("---")
        
        # Let user select conditions and classes
        all_conditions = experiment.conditions_list
        selected_conditions = st.multiselect(
            "Select conditions:", 
            all_conditions, 
            default=all_conditions
        )
        
        # Filter for conditions with multiple samples
        selected_conditions = [condition for condition in selected_conditions 
                             if len(experiment.individual_samples_list[experiment.conditions_list.index(condition)]) > 0]
        
        all_classes = continuation_df['ClassKey'].unique()
        selected_classes = st.multiselect(
            "Select lipid classes:", 
            all_classes, 
            default=all_classes
        )
        
        # Choose number of clusters for hierarchical clustering
        n_clusters = st.slider(
            "Number of clusters:", 
            min_value=2, 
            max_value=10, 
            value=5,
            help="Adjust the number of clusters in the hierarchical clustering. More clusters will create more detailed groupings."
        )
        
        if selected_conditions and selected_classes:
            # Filter data based on selections
            filtered_df, selected_samples = lp.LipidomicHeatmap.filter_data(
                continuation_df, 
                selected_conditions, 
                selected_classes, 
                experiment.conditions_list, 
                experiment.individual_samples_list
            )
            
            if filtered_df.empty:
                st.warning("No data available for the selected conditions and classes.")
                return None
            
            # Compute Z-scores
            with st.spinner("Computing Z-scores..."):
                z_scores_df = lp.LipidomicHeatmap.compute_z_scores(filtered_df)
            
            # Heatmap type selection
            heatmap_type = st.radio(
                "Select Heatmap Type", 
                ["Clustered", "Regular"], 
                index=0,
                help="Clustered heatmap groups similar lipids together. Regular heatmap preserves the original order."
            )
            
            # Generate and display the appropriate heatmap
            if heatmap_type == "Clustered":
                with st.spinner("Generating clustered heatmap..."):
                    clustered_heatmap = lp.LipidomicHeatmap.generate_clustered_heatmap(
                        z_scores_df, 
                        selected_samples, 
                        n_clusters
                    )
                    
                    # Display the clustered heatmap
                    st.plotly_chart(clustered_heatmap, use_container_width=True)
                    
                    # Add download options
                    col1, col2 = st.columns(2)
                    with col1:
                        plotly_svg_download_button(clustered_heatmap, "Lipidomic_Clustered_Heatmap.svg")
                    with col2:
                        csv_download = convert_df(z_scores_df.reset_index())
                        st.download_button(
                            "Download CSV", 
                            csv_download, 
                            'clustered_heatmap_data.csv', 
                            'text/csv'
                        )
                    
                    # Display cluster information
                    st.subheader("Cluster Composition")
                    st.markdown("""
                    The table below shows the percentage distribution of lipid classes within each cluster.
                    This helps identify which lipid classes are enriched in specific clusters.
                    """)
                    
                    # Get cluster percentages
                    _, class_percentages = lp.LipidomicHeatmap.identify_clusters_and_percentages(
                        z_scores_df, 
                        n_clusters
                    )
                    
                    # Display cluster percentages
                    if not class_percentages.empty:
                        # Format percentages to 1 decimal place
                        formatted_percentages = class_percentages.round(1)
                        st.dataframe(formatted_percentages)
                        
                        # Add download option for cluster composition
                        csv_download = convert_df(formatted_percentages.reset_index())
                        st.download_button(
                            "Download Cluster Composition CSV", 
                            csv_download, 
                            'cluster_composition.csv', 
                            'text/csv'
                        )
            else:
                with st.spinner("Generating regular heatmap..."):
                    regular_heatmap = lp.LipidomicHeatmap.generate_regular_heatmap(
                        z_scores_df, 
                        selected_samples
                    )
                    
                    # Display the regular heatmap
                    st.plotly_chart(regular_heatmap, use_container_width=True)
                    
                    # Add download options
                    col1, col2 = st.columns(2)
                    with col1:
                        plotly_svg_download_button(regular_heatmap, "Lipidomic_Regular_Heatmap.svg")
                    with col2:
                        csv_download = convert_df(z_scores_df.reset_index())
                        st.download_button(
                            "Download CSV", 
                            csv_download, 
                            'regular_heatmap_data.csv', 
                            'text/csv'
                        )
            
            # Remove the redundant separate download section
            # since we've added download options right below each heatmap
            
        else:
            st.warning("Please select at least one condition and one lipid class to generate the heatmap.")
    
    return regular_heatmap, clustered_heatmap

def generate_pdf_report(box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot, 
                     heatmap_figs, correlation_plots, abundance_bar_charts, abundance_pie_charts, 
                     saturation_plots, volcano_plots, pathway_visualization):
    pdf_buffer = io.BytesIO()
    
    try:
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        
        # Page 1: Box Plots
        # Convert Plotly figures to PNG for the box plots
        box_plot1_bytes = pio.to_image(box_plot_fig1, format='png', width=800, height=600, scale=2)
        box_plot2_bytes = pio.to_image(box_plot_fig2, format='png', width=800, height=600, scale=2)
        
        # Add first box plot
        img1 = ImageReader(io.BytesIO(box_plot1_bytes))
        pdf.drawImage(img1, 50, 400, width=500, height=300, preserveAspectRatio=True)
        
        # Add second box plot
        img2 = ImageReader(io.BytesIO(box_plot2_bytes))
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
            rt_bytes = pio.to_image(retention_time_plot, format='png', width=1000, height=700, scale=2)
            rt_img = ImageReader(io.BytesIO(rt_bytes))
            pdf.drawImage(rt_img, 50, 50, width=700, height=500, preserveAspectRatio=True)
        
        # Page 4: PCA Plot
        pdf.showPage()
        pdf.setPageSize(landscape(letter))
        if pca_plot is not None:
            pca_bytes = pio.to_image(pca_plot, format='png', width=1000, height=700, scale=2)
            pca_img = ImageReader(io.BytesIO(pca_bytes))
            pdf.drawImage(pca_img, 50, 50, width=700, height=500, preserveAspectRatio=True)
        
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
                # Convert Plotly figure to PNG bytes
                chart_bytes = pio.to_image(chart, format='png', width=1000, height=700, scale=2)
                chart_img = ImageReader(io.BytesIO(chart_bytes))
                pdf.drawImage(chart_img, 50, 50, width=700, height=500, preserveAspectRatio=True)
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