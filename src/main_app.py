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
from typing import Tuple, Optional

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
    
    # Grade configuration state (LipidSearch)
    if 'grade_config' not in st.session_state:
        st.session_state.grade_config = None
    
    # MS-DIAL configuration states
    if 'msdial_quality_config' not in st.session_state:
        st.session_state.msdial_quality_config = None
    if 'msdial_features' not in st.session_state:
        st.session_state.msdial_features = {}
    if 'msdial_use_normalized' not in st.session_state:
        st.session_state.msdial_use_normalized = False

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
    - **Versatile Data Input**: Import datasets from LipidSearch, MS-DIAL, Metabolomics Workbench, or generic CSV formats.
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
    - **Data Input**: Import data in flexible formats, including generic CSV, LipidSearch, MS-DIAL, and Metabolomics Workbench.
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
    describing LipidCruncher's features. The LipidSearch dataset is the original dataset containing raw, unprocessed values.
    Additionally, we provide the same data in Generic and Metabolomics Workbench formats, which include pre-normalized values for your convenience.

   
    
    The case study experiment includes three conditions with four replicates each:
    - **WT** (Wild Type): samples s1-s4
    - **ADGAT-DKO** (Double Knockout): samples s5-s8
    - **BQC** (Batch Quality Control): samples s9-s12
    """)
                
    st.subheader("Tip")
    st.markdown("""
    💡 **Important**: When you need to switch to a different analysis with a different dataset, 
    first refresh the page and start from the homepage. This ensures a clean session and prevents 
    any conflicts with previously loaded data.
    """)
    
    st.subheader("Support")
    st.markdown("""
    For reporting bugs, feature requests, or any questions or comments, please email lipidcruncher@gmail.com.
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
        
        st.markdown("Process, analyze and visualize lipidomic data from multiple sources.")
        
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
                # Show column mapping validation/correction BEFORE defining experiment
                if data_format in ['Generic Format', 'MS-DIAL']:
                    mapping_valid, corrected_df = display_column_mapping(df, data_format)
                    if not mapping_valid:
                        st.sidebar.error("Please correct column mappings to proceed.")
                        return
                    
                    # For MS-DIAL, DON'T apply corrections yet - wait for data type selection
                    # For Generic, apply corrections now
                    if data_format == 'Generic Format' and 'manual_column_correction' in st.session_state:
                        df = apply_manual_column_correction(df, data_format)
                
                confirmed, name_df, experiment, bqc_label, valid_samples, updated_df = process_experiment(df, data_format)
                
                st.session_state.confirmed = confirmed
                
                if valid_samples:
                    if confirmed:
                        update_session_state(name_df, experiment, bqc_label)
                        
                        df_to_clean = updated_df if updated_df is not None else df
                        
                        if st.session_state.module == "Data Cleaning, Filtering, & Normalization":
                            st.subheader("Data Standardization, Filtering, and Normalization Module")
                            
                            # Quality filtering configuration (format-specific)
                            grade_config = None
                            quality_config = None
                            
                            if data_format == 'LipidSearch 5.0':
                                # Show grade filtering UI for LipidSearch
                                with st.expander("Configure Grade Filtering (LipidSearch Only)", expanded=False):
                                    grade_config = get_grade_filtering_config(df_to_clean, data_format)
                                    st.session_state.grade_config = grade_config
                            
                            elif data_format == 'MS-DIAL':
                                # Always show data type selection first (if both raw and normalized data available)
                                get_msdial_data_type_selection()
                                
                                # Check if Total score was included in manual corrections
                                manual_correction = st.session_state.get('manual_column_correction')
                                has_total_score = True
                                
                                if manual_correction:
                                    # If manual corrections exist, check if TotalScore is included
                                    has_total_score = 'TotalScore' in manual_correction.get('metadata_mapping', {})
                                
                                if not has_total_score:
                                    st.info("ℹ️ Quality filtering unavailable - Total score column was not selected in column mapping.")
                                else:
                                    # Show quality filtering UI in expander below data type selection
                                    with st.expander("🔬 Quality Filtering Configuration", expanded=True):
                                        quality_config = get_msdial_quality_config()
                                        st.session_state.msdial_quality_config = quality_config
                            
                            # Pass config to clean_data
                            cleaned_df, intsta_df = clean_data(
                                df_to_clean, name_df, experiment, data_format, 
                                grade_config=grade_config,
                                quality_config=quality_config
                            )
                            
                            if cleaned_df is not None:
                                st.session_state.experiment = experiment
                                st.session_state.format_type = data_format
                                st.session_state.cleaned_df = cleaned_df
                                st.session_state.intsta_df = intsta_df
                                # Save original auto-detected standards for restoration when switching modes
                                st.session_state.original_auto_intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()
                                st.session_state.continuation_df = cleaned_df
                                
                                cleaned_df = display_cleaned_data(cleaned_df, intsta_df)
                                
                                if cleaned_df is not None:
                                    st.session_state.cleaned_df = cleaned_df
                                    st.session_state.continuation_df = cleaned_df
                                
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
                                            st.session_state.preserved_data = {
                                                'cleaned_df': st.session_state.cleaned_df,
                                                'intsta_df': st.session_state.intsta_df,
                                                'normalized_df': st.session_state.normalized_df,
                                                'continuation_df': st.session_state.continuation_df,
                                                'normalization_method': st.session_state.normalization_method,
                                                'selected_classes': st.session_state.selected_classes,
                                                'create_norm_dataset': st.session_state.create_norm_dataset,
                                                'preserved_intsta_df': st.session_state.get('preserved_intsta_df'),
                                                'protein_df': st.session_state.get('protein_df'),
                                                'standard_source_preference': st.session_state.get('standard_source_preference'),
                                                'class_standard_map': st.session_state.get('class_standard_map'),
                                                'standard_concentrations': st.session_state.get('standard_concentrations'),
                                                'grade_config': st.session_state.get('grade_config')
                                            }
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
                                    'create_norm_dataset': st.session_state.create_norm_dataset,
                                    'preserved_intsta_df': st.session_state.get('preserved_intsta_df'),
                                    'protein_df': st.session_state.get('protein_df'),
                                    'standard_source_preference': st.session_state.get('standard_source_preference'),
                                    'class_standard_map': st.session_state.get('class_standard_map'),
                                    'standard_concentrations': st.session_state.get('standard_concentrations'),
                                    'grade_config': st.session_state.get('grade_config')
                                }
                                st.experimental_rerun()
                    
                    else:
                        clear_session_state()
                        st.info("Please confirm your inputs in the sidebar to proceed with data cleaning and analysis.")
                else:
                    st.error("Please ensure your experiment description is valid before proceeding.")

        # Restore preserved state if returning from Quality Check & Analysis
        if 'preserved_data' in st.session_state and st.session_state.module == "Data Cleaning, Filtering, & Normalization":
            for key, value in st.session_state.preserved_data.items():
                if value is not None:
                    # CRITICAL FIX: Don't restore intsta_df if user is in "Upload Custom Standards" mode
                    # This prevents auto-detected standards from reappearing when switching modes
                    if key == 'intsta_df':
                        # Only restore intsta_df if we're in Automatic Detection mode OR if no mode is set yet
                        restored_preference = st.session_state.preserved_data.get('standard_source_preference', 'Automatic Detection')
                        if restored_preference == "Automatic Detection":
                            setattr(st.session_state, key, value)
                        # If in Upload Custom Standards mode, keep intsta_df empty unless custom file was uploaded
                        elif 'preserved_intsta_df' not in st.session_state.preserved_data or st.session_state.preserved_data['preserved_intsta_df'] is None:
                            st.session_state.intsta_df = pd.DataFrame()
                    else:
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
    st.session_state.grade_config = None  # Clear LipidSearch grade config
    # Clear MS-DIAL specific states
    st.session_state.msdial_quality_config = None
    st.session_state.msdial_features = {}
    st.session_state.msdial_use_normalized = False
    
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
    
    # IMPORTANT FIX: Make sure to preserve normalization settings through navigation
    if 'preserved_data' in st.session_state:
        # Restore preserved normalization settings when returning from other modules
        if 'preserved_intsta_df' in st.session_state.preserved_data:
            st.session_state.preserved_intsta_df = st.session_state.preserved_data['preserved_intsta_df']
        if 'protein_df' in st.session_state.preserved_data:
            st.session_state.protein_df = st.session_state.preserved_data['protein_df']
        if 'standard_source_preference' in st.session_state.preserved_data:
            st.session_state.standard_source_preference = st.session_state.preserved_data['standard_source_preference']
        if 'class_standard_map' in st.session_state.preserved_data:
            st.session_state.class_standard_map = st.session_state.preserved_data['class_standard_map']
        if 'standard_concentrations' in st.session_state.preserved_data:
            st.session_state.standard_concentrations = st.session_state.preserved_data['standard_concentrations']

def display_format_selection():
    return st.sidebar.selectbox(
        'Select Data Format',
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0', 'MS-DIAL']
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
        * Standardize lipid names (e.g., "LPC 16:0"  →  "LPC(16:0)")
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
    elif data_format == 'MS-DIAL':
        st.info("""
        **📋 MS-DIAL Format Requirements**

        **How to Export from MS-DIAL:**
        1. In MS-DIAL: File → Export → Alignment Result → CSV format
        2. Ensure the following columns are included in your export

        **REQUIRED Column (must use exact name):**
        * `Metabolite name` - Lipid/metabolite identifiers

        **OPTIONAL Quality Columns (use exact names for auto-detection):**
        * `Total score` - Quality score for filtering (0-100)
        * `MS/MS matched` - Whether MS/MS spectrum was matched (TRUE/FALSE)
        * `Average Rt(min)` - Retention time information
        * `Average Mz` - Mass-to-charge ratio

        **Sample Intensity Columns:**
        * Your export should include sample intensity columns (one per sample)
        * **CRITICAL:** Sample columns must be the LAST columns in your file

        **File Structure (Column Order):**

        **Option 1 - Raw data only:**
        * Structure: [...metadata columns...] [sample 1] [sample 2] ... [sample N]
        * Example with 7 samples: The LAST 7 columns must be your sample intensity columns

        **Option 2 - Both raw AND pre-normalized data:**
        * Structure: [...metadata columns...] [raw sample 1] ... [raw sample N] [Lipid IS] [normalized sample 1] ... [normalized sample N]
        * The `Lipid IS` column separates raw from normalized data
        * The `Lipid IS` column should contain the internal standard class used for normalization
        * Example with 7 samples: The LAST 15 columns must be: 7 raw + 1 Lipid IS + 7 normalized
        * You'll be able to choose which dataset to use after upload

        **Notes:**
        * ✓ Column names are case-sensitive and must match exactly
        * ✓ Metadata header rows are automatically skipped
        * ✓ Lipid class (ClassKey) is automatically inferred from lipid names
        * ✓ Hydroxyl notation (`;2O`, `;3O`) is preserved in lipid names
        * ✓ Internal standards with (d5), (d7), (d9), ISTD, or SPLASH patterns are auto-detected
        * ✓ After upload, you can review and correct column mappings if needed
        """)
    else:
        st.info("""
        **Dataset Requirements for Generic Format**
        
        IMPORTANT: Your dataset must contain ONLY these columns in this order:
        
        1. **First Column - Lipid Names:**
           * Can have any column name
           * Must contain lipid molecule identifiers
           * Will be standardized to match the standard format. For example:
             - LPC O-18:1         -> LPC(O-18:1)
             - Cer d18:0/C24:0    -> Cer(d18:0_C24:0)
             - CE 14:0;0          -> CE(14:0)
             - PA 16:0/18:1+O     -> PA(16:0_18:1+O)
             
        2. **Remaining Columns - Intensity Values Only:**
           * Can have any column names
           * Each column should contain intensity values for one sample
           * The number of intensity columns must match the total number of samples in your experiment
           * These columns will be automatically standardized to: intensity[s1], intensity[s2], ..., intensity[sN]
        
        ⚠️  If your dataset contains any additional columns, please remove them before uploading.
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
        
        elif data_format == 'MS-DIAL':
            # MS-DIAL format processing
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Store original data before standardization for potential manual corrections
            st.session_state.original_uploaded_df = df.copy()
            
            df, success, message = lp.DataFormatHandler.validate_and_preprocess(
                df,
                'msdial'
            )
            
            if not success:
                st.error(message)
                return None
            
            # Show info message with detected features
            st.info(message)
            return df
            
        else:
            # Standard CSV processing for other formats
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Store original data before standardization for potential manual corrections
            st.session_state.original_uploaded_df = df.copy()
            
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
    st.sidebar.subheader("Define Experiment")
    
    if data_format == 'Metabolomics Workbench' and 'workbench_conditions' in st.session_state:
        # Get conditions in the order they appear in the data
        conditions_in_order = [st.session_state.workbench_conditions[f's{i+1}'] 
                              for i in range(len(st.session_state.workbench_samples))]
        
        # Get unique conditions in order of first appearance
        unique_conditions = []
        seen = set()
        for condition in conditions_in_order:
            if condition not in seen:
                seen.add(condition)
                unique_conditions.append(condition)
        
        use_detected = st.sidebar.checkbox("Use detected experimental setup", value=True)
        
        if use_detected:
            # Use the detected conditions in original order
            n_conditions = len(unique_conditions)
            conditions_list = unique_conditions
            
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
            # Manual setup remains unchanged
            n_conditions = st.sidebar.number_input('Enter the number of conditions', 
                                                 min_value=1, max_value=20, value=1, step=1)
            conditions_list = [st.sidebar.text_input(f'Create a label for condition #{i + 1}') 
                             for i in range(n_conditions)]
            number_of_samples_list = [st.sidebar.number_input(f'Number of samples for condition #{i + 1}', 
                                                            min_value=1, max_value=1000, value=1, step=1) 
                                    for i in range(n_conditions)]
    else:
        # Non-Metabolomics Workbench setup remains unchanged
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
    
    # Rest of the function remains unchanged
    name_df, group_df, updated_df, valid_samples = process_group_samples(df, experiment, data_format)

    if not valid_samples:
        return False, None, None, None, False, None

    if st.session_state.grouping_complete:
        bqc_label = specify_bqc_samples(experiment)
        confirmed = confirm_user_inputs(group_df, experiment)
    else:
        bqc_label = None
        confirmed = False
        st.sidebar.error("Please complete sample grouping before proceeding.")

    return confirmed, name_df, experiment, bqc_label, valid_samples, updated_df

def display_column_mapping(df, data_format):
    """
    Display and optionally correct column mapping in the sidebar.
    Called BEFORE experiment is defined.
    """
    if st.session_state.get('column_mapping') is None:
        return True, None
    
    st.sidebar.subheader("Column Name Standardization")
    
    # Display initial mapping table
    mapping_df = st.session_state.column_mapping.copy()
    st.sidebar.dataframe(
        mapping_df.reset_index(drop=True),
        use_container_width=True
    )
    
    # For MS-DIAL, column names must be exact - no manual mapping needed
    # Just show what was detected and allow continuing
    if st.session_state.format_type == 'MS-DIAL':
        
        # Optional: Allow user to override sample column detection
        with st.sidebar.expander("🔧 Override Sample Detection (Optional)", expanded=False):
            st.write("Only change this if auto-detection incorrectly classified columns.")
            
            features = st.session_state.get('msdial_features', {})
            detected_samples = features.get('raw_sample_columns', [])
            all_columns = features.get('actual_columns', [])
            
            # Exclude known metadata columns
            available_for_samples = [
                col for col in all_columns 
                if col not in lp.DataFormatHandler.MSDIAL_METADATA_COLUMNS
            ]
            
            manual_samples = st.multiselect(
                "Sample columns:",
                options=available_for_samples,
                default=detected_samples,
                key='manual_sample_override',
                help="Select all columns containing sample intensity data"
            )
            
            if manual_samples and manual_samples != detected_samples:
                st.session_state.msdial_features['raw_sample_columns'] = manual_samples
                st.success(f"✓ Using {len(manual_samples)} manually selected samples")
        
        return True, None
    
    # For Generic Format, keep the existing Yes/No radio button flow
    # Ask if mappings look correct
    st.sidebar.write("Do the column mappings look correct?")
    mapping_correct = st.sidebar.radio(
        "",
        ['Yes', 'No'],
        key='column_mapping_correct'
    )
    
    if mapping_correct == 'Yes':
        return True, None
    else:
        # Show correction interface (Generic Format only)
        return handle_column_mapping_correction(df, mapping_df, data_format)

def handle_column_mapping_correction(mapping_df, experiment, data_format):
    """
    Handle manual correction of column mappings following the Group Samples pattern.
    
    Args:
        mapping_df: DataFrame with current column mappings
        experiment: Experiment object
        data_format: Data format string
        
    Returns:
        tuple: (mapping_valid, corrected_mapping_df)
    """
    # MS-DIAL should not reach here - it uses exact column names
    if data_format == 'MS-DIAL':
        st.sidebar.error("MS-DIAL format uses exact column names. Please ensure your export has correct column names.")
        return False, None
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Correct Column Mappings")
    
    # Initialize correction complete flag
    if 'column_mapping_complete' not in st.session_state:
        st.session_state.column_mapping_complete = False
    
    # Get all columns from the original data
    if data_format == 'Generic Format':
        # For Generic format, we need the original uploaded columns
        if 'original_name' in mapping_df.columns:
            all_original_cols = mapping_df['original_name'].tolist()
        else:
            # Fallback: get from original uploaded df
            all_original_cols = st.session_state.get('original_uploaded_df', df).columns.tolist()
    else:
        # For MS-DIAL, get all columns from features
        features = st.session_state.get('msdial_features', {})
        if 'actual_columns' in features:
            all_original_cols = features['actual_columns']
        elif 'original_name' in mapping_df.columns:
            all_original_cols = mapping_df['original_name'].tolist()
        else:
            # Fallback: get from original uploaded df
            all_original_cols = st.session_state.get('original_uploaded_df', df).columns.tolist()
    
    # Define metadata columns based on format
    if data_format == 'MS-DIAL':
        metadata_options = [
            ('Metabolite name', 'LipidMolec', True),
            ('Total score', 'TotalScore', False),
            ('MS/MS matched', 'MSMSMatched', False),
            ('Average Rt(min)', 'BaseRt', False),
            ('Average Mz', 'CalcMass', False)
        ]
        
        st.sidebar.write("""
        **Map your MS-DIAL columns:**
        
        📋 **Metadata Columns:**
        Select the corresponding column from your file for each field below.
        
        ℹ️ Note: ClassKey will be automatically inferred from Metabolite name.
        """)
    else:  # Generic Format
        metadata_options = [
            ('Lipid Name', 'LipidMolec', True),
            ('Lipid Class', 'ClassKey', False),
            ('Retention Time', 'BaseRt', False),
            ('Mass', 'CalcMass', False),
            ('Adduct', 'Adduct', False),
            ('Formula', 'Formula', False)
        ]
        
        st.sidebar.write("""
        **Map your columns:**
        
        📋 **Metadata Columns:**
        Select the corresponding column from your file for each field below.
        """)
    
    # Get current metadata columns from mapping
    current_metadata_map = {}
    if 'standardized_name' in mapping_df.columns and 'original_name' in mapping_df.columns:
        for _, row in mapping_df.iterrows():
            current_metadata_map[row['standardized_name']] = row['original_name']
    
    # Add N/A option for optional fields
    dropdown_options = ['N/A'] + all_original_cols
    
    # Create 6 dropdowns for metadata columns
    selected_metadata_mapping = {}
    st.sidebar.markdown("#### 📋 Metadata Column Mapping")
    
    for orig_label, std_name, is_required in metadata_options:
        label_text = f"{orig_label} {'(REQUIRED)' if is_required else '(optional)'}"
        
        # Get current selection
        current_selection = current_metadata_map.get(std_name, 'N/A')
        if current_selection not in dropdown_options:
            current_selection = 'N/A'
        
        selected_col = st.sidebar.selectbox(
            label_text,
            options=dropdown_options,
            index=dropdown_options.index(current_selection) if current_selection in dropdown_options else 0,
            key=f'metadata_{std_name}'
        )
        
        if selected_col != 'N/A':
            selected_metadata_mapping[std_name] = selected_col
    
    # Validate required field
    if 'LipidMolec' not in selected_metadata_mapping:
        st.sidebar.error("❌ Lipid name/identifier column is required")
        return False, None
    
    # Get selected metadata columns (for exclusion from sample columns)
    selected_metadata_cols = list(selected_metadata_mapping.values())
    
    # Box 2: Sample columns
    st.sidebar.markdown("#### 📊 Select Sample Intensity Columns")
    
    # Get auto-detected sample columns (always use these as default)
    if data_format == 'MS-DIAL':
        features = st.session_state.get('msdial_features', {})
        auto_detected_samples = features.get('raw_sample_columns', [])
    else:
        # For Generic format, get from original mapping
        auto_detected_samples = []
        if 'standardized_name' in mapping_df.columns and 'original_name' in mapping_df.columns:
            auto_detected_samples = [
                row['original_name'] 
                for _, row in mapping_df.iterrows() 
                if row['standardized_name'].startswith('intensity[')
            ]
    
    # Available columns = all columns minus selected metadata
    available_for_samples = [col for col in all_original_cols if col not in selected_metadata_cols]
    
    # Get sample count from DataFrame structure
    n_intensity_cols = len(auto_detected_samples)
    
    # Pre-populate with auto-detected samples
    default_samples = [col for col in auto_detected_samples if col in available_for_samples]
    
    selected_sample_cols = st.sidebar.multiselect(
        f"Choose sample intensity columns ({n_intensity_cols} detected):",
        options=available_for_samples,
        default=default_samples,
        key='sample_columns_selection',
        help="Select all columns containing sample intensity values"
    )
    
    # Validate selections
    if len(selected_sample_cols) >= 1:
        # Create corrected mapping
        corrected_mapping = []
        
        # Add metadata mappings
        for std_name, orig_col in selected_metadata_mapping.items():
            corrected_mapping.append({
                'standardized_name': std_name,
                'original_name': orig_col
            })
        
        # Add sample mappings
        for i, orig_col in enumerate(selected_sample_cols, 1):
            corrected_mapping.append({
                'standardized_name': f'intensity[s{i}]',
                'original_name': orig_col
            })
        
        corrected_mapping_df = pd.DataFrame(corrected_mapping)
        
        st.sidebar.success(f"✓ Selections valid: {len(selected_sample_cols)} sample columns, {len(selected_metadata_mapping)} metadata columns")
        
        # Show corrected mapping
        st.sidebar.markdown("#### ✅ Corrected Column Mapping")
        st.sidebar.dataframe(
            corrected_mapping_df.reset_index(drop=True),
            use_container_width=True
        )
        
        # Update session state with corrected mapping
        st.session_state.column_mapping = corrected_mapping_df
        st.session_state.n_intensity_cols = len(selected_sample_cols)
        st.session_state.column_mapping_complete = True
        
        # Store the manual correction info so we can apply it to the DataFrame
        st.session_state.manual_column_correction = {
            'metadata_mapping': selected_metadata_mapping,
            'sample_cols': selected_sample_cols,
            'sample_mapping': {orig: f'intensity[s{i}]' for i, orig in enumerate(selected_sample_cols, 1)}
        }
        
        return True, None
    else:
        # Show validation error
        if len(selected_sample_cols) < 1:
            st.sidebar.error(f"❌ At least one sample column is required")
        
        st.session_state.column_mapping_complete = False
        return False, None

def apply_manual_column_correction(df, data_format):
    """
    Apply manual column corrections to the DataFrame.
    This re-standardizes the DataFrame based on user's manual column selections.
    
    Args:
        df: DataFrame with automatically standardized columns (not used, kept for API compatibility)
        data_format: Data format string
        
    Returns:
        DataFrame with manually corrected column standardization
    """
    if 'manual_column_correction' not in st.session_state:
        return df
    
    if 'original_uploaded_df' not in st.session_state:
        st.error("Original data not available for re-standardization")
        return df
    
    correction = st.session_state.manual_column_correction
    orig_df = st.session_state.original_uploaded_df.copy()
    
    # Create a new standardized DataFrame
    standardized_data = {}
    
    if data_format == 'MS-DIAL':
        # For MS-DIAL, we need to handle the header row situation
        features = st.session_state.get('msdial_features', {})
        header_row_idx = features.get('header_row_index', -1)
        
        if header_row_idx >= 0:
            # Extract actual columns and data
            actual_columns = orig_df.iloc[header_row_idx].tolist()
            data_df = orig_df.iloc[header_row_idx + 1:].reset_index(drop=True)
        else:
            actual_columns = orig_df.columns.tolist()
            data_df = orig_df
        
        # Map original column names to their positions
        col_to_idx = {col: idx for idx, col in enumerate(actual_columns)}
        
        # Add metadata columns
        for std_col, orig_col in correction['metadata_mapping'].items():
            if orig_col in col_to_idx:
                col_idx = col_to_idx[orig_col]
                if std_col == 'LipidMolec':
                    # Standardize lipid names
                    standardized_data[std_col] = data_df.iloc[:, col_idx].apply(
                        lp.DataFormatHandler._standardize_lipid_name
                    )
                else:
                    standardized_data[std_col] = data_df.iloc[:, col_idx]
        
        # Always infer ClassKey from LipidMolec
        if 'LipidMolec' in standardized_data:
            standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(
                lp.DataFormatHandler._infer_class_key
            )
        
        # Add sample columns
        for orig_col, std_col in correction['sample_mapping'].items():
            if orig_col in col_to_idx:
                col_idx = col_to_idx[orig_col]
                standardized_data[std_col] = pd.to_numeric(
                    data_df.iloc[:, col_idx], errors='coerce'
                ).fillna(0)
    
    else:  # Generic Format
        # For Generic, columns are straightforward
        # Add metadata columns
        for std_col, orig_col in correction['metadata_mapping'].items():
            if orig_col in orig_df.columns:
                if std_col == 'LipidMolec':
                    # Standardize lipid names
                    standardized_data[std_col] = orig_df[orig_col].apply(
                        lp.DataFormatHandler._standardize_lipid_name
                    )
                else:
                    standardized_data[std_col] = orig_df[orig_col]
        
        # Always infer ClassKey from LipidMolec
        if 'LipidMolec' in standardized_data:
            standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(
                lp.DataFormatHandler._infer_class_key
            )
        
        # Add sample columns
        for orig_col, std_col in correction['sample_mapping'].items():
            if orig_col in orig_df.columns:
                standardized_data[std_col] = pd.to_numeric(
                    orig_df[orig_col], errors='coerce'
                ).fillna(0)
    
    return pd.DataFrame(standardized_data)

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
        total_samples = sum(experiment.number_of_samples_list)
        st.sidebar.write(f"There are a total of {total_samples} samples.")

        # Display information about sample and condition pairings
        for i, condition in enumerate(experiment.conditions_list):
            if condition and condition.strip():  # Make sure condition is not empty
                build_replicate_condition_pair(condition, experiment)
            else:
                st.sidebar.error(f"Empty condition found at index {i}")

        # Display a checkbox for the user to confirm the inputs
        return st.sidebar.checkbox("Confirm the inputs by checking this box")
        
    except Exception as e:
        st.sidebar.error(f"Error in confirming user inputs: {e}")
        st.write(f"Error in confirm_user_inputs: {e}")
        return False

def build_replicate_condition_pair(condition, experiment):
    """
    Display information about sample and condition pairings in the Streamlit sidebar.
    """
    try:
        index = experiment.conditions_list.index(condition)
        samples = experiment.individual_samples_list[index]

        if len(samples) > 5:
            display_text = f"- {samples[0]} to {samples[-1]} (total {len(samples)}) correspond to {condition}"
        else:
            display_text = f"- {'-'.join(samples)} correspond to {condition}"
        
        st.sidebar.write(display_text)
        
    except Exception as e:
        st.sidebar.error(f"Error displaying condition {condition}: {str(e)}")
        st.write(f"Error in build_replicate_condition_pair: {e}")

def clean_data(df, name_df, experiment, data_format, grade_config=None, quality_config=None):
    """
    Clean data using appropriate cleaner based on the format.
    
    Args:
        df: Input dataframe
        name_df: Name mapping dataframe
        experiment: Experiment object
        data_format: Format type string
        grade_config (dict, optional): Custom grade configuration for LipidSearch data
        quality_config (dict, optional): Quality filtering config for MS-DIAL data
    
    Returns:
        Tuple of (cleaned_df, intsta_df)
    """
    if data_format == 'LipidSearch 5.0':
        cleaner = lp.CleanLipidSearchData()
        cleaned_df = cleaner.data_cleaner(df, name_df, experiment, grade_config)
    elif data_format == 'MS-DIAL':
        cleaner = lp.CleanMSDIALData()
        cleaned_df = cleaner.data_cleaner(df, name_df, experiment, quality_config)
    else:
        cleaner = lp.CleanGenericData()
        cleaned_df = cleaner.data_cleaner(df, name_df, experiment)
    
    cleaned_df, intsta_df = cleaner.extract_internal_standards(cleaned_df)
    
    # Clear UI message about internal standards extraction
    if intsta_df is not None and not intsta_df.empty:
        num_standards = intsta_df['LipidMolec'].nunique()
        st.success(f"✓ **Internal Standards Extracted:** {num_standards} internal standard(s) separated for normalization (not filtered out)")
    
    return cleaned_df, intsta_df

def get_grade_filtering_config(df, format_type):
    """
    Display grade filtering UI within the data cleaning section.
    Only shown for LipidSearch format.
    
    Args:
        df (pd.DataFrame): The uploaded dataframe
        format_type (str): The data format type
        
    Returns:
        dict: Dictionary mapping lipid class to list of acceptable grades
              Returns None if not LipidSearch format or if using defaults
    """
    if format_type != 'LipidSearch 5.0':
        return None
    
    # Check if the required columns exist
    if 'ClassKey' not in df.columns or 'TotalGrade' not in df.columns:
        return None
    
    st.markdown("---")
    st.markdown("### 🎯 Grade Filtering Configuration")
    
    st.markdown("""
    LipidSearch assigns quality grades (A, B, C, D) to each lipid identification:
    - **Grade A**: Highest confidence identifications
    - **Grade B**: Good quality identifications
    - **Grade C**: Lower confidence identifications
    - **Grade D**: Lowest confidence identifications
    
    **Default Behavior:**
    - Most lipid classes: Accept grades A and B only
    - LPC and SM classes: Accept grades A, B, and C
    """)
    
    # Get unique classes from the data
    all_classes = sorted(df['ClassKey'].unique())
    
    # Option to use default or custom settings
    use_custom = st.radio(
        "Grade filtering mode:",
        ["Use Default Settings", "Customize by Class"],
        index=0,
        help="Default settings follow LipidSearch best practices. Customize if you have specific requirements.",
        key="grade_filter_mode"
    )
    
    if use_custom == "Use Default Settings":
        st.success("✓ Using default grade filtering: A/B for all classes, plus C for LPC and SM.")
        return None
    
    # Custom settings
    st.markdown("---")
    st.markdown("#### 🔬 Select Acceptable Grades by Lipid Class")
    st.info("💡 **Tip:** Select the grades you want to accept for each lipid class. Clear selections will exclude that class entirely.")
    
    grade_config = {}
    
    # Process each lipid class with better spacing and clarity
    for idx, lipid_class in enumerate(all_classes):
        # Add spacing between classes
        if idx > 0:
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Create a container for each class with visual separation
        with st.container():
            # Class header with colored background
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #1f77b4;">📊 {lipid_class}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Default grades based on class
            if lipid_class in ['LPC', 'SM']:
                default_grades = ['A', 'B', 'C']
            else:
                default_grades = ['A', 'B']
            
            # Multiselect with clear labeling
            selected_grades = st.multiselect(
                f"Select acceptable grades for **{lipid_class}** class:",
                options=['A', 'B', 'C', 'D'],
                default=default_grades,
                key=f"grade_select_{lipid_class}",
                help=f"Choose which quality grades to accept for {lipid_class} lipids"
            )
            
            # Visual feedback
            if not selected_grades:
                st.error(f"⚠️  **Warning:** No grades selected for {lipid_class}. This class will be completely excluded from analysis!")
            elif 'D' in selected_grades:
                st.warning(f"⚠️  **Caution:** Grade D included for {lipid_class}. This may include low-confidence identifications.")
            else:
                st.success(f"✓ {lipid_class}: {', '.join(selected_grades)} accepted")
            
            grade_config[lipid_class] = selected_grades
    
    st.markdown("---")
    
    # Summary of selections
    st.markdown("### 📊 Configuration Summary")
    
    # Count classes by grade selection
    abc_classes = [cls for cls, grades in grade_config.items() if set(grades) == {'A', 'B', 'C'}]
    ab_classes = [cls for cls, grades in grade_config.items() if set(grades) == {'A', 'B'}]
    abcd_classes = [cls for cls, grades in grade_config.items() if 'D' in grades]
    custom_classes = [cls for cls, grades in grade_config.items() 
                     if set(grades) not in [{'A', 'B', 'C'}, {'A', 'B'}] and 'D' not in grades]
    excluded_classes = [cls for cls, grades in grade_config.items() if not grades]
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
            st.markdown(f"**→ A, B only** ({len(ab_classes)} classes)")
            st.markdown(f"**→ A, B only** ({len(ab_classes)} classes)")
            st.caption(', '.join(ab_classes))
            st.markdown(f"**→ A, B, C** ({len(abc_classes)} classes)")
            st.markdown(f"**→ A, B, C** ({len(abc_classes)} classes)")
            st.caption(', '.join(abc_classes))
    
    with summary_col2:
        if abcd_classes:
            st.markdown(f"**⚠️  Includes D** ({len(abcd_classes)} classes)")
            st.caption(', '.join(abcd_classes))
        if custom_classes:
            st.markdown(f"**⚙️  Custom** ({len(custom_classes)} classes)")
            for cls in custom_classes:
                st.caption(f"• {cls}: {', '.join(grade_config[cls])}")
    
    if excluded_classes:
        st.error(f"**🚫 Excluded** ({len(excluded_classes)} classes): {', '.join(excluded_classes)}")
    
    return grade_config

def get_msdial_data_type_selection():
    """
    Display MS-DIAL data type selection UI (raw vs pre-normalized).
    This is separate from quality filtering to ensure it's always available
    when both data types exist, regardless of whether quality filtering is possible.
    
    Returns:
        None (sets st.session_state.msdial_use_normalized)
    """
    # Get the features detected during validation
    features = st.session_state.get('msdial_features', {})
    
    has_normalized_data = features.get('has_normalized_data', False)
    raw_samples = features.get('raw_sample_columns', [])
    norm_samples = features.get('normalized_sample_columns', [])
    
    # Data Type Selection (if normalized data is available)
    if has_normalized_data and len(norm_samples) > 0:
        st.markdown("---")
        st.markdown("#### 📊 Data Type Selection")
        st.markdown(f"""
        Your MS-DIAL export contains both raw and pre-normalized intensity values:
        - **Raw data**: {len(raw_samples)} sample columns
        - **Normalized data**: {len(norm_samples)} sample columns (after 'Lipid IS' column)
        """)
        
        data_type = st.radio(
            "Select which data to use:",
            [f"Raw intensity values ({len(raw_samples)} samples)", 
             f"Pre-normalized values ({len(norm_samples)} samples)"],
            index=0,
            key="msdial_data_type_selection",
            help="Choose raw data if you want to apply LipidCruncher's normalization. Choose pre-normalized if MS-DIAL already normalized your data."
        )
        
        use_normalized = "Pre-normalized" in data_type
        st.session_state.msdial_use_normalized = use_normalized
        
        if use_normalized:
            st.info("📌 Using pre-normalized data. LipidCruncher's internal standard normalization will be skipped.")
        else:
            st.info("📌 Using raw intensity data. You can apply normalization in the next step.")
        
        st.markdown("---")

def get_msdial_quality_config():
    """
    Display MS-DIAL quality filtering UI.
    
    Returns:
        dict: Quality configuration with 'total_score_threshold' and 'require_msms' keys
              Returns None if quality filtering is not available
    """
    # Get the features detected during validation
    features = st.session_state.get('msdial_features', {})
    
    quality_filtering_available = features.get('has_quality_score', False)
    msms_filtering_available = features.get('has_msms_matched', False)
    
    #Quality Filtering (if quality columns available)
    if quality_filtering_available or msms_filtering_available:
        # Case 1: Total score column is available - show full filtering UI
        if quality_filtering_available:
            # Define quality options
            quality_options = {
                'Strict (Score ≥80, MS/MS required)': {
                    'total_score_threshold': 80,
                    'require_msms': True
                },
                'Moderate (Score ≥60)': {
                    'total_score_threshold': 60,
                    'require_msms': False
                },
                'Permissive (Score ≥40)': {
                    'total_score_threshold': 40,
                    'require_msms': False
                },
                'No filtering': {
                    'total_score_threshold': 0,
                    'require_msms': False
                }
            }
            
            st.markdown("""
            MS-DIAL provides quality metrics for each identification. Choose a filtering level:
            - **Strict**: Only highest confidence IDs (publication-ready)
            - **Moderate**: Good balance of quality and coverage (recommended for exploration)
            - **Permissive**: Includes lower confidence IDs (discovery mode)
            - **No filtering**: Keep all identifications
            """)
            
            selected_option = st.radio(
                "Select quality filtering level:",
                list(quality_options.keys()),
                index=1,  # Default to Moderate
                key="msdial_quality_level"
            )
            
            quality_config = quality_options[selected_option].copy()
            
            # Show what this option means
            if selected_option == 'Strict (Score ≥80, MS/MS required)':
                st.success("✓ **Strict filtering**: Only highest confidence identifications. Ideal for publication-ready data.")
            elif selected_option == 'Moderate (Score ≥60)':
                st.info("✓ **Moderate filtering**: Good balance of quality and coverage. Recommended for exploratory analysis.")
            elif selected_option == 'Permissive (Score ≥40)':
                st.warning("⚠️ **Permissive filtering**: Includes lower confidence IDs. Use for discovery or when sample is limited.")
            else:
                st.warning("⚠️ **No filtering**: All identifications included. Quality may vary significantly.")
            
            # MS/MS validation option (if available)
            if msms_filtering_available:
                custom_msms = st.checkbox(
                    "Require MS/MS validation",
                    value=quality_config['require_msms'],
                    help="If checked, only lipids with MS/MS spectral match will be kept",
                    key="msdial_custom_msms"
                )
                quality_config['require_msms'] = custom_msms
            else:
                # Explain why MS/MS filtering is not available
                st.warning("💡 **Note:** MS/MS validation filtering is not available because the 'MS/MS matched' column was not found in your export. To enable this feature, re-export from MS-DIAL with the 'MS/MS matched' column included.")
                quality_config['require_msms'] = False
            
            # Show advanced options with a checkbox toggle
            show_advanced = st.checkbox("🔧 Customize score threshold", value=False, key="msdial_show_advanced")
            
            if show_advanced:
                st.markdown("**Override the preset Total Score threshold:**")
                custom_score = st.slider(
                    "Minimum Total Score:",
                    min_value=0,
                    max_value=100,
                    value=quality_config['total_score_threshold'],
                    step=5,
                    help="Lipids with scores below this threshold will be excluded",
                    key="msdial_custom_score"
                )
                quality_config['total_score_threshold'] = custom_score
            
            # Summary
            st.markdown("---")
            st.markdown("##### 📋 Current Settings")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Score threshold:** ≥{quality_config['total_score_threshold']}")
            with col2:
                st.write(f"**MS/MS required:** {'Yes' if quality_config['require_msms'] else 'No'}")
            
            return quality_config
        
        # Case 2: Only MS/MS matched column available (no Total score)
        elif msms_filtering_available:
            st.warning("""
            ⚠️ **Limited Quality Filtering Available**
            
            Your MS-DIAL export does not include the 'Total score' column, so score-based 
            filtering is not available. However, you can still filter based on MS/MS validation.
            
            💡 **To enable full quality filtering:** Re-export from MS-DIAL with the 'Total score' column included.
            """)
            
            st.markdown("---")
            
            # Only show MS/MS filtering option
            require_msms = st.checkbox(
                "✓ Require MS/MS validation",
                value=False,
                help="If checked, only lipids with MS/MS spectral match will be kept",
                key="msdial_msms_only"
            )
            
            quality_config = {
                'total_score_threshold': 0,  # No score filtering
                'require_msms': require_msms
            }
            
            # Summary
            st.markdown("---")
            st.markdown("##### 📋 Current Settings")
            st.write(f"**Score filtering:** Not available (no 'Total score' column)")
            st.write(f"**MS/MS validation required:** {'Yes' if require_msms else 'No'}")
            
            return quality_config
    
    else:
        st.warning("""
        **Note:** Quality filtering is not available for this dataset.
        
        Your MS-DIAL export does not include 'Total score' or 'MS/MS matched' columns.
        To enable quality filtering, re-export from MS-DIAL with these columns included.
        """)
        return None

def handle_standards_upload(normalizer):
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
            # Pass both cleaned_df and intsta_df from session state
            return normalizer.process_standards_file(
                standards_df,
                st.session_state.cleaned_df,
                st.session_state.intsta_df
            )
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error reading standards file: {str(e)}")
    return None


def manage_internal_standards(normalizer):
    """Handle internal standards management workflow with simplified approach."""
    
    # Initialize with automatic detection as default
    if 'standard_source_preference' not in st.session_state:
        st.session_state.standard_source_preference = "Automatic Detection"
    
    # Radio button for standards source
    standards_source = st.radio(
        "Select standards source:",
        ["Automatic Detection", "Upload Custom Standards"],
        index=0 if st.session_state.standard_source_preference == "Automatic Detection" else 1,
        key="standards_source_radio"
    )
    
    # Save the preference
    st.session_state.standard_source_preference = standards_source
    
    # Handle automatic detection
    if standards_source == "Automatic Detection":
        # Clear any custom standards when switching to automatic detection
        if 'preserved_intsta_df' in st.session_state:
            st.session_state.preserved_intsta_df = None
        
        # Restore original auto-detected standards if they exist
        if 'original_auto_intsta_df' in st.session_state:
            st.session_state.intsta_df = st.session_state.original_auto_intsta_df.copy()
        
        # Show automatically detected standards if available
        if not st.session_state.intsta_df.empty:
            st.success(f"✓ Found {len(st.session_state.intsta_df)} standards")
            st.write("Current standards:")
            display_data(st.session_state.intsta_df, "Current Standards", "current_standards.csv", "auto")
        else:
            st.info("No internal standards were automatically detected in the dataset")
    
    # Handle custom upload
    else:
        # CRITICAL FIX: Clear intsta_df immediately if no custom standards uploaded yet
        if 'preserved_intsta_df' not in st.session_state or st.session_state.preserved_intsta_df is None:
            st.session_state.intsta_df = pd.DataFrame()
        
        # Reset display area for custom standards
        upload_container = st.container()
        
        with upload_container:
            # Ask user about mode
            st.markdown("### Are all the standards present in your main dataset?")
            standards_in_dataset = st.radio(
                "Select one:",
                options=[
                    "Yes - Extract standards from my dataset",
                    "No - I'm uploading complete standards data"
                ],
                key="standards_mode_selection",
                help="Choose 'Yes' if the standards you want to use already exist in your uploaded dataset. Choose 'No' if you're uploading external standards with their intensity values."
            )
            
            # Convert to boolean
            use_legacy_mode = standards_in_dataset.startswith("Yes")
            
            # Show mode-specific instructions
            if use_legacy_mode:
                st.info("""
                **Instructions:**
                - Upload a CSV with just one column: lipid molecule names
                - These standards must exist in your main dataset
                - Intensity values will be extracted from your dataset
                - Column names will be automatically standardized
                """)
            else:
                st.info("""
                **Instructions:**
                - Upload a CSV with:
                  - 1st column: Lipid molecule names
                  - Remaining columns: Intensity values (one column per sample)
                - The number of intensity columns must match your main dataset sample count
                - Column names will be automatically standardized
                """)
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with standards", 
                type=['csv'],
                key="standards_file_uploader"
            )
            
            # CRITICAL FIX: If file uploader was cleared, reset both intsta_df and preserved_intsta_df
            if uploaded_file is None and 'preserved_intsta_df' in st.session_state and st.session_state.preserved_intsta_df is not None:
                st.session_state.intsta_df = pd.DataFrame()
                st.session_state.preserved_intsta_df = None
                # Show info message that standards were cleared
                st.info("No internal standards available for consistency plots.")
            
            if uploaded_file is not None:
                try:
                    standards_df = pd.read_csv(uploaded_file)
                    # Process the uploaded standards with mode selection
                    new_standards_df = normalizer.process_standards_file(
                        standards_df,
                        st.session_state.cleaned_df,
                        st.session_state.intsta_df,
                        standards_in_dataset=use_legacy_mode
                    )
                    
                    if new_standards_df is not None:
                        # Update both current and preserved standards
                        st.session_state.intsta_df = new_standards_df
                        st.session_state.preserved_intsta_df = new_standards_df.copy()
                        
                        st.success(f"✓ Successfully loaded {len(new_standards_df)} custom standards")
                        st.write("Loaded custom standards:")
                        display_data(new_standards_df, "Custom Standards", "custom_standards.csv", "uploaded")
                    
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Error reading standards file: {str(e)}")


def apply_zero_filter(cleaned_df, experiment, data_format, bqc_label=None):
    """
    Applies the zero-value filter to the cleaned dataframe.
   
    For each lipid species, removes the species if the BQC condition (if present) has 50% or more
    replicates with values <= threshold OR every non-BQC condition has 75% or more replicates
    with values <= threshold. If no BQC condition exists, only the non-BQC condition applies.
   
    Args:
        cleaned_df (pd.DataFrame): Cleaned dataframe
        experiment (Experiment): Experiment object
        data_format (str): Data format
        bqc_label (str, optional): Label for Batch Quality Control samples
       
    Returns:
        tuple: (filtered DataFrame, list of removed species)
    """
    default_threshold = 30000.0 if data_format == 'LipidSearch 5.0' else 0.0
    threshold = st.number_input(
        'Enter detection threshold (values <= this are considered zero for filtering)',
        min_value=0.0,
        value=default_threshold,
        step=1.0,
        help="For non-LipidSearch formats, 0 means only exact zeros are considered. Enter a value >0 if needed.",
        key="zero_filter_threshold"
    )
   
    # Get all lipid species before filtering
    all_species = cleaned_df['LipidMolec'].tolist()
   
    # List to keep track of rows to keep
    to_keep = []
   
    for idx, row in cleaned_df.iterrows():
        non_bqc_all_fail = True
        bqc_fail = False if bqc_label is None or bqc_label not in experiment.conditions_list else True  # Default to False if no valid BQC condition
       
        for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
            if not cond_samples:  # Skip empty conditions
                continue
           
            zero_count = 0
            n_samples = len(cond_samples)
           
            for sample in cond_samples:
                col = f'intensity[{sample}]'
                if col in cleaned_df.columns:
                    value = row[col]
                    if value <= threshold:
                        zero_count += 1
           
            # Set threshold based on whether the condition is BQC
            zero_threshold = 0.5 if experiment.conditions_list[cond_idx] == bqc_label else 0.75
           
            # Calculate zero proportion
            zero_proportion = zero_count / n_samples if n_samples > 0 else 1.0
           
            # Check if condition passes its threshold
            if zero_proportion < zero_threshold:
                if experiment.conditions_list[cond_idx] == bqc_label:
                    bqc_fail = False
                else:
                    non_bqc_all_fail = False
       
        # Retain lipid if BQC does not fail AND not all non-BQC conditions fail
        if not bqc_fail and not non_bqc_all_fail:
            to_keep.append(idx)
   
    filtered_df = cleaned_df.loc[to_keep].reset_index(drop=True)
   
    # Compute removed species
    removed_species = [species for species in all_species if species not in filtered_df['LipidMolec'].tolist()]
   
    return filtered_df, removed_species

def display_cleaned_data(unfiltered_df, intsta_df):
    """
    Display cleaned data and manage internal standards with simplified workflow.
    Includes grade filtering, zero filter application, and display of removed species.
    """
    # Update session state with new data if provided
    if unfiltered_df is not None:
        st.session_state.cleaned_df = unfiltered_df.copy()
        
        # CRITICAL: Only set intsta_df if user is in "Automatic Detection" mode
        # Don't overwrite if user has switched to "Upload Custom Standards"
        if 'standard_source_preference' not in st.session_state or st.session_state.standard_source_preference == "Automatic Detection":
            st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()
        # If in custom mode, preserve whatever is in intsta_df (could be empty or custom uploaded)
    
    # Create normalizer instance
    normalizer = lp.NormalizeData()
    
    # Display cleaned data
    with st.expander("Clean and Filter Data"):
        # Data cleaning process details - ALWAYS VISIBLE
        st.markdown("### Data Cleaning, Standardization and Filtering Process")
        st.markdown("""
        LipidCruncher performs a systematic cleaning and standardization process on your uploaded data
        to ensure consistency, reliability, and compatibility with downstream analyses. The specific
        procedures applied depend on your data format.
        """)
        
        # Tabs for different data formats
        data_format_tab = st.radio(
            "Select data format to view cleaning details:",
            ["LipidSearch Format", "Generic Format", "MS-DIAL Format", "Metabolomics Workbench"],
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
           
            4. **Quality Filtering**: By default, entries with grades 'A' and 'B' are retained for most classes,
               with grades 'A', 'B', and 'C' accepted for LPC and SM classes. You can customize this below.
           
            5. **Best Peak Selection**: For each unique lipid, the entry with the highest TotalSmpIDRate(%)
               is selected, as this indicates the most reliable measurement across samples.
           
            6. **Missing FA Keys Handling**: Rows with missing FA keys are removed, with exceptions
               for cholesterol (Ch) class molecules and deuterated standards.
           
            7. **Duplicate Removal**: Duplicate entries based on LipidMolec are removed.
           
            8. **Zero Filtering**: Removes lipid species where the BQC condition (if present) has 50% or more replicates with intensity values <= user-specified detection threshold OR every non-BQC condition has 75% or more replicates with such values.
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
               - LPC O-17:4  →  LPC(O-17:4)
               - Cer d18:0/C24:0  →  Cer(d18:0_C24:0)
               - CE 14:0;0  →  CE(14:0)
           
            3. **Class Key Extraction**: A 'ClassKey' column is generated by extracting the lipid class
               from the standardized lipid name (e.g., 'PC' from 'PC(16:0_18:1)').
           
            4. **Data Type Conversion**: Intensity columns are converted to numeric format, with any
               non-numeric entries replaced by zeros.
           
            5. **Invalid Lipid Removal**: Rows with invalid lipid names (empty strings, single special
               characters, strings with only special characters) are removed.
           
            6. **Duplicate Removal**: Duplicate entries based on LipidMolec are removed.
           
            7. **Zero Filtering**: Removes lipid species where the BQC condition (if present) has 50% or more replicates with intensity values <= user-specified detection threshold OR every non-BQC condition has 75% or more replicates with such values.
            """)
           
            st.info("This standardization process allows for consistent analysis regardless of the original format of your data.")
           
        elif data_format_tab == "MS-DIAL Format":
            st.markdown("#### Data Cleaning for MS-DIAL Format")
            st.markdown("""
            For MS-DIAL data, we perform the following standardization and cleaning steps:
           
            1. **Header Detection**: MS-DIAL exports contain metadata rows. LipidCruncher 
               automatically detects where the actual data begins by locating the row where 'Alignment ID' 
               contains numeric values.
           
            2. **Column Standardization**: We map MS-DIAL columns to LipidCruncher's standard format:
               - `Metabolite name` → `LipidMolec`
               - `ClassKey` → Inferred from `LipidMolec` (e.g., Cer(18:1;2O_24:0) → Cer)
               - `Average Rt(min)` → `BaseRt`
               - `Average Mz` → `CalcMass`
               - Sample intensity columns → `intensity[s1]`, `intensity[s2]`, etc.
           
            3. **Lipid Name Standardization**: Lipid names are standardized to consistent format while 
               preserving biologically important hydroxyl notation:
               - Cer 18:1;2O/24:0 → Cer(18:1;2O_24:0)
               - PC 16:0_18:1 → PC(16:0_18:1)
               - Hydroxyl groups (;0, ;2O, ;3O) are preserved
           
            4. **Quality Filtering**: MS-DIAL provides quality metrics that can be used for filtering:
               - **Total Score**: Composite confidence score (0-100) based on accurate mass, isotopic 
                 pattern, MS/MS spectrum match, and retention time
               - **MS/MS Matched**: TRUE/FALSE indicating whether MS/MS spectrum validation passed
               - Three filtering levels available: Strict (≥80 + MS/MS), Moderate (≥60), Permissive (≥40)
               - Configurable above in the quality filtering section
           
            5. **Data Type Selection**: If your MS-DIAL export contains both raw and pre-normalized data 
               (e.g., pmol/mg tissue after internal standard correction), you can choose which to use. 
               Select raw data to apply LipidCruncher's normalization, or pre-normalized if MS-DIAL 
               already performed normalization.
           
            6. **Data Type Conversion**: Intensity columns are converted to numeric format, with any 
               non-numeric entries replaced by zeros.
           
            7. **Invalid Lipid Removal**: Rows with invalid lipid names (empty strings, single special 
               characters) are removed.
           
            8. **Smart Deduplication**: When duplicate lipid identifications exist, the entry with the 
               highest Total Score is retained, ensuring the most confident identification is used.
           
            9. **Internal Standards Extraction**: Standards are automatically detected by patterns including:
               - Deuterated labels: (d5), (d7), (d9)
               - ISTD markers in ClassKey or Ontology columns
               - SPLASH LIPIDOMIX® standard patterns
           
            10. **Duplicate Removal**: Any remaining duplicate entries based on LipidMolec are removed.
           
            11. **Zero Filtering**: Removes lipid species where the BQC condition (if present) has 50% or more 
                replicates with intensity values ≤ user-specified detection threshold OR every non-BQC 
                condition has 75% or more replicates with such values.
            """)
           
            st.info("MS-DIAL integration provides flexible quality control with support for both raw and pre-normalized data workflows.")
        
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
           
            8. **Zero Filtering**: Removes lipid species where the BQC condition (if present) has 50% or more replicates with intensity values <= user-specified detection threshold OR every non-BQC condition has 75% or more replicates with such values.
            """)
           
            st.info("This process ensures that your Metabolomics Workbench data is properly formatted for analysis in LipidCruncher.")
       
        # Add a divider
        st.markdown("---")
       
        # Zero filtering section
        st.subheader("Zero Filtering")
        st.markdown("""
        This filter removes lipid species if the BQC condition (if present) has 50% or more replicates with intensity values
        less than or equal to the specified detection threshold (considered as zero or below detection) OR if every non-BQC
        condition has 75% or more such replicates. If no BQC condition exists, only the non-BQC condition applies.
        """)
        
        # Check if cleaned_df is valid
        if st.session_state.cleaned_df is None or st.session_state.cleaned_df.empty:
            st.error("No valid cleaned data available for zero filtering.")
            return None
        
        # Apply filter
        filtered_df, removed_species = apply_zero_filter(
            st.session_state.cleaned_df,
            st.session_state.experiment,
            st.session_state.format_type,
            bqc_label=st.session_state.bqc_label
        )
        
        # Display removed species immediately after the threshold input
        removed = len(removed_species)
        if removed > 0:
            st.info(f"Removed {removed} species based on zero filter ({removed / len(st.session_state.cleaned_df) * 100:.1f}%)")
            st.markdown("**Removed Species:**")
            removed_df = pd.DataFrame({'Removed LipidMolec': removed_species})
            st.dataframe(removed_df)
            csv = removed_df.to_csv(index=False)
            st.download_button(
                label="Download Removed Species",
                data=csv,
                file_name="removed_species.csv",
                mime="text/csv"
            )
        else:
            st.info("No species removed by zero filter")
        
        # Update session state with filtered data
        st.session_state.cleaned_df = filtered_df
        st.session_state.continuation_df = filtered_df
        
        # Display the filtered data within the main expander
        st.markdown("---")
        st.subheader("Final Cleaned and Filtered Data")
        st.write("This table shows your data after all cleaning steps, including zero filtering:")
        display_data(filtered_df, "Data", "final_cleaned_data.csv")
    
    # Internal standards management
    with st.expander("Manage Internal Standards"):
        # Internal standards detection details - ALWAYS VISIBLE
        st.markdown("### Internal Standards Detection")
        st.markdown("""
        LipidCruncher automatically identifies internal standards from the SPLASH LIPIDOMIX® Mass Spec Standard (Avanti Polar Lipids, Cat# 330707-1)
        by detecting patterns like "+D7" or ":(s)" notation in lipid names. If you use custom standards with different naming conventions, you can upload them using the
        option below.
        """)
           
        # Add a divider
        st.markdown("---")
       
        manage_internal_standards(normalizer)
        
        # CRITICAL FIX: Only show plots based on mode and data availability
        show_plots = False
        if st.session_state.standard_source_preference == "Automatic Detection":
            show_plots = not st.session_state.intsta_df.empty
        else:  # Upload Custom Standards
            show_plots = ('preserved_intsta_df' in st.session_state and 
                         st.session_state.preserved_intsta_df is not None and 
                         not st.session_state.preserved_intsta_df.empty)
        
        if show_plots:
            st.markdown("### Internal Standards Consistency Plots")
            st.markdown("""
            These stacked bar plots show the raw intensity values of internal standards across all samples, separated by class.
            For classes with multiple standards, bars are stacked by individual standard.
            Consistent bar heights across samples indicate good sample preparation and instrument performance.
            """)
           
            # Add multiselect for conditions
            conditions = st.session_state.experiment.conditions_list
            selected_conditions = st.multiselect(
                'Select conditions to display:',
                conditions,
                default=conditions
            )
           
            # Get selected samples in original order
            selected_samples = []
            for cond in selected_conditions:
                idx = conditions.index(cond)
                selected_samples.extend(st.session_state.experiment.individual_samples_list[idx])
           
            # Preserve original full order but filter to selected
            full_samples = st.session_state.experiment.full_samples_list
            selected_samples_ordered = [s for s in full_samples if s in selected_samples]
           
            # Generate plots with selected samples
            figs = lp.InternalStandardsPlotter.create_consistency_plots(
                st.session_state.intsta_df,
                selected_samples_ordered
            )
           
            if figs:
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
                   
                    # Add download options
                    col1, col2 = st.columns(2)
                    with col1:
                        plotly_svg_download_button(fig, f"internal_standards_consistency_{fig.layout.title.text.replace(' ', '_')}.svg")
                    with col2:
                        # Compute data for this fig
                        class_key = fig.layout.title.text.split(' for ')[-1]
                        class_df = st.session_state.intsta_df[st.session_state.intsta_df['ClassKey'] == class_key]
                        intensity_cols = [f'intensity[{s}]' for s in selected_samples_ordered if f'intensity[{s}]' in class_df.columns]
                        melted_df = pd.melt(
                            class_df,
                            id_vars=['LipidMolec'],
                            value_vars=intensity_cols,
                            var_name='Sample_Column',
                            value_name='Intensity'
                        )
                        melted_df['Sample'] = melted_df['Sample_Column'].str.replace('intensity[', '', regex=False).str.replace(']', '', regex=False)
                        csv = melted_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f'internal_standards_consistency_{class_key}.csv',
                            mime='text/csv'
                        )
            else:
                st.info("No internal standards available for consistency plots.")
        else:
            st.info("No internal standards available for consistency plots.")
   
    # Return the filtered dataframe for further processing
    return filtered_df
            
def rename_intensity_to_concentration(df):
    """Renames intensity columns to concentration columns at the end of normalization"""
    df = df.copy()
    rename_dict = {
        col: col.replace('intensity[', 'concentration[')
        for col in df.columns if col.startswith('intensity[')
    }
    return df.rename(columns=rename_dict)

def handle_data_normalization(cleaned_df, intsta_df, experiment, format_type):
    """Handle data normalization with improved session state persistence."""
    
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
    
    # FIXED: Always default to all classes if nothing is selected
    selected_classes = st.multiselect(
        'Select lipid classes you would like to analyze:',
        options=all_class_lst,
        default=all_class_lst if not st.session_state.selected_classes else st.session_state.selected_classes,
        key='temp_selected_classes',
        on_change=update_selected_classes
    )
    
    # Use the direct selection result
    if not selected_classes:
        st.warning("Please select at least one lipid class to proceed with normalization.")
        return None

    # Filter DataFrame based on selected classes
    filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()

    # Check if we have standards from intsta_df
    has_standards = not intsta_df.empty

    # Store standard source and intsta_df persistently in session state if not already present
    if 'standard_source' not in st.session_state:
        st.session_state.standard_source = "Automatic Detection" if has_standards else None
    
    # CRITICAL FIX: Only preserve CUSTOM uploaded standards, not auto-detected ones
    # preserved_intsta_df should ONLY be set when user uploads a custom file
    # Don't initialize it with auto-detected standards
    if 'preserved_intsta_df' not in st.session_state:
        st.session_state.preserved_intsta_df = None  # Start as None, only set by custom upload
    
    normalized_data_object = lp.NormalizeData()

    # No automatic fallback - user must explicitly select custom standards mode
    # and upload a file if they want to use custom standards

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

    # Determine available normalization options based on current standards availability
    normalization_options = ['None', 'Internal Standards', 'Protein-based', 'Both'] if has_standards else ['None', 'Protein-based']
    
    # IMPORTANT FIX: If user had previously chosen standards-based method but standards are now missing, add a notice
    if 'normalization_method' in st.session_state and st.session_state.normalization_method in ['Internal Standards', 'Both'] and not has_standards:
        st.warning("Your previous normalization method required internal standards, but none are currently available. Please upload standards or select a different method.")
    
    # Initialize or retrieve normalization method from session state
    if 'normalization_method' not in st.session_state:
        st.session_state.normalization_method = 'None'
    
    # Validate saved normalization method against current options
    if st.session_state.normalization_method not in normalization_options:
        # IMPORTANT FIX: Don't automatically reset to 'None' if standards might be uploaded
        if not ('normalization_method' in st.session_state and 
                st.session_state.normalization_method in ['Internal Standards', 'Both']):
            st.session_state.normalization_method = 'None'
            
    # Find the correct index to pre-select, handling the case where the method isn't available
    if st.session_state.normalization_method in normalization_options:
        default_index = normalization_options.index(st.session_state.normalization_method)
    else:
        default_index = 0
        
    # Select normalization method with session state persistence
    normalization_method = st.radio(
        "Select normalization method:",
        options=normalization_options,
        key='norm_method_selection',
        index=default_index
    )
    
    # Update session state
    st.session_state.normalization_method = normalization_method

    normalized_df = filtered_df.copy()

    if normalization_method != 'None':
        # Store normalization settings in session state
        if 'normalization_settings' not in st.session_state:
            st.session_state.normalization_settings = {}
        
        # Track whether we need to do standards normalization
        do_standards = normalization_method in ['Internal Standards', 'Both'] and has_standards
        
        # Handle Protein-based normalization
        if normalization_method in ['Protein-based', 'Both']:
            with st.expander("Enter Protein Concentration Data"):
                # IMPORTANT FIX: Preserve protein concentration inputs
                protein_df = None
                if 'protein_df' in st.session_state:
                    protein_df = st.session_state.protein_df
                
                new_protein_df = collect_protein_concentrations(experiment)
                
                if new_protein_df is not None:
                    # Update the stored protein_df in session state
                    st.session_state.protein_df = new_protein_df
                    protein_df = new_protein_df
                
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
                if 'ClassKey' in intsta_df.columns:
                    standards_by_class = intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
        
                # Process standards with session state preservation
                class_to_standard_map = process_class_standards(
                    selected_classes, 
                    standards_by_class, 
                    intsta_df,
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
                    intsta_df, 
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

def process_class_standards(selected_classes, standards_by_class, intsta_df, saved_mappings=None):
    """
    Process class-standard mapping with session state preservation, ensuring class-specific defaults.

    Args:
        selected_classes (list): List of lipid classes to normalize.
        standards_by_class (dict): Dictionary mapping ClassKey to list of LipidMolec standards.
        intsta_df (pd.DataFrame): DataFrame containing internal standards.
        saved_mappings (dict, optional): Previously saved class-to-standard mappings.

    Returns:
        dict or None: Mapping of lipid classes to selected standards, or None if invalid.
    """
    
    class_to_standard_map = {}
    all_available_standards = list(intsta_df['LipidMolec'].unique())
    
    # Handle case where no standards are available
    if not all_available_standards:
        st.error("No internal standards available for normalization.")
        return None
    
    for lipid_class in selected_classes:
        default_standard = None
        if saved_mappings and lipid_class in saved_mappings:
            default_standard = saved_mappings[lipid_class]
            if default_standard not in all_available_standards:
                default_standard = None

        if not default_standard:
            class_specific_standards = standards_by_class.get(lipid_class, [])
            if class_specific_standards:
                default_standard = class_specific_standards[0]
            else:
                default_standard = all_available_standards[0]
                st.warning(f"No specific standards available for {lipid_class}. Defaulting to {default_standard}.")

        default_idx = all_available_standards.index(default_standard) if default_standard in all_available_standards else 0

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
        
def display_data(df, title, filename, key_suffix=''):
    """
    Display a DataFrame in an expander with download option.
    
    Parameters:
        df (pd.DataFrame): DataFrame to display
        title (str): Title for the data section
        filename (str): Name of file for download
        key_suffix (str): Optional suffix to make widget keys unique
    """
    st.write(df)
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {title}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}_{key_suffix}"  # Add unique key
    )
    
def quality_check_and_analysis_module(continuation_df, intsta_df, experiment, bqc_label, format_type):
    """
    Updated quality check and analysis module with enhanced saturation plots and FACH.
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
    if 'fach_plots' not in st.session_state:
        st.session_state.fach_plots = {}  # Ensure this is initialized

    # Quality Check
    box_plot_fig1, box_plot_fig2 = display_box_plots(continuation_df, experiment)
    continuation_df, bqc_plot = conduct_bqc_quality_assessment(bqc_label, continuation_df, experiment)
   
    # Show retention time plots for formats that have retention time and mass data
    # (LipidSearch 5.0 always has these; MS-DIAL has them if Average Rt(min) and Average Mz were in the export)
    if format_type in ['LipidSearch 5.0', 'MS-DIAL']:
        retention_time_plot = display_retention_time_plots(continuation_df, format_type)
   
    # Pairwise Correlation Analysis
    selected_condition, corr_fig = analyze_pairwise_correlation(continuation_df, experiment)
    if selected_condition and corr_fig:
        st.session_state.correlation_plots[selected_condition] = corr_fig
   
    # PCA Analysis
    continuation_df, pca_plot = display_pca_analysis(continuation_df, experiment)
   
    st.subheader("Data Visualization, Interpretation, and Analysis Module")

    # Central statistical testing documentation
    with st.expander("📊 About Statistical Testing"):
        st.markdown("""
        ## 🧪 Statistical Testing in LipidCruncher

        LipidCruncher uses rigorous statistical methods to help you identify meaningful differences in your lipidomic data. This guide explains everything you need to know—no statistics PhD required!

        ---

        ### 🎯 The Big Picture

        When comparing lipid levels across conditions, we're asking: *"Is this difference real, or just random noise?"*

        Statistical tests give us a **p-value**—the probability of seeing this difference (or more extreme) by chance alone. Lower p-values = stronger evidence of a real difference.

        **The catch?** When you test many things at once, some will look significant just by luck. Test 100 lipids, and ~5 will have p < 0.05 purely by chance. That's where multiple testing corrections come in.

        ---

        ### 🔬 Available Tests

        | Test Type | 2 Conditions | 3+ Conditions | When to Use |
        |-----------|--------------|---------------|-------------|
        | **Parametric** | Welch's t-test | Welch's ANOVA | Default choice. Works great with log-transformed lipidomic data |
        | **Non-parametric** | Mann-Whitney U | Kruskal-Wallis | More conservative. Use when you want maximum rigor or have unusual distributions |

        **Why Welch's?** Unlike classic t-tests and ANOVA, Welch's versions don't assume equal variances across groups—important for biological data where variability often differs between conditions.

        **Why log transformation?** Lipidomic data is typically right-skewed (a few very high values). Log10 transformation makes the data more symmetric and suitable for parametric tests. This is standard practice in the field.

        ---

        ### 🎚️ Two-Level Correction Framework

        LipidCruncher uses a **two-level system** to control false discoveries at different scales:

        #### Level 1: Between-Class Correction
        *"Across all the lipid classes I'm testing, how do I control false discoveries?"*

        | Method | What It Does | Best For |
        |--------|--------------|----------|
        | **Uncorrected** | Raw p-values, no adjustment | Single pre-specified hypothesis |
        | **FDR** (Benjamini-Hochberg) | Controls the *proportion* of false discoveries | Exploratory analysis (recommended default) |
        | **Bonferroni** | Strictest control, adjusts for every test | Confirmatory studies where false positives are costly |

        #### Level 2: Within-Class Post-hoc Correction
        *"With 3+ conditions, which specific pairs are different?"*

        After a significant omnibus test (ANOVA/Kruskal-Wallis), post-hoc tests identify *which* groups differ:

        | Method | What It Does | Best For |
        |--------|--------------|----------|
        | **Uncorrected** | Raw pairwise p-values | Maximum discovery (higher false positive risk) |
        | **Tukey's HSD** | Controls family-wise error rate efficiently | Recommended for parametric tests |
        | **Bonferroni** | Most conservative pairwise correction | Maximum rigor |

        **Note:** Tukey's HSD is only available for parametric tests. When you select non-parametric analysis, the "Tukey's HSD" option automatically switches to Bonferroni-corrected Mann-Whitney U pairwise tests.

        ---

        ### 🤖 Auto Mode: Smart Defaults

        Don't want to think about statistics? Auto mode applies field-standard choices:

        - **Test type:** Parametric (Welch's) with log10 transformation
        - **Level 1:** Uncorrected for single class → FDR for multiple classes
        - **Level 2:** Uncorrected for ≤2 conditions → Tukey's HSD for 3+ conditions

        These defaults balance discovery power with false positive control for typical lipidomics experiments.

        ---

        ### 📊 The Testing Process

        **2 Conditions:** Direct pairwise test → Level 1 correction (if testing multiple classes)

        **3+ Conditions:**

        1. **Omnibus test** for each lipid class: "Are there ANY differences among groups?"
        2. **Level 1 correction** adjusts all omnibus p-values across classes
        3. **Post-hoc tests** run only for classes with significant omnibus results
        4. **Level 2 correction** adjusts pairwise p-values within each class

        **How the levels work together:**
        - Level 1 asks: "Which lipid classes show any difference?" (corrects across classes)
        - Level 2 asks: "Within those classes, which specific condition pairs differ?" (corrects within each class)

        ---

        ### ⚠️ Critical Best Practices

        1. **Pre-specify your hypotheses.** Decide which lipid classes to test *before* looking at results. Adding "interesting-looking" classes after the fact inflates false positives.

        2. **More tests = less power.** Each additional class or comparison dilutes your statistical power. Focus on what matters biologically.

        3. **Sample size matters.** With n=3 per group (common in lipidomics), non-parametric tests have very limited resolution—you may see identical p-values for many comparisons. Parametric tests with log transformation typically perform better.

        4. **Significance ≠ importance.** A tiny p-value doesn't mean a biologically meaningful effect. Always consider effect sizes (fold changes) alongside p-values.

        **Rule of thumb:** FDR + Tukey's HSD works well for most lipidomics analyses.
        """)

    # Saturation profile methodology documentation
    with st.expander("📊 About Saturation Profile Calculations"):
        st.markdown("""
        ## How SFA, MUFA, and PUFA Values Are Computed

        For each lipid molecule in your dataset, the algorithm:

        ### 1. Fatty Acid Chain Parsing
        Identifies individual fatty acid chains from the lipid name:
        - ✓ `PC(16:0_18:1)` → Successfully parses to chains: [16:0, 18:1]
        - ✓ `PE(18:0_20:4)` → Successfully parses to chains: [18:0, 20:4]
        - ✗ `PC(34:1)` → Cannot parse individual chains (consolidated format)

        ### 2. Classification by Saturation
        Counts double bonds in each chain:
        - **SFA** (0 double bonds): 16:0, 18:0, 24:0
        - **MUFA** (1 double bond): 16:1, 18:1, 24:1
        - **PUFA** (2+ double bonds): 18:2, 20:4, 22:6

        ### 3. Weighted Contribution Calculation
        Multiplies lipid concentration by fatty acid ratio.

        **Example:** `PC(16:0_18:1)` at 100 µM contributes:
        - SFA: 100 × 0.5 = 50 µM (1 of 2 chains is saturated)
        - MUFA: 100 × 0.5 = 50 µM (1 of 2 chains is monounsaturated)
        - PUFA: 100 × 0 = 0 µM (no polyunsaturated chains)

        **Why consolidated format fails:** `PC(36:2)` at 100 µM could be:
        - `PC(18:0_18:2)`: 50 µM SFA + 50 µM PUFA
        - `PC(18:1_18:1)`: 100 µM MUFA
        - `PC(16:0_20:2)`: 50 µM SFA + 50 µM PUFA
        - **Same total, completely different saturation profiles** — algorithm cannot determine which

        ### 4. Class-Level Summation
        All contributions within a lipid class are summed:
        - Total SFA = sum of all SFA contributions from all species
        - Total MUFA = sum of all MUFA contributions from all species
        - Total PUFA = sum of all PUFA contributions from all species

        ### Two Visualizations

        **Concentration Profile:** Absolute SFA, MUFA, PUFA values with error bars (standard deviation)

        **Percentage Distribution:** Relative proportions (always sums to 100%)

        ### Handling Consolidated Format Data

        LipidCruncher automatically detects lipids in consolidated format within your selected classes.

        **What happens if you keep them:** The algorithm classifies based on total double bonds only. 
        For example, `PC(34:1)` is treated as 100% MUFA, when it might actually be 50% SFA + 50% MUFA.

        **What happens if you exclude them:** The remaining lipids are classified accurately, 
        but you lose the abundance contribution from excluded species.

        You'll be prompted to review detected lipids and decide based on your analysis goals.
        """)
    
    # Analysis
    analysis_option = st.radio(
        "Select an analysis feature:",
        (
            "Class Level Breakdown - Bar Chart",
            "Class Level Breakdown - Pie Charts",
            "Class Level Breakdown - Saturation Plots (requires detailed fatty acid composition)",
            "Class Level Breakdown - Fatty Acid Composition Heatmaps",  # Added FACH option
            "Class Level Breakdown - Pathway Visualization (requires detailed fatty acid composition)",
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
    elif analysis_option == "Class Level Breakdown - Saturation Plots (requires detailed fatty acid composition)":
        saturation_plots = display_saturation_plots(experiment, continuation_df)
        st.session_state.saturation_plots.update(saturation_plots)
    elif analysis_option == "Class Level Breakdown - Fatty Acid Composition Heatmaps":
        fach_plots = display_fach_heatmaps(experiment, continuation_df)
        st.session_state.fach_plots.update(fach_plots)  # Store FACH plots
    elif analysis_option == "Class Level Breakdown - Pathway Visualization (requires detailed fatty acid composition)":
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
        "⚠️  Important: PDF Report Generation Guidelines\n\n"
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
                    st.session_state.pathway_visualization, st.session_state.fach_plots
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
        # Box plot analysis details - ALWAYS VISIBLE
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
    """
    Conducts quality assessment using BQC samples and generates a CoV scatter plot.
    """
    scatter_plot = None  # Initialize scatter_plot to None
    if bqc_label is not None:
        with st.expander("Quality Check Using BQC Samples"):
            # BQC analysis details - ALWAYS VISIBLE
            st.markdown("### Batch Quality Control (BQC) Analysis")
            st.markdown("""
            Batch Quality Control (BQC) analysis evaluates the reliability of your lipidomic data by analyzing the variability of measurements in BQC samples, which are special quality control samples run alongside your experiment. This analysis uses the Coefficient of Variation (CoV) to measure how consistent the measurements are for each lipid across these samples.

            **What is CoV?**  
            The Coefficient of Variation (CoV) is a measure of relative variability, calculated as:

            ```
            CoV = (Standard Deviation / Mean) × 100%
            ```

            A lower CoV indicates more consistent measurements, suggesting higher reliability. CoV is computed for each lipid species across all BQC samples.

            **Filtering Option:**  
            You can filter out lipids with high CoV values (above your chosen threshold) to focus on reliable data for downstream analyses. If you select "Yes" for filtering:  
            - Lipids with CoV above the threshold are listed in a table.  
            - You can review these lipids and choose to add back any you want to keep (e.g., lipids critical to your study).
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
            scatter_plot, prepared_df, reliable_data_percent, filtered_lipids = lp.BQCQualityCheck.generate_and_display_cov_plot(
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
            # Filtering section for CoV only
            if prepared_df is not None and not prepared_df.empty:
                st.subheader("Data Filtering Options")
                # CoV-based filtering
                filter_cov = st.radio(
                    f"Filter lipids with CoV >= {cov_threshold}%?",
                    ("No", "Yes"),
                    index=0
                )
                cov_to_keep = []
                if filter_cov == "Yes":
                    cov_filtered_df = prepared_df[prepared_df['cov'] >= cov_threshold][['LipidMolec', 'ClassKey', 'cov', 'mean']]
                    cov_filtered_df['cov'] = cov_filtered_df['cov'].round(2)
                    cov_filtered_df['mean'] = cov_filtered_df['mean'].round(4)
                    cov_filtered_df['Reason'] = f'CoV >= {cov_threshold}%'
                    if not cov_filtered_df.empty:
                        st.subheader("Lipids with High CoV")
                        st.markdown("""
                        The following lipids have CoV values above the threshold.
                        You can review them and select any to keep in the dataset.
                        """)
                        display_cols = ['LipidMolec', 'ClassKey', 'cov', 'mean', 'Reason']
                        st.dataframe(cov_filtered_df[display_cols])
                        cov_to_keep = st.multiselect(
                            "Select lipids to keep despite high CoV:",
                            options=cov_filtered_df['LipidMolec'],
                            format_func=lambda x: f"{x} (CoV: {cov_filtered_df[cov_filtered_df['LipidMolec'] == x]['cov'].iloc[0]}%, Mean: {cov_filtered_df[cov_filtered_df['LipidMolec'] == x]['mean'].iloc[0]})",
                            help="Select any lipids you want to retain in the dataset."
                        )
                    else:
                        st.info(f"No lipids with CoV >= {cov_threshold}% found.")
                else:
                    # If No, keep all (don't filter any high CoV)
                    cov_high = prepared_df[prepared_df['cov'] >= cov_threshold]['LipidMolec'].tolist()
                    cov_to_keep = cov_high
                # Apply filters
                filtered_df = data_df.copy()
                # No zero-based filtering, only CoV-based
                cov_to_remove = [lipid for lipid in prepared_df[prepared_df['cov'] >= cov_threshold]['LipidMolec'] if lipid not in cov_to_keep]
                to_remove = set(cov_to_remove)
                # Apply removal
                filtered_df = filtered_df[~filtered_df['LipidMolec'].isin(to_remove)]
                # Sort and reset
                filtered_df = filtered_df.sort_values(by='ClassKey').reset_index(drop=True)
                # Calculate statistics
                removed_count = len(data_df) - len(filtered_df)
                percentage_removed = round((removed_count / len(data_df)) * 100, 1) if len(data_df) > 0 else 0
                if removed_count > 0:
                    st.warning(f"Filtering removed {removed_count} lipid species ({percentage_removed:.1f}% of the dataset).")
                else:
                    st.success("No lipids were removed after filtering.")
                # Calculate added back counts for CoV filter
                kept_cov = len(cov_to_keep) if filter_cov == "Yes" else 0
                if kept_cov > 0:
                    st.info(f"Added back {kept_cov} lipids based on your selections despite filtering criteria.")
                st.write('Final filtered dataset:')
                st.dataframe(filtered_df)
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Data",
                    data=csv,
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
        format_type (str): The format of the input data (e.g., 'LipidSearch 5.0', 'MS-DIAL')

    Returns:
        plotly.graph_objs._figure.Figure or None: The multi-class retention time comparison plot if generated, else None.
    """
    # Check if required columns are present for retention time plots
    has_required_columns = 'BaseRt' in continuation_df.columns and 'CalcMass' in continuation_df.columns
    
    if (format_type in ['LipidSearch 5.0', 'MS-DIAL']) and has_required_columns:
        with st.expander('View Retention Time Plots: Check Sanity of Data'):
            # Add explanation about retention time analysis - ALWAYS VISIBLE
            st.markdown("### Retention Time Analysis")
            st.markdown("""
            Retention time analysis is a crucial quality check for lipidomic data. This visualization plots the retention time of each lipid against its calculated mass, allowing you to verify the consistency and reliability of lipid identification.
            
            **What is Retention Time?**  
            Retention time is the duration a molecule takes to travel through a chromatography column. It directly correlates with a lipid's hydrophobicityâ€"more hydrophobic lipids interact more strongly with the column and typically have longer retention times.
            
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
        # Correlation analysis details - ALWAYS VISIBLE
        st.markdown("### Pairwise Correlation Analysis")
        st.markdown("""
        Pairwise correlation analysis helps assess the reproducibility of your lipidomic measurements by calculating how closely related the measurements are between sample replicates.    **What is Correlation Analysis?**  
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
        - For biological replicates: Higher correlations suggest good reproducibility, while lower correlations may indicate potential issues or greater biological variation.
        - For technical replicates: Even higher correlations are typically expected, as they represent repeated measurements of the same sample, with lower values potentially signaling technical issues.
        
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
        # PCA analysis details - ALWAYS VISIBLE
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

def display_statistical_options():
    """
    Display UI components for statistical testing options with auto/manual mode selection.
    Returns the selected options as a dictionary.
    """
    
    # First choice: Auto vs Manual mode
    mode_choice = st.radio(
        "Select Analysis Mode:",
        options=["Manual", "Auto"],
        index=1,  # Default to Manual
        help="""
        • Manual: You control all statistical choices
        • Auto: Uses parametric tests with intelligent corrections based on your data
        """,
        horizontal=True
    )
    
    # Fixed significance threshold
    alpha = 0.05
    
    if mode_choice == "Auto":
        test_type = "auto"
        correction_method = "auto"
        posthoc_correction = "auto"
        
    else:
        # Manual mode - show full controls
        # Create two columns for options
        col1, col2 = st.columns(2)
        
        with col1:
            # Test type selection
            test_type = st.selectbox(
                "Statistical Test Type",
                options=["parametric", "non_parametric"],
                index=0,  # Default to parametric
                help="""
                • Parametric: Welch's t-test/ANOVA (assumes normal distribution after transformation)
                • Non-parametric: Mann-Whitney U/Kruskal-Wallis (more conservative, no distribution assumptions)
                """
            )
            
            correction_method = st.selectbox(
                "Between-Class Correction (Level 1)",
                options=["uncorrected", "fdr_bh", "bonferroni"],
                index=1,  # Default to uncorrected
                help="""
                • Uncorrected: No correction (good for single hypothesis)
                • FDR (Benjamini-Hochberg): Controls false discovery rate (for multiple classes)
                • Bonferroni: Conservative, controls family-wise error rate (very strict)
                """
            )
        
        with col2:
            posthoc_correction = st.selectbox(
                "Within-Class Correction (Level 2)",
                options=["uncorrected", "tukey", "bonferroni"],
                index=1,  # Default to Tukey's HSD
                help="""
                For 3+ conditions only:
                • Uncorrected: No pairwise correction
                • Tukey's HSD: Recommended for parametric tests; uses Bonferroni-corrected pairwise tests for non-parametric
                • Bonferroni: Bonferroni correction for all pairwise tests
                """
            )
    
    # Auto-transformation option (always available)
    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,  # Default to True
        help="""
        Automatically applies log10 transformation to all data. 
        Log10 transformation is standard practice in lipidomics as it often 
        normalizes skewed concentration data and is biologically interpretable.
        """
    )
    
    if mode_choice == "Manual":
        st.markdown("---")
        st.markdown("### 🔬 Current Settings Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- **Test Type**: {test_type.title()}")
            st.write(f"- **Level 1 Correction**: {correction_method.upper().replace('_', '-')}")
        with col2:
            st.write(f"- **Level 2 Correction**: {posthoc_correction.replace('_', ' ').title()}")
            st.write(f"- **Auto-transform**: {'Yes' if auto_transform else 'No'}")
    
    return {
        'test_type': test_type,
        'correction_method': correction_method,
        'posthoc_correction': posthoc_correction,
        'alpha': alpha,
        'auto_transform': auto_transform,
        'mode_choice': mode_choice
    }

def display_detailed_statistical_results(statistical_results, selected_conditions):
    """
    Display detailed statistical test results in an expandable section.
    """
    show_detailed_stats = st.checkbox("Show detailed statistical analysis", key="show_detailed_stats")
    if show_detailed_stats:
        st.write("### Detailed Statistical Test Results")
        
        # Create a table for statistical results
        results_data = []
        for lipid_class, results in statistical_results.items():
            if lipid_class.startswith('_'):  # Skip metadata
                continue
                
            p_value = results['p-value']
            adj_p_value = results.get('adjusted p-value', p_value)
            transformation = results.get('transformation', 'none')
            
            result_row = {
                'Lipid Class': lipid_class,
                'Test': results['test'],
                'Statistic': f"{results['statistic']:.3f}",
                'p-value': f"{p_value:.3f}",
                'Adjusted p-value': f"{adj_p_value:.3f}",
                'Transformation': transformation.title(),
                'Significant': '✓' if adj_p_value < 0.05 else '—'
            }
            results_data.append(result_row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
        
        # Show detailed post-hoc results for multi-group comparisons
        if len(selected_conditions) > 2:
            st.write("### Post-hoc Test Results")
            
            # Check if post-hoc correction was requested
            posthoc_correction = statistical_results.get('_parameters', {}).get('posthoc_correction', 'uncorrected')
            
            if posthoc_correction == "uncorrected":
                st.write("Post-hoc analysis was not performed (correction method: uncorrected).")
            else:
                # Count significant omnibus tests
                significant_omnibus_count = 0
                has_posthoc_results = False
                
                for lipid_class, results in statistical_results.items():
                    if lipid_class.startswith('_'):  # Skip metadata
                        continue
                        
                    # Check if this was a significant omnibus test
                    if (results.get('test', '') in ["One-way ANOVA", "Welch's ANOVA", "One-way ANOVA (fallback)", 
                                                   "One-way ANOVA (Welch's unavailable)", "Kruskal-Wallis"] and
                        results.get('p-value', 1) <= 0.05):
                        significant_omnibus_count += 1
                        
                        # Check if post-hoc results exist
                        if results.get('tukey_results'):
                            has_posthoc_results = True
                            st.write(f"**{lipid_class}** ({results['test']})")
                            
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
                                    'Significant': significance,
                                    'Method': tukey['method']
                                })
                            
                            if tukey_data:
                                tukey_df = pd.DataFrame(tukey_data)
                                st.dataframe(tukey_df, use_container_width=True)
                                st.markdown("---")
                
                # Display appropriate message based on results
                if significant_omnibus_count == 0:
                    st.write("No significant omnibus tests found for post-hoc analysis.")
                elif not has_posthoc_results:
                    st.write(f"Found {significant_omnibus_count} significant omnibus test(s), but post-hoc analysis failed to complete.")

def apply_auto_mode_logic(selected_classes, selected_conditions):
    """
    Auto mode logic for intelligent defaults.
    
    Args:
        selected_classes (list): List of selected lipid classes
        selected_conditions (list): List of selected conditions
        
    Returns:
        dict: Dictionary containing auto recommendations and rationales
    """
    # Auto Level 1 (Between-Class) Correction
    if len(selected_classes) == 1:
        auto_correction_method = "uncorrected"
        auto_rationale = "Single class  →  no between-class correction needed"
    elif len(selected_classes) <= 5:
        auto_correction_method = "fdr_bh"
        auto_rationale = "Few classes (≤5)  →  FDR balances discovery and control"
    else:
        auto_correction_method = "fdr_bh"
        auto_rationale = "Multiple classes  →  FDR recommended for exploration"
    
    # Auto Level 2 (Within-Class) Correction
    # Tukey's HSD is designed to control FWER for any number of pairwise comparisons
    # It's more powerful than Bonferroni because it accounts for the correlation structure
    if len(selected_conditions) <= 2:
        auto_posthoc_correction = "uncorrected"
        auto_posthoc_rationale = "≤2 conditions → no post-hoc needed"
    else:
        auto_posthoc_correction = "tukey"
        auto_posthoc_rationale = "3+ conditions → Tukey's HSD (controls FWER)"
    
    return {
        'correction_method': auto_correction_method,
        'correction_rationale': auto_rationale,
        'posthoc_correction': auto_posthoc_correction,
        'posthoc_rationale': auto_posthoc_rationale
    }

def display_abundance_bar_charts(experiment, continuation_df):
    """Display abundance bar charts with auto mode integration."""
    with st.expander("Class Concentration Bar Chart"):
        # Short description
        st.markdown("""
        Compare total lipid class concentrations across experimental conditions. 
        Each bar shows the sum of all species within a class, with error bars indicating variability between replicates.
        """)
        
        # Get valid conditions (more than one sample)
        valid_conditions = [
            cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) 
            if num_samples > 1
        ]
        
        if not valid_conditions:
            st.warning("No conditions with multiple samples found. Bar chart analysis requires at least two replicates per condition.")
            return None, None
        
        # --- Statistical Options Section ---
        st.markdown("---")
        st.markdown("#### ⚙️ Statistical Options")
        stat_options = display_statistical_options()
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_conditions_list = st.multiselect(
                'Conditions', 
                valid_conditions, 
                valid_conditions,
                key='conditions_select'
            )
        with col2:
            selected_classes_list = st.multiselect(
                'Lipid Classes',
                list(continuation_df['ClassKey'].value_counts().index), 
                list(continuation_df['ClassKey'].value_counts().index),
                key='classes_select'
            )
        
        # Apply auto mode logic if selected
        if stat_options['mode_choice'] == "Auto":
            auto_settings = apply_auto_mode_logic(selected_classes_list, selected_conditions_list)
            stat_options['test_type'] = "auto"
            stat_options['correction_method'] = auto_settings['correction_method']
            stat_options['posthoc_correction'] = auto_settings['posthoc_correction']
        
        linear_fig, log10_fig = None, None
        
        if selected_conditions_list and selected_classes_list:
            # Perform statistical tests
            with st.spinner("Performing statistical analysis..."):
                statistical_results = lp.AbundanceBarChart.perform_statistical_tests(
                    continuation_df, 
                    experiment, 
                    selected_conditions_list, 
                    selected_classes_list,
                    test_type=stat_options['test_type'],
                    correction_method=stat_options['correction_method'],
                    posthoc_correction=stat_options['posthoc_correction'],
                    alpha=stat_options['alpha'],
                    auto_transform=stat_options['auto_transform']
                )
            
            # --- Results Section ---
            st.markdown("---")
            st.markdown("#### 📊 Results")
            
            # Scale selection
            scale_choice = st.radio(
                "Select scale:",
                options=["Log10 Scale", "Linear Scale"],
                index=0,  # Default to Log10
                horizontal=True
            )
            
            # Generate chart based on selection
            mode = 'log10 scale' if scale_choice == "Log10 Scale" else 'linear scale'
            filename_suffix = "log10" if scale_choice == "Log10 Scale" else "linear"
            
            with st.spinner("Generating chart..."):
                fig, abundance_df = lp.AbundanceBarChart.create_abundance_bar_chart(
                    df=continuation_df,
                    full_samples_list=experiment.full_samples_list,
                    individual_samples_list=experiment.individual_samples_list,
                    conditions_list=experiment.conditions_list,
                    selected_conditions=selected_conditions_list,
                    selected_classes=selected_classes_list,
                    mode=mode,
                    anova_results=statistical_results
                )
            
            if fig is not None and abundance_df is not None and not abundance_df.empty:
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    svg_bytes = fig.to_image(format="svg")
                    svg_string = svg_bytes.decode('utf-8')
                    st.download_button(
                        label="Download SVG",
                        data=svg_string,
                        file_name=f"abundance_bar_chart_{filename_suffix}.svg",
                        mime="image/svg+xml"
                    )
                with col2:
                    csv_data = convert_df(abundance_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f'abundance_bar_chart_{filename_suffix}.csv',
                        mime='text/csv',
                        key='abundance_chart_download'
                    )
            
            # Store both for PDF report (generate the other one in background)
            other_mode = 'linear scale' if scale_choice == "Log10 Scale" else 'log10 scale'
            other_fig, _ = lp.AbundanceBarChart.create_abundance_bar_chart(
                df=continuation_df,
                full_samples_list=experiment.full_samples_list,
                individual_samples_list=experiment.individual_samples_list,
                conditions_list=experiment.conditions_list,
                selected_conditions=selected_conditions_list,
                selected_classes=selected_classes_list,
                mode=other_mode,
                anova_results=statistical_results
            )
            
            if scale_choice == "Log10 Scale":
                log10_fig, linear_fig = fig, other_fig
            else:
                linear_fig, log10_fig = fig, other_fig
            
            # --- Detailed Statistics Section (at the end) ---
            st.markdown("---")
            st.markdown("#### 🔍 Detailed Statistics")
            display_detailed_statistical_results(statistical_results, selected_conditions_list)
            
            # Note about excluded conditions
            removed_conditions = set(experiment.conditions_list) - set(valid_conditions)
            if removed_conditions:
                st.info(f"Note: The following conditions were excluded due to having only one sample: {', '.join(removed_conditions)}")
                
        else:
            st.warning("Please select at least one condition and one class to create the charts.")
        
        return linear_fig, log10_fig

def display_abundance_pie_charts(experiment, continuation_df):
    """Display pie charts showing the proportional distribution of lipid classes for each condition."""
    pie_charts = {}
    with st.expander("Class Concentration Pie Chart"):
        # Short description
        st.markdown("""
        Visualize the relative proportions of lipid classes within each condition.
        """)
        
        # Get all lipid classes
        full_samples_list = experiment.full_samples_list
        all_classes = lp.AbundancePieChart.get_all_classes(continuation_df, full_samples_list)
        
        # Get conditions with multiple samples
        conditions_with_samples = [
            (condition, samples) 
            for condition, samples in zip(experiment.conditions_list, experiment.individual_samples_list)
            if len(samples) > 1
        ]
        
        if not conditions_with_samples:
            st.warning("No conditions with multiple samples found.")
            return pie_charts
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        selected_classes_list = st.multiselect(
            'Lipid Classes', 
            all_classes, 
            all_classes,
            key='pie_chart_classes'
        )
        
        if not selected_classes_list:
            st.warning("Please select at least one lipid class to create the pie charts.")
            return pie_charts
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("#### 📊 Results")
        
        # Filter dataframe and generate color mapping
        filtered_df = lp.AbundancePieChart.filter_df_for_selected_classes(
            continuation_df, full_samples_list, selected_classes_list
        )
        color_mapping = lp.AbundancePieChart._generate_color_mapping(selected_classes_list)
        
        # Create chart for each condition
        for condition, samples in conditions_with_samples:
            st.markdown(f"##### {condition}")
            fig, df = lp.AbundancePieChart.create_pie_chart(
                filtered_df, full_samples_list, condition, samples, color_mapping
            )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                pie_charts[condition] = fig
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    plotly_svg_download_button(fig, f"abundance_pie_chart_{condition}.svg")
                with col2:
                    available_samples = [
                        sample for sample in samples 
                        if f"concentration[{sample}]" in filtered_df.columns
                    ]
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
                            mime='text/csv',
                            key=f'pie_chart_csv_{condition}'
                        )
    
    return pie_charts

def display_saturation_statistical_options():
    """
    Display UI components for statistical testing options with auto/manual mode selection.
    Specialized for saturation plot analysis.
    """
    
    # First choice: Auto vs Manual mode
    mode_choice = st.radio(
        "Select Analysis Mode:",
        options=["Manual", "Auto"],
        index=1,  # Default to Manual
        help="""
        • Manual: You control all statistical choices
        • Auto: Uses parametric tests with intelligent corrections based on your data
        """,
        horizontal=True
    )
    
    # Fixed significance threshold
    alpha = 0.05
    
    if mode_choice == "Auto":
        test_type = "auto"
        correction_method = "auto"
        posthoc_correction = "auto"
        
    else:
        # Manual mode - show full controls
        col1, col2 = st.columns(2)
        
        with col1:
            test_type = st.selectbox(
                "Statistical Test Type",
                options=["parametric", "non_parametric"],
                index=0,
                help="""
                • Parametric: Welch's t-test/ANOVA (assumes log-normal distribution after transformation)
                • Non-parametric: Mann-Whitney U/Kruskal-Wallis (more conservative, no distribution assumptions)
                """
            )
            
            correction_method = st.selectbox(
                "Between Class/FA Type Correction (Level 1)",
                options=["uncorrected", "fdr_bh", "bonferroni"],
                index=1,
                help="""
                Note: Tests 3 fatty acid types (SFA, MUFA, PUFA) per lipid class
                • Uncorrected: No correction (good for single class analysis)
                • FDR (Benjamini-Hochberg): Controls false discovery rate (for multiple classes)
                • Bonferroni: Conservative, controls family-wise error rate (very strict)
                """
            )
        
        with col2:
            posthoc_correction = st.selectbox(
                "Within Class/FA Type Correction (Level 2)",
                options=["uncorrected", "tukey", "bonferroni"],
                index=1,  # Default to Tukey's HSD
                help="""
                For 3+ conditions only:
                • Uncorrected: No pairwise correction
                • Tukey's HSD: Recommended for parametric tests; uses Bonferroni-corrected pairwise tests for non-parametric
                • Bonferroni: Bonferroni correction for all pairwise tests
                """
            )
    
    # Auto-transformation option (always available)
    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,
        help="""
        Automatically applies log10 transformation to all data. 
        Log10 transformation is standard practice in lipidomics as it often 
        normalizes skewed concentration data and is biologically interpretable.
        """
    )
    
    # Show current settings for manual mode only
    if mode_choice == "Manual":
        st.markdown("---")
        st.markdown("### 🔬 Current Settings Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- **Test Type**: {test_type.title()}")
            st.write(f"- **Level 1 Correction**: {correction_method.upper().replace('_', '-')}")
        with col2:
            st.write(f"- **Level 2 Correction**: {posthoc_correction.replace('_', ' ').title()}")
            st.write(f"- **Auto-transform**: {'Yes' if auto_transform else 'No'}")
    
    return {
        'test_type': test_type,
        'correction_method': correction_method,
        'posthoc_correction': posthoc_correction,
        'alpha': alpha,
        'auto_transform': auto_transform,
        'mode_choice': mode_choice
    }

def display_saturation_detailed_statistical_results(statistical_results, selected_conditions):
    """Display detailed statistical test results in an expandable section."""
    show_detailed_stats = st.checkbox("Show detailed statistical analysis", key="show_detailed_saturation_stats")
    if show_detailed_stats:
        st.write("### Detailed Statistical Test Results")
        
        # Create a table for statistical results
        results_data = []
        for lipid_class, class_results in statistical_results.items():
            if lipid_class.startswith('_'):  # Skip metadata
                continue
                
            for fa_type, results in class_results.items():
                p_value = results['p-value']
                adj_p_value = results.get('adjusted p-value', p_value)
                transformation = results.get('transformation', 'none')
                
                result_row = {
                    'Lipid Class': lipid_class,
                    'Fatty Acid Type': fa_type,
                    'Test': results['test'],
                    'Statistic': f"{results['statistic']:.3f}",
                    'p-value': f"{p_value:.3f}",
                    'Adjusted p-value': f"{adj_p_value:.3f}" if not np.isnan(adj_p_value) else "NaN",
                    'Transformation': transformation.title(),
                    'Significant': '✓' if adj_p_value < 0.05 else '—'
                }
                results_data.append(result_row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
        
        # Show detailed post-hoc results for multi-group comparisons
        if len(selected_conditions) > 2:
            st.write("### Post-hoc Test Results")
            
            posthoc_correction = statistical_results.get('_parameters', {}).get('posthoc_correction', 'uncorrected')
            
            if posthoc_correction == "uncorrected":
                st.write("Post-hoc analysis was not performed (correction method: uncorrected).")
            else:
                # Count significant omnibus tests
                significant_omnibus_count = 0
                has_posthoc_results = False
                
                for lipid_class, class_results in statistical_results.items():
                    if lipid_class.startswith('_'):  # Skip metadata
                        continue
                        
                    for fa_type, results in class_results.items():
                        # Check if this was a significant omnibus test using adjusted p-value
                        adj_p_value = results.get('adjusted p-value', results.get('p-value', 1))
                        if (results.get('test', '') in ["One-way ANOVA", "Welch's ANOVA", "One-way ANOVA (fallback)", 
                                                       "One-way ANOVA (Welch's unavailable)", "Kruskal-Wallis"] and
                            adj_p_value <= 0.05):
                            significant_omnibus_count += 1
                            
                            # Check if post-hoc results exist
                            if results.get('tukey_results'):
                                has_posthoc_results = True
                                st.write(f"**{lipid_class} - {fa_type}** ({results['test']})")
                                
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
                                        'Significant': significance,
                                        'Method': tukey['method']
                                    })
                                
                                if tukey_data:
                                    tukey_df = pd.DataFrame(tukey_data)
                                    st.dataframe(tukey_df, use_container_width=True)
                                    st.markdown("---")
                
                # Display appropriate message based on results
                if significant_omnibus_count == 0:
                    st.write("No significant omnibus tests found for post-hoc analysis.")
                elif not has_posthoc_results:
                    st.write(f"Found {significant_omnibus_count} significant omnibus test(s), but post-hoc analysis failed to complete.")

def display_saturation_compatibility_warning(continuation_df):
    """Display compatibility warning for datasets without detailed fatty acid composition."""
    # Check if we have any detailed FA compositions
    has_detailed_fa = any('_' in str(lipid) for lipid in continuation_df['LipidMolec'])
    
    if not has_detailed_fa:
        st.warning("""
        ⚠️  **Data Format Issue**: Saturation plots require detailed fatty acid composition 
        (e.g., PC(16:0_18:1)) to accurately classify chains as saturated, monounsaturated, 
        or polyunsaturated. Your data appears to use consolidated total composition 
        (e.g., PC(34:1)) which only shows total carbons and double bonds across all chains. 
        This prevents accurate saturation analysis.
        
        **What this means:**
        - `PC(16:0_18:1)` ✓ Shows individual chains: 16:0 (saturated) and 18:1 (monounsaturated)
        - `PC(34:1)` — Only shows totals: 34 carbons, 1 double bond (could be distributed many ways)
        
        The analysis may produce inaccurate results because the algorithm cannot determine 
        which chains are saturated vs. unsaturated.
        """)

def display_excluded_conditions_warning(experiment, selected_conditions):
    """Display warning about conditions excluded due to insufficient samples."""
    # Get valid conditions (more than one sample)
    valid_conditions = [
        cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) 
        if num_samples > 1
    ]
    
    # Check if any conditions were removed due to having only one sample
    removed_conditions = set(selected_conditions) - set(valid_conditions)
    if removed_conditions:
        st.info(f"Note: The following conditions were excluded due to having only one sample: {', '.join(removed_conditions)}")

def display_saturation_plots(experiment, continuation_df):
    """Enhanced saturation plot display with rigorous statistical methodology."""
    saturation_plots = {}
    with st.expander("Class Level Breakdown - Saturation Plots"):
        # Short description
        st.markdown("""
        Analyze fatty acid composition within lipid classes: SFA (saturated), MUFA (monounsaturated), and PUFA (polyunsaturated).
        """)
        
        # Compatibility warning (if no detailed FA)
        display_saturation_compatibility_warning(continuation_df)
        
        # Get valid conditions (more than one sample)
        valid_conditions = [
            cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) 
            if num_samples > 1
        ]
        
        if not valid_conditions:
            st.warning("No conditions with multiple samples found. Saturation analysis requires at least two replicates per condition.")
            return saturation_plots
        
        # --- Statistical Options Section ---
        st.markdown("---")
        st.markdown("#### ⚙️ Statistical Options")
        stat_options = display_saturation_statistical_options()
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_conditions_list = st.multiselect(
                'Conditions', 
                valid_conditions, 
                valid_conditions,
                key='saturation_conditions_select'
            )
        with col2:
            selected_classes_list = st.multiselect(
                'Lipid Classes', 
                list(continuation_df['ClassKey'].value_counts().index), 
                list(continuation_df['ClassKey'].value_counts().index),
                key='saturation_classes_select'
            )
        
        if not selected_conditions_list or not selected_classes_list:
            st.warning("Please select at least one condition and one class to generate saturation plots.")
            return saturation_plots
        
        # --- Consolidated Format Handling ---
        filtered_continuation_df = continuation_df.copy()
        consolidated_lipids_dict = lp.SaturationPlot.identify_consolidated_lipids(continuation_df, selected_classes_list)
        
        if consolidated_lipids_dict:
            st.markdown("---")
            st.markdown("#### ⚠️ Consolidated Format Lipids")
            st.markdown("""
**No perfect solution exists for consolidated format lipids:**
- **Include:** The lipid's abundance is counted, but SFA/MUFA/PUFA classification is based only on total double bonds (inaccurate)
- **Exclude:** Classification is accurate for remaining lipids, but you lose abundance from excluded species

Review the table and list below. Some may be legitimate single-chain lipids (accurate as-is). 
For multi-chain lipids, decide based on your analysis goals.
            """)
            
            # Build summary table
            summary_data = []
            for lipid_class in selected_classes_list:
                total_in_class = len(continuation_df[continuation_df['ClassKey'] == lipid_class])
                consolidated_in_class = len(consolidated_lipids_dict.get(lipid_class, []))
                if total_in_class > 0:
                    pct = (consolidated_in_class / total_in_class) * 100
                    summary_data.append({
                        'Class': lipid_class,
                        'Total Lipids': total_in_class,
                        'Consolidated': consolidated_in_class,
                        '% Consolidated': f"{pct:.1f}%"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                # Only show classes with consolidated lipids or sort to show affected ones first
                summary_df = summary_df.sort_values('Consolidated', ascending=False)
                st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)
            
            st.caption(
                "Note: True single-chain classes (LPC, LPE, LPA, LPG, LPI, LPS, MAG, FFA, CE, ChE) "
                "are excluded from this check — their format is always accurate."
            )
            
            # Flatten all consolidated lipids for multiselect
            all_consolidated = []
            for lipid_class, lipids in consolidated_lipids_dict.items():
                all_consolidated.extend([(lipid, lipid_class) for lipid in lipids])
            
            display_options = [f"{lipid} ({lipid_class})" for lipid, lipid_class in all_consolidated]
            lipid_to_display = {f"{lipid} ({lipid_class})": lipid for lipid, lipid_class in all_consolidated}
            
            lipids_to_exclude_display = st.multiselect(
                f'Select lipids to exclude ({len(all_consolidated)} detected):',
                display_options,
                default=[],
                help="Exclude consolidated format lipids with multiple fatty acid chains."
            )
            
            lipids_to_exclude = [lipid_to_display[display] for display in lipids_to_exclude_display]
            
            if lipids_to_exclude:
                filtered_continuation_df = continuation_df[~continuation_df['LipidMolec'].isin(lipids_to_exclude)].copy()
                st.success(f"✓ Excluded {len(lipids_to_exclude)} lipids from analysis")
        
        # Apply auto mode logic if selected
        if stat_options['mode_choice'] == "Auto":
            auto_settings = lp.SaturationPlot.apply_auto_mode_logic(selected_classes_list, selected_conditions_list)
            stat_options['test_type'] = "auto"
            stat_options['correction_method'] = auto_settings['correction_method']
            stat_options['posthoc_correction'] = auto_settings['posthoc_correction']
        
        # Perform statistical tests
        with st.spinner("Performing statistical analysis..."):
            statistical_results = lp.SaturationPlot.perform_statistical_tests(
                filtered_continuation_df,
                experiment, 
                selected_conditions_list, 
                selected_classes_list,
                test_type=stat_options['test_type'],
                correction_method=stat_options['correction_method'],
                posthoc_correction=stat_options['posthoc_correction'],
                alpha=stat_options['alpha'],
                auto_transform=stat_options['auto_transform']
            )
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("#### 📊 Results")
        
        # Plot options
        col1, col2 = st.columns(2)
        with col1:
            plot_type = st.radio(
                "Plot type:",
                options=["Concentration", "Percentage", "Both"],
                index=0,
                horizontal=True
            )
        with col2:
            show_significance = st.checkbox(
                "Show significance asterisks",
                value=False,
                help="Display *, **, *** on plots"
            )
        
        # Generate plots
        with st.spinner("Generating saturation plots..."):
            plots = lp.SaturationPlot.create_plots(
                filtered_continuation_df,
                experiment, 
                selected_conditions_list, 
                statistical_results,
                show_significance=show_significance
            )
            
            if plots:
                for lipid_class, (main_plot, percentage_plot, plot_data) in plots.items():
                    st.subheader(f"{lipid_class}")
                    
                    # Show concentration plot
                    if plot_type in ["Concentration", "Both"]:
                        st.plotly_chart(main_plot, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            svg_bytes = main_plot.to_image(format="svg")
                            st.download_button(
                                label="Download SVG",
                                data=svg_bytes.decode('utf-8'),
                                file_name=f"saturation_concentration_{lipid_class}.svg",
                                mime="image/svg+xml",
                                key=f"sat_conc_svg_{lipid_class}"
                            )
                        with col2:
                            st.download_button(
                                label="Download CSV",
                                data=plot_data.to_csv(index=False).encode('utf-8'),
                                file_name=f"saturation_concentration_{lipid_class}.csv",
                                mime="text/csv",
                                key=f"sat_conc_csv_{lipid_class}"
                            )
                    
                    # Show percentage plot
                    if plot_type in ["Percentage", "Both"]:
                        st.plotly_chart(percentage_plot, use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            svg_bytes = percentage_plot.to_image(format="svg")
                            st.download_button(
                                label="Download SVG",
                                data=svg_bytes.decode('utf-8'),
                                file_name=f"saturation_percentage_{lipid_class}.svg",
                                mime="image/svg+xml",
                                key=f"sat_pct_svg_{lipid_class}"
                            )
                        with col2:
                            percentage_data = lp.SaturationPlot._calculate_percentage_df(plot_data)
                            st.download_button(
                                label="Download CSV",
                                data=percentage_data.to_csv(index=False).encode('utf-8'),
                                file_name=f"saturation_percentage_{lipid_class}.csv",
                                mime="text/csv",
                                key=f"sat_pct_csv_{lipid_class}"
                            )
                    
                    saturation_plots[lipid_class] = {'main': main_plot, 'percentage': percentage_plot}
            else:
                st.warning("No plots could be generated. Check that selected classes have sufficient data.")
        
        # --- Detailed Statistics Section ---
        st.markdown("---")
        st.markdown("#### 🔍 Detailed Statistics")
        display_saturation_detailed_statistical_results(statistical_results, selected_conditions_list)
        
        # Note about excluded conditions
        display_excluded_conditions_warning(experiment, selected_conditions_list)
    
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
        # Short description
        st.markdown("""
        Compare lipid class abundance and saturation between two conditions. 
        Circle size represents fold change; circle color represents saturation ratio.
        """)
        
        # Calculation methodology
        st.markdown("**Fold Change** (determines circle size):")
        st.code("Fold Change = Mean_concentration_experimental / Mean_concentration_control", language=None)
        st.markdown("Values >1 indicate increase in experimental condition; <1 indicate decrease.")
        
        st.markdown("**Saturation Ratio** (determines circle color, range 0–1):")
        st.code("Saturation Ratio = Total_saturated_chains / Total_chains (summed across all species in class)", language=None)
        st.markdown("""
        Each lipid species contributes its chain count (not weighted by concentration). 
        Chains with 0 double bonds (e.g., 16:0) are saturated; chains with ≥1 double bonds (e.g., 18:1) are unsaturated.
        """)
        
        # Compatibility warning (only if no detailed FA)
        has_detailed_fa = any('_' in str(lipid) for lipid in continuation_df['LipidMolec'])
        if not has_detailed_fa:
            st.warning("""
            ⚠️  Note: Saturation ratios require detailed fatty acid composition (e.g., PC(16:0_18:1)).
            Your data uses total composition format, which affects saturation accuracy. Fold changes remain accurate.
            """)
        
        # Get valid conditions (more than one replicate)
        valid_conditions = [
            cond for cond, num_samples in zip(experiment.conditions_list, experiment.number_of_samples_list) 
            if num_samples > 1
        ]
        
        if len(valid_conditions) < 2:
            st.warning("At least two conditions with multiple replicates are required for pathway visualization.")
            return None
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            control_condition = st.selectbox('Control Condition', valid_conditions, index=0, key='pathway_control')
        with col2:
            experimental_options = [c for c in valid_conditions if c != control_condition]
            experimental_condition = st.selectbox('Experimental Condition', experimental_options, index=0, key='pathway_experimental')
        
        if not control_condition or not experimental_condition:
            st.info("Select both control and experimental conditions to generate the visualization.")
            return None
        
        # --- Consolidated Format Handling ---
        filtered_df = continuation_df.copy()
        all_classes = list(continuation_df['ClassKey'].unique())
        consolidated_lipids_dict = lp.SaturationPlot.identify_consolidated_lipids(continuation_df, all_classes)
        
        if consolidated_lipids_dict:
            st.markdown("---")
            st.markdown("#### ⚠️ Consolidated Format Lipids")
            st.markdown("""
            Consolidated format lipids (e.g., PC(34:1)) cannot be accurately classified for saturation ratio 
            because individual chain composition is unknown. This affects circle **colors**, not sizes.
            """)
            
            # Build summary table
            summary_data = []
            for lipid_class in all_classes:
                total_in_class = len(continuation_df[continuation_df['ClassKey'] == lipid_class])
                consolidated_in_class = len(consolidated_lipids_dict.get(lipid_class, []))
                if consolidated_in_class > 0:
                    pct = (consolidated_in_class / total_in_class) * 100
                    summary_data.append({
                        'Class': lipid_class,
                        'Total Lipids': total_in_class,
                        'Consolidated': consolidated_in_class,
                        '% Consolidated': f"{pct:.1f}%"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Consolidated', ascending=False)
                st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)
            
            st.caption(
                "Note: Single-chain classes (LPC, LPE, LPA, LPG, LPI, LPS, MAG, FFA, CE, ChE) "
                "are excluded — their format is always accurate."
            )
            
            # Flatten all consolidated lipids for multiselect
            all_consolidated = []
            for lipid_class, lipids in consolidated_lipids_dict.items():
                all_consolidated.extend([(lipid, lipid_class) for lipid in lipids])
            
            display_options = [f"{lipid} ({lipid_class})" for lipid, lipid_class in all_consolidated]
            lipid_to_display = {f"{lipid} ({lipid_class})": lipid for lipid, lipid_class in all_consolidated}
            
            lipids_to_exclude_display = st.multiselect(
                f'Select lipids to exclude from saturation calculation ({len(all_consolidated)} detected):',
                display_options,
                default=[],
                help="Excluding consolidated lipids improves saturation ratio accuracy but removes their abundance contribution.",
                key='pathway_exclude_lipids'
            )
            
            lipids_to_exclude = [lipid_to_display[display] for display in lipids_to_exclude_display]
            
            if lipids_to_exclude:
                filtered_df = continuation_df[~continuation_df['LipidMolec'].isin(lipids_to_exclude)].copy()
                st.success(f"✓ Excluded {len(lipids_to_exclude)} lipids from saturation calculation")
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("#### 📊 Results")
        
        with st.spinner("Generating pathway visualization..."):
            # Calculate saturation ratios (using filtered df for accuracy)
            class_saturation_ratio_df = lp.PathwayViz.calculate_class_saturation_ratio(filtered_df)
            
            # Calculate fold changes (using original df for complete abundance)
            class_fold_change_df = lp.PathwayViz.calculate_class_fold_change(
                continuation_df, experiment, control_condition, experimental_condition
            )
            
            # Generate the visualization
            fig, pathway_dict = lp.PathwayViz.create_pathway_viz(
                class_fold_change_df, class_saturation_ratio_df, control_condition, experimental_condition
            )
            
            if fig is not None and pathway_dict:
                st.pyplot(fig)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    matplotlib_svg_download_button(fig, f"pathway_visualization_{control_condition}_vs_{experimental_condition}.svg")
                with col2:
                    # Prepare CSV data
                    pathway_df = pd.DataFrame({
                        'Lipid Class': pathway_dict['class'],
                        'Fold Change': pathway_dict['abundance ratio'],
                        'Saturation Ratio': pathway_dict['saturated fatty acids ratio']
                    })
                    pathway_df['Absolute Change'] = abs(pathway_df['Fold Change'] - 1)
                    pathway_df = pathway_df.sort_values('Absolute Change', ascending=False)
                    pathway_df = pathway_df.drop('Absolute Change', axis=1)
                    pathway_df['Fold Change'] = pathway_df['Fold Change'].apply(lambda x: f"{x:.2f}")
                    pathway_df['Saturation Ratio'] = pathway_df['Saturation Ratio'].apply(lambda x: f"{x:.2f}")
                    
                    csv_download = convert_df(pathway_df)
                    st.download_button(
                        label="Download CSV",
                        data=csv_download,
                        file_name='pathway_visualization_data.csv',
                        mime='text/csv',
                        key='pathway_csv_download'
                    )
                
                # Data summary
                st.markdown(f"**Data Summary:** Comparing {experimental_condition} to {control_condition}")
                st.dataframe(pathway_df, use_container_width=True)
                
                return fig
            else:
                st.warning("Unable to generate pathway visualization due to insufficient data.")
                st.markdown("""
                This could be due to:
                - Missing lipid classes needed for the visualization
                - Insufficient data for calculating fold changes
                
                Try using a dataset with more comprehensive lipid coverage.
                """)
        
        return None
                
def display_volcano_statistical_options():
    """
    Display UI components for volcano plot statistical testing options.
    """
    st.subheader("Statistical Testing Options")
    
    # Fixed significance threshold
    alpha = 0.05
    
    # Manual mode controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Test type selection
        test_type = st.selectbox(
            "Statistical Test Type",
            options=["parametric", "non_parametric"],
            index=0,  # Default to parametric
            help="""
            • Parametric: Welch's t-test (assumes log-normal distribution after transformation)
            • Non-parametric: Mann-Whitney U (more conservative, no distribution assumptions)
            """
        )
        
    with col2:
        correction_method = st.selectbox(
            "Multiple Testing Correction",
            options=["uncorrected", "fdr_bh", "bonferroni"],
            index=1,  # Default to uncorrected
            help="""
            • Uncorrected: No correction (good for targeted hypothesis testing)
            • FDR (Benjamini-Hochberg): Controls false discovery rate (recommended for exploratory analysis)
            • Bonferroni: Conservative, controls family-wise error rate (very strict)
            """
        )

    # Auto-transformation option
    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,  # Default to True
        help="""
        Automatically applies log10 transformation to all data. 
        Log10 transformation is standard practice in lipidomics as it often 
        normalizes skewed concentration data and is biologically interpretable.
        """
    )
    
    # Show current settings
    st.markdown("---")
    st.markdown("### 🔬 Current Settings Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"- **Test Type**: {test_type.title()}")
        st.write(f"- **Correction**: {correction_method.upper().replace('_', '-')}")
    with col2:
        st.write(f"- **Significance**: α = {alpha}")
        st.write(f"- **Auto-transform**: {'Yes' if auto_transform else 'No'}")
    
    return {
        'test_type': test_type,
        'correction_method': correction_method,
        'alpha': alpha,
        'auto_transform': auto_transform
    }

def apply_volcano_auto_mode_logic(n_lipids_tested):
    """
    Auto mode logic for volcano plot analysis.
    
    Args:
        n_lipids_tested (int): Number of lipid species being tested
        
    Returns:
        dict: Dictionary containing auto recommendations and rationales
    """
    # Auto correction method based on number of tests
    if n_lipids_tested <= 10:
        auto_correction_method = "uncorrected"
        auto_rationale = "Few lipids (≤10)  →  no correction needed"
    elif n_lipids_tested <= 50:
        auto_correction_method = "fdr_bh"
        auto_rationale = "Moderate number of lipids  →  FDR balances discovery and control"
    else:
        auto_correction_method = "fdr_bh"  # Still use FDR for large numbers
        auto_rationale = "Many lipids  →  FDR recommended for exploration"
    
    return {
        'correction_method': auto_correction_method,
        'correction_rationale': auto_rationale
    }

def display_volcano_detailed_statistical_results(statistical_results, control_condition, experimental_condition):
    """
    Display detailed statistical test results for volcano plot in an expandable section.
    """
    show_detailed_stats = st.checkbox("Show detailed statistical analysis", key="show_detailed_volcano_stats")
    if show_detailed_stats:
        st.write("### Detailed Statistical Test Results")
        
        # Extract parameters
        params = statistical_results.get('_parameters', {})
        n_tests = params.get('n_tests_performed', 0)
        
        st.write(f"**Comparison**: {experimental_condition} vs {control_condition}")
        st.write(f"**Tests performed**: {n_tests} individual lipid species")
        st.write(f"**Test method**: {params.get('test_type', 'Unknown').title()}")
        st.write(f"**Correction method**: {params.get('correction_method', 'Unknown').replace('_', '-').upper()}")
        st.write(f"**Transformation**: {'Log10' if params.get('auto_transform', False) else 'None'}")
        
        # Create a table for statistical results
        results_data = []
        for lipid_name, results in statistical_results.items():
            if lipid_name.startswith('_'):  # Skip metadata
                continue
                
            p_value = results['p-value']
            adj_p_value = results.get('adjusted_p_value', p_value)
            fold_change = results['fold_change']
            log2_fc = results['log2_fold_change']
            
            result_row = {
                'Lipid': lipid_name,
                'Class': results['class_key'],
                'Log2 FC': f"{log2_fc:.3f}",
                'Fold Change': f"{fold_change:.3f}",
                'p-value': f"{p_value:.2e}",
                'Adj. p-value': f"{adj_p_value:.2e}",
                'Test': results['test'],
                'Significant': '✓' if results['significant'] else '—'
            }
            results_data.append(result_row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            
            # Sort by absolute log2 fold change to show most changed lipids first
            results_df['abs_log2_fc'] = results_df['Log2 FC'].astype(float).abs()
            results_df = results_df.sort_values('abs_log2_fc', ascending=False)
            results_df = results_df.drop('abs_log2_fc', axis=1)
            
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download option
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f'volcano_statistical_results_{experimental_condition}_vs_{control_condition}.csv',
                mime='text/csv'
            )
            
            # Summary statistics
            st.write("### Summary Statistics")
            n_significant = sum(1 for r in statistical_results.values() 
                              if isinstance(r, dict) and r.get('significant', False))
            n_upregulated = sum(1 for r in statistical_results.values() 
                               if isinstance(r, dict) and r.get('significant', False) and r.get('log2_fold_change', 0) > 0)
            n_downregulated = sum(1 for r in statistical_results.values() 
                                 if isinstance(r, dict) and r.get('significant', False) and r.get('log2_fold_change', 0) < 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tested", n_tests)
            with col2:
                st.metric("Significant", n_significant)
            with col3:
                st.metric("Upregulated", n_upregulated)
            with col4:
                st.metric("Downregulated", n_downregulated)

def display_volcano_plot(experiment, continuation_df):
    """
    Volcano plot display function with rigorous statistical methodology integration,
    supporting both top N significant lipid labeling and user-selected permanent labels.
    """
    volcano_plots = {}
    with st.expander("Species Level Breakdown - Volcano Plot"):
        # Show volcano plot analysis details
        st.markdown("### Volcano Plot Analysis")
        st.markdown("""
        The volcano plot identifies significantly altered lipid species between two experimental conditions.
        See the **"About Statistical Testing"** expander above for details on test selection and corrections.

        **🎯 What the Plot Shows:**

        - **X-axis (Log2 Fold Change)**: Magnitude and direction of change
          - Each unit = 2-fold change (e.g., +1 = 2× increase, -1 = 2× decrease)
          - Calculated from original concentration values for biological interpretability

        - **Y-axis**: -log10(p-value) — higher = more significant
        - **Red Dashed Lines**: Configurable significance and fold-change thresholds

        **⚠️ Non-Parametric Test Note:**

        Mann-Whitney U tests with small sample sizes (n=3-6) often produce identical p-values for many lipids,
        creating horizontal lines in the plot. This is expected behavior. Parametric tests (Welch's t-test) 
        with log transformation provide better resolution for typical lipidomic data.
        """)
        st.markdown("---")

        # Get statistical options using the specialized UI function
        stat_options = display_volcano_statistical_options()
        st.markdown("---")

        # Condition selection
        conditions_with_replicates = [
            condition for index, condition in enumerate(experiment.conditions_list)
            if experiment.number_of_samples_list[index] > 1
        ]

        if len(conditions_with_replicates) <= 1:
            st.error('You need at least two conditions with more than one replicate to create a volcano plot.')
            return volcano_plots

        control_condition = st.selectbox('Select Control Condition', conditions_with_replicates)
        available_experimental = [cond for cond in conditions_with_replicates if cond != control_condition]

        if not available_experimental:
            st.error("Need at least two different conditions with multiple replicates.")
            return volcano_plots

        default_experimental = available_experimental[0] if available_experimental else conditions_with_replicates[0]
        experimental_condition = st.selectbox('Select Experimental Condition', available_experimental,
                                             index=0 if available_experimental else 0)

        # Class selection
        selected_classes_list = st.multiselect(
            'Select lipid classes to include:',
            list(continuation_df['ClassKey'].value_counts().index),
            list(continuation_df['ClassKey'].value_counts().index),
            key='volcano_classes_select'
        )

        # Threshold settings
        col1, col2 = st.columns(2)
        with col1:
            p_value_threshold = st.number_input(
                'Significance threshold (p-value)',
                min_value=0.001, max_value=0.1, value=0.05, step=0.001,
                help="P-value threshold for statistical significance"
            )
        with col2:
            fold_change_threshold = st.number_input(
                'Biological significance threshold (fold change)',
                min_value=1.1, max_value=5.0, value=2.0, step=0.1,
                help="Minimum fold change considered biologically significant"
            )

        q_value_threshold = -np.log10(p_value_threshold)
        log2_fc_threshold = np.log2(fold_change_threshold)

        # Display options
        hide_non_significant = st.checkbox(
            'Hide non-significant data points',
            value=False,
            help="Hide points that don't meet both statistical and biological significance thresholds"
        )

        # Initialize session state for custom labels and positions
        if 'custom_labeled_lipids' not in st.session_state:
            st.session_state.custom_labeled_lipids = []
        if 'custom_label_positions' not in st.session_state:
            st.session_state.custom_label_positions = {}

        # Define use_adjusted_p based on correction_method
        use_adjusted_p = stat_options['correction_method'] != "uncorrected"

        # Generate enhanced volcano plot
        if selected_classes_list and control_condition != experimental_condition:
            with st.spinner("Performing enhanced statistical analysis..."):
                try:
                    # Get samples for conditions
                    control_samples, experimental_samples = lp.VolcanoPlot._get_samples_for_conditions(
                        experiment, control_condition, experimental_condition
                    )
                    df_processed, control_cols, experimental_cols = lp.VolcanoPlot._prepare_data(
                        continuation_df, control_samples, experimental_samples
                    )
                    df_processed = df_processed[df_processed['ClassKey'].isin(selected_classes_list)]
                    statistical_results = lp.VolcanoPlot.perform_statistical_tests(
                        df_processed, control_cols, experimental_cols,
                        stat_options['test_type'], stat_options['correction_method'],
                        stat_options['alpha'], stat_options['auto_transform']
                    )
                    volcano_df, removed_lipids_df = lp.VolcanoPlot._format_results_enhanced(
                        df_processed, statistical_results, control_cols, experimental_cols
                    )

                    # Display detailed statistical results
                    display_volcano_detailed_statistical_results(
                        statistical_results, control_condition, experimental_condition
                    )

                    st.markdown("---")
                    st.subheader("Volcano Plot")

                    # Restore top N labeling option
                    top_n_labels = st.number_input(
                        'Number of top lipids to label (ranked by p-value):',
                        min_value=0, max_value=50, step=1,
                        value=0,
                        key='top_n_labels',
                        help="Enter the number of most significant lipids to label on the plot automatically."
                    )

                    # Add multiselect for user to choose additional lipids to label
                    available_lipids = volcano_df['LipidMolec'].tolist()
                    st.session_state.custom_labeled_lipids = st.multiselect(
                        'Select additional lipids to label on the plot (in addition to top N):',
                        available_lipids,
                        default=st.session_state.custom_labeled_lipids,
                        key='custom_lipid_labels'
                    )

                    # Combine top N and user-selected lipids
                    p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
                    top_n_lipids = volcano_df.sort_values(p_col, ascending=False).head(top_n_labels)['LipidMolec'].tolist()
                    all_lipids_to_label = list(set(top_n_lipids + st.session_state.custom_labeled_lipids))

                    # Label adjustment checkbox
                    adjust_labels = st.checkbox("Enable label adjustment", value=False)

                    # If adjust_labels, show adjustment inputs for all labeled lipids
                    if adjust_labels and all_lipids_to_label:
                        with st.container():
                            st.markdown("### Label Position Adjustments")
                            for lipid in all_lipids_to_label:
                                key = f"pos_{lipid}"
                                if key not in st.session_state.custom_label_positions:
                                    st.session_state.custom_label_positions[key] = (0.0, 0.0)

                                st.markdown(f"**{lipid}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    x_offset = st.number_input(
                                        "X offset",
                                        value=st.session_state.custom_label_positions[key][0],
                                        step=0.1,
                                        key=f"x_offset_{lipid}"
                                    )
                                with col2:
                                    y_offset = st.number_input(
                                        "Y offset",
                                        value=st.session_state.custom_label_positions[key][1],
                                        step=0.1,
                                        key=f"y_offset_{lipid}"
                                    )

                                st.session_state.custom_label_positions[key] = (x_offset, y_offset)

                    # Create color mapping
                    color_mapping = lp.VolcanoPlot._generate_color_mapping(volcano_df)

                    # Create plot with all selected labels
                    plot = lp.VolcanoPlot._create_plot_enhanced(
                        volcano_df, color_mapping, q_value_threshold, hide_non_significant,
                        use_adjusted_p, log2_fc_threshold, top_n_labels=0  # Set to 0 to handle labeling manually
                    )

                    # Add annotations for both top N and user-selected lipids
                    if all_lipids_to_label:
                        plot.layout.annotations = []  # Clear existing annotations
                        char_width = 0.05
                        label_height = 0.3
                        buffer = 0.2
                        placed_boxes = []
                        candidates = [
                            (0.1, 0.3, 'left'), (-0.1, 0.3, 'right'),
                            (0.1, -0.3, 'left'), (-0.1, -0.3, 'right'),
                            (0.3, 0.1, 'left'), (0.3, -0.1, 'left'),
                            (-0.3, 0.1, 'right'), (-0.3, -0.1, 'right')
                        ]

                        for lipid in all_lipids_to_label:
                            row = volcano_df[volcano_df['LipidMolec'] == lipid].iloc[0]
                            color = color_mapping[row['ClassKey']]
                            point_x = row['FoldChange']
                            point_y = row[p_col]
                            key = f"pos_{lipid}"
                            custom_x, custom_y = st.session_state.custom_label_positions.get(key, (0.0, 0.0))
                            w = len(lipid) * char_width
                            placed = False

                            for cand_i, (dx, dy, align) in enumerate(candidates):
                                scale = 1 + (cand_i // len(candidates)) * 0.5
                                label_x = point_x + dx * scale + custom_x
                                label_y = point_y + dy * scale + custom_y

                                if align == 'left':
                                    left = label_x
                                    right = label_x + w
                                else:
                                    left = label_x - w
                                    right = label_x

                                bottom = label_y - label_height / 2
                                top = label_y + label_height / 2

                                overlap = False
                                for box in placed_boxes:
                                    if not (right < box['left'] - buffer or left > box['right'] + buffer or
                                            top < box['bottom'] - buffer or bottom > box['top'] + buffer):
                                        overlap = True
                                        break

                                if not overlap:
                                    plot.add_annotation(
                                        x=label_x,
                                        y=label_y,
                                        text=lipid,
                                        showarrow=False,
                                        font=dict(color=color, size=12),
                                        align=align
                                    )
                                    plot.add_annotation(
                                        x=point_x,
                                        y=point_y,
                                        ax=label_x,
                                        ay=label_y,
                                        axref='x',
                                        ayref='y',
                                        text='',
                                        showarrow=True,
                                        arrowhead=1,
                                        arrowsize=1,
                                        arrowwidth=1,
                                        arrowcolor='black'
                                    )
                                    placed_boxes.append({
                                        'left': left - buffer,
                                        'right': right + buffer,
                                        'bottom': bottom - buffer,
                                        'top': top + buffer
                                    })
                                    placed = True
                                    break

                            if not placed:
                                align_fallback = 'left' if point_x > 0 else 'right'
                                label_x_fallback = point_x + (0.1 if align_fallback == 'left' else -0.1) + custom_x
                                label_y_fallback = point_y + 0.2 + custom_y
                                plot.add_annotation(
                                    x=label_x_fallback,
                                    y=label_y_fallback,
                                    text=lipid,
                                    showarrow=False,
                                    font=dict(color=color, size=12),
                                    align=align_fallback
                                )
                                plot.add_annotation(
                                    x=point_x,
                                    y=point_y,
                                    ax=label_x_fallback,
                                    ay=label_y_fallback,
                                    axref='x',
                                    ayref='y',
                                    text='',
                                    showarrow=True,
                                    arrowhead=1,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor='black'
                                )

                    # Display main volcano plot
                    st.plotly_chart(plot, use_container_width=True)
                    volcano_plots['main'] = plot

                    # Add download options for main plot
                    col1, col2 = st.columns(2)
                    with col1:
                        svg_bytes = plot.to_image(format="svg")
                        svg_string = svg_bytes.decode('utf-8')
                        st.download_button(
                            label="Download SVG",
                            data=svg_string,
                            file_name=f"volcano_plot_{experimental_condition}_vs_{control_condition}.svg",
                            mime="image/svg+xml"
                        )
                    with col2:
                        png_bytes = plot.to_image(format="png")
                        st.download_button(
                            label="Download PNG",
                            data=png_bytes,
                            file_name=f"volcano_plot_{experimental_condition}_vs_{control_condition}.png",
                            mime="image/png"
                        )

                    st.markdown("---")

                    # Generate and display the concentration vs. fold change plot
                    st.subheader("Concentration vs. Fold Change Plot")
                    st.markdown("""
                    This plot shows the relationship between a lipid's abundance and its fold change,
                    using the same statistical framework as the main volcano plot.
                    """)
                    concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(
                        volcano_df, color_mapping, q_value_threshold, hide_non_significant, use_adjusted_p=True
                    )
                    st.plotly_chart(concentration_vs_fold_change_plot, use_container_width=True)
                    volcano_plots['concentration_vs_fold_change'] = concentration_vs_fold_change_plot

                    # Add download options for concentration plot
                    col1, col2 = st.columns(2)
                    with col1:
                        svg_bytes = concentration_vs_fold_change_plot.to_image(format="svg")
                        svg_string = svg_bytes.decode('utf-8')
                        st.download_button(
                            label="Download SVG",
                            data=svg_string,
                            file_name=f"concentration_vs_fold_change_{experimental_condition}_vs_{control_condition}.svg",
                            mime="image/svg+xml"
                        )
                    with col2:
                        png_bytes = concentration_vs_fold_change_plot.to_image(format="png")
                        st.download_button(
                            label="Download PNG",
                            data=png_bytes,
                            file_name=f"concentration_vs_fold_change_{experimental_condition}_vs_{control_condition}.png",
                            mime="image/png"
                        )

                    st.markdown("---")

                    # Lipid concentration distribution
                    st.subheader("Individual Lipid Analysis")
                    st.markdown("""
                    Select specific lipids to examine their concentration distributions in detail.
                    """)
                    all_classes = list(volcano_df['ClassKey'].unique())
                    selected_class = st.selectbox('Select Lipid Class for Detailed Analysis:', all_classes)

                    if selected_class:
                        class_lipids = volcano_df[volcano_df['ClassKey'] == selected_class]['LipidMolec'].unique().tolist()
                        # Default to most significantly changed lipids
                        sorted_lipids = volcano_df[volcano_df['ClassKey'] == selected_class].sort_values(
                            '-log10(adjusted_pValue)', ascending=False
                        )['LipidMolec'].tolist()
                        selected_lipids = st.multiselect(
                            'Select Lipids for Detailed View:',
                            class_lipids,
                            default=sorted_lipids[:3] if len(sorted_lipids) >= 3 else sorted_lipids
                        )

                        if selected_lipids:
                            selected_conditions = [control_condition, experimental_condition]
                            with st.spinner("Generating concentration distribution plot..."):
                                plot_df = lp.VolcanoPlot.create_concentration_distribution_data(
                                    continuation_df, selected_lipids, selected_conditions, experiment
                                )
                                fig = lp.VolcanoPlot.create_concentration_distribution_plot(
                                    plot_df, selected_lipids, selected_conditions
                                )
                                st.pyplot(fig)
                                volcano_plots['concentration_distribution'] = fig

                                # Add download options
                                col1, col2 = st.columns(2)
                                with col1:
                                    buf = io.BytesIO()
                                    fig.savefig(buf, format='svg', bbox_inches='tight')
                                    buf.seek(0)
                                    svg_string = buf.getvalue().decode('utf-8')
                                    st.download_button(
                                        label="Download SVG",
                                        data=svg_string,
                                        file_name=f"concentration_distribution_{experimental_condition}_vs_{control_condition}.svg",
                                        mime="image/svg+xml"
                                    )
                                with col2:
                                    csv_data = plot_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv_data,
                                        file_name=f"concentration_distribution_data_{experimental_condition}_vs_{control_condition}.csv",
                                        mime="text/csv"
                                    )

                    st.markdown("---")

                    # Display excluded lipids information
                    st.subheader("Excluded Lipids")
                    st.markdown("""
                    The following lipids were excluded from the statistical analysis. This provides transparency
                    about which lipids couldn't be analyzed and why.
                    """)
                    if not removed_lipids_df.empty:
                        st.dataframe(removed_lipids_df)
                        csv_excluded = removed_lipids_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download CSV",
                            data=csv_excluded,
                            file_name=f"excluded_lipids_{experimental_condition}_vs_{control_condition}.csv",
                            mime="text/csv"
                        )

                        # Show summary of exclusion reasons
                        exclusion_summary = removed_lipids_df['Reason'].value_counts()
                        st.write("**Exclusion Reasons Summary:**")
                        for reason, count in exclusion_summary.items():
                            st.write(f"- {reason}: {count} lipids")
                    else:
                        st.success("→ No lipids were excluded from the analysis!")

                except Exception as e:
                    st.error(f"Error generating volcano plot: {str(e)}")
                    import traceback
                    st.error(f"Debug info: {traceback.format_exc()}")

        else:
            if not selected_classes_list:
                st.warning("Please select at least one lipid class to generate the volcano plot.")
            if control_condition == experimental_condition:
                st.warning("Please select different conditions for control and experimental groups.")

    return volcano_plots

def display_lipidomic_heatmap(experiment, continuation_df):
    """
    Display a user interface for creating and interacting with lipidomic heatmaps.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomic data
        
    Returns:
        tuple: A tuple containing the regular heatmap figure and clustered heatmap figure
    """
    regular_heatmap = None
    clustered_heatmap = None
    
    with st.expander("Species Level Breakdown - Lipidomic Heatmap"):
        # Heatmap analysis details - ALWAYS VISIBLE
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
            
        else:
            st.warning("Please select at least one condition and one lipid class to generate the heatmap.")
    
    return regular_heatmap, clustered_heatmap

def display_fach_heatmaps(experiment, continuation_df):
    """
    Display Fatty Acid Composition Heatmaps (FACH) for a selected lipid class and conditions.
    Returns a dictionary of generated plots for PDF inclusion.
    
    Args:
        experiment: Experiment object containing experimental setup information
        continuation_df: DataFrame containing the lipidomic data
    
    Returns:
        dict: Dictionary with lipid class as key and Plotly figure as value
    """
    fach_plots = {}
    with st.expander("Class Level Breakdown - Fatty Acid Composition Heatmaps"):
        # Short description
        st.markdown("""
        Visualize lipid distribution by carbon chain length (y-axis) and double bonds (x-axis). 
        Color intensity shows proportional abundance within each class. Species with identical totals are aggregated.
        """)
        
        # Display compatibility warning if no detailed FA composition
        has_detailed_fa = any('_' in str(lipid) for lipid in continuation_df['LipidMolec'])
        if not has_detailed_fa:
            st.warning("""
            ⚠️  Note: FACH works best with detailed fatty acid composition (e.g., PC(16:0_18:1)).
            Your data appears to use total composition (e.g., PC(34:1)), which may affect accuracy.
            """)
        
        # Get valid conditions
        valid_conditions = [c for i, c in enumerate(experiment.conditions_list) if experiment.number_of_samples_list[i] > 0]
        
        if not valid_conditions:
            st.warning("No conditions with samples found.")
            return fach_plots
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            all_classes = sorted(continuation_df['ClassKey'].unique())
            selected_class = st.selectbox('Lipid Class', all_classes, key='fach_class_select')
        with col2:
            selected_conditions = st.multiselect(
                'Conditions', 
                valid_conditions, 
                default=valid_conditions[:2], 
                key='fach_conditions_select'
            )
        
        if not selected_class or not selected_conditions:
            st.info("Select a lipid class and at least one condition to generate the heatmap.")
            return fach_plots
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("#### 📊 Results")
        
        with st.spinner("Generating Fatty Acid Composition Heatmap..."):
            data_dict = lp.FACH.prepare_fach_data(continuation_df, experiment, selected_class, selected_conditions)
            
            if data_dict:
                fig = lp.FACH.create_fach_heatmap(data_dict)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    fach_plots[selected_class] = fig
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        plotly_svg_download_button(fig, f"fach_{selected_class}.svg")
                    with col2:
                        # Combined CSV download
                        combined_df = pd.concat([df.assign(Condition=cond) for cond, df in data_dict.items()])
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"fach_data_{selected_class}.csv",
                            mime="text/csv",
                            key=f"fach_download_{selected_class}"
                        )
            else:
                st.warning("No data available for the selected class and conditions. Ensure the lipid class contains parseable lipid names (e.g., PC(34:1) or PC(16:0_18:1)).")
    
    return fach_plots

def generate_pdf_report(box_plot_fig1, box_plot_fig2, bqc_plot, retention_time_plot, pca_plot,
                       heatmap_figs, correlation_plots, abundance_bar_charts, abundance_pie_charts,
                       saturation_plots, volcano_plots, pathway_visualization, fach_plots=None):
    """
    Generate a PDF report containing all generated plots.
    
    Args:
        box_plot_fig1: Plotly figure for missing values distribution
        box_plot_fig2: Plotly figure for box plot of non-zero concentrations
        bqc_plot: Plotly figure for BQC quality check
        retention_time_plot: Plotly figure for retention time plot
        pca_plot: Plotly figure for PCA analysis
        heatmap_figs: Dictionary of heatmap figures (regular and clustered)
        correlation_plots: Dictionary of correlation plot figures
        abundance_bar_charts: Dictionary of abundance bar chart figures
        abundance_pie_charts: Dictionary of abundance pie chart figures
        saturation_plots: Dictionary of saturation plot figures
        volcano_plots: Dictionary of volcano plot figures
        pathway_visualization: Matplotlib figure for pathway visualization
        fach_plots: Dictionary of FACH figures (default None)
    
    Returns:
        BytesIO: Buffer containing the generated PDF
    """
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
        
        # Add FACH Plots
        if fach_plots:
            for lipid_class, fach_plot in fach_plots.items():
                if fach_plot is not None:
                    pdf.showPage()
                    pdf.setPageSize(landscape(letter))
                    fach_bytes = pio.to_image(fach_plot, format='png', width=1000, height=600, scale=2)
                    fach_img = ImageReader(io.BytesIO(fach_bytes))
                    pdf.drawImage(fach_img, 50, 100, width=700, height=400, preserveAspectRatio=True)
                    pdf.drawString(50, 80, f"Fatty Acid Composition Heatmap for {lipid_class}")
        
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