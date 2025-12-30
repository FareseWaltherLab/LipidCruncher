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

def load_module_image(filename, caption=None):
    """Helper function to load and display a module PDF as an image."""
    try:
        pdf_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(pdf_path):
            images = convert_from_path(pdf_path, dpi=300)
            if images:
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='PNG')
                st.image(img_byte_arr.getvalue(), caption=caption, use_column_width=True)
                return True
    except Exception as e:
        st.warning(f"Could not load {filename}: {str(e)}")
    return False

def display_landing_page():
    """Display the LipidCruncher landing page."""
    # Logo
    try:
        logo_path = os.path.join(IMAGES_DIR, 'new_logo.tif')
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=720)
        else:
            st.error(f"Logo file not found at {logo_path}")
            st.header("LipidCruncher")
    except Exception as e:
        st.error(f"Failed to load logo: {str(e)}")
        st.header("LipidCruncher")

    # Tagline
    st.markdown("""
    *An open-source platform for processing, visualizing, and analyzing lipidomic data.*
    
    Built by [The Farese & Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther) 
    to bridge the gap between lipidomic data generation and biological insight—no bioinformatics expertise required.
    """)

    # Quick highlights
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**📂 4 Formats**<br>LipidSearch, MS-DIAL, Generic, Metabolomics Workbench", unsafe_allow_html=True)
    col2.markdown("**🔬 QC + Normalization**<br>Integrated quality control with flexible normalization", unsafe_allow_html=True)
    col3.markdown("**📊 Lipid-Specific Viz**<br>Saturation profiles, pathway maps, lipidomic heatmap", unsafe_allow_html=True)
    col4.markdown("**📈 Publication-Ready**<br>Interactive plots, SVG export, PDF reports", unsafe_allow_html=True)

    st.markdown("---")

    # The Three Modules
    st.subheader("🚀 How It Works")
    st.markdown("LipidCruncher guides you through three intuitive modules:")

    # Module 1
    st.markdown("#### Module 1: Standardize & Normalize")
    st.markdown("""
    **Get your data analysis-ready in minutes.** Import data from LipidSearch, MS-DIAL, Metabolomics Workbench, or a generic CSV format. 
    Define your experiment by assigning samples to conditions, then apply automatic column standardization, 
    filtering (duplicates, empty rows, zero values), and flexible normalization (internal standards, protein concentration, or both). 
    Internal standards consistency plots help verify sample preparation and instrument performance.
    """)
    load_module_image('module1.pdf')

    st.markdown("---")

    # Module 2
    st.markdown("#### Module 2: Quality Check")
    st.markdown("""
    **Trust your data before you analyze it.** Box plots assess data quality and validate normalization—replicates 
    within a condition should exhibit similar medians and interquartile ranges. CoV analysis of batch quality control (BQC) 
    samples evaluates measurement precision. Correlation heatmaps and PCA detect outliers and visualize sample clustering.
    """)
    load_module_image('module2.pdf')

    st.markdown("---")

    # Module 3
    st.markdown("#### Module 3: Visualize & Analyze")
    st.markdown("""
    **Turn complex lipid profiles into biological insights.** Bar & pie charts, volcano plots, saturation profiles (SFA, MUFA, PUFA), 
    metabolic pathway mapping, clustered heatmaps, and fatty acid composition analysis—all interactive with SVG/CSV export.
    """)
    load_module_image('module3.pdf')

    st.markdown("---")

    # Call to Action
    st.subheader("🎯 Ready to Crunch?")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Crunching", use_container_width=True):
            st.session_state.page = 'app'
            st.experimental_rerun()

    st.markdown("---")

    # Resources in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📚 Resources")
        st.markdown("""
        - 📄 [Read our paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1)
        - 💻 [Source code on GitHub](https://github.com/FareseWaltherLab/LipidCruncher)
        - 📊 [Sample datasets](https://github.com/FareseWaltherLab/LipidCruncher/tree/main/sample_datasets)
        """)

    with col2:
        st.markdown("#### 🧪 Try Our Test Data")
        st.markdown("""
        No dataset? No problem! Our sample data includes:
        - **WT** (Wild Type): 4 replicates
        - **ADGAT-DKO** (Double Knockout): 4 replicates  
        - **BQC** (Batch Quality Control): 4 replicates
        """)

    st.markdown("---")

    # Footer
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 💡 Pro Tip")
        st.markdown("Starting a new analysis? Refresh the page first to ensure a clean session.")
    with col2:
        st.markdown("#### 📧 Support")
        st.markdown("Questions, bugs, or feature requests? Email **abdih@mskcc.org**")

def main():
    """Main function for the Lipidomics Analysis Module Streamlit application."""
    initialize_session_state()

    if st.session_state.page == 'landing':
        display_landing_page()
    elif st.session_state.page == 'app':
        try:
            logo_path = os.path.join(IMAGES_DIR, 'new_logo.tif')
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
                            
                            # FIRST: Show documentation expander
                            display_data_processing_docs(data_format)
                            
                            # Quality filtering configuration (format-specific)
                            grade_config = None
                            quality_config = None
                            
                            if data_format == 'LipidSearch 5.0':
                                # Show grade filtering UI for LipidSearch
                                with st.expander("⚙️ Configure Grade Filtering", expanded=False):
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
                                    st.info("ℹ️ Quality filtering unavailable — Total score column was not selected in column mapping.")
                                else:
                                    # Show quality filtering UI in expander below data type selection
                                    with st.expander("⚙️ Configure Quality Filtering", expanded=False):
                                        quality_config = get_msdial_quality_config()
                                        st.session_state.msdial_quality_config = quality_config
                                        
                                        # Display filter messages from previous run (if any)
                                        if 'msdial_filter_messages' in st.session_state and st.session_state.msdial_filter_messages:
                                            st.markdown("---")
                                            st.markdown("**Filter Results:**")
                                            for msg in st.session_state.msdial_filter_messages:
                                                st.info(msg)
                            
                            # Pass config to clean_data
                            cleaned_df, intsta_df = clean_data(
                                df_to_clean, name_df, experiment, data_format, 
                                grade_config=grade_config,
                                quality_config=quality_config
                            )
                            
                            # Display MS-DIAL filter results (if any) right after cleaning
                            if data_format == 'MS-DIAL' and 'msdial_filter_messages' in st.session_state:
                                for msg in st.session_state.msdial_filter_messages:
                                    st.info(msg)
                                # Clear after displaying so they don't persist incorrectly
                                del st.session_state.msdial_filter_messages
                            
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
                        st.info("Please confirm your inputs in the sidebar to proceed with data filtering and analysis.")
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
    """Display format-specific requirements in a collapsible, easy-to-read format."""
    
    if data_format == 'Metabolomics Workbench':
        with st.expander("📋 Data Format Requirements", expanded=False):
            st.markdown("""
### 🔬 Metabolomics Workbench Format

**Required Structure:**

| Component | Description |
|-----------|-------------|
| `MS_METABOLITE_DATA_START` | Section start marker |
| Row 1 | Sample names |
| Row 2 | Condition labels (one per sample) |
| Row 3+ | Lipid data (name in first column) |
| `MS_METABOLITE_DATA_END` | Section end marker |

**Example:**
```
MS_METABOLITE_DATA_START
Samples,Sample1,Sample2,Sample3,Sample4
Factors,WT,WT,KO,KO
LPC(16:0),234.5,256.7,189.3,201.4
PE(18:0_20:4),456.7,478.2,390.1,405.6
MS_METABOLITE_DATA_END
```

**✨ Auto-processing:** Lipid names standardized, intensity columns created, conditions extracted.
            """)
    
    elif data_format == 'LipidSearch 5.0':
        with st.expander("📋 Data Format Requirements", expanded=False):
            st.markdown("""
### 🔬 LipidSearch 5.0 Format

**Required Columns:**

| Column | Description |
|--------|-------------|
| `LipidMolec` | Lipid molecule identifier |
| `ClassKey` | Lipid class (e.g., PC, PE, TG) |
| `CalcMass` | Calculated mass |
| `BaseRt` | Retention time |
| `TotalGrade` | Quality grade (A/B/C/D) |
| `TotalSmpIDRate(%)` | Sample identification rate |
| `FAKey` | Fatty acid key |
| `MeanArea[s1]`, `MeanArea[s2]`, ... | Intensity per sample |

**Example column structure:**
```
LipidMolec | ClassKey | CalcMass | BaseRt | TotalGrade | ... | MeanArea[s1] | MeanArea[s2] | ...
```

**💡 Tip:** Export directly from LipidSearch — column names should match automatically.
            """)
    
    elif data_format == 'MS-DIAL':
        with st.expander("📋 Data Format Requirements", expanded=False):
            st.markdown("""
### 🔬 MS-DIAL Format

**How to Export:** File → Export → Alignment Result → CSV

**Required Columns:**

| Column | Description |
|--------|-------------|
| `Metabolite name` | Lipid identifiers (exact name required) |
| Sample columns | Intensity values — **must be LAST columns** |

**Optional Columns** (enable extra features):

| Column | Feature Enabled |
|--------|-----------------|
| `Total score` | Quality filtering (0-100) |
| `MS/MS matched` | MS/MS validation filter |
| `Average Rt(min)` | Retention time plots |
| `Average Mz` | Retention time plots |

---

**📁 File Structure Options:**

**Option A — Raw data only:**
```
[metadata cols...] [sample1] [sample2] ... [sampleN]
```

**Option B — Raw + Pre-normalized:**
```
[metadata cols...] [raw1]...[rawN] [Lipid IS] [norm1]...[normN]
```
The `Lipid IS` column separates raw from normalized data. You'll choose which to use after upload.

---

**✨ Auto-features:**
- ClassKey inferred from lipid names
- Hydroxyl notation preserved (`;2O`, `;3O`)
- Internal standards detected: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH`
- Column mappings reviewable after upload
            """)
    
    else:  # Generic Format
        with st.expander("📋 Data Format Requirements", expanded=False):
            st.markdown("""
### 🔬 Generic Format

**Simple structure — just two things:**

| Position | Content |
|----------|---------|
| Column 1 | Lipid names |
| Columns 2+ | Sample intensities (one column per sample) |

**⚠️ Important:** No extra columns allowed! Remove any metadata columns before upload.

---

**Lipid Name Standardization Examples:**

| Your Format | → Standardized |
|-------------|----------------|
| `LPC O-18:1` | `LPC(O-18:1)` |
| `Cer d18:1;2O/24:0` | `Cer(d18:1;2O_24:0)` |
| `SM 18:1;2O/16:0` | `SM(18:1;2O_16:0)` |
| `PA 16:0/18:1` | `PA(16:0_18:1)` |

---

**✨ Auto-features:**
- Lipid names standardized to `Class(chains)` format
- Hydroxyl notation preserved (`;2O`, `;3O`)
- ClassKey extracted from lipid names
- Intensity columns renamed to `intensity[s1]`, `intensity[s2]`, ...
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
    
    # Store count for display elsewhere (don't show message here)
    if intsta_df is not None and not intsta_df.empty:
        st.session_state.extracted_standards_count = intsta_df['LipidMolec'].nunique()
    else:
        st.session_state.extracted_standards_count = 0
    
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
    
    # Get unique classes from the data
    all_classes = sorted(df['ClassKey'].unique())
    
    # Option to use default or custom settings
    use_custom = st.radio(
        "Grade filtering mode:",
        ["Use Default Settings", "Customize by Class"],
        index=0,
        horizontal=True,
        key="grade_filter_mode"
    )
    
    if use_custom == "Use Default Settings":
        st.success("✓ Default: A/B for all classes, plus C for LPC and SM.")
        return None
    
    # Custom settings
    st.markdown("---")
    
    grade_config = {}
    
    # Create columns for compact layout
    cols = st.columns(3)
    
    for idx, lipid_class in enumerate(all_classes):
        with cols[idx % 3]:
            # Default grades based on class
            if lipid_class in ['LPC', 'SM']:
                default_grades = ['A', 'B', 'C']
            else:
                default_grades = ['A', 'B']
            
            selected_grades = st.multiselect(
                f"**{lipid_class}**",
                options=['A', 'B', 'C', 'D'],
                default=default_grades,
                key=f"grade_select_{lipid_class}"
            )
            
            if not selected_grades:
                st.error("⚠️ Will be excluded!")
            
            grade_config[lipid_class] = selected_grades
    
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
        st.markdown("##### 📊 Data Type Selection")
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
    
    if not quality_filtering_available and not msms_filtering_available:
        st.warning("Quality filtering unavailable — no 'Total score' or 'MS/MS matched' columns found.")
        return None
    
    # Case 1: Total score column is available
    if quality_filtering_available:
        quality_options = {
            'Strict (Score ≥80, MS/MS required)': {'total_score_threshold': 80, 'require_msms': True},
            'Moderate (Score ≥60)': {'total_score_threshold': 60, 'require_msms': False},
            'Permissive (Score ≥40)': {'total_score_threshold': 40, 'require_msms': False},
            'No filtering': {'total_score_threshold': 0, 'require_msms': False}
        }
        
        selected_option = st.radio(
            "Quality filtering level:",
            list(quality_options.keys()),
            index=1,  # Default to Moderate
            horizontal=True,
            key="msdial_quality_level"
        )
        
        quality_config = quality_options[selected_option].copy()
        
        # MS/MS validation override (if available)
        if msms_filtering_available:
            col1, col2 = st.columns(2)
            with col1:
                custom_msms = st.checkbox(
                    "Require MS/MS validation",
                    value=quality_config['require_msms'],
                    key="msdial_custom_msms"
                )
                quality_config['require_msms'] = custom_msms
        
        # Advanced: custom score threshold (using checkbox since we may be inside an expander)
        show_custom = st.checkbox("Customize score threshold", value=False, key="msdial_show_custom_threshold")
        if show_custom:
            custom_score = st.slider(
                "Minimum Total Score:",
                min_value=0,
                max_value=100,
                value=quality_config['total_score_threshold'],
                step=5,
                key="msdial_custom_score"
            )
            quality_config['total_score_threshold'] = custom_score
        
        # Summary
        st.success(f"✓ Score ≥{quality_config['total_score_threshold']}, MS/MS {'required' if quality_config['require_msms'] else 'optional'}")
        
        return quality_config
    
    # Case 2: Only MS/MS matched column available
    elif msms_filtering_available:
        st.info("Score filtering unavailable (no 'Total score' column). MS/MS filtering only.")
        
        require_msms = st.checkbox(
            "Require MS/MS validation",
            value=False,
            key="msdial_msms_only"
        )
        
        return {
            'total_score_threshold': 0,
            'require_msms': require_msms
        }
    
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
    
    st.markdown("##### 🏷️ Standards Source")
    
    # Radio button for standards source
    standards_source = st.radio(
        "Standards source:",
        ["Automatic Detection", "Upload Custom Standards"],
        index=0 if st.session_state.standard_source_preference == "Automatic Detection" else 1,
        horizontal=True,
        key="standards_source_radio",
        label_visibility="collapsed"
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
            display_data(st.session_state.intsta_df, "Detected Standards", "detected_standards.csv", "auto")
        else:
            st.warning("No internal standards automatically detected in dataset.")
    
    # Handle custom upload
    else:
        # Clear intsta_df immediately if no custom standards uploaded yet
        if 'preserved_intsta_df' not in st.session_state or st.session_state.preserved_intsta_df is None:
            st.session_state.intsta_df = pd.DataFrame()
        
        st.markdown("**Are standards present in your main dataset?**")
        standards_in_dataset = st.radio(
            "Standards location:",
            options=[
                "Yes — Extract from dataset",
                "No — Uploading complete standards data"
            ],
            key="standards_mode_selection",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        use_legacy_mode = standards_in_dataset.startswith("Yes")
        
        # Mode-specific format guidance
        if use_legacy_mode:
            st.markdown("**CSV format:** Single column with lipid names (must exist in your dataset)")
        else:
            st.markdown("**CSV format:** 1st column = lipid names, remaining columns = intensity values per sample")
        
        uploaded_file = st.file_uploader(
            "Upload standards CSV", 
            type=['csv'],
            key="standards_file_uploader"
        )
        
        # If file uploader was cleared, reset standards
        if uploaded_file is None and 'preserved_intsta_df' in st.session_state and st.session_state.preserved_intsta_df is not None:
            st.session_state.intsta_df = pd.DataFrame()
            st.session_state.preserved_intsta_df = None
            st.warning("Standards cleared.")
        
        if uploaded_file is not None:
            try:
                standards_df = pd.read_csv(uploaded_file)
                new_standards_df = normalizer.process_standards_file(
                    standards_df,
                    st.session_state.cleaned_df,
                    st.session_state.intsta_df,
                    standards_in_dataset=use_legacy_mode
                )
                
                if new_standards_df is not None:
                    st.session_state.intsta_df = new_standards_df
                    st.session_state.preserved_intsta_df = new_standards_df.copy()
                    
                    st.success(f"✓ Loaded {len(new_standards_df)} custom standards")
                    display_data(new_standards_df, "Custom Standards", "custom_standards.csv", "uploaded")
                
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

def apply_zero_filter(cleaned_df, experiment, data_format, bqc_label=None, 
                      bqc_threshold=0.5, non_bqc_threshold=0.75):
    """
    Applies the zero-value filter to the cleaned dataframe.
    
    Args:
        cleaned_df (pd.DataFrame): Cleaned dataframe
        experiment (Experiment): Experiment object
        data_format (str): Data format
        bqc_label (str, optional): Label for Batch Quality Control samples
        bqc_threshold (float): Proportion threshold for BQC condition (default 0.5 = 50%)
        non_bqc_threshold (float): Proportion threshold for non-BQC conditions (default 0.75 = 75%)
        
    Returns:
        tuple: (filtered DataFrame, list of removed species)
    """
    default_detection = 30000.0 if data_format == 'LipidSearch 5.0' else 0.0
    detection_threshold = st.number_input(
        'Detection threshold (values ≤ this are considered zero)',
        min_value=0.0,
        value=default_detection,
        step=1.0,
        help="For non-LipidSearch formats, 0 means only exact zeros. Increase if your data has a noise floor.",
        key="zero_filter_detection_threshold"
    )
    
    # Get all lipid species before filtering
    all_species = cleaned_df['LipidMolec'].tolist()
    
    # List to keep track of rows to keep
    to_keep = []
    
    has_bqc = bqc_label is not None and bqc_label in experiment.conditions_list
    
    for idx, row in cleaned_df.iterrows():
        non_bqc_all_fail = True
        bqc_fail = True if has_bqc else False
        
        for cond_idx, cond_samples in enumerate(experiment.individual_samples_list):
            if not cond_samples:
                continue
            
            zero_count = 0
            n_samples = len(cond_samples)
            
            for sample in cond_samples:
                col = f'intensity[{sample}]'
                if col in cleaned_df.columns:
                    value = row[col]
                    if value <= detection_threshold:
                        zero_count += 1
            
            # Use the appropriate threshold
            is_bqc_condition = experiment.conditions_list[cond_idx] == bqc_label
            threshold = bqc_threshold if is_bqc_condition else non_bqc_threshold
            
            zero_proportion = zero_count / n_samples if n_samples > 0 else 1.0
            
            if zero_proportion < threshold:
                if is_bqc_condition:
                    bqc_fail = False
                else:
                    non_bqc_all_fail = False
        
        if not bqc_fail and not non_bqc_all_fail:
            to_keep.append(idx)
    
    filtered_df = cleaned_df.loc[to_keep].reset_index(drop=True)
    removed_species = [s for s in all_species if s not in filtered_df['LipidMolec'].tolist()]
    
    return filtered_df, removed_species

def display_data_processing_docs(data_format):
    """Display documentation about data standardization and filtering for the current format."""
    
    with st.expander("📖 About Data Standardization and Filtering", expanded=False):
        
        cleaning_docs = {
            'LipidSearch 5.0': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | Extract LipidMolec, ClassKey, CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey, MeanArea columns |
| 2. Data Type Conversion | Convert MeanArea to numeric (non-numeric → 0) |
| 3. Lipid Name Standardization | Standardize to `Class(chains)` format |
| 4. Grade Filtering | Filter by quality grade (**configurable below**) |
| 5. Best Peak Selection | Keep entry with highest TotalSmpIDRate(%) per lipid |
| 6. Missing FA Keys | Remove rows without FAKey (except Ch class, deuterated standards) |
| 7. Duplicate Removal | Remove duplicates by LipidMolec |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### ⚙️ Grade Filtering (Configurable)

LipidSearch assigns quality grades to each identification:

| Grade | Confidence | Default Action |
|-------|------------|----------------|
| A | Highest | ✓ Keep |
| B | Good | ✓ Keep |
| C | Lower | ✓ Keep for LPC/SM only |
| D | Lowest | ✗ Remove |

**→ Configure in "Configure Grade Filtering" section below.**
            """,
            
            'MS-DIAL': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Header Detection | Auto-detect data start row (skip metadata rows) |
| 2. Column Mapping | `Metabolite name` → LipidMolec, `Average Rt(min)` → BaseRt, `Average Mz` → CalcMass |
| 3. ClassKey Inference | Extract class from lipid name (e.g., `Cer(18:1;2O_24:0)` → `Cer`) |
| 4. Lipid Name Standardization | Standardize format, preserve hydroxyl notation (`;2O`, `;3O`) |
| 5. Quality Filtering | Filter by Total Score and/or MS/MS validation (**configurable below**) |
| 6. Data Type Selection | Choose raw or pre-normalized (if both available) |
| 7. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 8. Smart Deduplication | Keep entry with highest Total Score per lipid |
| 9. Internal Standards | Auto-detect: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH` patterns |
| 10. Duplicate Removal | Remove remaining duplicates by LipidMolec |
| 11. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### ⚙️ Quality Filtering (Configurable)

MS-DIAL provides quality metrics for filtering:

| Preset | Total Score | MS/MS Required | Use Case |
|--------|-------------|----------------|----------|
| Strict | ≥80 | Yes | Publication-ready |
| Moderate | ≥60 | No | Exploratory analysis |
| Permissive | ≥40 | No | Discovery |

**→ Configure in "Configure Quality Filtering" section below.**
            """,
            
            'Metabolomics Workbench': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Section Extraction | Extract data between `MS_METABOLITE_DATA_START` and `MS_METABOLITE_DATA_END` |
| 2. Header Processing | Row 1 → sample names, Row 2 → conditions |
| 3. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 4. Lipid Name Standardization | Standardize to `Class(chains)` format |
| 5. ClassKey Extraction | Extract class from lipid name |
| 6. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 7. Conditions Storage | Store conditions for experiment setup suggestions |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
            """,
            
            'Generic Format': """
### 🔬 Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 2. Lipid Name Standardization | Standardize to `Class(chains)` format, preserve hydroxyl notation |
| 3. ClassKey Extraction | Extract class from lipid name (e.g., `PC(16:0_18:1)` → `PC`) |
| 4. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 5. Invalid Lipid Removal | Remove empty names, single special characters |
| 6. Duplicate Removal | Remove duplicates by LipidMolec |
| 7. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
            """
        }
        
        # Display documentation for current format
        st.markdown(cleaning_docs.get(data_format, cleaning_docs['Generic Format']))
        
        # Zero filtering explanation (applies to all formats)
        st.markdown("---")
        st.markdown("""
#### 🔧 Zero Filtering (Configurable)

Removes lipid species with too many zero/below-detection values:

| Condition Type | Default Threshold | Action |
|----------------|-------------------|--------|
| BQC (if present) | ≥50% zeros | Remove species |
| All non-BQC conditions | ≥75% zeros each | Remove species |

*Thresholds are adjustable in "Configure Zero Filtering" section below.*
        """)
        
        # Show internal standards extraction result
        standards_count = st.session_state.get('extracted_standards_count', 0)
        if standards_count > 0:
            st.markdown("---")
            st.success(f"✓ **Internal Standards Extracted:** {standards_count} internal standard(s) separated for normalization (not filtered out)")

def display_cleaned_data(unfiltered_df, intsta_df):
    """
    Display cleaned data and manage internal standards with simplified workflow.
    Includes zero filter application and display of removed species.
    """
    # Update session state with new data if provided
    if unfiltered_df is not None:
        st.session_state.cleaned_df = unfiltered_df.copy()
        
        # CRITICAL: Only set intsta_df if user is in "Automatic Detection" mode
        if 'standard_source_preference' not in st.session_state or st.session_state.standard_source_preference == "Automatic Detection":
            st.session_state.intsta_df = intsta_df.copy() if intsta_df is not None else pd.DataFrame()
    
    # Create normalizer instance
    normalizer = lp.NormalizeData()
    
    # Check if experiment has BQC samples
    bqc_label = st.session_state.get('bqc_label')
    has_bqc = bqc_label is not None and bqc_label in st.session_state.experiment.conditions_list
    
    # ==========================================================================
    # Configure Zero Filtering expander
    # ==========================================================================
    with st.expander("⚙️ Configure Zero Filtering", expanded=False):
        
        # Check if cleaned_df is valid
        if st.session_state.cleaned_df is None or st.session_state.cleaned_df.empty:
            st.error("No valid cleaned data available for zero filtering.")
            return None
        
        st.markdown("Adjust thresholds for removing lipid species with too many zero/below-detection values.")
        
        # Threshold sliders
        col1, col2 = st.columns(2)
        
        with col1:
            non_bqc_pct = st.slider(
                "Non-BQC threshold (%)",
                min_value=50,
                max_value=100,
                value=75,
                step=5,
                help="Remove species if ALL non-BQC conditions have ≥ this % zeros",
                key="non_bqc_zero_threshold"
            )
            non_bqc_threshold = non_bqc_pct / 100.0
        
        with col2:
            if has_bqc:
                bqc_pct = st.slider(
                    f"BQC threshold (%) — {bqc_label}",
                    min_value=25,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Remove species if BQC condition has ≥ this % zeros",
                    key="bqc_zero_threshold"
                )
                bqc_threshold = bqc_pct / 100.0
            else:
                bqc_threshold = 0.5
                st.info("No BQC condition — only non-BQC threshold applies.")
        
        # Apply filter with user-selected thresholds
        filtered_df, removed_species = apply_zero_filter(
            st.session_state.cleaned_df,
            st.session_state.experiment,
            st.session_state.format_type,
            bqc_label=bqc_label,
            bqc_threshold=bqc_threshold,
            non_bqc_threshold=non_bqc_threshold
        )
        
        # Display results
        removed = len(removed_species)
        if removed > 0:
            st.warning(f"**Result:** Removed {removed} species ({removed / len(st.session_state.cleaned_df) * 100:.1f}% of dataset)")
            removed_df = pd.DataFrame({'LipidMolec': removed_species})
            st.dataframe(removed_df, use_container_width=True, height=150)
            csv = removed_df.to_csv(index=False)
            st.download_button(
                label="Download Removed Species",
                data=csv,
                file_name="removed_species.csv",
                mime="text/csv",
                key="download_removed_species"
            )
        else:
            st.success("**Result:** No species removed")
        
        # Update session state with filtered data
        st.session_state.cleaned_df = filtered_df
        st.session_state.continuation_df = filtered_df
    
    # ==========================================================================
    # Final Filtered Data (OUTSIDE expander)
    # ==========================================================================
    st.markdown("##### 📋 Final Filtered Data (Pre-Normalization)")
    display_data(filtered_df, "Data", "final_filtered_data.csv", key_suffix="filtered")
    
    # ==========================================================================
    # Manage Internal Standards expander
    # ==========================================================================
    with st.expander("Manage Internal Standards"):
        st.markdown("""
        Auto-detection identifies deuterated standards (`(d5)`, `(d7)`, `(d9)`), 
        `ISTD`/`IS` markers in class names, and SPLASH LIPIDOMIX® patterns.
        """)
        manage_internal_standards(normalizer)
        
        # Show plots if standards are available
        show_plots = False
        if st.session_state.get('standard_source_preference') == "Automatic Detection":
            show_plots = not st.session_state.intsta_df.empty
        else:
            show_plots = ('preserved_intsta_df' in st.session_state and 
                         st.session_state.preserved_intsta_df is not None and 
                         not st.session_state.preserved_intsta_df.empty)
        
        if show_plots:
            st.markdown("---")
            st.markdown("##### 📊 Internal Standards Consistency")
            st.markdown("*Consistent bar heights across samples indicate good sample preparation and instrument performance.*")
            
            conditions = st.session_state.experiment.conditions_list
            selected_conditions = st.multiselect(
                'Select conditions:',
                conditions,
                default=conditions,
                key='standards_conditions_select'
            )
            
            selected_samples = []
            for cond in selected_conditions:
                idx = conditions.index(cond)
                selected_samples.extend(st.session_state.experiment.individual_samples_list[idx])
            
            full_samples = st.session_state.experiment.full_samples_list
            selected_samples_ordered = [s for s in full_samples if s in selected_samples]
            
            if selected_samples_ordered:
                intsta_to_plot = st.session_state.preserved_intsta_df if 'preserved_intsta_df' in st.session_state and st.session_state.preserved_intsta_df is not None and not st.session_state.preserved_intsta_df.empty else st.session_state.intsta_df
                
                if intsta_to_plot is not None and not intsta_to_plot.empty:
                    plots = lp.InternalStandardsPlotter.create_consistency_plots(
                        intsta_to_plot,
                        selected_samples_ordered
                    )
                    
                    if plots:
                        for fig in plots:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No standards data available for plotting.")
    
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
    
    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes
    
    # ==========================================================================
    # About Normalization Methods (documentation)
    # ==========================================================================
    with st.expander("📖 About Normalization Methods", expanded=False):
        st.markdown("""
### Normalization Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **None** | Raw values | Data already normalized externally |
| **Internal Standards** | `(Intensity_lipid / Intensity_standard) × Conc_standard` | Correct for extraction/ionization variability |
| **Protein-based** | `Intensity_lipid / Protein_conc` | Normalize to starting material (e.g., BCA assay) |
| **Both** | `(Intensity_lipid / Intensity_standard) × (Conc_standard / Protein_conc)` | Combined correction |

After normalization, `intensity[...]` columns become `concentration[...]` columns.
        """)
    
    # ==========================================================================
    # Class Selection
    # ==========================================================================
    st.markdown("##### 🎯 Select Lipid Classes")
    selected_classes = st.multiselect(
        'Classes to analyze:',
        options=all_class_lst,
        default=all_class_lst if not st.session_state.selected_classes else st.session_state.selected_classes,
        key='temp_selected_classes',
        on_change=update_selected_classes
    )
    
    if not selected_classes:
        st.warning("Select at least one lipid class to proceed.")
        return None

    # Filter DataFrame based on selected classes
    filtered_df = cleaned_df[cleaned_df['ClassKey'].isin(selected_classes)].copy()

    # Check if we have standards
    has_standards = not intsta_df.empty

    # Store standard source persistently
    if 'standard_source' not in st.session_state:
        st.session_state.standard_source = "Automatic Detection" if has_standards else None
    
    if 'preserved_intsta_df' not in st.session_state:
        st.session_state.preserved_intsta_df = None

    normalized_data_object = lp.NormalizeData()

    # ==========================================================================
    # Normalization Method Selection
    # ==========================================================================
    st.markdown("##### ⚙️ Normalization Method")
    
    # Determine available options
    normalization_options = ['None (pre-normalized data)', 'Internal Standards', 'Protein-based', 'Both'] if has_standards else ['None (pre-normalized data)', 'Protein-based']
    
    if not has_standards:
        st.markdown("*Internal standards options unavailable — no standards detected or uploaded.*")
    
    # Initialize the radio key if it doesn't exist or if saved method is no longer available
    if 'norm_method_selection' not in st.session_state:
        st.session_state.norm_method_selection = 'None (pre-normalized data)'
    
    # Handle case where saved method is no longer available (e.g., standards removed)
    if st.session_state.norm_method_selection not in normalization_options:
        st.session_state.norm_method_selection = 'None (pre-normalized data)'
    
    normalization_method = st.radio(
        "Method:",
        options=normalization_options,
        key='norm_method_selection',
        horizontal=True
    )
    
    # Keep legacy session state in sync for backward compatibility
    st.session_state.normalization_method = normalization_method

    normalized_df = filtered_df.copy()

    if normalization_method != 'None (pre-normalized data)':
        if 'normalization_settings' not in st.session_state:
            st.session_state.normalization_settings = {}
        
        do_standards = normalization_method in ['Internal Standards', 'Both'] and has_standards
        
        # ----------------------------------------------------------------------
        # Protein-based normalization
        # ----------------------------------------------------------------------
        if normalization_method in ['Protein-based', 'Both']:
            with st.expander("⚙️ Protein Concentration Data", expanded=True):
                # Get fresh protein data from user input
                new_protein_df = collect_protein_concentrations(experiment)
                
                if new_protein_df is not None:
                    # User provided valid data - apply normalization
                    st.session_state.protein_df = new_protein_df
                    try:
                        normalized_df = normalized_data_object.normalize_using_bca(
                            normalized_df, 
                            new_protein_df, 
                            preserve_prefix=True
                        )
                        st.session_state.normalization_settings['protein'] = {'protein_df': new_protein_df}
                        st.success("✓ Protein normalization applied")
                    except Exception as e:
                        st.error(f"Protein normalization error: {str(e)}")
                        return None
                else:
                    # No data yet (waiting for CSV upload) - don't apply normalization
                    if 'protein' in st.session_state.get('normalization_settings', {}):
                        del st.session_state.normalization_settings['protein']

        # ----------------------------------------------------------------------
        # Internal standards normalization
        # ----------------------------------------------------------------------
        if do_standards:
            with st.expander("⚙️ Internal Standards Mapping", expanded=True):
                intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
                if not intensity_cols:
                    st.error("Standards data missing intensity columns.")
                    return None

                # Group standards by class
                standards_by_class = {}
                if 'ClassKey' in intsta_df.columns:
                    standards_by_class = intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()
        
                class_to_standard_map = process_class_standards(
                    selected_classes, 
                    standards_by_class, 
                    intsta_df,
                    st.session_state.get('class_standard_map', {})
                )
                
                if not class_to_standard_map:
                    return None

                st.session_state.class_standard_map = class_to_standard_map

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

    # Rename intensity → concentration
    normalized_df = rename_intensity_to_concentration(normalized_df)
    
    # Add back LipidSearch essential columns
    if normalized_df is not None and format_type == 'LipidSearch 5.0':
        for col, values in stored_columns.items():
            normalized_df[col] = values

    # ==========================================================================
    # Final Normalized Data
    # ==========================================================================
    if normalized_df is not None:
        st.markdown("##### 📊 Final Normalized Data")
        st.dataframe(normalized_df, use_container_width=True)
        csv = normalized_df.to_csv(index=False)
        st.download_button(
            label="Download Normalized Data",
            data=csv,
            file_name="normalized_data.csv",
            mime="text/csv",
            key="download_normalized_data"
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
    
    # Identify classes without specific standards and show warnings at the top
    for lipid_class in selected_classes:
        class_specific_standards = standards_by_class.get(lipid_class, [])
        if not class_specific_standards:
            # Determine what standard will be used (saved mapping or first available)
            if saved_mappings and lipid_class in saved_mappings and saved_mappings[lipid_class] in all_available_standards:
                default_standard = saved_mappings[lipid_class]
            else:
                default_standard = all_available_standards[0]
            st.warning(f"No specific standards available for {lipid_class}. Defaulting to {default_standard}.")
    
    for lipid_class in selected_classes:
        # Determine default standard
        default_standard = None
        
        # Try saved mapping first
        if saved_mappings and lipid_class in saved_mappings:
            default_standard = saved_mappings[lipid_class]
            if default_standard not in all_available_standards:
                default_standard = None

        # If no saved mapping, use class-specific or fall back to first available
        if not default_standard:
            class_specific_standards = standards_by_class.get(lipid_class, [])
            if class_specific_standards:
                default_standard = class_specific_standards[0]
            else:
                default_standard = all_available_standards[0]

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
        pd.DataFrame or None: A DataFrame with 'Sample' and 'Concentration' columns,
                              or None if input is incomplete
    """
    # Initialize method selection key if not present
    if 'protein_input_method' not in st.session_state:
        st.session_state.protein_input_method = "Upload CSV File"
    
    # Track previous method to detect changes
    prev_method = st.session_state.get('protein_input_method_prev')
    
    method = st.radio(
        "Input method:",
        ["Manual Input", "Upload CSV File"],
        key='protein_input_method',
        horizontal=True
    )
    
    # Detect method change and clear stale data
    if prev_method is not None and prev_method != method:
        # Method changed - clear the stored protein_df to prevent stale data
        if 'protein_df' in st.session_state:
            del st.session_state.protein_df
        if 'normalization_settings' in st.session_state and 'protein' in st.session_state.normalization_settings:
            del st.session_state.normalization_settings['protein']
    
    # Update previous method tracker
    st.session_state.protein_input_method_prev = method
    
    if method == "Manual Input":
        protein_concentrations = {}
        
        # Use columns for compact layout
        cols = st.columns(3)
        for idx, sample in enumerate(experiment.full_samples_list):
            with cols[idx % 3]:
                concentration = st.number_input(
                    f'{sample}:',
                    min_value=0.0, 
                    max_value=1000000.0, 
                    value=1.0, 
                    step=0.1,
                    key=f"protein_{sample}"
                )
                protein_concentrations[sample] = concentration

        protein_df = pd.DataFrame(list(protein_concentrations.items()), 
                                columns=['Sample', 'Concentration'])
        return protein_df
    
    else:  # Upload CSV
        st.markdown("**CSV format:** Single column named `Concentration` with one value per sample (in order).")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv", key="protein_csv_upload")
        
        if uploaded_file is not None:
            try:
                protein_df = pd.read_csv(uploaded_file)
                
                if 'Concentration' not in protein_df.columns:
                    st.error(f"CSV must contain a column named 'Concentration'. Found: {list(protein_df.columns)}")
                    return None
                    
                if len(protein_df) != len(experiment.full_samples_list):
                    st.error(f"Row count ({len(protein_df)}) doesn't match sample count ({len(experiment.full_samples_list)})")
                    return None
                
                protein_df['Concentration'] = pd.to_numeric(protein_df['Concentration'], errors='coerce')
                if protein_df['Concentration'].isna().any():
                    st.error("Some concentration values couldn't be converted to numbers.")
                    return None
                
                protein_df['Sample'] = experiment.full_samples_list
                st.success(f"✓ Loaded {len(protein_df)} concentration values")
                
                return protein_df
                
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
                return None
        
        # No file uploaded yet - show info message
        st.info("Please upload a CSV file with protein concentrations.")
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
                    st.session_state.pathway_visualization, st.session_state.fach_plots,
                    experiment=experiment, format_type=format_type
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
    
    with st.expander('View Distributions of AUC: Scan Data & Detect Atypical Patterns'):
        # Concise description
        st.markdown("""
        Assess data quality and identify anomalies. Two diagnostic visualizations help detect technical issues or outliers.
        """)
        
        st.markdown("**What to look for:** Similar patterns across replicates indicate good reproducibility. Large differences may signal quality issues.")
        
        # Creating a deep copy for visualization
        visualization_df = continuation_df.copy(deep=True)
        
        # Ensure the columns reflect the current state of the DataFrame
        current_samples = [
            sample for sample in experiment.full_samples_list 
            if f'concentration[{sample}]' in visualization_df.columns
        ]
        
        mean_area_df = lp.BoxPlot.create_mean_area_df(visualization_df, current_samples)
        zero_values_percent_list = lp.BoxPlot.calculate_missing_values_percentage(mean_area_df)
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("##### 📈 Results")
        
        # Missing Values Distribution
        st.markdown("###### Missing Values Distribution")
        st.markdown("Percentage of zero values per sample. High percentages may indicate lower sensitivity or technical issues.")
        
        fig1 = lp.BoxPlot.plot_missing_values(current_samples, zero_values_percent_list)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Download buttons for missing values
        missing_values_data = np.vstack((current_samples, zero_values_percent_list)).T
        missing_values_df = pd.DataFrame(missing_values_data, columns=["Sample", "Percentage Missing"])
        missing_values_csv = convert_df(missing_values_df)
        
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(fig1, "missing_values_distribution.svg")
        with col2:
            st.download_button(
                label="Download CSV",
                data=missing_values_csv,
                file_name="missing_values_data.csv",
                mime='text/csv',
                key=f"download_missing_values_data_{unique_id}"
            )
        
        st.markdown("---")
        
        # Box Plot of Non-Zero Concentrations
        st.markdown("###### Concentration Distribution")
        st.markdown("Log10-transformed non-zero concentrations. Box = IQR (25th-75th percentile), line = median, points = outliers.")
        
        fig2 = lp.BoxPlot.plot_box_plot(mean_area_df, current_samples)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Download buttons for box plot
        csv_data = convert_df(mean_area_df)
        
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(fig2, "box_plot.svg")
        with col2:
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
    scatter_plot = None
    if bqc_label is not None:
        with st.expander("Quality Check Using BQC Samples"):
            # Concise description with formula
            st.markdown("""
            Evaluate measurement reliability using Batch Quality Control (BQC) samples. Lower CoV = more consistent measurements.
            """)
            
            st.markdown("**Coefficient of Variation (CoV):**")
            st.code("CoV = (Standard_Deviation / Mean) × 100%", language=None)
            st.markdown("Blue points are below threshold (reliable), red points are above (variable).")
            
            # --- Settings Section ---
            st.markdown("---")
            st.markdown("##### ⚙️ Settings")
            
            cov_threshold = st.number_input(
                'CoV Threshold (%)',
                min_value=10,
                max_value=1000,
                value=30,
                step=1,
                help="Points above threshold highlighted in red.",
                key='bqc_cov_threshold'
            )
            
            # --- Results Section ---
            st.markdown("---")
            st.markdown("##### 📈 Results")
            
            bqc_sample_index = experiment.conditions_list.index(bqc_label)
            scatter_plot, prepared_df, reliable_data_percent, filtered_lipids = lp.BQCQualityCheck.generate_and_display_cov_plot(
                data_df,
                experiment,
                bqc_sample_index,
                cov_threshold=cov_threshold
            )
            
            st.plotly_chart(scatter_plot, use_container_width=True)
            
            # Reliability assessment
            if reliable_data_percent >= 80:
                st.success(f"{reliable_data_percent:.1f}% of datapoints are reliable (CoV < {cov_threshold}%).")
            elif reliable_data_percent >= 50:
                st.warning(f"{reliable_data_percent:.1f}% of datapoints are reliable (CoV < {cov_threshold}%).")
            else:
                st.error(f"Less than 50% of datapoints are reliable (CoV < {cov_threshold}%).")
            
            # Download buttons
            csv_data = convert_df(prepared_df[['LipidMolec', 'cov', 'mean']].dropna())
            col1, col2 = st.columns(2)
            with col1:
                plotly_svg_download_button(scatter_plot, "bqc_quality_check.svg")
            with col2:
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="cov_plot_data.csv",
                    mime='text/csv',
                    key='bqc_csv_download'
                )
            
            # --- Filtering Section ---
            if prepared_df is not None and not prepared_df.empty:
                st.markdown("---")
                st.markdown("##### 🔧 Data Filtering")
                
                filter_cov = st.radio(
                    f"Filter lipids with CoV ≥ {cov_threshold}%?",
                    ("No", "Yes"),
                    index=0,
                    horizontal=True,
                    key='bqc_filter_choice'
                )
                
                cov_to_keep = []
                if filter_cov == "Yes":
                    cov_filtered_df = prepared_df[prepared_df['cov'] >= cov_threshold][['LipidMolec', 'ClassKey', 'cov', 'mean']]
                    cov_filtered_df['cov'] = cov_filtered_df['cov'].round(2)
                    cov_filtered_df['mean'] = cov_filtered_df['mean'].round(4)
                    
                    if not cov_filtered_df.empty:
                        st.markdown("###### Lipids Above Threshold")
                        st.dataframe(cov_filtered_df, use_container_width=True)
                        
                        cov_to_keep = st.multiselect(
                            "Keep despite high CoV:",
                            options=cov_filtered_df['LipidMolec'].tolist(),
                            format_func=lambda x: f"{x} (CoV: {cov_filtered_df[cov_filtered_df['LipidMolec'] == x]['cov'].iloc[0]}%)",
                            help="Select lipids to retain in the dataset.",
                            key='bqc_lipids_to_keep'
                        )
                    else:
                        st.info(f"No lipids with CoV ≥ {cov_threshold}% found.")
                else:
                    cov_high = prepared_df[prepared_df['cov'] >= cov_threshold]['LipidMolec'].tolist()
                    cov_to_keep = cov_high
                
                # Apply filters
                filtered_df = data_df.copy()
                cov_to_remove = [lipid for lipid in prepared_df[prepared_df['cov'] >= cov_threshold]['LipidMolec'] if lipid not in cov_to_keep]
                filtered_df = filtered_df[~filtered_df['LipidMolec'].isin(set(cov_to_remove))]
                filtered_df = filtered_df.sort_values(by='ClassKey').reset_index(drop=True)
                
                # Summary
                removed_count = len(data_df) - len(filtered_df)
                percentage_removed = round((removed_count / len(data_df)) * 100, 1) if len(data_df) > 0 else 0
                
                if removed_count > 0:
                    st.warning(f"Removed {removed_count} lipids ({percentage_removed:.1f}% of dataset).")
                else:
                    st.success("No lipids removed.")
                
                kept_cov = len(cov_to_keep) if filter_cov == "Yes" else 0
                if kept_cov > 0:
                    st.info(f"Retained {kept_cov} lipids despite high CoV.")
                
                # Filtered dataset
                st.markdown("###### Filtered Dataset")
                st.dataframe(filtered_df, use_container_width=True)
                
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data",
                    data=csv,
                    file_name='filtered_data.csv',
                    mime='text/csv',
                    key='bqc_filtered_download'
                )
                
                data_df = filtered_df
    
    return data_df, scatter_plot

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
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the selected condition and the matplotlib figure, or (None, None) if no plot was generated.
    """
    with st.expander('Pairwise Correlation Analysis'):
        # Concise description
        st.markdown("""
        Assess reproducibility by calculating Pearson correlation coefficients between sample replicates.
        """)
        
        st.markdown("**Interpretation:** Values close to 1 = similar patterns (good). Blue = higher correlation, Red = lower correlation.")
        
        # Filter out conditions with only one replicate
        multi_replicate_conditions = [
            condition for condition, num_samples 
            in zip(experiment.conditions_list, experiment.number_of_samples_list) 
            if num_samples > 1
        ]
        
        if not multi_replicate_conditions:
            st.error("No conditions with multiple replicates found. Correlation analysis requires at least two replicates.")
            return None, None
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("##### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_condition = st.selectbox(
                'Condition', 
                multi_replicate_conditions,
                key='corr_condition'
            )
        with col2:
            sample_type = st.selectbox(
                'Sample Type', 
                ['biological replicates', 'technical replicates'],
                help="Biological: different samples, same condition. Technical: repeated measurements, same sample.",
                key='corr_sample_type'
            )
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("##### 📈 Results")
        
        condition_index = experiment.conditions_list.index(selected_condition)
        mean_area_df = lp.Correlation.prepare_data_for_correlation(
            continuation_df, experiment.individual_samples_list, condition_index
        )
        correlation_df, v_min, thresh = lp.Correlation.compute_correlation(mean_area_df, sample_type)
        fig = lp.Correlation.render_correlation_plot(
            correlation_df, v_min, thresh, experiment.conditions_list[condition_index]
        )
        
        st.pyplot(fig)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            matplotlib_svg_download_button(fig, f"correlation_plot_{selected_condition}.svg")
        with col2:
            csv_download = convert_df(correlation_df)
            st.download_button(
                label="Download CSV",
                data=csv_download,
                file_name=f'correlation_matrix_{selected_condition}.csv',
                mime='text/csv',
                key='corr_csv_download'
            )
        
        # Correlation matrix table
        st.markdown("###### Correlation Coefficients")
        st.dataframe(correlation_df, use_container_width=True)
        
        # Flag low correlations
        min_threshold = 0.7 if sample_type == 'biological replicates' else 0.8
        low_correlations = []
        for i in range(len(correlation_df.columns)):
            for j in range(i + 1, len(correlation_df.columns)):
                if correlation_df.iloc[j, i] < min_threshold:
                    low_correlations.append(
                        f"{correlation_df.columns[i]} ↔ {correlation_df.index[j]}: {correlation_df.iloc[j, i]:.3f}"
                    )
        
        if low_correlations:
            st.warning(f"**Low correlations detected** (< {min_threshold}): " + " | ".join(low_correlations))
        
        return selected_condition, fig
        
def display_pca_analysis(continuation_df, experiment):
    """
    Displays the PCA analysis interface in the Streamlit app and generates a PCA plot using Plotly.
    
    Args:
        continuation_df (pd.DataFrame): The DataFrame containing the normalized or cleaned data.
        experiment (Experiment): The experiment object with details of the conditions and samples.
    
    Returns:
        tuple: A tuple containing the updated continuation_df and the PCA plot.
    """
    from scipy.stats import chi2
    
    pca_plot = None
    with st.expander("Principal Component Analysis (PCA)"):
        # Concise description
        st.markdown("""
        Visualize sample clustering based on lipid profiles. Each point = one sample, ellipses = 95% confidence intervals.
        """)
        
        st.markdown("**Interpretation:** Clustered points = similar profiles. Separated clusters = distinct conditions. Outliers fall outside ellipses.")
        
        # --- Settings Section ---
        st.markdown("---")
        st.markdown("##### ⚙️ Settings")
        
        samples_to_remove = st.multiselect(
            'Exclude Samples (optional)',
            experiment.full_samples_list,
            help="Exclude suspected outliers from analysis.",
            key='pca_samples_remove'
        )
        
        if samples_to_remove:
            remaining_count = len(experiment.full_samples_list) - len(samples_to_remove)
            if remaining_count < 2:
                st.error('At least two samples required for PCA.')
                return continuation_df, pca_plot
            else:
                continuation_df = experiment.remove_bad_samples(samples_to_remove, continuation_df)
                st.warning(f"⚠️ {len(samples_to_remove)} sample(s) excluded from **all downstream analyses**, not just PCA.")
                st.success(f"Proceeding with {remaining_count} samples.")
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("##### 📈 Results")
        
        pca_plot, pca_df = lp.PCAAnalysis.plot_pca(
            continuation_df, experiment.full_samples_list, experiment.extensive_conditions_list
        )
        st.plotly_chart(pca_plot, use_container_width=True)
        
        # Download buttons
        csv_data = convert_df(pca_df)
        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(pca_plot, "pca_plot.svg")
        with col2:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="pca_data.csv",
                mime="text/csv",
                key='pca_csv_download'
            )
        
        # Outlier detection
        all_outliers = []
        for condition in pca_df['Condition'].unique():
            condition_df = pca_df[pca_df['Condition'] == condition]
            if len(condition_df) >= 3:
                center = condition_df[['PC1', 'PC2']].mean().values
                cov = np.cov(condition_df['PC1'], condition_df['PC2'])
                
                try:
                    inv_cov = np.linalg.inv(cov)
                    for _, row in condition_df.iterrows():
                        point = np.array([row['PC1'], row['PC2']])
                        dist = np.sqrt(np.dot(np.dot((point - center), inv_cov), (point - center).T))
                        if dist > np.sqrt(chi2.ppf(0.95, 2)):
                            all_outliers.append(f"{row['Sample']} ({condition})")
                except np.linalg.LinAlgError:
                    pass
        
        if all_outliers:
            st.warning(f"**Potential outliers** (outside 95% ellipse): " + " | ".join(all_outliers))
    
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
        #
        stat_options = display_statistical_options()
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("##### 🎯 Data Selection")
        
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
            st.markdown("##### 📊 Results")
            
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
            st.markdown("##### 🔍 Detailed Statistics")
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
        st.markdown("##### 🎯 Data Selection")
        
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
            st.markdown(f"###### {condition}")
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
            st.markdown("##### ⚠️ Consolidated Format Lipids")
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
        st.markdown("##### 📊 Results")
        
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
    alpha = 0.05
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_type = st.selectbox(
            "Test Type",
            options=["parametric", "non_parametric"],
            index=0,
            help="Parametric: Welch's t-test | Non-parametric: Mann-Whitney U",
            key='volcano_test_type'
        )
        
    with col2:
        correction_method = st.selectbox(
            "Multiple Testing Correction",
            options=["uncorrected", "fdr_bh", "bonferroni"],
            index=1,
            help="Uncorrected | FDR (recommended) | Bonferroni (strict)",
            key='volcano_correction'
        )

    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,
        help="Standard practice in lipidomics to normalize skewed data.",
        key='volcano_transform'
    )
    
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
        # Short description with formula
        st.markdown("""
        Identify significantly altered lipid species between two conditions.
        """)
        
        st.markdown("**Fold Change** (x-axis, log2-transformed):")
        st.code("Fold Change = Mean_concentration_experimental / Mean_concentration_control", language=None)
        st.markdown('**Significance** (y-axis): -log10(p-value). Higher = more significant. See "About Statistical Testing" for methodology.')
        
        # Get valid conditions
        conditions_with_replicates = [
            condition for index, condition in enumerate(experiment.conditions_list)
            if experiment.number_of_samples_list[index] > 1
        ]

        if len(conditions_with_replicates) < 2:
            st.warning('At least two conditions with multiple replicates are required for volcano plot.')
            return volcano_plots
        
        # --- Statistical Options Section ---
        st.markdown("---")
        st.markdown("##### ⚙️ Statistical Options")
        stat_options = display_volcano_statistical_options()
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("##### 🎯 Data Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            control_condition = st.selectbox('Control Condition', conditions_with_replicates, key='volcano_control')
        with col2:
            available_experimental = [c for c in conditions_with_replicates if c != control_condition]
            experimental_condition = st.selectbox('Experimental Condition', available_experimental, key='volcano_experimental')

        if not available_experimental:
            st.error("Need at least two different conditions with multiple replicates.")
            return volcano_plots

        selected_classes_list = st.multiselect(
            'Lipid Classes',
            list(continuation_df['ClassKey'].value_counts().index),
            list(continuation_df['ClassKey'].value_counts().index),
            key='volcano_classes_select'
        )

        # Initialize session state for custom labels
        if 'custom_labeled_lipids' not in st.session_state:
            st.session_state.custom_labeled_lipids = []
        if 'custom_label_positions' not in st.session_state:
            st.session_state.custom_label_positions = {}

        use_adjusted_p = stat_options['correction_method'] != "uncorrected"

        # Validation
        if not selected_classes_list:
            st.warning("Please select at least one lipid class.")
            return volcano_plots
        if control_condition == experimental_condition:
            st.warning("Please select different conditions for control and experimental groups.")
            return volcano_plots
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("##### 📈 Results")
        
        # Significance & Display Settings (applies to all plots)
        st.markdown("**Significance & Display Settings**")
        col1, col2 = st.columns(2)
        with col1:
            p_value_threshold = st.number_input(
                'P-value threshold',
                min_value=0.001, max_value=0.1, value=0.05, step=0.001,
                help="Statistical significance cutoff",
                key='volcano_pval'
            )
        with col2:
            fold_change_threshold = st.number_input(
                'Fold change threshold',
                min_value=1.1, max_value=5.0, value=2.0, step=0.1,
                help="Biological significance cutoff",
                key='volcano_fc'
            )

        q_value_threshold = -np.log10(p_value_threshold)
        log2_fc_threshold = np.log2(fold_change_threshold)

        hide_non_significant = st.checkbox(
            'Hide non-significant points',
            value=False,
            help="Hide points that don't meet both thresholds",
            key='volcano_hide'
        )

        with st.spinner("Performing statistical analysis..."):
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

                # --- Volcano Plot ---
                st.markdown("---")
                st.markdown("###### Volcano Plot")
                
                # Labeling options
                st.markdown("**Labeling Options**")
                col1, col2 = st.columns(2)
                with col1:
                    top_n_labels = st.number_input(
                        'Top N lipids to label:',
                        min_value=0, max_value=50, step=1, value=0,
                        key='top_n_labels',
                        help="Most significant lipids to label automatically"
                    )
                with col2:
                    adjust_labels = st.checkbox("Enable label adjustment", value=False, key='volcano_adjust')

                available_lipids = volcano_df['LipidMolec'].tolist()
                st.session_state.custom_labeled_lipids = st.multiselect(
                    'Additional lipids to label:',
                    available_lipids,
                    default=st.session_state.custom_labeled_lipids,
                    key='custom_lipid_labels'
                )

                # Combine top N and user-selected lipids
                p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
                top_n_lipids = volcano_df.sort_values(p_col, ascending=False).head(top_n_labels)['LipidMolec'].tolist()
                all_lipids_to_label = list(set(top_n_lipids + st.session_state.custom_labeled_lipids))

                # Label adjustment UI
                if adjust_labels and all_lipids_to_label:
                    with st.container():
                        st.markdown("**Label Position Adjustments:**")
                        for lipid in all_lipids_to_label:
                            key = f"pos_{lipid}"
                            if key not in st.session_state.custom_label_positions:
                                st.session_state.custom_label_positions[key] = (0.0, 0.0)

                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.text(lipid)
                            with col2:
                                x_offset = st.number_input(
                                    "X", value=st.session_state.custom_label_positions[key][0],
                                    step=0.1, key=f"x_offset_{lipid}", label_visibility="collapsed"
                                )
                            with col3:
                                y_offset = st.number_input(
                                    "Y", value=st.session_state.custom_label_positions[key][1],
                                    step=0.1, key=f"y_offset_{lipid}", label_visibility="collapsed"
                                )
                            st.session_state.custom_label_positions[key] = (x_offset, y_offset)

                # Create color mapping and plot
                color_mapping = lp.VolcanoPlot._generate_color_mapping(volcano_df)
                
                plot = lp.VolcanoPlot._create_plot_enhanced(
                    volcano_df, color_mapping, q_value_threshold, hide_non_significant,
                    use_adjusted_p, log2_fc_threshold, top_n_labels=0
                )

                # Add annotations for labeled lipids
                if all_lipids_to_label:
                    plot.layout.annotations = []
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
                                    x=label_x, y=label_y, text=lipid,
                                    showarrow=False, font=dict(color=color, size=12), align=align
                                )
                                plot.add_annotation(
                                    x=point_x, y=point_y, ax=label_x, ay=label_y,
                                    axref='x', ayref='y', text='',
                                    showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1, arrowcolor='black'
                                )
                                placed_boxes.append({
                                    'left': left - buffer, 'right': right + buffer,
                                    'bottom': bottom - buffer, 'top': top + buffer
                                })
                                placed = True
                                break

                        if not placed:
                            align_fallback = 'left' if point_x > 0 else 'right'
                            label_x_fallback = point_x + (0.1 if align_fallback == 'left' else -0.1) + custom_x
                            label_y_fallback = point_y + 0.2 + custom_y
                            plot.add_annotation(
                                x=label_x_fallback, y=label_y_fallback, text=lipid,
                                showarrow=False, font=dict(color=color, size=12), align=align_fallback
                            )
                            plot.add_annotation(
                                x=point_x, y=point_y, ax=label_x_fallback, ay=label_y_fallback,
                                axref='x', ayref='y', text='',
                                showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1, arrowcolor='black'
                            )

                # Display plot
                st.plotly_chart(plot, use_container_width=True)
                volcano_plots['main'] = plot

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    svg_bytes = plot.to_image(format="svg")
                    st.download_button(
                        label="Download SVG", data=svg_bytes.decode('utf-8'),
                        file_name=f"volcano_plot_{experimental_condition}_vs_{control_condition}.svg",
                        mime="image/svg+xml", key='volcano_svg'
                    )
                with col2:
                    png_bytes = plot.to_image(format="png")
                    st.download_button(
                        label="Download PNG", data=png_bytes,
                        file_name=f"volcano_plot_{experimental_condition}_vs_{control_condition}.png",
                        mime="image/png", key='volcano_png'
                    )

                # Detailed statistical results
                show_detailed = st.checkbox("Show detailed statistical results", value=False, key='volcano_show_stats')
                if show_detailed:
                    display_volcano_detailed_statistical_results(
                        statistical_results, control_condition, experimental_condition
                    )

                # --- Concentration vs Fold Change Plot ---
                st.markdown("---")
                st.markdown("###### Concentration vs. Fold Change")
                st.markdown("Relationship between lipid abundance and fold change.")
                
                concentration_vs_fold_change_plot, download_df = lp.VolcanoPlot._create_concentration_vs_fold_change_plot(
                    volcano_df, color_mapping, q_value_threshold, hide_non_significant, use_adjusted_p=True
                )
                st.plotly_chart(concentration_vs_fold_change_plot, use_container_width=True)
                volcano_plots['concentration_vs_fold_change'] = concentration_vs_fold_change_plot

                col1, col2 = st.columns(2)
                with col1:
                    svg_bytes = concentration_vs_fold_change_plot.to_image(format="svg")
                    st.download_button(
                        label="Download SVG", data=svg_bytes.decode('utf-8'),
                        file_name=f"conc_vs_fc_{experimental_condition}_vs_{control_condition}.svg",
                        mime="image/svg+xml", key='conc_fc_svg'
                    )
                with col2:
                    png_bytes = concentration_vs_fold_change_plot.to_image(format="png")
                    st.download_button(
                        label="Download PNG", data=png_bytes,
                        file_name=f"conc_vs_fc_{experimental_condition}_vs_{control_condition}.png",
                        mime="image/png", key='conc_fc_png'
                    )

                # --- Individual Lipid Analysis ---
                st.markdown("---")
                st.markdown("##### Individual Lipid Analysis")
                st.markdown("Examine concentration distributions for specific lipids.")
                
                all_classes = list(volcano_df['ClassKey'].unique())
                col1, col2 = st.columns(2)
                with col1:
                    selected_class = st.selectbox('Lipid Class:', all_classes, key='volcano_detail_class')
                with col2:
                    if selected_class:
                        class_lipids = volcano_df[volcano_df['ClassKey'] == selected_class]['LipidMolec'].unique().tolist()
                        sorted_lipids = volcano_df[volcano_df['ClassKey'] == selected_class].sort_values(
                            '-log10(adjusted_pValue)', ascending=False
                        )['LipidMolec'].tolist()
                        
                        selected_lipids = st.multiselect(
                            'Lipids:',
                            class_lipids,
                            default=sorted_lipids[:3] if len(sorted_lipids) >= 3 else sorted_lipids,
                            key='volcano_detail_lipids'
                        )
                    else:
                        selected_lipids = []

                if selected_lipids:
                    selected_conditions = [control_condition, experimental_condition]
                    with st.spinner("Generating distribution plot..."):
                        plot_df = lp.VolcanoPlot.create_concentration_distribution_data(
                            continuation_df, selected_lipids, selected_conditions, experiment
                        )
                        fig = lp.VolcanoPlot.create_concentration_distribution_plot(
                            plot_df, selected_lipids, selected_conditions
                        )
                        st.pyplot(fig)
                        volcano_plots['concentration_distribution'] = fig

                        col1, col2 = st.columns(2)
                        with col1:
                            buf = io.BytesIO()
                            fig.savefig(buf, format='svg', bbox_inches='tight')
                            buf.seek(0)
                            st.download_button(
                                label="Download SVG", data=buf.getvalue().decode('utf-8'),
                                file_name=f"conc_dist_{experimental_condition}_vs_{control_condition}.svg",
                                mime="image/svg+xml", key='dist_svg'
                            )
                        with col2:
                            st.download_button(
                                label="Download CSV", data=plot_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"conc_dist_data_{experimental_condition}_vs_{control_condition}.csv",
                                mime="text/csv", key='dist_csv'
                            )

                # --- Excluded Lipids ---
                st.markdown("---")
                st.markdown("##### Excluded Lipids")
                if not removed_lipids_df.empty:
                    st.dataframe(removed_lipids_df, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download CSV", data=removed_lipids_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"excluded_lipids_{experimental_condition}_vs_{control_condition}.csv",
                            mime="text/csv", key='excluded_csv'
                        )
                    with col2:
                        exclusion_summary = removed_lipids_df['Reason'].value_counts()
                        summary_text = " | ".join([f"{reason}: {count}" for reason, count in exclusion_summary.items()])
                        st.markdown(f"**Reasons:** {summary_text}")
                else:
                    st.success("✓ No lipids were excluded from the analysis.")

            except Exception as e:
                st.error(f"Error generating volcano plot: {str(e)}")
                import traceback
                st.error(f"Debug info: {traceback.format_exc()}")

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
        # Concise description with formula
        st.markdown("""
        Visualize lipid abundance patterns across samples. Colors represent Z-scores (standardized values).
        """)
        
        st.markdown("**Z-score** (color scale):")
        st.code("Z-score = (Value - Mean) / Standard_Deviation", language=None)
        st.markdown("Red = above average, Blue = below average, White = average. Enables comparison across lipids with different absolute abundances.")
        
        # --- Data Selection Section ---
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")
        
        all_conditions = experiment.conditions_list
        all_classes = list(continuation_df['ClassKey'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_conditions = st.multiselect(
                "Conditions", 
                all_conditions, 
                default=all_conditions,
                key='heatmap_conditions'
            )
        with col2:
            selected_classes = st.multiselect(
                "Lipid Classes", 
                all_classes, 
                default=all_classes,
                key='heatmap_classes'
            )
        
        # Filter for conditions with samples
        selected_conditions = [condition for condition in selected_conditions 
                             if len(experiment.individual_samples_list[experiment.conditions_list.index(condition)]) > 0]
        
        if not selected_conditions or not selected_classes:
            st.warning("Please select at least one condition and one lipid class.")
            return regular_heatmap, clustered_heatmap
        
        # --- Heatmap Settings Section ---
        st.markdown("---")
        st.markdown("#### ⚙️ Heatmap Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            heatmap_type = st.radio(
                "Heatmap Type", 
                ["Clustered", "Regular"], 
                index=0,
                help="Clustered: groups similar lipids. Regular: preserves original order.",
                key='heatmap_type'
            )
        with col2:
            if heatmap_type == "Clustered":
                n_clusters = st.slider(
                    "Number of Clusters", 
                    min_value=2, 
                    max_value=10, 
                    value=5,
                    help="More clusters = more detailed groupings.",
                    key='heatmap_clusters'
                )
            else:
                st.markdown("")  # Empty placeholder for alignment
                n_clusters = 5  # Default value (not used for regular)
        
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
            return regular_heatmap, clustered_heatmap
        
        # --- Results Section ---
        st.markdown("---")
        st.markdown("#### 📈 Results")
        
        # Compute Z-scores
        with st.spinner("Computing Z-scores..."):
            z_scores_df = lp.LipidomicHeatmap.compute_z_scores(filtered_df)
        
        # Generate and display the appropriate heatmap
        if heatmap_type == "Clustered":
            with st.spinner("Generating clustered heatmap..."):
                clustered_heatmap = lp.LipidomicHeatmap.generate_clustered_heatmap(
                    z_scores_df, 
                    selected_samples, 
                    n_clusters
                )
                
                st.plotly_chart(clustered_heatmap, use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    plotly_svg_download_button(clustered_heatmap, "lipidomic_clustered_heatmap.svg")
                with col2:
                    csv_download = convert_df(z_scores_df.reset_index())
                    st.download_button(
                        "Download CSV", 
                        csv_download, 
                        'clustered_heatmap_data.csv', 
                        'text/csv',
                        key='clustered_heatmap_csv'
                    )
                
                # Cluster composition
                st.markdown("---")
                st.markdown("##### Cluster Composition")
                
                composition_view = st.radio(
                    "Show composition by:",
                    ["Species Count", "Total Concentration"],
                    horizontal=True,
                    key='cluster_composition_view',
                    help="Species Count: % of lipid species. Total Concentration: % of summed abundance."
                )
                
                if composition_view == "Species Count":
                    _, class_percentages = lp.LipidomicHeatmap.identify_clusters_and_percentages(
                        z_scores_df, 
                        n_clusters
                    )
                    st.markdown("Percentage of lipid species within each cluster.")
                else:
                    class_percentages = lp.LipidomicHeatmap.identify_clusters_and_concentration_percentages(
                        z_scores_df,
                        filtered_df,
                        n_clusters
                    )
                    st.markdown("Percentage of total concentration within each cluster.")
                
                if not class_percentages.empty:
                    formatted_percentages = class_percentages.round(1)
                    st.dataframe(formatted_percentages, use_container_width=True)
                    
                    csv_download = convert_df(formatted_percentages.reset_index())
                    st.download_button(
                        "Download CSV", 
                        csv_download, 
                        f'cluster_composition_{composition_view.lower().replace(" ", "_")}.csv', 
                        'text/csv',
                        key='cluster_composition_csv'
                    )
        else:
            with st.spinner("Generating regular heatmap..."):
                regular_heatmap = lp.LipidomicHeatmap.generate_regular_heatmap(
                    z_scores_df, 
                    selected_samples
                )
                
                st.plotly_chart(regular_heatmap, use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    plotly_svg_download_button(regular_heatmap, "lipidomic_regular_heatmap.svg")
                with col2:
                    csv_download = convert_df(z_scores_df.reset_index())
                    st.download_button(
                        "Download CSV", 
                        csv_download, 
                        'regular_heatmap_data.csv', 
                        'text/csv',
                        key='regular_heatmap_csv'
                    )
    
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
                       saturation_plots, volcano_plots, pathway_visualization, fach_plots=None,
                       experiment=None, format_type=None):
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
        experiment: Experiment object with experimental setup info (default None)
        format_type: String indicating data format type (default None)
    
    Returns:
        BytesIO: Buffer containing the generated PDF
    """
    from datetime import datetime
    
    pdf_buffer = io.BytesIO()
    
    try:
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        page_width, page_height = letter
        
        # === Page 1: Metadata / Cover Page ===
        # Title
        pdf.setFont("Helvetica-Bold", 24)
        pdf.drawCentredString(page_width / 2, page_height - 100, "LipidCruncher Analysis Report")
        
        # Subtitle
        pdf.setFont("Helvetica", 14)
        pdf.drawCentredString(page_width / 2, page_height - 130, "Quality Check & Lipidomic Analysis Summary")
        
        # Horizontal line
        pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
        pdf.setLineWidth(1)
        pdf.line(50, page_height - 150, page_width - 50, page_height - 150)
        
        # Report metadata section
        y_position = page_height - 190
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_position, "Report Information")
        y_position -= 25
        
        pdf.setFont("Helvetica", 11)
        pdf.drawString(70, y_position, f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        y_position -= 20
        
        if format_type:
            pdf.drawString(70, y_position, f"Data Format: {format_type}")
            y_position -= 20
        
        # Experiment details section
        if experiment:
            y_position -= 15
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(50, y_position, "Experimental Design")
            y_position -= 25
            
            pdf.setFont("Helvetica", 11)
            pdf.drawString(70, y_position, f"Number of Conditions: {experiment.n_conditions}")
            y_position -= 20
            
            pdf.drawString(70, y_position, f"Total Samples: {len(experiment.full_samples_list)}")
            y_position -= 25
            
            # Conditions table header
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(70, y_position, "Condition")
            pdf.drawString(250, y_position, "Samples")
            pdf.drawString(400, y_position, "Sample IDs")
            y_position -= 5
            pdf.line(70, y_position, page_width - 50, y_position)
            y_position -= 15
            
            # Conditions details
            pdf.setFont("Helvetica", 10)
            for i, (condition, n_samples, samples) in enumerate(zip(
                experiment.conditions_list,
                experiment.number_of_samples_list,
                experiment.individual_samples_list
            )):
                if y_position < 150:
                    break
                pdf.drawString(70, y_position, str(condition)[:25])
                pdf.drawString(250, y_position, str(n_samples))
                samples_str = ", ".join(samples[:5])
                if len(samples) > 5:
                    samples_str += f" ... (+{len(samples) - 5} more)"
                pdf.drawString(400, y_position, samples_str[:35])
                y_position -= 18
        
        # Analyses included section
        y_position -= 20
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_position, "Analyses Included in This Report")
        y_position -= 25
        
        pdf.setFont("Helvetica", 10)
        analyses = []
        analyses.append("• Distribution Box Plots")
        if bqc_plot is not None:
            analyses.append("• BQC Quality Assessment")
        if retention_time_plot is not None:
            analyses.append("• Retention Time Analysis")
        if pca_plot is not None:
            analyses.append("• Principal Component Analysis (PCA)")
        if correlation_plots:
            analyses.append(f"• Pairwise Correlation ({len(correlation_plots)} condition(s))")
        if abundance_bar_charts and any(v is not None for v in abundance_bar_charts.values()):
            analyses.append("• Class Concentration Bar Charts")
        if abundance_pie_charts:
            analyses.append(f"• Class Concentration Pie Charts ({len(abundance_pie_charts)} condition(s))")
        if saturation_plots:
            analyses.append(f"• Saturation Profiles ({len(saturation_plots)} class(es))")
        if fach_plots:
            analyses.append(f"• Fatty Acid Composition Heatmaps ({len(fach_plots)} class(es))")
        if volcano_plots:
            analyses.append("• Volcano Plot Analysis")
        if pathway_visualization is not None:
            analyses.append("• Lipid Pathway Visualization")
        if heatmap_figs:
            analyses.append(f"• Lipidomic Heatmap(s) ({len(heatmap_figs)} type(s))")
        
        for analysis in analyses:
            if y_position < 80:
                break
            pdf.drawString(70, y_position, analysis)
            y_position -= 16
        
        # Footer
        pdf.setFont("Helvetica-Oblique", 9)
        pdf.drawCentredString(page_width / 2, 50, "Generated by LipidCruncher - The Farese and Walther Lab")
        pdf.drawCentredString(page_width / 2, 38, "https://github.com/FareseWaltherLab/LipidCruncher")
        
        # === Page 2: Box Plots ===
        pdf.showPage()
        pdf.setPageSize(letter)
        
        # Convert Plotly figures to PNG for the box plots
        box_plot1_bytes = pio.to_image(box_plot_fig1, format='png', width=800, height=600, scale=2)
        box_plot2_bytes = pio.to_image(box_plot_fig2, format='png', width=800, height=600, scale=2)
        
        # Add first box plot
        img1 = ImageReader(io.BytesIO(box_plot1_bytes))
        pdf.drawImage(img1, 50, 400, width=500, height=300, preserveAspectRatio=True)
        
        # Add second box plot
        img2 = ImageReader(io.BytesIO(box_plot2_bytes))
        pdf.drawImage(img2, 50, 50, width=500, height=300, preserveAspectRatio=True)
        
        # === BQC Plot (only if exists) ===
        if bqc_plot is not None:
            pdf.showPage()
            pdf.setPageSize(letter)
            bqc_bytes = pio.to_image(bqc_plot, format='png', width=800, height=600, scale=2)
            bqc_img = ImageReader(io.BytesIO(bqc_bytes))
            pdf.drawImage(bqc_img, 50, 100, width=500, height=400, preserveAspectRatio=True)
            pdf.drawString(50, 80, "BQC Quality Assessment")
        
        # === Retention Time Plot (only if exists) ===
        if retention_time_plot is not None:
            pdf.showPage()
            pdf.setPageSize(landscape(letter))
            rt_bytes = pio.to_image(retention_time_plot, format='png', width=1000, height=700, scale=2)
            rt_img = ImageReader(io.BytesIO(rt_bytes))
            pdf.drawImage(rt_img, 50, 50, width=700, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 30, "Retention Time vs. Mass Plot")
        
        # === PCA Plot (only if exists) ===
        if pca_plot is not None:
            pdf.showPage()
            pdf.setPageSize(landscape(letter))
            pca_bytes = pio.to_image(pca_plot, format='png', width=1000, height=700, scale=2)
            pca_img = ImageReader(io.BytesIO(pca_bytes))
            pdf.drawImage(pca_img, 50, 50, width=700, height=500, preserveAspectRatio=True)
            pdf.drawString(50, 30, "Principal Component Analysis (PCA)")
        
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
    """
    Add a heatmap figure to the PDF with proper scaling to prevent label cut-off.
    Optimized for heatmaps with many lipids (rows) and fewer samples (columns).
    
    Args:
        pdf: ReportLab canvas object
        heatmap_fig: Plotly figure object containing the heatmap
        title: String title for the heatmap
    """
    import copy
    
    pdf.showPage()
    pdf.setPageSize(letter)  # Portrait orientation - better for tall heatmaps
    page_width, page_height = letter
    
    # Create a copy of the figure to modify without affecting the original
    fig_copy = copy.deepcopy(heatmap_fig)
    
    # Update the figure layout: more left margin, larger y-axis font
    fig_copy.update_layout(
        margin=dict(l=180, r=100, t=60, b=60),
        yaxis=dict(tickfont=dict(size=11)),  # Larger, readable font
        xaxis=dict(tickfont=dict(size=10)),
    )
    
    # Export with tall aspect ratio (narrow and tall)
    heatmap_bytes = pio.to_image(
        fig_copy, 
        format='png', 
        width=900,    # Narrower
        height=1200,  # Taller - more room for lipid names
        scale=2
    )
    heatmap_img = ImageReader(io.BytesIO(heatmap_bytes))
    
    # Page margins: 30pt left/right, 50pt bottom, 40pt top
    available_width = page_width - 60
    available_height = page_height - 90
    
    # Maintain aspect ratio (900/1200 = 0.75)
    img_aspect = 900 / 1200
    page_aspect = available_width / available_height
    
    if img_aspect > page_aspect:
        img_width = available_width
        img_height = available_width / img_aspect
    else:
        img_height = available_height
        img_width = available_height * img_aspect
    
    # Center horizontally
    x_position = (page_width - img_width) / 2
    y_position = 50 + (available_height - img_height) / 2
    
    pdf.drawImage(heatmap_img, x_position, y_position, width=img_width, height=img_height, preserveAspectRatio=True)
    
    # Add title below the image
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(x_position, y_position - 20, title)

if __name__ == "__main__":
    main()