"""
LipidCruncher v2.0 - Refactored Main Application

Single-page app using only refactored architecture.
Matches old main_app.py UI exactly.

This application combines upload, configuration, data cleaning,
zero filtering, and normalization into a single unified workflow.

Author: Hamed Abdi / Claude
Date: November 2025
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Add src to path
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR / 'src'
IMAGES_DIR = SCRIPT_DIR / 'images'
sys.path.insert(0, str(SRC_DIR))

# Import refactored components
from lipidcruncher.core.models.experiment import ExperimentConfig
from lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService
from lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from lipidcruncher.core.services.zero_filtering_service import ZeroFilteringService
from lipidcruncher.workflows.normalization_workflow import NormalizationWorkflow

# Import sample grouping service (assumes it's in the same directory or installed)
from lipidcruncher.core.services.sample_grouping_service import SampleGroupingService

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    # Page control
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    
    # Data at each stage
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'intsta_df' not in st.session_state:
        st.session_state.intsta_df = None
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    if 'normalized_df' not in st.session_state:
        st.session_state.normalized_df = None
    
    # Configuration
    if 'experiment_config' not in st.session_state:
        st.session_state.experiment_config = None
    if 'format_type' not in st.session_state:
        st.session_state.format_type = None
    if 'grade_config' not in st.session_state:
        st.session_state.grade_config = None
    if 'bqc_label' not in st.session_state:
        st.session_state.bqc_label = None
    
    # Process control
    if 'confirmed' not in st.session_state:
        st.session_state.confirmed = False
    if 'removed_species' not in st.session_state:
        st.session_state.removed_species = []
    
    # Normalization settings
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = []
    if 'normalization_method' not in st.session_state:
        st.session_state.normalization_method = 'None'

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="LipidCruncher v2.0",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    if st.session_state.page == 'landing':
        display_landing_page()
    elif st.session_state.page == 'app':
        display_main_app()

# ============================================================================
# LANDING PAGE
# ============================================================================

def display_landing_page():
    """Display the LipidCruncher landing page."""
    # Load and display logo
    try:
        logo_path = IMAGES_DIR / 'logo.tif'
        if logo_path.exists():
            logo = Image.open(logo_path)
            st.image(logo, width=720)
        else:
            st.header("LipidCruncher v2.0")
    except Exception:
        st.header("LipidCruncher v2.0")
    
    st.markdown("""
    **LipidCruncher** is an open-source web platform developed by **[The Farese and Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther)** 
    to transform lipidomic data analysis. Designed to overcome traditional challenges like manual spreadsheet handling and 
    insufficient quality assessment, LipidCruncher offers a comprehensive solution that streamlines data standardization, 
    normalization, and rigorous quality control while providing powerful visualization tools.
    """)
    
    st.markdown("---")
    
    # Module selection
    st.subheader("📊 Select Analysis Module")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 🧹 Data Cleaning, Filtering, & Normalization
        
        **Comprehensive data processing pipeline:**
        - Upload data from multiple formats (LipidSearch, Generic CSV, Metabolomics Workbench)
        - Configure your experiment (conditions, samples, replicates)
        - Clean and filter data based on quality metrics
        - Normalize using internal standards and/or protein concentrations
        - Export processed data for downstream analysis
        
        **Best for:** Initial data processing and preparation
        """)
        
        if st.button("▶️ Start Data Processing", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.experimental_rerun()
    
    with col2:
        st.warning("""
        ### 📈 Quality Check & Visualization
        
        **Advanced analysis and visualization:**
        - Volcano plots for differential analysis
        - PCA and hierarchical clustering
        - Abundance charts and heatmaps
        - Pathway analysis and enrichment
        - Statistical testing and comparisons
        
        **Status:** Coming in Phase 3B (visualization refactoring)
        
        *Note: This module is being refactored to use the new architecture.*
        """)
        
        st.button("📊 Visualization Module", disabled=True, use_container_width=True)
    
    st.markdown("---")
    st.caption("LipidCruncher v2.0 | Refactored Architecture | November 2025")

# ============================================================================
# MAIN APPLICATION PAGE
# ============================================================================

def display_main_app():
    """Display main application page with full pipeline."""
    # Display logo
    try:
        logo_path = IMAGES_DIR / 'logo.tif'
        if logo_path.exists():
            logo = Image.open(logo_path)
            st.image(logo, width=720)
        else:
            st.header("LipidCruncher v2.0")
    except Exception:
        st.header("LipidCruncher v2.0")
    
    st.markdown("Process, analyze and visualize lipidomics data from multiple sources.")
    
    # Sidebar: Configuration and upload
    render_sidebar()
    
    # Main page: Sequential data pipeline (only shown after confirmation)
    if st.session_state.confirmed:
        render_data_pipeline()
    else:
        # Landing page with format requirements
        st.markdown("## 📋 Data Format Requirements")
        st.markdown("Please review the requirements for your data format before uploading.")
        
        # Show format requirements based on selected format
        if st.session_state.format_type:
            display_format_requirements_main(st.session_state.format_type)
        
        st.info("👈 Please configure your experiment in the sidebar and confirm your inputs to proceed.")
    
    # Back to home button
    st.markdown("---")
    if st.button("🏠 Back to Home"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.session_state.page = 'landing'
        st.experimental_rerun()

# ============================================================================
# SIDEBAR: FORMAT SELECTION, UPLOAD, AND CONFIGURATION
# ============================================================================

def render_sidebar():
    """Render sidebar with format selection, upload, and experiment configuration."""
    st.sidebar.header("📁 Data Upload")
    
    # Format selection
    data_format = st.sidebar.selectbox(
        'Select Data Format',
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0']
    )
    
    st.session_state.format_type = data_format
    
    # Display format requirements
    display_format_requirements(data_format)
    
    # File uploader
    file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
    
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset',
        type=file_types
    )
    
    if uploaded_file:
        # Read and store raw data
        try:
            if data_format == 'Metabolomics Workbench':
                # Read as text to preserve structure
                raw_text = uploaded_file.read().decode('utf-8')
                st.session_state.raw_df = raw_text
                df = raw_text
                st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
                
                # Extract conditions immediately for auto-config
                st.session_state.metabolomics_conditions = extract_metabolomics_conditions(raw_text)
            else:
                # Read as DataFrame
                df = pd.read_csv(uploaded_file, sep=',' if uploaded_file.name.endswith('.csv') else '\t')
                st.session_state.raw_df = df.copy()
                st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
                st.sidebar.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.session_state.metabolomics_conditions = None
            
            # Experiment configuration
            render_experiment_config(df, data_format)
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

def display_format_requirements(data_format: str):
    """Display format requirements in an expander in sidebar."""
    with st.sidebar.expander("📋 Format Requirements", expanded=False):
        if data_format == 'Metabolomics Workbench':
            st.info("""
            **Dataset Requirements for Metabolomics Workbench Format**
            
            The file must be a CSV containing:
            1. Required section markers:
               * MS_METABOLITE_DATA_START
               * MS_METABOLITE_DATA_END
               
            2. Three essential rows in the data section:
               * Row 1: Sample names
               * Row 2: Experimental conditions
               * Row 3+: Lipid measurements
            """)
        elif data_format == 'LipidSearch 5.0':
            st.info("""
            **Dataset Requirements for LipidSearch 5.0**
            
            Required columns:
            * `LipidMolec`: Lipid molecule identifier
            * `ClassKey`: Lipid class classification
            * `CalcMass`: Calculated mass
            * `BaseRt`: Base retention time
            * `TotalGrade`: Quality grade (A, B, C, D)
            * `TotalSmpIDRate(%)`: Sample identification rate
            * `FAKey`: Fatty acid key
            * `MeanArea[s1]`, `MeanArea[s2]`, etc.: Sample intensities
            """)
        else:  # Generic Format
            st.info("""
            **Dataset Requirements for Generic Format**
            
            Your dataset must contain ONLY these columns:
            
            1. **First Column - Lipid Names:**
               * Can have any column name
               * Must contain lipid molecule identifiers
               
            2. **Remaining Columns - Intensity Values Only:**
               * Each column = one sample
               * Number of columns must match total number of samples
               * Will be standardized to: intensity[s1], intensity[s2], ...
            
            ⚠️ Remove any additional columns before uploading.
            """)

def display_format_requirements_main(data_format: str):
    """Display format requirements on main page (not in expander)."""
    if data_format == 'Metabolomics Workbench':
        st.markdown("### Metabolomics Workbench Format")
        st.info("""
        **Dataset Requirements:**
        
        The file must be a CSV containing:
        1. **Required section markers:**
           * `MS_METABOLITE_DATA_START`
           * `MS_METABOLITE_DATA_END`
           
        2. **Three essential rows in the data section:**
           * Row 1: Sample names
           * Row 2: Experimental conditions (one per sample)
           * Row 3+: Lipid measurements with lipid names in first column
           
        **Example:**
        ```
        MS_METABOLITE_DATA_START
        Samples,Sample1,Sample2,Sample3,Sample4
        Factors,WT,WT,KO,KO
        LPC(16:0),234.5,256.7,189.3,201.4
        PE(18:0_20:4),456.7,478.2,390.1,405.6
        MS_METABOLITE_DATA_END
        ```
        """)
    elif data_format == 'LipidSearch 5.0':
        st.markdown("### LipidSearch 5.0 Format")
        st.info("""
        **Required Columns:**
        * `LipidMolec`: Lipid molecule identifier
        * `ClassKey`: Lipid class classification
        * `CalcMass`: Calculated mass
        * `BaseRt`: Base retention time
        * `TotalGrade`: Quality grade (A, B, C, D)
        * `TotalSmpIDRate(%)`: Sample identification rate
        * `FAKey`: Fatty acid key
        * `MeanArea[s1]`, `MeanArea[s2]`, etc.: Sample intensities
        
        **Note:** Each sample must have a corresponding MeanArea column.
        For 10 samples, you need: MeanArea[s1], MeanArea[s2], ..., MeanArea[s10]
        """)
    else:  # Generic Format
        st.markdown("### Generic Format")
        st.info("""
        **Dataset Requirements:**
        
        Your dataset must contain ONLY these columns in this order:
        
        1. **First Column - Lipid Names:**
           * Can have any column name
           * Must contain lipid molecule identifiers
           * Will be standardized. Examples:
             - `LPC O-18:1` → `LPC(O-18:1)`
             - `Cer d18:0/C24:0` → `Cer(d18:0_C24:0)`
             - `CE 14:0;0` → `CE(14:0)`
             
        2. **Remaining Columns - Intensity Values Only:**
           * Each column = one sample
           * Can have any column names
           * Number of columns must match total number of samples
           * Will be standardized to: `intensity[s1]`, `intensity[s2]`, ...
        
        ⚠️ **Important:** Remove any additional columns before uploading.
        Only lipid names column + intensity columns should be present.
        
        **Note:** Lipid class (e.g., LPC, Cer, CE) will be automatically extracted from lipid names.
        """)

def render_experiment_config(df, data_format: str):
    """Render experiment configuration section in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Experiment Setup")
    
    # Check for Metabolomics Workbench auto-detected conditions
    if data_format == 'Metabolomics Workbench' and 'metabolomics_conditions' in st.session_state and st.session_state.metabolomics_conditions:
        conditions_map = st.session_state.metabolomics_conditions
        
        # Get unique conditions in order of appearance
        conditions_in_order = [conditions_map[f's{i+1}'] for i in range(len(conditions_map))]
        unique_conditions = []
        seen = set()
        for condition in conditions_in_order:
            if condition not in seen:
                seen.add(condition)
                unique_conditions.append(condition)
        
        use_detected = st.sidebar.checkbox("Use detected experimental setup", value=True, key="use_detected_metabolomics")
        
        if use_detected:
            # Auto-populate from detected conditions
            n_conditions = len(unique_conditions)
            conditions_list = unique_conditions
            
            # Count samples per condition
            sample_counts = {}
            for condition in conditions_list:
                count = sum(1 for x in conditions_map.values() if x == condition)
                sample_counts[condition] = count
            
            number_of_samples_list = [sample_counts[condition] for condition in conditions_list]
            
            # Display the detected setup
            st.sidebar.success("✅ Using detected setup:")
            for cond, count in zip(conditions_list, number_of_samples_list):
                st.sidebar.write(f"• {cond}: {count} samples")
        else:
            # Manual configuration
            n_conditions, conditions_list, number_of_samples_list = manual_experiment_config()
    else:
        # Manual configuration for other formats
        n_conditions, conditions_list, number_of_samples_list = manual_experiment_config()
    
    # Validate inputs
    if not all(conditions_list):
        st.sidebar.error("⚠️ All condition labels must be non-empty.")
        return
    
    # Create ExperimentConfig
    try:
        experiment_config = ExperimentConfig(
            n_conditions=n_conditions,
            conditions_list=conditions_list,
            number_of_samples_list=number_of_samples_list
        )
        st.session_state.experiment_config = experiment_config
        
        # Group samples section
        render_group_samples(df, experiment_config, data_format)
        
        # BQC specification
        render_bqc_specification(experiment_config)
        
        # Confirmation
        render_confirmation(experiment_config)
        
    except Exception as e:
        st.sidebar.error(f"❌ Error creating experiment configuration: {str(e)}")

def manual_experiment_config():
    """Manual experiment configuration UI."""
    n_conditions = st.sidebar.number_input(
        'Enter the number of conditions',
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )
    
    conditions_list = []
    number_of_samples_list = []
    
    for i in range(n_conditions):
        conditions_list.append(
            st.sidebar.text_input(
                f'Label for condition #{i + 1}',
                key=f'condition_{i}',
                value=f'Cond{i+1}'
            )
        )
        number_of_samples_list.append(
            st.sidebar.number_input(
                f'Number of samples for condition #{i + 1}',
                min_value=1,
                max_value=1000,
                value=3,
                step=1,
                key=f'n_samples_{i}'
            )
        )
    
    return n_conditions, conditions_list, number_of_samples_list

def render_group_samples(df, experiment_config: ExperimentConfig, data_format: str):
    """Render group samples section."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👥 Group Samples")
    
    # Build initial grouping
    group_df = pd.DataFrame({
        'sample name': experiment_config.full_samples_list,
        'condition': experiment_config.extensive_conditions_list
    })
    
    st.sidebar.write(group_df)
    
    # Ask if samples are properly grouped
    st.sidebar.write("Are your samples properly grouped together?")
    grouping_answer = st.sidebar.radio(
        'Grouping Status',
        ['Yes', 'No'],
        index=0,
        key='grouping_radio',
        label_visibility='collapsed'
    )
    
    # Handle manual regrouping
    if grouping_answer == 'No':
        render_manual_grouping(df, experiment_config, data_format)

def render_manual_grouping(df, experiment_config: ExperimentConfig, data_format: str):
    """Handle manual sample grouping with service."""
    st.sidebar.write("**Manual Sample Grouping**")
    st.sidebar.info("Select which samples belong to each condition.")
    
    # Need preprocessing first to get actual sample names
    if st.session_state.preprocessed_df is None:
        preprocessing_service = FormatPreprocessingService()
        format_mapping = {
            'LipidSearch 5.0': 'lipidsearch',
            'Generic Format': 'generic',
            'Metabolomics Workbench': 'metabolomics_workbench'
        }
        format_type = format_mapping[data_format]
        
        temp_preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(
            df,
            format_type=format_type
        )
        
        if not success:
            st.sidebar.error(f"❌ {message}")
            return
    else:
        temp_preprocessed_df = st.session_state.preprocessed_df
    
    # Extract sample names using service
    grouping_service = SampleGroupingService()
    available_samples = grouping_service.extract_sample_names_from_dataframe(temp_preprocessed_df)
    
    # Let user select samples for each condition
    selections = {}
    remaining_samples = available_samples.copy()
    
    for condition, n_samples in zip(
        experiment_config.conditions_list,
        experiment_config.number_of_samples_list
    ):
        st.sidebar.write(f"**Select {n_samples} samples for {condition}:**")
        
        selected = st.sidebar.multiselect(
            f'Samples for {condition}',
            options=remaining_samples,
            key=f'manual_group_{condition}'
        )
        
        selections[condition] = selected
        
        # Show warning if wrong number selected
        if len(selected) != n_samples:
            st.sidebar.warning(f"⚠️ Please select exactly {n_samples} samples")
        else:
            # Remove selected samples from remaining
            remaining_samples = [s for s in remaining_samples if s not in selected]
    
    # Validate selections using service
    valid, error_message = grouping_service.validate_selections(
        selections,
        experiment_config.conditions_list,
        experiment_config.number_of_samples_list
    )
    
    if not valid:
        st.sidebar.warning(f"⚠️ {error_message}")
        return
    
    # Reorder DataFrame using service
    try:
        reordered_df = grouping_service.reorder_dataframe(
            temp_preprocessed_df,
            selections,
            experiment_config.conditions_list
        )
        
        # Store reordered data
        st.session_state.preprocessed_df = reordered_df
        
        # Show new sample order
        name_mapping = grouping_service.create_name_mapping(
            selections,
            experiment_config.conditions_list
        )
        
        st.sidebar.write("**New sample order:**")
        st.sidebar.dataframe(name_mapping)
        
    except Exception as e:
        st.sidebar.error(f"❌ Error reordering: {str(e)}")

def render_bqc_specification(experiment_config: ExperimentConfig):
    """Render BQC specification section."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Batch Quality Control (BQC)")
    
    bqc_ans = st.sidebar.radio(
        'Do you have BQC samples?',
        ['Yes', 'No'],
        index=1,
        key='bqc_radio'
    )
    
    if bqc_ans == 'Yes':
        # Only show conditions with 2+ samples
        conditions_with_multiple = [
            condition for condition, n_samples
            in zip(experiment_config.conditions_list, experiment_config.number_of_samples_list)
            if n_samples > 1
        ]
        
        if conditions_with_multiple:
            bqc_label = st.sidebar.radio(
                'Which label corresponds to BQC samples?',
                conditions_with_multiple,
                index=0,
                key='bqc_label_radio'
            )
            st.session_state.bqc_label = bqc_label
        else:
            st.sidebar.warning("⚠️ No conditions with 2+ samples available for BQC")
            st.session_state.bqc_label = None
    else:
        st.session_state.bqc_label = None

def render_confirmation(experiment_config: ExperimentConfig):
    """Render confirmation section."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Confirm Inputs")
    
    # Show summary
    total_samples = len(experiment_config.full_samples_list)
    st.sidebar.write(f"**Total samples:** {total_samples}")
    
    for i, (condition, n_samples) in enumerate(zip(
        experiment_config.conditions_list,
        experiment_config.number_of_samples_list
    )):
        samples = experiment_config.individual_samples_list[i]
        if len(samples) > 5:
            st.sidebar.write(f"• {samples[0]} to {samples[-1]} ({len(samples)} samples) → {condition}")
        else:
            st.sidebar.write(f"• {'-'.join(samples)} → {condition}")
    
    if st.session_state.bqc_label:
        st.sidebar.write(f"**BQC Condition:** {st.session_state.bqc_label}")
    
    confirmed = st.sidebar.checkbox(
        "✅ Confirm the inputs by checking this box",
        key='confirm_checkbox'
    )
    st.session_state.confirmed = confirmed

# ============================================================================
# MAIN PIPELINE: DATA PROCESSING
# ============================================================================

def render_data_pipeline():
    """Render the main data processing pipeline sections."""
    # Preprocess data (if not already done during grouping)
    if st.session_state.preprocessed_df is None:
        preprocess_data()
    
    # Show configuration summary
    display_configuration_summary()
    
    # Section 1: Data Cleaning and Zero Filtering (includes grade filtering for LipidSearch)
    render_data_cleaning()
    
    # Section 2: Internal Standards Management
    if st.session_state.intsta_df is not None and not st.session_state.intsta_df.empty:
        render_standards_management()
    
    # Section 3: Normalization (always visible, no expander)
    render_normalization()

def extract_metabolomics_conditions(text_data: str) -> dict:
    """Extract conditions mapping from Metabolomics Workbench format."""
    try:
        # Extract data section
        start_idx = text_data.find('MS_METABOLITE_DATA_START')
        end_idx = text_data.find('MS_METABOLITE_DATA_END')
        data_section = text_data[start_idx:end_idx].strip()
        data_lines = [line.strip() for line in data_section.split('\n') if line.strip()]
        
        # Get samples (row 2) and factors (row 3)
        samples_line = data_lines[1]
        factors_line = data_lines[2]
        
        sample_names = [s.strip() for s in samples_line.split(',')[1:]]  # Skip "Samples" header
        conditions = [c.strip() for c in factors_line.split(',')[1:]]  # Skip "Factors" header
        
        # Create mapping {s1: condition, s2: condition, ...}
        conditions_map = {f's{i+1}': conditions[i] for i in range(len(sample_names))}
        
        return conditions_map
    except Exception as e:
        return {}

def preprocess_data():
    """Preprocess uploaded data."""
    if st.session_state.raw_df is None:
        st.error("❌ No data uploaded")
        return
    
    preprocessing_service = FormatPreprocessingService()
    
    # Map UI format names to service format types
    format_mapping = {
        'LipidSearch 5.0': 'lipidsearch',
        'Generic Format': 'generic',
        'Metabolomics Workbench': 'metabolomics_workbench'
    }
    
    format_type = format_mapping[st.session_state.format_type]
    
    with st.spinner("Preprocessing data..."):
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(
            st.session_state.raw_df,
            format_type=format_type
        )
    
    if not success:
        st.error(f"❌ Preprocessing failed: {message}")
        st.stop()
    
    st.success(f"✅ {message}")
    st.session_state.preprocessed_df = preprocessed_df
    
    # Check total samples match
    intensity_cols = [col for col in preprocessed_df.columns if col.startswith('intensity[')]
    total_samples_actual = len(intensity_cols)
    total_samples_expected = len(st.session_state.experiment_config.full_samples_list)
    
    if total_samples_expected != total_samples_actual:
        st.error(
            f"⚠️ Mismatch: You specified {total_samples_expected} total samples, "
            f"but the data has {total_samples_actual} samples. "
            f"Please adjust your experiment configuration."
        )
        st.stop()

def display_configuration_summary():
    """Display configuration summary."""
    with st.expander("📊 Configuration Summary", expanded=False):
        config = st.session_state.experiment_config
        st.write(f"**Format:** {st.session_state.format_type}")
        st.write(f"**Total samples:** {len(config.full_samples_list)}")
        st.write(f"**Conditions:** {', '.join(config.conditions_list)}")
        
        for i, (condition, n_samples) in enumerate(zip(
            config.conditions_list,
            config.number_of_samples_list
        )):
            samples = config.individual_samples_list[i]
            if len(samples) > 5:
                st.write(f"- **{condition}:** {samples[0]} to {samples[-1]} ({len(samples)} samples)")
            else:
                st.write(f"- **{condition}:** {'-'.join(samples)}")
        
        if st.session_state.bqc_label:
            st.write(f"**BQC Condition:** {st.session_state.bqc_label}")

def render_data_cleaning():
    """Render data cleaning and zero filtering section."""
    st.markdown("---")
    st.subheader("🧹 Data Cleaning and Filtering")
    
    with st.expander("Clean and Filter Data", expanded=True):
        # Grade filtering UI (LipidSearch only) - MUST be configured before cleaning
        if st.session_state.format_type == 'LipidSearch 5.0':
            st.markdown("### 🎯 Grade Filtering (LipidSearch Only)")
            st.markdown("""
            LipidSearch assigns quality grades (A, B, C, D) to each lipid identification:
            - **Grade A**: Highest confidence
            - **Grade B**: Good quality
            - **Grade C**: Lower confidence
            - **Grade D**: Lowest confidence
            
            **Default:** Accept A/B for all classes, plus C for LPC and SM.
            """)
            
            if 'ClassKey' in st.session_state.preprocessed_df.columns:
                # Get unique classes
                all_classes = sorted(st.session_state.preprocessed_df['ClassKey'].unique())
                st.write(f"**Lipid classes found:** {len(all_classes)}")
                
                # Option to use default or custom
                use_custom = st.radio(
                    "Grade filtering mode:",
                    ["Use Default Settings", "Customize by Class"],
                    index=0,
                    key="grade_filter_mode"
                )
                
                if use_custom == "Use Default Settings":
                    st.success("✅ Using default grade filtering.")
                    st.session_state.grade_config = None
                else:
                    # Custom settings
                    st.markdown("#### 📋 Select Acceptable Grades by Lipid Class")
                    grade_config = {}
                    
                    for lipid_class in all_classes:
                        # Default grades
                        if lipid_class in ['LPC', 'SM']:
                            default_grades = ['A', 'B', 'C']
                        else:
                            default_grades = ['A', 'B']
                        
                        selected_grades = st.multiselect(
                            f"Acceptable grades for **{lipid_class}**:",
                            options=['A', 'B', 'C', 'D'],
                            default=default_grades,
                            key=f"grade_select_{lipid_class}"
                        )
                        
                        if not selected_grades:
                            st.warning(f"⚠️ No grades selected for {lipid_class} - will be excluded!")
                        
                        grade_config[lipid_class] = selected_grades
                    
                    st.session_state.grade_config = grade_config
            else:
                st.warning("⚠️ ClassKey column missing - skipping grade filtering")
            
            st.markdown("---")  # Separator between grade filtering and data cleaning
        
        # Auto-clean if not already done
        if st.session_state.cleaned_df is None:
            clean_data_only()
        
        # Display cleaned data
        if st.session_state.cleaned_df is not None:
            # Zero filtering UI (always show)
            st.markdown("### 🔍 Zero Filtering")
            st.markdown("""
            Remove lipid species based on zero/near-zero value prevalence:
            - **BQC condition:** Remove if ≥50% of replicates are below threshold
            - **Non-BQC conditions:** Remove if ≥75% of replicates in ALL conditions are below threshold
            """)
            
            # Set default threshold based on format
            default_threshold = 30000.0 if st.session_state.format_type == 'LipidSearch 5.0' else 0.0
            step_size = 1000.0 if st.session_state.format_type == 'LipidSearch 5.0' else 0.1
            
            threshold = st.number_input(
                "Zero filter threshold:",
                min_value=0.0,
                value=default_threshold,
                step=step_size,
                help="Values ≤ this are considered zero",
                key="zero_filter_threshold"
            )
            
            # Apply zero filtering whenever threshold changes
            apply_zero_filtering_simple(st.session_state.cleaned_df, threshold)
            
            # Display cleaning summary messages (after zero filtering, before data)
            if st.session_state.intsta_df is not None and not st.session_state.intsta_df.empty:
                st.success(f"✅ Extracted {len(st.session_state.intsta_df)} internal standards")
            st.success(f"✅ Data cleaned: {st.session_state.cleaned_df.shape[0]} lipids")
            
            # Display results
            if st.session_state.filtered_df is not None:
                display_cleaning_results()

def clean_data_only():
    """Clean data without zero filtering."""
    cleaning_service = DataCleaningService()
    
    with st.spinner("Cleaning data..."):
        try:
            if st.session_state.format_type == 'LipidSearch 5.0':
                cleaned_df = cleaning_service.clean_lipidsearch_data(
                    st.session_state.preprocessed_df,
                    st.session_state.experiment_config,
                    grade_config=st.session_state.grade_config
                )
            else:
                cleaned_df = cleaning_service.clean_generic_data(
                    st.session_state.preprocessed_df,
                    st.session_state.experiment_config
                )
            
            # Extract internal standards
            cleaned_df, intsta_df = cleaning_service.extract_internal_standards(cleaned_df)
            
            st.session_state.cleaned_df = cleaned_df
            st.session_state.intsta_df = intsta_df
            # Save original auto-detected standards for later restoration
            st.session_state.original_intsta_df = intsta_df.copy()
            
        except Exception as e:
            st.error(f"❌ Error during cleaning: {str(e)}")

def apply_zero_filtering_simple(cleaned_df: pd.DataFrame, threshold: float):
    """Apply zero filtering with given threshold."""
    zero_service = ZeroFilteringService()
    
    filtered_df, removed_lipids = zero_service.filter_by_zeros(
        df=cleaned_df,
        experiment_config=st.session_state.experiment_config,
        threshold=threshold,
        bqc_label=st.session_state.bqc_label
    )
    
    st.session_state.filtered_df = filtered_df
    st.session_state.removed_species = removed_lipids
    
    removed_count = len(removed_lipids)
    if removed_count > 0:
        removal_pct = (removed_count / len(cleaned_df)) * 100
        st.info(f"Removed {removed_count} species ({removal_pct:.1f}%)")
    else:
        st.success("✅ No species removed by zero filter")

def display_cleaning_results():
    """Display cleaning and filtering results."""
    st.markdown("### 📊 Final Cleaned and Filtered Data")
    st.write(f"**Shape:** {st.session_state.filtered_df.shape}")
    
    # Show data
    st.dataframe(st.session_state.filtered_df.head(20))
    
    # Download button
    csv = st.session_state.filtered_df.to_csv(index=False)
    st.download_button(
        label="💾 Download Filtered Data",
        data=csv,
        file_name="cleaned_filtered_data.csv",
        mime="text/csv",
        key="download_filtered"
    )
    
    # Show removed species if any
    if st.session_state.removed_species:
        st.markdown(f"**🗑️ Removed Species ({len(st.session_state.removed_species)}):**")
        removed_df = pd.DataFrame({'Removed LipidMolec': st.session_state.removed_species})
        st.dataframe(removed_df)

def render_standards_management():
    """Render internal standards management section."""
    with st.expander("🔬 Manage Internal Standards", expanded=False):
        # Initialize preference if not exists
        if 'standard_source_preference' not in st.session_state:
            st.session_state.standard_source_preference = "Automatic Detection"
        
        # Radio button for standards source
        standards_source = st.radio(
            "Select standards source:",
            ["Automatic Detection", "Upload Custom Standards"],
            key="standards_source_radio"
        )
        
        # Save the preference
        st.session_state.standard_source_preference = standards_source
        
        # Handle automatic detection
        if standards_source == "Automatic Detection":
            # Clear any custom standards when switching to automatic detection
            if 'custom_intsta_df' in st.session_state:
                st.session_state.custom_intsta_df = None
            
            # Restore original auto-detected standards
            if 'original_intsta_df' in st.session_state:
                st.session_state.intsta_df = st.session_state.original_intsta_df.copy()
            
            # Show automatically detected standards
            st.markdown("### 📊 Automatically Detected Standards")
            if not st.session_state.intsta_df.empty:
                st.success(f"✅ Found {len(st.session_state.intsta_df)} standards")
                st.dataframe(st.session_state.intsta_df)
                
                # Download button
                csv = st.session_state.intsta_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Internal Standards",
                    data=csv,
                    file_name="internal_standards.csv",
                    mime="text/csv",
                    key="download_standards_auto"
                )
            else:
                st.info("No internal standards were automatically detected in the dataset")
        
        # Handle custom upload
        else:
            st.markdown("### 📤 Upload Custom Standards")
            st.info("""
            You can upload your own standards file. Requirements:
            - CSV file (.csv)
            - Must contain column: 'LipidMolec'
            - The standards must exist in your dataset
            """)
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with standards", 
                type=['csv'],
                key="standards_file_uploader"
            )
            
            if uploaded_file is not None:
                from lipidcruncher.core.services.standards_service import StandardsService
                
                try:
                    standards_df = pd.read_csv(uploaded_file)
                    
                    # Process the uploaded standards
                    service = StandardsService()
                    new_standards_df = service.process_standards_file(
                        standards_df,
                        st.session_state.cleaned_df,
                        st.session_state.intsta_df
                    )
                    
                    if new_standards_df is not None:
                        # Store custom standards separately
                        st.session_state.custom_intsta_df = new_standards_df.copy()
                        # Update current standards
                        st.session_state.intsta_df = new_standards_df
                        
                        st.success(f"✅ Successfully loaded {len(new_standards_df)} custom standards")
                        st.markdown("### 📊 Loaded Custom Standards")
                        st.dataframe(new_standards_df)
                        
                        # Download button
                        csv = new_standards_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Custom Standards",
                            data=csv,
                            file_name="custom_standards.csv",
                            mime="text/csv",
                            key="download_standards_custom"
                        )
                    
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Error reading standards file: {str(e)}")
            
            elif 'custom_intsta_df' in st.session_state and st.session_state.custom_intsta_df is not None:
                # Show previously uploaded standards
                st.success(f"✅ Using previously uploaded standards ({len(st.session_state.custom_intsta_df)} standards)")
                st.markdown("### 📊 Current Custom Standards")
                st.dataframe(st.session_state.custom_intsta_df)
                
                # Use previously uploaded standards
                st.session_state.intsta_df = st.session_state.custom_intsta_df.copy()
                
                # Download button
                csv = st.session_state.custom_intsta_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Custom Standards",
                    data=csv,
                    file_name="custom_standards.csv",
                    mime="text/csv",
                    key="download_standards_custom_prev"
                )

def render_normalization():
    """Render normalization section (always visible, no expander)."""
    st.markdown("---")
    st.subheader("⚖️ Data Normalization")
    
    if st.session_state.filtered_df is None:
        st.info("ℹ️ Please complete data cleaning first.")
        return
    
    # About normalization
    with st.expander("About Normalization Methods", expanded=False):
        st.markdown("""
        **None**: Use raw intensity values without normalization.
        
        **Internal Standards**: Normalize using spiked-in internal standards of known concentration.
```
        Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard
```
        
        **Protein-based**: Normalize against protein concentration (e.g., BCA assay).
```
        Concentration = Intensity_lipid / Protein_concentration
```
        
        **Both**: Apply both internal standards and protein normalization.
```
        Concentration = (Intensity_lipid / Intensity_standard) × (Concentration_standard / Protein_concentration)
```
        """)
    
    # Lipid class selection
    st.markdown("### 🎯 Select Lipid Classes")
    
    if 'ClassKey' not in st.session_state.filtered_df.columns:
        st.error("❌ ClassKey column required for normalization")
        return
    
    all_classes = sorted(st.session_state.filtered_df['ClassKey'].unique())
    
    # Initialize or reinitialize if empty
    if 'class_selector' not in st.session_state or not st.session_state.class_selector:
        st.session_state.class_selector = all_classes.copy()
    
    # Filter to only classes present in current data
    valid_classes = [c for c in st.session_state.class_selector if c in all_classes]
    if len(valid_classes) != len(st.session_state.class_selector):
        st.session_state.class_selector = valid_classes if valid_classes else all_classes.copy()
    
    selected_classes = st.multiselect(
        'Select lipid classes to analyze:',
        options=all_classes,
        key='class_selector'
    )
    
    if not selected_classes:
        st.warning("⚠️ Please select at least one lipid class.")
        return
    
    st.session_state.selected_classes = selected_classes
    
    # Filter dataframe to selected classes
    filtered_for_norm = st.session_state.filtered_df[
        st.session_state.filtered_df['ClassKey'].isin(selected_classes)
    ].copy()
    
    st.info(f"Selected {len(selected_classes)} classes with {len(filtered_for_norm)} lipid species")
    
    # Normalization method selection
    st.markdown("### ⚖️ Normalization Method")
    
    # Determine available methods
    has_standards = (st.session_state.intsta_df is not None and 
                    not st.session_state.intsta_df.empty)
    
    if has_standards:
        normalization_options = ['None', 'Internal Standards', 'Protein-based', 'Both']
        st.success(f"✅ Internal standards available ({len(st.session_state.intsta_df)} species)")
    else:
        normalization_options = ['None', 'Protein-based']
        st.info("ℹ️ No internal standards detected. Only 'None' and 'Protein-based' available.")
    
    # Method selection
    current_index = 0
    if st.session_state.normalization_method in normalization_options:
        current_index = normalization_options.index(st.session_state.normalization_method)
    
    normalization_method = st.radio(
        "Select normalization method:",
        options=normalization_options,
        index=current_index,
        key='norm_method_radio'
    )
    
    st.session_state.normalization_method = normalization_method
    
    # Perform normalization based on method
    if normalization_method == 'None':
        # Just rename columns
        normalized_df = filtered_for_norm.copy()
        intensity_cols = [col for col in normalized_df.columns if col.startswith('intensity[')]
        for col in intensity_cols:
            new_col = col.replace('intensity[', 'concentration[')
            normalized_df.rename(columns={col: new_col}, inplace=True)
        st.session_state.normalized_df = normalized_df
        display_normalization_results()
        
    elif normalization_method == 'Internal Standards':
        # Collect standards mapping
        normalized_df = collect_and_apply_standards(filtered_for_norm, st.session_state.intsta_df)
        if normalized_df is not None:
            st.session_state.normalized_df = normalized_df
            display_normalization_results()
            
    elif normalization_method == 'Protein-based':
        # Collect protein concentrations
        normalized_df = collect_and_apply_protein(filtered_for_norm)
        if normalized_df is not None:
            st.session_state.normalized_df = normalized_df
            display_normalization_results()
            
    elif normalization_method == 'Both':
        # Collect both
        normalized_df = collect_and_apply_both(filtered_for_norm, st.session_state.intsta_df)
        if normalized_df is not None:
            st.session_state.normalized_df = normalized_df
            display_normalization_results()

def collect_and_apply_standards(df: pd.DataFrame, intsta_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Collect internal standards mapping and apply normalization."""
    from lipidcruncher.core.services.normalization_service import NormalizationService
    from lipidcruncher.core.models.normalization import NormalizationConfig
    
    with st.expander("📋 Configure Internal Standards Normalization", expanded=True):
        st.markdown("Map each lipid class to its internal standard:")
        
        # Get unique classes in data and standards
        lipid_classes = sorted(df['ClassKey'].unique())
        available_standards = intsta_df['LipidMolec'].tolist()
        
        # Collect mapping
        standards_mapping = {}
        concentrations = {}
        
        for lipid_class in lipid_classes:
            col1, col2 = st.columns(2)
            
            with col1:
                standard = st.selectbox(
                    f"Standard for **{lipid_class}**:",
                    options=['None'] + available_standards,
                    key=f'standard_{lipid_class}'
                )
                standards_mapping[lipid_class] = standard if standard != 'None' else None
            
            with col2:
                if standard != 'None':
                    conc = st.number_input(
                        f"Concentration (pmol):",
                        min_value=0.0,
                        value=100.0,
                        step=10.0,
                        key=f'conc_{lipid_class}'
                    )
                    concentrations[standard] = conc
        
        # Check if all classes have standards
        missing = [c for c in lipid_classes if not standards_mapping.get(c)]
        if missing:
            st.warning(f"⚠️ No standards selected for: {', '.join(missing)}")
            return None
    
    # Create NormalizationConfig
    try:
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=lipid_classes,
            internal_standards=standards_mapping,
            intsta_concentrations=concentrations
        )
        
        service = NormalizationService()
        normalized_df = service.normalize_by_internal_standards(
            df=df,
            config=config,
            experiment=st.session_state.experiment_config,
            intsta_df=intsta_df
        )
        st.success("✅ Normalization by internal standards complete!")
        return normalized_df
    except Exception as e:
        st.error(f"❌ Normalization failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def collect_and_apply_protein(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Collect protein concentrations and apply normalization."""
    from lipidcruncher.core.services.normalization_service import NormalizationService
    from lipidcruncher.core.models.normalization import NormalizationConfig
    
    with st.expander("📋 Configure Protein Normalization", expanded=True):
        st.markdown("Enter protein concentration for each sample (e.g., mg/mL):")
        
        # Collect protein concentrations
        protein_data = []
        
        for sample in st.session_state.experiment_config.full_samples_list:
            conc = st.number_input(
                f"Protein concentration for **{sample}**:",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f'protein_{sample}'
            )
            protein_data.append({'Sample': sample, 'Concentration': conc})
        
        protein_df = pd.DataFrame(protein_data)
    
    # Get lipid classes
    lipid_classes = sorted(df['ClassKey'].unique())
    
    # Create NormalizationConfig
    try:
        config = NormalizationConfig(
            method='protein',
            selected_classes=lipid_classes
        )
        
        service = NormalizationService()
        normalized_df = service.normalize_by_protein(
            df=df,
            config=config,
            protein_df=protein_df
        )
        st.success("✅ Protein normalization complete!")
        return normalized_df
    except Exception as e:
        st.error(f"❌ Normalization failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def collect_and_apply_both(df: pd.DataFrame, intsta_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Collect both standards and protein, apply normalization."""
    from lipidcruncher.core.services.normalization_service import NormalizationService
    from lipidcruncher.core.models.normalization import NormalizationConfig
    
    lipid_classes = sorted(df['ClassKey'].unique())
    available_standards = intsta_df['LipidMolec'].tolist()
    
    # First, internal standards
    with st.expander("📋 Step 1: Map Internal Standards", expanded=True):
        standards_mapping = {}
        concentrations = {}
        
        for lipid_class in lipid_classes:
            col1, col2 = st.columns(2)
            
            with col1:
                standard = st.selectbox(
                    f"Standard for **{lipid_class}**:",
                    options=['None'] + available_standards,
                    key=f'both_standard_{lipid_class}'
                )
                standards_mapping[lipid_class] = standard if standard != 'None' else None
            
            with col2:
                if standard != 'None':
                    conc = st.number_input(
                        f"Concentration (pmol):",
                        min_value=0.0,
                        value=100.0,
                        step=10.0,
                        key=f'both_conc_{lipid_class}'
                    )
                    concentrations[standard] = conc
        
        missing = [c for c in lipid_classes if not standards_mapping.get(c)]
        if missing:
            st.warning(f"⚠️ No standards selected for: {', '.join(missing)}")
            return None
    
    # Then, protein concentrations
    with st.expander("📋 Step 2: Enter Protein Concentrations", expanded=True):
        protein_data = []
        
        for sample in st.session_state.experiment_config.full_samples_list:
            conc = st.number_input(
                f"Protein concentration for **{sample}**:",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key=f'both_protein_{sample}'
            )
            protein_data.append({'Sample': sample, 'Concentration': conc})
        
        protein_df = pd.DataFrame(protein_data)
    
    # Apply both normalizations
    try:
        # Convert protein_df to dict for config
        protein_concentrations = dict(zip(protein_df['Sample'], protein_df['Concentration']))
        
        config = NormalizationConfig(
            method='both',
            selected_classes=lipid_classes,
            internal_standards=standards_mapping,
            intsta_concentrations=concentrations,
            protein_concentrations=protein_concentrations
        )
        
        service = NormalizationService()
        normalized_df = service.normalize(
            df=df,
            config=config,
            experiment=st.session_state.experiment_config,
            intsta_df=intsta_df,
            protein_df=protein_df
        )
        
        st.success("✅ Both normalizations complete!")
        return normalized_df
    except Exception as e:
        st.error(f"❌ Normalization failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_normalization_results():
    """Display normalization results."""
    st.markdown("### 📊 Normalized Dataset")
    st.write(f"**Shape:** {st.session_state.normalized_df.shape}")
    
    # Show data
    st.dataframe(st.session_state.normalized_df.head(20))
    
    # Download button
    csv = st.session_state.normalized_df.to_csv(index=False)
    st.download_button(
        label="💾 Download Normalized Data",
        data=csv,
        file_name="normalized_data.csv",
        mime="text/csv",
        key="download_normalized"
    )
    
    # Summary
    with st.expander("📊 Normalization Summary", expanded=False):
        st.write(f"**Method used:** {st.session_state.normalization_method}")
        st.write(f"**Lipid classes:** {', '.join(st.session_state.selected_classes)}")
        
        if 'ClassKey' in st.session_state.normalized_df.columns:
            class_counts = st.session_state.normalized_df['ClassKey'].value_counts()
            st.write("**Species per class:**")
            st.dataframe(class_counts)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()