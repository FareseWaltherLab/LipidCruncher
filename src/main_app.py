"""
LipidCruncher - Lipidomics Data Analysis Application

Refactored architecture:
    UI Layer (this file)
        → Workflows (app/workflows/)
            → Adapters (app/adapters/)
                → Services (app/services/)
                    → Models (app/models/)

Reference: old_main_app.py contains the original monolithic implementation.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="LipidCruncher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
BASE_DIR = Path(__file__).parent

# =============================================================================
# Imports - Refactored Components
# =============================================================================

from app.adapters.streamlit_adapter import StreamlitAdapter, SessionState
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.services.format_detection import FormatDetectionService, DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.services.zero_filtering import ZeroFilterConfig
from app.workflows.data_ingestion import DataIngestionWorkflow, IngestionConfig, IngestionResult
from app.workflows.normalization import NormalizationWorkflow, NormalizationWorkflowConfig, NormalizationWorkflowResult
from app.ui.landing_page import display_landing_page, display_logo
from app.ui.format_requirements import display_format_requirements
from app.ui.zero_filtering import display_zero_filtering_config
from app.ui.content import (
    get_processing_docs,
    ZERO_FILTERING_DOCS,
    NORMALIZATION_METHODS_DOCS,
    STANDARDS_EXTRACT_HELP,
    STANDARDS_COMPLETE_HELP,
    PROTEIN_CSV_HELP,
)
from app.ui.sidebar import (
    display_format_selection,
    load_sample_dataset,
    display_file_upload,
    standardize_uploaded_data,
    display_column_mapping,
    display_sample_grouping,
)

# Legacy modules for UI compatibility (DataFormatHandler used in main content area)
from lipidomics.data_format_handler import DataFormatHandler


# =============================================================================
# Streamlit Compatibility
# =============================================================================

def safe_rerun():
    """Rerun the app, compatible with both old and new Streamlit versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


# =============================================================================
# UI Components - Data Processing Documentation
# =============================================================================

def display_data_processing_docs(data_format: str):
    """Display format-specific data processing documentation."""
    with st.expander("📖 About Data Standardization and Filtering", expanded=False):
        st.markdown(get_processing_docs(data_format))
        st.markdown("---")
        st.markdown(ZERO_FILTERING_DOCS)


# =============================================================================
# UI Components - Format-Specific Filtering
# =============================================================================

def display_grade_filtering_config(df: pd.DataFrame) -> dict:
    """
    Display LipidSearch grade filtering configuration.

    Returns:
        dict: Grade config mapping class to acceptable grades, or None for defaults
    """
    # Check if the required columns exist
    if 'ClassKey' not in df.columns or 'TotalGrade' not in df.columns:
        return None

    # Get unique classes from the data
    all_classes = sorted(df['ClassKey'].dropna().unique())
    if not all_classes:
        return None

    with st.expander("⚙️ Configure Grade Filtering", expanded=False):
        # Initialize session state for persistence
        if 'grade_filter_mode' not in st.session_state:
            st.session_state.grade_filter_mode = "Use Default Settings"
        if 'grade_selections' not in st.session_state:
            st.session_state.grade_selections = {}

        options = ["Use Default Settings", "Customize by Class"]
        widget_key = "grade_filter_mode_radio"

        # Initialize widget key from persisted value BEFORE rendering
        persisted_value = st.session_state.get('grade_filter_mode', "Use Default Settings")
        if persisted_value in options:
            st.session_state[widget_key] = persisted_value

        def on_grade_mode_change():
            st.session_state.grade_filter_mode = st.session_state[widget_key]

        use_custom = st.radio(
            "Grade filtering mode:",
            options,
            horizontal=True,
            key=widget_key,
            on_change=on_grade_mode_change
        )
        st.session_state.grade_filter_mode = use_custom

        if use_custom == "Use Default Settings":
            st.success("✓ Default: A/B for all classes, plus C for LPC and SM.")
            return None

        # Custom settings
        st.markdown("---")
        grade_config = {}
        cols = st.columns(3)

        for idx, lipid_class in enumerate(all_classes):
            with cols[idx % 3]:
                # Get saved or default grades
                if lipid_class in st.session_state.grade_selections:
                    default_grades = st.session_state.grade_selections[lipid_class]
                elif lipid_class in ['LPC', 'SM']:
                    default_grades = ['A', 'B', 'C']
                else:
                    default_grades = ['A', 'B']

                selected_grades = st.multiselect(
                    f"**{lipid_class}**",
                    options=['A', 'B', 'C', 'D'],
                    default=default_grades,
                    key=f"grade_select_{lipid_class}"
                )

                st.session_state.grade_selections[lipid_class] = selected_grades

                if not selected_grades:
                    st.error("⚠️ Will be excluded!")

                grade_config[lipid_class] = selected_grades

        return grade_config


def display_msdial_data_type_selection():
    """Display MS-DIAL data type selection (raw vs pre-normalized)."""
    features = st.session_state.get('msdial_features', {})
    has_normalized_data = features.get('has_normalized_data', False)
    raw_samples = features.get('raw_sample_columns', [])
    norm_samples = features.get('normalized_sample_columns', [])

    if has_normalized_data and len(norm_samples) > 0:
        st.markdown("##### 📊 Data Type Selection")
        st.markdown(f"""
Your MS-DIAL export contains both raw and pre-normalized intensity values:
- **Raw data**: {len(raw_samples)} sample columns
- **Normalized data**: {len(norm_samples)} sample columns (after 'Lipid IS' column)
        """)

        options = [
            f"Raw intensity values ({len(raw_samples)} samples)",
            f"Pre-normalized values ({len(norm_samples)} samples)"
        ]

        # Initialize widget key from persisted index BEFORE rendering radio
        # This ensures consistency when navigating between pages
        widget_key = "msdial_data_type_radio"
        persisted_index = st.session_state.get('msdial_data_type_index', 0)
        if persisted_index < len(options):
            st.session_state[widget_key] = options[persisted_index]

        def on_data_type_change():
            """Callback to update session state immediately when selection changes."""
            selection = st.session_state[widget_key]
            new_index = options.index(selection) if selection in options else 0
            st.session_state.msdial_data_type_index = new_index
            st.session_state.msdial_use_normalized = "Pre-normalized" in selection
            # Clear cached data to force re-standardization with new selection
            st.session_state.standardized_df = None
            st.session_state.cleaned_df = None
            st.session_state.pre_filter_df = None
            st.session_state.continuation_df = None

        data_type = st.radio(
            "Select which data to use:",
            options,
            key=widget_key,
            on_change=on_data_type_change,
            help="Choose raw data if you want to apply LipidCruncher's normalization. Choose pre-normalized if MS-DIAL already normalized your data."
        )

        # Keep session state in sync after render
        use_normalized = "Pre-normalized" in data_type
        st.session_state.msdial_data_type_index = options.index(data_type) if data_type in options else 0
        st.session_state.msdial_use_normalized = use_normalized

        if use_normalized:
            st.info("📌 Using pre-normalized data. LipidCruncher's internal standard normalization will be skipped.")
        else:
            st.info("📌 Using raw intensity data. You can apply normalization in the next step.")

        st.markdown("---")


def display_quality_filtering_config() -> dict:
    """
    Display MS-DIAL quality filtering configuration.

    Returns:
        dict: Quality config with 'total_score_threshold' and 'require_msms' keys
    """
    features = st.session_state.get('msdial_features', {})
    quality_filtering_available = features.get('has_quality_score', False)
    msms_filtering_available = features.get('has_msms_matched', False)

    if not quality_filtering_available and not msms_filtering_available:
        st.warning("Quality filtering unavailable — no 'Total score' or 'MS/MS matched' columns found.")
        return None

    with st.expander("⚙️ Configure Quality Filtering", expanded=False):
        # Initialize session state
        if 'msdial_quality_level' not in st.session_state:
            st.session_state.msdial_quality_level = 'Moderate (Score ≥60)'

        if quality_filtering_available:
            quality_options = {
                'Strict (Score ≥80, MS/MS required)': {'total_score_threshold': 80, 'require_msms': True},
                'Moderate (Score ≥60)': {'total_score_threshold': 60, 'require_msms': False},
                'Permissive (Score ≥40)': {'total_score_threshold': 40, 'require_msms': False},
                'No filtering': {'total_score_threshold': 0, 'require_msms': False}
            }
            quality_options_list = list(quality_options.keys())
            widget_key = "msdial_quality_level_radio"

            # Initialize widget key from persisted value BEFORE rendering
            persisted_value = st.session_state.get('msdial_quality_level', 'Moderate (Score ≥60)')
            if persisted_value in quality_options_list:
                st.session_state[widget_key] = persisted_value

            def on_quality_level_change():
                st.session_state.msdial_quality_level = st.session_state[widget_key]

            selected_option = st.radio(
                "Quality filtering level:",
                quality_options_list,
                horizontal=True,
                key=widget_key,
                on_change=on_quality_level_change
            )
            st.session_state.msdial_quality_level = selected_option

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

            # Advanced: custom score threshold
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
            st.markdown("---")
            st.markdown(f"**Current settings:** Score ≥ {quality_config['total_score_threshold']}, "
                       f"MS/MS required: {'Yes' if quality_config['require_msms'] else 'No'}")

            # Show filter results only if they match the current config
            # (results are stored after workflow runs, so they reflect previous run with same config)
            last_config = st.session_state.get('last_quality_config')
            if (last_config and
                last_config.get('total_score_threshold') == quality_config['total_score_threshold'] and
                last_config.get('require_msms') == quality_config['require_msms']):
                ingestion_result = st.session_state.get('ingestion_result')
                if ingestion_result and ingestion_result.cleaning_messages:
                    st.markdown("**Filter Results:**")
                    for msg in ingestion_result.cleaning_messages:
                        st.info(msg)

            return quality_config

        elif msms_filtering_available:
            # Only MS/MS filtering available
            require_msms = st.checkbox(
                "Require MS/MS validation",
                value=False,
                key="msdial_msms_only"
            )
            return {'total_score_threshold': 0, 'require_msms': require_msms}

    return None



# =============================================================================
# UI Components - Manage Internal Standards
# =============================================================================

def display_manage_internal_standards(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Display the Manage Internal Standards expander.

    Allows users to:
    - Use automatically detected standards
    - Upload custom standards (from dataset or external)
    - Clear custom standards
    - View consistency plots for internal standards

    Args:
        cleaned_df: The cleaned data DataFrame (for extracting standards)
        auto_detected_df: Auto-detected internal standards DataFrame

    Returns:
        The active internal standards DataFrame to use for normalization
    """
    from app.services.standards import StandardsService
    from app.ui.standards_plots import display_standards_consistency_plots

    # Track the active standards DataFrame (set in branches below)
    active_standards_df = pd.DataFrame()

    with st.expander("Manage Internal Standards", expanded=False):
        st.markdown("""
Auto-detection identifies deuterated standards (`(d5)`, `(d7)`, `(d9)`),
`ISTD`/`IS` markers in class names, and SPLASH LIPIDOMIX® patterns.
        """)

        # Initialize session state for standards management
        if 'standards_source' not in st.session_state:
            st.session_state.standards_source = "Automatic Detection"
        if 'custom_standards_df' not in st.session_state:
            st.session_state.custom_standards_df = None
        if 'custom_standards_mode' not in st.session_state:
            st.session_state.custom_standards_mode = None  # Track which upload mode was used
        if 'original_auto_intsta_df' not in st.session_state:
            st.session_state.original_auto_intsta_df = auto_detected_df

        # Standards source selection
        st.markdown("##### 🏷️ Standards Source")

        standards_source = st.radio(
            "Standards source:",
            ["Automatic Detection", "Upload Custom Standards"],
            horizontal=True,
            key="standards_source_radio",
            label_visibility="collapsed"
        )
        st.session_state.standards_source = standards_source

        if standards_source == "Automatic Detection":
            # Clear custom standards when switching to automatic
            if st.session_state.custom_standards_df is not None:
                st.session_state.custom_standards_df = None
                st.session_state.custom_standards_mode = None

            # Show auto-detected standards
            if auto_detected_df is not None and not auto_detected_df.empty:
                st.success(f"✓ Found {len(auto_detected_df)} standards")
                st.dataframe(auto_detected_df, use_container_width=True)

                # Download button
                csv = auto_detected_df.to_csv(index=False)
                st.download_button(
                    label="Download Detected Standards",
                    data=csv,
                    file_name="detected_standards.csv",
                    mime="text/csv",
                    key="download_auto_standards"
                )
                active_standards_df = auto_detected_df
            else:
                st.warning("No internal standards automatically detected in dataset.")
                active_standards_df = pd.DataFrame()

        else:  # Upload Custom Standards
            st.markdown("---")

            # Mode selection: standards in dataset or external
            st.markdown("**Are standards present in your main dataset?**")

            standards_location = st.radio(
                "Standards location:",
                options=[
                    "Yes — Extract from dataset",
                    "No — Uploading complete standards data"
                ],
                key="standards_location_radio",
                horizontal=True,
                label_visibility="collapsed"
            )

            use_extract_mode = "Yes" in standards_location
            current_mode = "extract" if use_extract_mode else "complete"

            # Clear custom standards if user switched between upload modes
            if (st.session_state.custom_standards_mode is not None and
                st.session_state.custom_standards_mode != current_mode and
                st.session_state.custom_standards_df is not None):
                st.session_state.custom_standards_df = None
                st.session_state.custom_standards_mode = None

            # Format guidance
            st.markdown("---")
            if use_extract_mode:
                st.markdown(STANDARDS_EXTRACT_HELP)
            else:
                st.markdown(STANDARDS_COMPLETE_HELP)

            # File uploader
            uploaded_file = st.file_uploader(
                "Upload standards CSV",
                type=['csv'],
                key="standards_file_uploader"
            )

            # Show preserved custom standards if file uploader is empty
            if uploaded_file is None and st.session_state.custom_standards_df is not None:
                st.success(f"✓ Using {len(st.session_state.custom_standards_df)} custom standards")
                st.dataframe(st.session_state.custom_standards_df, use_container_width=True)

                # Clear button
                if st.button("Clear Custom Standards", key="clear_custom_standards"):
                    st.session_state.custom_standards_df = None
                    st.session_state.custom_standards_mode = None
                    st.session_state.standards_source = "Automatic Detection"
                    safe_rerun()

                active_standards_df = st.session_state.custom_standards_df

            # Process uploaded file
            elif uploaded_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)

                    # Process using StandardsService
                    result = StandardsService.process_standards_file(
                        uploaded_df=uploaded_df,
                        cleaned_df=cleaned_df,
                        standards_in_dataset=use_extract_mode
                    )

                    if result.standards_df is not None and not result.standards_df.empty:
                        st.session_state.custom_standards_df = result.standards_df
                        st.session_state.custom_standards_mode = current_mode  # Track which mode was used

                        # For external standards, remove them from main dataset if present
                        if not use_extract_mode:
                            filtered_df, removed_lipids = StandardsService.remove_standards_from_dataset(
                                cleaned_df,
                                result.standards_df
                            )
                            if removed_lipids:
                                st.session_state.cleaned_df = filtered_df
                                preview = removed_lipids[:5]
                                more = f"... and {len(removed_lipids) - 5} more" if len(removed_lipids) > 5 else ""
                                st.warning(
                                    f"⚠️ Removed {len(removed_lipids)} standard(s) from main dataset: "
                                    f"{', '.join(preview)}{more}"
                                )

                        # Show processing info
                        if result.duplicates_removed > 0:
                            st.info(f"Removed {result.duplicates_removed} duplicate standard(s).")

                        st.success(f"✓ Loaded {result.standards_count} custom standards (mode: {result.source_mode})")
                        st.dataframe(result.standards_df, use_container_width=True)

                        # Download button
                        csv = result.standards_df.to_csv(index=False)
                        st.download_button(
                            label="Download Custom Standards",
                            data=csv,
                            file_name="custom_standards.csv",
                            mime="text/csv",
                            key="download_custom_standards"
                        )

                        active_standards_df = result.standards_df
                    else:
                        st.error("No valid standards found in uploaded file.")
                        active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()

                except ValueError as ve:
                    st.error(str(ve))
                    active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    active_standards_df = auto_detected_df if auto_detected_df is not None else pd.DataFrame()

            else:
                # No file uploaded yet, use auto-detected as fallback
                if auto_detected_df is not None and not auto_detected_df.empty:
                    st.info("Upload a CSV file or switch to Automatic Detection.")
                    active_standards_df = auto_detected_df
                else:
                    st.warning("No standards available. Please upload a standards file.")
                    active_standards_df = pd.DataFrame()

        # Show consistency plots if we have standards and experiment config
        if (active_standards_df is not None and
            not active_standards_df.empty and
            'experiment' in st.session_state and
            st.session_state.experiment is not None):
            display_standards_consistency_plots(
                intsta_df=active_standards_df,
                experiment=st.session_state.experiment
            )

    return active_standards_df


# =============================================================================
# UI Components - Normalization
# =============================================================================

def _display_class_selection(cleaned_df: pd.DataFrame) -> list:
    """Display lipid class selection UI. Returns selected classes or empty list."""
    available_classes = NormalizationWorkflow.get_available_classes(cleaned_df)

    # Initialize session state for persistence (matches old app pattern)
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = available_classes.copy()

    # Validate that saved classes still exist in available classes
    # (data may have changed due to new file or config changes)
    valid_saved_classes = [c for c in st.session_state.selected_classes if c in available_classes]

    # Determine default: use valid saved classes, or all available if none saved
    default_classes = valid_saved_classes if valid_saved_classes else available_classes

    def update_selected_classes():
        st.session_state.selected_classes = st.session_state.temp_selected_classes

    selected_classes = st.multiselect(
        'Classes to analyze:',
        options=available_classes,
        default=default_classes,
        key='temp_selected_classes',
        on_change=update_selected_classes
    )

    return selected_classes


def _display_internal_standards_config(intsta_df: pd.DataFrame, selected_classes: list) -> tuple:
    """Display internal standards configuration UI. Returns (internal_standards, intsta_concentrations) or (None, None)."""
    if intsta_df is None or intsta_df.empty:
        st.error("No internal standards detected. Please upload standards or use a different method.")
        return None, None

    standards_by_class = NormalizationWorkflow.get_standards_by_class(intsta_df)
    all_standards = NormalizationWorkflow.get_available_standards(intsta_df)

    if not all_standards:
        st.error("No internal standards available.")
        return None, None

    internal_standards = {}
    intsta_concentrations = {}

    # Retrieve saved mappings for session state persistence
    saved_mappings = st.session_state.get('class_standard_map', {})

    with st.expander("⚙️ Internal Standards Mapping", expanded=True):
        # Show warnings for classes without specific standards
        for lipid_class in selected_classes:
            class_standards = standards_by_class.get(lipid_class, [])
            if not class_standards:
                if saved_mappings and lipid_class in saved_mappings and saved_mappings[lipid_class] in all_standards:
                    default_standard = saved_mappings[lipid_class]
                else:
                    default_standard = all_standards[0]
                st.warning(f"No specific standards available for {lipid_class}. Defaulting to {default_standard}.")

        # Class-to-standard mapping (double-column layout)
        for lipid_class in selected_classes:
            # Determine default standard (saved mapping -> class-specific -> first available)
            default_std = None
            if saved_mappings and lipid_class in saved_mappings:
                default_std = saved_mappings[lipid_class]
                if default_std not in all_standards:
                    default_std = None
            if not default_std:
                class_standards = standards_by_class.get(lipid_class, [])
                default_std = class_standards[0] if class_standards else all_standards[0]

            default_idx = all_standards.index(default_std) if default_std in all_standards else 0

            col1, col2 = st.columns([1, 2])
            with col1:
                st.text(lipid_class)
            with col2:
                selected_std = st.selectbox(
                    f'Select internal standard for {lipid_class}',
                    options=all_standards,
                    index=default_idx,
                    key=f'standard_selection_{lipid_class}',
                    label_visibility="collapsed"
                )
                internal_standards[lipid_class] = selected_std

        # Save mapping to session state for persistence
        st.session_state.class_standard_map = internal_standards

        # Standard concentrations (always visible, not optional)
        if st.session_state.get('standard_concentrations') is None:
            st.session_state.standard_concentrations = {}

        st.write("Enter the concentration of each selected internal standard (µM):")

        all_concentrations_valid = True
        for std in set(internal_standards.values()):
            widget_key = f"conc_{std}"
            # Initialize from preserved concentrations if not already set
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.standard_concentrations.get(std, 1.0)

            conc = st.number_input(
                f"Concentration (µM) for {std}",
                min_value=0.0,
                step=0.1,
                key=widget_key
            )

            # Sync back for session preservation
            st.session_state.standard_concentrations[std] = conc

            if conc <= 0:
                st.error(f"Please enter a valid concentration for {std}")
                all_concentrations_valid = False
            intsta_concentrations[std] = conc

        if not all_concentrations_valid:
            st.error("Please enter valid concentrations for all standards")
            return None, None

    return internal_standards, intsta_concentrations


def _display_protein_config(experiment: ExperimentConfig) -> dict:
    """Display protein concentration configuration UI. Returns protein_concentrations dict or None."""

    sample_names = experiment.full_samples_list
    protein_concentrations = {}

    with st.expander("⚙️ Protein Concentration Data", expanded=True):
        # Initialize method selection key if not present
        if 'protein_input_method' not in st.session_state:
            st.session_state.protein_input_method = "Manual Input"

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
            if 'protein_df' in st.session_state:
                del st.session_state.protein_df
            for sample in sample_names:
                widget_key = f"protein_{sample}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

        st.session_state.protein_input_method_prev = method

        if method == "Manual Input":
            # Get preserved protein data for restoring values
            preserved_protein_df = st.session_state.get('protein_df')
            preserved_values = {}
            if preserved_protein_df is not None and isinstance(preserved_protein_df, dict):
                preserved_values = preserved_protein_df
            elif preserved_protein_df is not None and hasattr(preserved_protein_df, 'columns'):
                if 'Sample' in preserved_protein_df.columns and 'Concentration' in preserved_protein_df.columns:
                    preserved_values = dict(zip(preserved_protein_df['Sample'], preserved_protein_df['Concentration']))

            # Initialize session state for all samples BEFORE widgets render
            for sample in sample_names:
                widget_key = f"protein_{sample}"
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = float(preserved_values.get(sample, 1.0))

            # 3-column flat grid layout (matches old app)
            cols = st.columns(3)
            for idx, sample in enumerate(sample_names):
                with cols[idx % 3]:
                    concentration = st.number_input(
                        f'{sample}:',
                        min_value=0.0,
                        max_value=1000000.0,
                        step=0.1,
                        key=f"protein_{sample}"
                    )
                    protein_concentrations[sample] = concentration

            # Preserve in session state
            st.session_state.protein_df = protein_concentrations.copy()
            return protein_concentrations

        else:  # Upload CSV File
            st.markdown(PROTEIN_CSV_HELP)

            preserved_protein_df = st.session_state.get('protein_df')

            uploaded_file = st.file_uploader("Upload CSV", type="csv", key="protein_csv_upload")

            if uploaded_file is not None:
                try:
                    csv_df = pd.read_csv(uploaded_file)

                    if 'Concentration' not in csv_df.columns:
                        st.error(f"CSV must contain a column named 'Concentration'. Found: {list(csv_df.columns)}")
                        return None

                    if len(csv_df) != len(sample_names):
                        st.error(f"Row count ({len(csv_df)}) doesn't match sample count ({len(sample_names)})")
                        return None

                    csv_df['Concentration'] = pd.to_numeric(csv_df['Concentration'], errors='coerce')
                    if csv_df['Concentration'].isna().any():
                        st.error("Some concentration values couldn't be converted to numbers.")
                        return None

                    # Convert to dict for NormalizationConfig compatibility
                    protein_concentrations = dict(zip(sample_names, csv_df['Concentration'].tolist()))
                    st.session_state.protein_df = protein_concentrations.copy()
                    st.success(f"✓ Loaded {len(protein_concentrations)} concentration values")
                    return protein_concentrations

                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    return None

            # No new file uploaded - check preserved data
            if preserved_protein_df is not None and isinstance(preserved_protein_df, dict):
                if len(preserved_protein_df) == len(sample_names):
                    st.success(f"✓ Using previously loaded {len(preserved_protein_df)} concentration values")
                    return preserved_protein_df

            st.info("Please upload a CSV file with protein concentrations.")
            return None


def _run_normalization(
    cleaned_df: pd.DataFrame,
    intsta_df: pd.DataFrame,
    experiment: ExperimentConfig,
    data_format: str,
    config_method: str,
    selected_classes: list,
    internal_standards: dict,
    intsta_concentrations: dict,
    protein_concentrations: dict
) -> NormalizationWorkflowResult:
    """Run normalization workflow and return result (cached for performance)."""
    try:
        format_map = {
            'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
            'MS-DIAL': DataFormat.MSDIAL,
            'Generic Format': DataFormat.GENERIC,
            'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
        }

        # Use cached adapter method for performance
        df_hash = StreamlitAdapter.compute_df_hash(cleaned_df)
        experiment_dict = StreamlitAdapter.experiment_to_dict(experiment)
        format_type_value = format_map.get(data_format, DataFormat.GENERIC).value

        # Call cached normalization workflow
        (
            normalized_df,
            success,
            method_applied,
            removed_standards,
            validation_errors,
            validation_warnings,
        ) = StreamlitAdapter.run_normalization(
            _df_hash=df_hash,
            df=cleaned_df,
            experiment_dict=experiment_dict,
            method=config_method,
            selected_classes=selected_classes,
            format_type=format_type_value,
            internal_standards=internal_standards if internal_standards else None,
            intsta_concentrations=intsta_concentrations if intsta_concentrations else None,
            protein_concentrations=protein_concentrations if protein_concentrations else None,
            intsta_df=intsta_df,
        )

        # Reconstruct result object for compatibility
        result = NormalizationWorkflowResult(
            normalized_df=normalized_df,
            success=success,
            method_applied=method_applied,
            removed_standards=removed_standards,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
        )

        if result.success:
            st.session_state.normalization_result = result
            st.session_state.normalized_df = result.normalized_df
            st.session_state.continuation_df = result.normalized_df

        return result

    except Exception as e:
        st.error(f"Normalization error: {e}")
        return None


def display_normalization_ui(cleaned_df: pd.DataFrame, intsta_df: pd.DataFrame, experiment: ExperimentConfig, data_format: str):
    """Display normalization options and apply normalization automatically."""
    # About Normalization Methods (documentation)
    with st.expander("📖 About Normalization Methods", expanded=False):
        st.markdown(NORMALIZATION_METHODS_DOCS)

    # Class selection
    st.markdown("##### 🎯 Select Lipid Classes")
    selected_classes = _display_class_selection(cleaned_df)
    if not selected_classes:
        st.warning("Please select at least one lipid class.")
        return None

    # Check if we have standards available
    has_standards = intsta_df is not None and not intsta_df.empty

    # Check if using pre-normalized MS-DIAL data (IS normalization not applicable)
    is_msdial_prenormalized = (data_format == 'MS-DIAL' and
                               st.session_state.get('msdial_use_normalized', False))

    # Determine available normalization options
    if is_msdial_prenormalized:
        normalization_options = ['None (pre-normalized data)', 'Protein-based']
        st.markdown("*Internal standards options unavailable — using pre-normalized MS-DIAL data.*")
    elif has_standards:
        normalization_options = ['None (pre-normalized data)', 'Internal Standards', 'Protein-based', 'Both']
    else:
        normalization_options = ['None (pre-normalized data)', 'Protein-based']
        st.markdown("*Internal standards options unavailable — no standards detected or uploaded.*")

    # Initialize session state for normalization method
    if 'norm_method_selection' not in st.session_state:
        st.session_state['norm_method_selection'] = 'None (pre-normalized data)'

    # Handle case where saved method is no longer available (e.g., standards removed)
    current_selection = st.session_state.get('norm_method_selection')
    if current_selection not in normalization_options:
        st.session_state['norm_method_selection'] = 'None (pre-normalized data)'

    # Method selection
    st.markdown("##### ⚙️ Normalization Method")
    method = st.radio(
        "Method:",
        options=normalization_options,
        horizontal=True,
        key='norm_method_selection'
    )

    method_map = {
        'None (pre-normalized data)': 'none',
        'Internal Standards': 'internal_standard',
        'Protein-based': 'protein',
        'Both': 'both'
    }
    config_method = method_map.get(method, 'none')

    # Collect configuration based on method
    internal_standards = {}
    intsta_concentrations = {}
    protein_concentrations = {}

    if method in ['Internal Standards', 'Both']:
        internal_standards, intsta_concentrations = _display_internal_standards_config(intsta_df, selected_classes)
        if internal_standards is None:
            return None

    if method in ['Protein-based', 'Both']:
        protein_concentrations = _display_protein_config(experiment)
        if protein_concentrations is None:
            return None

    # Apply normalization automatically (no button needed)
    result = _run_normalization(
        cleaned_df, intsta_df, experiment, data_format,
        config_method, selected_classes,
        internal_standards, intsta_concentrations, protein_concentrations
    )

    # Display results
    if result and result.success:
        st.markdown("##### 📊 Final Normalized Data")

        if result.normalized_df is not None:
            st.dataframe(result.normalized_df, use_container_width=True)
            csv = result.normalized_df.to_csv(index=False)
            st.download_button(
                label="Download Normalized Data",
                data=csv,
                file_name="normalized_data.csv",
                mime="text/csv",
                key="download_normalized_data"
            )
    elif result:
        for error in result.validation_errors:
            st.error(error)

    return st.session_state.get('normalized_df')


# =============================================================================
# Main App Page
# =============================================================================

def display_app_page():
    """Display the main application page (Module 1) with centered layout matching landing page."""
    # Centered layout matching landing page width
    _, center, _ = st.columns([1, 3, 1])

    # Sidebar: Format selection (must happen before center content that depends on it)
    data_format = display_format_selection()

    with center:
        display_logo(centered=True)
        st.markdown("Process, analyze and visualize lipidomic data from multiple sources.")
        display_format_requirements(data_format)

    # Sidebar: File upload
    raw_df = display_file_upload(data_format)

    if raw_df is None:
        with center:
            st.info("Upload a dataset or load sample data to begin.")

            # Back to landing button
            if st.button("← Back to Home"):
                st.session_state.page = 'landing'
                safe_rerun()
        return

    # Standardize data if not already done
    if st.session_state.get('standardized_df') is None:
        standardized_df = standardize_uploaded_data(raw_df, data_format)
        if standardized_df is None:
            return
        st.session_state.standardized_df = standardized_df
    else:
        standardized_df = st.session_state.standardized_df

    # Sidebar: Column Name Standardization (display mapping)
    mapping_valid, modified_df = display_column_mapping(standardized_df, data_format)
    if not mapping_valid:
        return

    # Use modified df if sample override was applied
    if modified_df is not None:
        standardized_df = modified_df
        st.session_state.standardized_df = standardized_df

    # Use standardized df for sample grouping
    df = standardized_df

    # Sidebar: Sample grouping (includes experiment def, grouping, BQC, confirm)
    experiment, bqc_label = display_sample_grouping(df, data_format)

    if experiment is None:
        with center:
            st.info("Configure your experiment in the sidebar and check 'Confirm Inputs' to proceed.")
        return

    # After sample grouping, use the potentially reordered DataFrame from session state
    # (display_sample_grouping updates st.session_state.standardized_df when user regroups samples)
    df = st.session_state.standardized_df

    # ==========================================================================
    # Main area: Processing Module (Automatic Flow)
    # ==========================================================================
    with center:
        st.subheader("Data Standardization, Filtering, and Normalization")

        # Step 1: Data processing documentation
        display_data_processing_docs(data_format)

        # Step 2: Format-specific filtering configuration
        grade_config = None
        quality_config = None
        quality_config_dict = None

        if data_format == 'LipidSearch 5.0':
            grade_config_dict = display_grade_filtering_config(raw_df)
            if grade_config_dict is not None:
                grade_config = GradeFilterConfig(grade_config=grade_config_dict)

        elif data_format == 'MS-DIAL':
            # Data type selection (raw vs normalized)
            display_msdial_data_type_selection()

            # Quality filtering
            quality_config_dict = display_quality_filtering_config()
            if quality_config_dict is not None:
                quality_config = QualityFilterConfig(
                    total_score_threshold=quality_config_dict.get('total_score_threshold', 0),
                    require_msms=quality_config_dict.get('require_msms', False)
                )

        # Step 3: Automatic data processing (runs on every confirmation)
        # Build config hash to detect when settings change
        config_hash = f"{data_format}_{grade_config}_{quality_config}"

        # Process data automatically (no button needed)
        format_map = {
            'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
            'MS-DIAL': DataFormat.MSDIAL,
            'Generic Format': DataFormat.GENERIC,
            'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
        }

        # Run ingestion workflow (clean data WITHOUT zero filtering - user configures thresholds after)
        # Use cached adapter method for performance
        df_hash = StreamlitAdapter.compute_df_hash(df)
        experiment_dict = StreamlitAdapter.experiment_to_dict(experiment)
        format_type_value = format_map.get(data_format).value if format_map.get(data_format) else None
        grade_config_dict_for_cache = StreamlitAdapter.config_to_dict(grade_config)
        quality_config_dict_for_cache = StreamlitAdapter.config_to_dict(quality_config)

        # Capture previous config BEFORE running workflow (to detect config changes)
        prev_quality_config = st.session_state.get('last_quality_config')

        try:
            # Call cached ingestion workflow
            (
                cleaned_df,
                intsta_df,
                detected_format,
                is_valid,
                validation_errors,
                validation_warnings,
                cleaning_messages,
            ) = StreamlitAdapter.run_ingestion(
                _df_hash=df_hash,
                df=df,
                experiment_dict=experiment_dict,
                format_type=format_type_value,
                bqc_label=bqc_label,
                apply_zero_filter=False,  # Don't apply yet - let user configure
                grade_config_dict=grade_config_dict_for_cache,
                quality_config_dict=quality_config_dict_for_cache,
            )

            # Reconstruct result object for compatibility
            result = IngestionResult(
                detected_format=DataFormat(detected_format),
                cleaned_df=cleaned_df,
                internal_standards_df=intsta_df,
                is_valid=is_valid,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                cleaning_messages=cleaning_messages,
            )

            st.session_state.ingestion_result = result
            st.session_state.pre_filter_df = result.cleaned_df  # Store pre-filter version
            # Store the config used so we can verify results match current settings
            if quality_config_dict is not None:
                st.session_state.last_quality_config = quality_config_dict.copy()

            if result.is_valid:
                st.session_state.cleaned_df = result.cleaned_df
                st.session_state.intsta_df = result.internal_standards_df
                st.session_state.continuation_df = result.cleaned_df
        except Exception as e:
            st.error(f"Processing error: {e}")
            return

        # Step 4: Display results (automatic - no button needed)
        if not result.is_valid:
            for error in result.validation_errors:
                st.error(error)
            return

        # Show warnings
        for warning in result.validation_warnings:
            st.warning(warning)

        # If quality config changed, rerun so the expander shows updated results
        if data_format == 'MS-DIAL' and quality_config_dict:
            config_changed = not (prev_quality_config and
                                 prev_quality_config.get('total_score_threshold') == quality_config_dict.get('total_score_threshold') and
                                 prev_quality_config.get('require_msms') == quality_config_dict.get('require_msms'))
            if config_changed:
                safe_rerun()

        # Step 5: Zero Filtering Configuration (interactive)
        pre_filter_df = st.session_state.get('pre_filter_df', result.cleaned_df)
        if pre_filter_df is not None and not pre_filter_df.empty:
            filtered_df, removed_species, zero_config = display_zero_filtering_config(
                pre_filter_df, experiment, bqc_label,
                data_format=st.session_state.get('format_type')
            )

            # Update session state with filtered data
            if filtered_df is not None:
                st.session_state.cleaned_df = filtered_df
                st.session_state.continuation_df = filtered_df

        # Step 6: Show final filtered data (outside expander, matching old app)
        # Sort by ClassKey so all species of the same class are grouped together
        if 'ClassKey' in st.session_state.cleaned_df.columns:
            st.session_state.cleaned_df = st.session_state.cleaned_df.sort_values('ClassKey').reset_index(drop=True)
            st.session_state.continuation_df = st.session_state.cleaned_df

        st.markdown("##### 📋 Final Filtered Data (Pre-Normalization)")
        st.dataframe(st.session_state.cleaned_df, use_container_width=True)
        csv = st.session_state.cleaned_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="final_filtered_data.csv",
            mime="text/csv",
            key="download_filtered_data"
        )

        # Step 7: Manage Internal Standards
        auto_detected_intsta_df = st.session_state.get('intsta_df', result.internal_standards_df)
        intsta_df = display_manage_internal_standards(
            cleaned_df=st.session_state.cleaned_df,
            auto_detected_df=auto_detected_intsta_df
        )
        # Update session state with the active standards
        st.session_state.intsta_df = intsta_df

        # Step 8: Normalization (automatic - no button needed)
        display_normalization_ui(
            cleaned_df=st.session_state.cleaned_df,
            intsta_df=intsta_df,
            experiment=experiment,
            data_format=data_format
        )

        # Navigation
        st.markdown("---")
        if st.button("← Back to Home"):
            st.session_state.page = 'landing'
            StreamlitAdapter.reset_data_state()
            safe_rerun()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main application entry point."""
    if st.session_state.page == 'landing':
        display_landing_page()
    elif st.session_state.page == 'app':
        display_app_page()


if __name__ == "__main__":
    main()
