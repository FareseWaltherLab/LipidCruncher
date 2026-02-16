"""
Data Processing UI Components for LipidCruncher main content area.

This module contains:
- display_data_processing_docs: Format-specific processing documentation
- display_grade_filtering_config: LipidSearch grade filtering configuration
- display_msdial_data_type_selection: MS-DIAL data type selection (raw vs pre-normalized)
- display_quality_filtering_config: MS-DIAL quality filtering configuration
- build_filter_configs: Build format-specific filter config objects from UI widgets
- run_ingestion_pipeline: Execute cached ingestion workflow and update session state
- display_final_filtered_data: Display final filtered data table with download
"""

import streamlit as st
import pandas as pd

from app.adapters.streamlit_adapter import StreamlitAdapter


from app.services.format_detection import DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.workflows.data_ingestion import IngestionResult
from app.ui.content import get_processing_docs, ZERO_FILTERING_DOCS


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
            quality_config = {'total_score_threshold': 0, 'require_msms': require_msms}

            # Summary
            st.markdown("---")
            st.markdown(f"**Current settings:** MS/MS required: {'Yes' if require_msms else 'No'}")

            # Show filter results if they match the current config
            last_config = st.session_state.get('last_quality_config')
            if (last_config and
                last_config.get('total_score_threshold') == 0 and
                last_config.get('require_msms') == require_msms):
                ingestion_result = st.session_state.get('ingestion_result')
                if ingestion_result and ingestion_result.cleaning_messages:
                    st.markdown("**Filter Results:**")
                    for msg in ingestion_result.cleaning_messages:
                        st.info(msg)

            return quality_config

    return None


# =============================================================================
# Pipeline Orchestration
# =============================================================================

FORMAT_MAP = {
    'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
    'MS-DIAL': DataFormat.MSDIAL,
    'Generic Format': DataFormat.GENERIC,
    'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
}


def build_filter_configs(data_format: str, raw_df: pd.DataFrame = None):
    """
    Display format-specific filter configuration UI and build config objects.

    Args:
        data_format: The selected data format string
        raw_df: Raw DataFrame (needed for LipidSearch grade filtering)

    Returns:
        Tuple of (grade_config, quality_config, quality_config_dict)
    """
    grade_config = None
    quality_config = None
    quality_config_dict = None

    if data_format == 'LipidSearch 5.0':
        grade_config_dict = display_grade_filtering_config(raw_df)
        if grade_config_dict is not None:
            grade_config = GradeFilterConfig(grade_config=grade_config_dict)

    elif data_format == 'MS-DIAL':
        display_msdial_data_type_selection()
        quality_config_dict = display_quality_filtering_config()
        if quality_config_dict is not None:
            quality_config = QualityFilterConfig(
                total_score_threshold=quality_config_dict.get('total_score_threshold', 0),
                require_msms=quality_config_dict.get('require_msms', False)
            )

    return grade_config, quality_config, quality_config_dict


def run_ingestion_pipeline(df, experiment, bqc_label, data_format,
                           grade_config, quality_config, quality_config_dict):
    """
    Execute the ingestion workflow and update session state.

    Handles cached execution, session state updates, error/warning display,
    and MS-DIAL quality config change detection.

    Args:
        df: Standardized DataFrame
        experiment: ExperimentConfig
        bqc_label: BQC label string or None
        data_format: The selected data format string
        grade_config: GradeFilterConfig or None
        quality_config: QualityFilterConfig or None
        quality_config_dict: Raw quality config dict or None

    Returns:
        IngestionResult if successful, None if error or invalid
    """
    # Prepare cache parameters
    df_hash = StreamlitAdapter.compute_df_hash(df)
    experiment_dict = StreamlitAdapter.experiment_to_dict(experiment)
    format_enum = FORMAT_MAP.get(data_format)
    format_type_value = format_enum.value if format_enum else None
    grade_config_dict_for_cache = StreamlitAdapter.config_to_dict(grade_config)
    quality_config_dict_for_cache = StreamlitAdapter.config_to_dict(quality_config)

    # Capture previous config BEFORE running workflow (to detect config changes)
    prev_quality_config = st.session_state.get('last_quality_config')

    try:
        (
            cleaned_df, intsta_df, detected_format,
            is_valid, validation_errors, validation_warnings, cleaning_messages,
        ) = StreamlitAdapter.run_ingestion(
            _df_hash=df_hash,
            df=df,
            experiment_dict=experiment_dict,
            format_type=format_type_value,
            bqc_label=bqc_label,
            apply_zero_filter=False,
            grade_config_dict=grade_config_dict_for_cache,
            quality_config_dict=quality_config_dict_for_cache,
        )

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
        st.session_state.pre_filter_df = result.cleaned_df
        if quality_config_dict is not None:
            st.session_state.last_quality_config = quality_config_dict.copy()

        if result.is_valid:
            st.session_state.cleaned_df = result.cleaned_df
            st.session_state.intsta_df = result.internal_standards_df
            st.session_state.continuation_df = result.cleaned_df
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None

    # Display validation results
    if not result.is_valid:
        for error in result.validation_errors:
            st.error(error)
        return None

    for warning in result.validation_warnings:
        st.warning(warning)

    # MS-DIAL: rerun if quality config changed so expander shows updated results
    if data_format == 'MS-DIAL' and quality_config_dict:
        config_changed = not (
            prev_quality_config
            and prev_quality_config.get('total_score_threshold') == quality_config_dict.get('total_score_threshold')
            and prev_quality_config.get('require_msms') == quality_config_dict.get('require_msms')
        )
        if config_changed:
            st.rerun()

    return result


def display_final_filtered_data(cleaned_df: pd.DataFrame):
    """Display the final filtered data table with download button.

    Sorts by ClassKey if available and updates session state.
    """
    if 'ClassKey' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values('ClassKey').reset_index(drop=True)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.continuation_df = cleaned_df

    st.markdown("##### 📋 Final Filtered Data (Pre-Normalization)")
    st.dataframe(cleaned_df, use_container_width=True)
    csv = cleaned_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data",
        data=csv,
        file_name="final_filtered_data.csv",
        mime="text/csv",
        key="download_filtered_data"
    )
