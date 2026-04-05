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

import logging

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.download_utils import csv_download_button

from app.constants import (
    get_format_display_to_enum,
    LIPIDSEARCH_GRADE_OPTIONS,
    LIPIDSEARCH_DEFAULT_GRADES,
    LIPIDSEARCH_RELAXED_GRADE_CLASSES,
    LIPIDSEARCH_RELAXED_GRADES,
    MSDIAL_QUALITY_PRESETS,
    MSDIAL_DEFAULT_QUALITY_LEVEL,
)
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
                elif lipid_class in LIPIDSEARCH_RELAXED_GRADE_CLASSES:
                    default_grades = LIPIDSEARCH_RELAXED_GRADES
                else:
                    default_grades = LIPIDSEARCH_DEFAULT_GRADES

                selected_grades = st.multiselect(
                    f"**{lipid_class}**",
                    options=LIPIDSEARCH_GRADE_OPTIONS,
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
            st.session_state.msdial_quality_level = MSDIAL_DEFAULT_QUALITY_LEVEL

        if quality_filtering_available:
            quality_config = _display_score_filtering(msms_filtering_available)
        elif msms_filtering_available:
            quality_config = _display_msms_only_filtering()
        else:
            return None

        _display_quality_filter_summary(quality_config)
        _display_cached_filter_results(quality_config)
        return quality_config

    return None


def _display_score_filtering(msms_filtering_available: bool) -> dict:
    """Display quality score filtering with optional MS/MS and custom threshold."""
    quality_options_list = list(MSDIAL_QUALITY_PRESETS.keys())
    widget_key = "msdial_quality_level_radio"

    # Initialize widget key from persisted value BEFORE rendering
    persisted_value = st.session_state.get('msdial_quality_level', MSDIAL_DEFAULT_QUALITY_LEVEL)
    if persisted_value in quality_options_list:
        st.session_state[widget_key] = persisted_value

    def on_quality_level_change():
        st.session_state.msdial_quality_level = st.session_state[widget_key]
        # Sync MS/MS checkbox with the new preset value
        new_preset = MSDIAL_QUALITY_PRESETS.get(st.session_state[widget_key], {})
        st.session_state['msdial_custom_msms'] = new_preset.get('require_msms', False)

    selected_option = st.radio(
        "Quality filtering level:",
        quality_options_list,
        horizontal=True,
        key=widget_key,
        on_change=on_quality_level_change
    )
    st.session_state.msdial_quality_level = selected_option

    quality_config = MSDIAL_QUALITY_PRESETS[selected_option].copy()

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

    return quality_config


def _display_msms_only_filtering() -> dict:
    """Display MS/MS-only filtering when no quality score column is available."""
    require_msms = st.checkbox(
        "Require MS/MS validation",
        value=False,
        key="msdial_msms_only"
    )
    return {'total_score_threshold': 0, 'require_msms': require_msms}


def _display_quality_filter_summary(quality_config: dict):
    """Display current quality filter settings summary."""
    st.markdown("---")
    st.markdown(f"**Current settings:** Score ≥ {quality_config['total_score_threshold']}, "
               f"MS/MS required: {'Yes' if quality_config['require_msms'] else 'No'}")


def _display_cached_filter_results(quality_config: dict):
    """Display filter results from previous workflow run if config matches."""
    last_config = st.session_state.get('last_quality_config')
    if (last_config and
        last_config.total_score_threshold == quality_config['total_score_threshold'] and
        last_config.require_msms == quality_config['require_msms']):
        ingestion_result = st.session_state.get('ingestion_result')
        if ingestion_result and ingestion_result.cleaning_messages:
            st.markdown("**Filter Results:**")
            for msg in ingestion_result.cleaning_messages:
                st.info(msg)


# =============================================================================
# Pipeline Orchestration
# =============================================================================

FORMAT_MAP = None  # Lazy-loaded below


def _get_format_map():
    global FORMAT_MAP
    if FORMAT_MAP is None:
        FORMAT_MAP = get_format_display_to_enum()
    return FORMAT_MAP


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
    format_enum = _get_format_map().get(data_format)

    # Capture previous config BEFORE running workflow (to detect config changes)
    prev_quality_config = st.session_state.get('last_quality_config')

    try:
        (
            cleaned_df, intsta_df, detected_format,
            is_valid, validation_errors, validation_warnings, cleaning_messages,
        ) = StreamlitAdapter.run_ingestion(
            df=df,
            experiment=experiment,
            data_format=format_enum,
            bqc_label=bqc_label,
            apply_zero_filter=False,
            grade_config=grade_config,
            quality_config=quality_config,
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
        if quality_config is not None:
            st.session_state.last_quality_config = quality_config

        if result.is_valid:
            st.session_state.cleaned_df = result.cleaned_df
            st.session_state.intsta_df = result.internal_standards_df
            st.session_state.continuation_df = result.cleaned_df
    except (ValueError, KeyError) as e:
        logger.error("Data processing error: %s", e)
        st.error(f"Data processing failed: {e}")
        return None

    # Display validation results
    if not result.is_valid:
        for error in result.validation_errors:
            st.error(error)
        return None

    for warning in result.validation_warnings:
        st.warning(warning)

    # MS-DIAL: rerun if quality config changed so expander shows updated results
    if data_format == 'MS-DIAL' and quality_config is not None:
        config_changed = (prev_quality_config != quality_config)
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
    csv_download_button(cleaned_df, "final_filtered_data.csv", key="download_filtered_data")
