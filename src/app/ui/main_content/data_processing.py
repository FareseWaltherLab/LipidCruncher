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
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.download_utils import csv_download_button, named_dataframe

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL,
    resolve_format_enum,
    LIPIDSEARCH_GRADE_OPTIONS,
    LIPIDSEARCH_DEFAULT_GRADES,
    LIPIDSEARCH_RELAXED_GRADE_CLASSES,
    LIPIDSEARCH_RELAXED_GRADES,
    MSDIAL_QUALITY_PRESETS,
    MSDIAL_DEFAULT_QUALITY_LEVEL,
)
from app.services.data_cleaning import (
    ADDUCT_NOTE_PREFIX, ANY_ADDUCT, GradeFilterConfig, QualityFilterConfig,
)
from app.ui.content import get_processing_docs, ZERO_FILTERING_DOCS
from app.ui.sample_labels import names_for

logger = logging.getLogger(__name__)


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
        st.info(
            "ℹ️ The **default settings** (A/B for all classes, plus C for LPC and SM) "
            "are recommended if you are unsure. Choose **Allow C for all classes** to "
            "relax quality across the board, or **Customize by Class** for finer control."
        )

        # Initialize session state for persistence
        if 'grade_filter_mode' not in st.session_state:
            st.session_state.grade_filter_mode = "Use Default Settings"
        if 'grade_selections' not in st.session_state:
            st.session_state.grade_selections = {}

        options = ["Use Default Settings", "Allow C for all classes", "Customize by Class"]
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

        if use_custom == "Allow C for all classes":
            st.success("✓ Allowing grades A/B/C for every class (D excluded).")
            return {lipid_class: list(LIPIDSEARCH_RELAXED_GRADES) for lipid_class in all_classes}

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


_DEFAULT_ADDUCT = "Default"


def display_adduct_config(df: pd.DataFrame) -> Tuple[bool, Optional[Dict[str, str]]]:
    """LipidSearch adduct options, when the export names each row's adduct.

    Returns:
        Whether to keep species seen only in other adducts than their
        class's adduct, and the user's adduct per class (None: defaults).
    """
    if 'MainIon' not in df.columns:
        return False, None

    if 'keep_other_adducts' not in st.session_state:
        st.session_state.keep_other_adducts = False
    widget_key = "keep_other_adducts_checkbox"
    st.session_state[widget_key] = st.session_state.keep_other_adducts

    def on_change():
        st.session_state.keep_other_adducts = st.session_state[widget_key]

    with st.expander("⚙️ Configure Adducts", expanded=False):
        st.info(
            "ℹ️ A class with its own internal standard is quantified in the "
            "standard's adduct (e.g. PC as [M+H]+ when the PC standard is "
            "[M+H]+), so species and standard respond alike. Species of that "
            "class seen only in other adducts are **removed** by default: "
            "their signal is on a different scale from the standard's."
        )
        st.checkbox(
            "Keep species measured in other adducts",
            key=widget_key,
            on_change=on_change,
            help="Keeps more species, but their concentrations from the "
                 "standard are less accurate. The note after processing "
                 "lists them per class.",
        )
        class_adducts = _display_class_adduct_choices(df)
    return st.session_state.keep_other_adducts, class_adducts


def _display_class_adduct_choices(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Power users: choose the adduct each class is quantified from."""
    if 'customize_adducts' not in st.session_state:
        st.session_state.customize_adducts = False
    if 'adduct_choices' not in st.session_state:
        st.session_state.adduct_choices = {}
    widget_key = "customize_adducts_checkbox"
    st.session_state[widget_key] = st.session_state.customize_adducts

    def on_change():
        st.session_state.customize_adducts = st.session_state[widget_key]

    st.checkbox(
        "Choose the adduct per class",
        key=widget_key,
        on_change=on_change,
        help="**Default**: the class's internal standard's adduct, or each "
             "species' best-graded identification if the class has no "
             "standard of its own. An adduct: quantify the class in that "
             "adduct (species not seen in it follow the option above). "
             f"**{ANY_ADDUCT}**: each species' best-graded identification.",
    )
    if not st.session_state.customize_adducts:
        return None

    ions = df.assign(_ion=df['MainIon'].astype(str).str.strip())
    per_class = {
        cls: sorted(set(group['_ion']))
        for cls, group in ions.groupby('ClassKey')
        if group['_ion'].nunique() > 1
    }
    if not per_class:
        st.caption("Every class was measured in one adduct: nothing to choose.")
        return None

    choices = {}
    cols = st.columns(3)
    for idx, (cls, adducts) in enumerate(sorted(per_class.items())):
        options = [_DEFAULT_ADDUCT, *adducts, ANY_ADDUCT]
        saved = st.session_state.adduct_choices.get(cls, _DEFAULT_ADDUCT)
        with cols[idx % 3]:
            choice = st.selectbox(
                f"**{cls}**", options,
                index=options.index(saved) if saved in options else 0,
                key=f"adduct_select_{cls}",
            )
        st.session_state.adduct_choices[cls] = choice
        if choice != _DEFAULT_ADDUCT:
            choices[cls] = choice
    return choices or None


def display_msdial_data_type_selection():
    """Display MS-DIAL data type selection (raw vs pre-normalized)."""
    features = st.session_state.get('msdial_features', None)
    has_normalized_data = getattr(features, 'has_normalized_data', False)
    raw_samples = getattr(features, 'raw_sample_columns', [])
    norm_samples = getattr(features, 'normalized_sample_columns', [])

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
            # Clear column mapping so it gets rebuilt from fresh standardization
            # (otherwise the stale override mapping causes incorrect column lookups)
            st.session_state.column_mapping = None
            st.session_state.n_intensity_cols = None
            # Drop the sample-regroup snapshot. It was captured under the old
            # raw/normalized selection; left in place, the regroup step would
            # re-apply it and silently revert this toggle (showing raw values).
            StreamlitAdapter.drop_regroup_snapshot()

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
    features = st.session_state.get('msdial_features', None)
    quality_filtering_available = getattr(features, 'has_quality_score', False)
    msms_filtering_available = getattr(features, 'has_msms_matched', False)

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

    if data_format == FORMAT_LIPIDSEARCH:
        grade_config_dict = display_grade_filtering_config(raw_df)
        keep_other_adducts, class_adducts = display_adduct_config(raw_df)
        if grade_config_dict is not None or keep_other_adducts or class_adducts:
            grade_config = GradeFilterConfig(
                grade_config=grade_config_dict,
                keep_other_adducts=keep_other_adducts,
                class_adducts=class_adducts,
            )

    elif data_format == FORMAT_MSDIAL:
        display_msdial_data_type_selection()
        quality_config_dict = display_quality_filtering_config()
        if quality_config_dict is not None:
            quality_config = QualityFilterConfig(
                total_score_threshold=quality_config_dict.get('total_score_threshold', 0),
                require_msms=quality_config_dict.get('require_msms', False)
            )

    return grade_config, quality_config, quality_config_dict


def run_ingestion_pipeline(df, experiment, data_format,
                           grade_config, quality_config, quality_config_dict):
    """
    Execute the ingestion workflow and update session state.

    Handles cached execution, session state updates, error/warning display,
    and MS-DIAL quality config change detection.

    Args:
        df: Standardized DataFrame
        experiment: ExperimentConfig
        data_format: The selected data format string
        grade_config: GradeFilterConfig or None
        quality_config: QualityFilterConfig or None
        quality_config_dict: Raw quality config dict or None

    Returns:
        IngestionResult if successful, None if error or invalid
    """
    format_enum = resolve_format_enum(data_format)

    # Capture previous config BEFORE running workflow (to detect config changes)
    prev_quality_config = st.session_state.get('last_quality_config')

    try:
        result = StreamlitAdapter.run_ingestion(
            df=df,
            experiment=experiment,
            data_format=format_enum,
            grade_config=grade_config,
            quality_config=quality_config,
        )

        st.session_state.ingestion_result = result
        st.session_state.grade_config = grade_config
        if quality_config is not None:
            st.session_state.last_quality_config = quality_config
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

    # LipidSearch: which adduct each class was quantified from, what was
    # removed for not matching its standard, and which classes mix adducts.
    for msg in result.cleaning_messages:
        if msg.startswith(ADDUCT_NOTE_PREFIX):
            st.info(msg)

    # MS-DIAL: rerun if quality config changed so expander shows updated results
    if data_format == FORMAT_MSDIAL and quality_config is not None:
        config_changed = (prev_quality_config != quality_config)
        if config_changed:
            st.rerun()

    return result


def display_final_filtered_data(cleaned_df: pd.DataFrame):
    """Display the final filtered data table with download button.

    ``cleaned_df`` is what normalization receives
    (``DataIngestionWorkflow.analysis_frame``).
    """
    st.markdown("##### 📋 Final Filtered Data (Pre-Normalization)")
    named_dataframe(
        cleaned_df, sample_names=names_for('processing'),
        width='stretch',
    )
    csv_download_button(
        cleaned_df, "final_filtered_data.csv", key="download_filtered_data",
        sample_names=names_for('processing'),
    )
