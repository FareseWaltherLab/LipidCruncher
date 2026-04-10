"""
LipidCruncher - Lipidomics Data Analysis Application

Architecture:
    UI Layer (this file)
        → Workflows (app/workflows/)
            → Adapters (app/adapters/)
                → Services (app/services/)
                    → Models (app/models/)

"""

import logging
from typing import Optional

import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="LipidCruncher",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# =============================================================================
# Imports
# =============================================================================

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS, PAGE_LANDING, PAGE_APP, FORMAT_MSDIAL, Module, Page
from app.ui.landing_page import display_landing_page, display_logo
from app.ui.format_requirements import display_format_requirements
from app.ui.zero_filtering import display_zero_filtering_config
from app.ui.sidebar import (
    display_format_selection,
    display_file_upload,
    standardize_uploaded_data,
    display_column_mapping,
    display_sample_grouping,
)
from app.ui.main_content import (
    display_data_processing_docs,
    build_filter_configs,
    run_ingestion_pipeline,
    display_final_filtered_data,
    display_manage_internal_standards,
    display_normalization_ui,
    display_quality_check_module,
    display_analysis_module,
)


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


# =============================================================================
# Main App Page
# =============================================================================

def _reset_qc_state() -> None:
    """Clear all Quality Check session state."""
    StreamlitAdapter.reset_module_state(
        'qc_', '_preserved_bqc_', '_preserved_rt_', '_preserved_pca_',
    )


def _reset_analysis_state() -> None:
    """Clear all Module 3 analysis session state."""
    StreamlitAdapter.reset_module_state('analysis_')


def display_app_page() -> None:
    """Display the main application page with module routing."""
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
            st.info("Upload a dataset or load sample data from the sidebar (left panel) to begin.")

            # Back to landing button
            if st.button("← Back to Home", key="back_home_no_data"):
                st.session_state.page = PAGE_LANDING
                st.rerun()
        return

    # Standardize data if not already done
    if st.session_state.get('standardized_df') is None:
        standardized_df = standardize_uploaded_data(raw_df, data_format)
        if standardized_df is None:
            return
        st.session_state.standardized_df = standardized_df

        # For MS-DIAL: restore sample override widget after re-standardization
        # (re-validation resets msdial_features, so we need to ensure the
        # multiselect widget still has the overridden selection)
        if data_format == FORMAT_MSDIAL:
            override_samples = st.session_state.get('_msdial_override_samples')
            if override_samples:
                st.session_state['manual_sample_override'] = override_samples
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
    # Main area: Module routing
    # ==========================================================================
    current_module = st.session_state.get('module', MODULE_DATA_PROCESSING)

    if current_module == MODULE_DATA_PROCESSING:
        with center:
            _display_module1(df, raw_df, experiment, bqc_label, data_format)

    elif current_module == MODULE_QC_ANALYSIS:
        with center:
            _display_module2_and_3(experiment, bqc_label, data_format)

    else:
        logger.error("Unknown module: %s — falling back to data processing", current_module)
        st.session_state.module = MODULE_DATA_PROCESSING
        st.rerun()


# =============================================================================
# Module 1: Decomposed Steps
# =============================================================================

def _run_zero_filtering(result) -> None:
    """Step 4: Apply zero filtering to cleaned data."""
    pre_filter_df = st.session_state.get('pre_filter_df', result.cleaned_df)
    if pre_filter_df is not None and not pre_filter_df.empty:
        filtered_df, removed_species, zero_config = display_zero_filtering_config(
            pre_filter_df, st.session_state.get('experiment'), st.session_state.get('bqc_label'),
            data_format=st.session_state.get('format_type')
        )
        if filtered_df is not None:
            st.session_state.cleaned_df = filtered_df
            st.session_state.continuation_df = filtered_df


def _run_internal_standards(result) -> None:
    """Step 6: Manage internal standards."""
    auto_detected_intsta_df = st.session_state.get('intsta_df', result.internal_standards_df)
    intsta_df = display_manage_internal_standards(
        cleaned_df=st.session_state.cleaned_df,
        auto_detected_df=auto_detected_intsta_df
    )
    st.session_state.intsta_df = intsta_df


def _display_module1_navigation() -> None:
    """Module 1 navigation buttons."""
    st.markdown("---")
    normalized_df = st.session_state.get('normalized_df')
    if normalized_df is not None:
        if st.button("Next: Quality Check, Visualization & Analysis →"):
            _reset_qc_state()
            _reset_analysis_state()
            st.session_state.module = MODULE_QC_ANALYSIS
            st.rerun()

    if st.button("← Back to Home", key="back_home_module1"):
        st.session_state.page = PAGE_LANDING
        StreamlitAdapter.reset_data_state()
        st.rerun()


def _display_module1(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    experiment: 'ExperimentConfig',
    bqc_label: Optional[str],
    data_format: str,
) -> None:
    """Display Module 1: Data Standardization, Filtering, and Normalization."""
    st.subheader("Data Standardization, Filtering, and Normalization")

    # Step 1: Data processing documentation
    display_data_processing_docs(data_format)

    # Step 2: Format-specific filtering configuration
    grade_config, quality_config, quality_config_dict = build_filter_configs(data_format, raw_df)

    # Step 3: Run ingestion pipeline (cached)
    result = run_ingestion_pipeline(
        df, experiment, bqc_label, data_format,
        grade_config, quality_config, quality_config_dict
    )
    if result is None:
        return

    # Step 4: Zero Filtering Configuration (interactive)
    _run_zero_filtering(result)

    # Step 5: Final filtered data
    display_final_filtered_data(st.session_state.cleaned_df)

    # Step 6: Manage Internal Standards
    _run_internal_standards(result)

    # Step 7: Normalization
    display_normalization_ui(
        cleaned_df=st.session_state.cleaned_df,
        intsta_df=st.session_state.intsta_df,
        experiment=experiment,
        data_format=data_format
    )

    # Navigation
    _display_module1_navigation()


def _display_module2_and_3(
    experiment: 'ExperimentConfig',
    bqc_label: Optional[str],
    data_format: str,
) -> None:
    """Display Module 2 (Quality Check) and Module 3 (Visualize & Analyze) on the same page."""
    # Use normalized data as input for QC
    continuation_df = st.session_state.get('normalized_df')
    if continuation_df is None:
        st.error("No normalized data available. Please complete Module 1 first.")
        if st.button("← Back to Data Processing", key="back_processing_error"):
            st.session_state.module = MODULE_DATA_PROCESSING
            st.rerun()
        return

    # --- Module 2: Quality Check ---
    qc_df, updated_experiment = display_quality_check_module(
        continuation_df=continuation_df,
        experiment=experiment,
        bqc_label=bqc_label,
        format_type=data_format,
    )

    # Store results for downstream use
    st.session_state.qc_continuation_df = qc_df

    # --- Module 3: Visualize & Analyze ---
    st.markdown("---")

    # Use QC output as analysis input
    analysis_df = qc_df if qc_df is not None else continuation_df

    display_analysis_module(
        df=analysis_df,
        experiment=updated_experiment,
        bqc_label=bqc_label,
        format_type=data_format,
    )

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Data Processing", key="back_processing_module2"):
            _reset_qc_state()
            _reset_analysis_state()
            st.session_state.module = MODULE_DATA_PROCESSING
            st.rerun()
    with col2:
        if st.button("← Back to Home", key="back_home_module2"):
            st.session_state.page = PAGE_LANDING
            StreamlitAdapter.reset_data_state()
            st.rerun()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main application entry point."""
    page = st.session_state.get('page', PAGE_LANDING)

    if page == PAGE_LANDING:
        display_landing_page()
    elif page == PAGE_APP:
        display_app_page()
    else:
        logger.error("Unknown page: %s — resetting to landing", page)
        st.session_state.page = PAGE_LANDING
        st.rerun()


if __name__ == "__main__":
    main()