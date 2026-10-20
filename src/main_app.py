"""
LipidCruncher - Lipidomics Data Analysis Application

Architecture:
    UI Layer (this file, app/ui/)
        → Adapter (app/adapters/)
            → Workflows (app/workflows/)
                → Services (app/services/)
                    → Models (app/models/)

"""

import logging
from typing import Optional, Tuple

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
from app.workflows.data_ingestion import DataIngestionWorkflow
from app.constants import (
    MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS, PAGE_LANDING, PAGE_APP, FORMAT_MSDIAL,
    FORMAT_METABOLOMICS_WORKBENCH,
)
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.experiment import ExperimentConfig


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


# =============================================================================
# Main App Page
# =============================================================================

def _display_session_reset_button() -> None:
    """Sidebar control that soft-resets the session while keeping the upload.

    Clears experiment setup, processing, QC/analysis state and the cache, then
    returns the user to the Define Experiment step (via rerun, no page refresh).
    """
    if st.sidebar.button(
        "🔄 Reset session (keep uploaded data)",
        key="reset_session",
        help="Clear experiment setup, processing, and cached results, then "
             "return to the Define Experiment step. Your uploaded data is kept.",
        width='stretch',
    ):
        StreamlitAdapter.reset_to_experiment_setup()
        st.rerun()
    st.sidebar.markdown("---")


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

    # A format change re-standardizes the same upload; state derived from the
    # previous format's standardization would otherwise be reused. Before the
    # upload step, which itself parses Metabolomics Workbench text: run after
    # it, the reset wiped that parse in the same run.
    cached_format = st.session_state.get('format_type')
    if cached_format is not None and cached_format != data_format:
        StreamlitAdapter.reset_format_derived_state()

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

    # Workbench uploads are parsed from text before this point, so there is
    # no raw table left to compare.
    if data_format != FORMAT_METABOLOMICS_WORKBENCH:
        looks_like = StreamlitAdapter.detect_format_mismatch(raw_df, data_format)
        if looks_like is not None:
            st.sidebar.warning(
                f"This file looks like **{looks_like.value}** data, but "
                f"**{data_format}** is selected above. If so, switch the format "
                "so the file is read correctly."
            )

    # Sidebar: session reset — available from any step once data is loaded.
    _display_session_reset_button()

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

    # After sample grouping, use the frame Module 1 should run on: the
    # (possibly regrouped) standardized data minus any samples excluded in
    # the sidebar, with survivors renumbered to match `experiment`.
    df = st.session_state.processing_frame.df

    # ==========================================================================
    # Main area: Module routing
    # ==========================================================================
    current_module = st.session_state.get('module', MODULE_DATA_PROCESSING)

    if current_module == MODULE_DATA_PROCESSING:
        with center:
            _display_module1(df, raw_df, experiment, data_format)

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

def _run_zero_filtering(result) -> pd.DataFrame:
    """Step 4: Apply zero filtering to the cleaned data; return the result."""
    cleaned_df = result.cleaned_df
    if cleaned_df is None or cleaned_df.empty:
        return cleaned_df
    filtered_df, removed_species, zero_config = display_zero_filtering_config(
        cleaned_df, st.session_state.get('experiment'), st.session_state.get('bqc_label'),
        data_format=st.session_state.get('format_type')
    )
    return cleaned_df if filtered_df is None else filtered_df


def _display_module1_navigation() -> None:
    """Module 1 navigation buttons."""
    st.markdown("---")
    normalized_df = st.session_state.get('normalized_df')
    if normalized_df is not None:
        if st.button("Next: Quality Check, Visualization & Analysis →"):
            StreamlitAdapter.go_to_quality_check()
            st.rerun()

    if st.button("← Back to Home", key="back_home_module1"):
        StreamlitAdapter.back_to_home()
        st.rerun()


def _display_module1(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    experiment: 'ExperimentConfig',
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
        df, experiment, data_format,
        grade_config, quality_config, quality_config_dict
    )
    if result is None:
        return

    # Step 4: Zero Filtering Configuration (interactive)
    filtered_df = _run_zero_filtering(result)

    # Step 5: Final filtered data. Shown here, but filled after step 6: the
    # standards chosen there leave the data, so only then is it final.
    final_data_slot = st.container()

    # Step 6: Manage Internal Standards
    intsta_df = display_manage_internal_standards(
        cleaned_df=filtered_df,
        auto_detected_df=result.internal_standards_df,
    )
    cleaned_df = DataIngestionWorkflow.analysis_frame(filtered_df, intsta_df)
    st.session_state.cleaned_df = cleaned_df
    st.session_state.intsta_df = intsta_df
    with final_data_slot:
        display_final_filtered_data(cleaned_df)

    # Step 7: Normalization
    display_normalization_ui(
        cleaned_df=cleaned_df,
        intsta_df=intsta_df,
        experiment=experiment,
        data_format=data_format
    )

    # Navigation
    _display_module1_navigation()


def _module1_handoff(
    experiment: 'ExperimentConfig', bqc_label: Optional[str],
) -> Tuple['ExperimentConfig', Optional[str]]:
    """Return the sidebar setup Module 1 ran with, frozen on entry to Module 2.

    The sidebar keeps running here, so an exclusion or regroup made now would
    pair a new experiment with the normalized data Module 1 already computed.
    Module 2/3 keep the setup that produced that data and say how to apply
    the change.
    """
    state = st.session_state
    if state.get('qc_input_frame') is None:
        state.qc_input_frame = state.processing_frame
        state.qc_input_bqc_label = bqc_label
    elif (
        experiment != state.qc_input_frame.experiment
        or bqc_label != state.qc_input_bqc_label
        or not state.processing_frame.df.equals(state.qc_input_frame.df)
    ):
        st.warning(
            "The sidebar setup changed after Module 1 ran. Quality Check and "
            "the analyses below still use the samples and grouping Module 1 "
            "processed; go **← Back to Data Processing** to apply the change."
        )
    return state.qc_input_frame.experiment, state.qc_input_bqc_label


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

    experiment, bqc_label = _module1_handoff(experiment, bqc_label)

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
            StreamlitAdapter.back_to_data_processing()
            st.rerun()
    with col2:
        if st.button("← Back to Home", key="back_home_module2"):
            StreamlitAdapter.back_to_home()
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