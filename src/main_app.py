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

# =============================================================================
# Imports - Refactored Components
# =============================================================================

from app.adapters.streamlit_adapter import StreamlitAdapter
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
)


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


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
        if data_format == 'MS-DIAL':
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
    # Main area: Processing Module (Automatic Flow)
    # ==========================================================================
    with center:
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
        pre_filter_df = st.session_state.get('pre_filter_df', result.cleaned_df)
        if pre_filter_df is not None and not pre_filter_df.empty:
            filtered_df, removed_species, zero_config = display_zero_filtering_config(
                pre_filter_df, experiment, bqc_label,
                data_format=st.session_state.get('format_type')
            )
            if filtered_df is not None:
                st.session_state.cleaned_df = filtered_df
                st.session_state.continuation_df = filtered_df

        # Step 5: Final filtered data
        display_final_filtered_data(st.session_state.cleaned_df)

        # Step 6: Manage Internal Standards
        auto_detected_intsta_df = st.session_state.get('intsta_df', result.internal_standards_df)
        intsta_df = display_manage_internal_standards(
            cleaned_df=st.session_state.cleaned_df,
            auto_detected_df=auto_detected_intsta_df
        )
        st.session_state.intsta_df = intsta_df

        # Step 7: Normalization
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
            st.rerun()


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
