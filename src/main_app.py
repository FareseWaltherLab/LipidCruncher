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

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.format_detection import DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.workflows.data_ingestion import IngestionResult
from app.ui.landing_page import display_landing_page, display_logo
from app.ui.format_requirements import display_format_requirements
from app.ui.zero_filtering import display_zero_filtering_config
from app.ui.sidebar import (
    display_format_selection,
    load_sample_dataset,
    display_file_upload,
    standardize_uploaded_data,
    display_column_mapping,
    display_sample_grouping,
)
from app.ui.main_content import (
    display_data_processing_docs,
    display_grade_filtering_config,
    display_msdial_data_type_selection,
    display_quality_filtering_config,
    display_manage_internal_standards,
    display_normalization_ui,
)


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
