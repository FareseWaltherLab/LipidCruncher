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
SAMPLE_DATA_DIR = BASE_DIR.parent / "sample_datasets"

# =============================================================================
# Imports - Refactored Components
# =============================================================================

from app.adapters.streamlit_adapter import StreamlitAdapter, SessionState
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.services.format_detection import FormatDetectionService, DataFormat
from app.workflows.data_ingestion import DataIngestionWorkflow, IngestionConfig
from app.workflows.normalization import NormalizationWorkflow, NormalizationWorkflowConfig, NormalizationWorkflowResult
from app.ui.landing_page import display_landing_page, display_logo
from app.ui.format_requirements import display_format_requirements


# =============================================================================
# Session State Initialization
# =============================================================================

StreamlitAdapter.initialize_session_state()


# =============================================================================
# UI Components - Format Selection & File Upload
# =============================================================================

def display_format_selection() -> str:
    """Display format selection dropdown in sidebar."""
    return st.sidebar.selectbox(
        'Select Data Format',
        ['Generic Format', 'Metabolomics Workbench', 'LipidSearch 5.0', 'MS-DIAL']
    )


def load_sample_dataset(data_format: str) -> pd.DataFrame:
    """Load a sample dataset for the selected format."""
    sample_files = {
        'LipidSearch 5.0': 'lipidsearch5_sample_dataset.csv',
        'MS-DIAL': 'msdial_test_dataset.csv',
        'Generic Format': 'generic_sample_dataset.csv',
        'Metabolomics Workbench': 'metabolomic_workbench_sample_data.csv',
    }

    filename = sample_files.get(data_format)
    if filename:
        filepath = SAMPLE_DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
    return None


def display_file_upload(data_format: str) -> pd.DataFrame:
    """Display file upload widget and sample data option."""

    # Sample data option
    with st.sidebar.expander("Try Sample Data", expanded=False):
        if st.button(f"Load {data_format} Sample", key="load_sample"):
            sample_df = load_sample_dataset(data_format)
            if sample_df is not None:
                st.session_state.using_sample_data = True
                st.session_state.raw_df = sample_df
                st.sidebar.success(f"Loaded sample dataset!")
                return sample_df

    # Check if using sample data
    if st.session_state.get('using_sample_data') and st.session_state.get('raw_df') is not None:
        if st.sidebar.button("Clear & Upload Your Data"):
            st.session_state.using_sample_data = False
            StreamlitAdapter.reset_data_state()
            st.experimental_rerun()
        return st.session_state.raw_df

    # File upload
    file_types = ['csv'] if data_format == 'Metabolomics Workbench' else ['csv', 'txt']
    uploaded_file = st.sidebar.file_uploader(
        f'Upload your {data_format} dataset',
        type=file_types,
        help="Limit 800MB per file"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None

    return None


# =============================================================================
# UI Components - Sample Grouping
# =============================================================================

def detect_sample_columns(df: pd.DataFrame, data_format: str) -> list:
    """Detect intensity/sample columns based on format."""
    if data_format == 'LipidSearch 5.0':
        return [col for col in df.columns if col.startswith('MeanArea[')]
    elif data_format == 'MS-DIAL':
        # MS-DIAL: columns after metadata are samples
        metadata_cols = FormatDetectionService.MSDIAL_METADATA_COLUMNS
        return [col for col in df.columns if col not in metadata_cols]
    else:
        # Generic: columns after LipidMolec/ClassKey
        metadata = {'LipidMolec', 'ClassKey'}
        return [col for col in df.columns if col not in metadata]


def extract_sample_names(columns: list, data_format: str) -> list:
    """Extract clean sample names from column names."""
    if data_format == 'LipidSearch 5.0':
        # Extract name from MeanArea[name]
        names = []
        for col in columns:
            if col.startswith('MeanArea[') and col.endswith(']'):
                names.append(col[9:-1])  # Remove MeanArea[ and ]
            else:
                names.append(col)
        return names
    return columns


def display_sample_grouping(df: pd.DataFrame, data_format: str):
    """Display sample grouping UI in sidebar."""
    st.sidebar.subheader('Group Samples')

    # Detect sample columns
    sample_cols = detect_sample_columns(df, data_format)
    sample_names = extract_sample_names(sample_cols, data_format)

    if not sample_cols:
        st.sidebar.error("No sample columns detected!")
        return None, None

    st.sidebar.info(f"Detected {len(sample_cols)} samples")

    # Display sample list
    with st.sidebar.expander("View Samples", expanded=False):
        for i, name in enumerate(sample_names, 1):
            st.text(f"{i}. {name}")

    # Experiment definition
    st.sidebar.subheader("Define Experiment")

    n_conditions = st.sidebar.number_input(
        'Number of conditions',
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    conditions_list = []
    number_of_samples_list = []

    for i in range(n_conditions):
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            cond_name = st.text_input(
                f'Condition {i + 1}',
                value=f'Condition_{i + 1}',
                key=f'cond_name_{i}'
            )
            conditions_list.append(cond_name)
        with col2:
            n_samples = st.number_input(
                'Samples',
                min_value=1,
                max_value=len(sample_cols),
                value=min(3, len(sample_cols)),
                key=f'n_samples_{i}'
            )
            number_of_samples_list.append(n_samples)

    # Validate total samples
    total_assigned = sum(number_of_samples_list)
    if total_assigned != len(sample_cols):
        st.sidebar.warning(
            f"Assigned {total_assigned} samples but dataset has {len(sample_cols)}. "
            "Please adjust sample counts."
        )
        return None, None

    # BQC detection
    bqc_label = None
    for cond in conditions_list:
        if 'bqc' in cond.lower() or 'qc' in cond.lower():
            bqc_label = cond
            break

    if bqc_label:
        st.sidebar.info(f"Detected BQC condition: {bqc_label}")

    # Confirm button
    if st.sidebar.button("Confirm Experiment Setup", type="primary"):
        try:
            experiment = ExperimentConfig(
                n_conditions=n_conditions,
                conditions_list=conditions_list,
                number_of_samples_list=number_of_samples_list
            )
            st.session_state.experiment = experiment
            st.session_state.bqc_label = bqc_label
            st.session_state.confirmed = True
            st.sidebar.success("Experiment confirmed!")
            return experiment, bqc_label
        except Exception as e:
            st.sidebar.error(f"Invalid experiment setup: {e}")
            return None, None

    # Return existing experiment if already confirmed
    if st.session_state.get('confirmed'):
        return st.session_state.get('experiment'), st.session_state.get('bqc_label')

    return None, None


# =============================================================================
# UI Components - Data Processing Display
# =============================================================================

def display_data_preview(df: pd.DataFrame, title: str = "Data Preview"):
    """Display a preview of the DataFrame."""
    with st.expander(title, expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df)} rows, {len(df.columns)} columns")


def display_processing_results(result):
    """Display results from DataIngestionWorkflow."""
    if not result.is_valid:
        for error in result.validation_errors:
            st.error(error)
        return

    # Success message
    st.success(f"Data cleaned successfully! Format: {result.detected_format.value}")

    # Cleaning messages
    if result.cleaning_messages:
        with st.expander("Processing Details", expanded=False):
            for msg in result.cleaning_messages:
                st.info(msg)

    # Zero filtering stats
    if result.zero_filtered:
        st.info(
            f"Zero filtering: {result.species_before_filter} → {result.species_after_filter} species "
            f"({result.species_removed_count} removed, {result.removal_percentage:.1f}%)"
        )

    # Warnings
    for warning in result.validation_warnings:
        st.warning(warning)

    # Display cleaned data
    if result.cleaned_df is not None:
        display_data_preview(result.cleaned_df, "Cleaned Data")

    # Internal standards
    if result.internal_standards_df is not None and not result.internal_standards_df.empty:
        with st.expander(f"Internal Standards ({len(result.internal_standards_df)} detected)", expanded=False):
            st.dataframe(result.internal_standards_df, use_container_width=True)


# =============================================================================
# UI Components - Normalization
# =============================================================================

def display_normalization_ui(cleaned_df: pd.DataFrame, intsta_df: pd.DataFrame, experiment: ExperimentConfig, data_format: str):
    """Display normalization options and apply normalization."""
    st.subheader("Normalization")

    # Get available classes
    available_classes = NormalizationWorkflow.get_available_classes(cleaned_df)

    # Class selection
    with st.expander("Select Lipid Classes", expanded=True):
        select_all = st.checkbox("Select All Classes", value=True)
        if select_all:
            selected_classes = available_classes
        else:
            selected_classes = st.multiselect(
                "Classes to include:",
                options=available_classes,
                default=available_classes
            )

    if not selected_classes:
        st.warning("Please select at least one lipid class.")
        return None

    # Normalization method selection
    st.markdown("**Normalization Method**")
    method = st.radio(
        "Choose a method:",
        options=['None', 'Internal Standards', 'Protein Concentration', 'Both'],
        horizontal=True,
        help="'Both' applies internal standards first, then protein concentration."
    )

    # Map UI method names to config values
    method_map = {
        'None': 'none',
        'Internal Standards': 'internal_standard',
        'Protein Concentration': 'protein',
        'Both': 'both'
    }
    config_method = method_map[method]

    # Internal standards configuration
    internal_standards = {}
    intsta_concentrations = {}

    if method in ['Internal Standards', 'Both']:
        st.markdown("---")
        st.markdown("**Internal Standards Configuration**")

        if intsta_df is None or intsta_df.empty:
            st.error("No internal standards detected. Please upload standards or use a different method.")
            return None

        # Get standards by class
        standards_by_class = NormalizationWorkflow.get_standards_by_class(intsta_df)
        all_standards = NormalizationWorkflow.get_available_standards(intsta_df)

        if not all_standards:
            st.error("No internal standards available.")
            return None

        st.info(f"Detected {len(all_standards)} internal standards")

        # Show mapping UI
        with st.expander("Standard-to-Class Mapping", expanded=True):
            for lipid_class in selected_classes:
                class_standards = standards_by_class.get(lipid_class, [])
                default_std = class_standards[0] if class_standards else all_standards[0]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text(lipid_class)
                with col2:
                    selected_std = st.selectbox(
                        f"Standard for {lipid_class}",
                        options=all_standards,
                        index=all_standards.index(default_std) if default_std in all_standards else 0,
                        key=f"std_{lipid_class}",
                        label_visibility="collapsed"
                    )
                    internal_standards[lipid_class] = selected_std

        # Standard concentrations (optional)
        with st.expander("Standard Concentrations (optional)", expanded=False):
            st.caption("Leave at 1.0 for relative normalization, or enter actual concentrations.")
            for std in set(internal_standards.values()):
                conc = st.number_input(
                    f"{std} concentration:",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1,
                    key=f"conc_{std}"
                )
                intsta_concentrations[std] = conc

    # Protein concentration configuration
    protein_concentrations = {}

    if method in ['Protein Concentration', 'Both']:
        st.markdown("---")
        st.markdown("**Protein Concentrations**")

        # Get sample names from experiment
        sample_names = experiment.full_samples_list

        with st.expander("Enter Protein Concentrations", expanded=True):
            st.caption("Enter protein concentration for each sample (e.g., mg/mL).")

            # Input method choice
            input_method = st.radio(
                "Input method:",
                ["Manual entry", "Same for all"],
                horizontal=True
            )

            if input_method == "Same for all":
                default_conc = st.number_input(
                    "Protein concentration for all samples:",
                    min_value=0.0001,
                    value=1.0,
                    step=0.1
                )
                for sample in sample_names:
                    protein_concentrations[sample] = default_conc
            else:
                # Group by condition for easier entry
                for i, condition in enumerate(experiment.conditions_list):
                    st.markdown(f"**{condition}**")
                    start_idx = sum(experiment.number_of_samples_list[:i])
                    end_idx = start_idx + experiment.number_of_samples_list[i]
                    condition_samples = sample_names[start_idx:end_idx]

                    cols = st.columns(min(len(condition_samples), 4))
                    for j, sample in enumerate(condition_samples):
                        with cols[j % len(cols)]:
                            conc = st.number_input(
                                sample,
                                min_value=0.0001,
                                value=1.0,
                                step=0.1,
                                key=f"protein_{sample}"
                            )
                            protein_concentrations[sample] = conc

    # Apply normalization button
    st.markdown("---")
    if st.button("Apply Normalization", type="primary"):
        with st.spinner("Normalizing data..."):
            try:
                # Create normalization config
                norm_config = NormalizationConfig(
                    method=config_method,
                    selected_classes=selected_classes,
                    internal_standards=internal_standards if internal_standards else None,
                    intsta_concentrations=intsta_concentrations if intsta_concentrations else None,
                    protein_concentrations=protein_concentrations if protein_concentrations else None
                )

                # Map UI format to DataFormat enum
                format_map = {
                    'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
                    'MS-DIAL': DataFormat.MSDIAL,
                    'Generic Format': DataFormat.GENERIC,
                    'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
                }

                # Create workflow config
                workflow_config = NormalizationWorkflowConfig(
                    experiment=experiment,
                    normalization=norm_config,
                    data_format=format_map.get(data_format, DataFormat.GENERIC)
                )

                # Run workflow
                result = NormalizationWorkflow.run(
                    df=cleaned_df,
                    config=workflow_config,
                    intsta_df=intsta_df
                )

                if result.success:
                    st.session_state.normalization_result = result
                    st.session_state.normalized_df = result.normalized_df
                    st.session_state.continuation_df = result.normalized_df
                    st.success(f"Normalization complete! Method: {result.method_applied}")
                else:
                    for error in result.validation_errors:
                        st.error(error)

            except Exception as e:
                st.error(f"Normalization error: {e}")

    # Display normalization results
    if st.session_state.get('normalization_result'):
        result = st.session_state.normalization_result
        if result.success:
            # Show statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Lipids", result.lipids_after)
            col2.metric("Samples", result.samples_processed)
            col3.metric("Standards Removed", len(result.removed_standards))

            if result.removed_standards:
                with st.expander("Removed Standards", expanded=False):
                    for std in result.removed_standards:
                        st.text(f"• {std}")

            # Show normalized data preview
            if result.normalized_df is not None:
                display_data_preview(result.normalized_df, "Normalized Data")

    return st.session_state.get('normalized_df')


# =============================================================================
# Main App Page
# =============================================================================

def display_app_page():
    """Display the main application page (Module 1)."""
    display_logo()
    st.markdown("Process, analyze and visualize lipidomic data from multiple sources.")

    # Sidebar: Format selection
    data_format = display_format_selection()
    display_format_requirements(data_format)

    # Sidebar: File upload
    df = display_file_upload(data_format)

    if df is None:
        st.info("Upload a dataset or load sample data to begin.")

        # Back to landing button
        if st.button("← Back to Home"):
            st.session_state.page = 'landing'
            st.experimental_rerun()
        return

    # Show data preview
    display_data_preview(df, "Uploaded Data")

    # Sidebar: Sample grouping
    experiment, bqc_label = display_sample_grouping(df, data_format)

    if experiment is None:
        st.info("Configure your experiment in the sidebar to proceed.")
        return

    # Main area: Processing Module
    st.subheader("Data Standardization, Filtering, and Normalization")

    # Process data button
    if st.button("Process Data", type="primary"):
        with st.spinner("Processing data..."):
            # Map UI format to DataFormat enum
            format_map = {
                'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
                'MS-DIAL': DataFormat.MSDIAL,
                'Generic Format': DataFormat.GENERIC,
                'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
            }

            config = IngestionConfig(
                experiment=experiment,
                data_format=format_map.get(data_format),
                bqc_label=bqc_label,
                apply_zero_filter=True,
            )

            try:
                result = DataIngestionWorkflow.run(df, config)
                st.session_state.ingestion_result = result

                if result.is_valid:
                    st.session_state.cleaned_df = result.cleaned_df
                    st.session_state.intsta_df = result.internal_standards_df
                    st.session_state.continuation_df = result.cleaned_df
            except Exception as e:
                st.error(f"Processing error: {e}")

    # Display results if available
    if st.session_state.get('ingestion_result'):
        result = st.session_state.ingestion_result
        display_processing_results(result)

        # Normalization UI (only if ingestion was successful)
        if result.is_valid and result.cleaned_df is not None:
            st.markdown("---")
            display_normalization_ui(
                cleaned_df=result.cleaned_df,
                intsta_df=result.internal_standards_df,
                experiment=experiment,
                data_format=data_format
            )

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = 'landing'
            StreamlitAdapter.reset_data_state()
            st.experimental_rerun()


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
