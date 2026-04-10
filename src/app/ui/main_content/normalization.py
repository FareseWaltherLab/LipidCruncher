"""
Normalization UI Components for LipidCruncher main content area.

This module contains:
- display_normalization_ui: Main normalization orchestration UI
- _display_class_selection: Lipid class selection widget
- _display_internal_standards_config: Standards mapping and concentrations
- _display_protein_config: Protein concentration input (manual or CSV)
- _run_normalization: Cached normalization workflow execution
"""

import logging
from typing import Optional

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

from app.models.experiment import ExperimentConfig
from app.workflows.normalization import NormalizationWorkflow, NormalizationWorkflowResult
from app.constants import FORMAT_MSDIAL, resolve_format_enum
from app.services.format_detection import DataFormat
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.content import NORMALIZATION_METHODS_DOCS, PROTEIN_CSV_HELP
from app.ui.download_utils import csv_download_button


# =============================================================================
# Helper Functions - Class Selection
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


# =============================================================================
# Helper Functions - Internal Standards Configuration
# =============================================================================

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


# =============================================================================
# Helper Functions - Protein Configuration
# =============================================================================

def _display_manual_protein_input(sample_names: list) -> dict:
    """Display manual protein concentration input as a 3-column grid.

    Returns protein_concentrations dict.
    """
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
    protein_concentrations = {}
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

    st.session_state.protein_df = protein_concentrations.copy()
    return protein_concentrations


def _display_csv_protein_upload(sample_names: list) -> dict:
    """Display CSV upload for protein concentrations.

    Returns protein_concentrations dict or None.
    """
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

            protein_concentrations = dict(zip(sample_names, csv_df['Concentration'].tolist()))
            st.session_state.protein_df = protein_concentrations.copy()
            st.success(f"✓ Loaded {len(protein_concentrations)} concentration values")
            return protein_concentrations

        except UnicodeDecodeError as e:
            logger.error("Protein CSV encoding error: %s", e)
            st.error(
                "Could not read the protein concentration CSV. The file may use an unsupported encoding. "
                "Please save it as a UTF-8 encoded CSV and try again."
            )
            return None
        except pd.errors.EmptyDataError as e:
            logger.error("Empty protein CSV: %s", e)
            st.error("The uploaded protein concentration CSV appears to be empty. Please check the file and try again.")
            return None
        except pd.errors.ParserError as e:
            logger.error("Protein CSV parse error: %s", e)
            st.error(
                "Could not parse the protein concentration CSV. The file may be corrupted or improperly "
                "formatted. Please ensure it is a valid CSV file with proper delimiters."
            )
            return None
        except KeyError as e:
            logger.error("Protein CSV missing column: %s", e)
            st.error(f"Missing required column in protein CSV: {e}. Expected a 'Concentration' column.")
            return None
        except ValueError as e:
            logger.error("Protein CSV value error: %s", e)
            st.error(f"Invalid values in protein CSV: {e}")
            return None

    # No new file uploaded - check preserved data
    if preserved_protein_df is not None and isinstance(preserved_protein_df, dict):
        if len(preserved_protein_df) == len(sample_names):
            st.success(f"✓ Using previously loaded {len(preserved_protein_df)} concentration values")
            return preserved_protein_df

    st.info("Please upload a CSV file with protein concentrations.")
    return None


def _display_protein_config(experiment: ExperimentConfig) -> dict:
    """Display protein concentration configuration UI. Returns protein_concentrations dict or None."""

    sample_names = experiment.full_samples_list

    with st.expander("⚙️ Protein Concentration Data", expanded=True):
        # Restore method selection from preserved state (lost during module navigation)
        if st.session_state.get('protein_input_method') is None:
            preserved = st.session_state.get('protein_input_method_prev')
            st.session_state.protein_input_method = (
                preserved if preserved in ["Manual Input", "Upload CSV File"]
                else "Manual Input"
            )

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
            return _display_manual_protein_input(sample_names)
        else:
            return _display_csv_protein_upload(sample_names)


# =============================================================================
# Helper Functions - Run Normalization
# =============================================================================

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
        from app.models.normalization import NormalizationConfig

        norm_config = NormalizationConfig(
            method=config_method,
            selected_classes=selected_classes,
            internal_standards=internal_standards if internal_standards else None,
            intsta_concentrations=intsta_concentrations if intsta_concentrations else None,
            protein_concentrations=protein_concentrations if protein_concentrations else None,
        )

        result = StreamlitAdapter.run_normalization(
            df=cleaned_df,
            experiment=experiment,
            normalization=norm_config,
            data_format=resolve_format_enum(data_format),
            intsta_df=intsta_df,
        )

        if result.success:
            st.session_state.normalization_result = result
            st.session_state.normalized_df = result.normalized_df
            st.session_state.continuation_df = result.normalized_df

        return result

    except (ValueError, KeyError) as e:
        logger.error("Normalization error: %s", e)
        st.error(f"Normalization failed: {e}")
        return None


# =============================================================================
# UI Components - Normalization
# =============================================================================

def display_normalization_ui(cleaned_df: pd.DataFrame, intsta_df: pd.DataFrame, experiment: ExperimentConfig, data_format: str) -> Optional[pd.DataFrame]:
    """Display normalization options and apply normalization automatically."""
    with st.expander("📖 About Normalization Methods", expanded=False):
        st.markdown(NORMALIZATION_METHODS_DOCS)
        st.info(
            "**When to use Total Intensity normalization:** This method corrects for differences in sample "
            "loading or injection volume without requiring internal standards or protein concentrations. It is "
            "best suited for experiments where the total lipid content is not expected to change between "
            "conditions (e.g., comparing lipid profiles across similar tissue samples)."
        )
        st.warning(
            "**Caution:** Total Intensity normalization converts intensities to relative proportions. If total "
            "lipid content truly differs between conditions (e.g., lipodystrophy models, DGAT knockouts, "
            "starvation vs. fed), a real change in one abundant class will cause apparent changes in all other "
            "classes — even if they did not change. For such experiments, use Internal Standards or "
            "Protein-based normalization instead."
        )

    # Class selection
    st.markdown("##### 🎯 Select Lipid Classes")
    selected_classes = _display_class_selection(cleaned_df)
    if not selected_classes:
        st.warning("Please select at least one lipid class.")
        return None

    # Method selection
    normalization_options = _get_normalization_options(intsta_df, data_format)
    method = _display_method_selection(normalization_options)

    method_map = {
        'None (pre-normalized data)': 'none',
        'Internal Standards': 'internal_standard',
        'Protein-based': 'protein',
        'Internal Standards + Protein': 'both',
        'Total Intensity': 'total_intensity',
    }
    config_method = method_map.get(method, 'none')

    # Collect configuration based on method
    internal_standards, intsta_concentrations, protein_concentrations = (
        _collect_method_config(method, intsta_df, selected_classes, experiment)
    )
    if internal_standards is None or protein_concentrations is None:
        return None

    # Apply normalization
    result = _run_normalization(
        cleaned_df, intsta_df, experiment, data_format,
        config_method, selected_classes,
        internal_standards, intsta_concentrations, protein_concentrations
    )

    if result is not None:
        _display_normalization_results(result)
    return st.session_state.get('normalized_df')


def _get_normalization_options(intsta_df, data_format: str) -> list:
    """Determine available normalization options based on standards and data format."""
    has_standards = intsta_df is not None and not intsta_df.empty
    is_msdial_prenormalized = (data_format == FORMAT_MSDIAL and
                               st.session_state.get('msdial_use_normalized', False))

    if is_msdial_prenormalized:
        st.markdown("*Internal standards options unavailable — using pre-normalized MS-DIAL data.*")
        return ['None (pre-normalized data)', 'Protein-based', 'Total Intensity']
    elif has_standards:
        return ['None (pre-normalized data)', 'Internal Standards', 'Protein-based', 'Internal Standards + Protein', 'Total Intensity']
    else:
        st.markdown("*Internal standards options unavailable — no standards detected or uploaded.*")
        return ['None (pre-normalized data)', 'Protein-based', 'Total Intensity']


def _display_method_selection(normalization_options: list) -> str:
    """Display normalization method radio and handle session state persistence."""
    widget_key = 'norm_method_selection'
    persisted_value = st.session_state.get('_preserved_norm_method_selection', 'None (pre-normalized data)')
    if persisted_value in normalization_options:
        st.session_state[widget_key] = persisted_value
    else:
        st.session_state[widget_key] = 'None (pre-normalized data)'

    def on_norm_method_change():
        st.session_state._preserved_norm_method_selection = st.session_state[widget_key]

    st.markdown("##### ⚙️ Normalization Method")
    method = st.radio(
        "Method:",
        options=normalization_options,
        horizontal=True,
        key=widget_key,
        on_change=on_norm_method_change
    )
    st.session_state._preserved_norm_method_selection = method
    return method


def _collect_method_config(method: str, intsta_df, selected_classes: list, experiment) -> tuple:
    """Collect IS and protein config based on selected method.

    Returns (internal_standards, intsta_concentrations, protein_concentrations).
    Returns None for internal_standards or protein_concentrations if config is incomplete.
    """
    internal_standards = {}
    intsta_concentrations = {}
    protein_concentrations = {}

    if method in ['Internal Standards', 'Internal Standards + Protein']:
        internal_standards, intsta_concentrations = _display_internal_standards_config(intsta_df, selected_classes)
        if internal_standards is None:
            return None, None, None

    if method in ['Protein-based', 'Internal Standards + Protein']:
        protein_concentrations = _display_protein_config(experiment)
        if protein_concentrations is None:
            return None, None, None
        # Validate: zero concentrations cannot be used as denominators
        zero_samples = [s for s, c in protein_concentrations.items() if c <= 0]
        if zero_samples:
            st.error(
                f"Protein concentration must be greater than zero for all samples. "
                f"The following samples have zero or negative values: {', '.join(zero_samples)}"
            )
            return None, None, None

    return internal_standards, intsta_concentrations, protein_concentrations


def _display_normalization_results(result) -> None:
    """Display normalization results or errors."""
    if result and result.success:
        st.markdown("##### 📊 Final Normalized Data")
        if result.normalized_df is not None:
            st.dataframe(result.normalized_df, use_container_width=True)
            csv_download_button(result.normalized_df, "normalized_data.csv", key="download_normalized_data")
    elif result:
        for error in result.validation_errors:
            st.error(error)
