"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
This module contains all Streamlit-specific code that wraps the pure business logic services.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from ..workflows.data_ingestion import DataIngestionWorkflow, IngestionConfig, IngestionResult
from ..workflows.normalization import NormalizationWorkflow, NormalizationWorkflowConfig, NormalizationWorkflowResult


@dataclass
class SessionState:
    """
    Type-safe container for all session state variables.

    Defines the complete shape of session state with default values.
    Use this as a reference for what keys exist in session state.

    Ownership model — each key is owned by one UI file:
      file_upload.py       → raw_df, using_sample_data, sample_data_file
      column_mapping.py    → standardized_df, column_mapping, n_intensity_cols,
                             format_type, msdial_features, msdial_sample_names,
                             _msdial_override_samples
      experiment_config.py → workbench_conditions, workbench_samples
      sample_grouping.py   → experiment, bqc_label, confirmed, grouping_complete,
                             original_column_order
      data_processing.py   → cleaned_df, intsta_df, continuation_df, ingestion_result,
                             pre_filter_df, grade_config, grade_filter_mode_saved,
                             grade_selections_saved, last_quality_config,
                             msdial_quality_config, msdial_use_normalized,
                             msdial_data_type_index, msdial_quality_level_index
      internal_standards.py → original_auto_intsta_df, preserved_intsta_df,
                              preserved_standards_mode, custom_standards_df,
                              custom_standards_mode, standards_source
      zero_filtering.py    → _zero_filter_format, _preserved_zero_filter_*
      normalization.py     → normalized_df, normalization_method, normalization_inputs,
                             selected_classes, create_norm_dataset, norm_method_selection,
                             _preserved_norm_method_selection, normalization_result,
                             class_standard_map, standard_concentrations, protein_df,
                             protein_input_method, protein_input_method_prev
      quality_check.py     → qc_continuation_df, qc_bqc_plot, qc_cov_threshold,
                             qc_correlation_plots, qc_pca_plot, qc_samples_removed
      main_app.py          → page, module
    """
    # --- App-level routing (owner: main_app.py) ---
    page: str = 'landing'
    module: str = "Data Cleaning, Filtering, & Normalization"

    # --- File upload (owner: file_upload.py) ---
    raw_df: Optional[pd.DataFrame] = None
    using_sample_data: bool = False
    sample_data_file: Optional[str] = None

    # --- Column mapping (owner: column_mapping.py) ---
    standardized_df: Optional[pd.DataFrame] = None
    column_mapping: Optional[pd.DataFrame] = None
    n_intensity_cols: Optional[int] = None
    format_type: Optional[DataFormat] = None
    msdial_features: Dict[str, Any] = field(default_factory=dict)
    msdial_sample_names: Optional[List[str]] = None
    _msdial_override_samples: Optional[List[str]] = None

    # --- Experiment config (owner: experiment_config.py) ---
    workbench_conditions: Optional[List[str]] = None
    workbench_samples: Optional[Dict] = None

    # --- Sample grouping (owner: sample_grouping.py) ---
    experiment: Optional[ExperimentConfig] = None
    bqc_label: Optional[str] = None
    confirmed: bool = False
    grouping_complete: bool = True
    original_column_order: Optional[List[str]] = None

    # --- Data processing (owner: data_processing.py) ---
    cleaned_df: Optional[pd.DataFrame] = None
    intsta_df: Optional[pd.DataFrame] = None
    continuation_df: Optional[pd.DataFrame] = None
    pre_filter_df: Optional[pd.DataFrame] = None
    ingestion_result: Any = None
    grade_config: Optional[GradeFilterConfig] = None
    grade_filter_mode_saved: int = 0  # 0 = default, 1 = customize
    grade_selections_saved: Dict[str, List[str]] = field(default_factory=dict)
    last_quality_config: Any = None
    msdial_quality_config: Optional[QualityFilterConfig] = None
    msdial_use_normalized: bool = False
    msdial_data_type_index: int = 0  # 0 = raw, 1 = pre-normalized
    msdial_quality_level_index: int = 1  # 0 = relaxed, 1 = moderate, 2 = strict

    # --- Internal standards (owner: internal_standards.py) ---
    original_auto_intsta_df: Optional[pd.DataFrame] = None
    preserved_intsta_df: Optional[pd.DataFrame] = None
    preserved_standards_mode: Optional[str] = None
    custom_standards_df: Optional[pd.DataFrame] = None
    custom_standards_mode: Optional[str] = None
    standards_source: Optional[str] = None

    # --- Zero filtering (owner: zero_filtering.py) ---
    _zero_filter_format: Optional[str] = None
    _preserved_zero_filter_detection_threshold: Optional[float] = None
    _preserved_non_bqc_zero_threshold: Optional[float] = None
    _preserved_bqc_zero_threshold: Optional[float] = None

    # --- Normalization (owner: normalization.py) ---
    normalized_df: Optional[pd.DataFrame] = None
    normalization_method: str = 'None'
    normalization_inputs: Dict[str, Any] = field(default_factory=dict)
    selected_classes: List[str] = field(default_factory=list)
    create_norm_dataset: bool = False
    norm_method_selection: Optional[str] = None
    _preserved_norm_method_selection: Optional[str] = None
    normalization_result: Any = None
    class_standard_map: Optional[Dict[str, str]] = None
    standard_concentrations: Optional[Dict[str, float]] = None
    protein_df: Optional[pd.DataFrame] = None
    protein_input_method: Optional[str] = None
    protein_input_method_prev: Optional[str] = None

    # --- Quality Check (owner: quality_check.py) ---
    qc_continuation_df: Optional[pd.DataFrame] = None
    qc_bqc_plot: Any = None
    qc_cov_threshold: int = 30
    qc_correlation_plots: Dict[str, Any] = field(default_factory=dict)
    qc_pca_plot: Any = None
    qc_samples_removed: List[str] = field(default_factory=list)


# Keys that should NOT be reset when starting a fresh analysis
# (they control app-level routing, not data-specific state)
_PRESERVE_ON_RESET = {'page', 'module'}

# Widget keys created by Streamlit widgets (not in SessionState dataclass).
# Must be removed on reset to prevent stale widget state.
_WIDGET_KEYS = {
    # Sidebar widgets (file_upload, experiment_config, sample_grouping, confirm_inputs)
    'manual_sample_override', 'grouping_radio',
    'bqc_radio', 'bqc_label_radio', 'confirm_checkbox',
    # Data processing widgets (data_processing.py)
    'grade_filter_mode', 'grade_selections',
    'grade_filter_mode_radio',
    'msdial_quality_level', 'msdial_data_type_radio',
    'msdial_quality_level_radio',
    'msdial_custom_msms', 'msdial_show_custom_threshold', 'msdial_custom_score',
    # Zero filtering widgets (zero_filtering.py)
    'zero_filter_detection_threshold',
    # Standards widgets (internal_standards.py)
    'standards_source_radio', 'standards_location_radio', 'standards_conditions_select',
    # Normalization widgets (normalization.py)
    'temp_selected_classes', 'protein_csv_upload',
    # Quality Check widgets (quality_check.py)
    'bqc_cov_threshold', 'bqc_filter_choice', 'bqc_lipids_to_keep',
    'bqc_csv_download', 'bqc_filtered_download',
    'rt_viewing_mode', 'rt_class_selection',
    'qc_rt_svg_comparison', 'rt_csv_comparison',
    'corr_condition', 'corr_csv_download', 'qc_corr_svg',
    'pca_samples_remove', 'pca_csv_download',
}

# Prefixes for dynamic widget keys (created with f-strings like `protein_{sample}`).
# All matching keys are removed on reset via pattern-based cleanup.
_DYNAMIC_KEY_PREFIXES = (
    'protein_',         # protein concentrations per sample (normalization.py)
    'conc_',            # internal standard concentrations (normalization.py)
    'standard_selection_',  # lipid class → standard mapping (normalization.py)
    'grade_select_',    # grade selection per class (data_processing.py)
    'cond_name_',       # condition names by index (experiment_config.py)
    'n_samples_',       # samples per condition by index (experiment_config.py)
    'select_',          # multiselect for condition samples (sample_grouping.py)
    'qc_rt_svg_individual_',  # individual RT SVG downloads (quality_check.py)
    'rt_csv_individual_',     # individual RT CSV downloads (quality_check.py)
)


class StreamlitAdapter:
    """
    Adapter for managing Streamlit session state and caching.

    Responsibilities:
    1. Initialize session state with default values
    2. Provide type-safe accessors for session state
    3. Provide cached wrappers for service calls
    4. Handle UI data conversion to service models

    All methods are static - no instance state is stored.
    """

    # ==================== Session State Initialization ====================

    @staticmethod
    def initialize_session_state() -> None:
        """
        Initialize all session state variables with default values.

        Call this at the beginning of main() to ensure all required
        session state keys exist before they are accessed.
        Derives defaults from the SessionState dataclass.
        """
        defaults = SessionState()
        for f in fields(SessionState):
            if f.name not in st.session_state:
                st.session_state[f.name] = getattr(defaults, f.name)

    @staticmethod
    def reset_data_state() -> None:
        """
        Reset all data-related session state for a fresh upload.

        Resets every SessionState field to its default except page/module
        (app-level routing). Also removes Streamlit widget keys (both static
        and dynamic) that may hold stale state from the previous dataset.
        """
        defaults = SessionState()
        for f in fields(SessionState):
            if f.name not in _PRESERVE_ON_RESET:
                st.session_state[f.name] = getattr(defaults, f.name)
        # Remove static widget keys
        for key in _WIDGET_KEYS:
            st.session_state.pop(key, None)
        # Remove dynamic widget keys by prefix (e.g. protein_{sample}, conc_{std})
        dynamic_keys = [
            key for key in list(st.session_state.keys())
            if isinstance(key, str) and key.startswith(_DYNAMIC_KEY_PREFIXES)
        ]
        for key in dynamic_keys:
            st.session_state.pop(key, None)

    @staticmethod
    def reset_normalization_state() -> None:
        """Reset normalization-related session state."""
        st.session_state.normalization_method = 'None'
        st.session_state.normalization_inputs = {}
        st.session_state.selected_classes = []
        st.session_state.create_norm_dataset = False
        st.session_state.normalized_df = None
        st.session_state.norm_method_selection = None
        st.session_state.normalization_result = None
        st.session_state.class_standard_map = None
        st.session_state.standard_concentrations = None
        st.session_state.protein_df = None
        st.session_state.protein_input_method = None
        st.session_state.protein_input_method_prev = None

    # ==================== Service Wrappers ====================

    @staticmethod
    def detect_format(df: pd.DataFrame) -> DataFormat:
        """
        Detect data format from DataFrame.

        This is a thin wrapper around FormatDetectionService.
        Not cached because format detection is fast (just column checks).

        Args:
            df: DataFrame to detect format for

        Returns:
            Detected DataFormat
        """
        return FormatDetectionService.detect_format(df)

    # ==================== Cached Workflow Wrappers ====================

    @staticmethod
    @st.cache_data(
        show_spinner="Processing data...",
        hash_funcs={ExperimentConfig: hash, GradeFilterConfig: hash, QualityFilterConfig: hash},
    )
    def run_ingestion(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        data_format: Optional[DataFormat] = None,
        bqc_label: Optional[str] = None,
        apply_zero_filter: bool = False,
        grade_config: Optional[GradeFilterConfig] = None,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> Tuple[
        Optional[pd.DataFrame],  # cleaned_df
        Optional[pd.DataFrame],  # internal_standards_df
        str,                     # detected_format
        bool,                    # is_valid
        List[str],               # validation_errors
        List[str],               # validation_warnings
        List[str],               # cleaning_messages
    ]:
        """
        Cached data ingestion workflow wrapper.

        Args:
            df: Raw DataFrame to process
            experiment: Experiment configuration
            data_format: Data format enum
            bqc_label: BQC condition label
            apply_zero_filter: Whether to apply zero filtering
            grade_config: Grade filter config (LipidSearch only)
            quality_config: Quality filter config (MS-DIAL only)

        Returns:
            Tuple of (cleaned_df, intsta_df, format, is_valid, errors, warnings, messages)
        """
        config = IngestionConfig(
            experiment=experiment,
            data_format=data_format,
            bqc_label=bqc_label,
            apply_zero_filter=apply_zero_filter,
            grade_config=grade_config,
            quality_config=quality_config,
        )

        result = DataIngestionWorkflow.run(df, config)

        return (
            result.cleaned_df,
            result.internal_standards_df,
            result.detected_format.value,
            result.is_valid,
            result.validation_errors,
            result.validation_warnings,
            result.cleaning_messages,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Normalizing data...",
        hash_funcs={ExperimentConfig: hash, NormalizationConfig: hash},
    )
    def run_normalization(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        normalization: NormalizationConfig,
        data_format: DataFormat = DataFormat.GENERIC,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> Tuple[
        Optional[pd.DataFrame],  # normalized_df
        bool,                    # success
        str,                     # method_applied
        List[str],               # removed_standards
        List[str],               # validation_errors
        List[str],               # validation_warnings
    ]:
        """
        Cached normalization workflow wrapper.

        Args:
            df: Cleaned DataFrame to normalize
            experiment: Experiment configuration
            normalization: Normalization configuration
            data_format: Data format enum
            intsta_df: Internal standards DataFrame

        Returns:
            Tuple of (normalized_df, success, method_applied, removed_standards, errors, warnings)
        """
        workflow_config = NormalizationWorkflowConfig(
            experiment=experiment,
            normalization=normalization,
            data_format=data_format
        )

        result = NormalizationWorkflow.run(
            df=df,
            config=workflow_config,
            intsta_df=intsta_df
        )

        return (
            result.normalized_df,
            result.success,
            result.method_applied,
            result.removed_standards,
            result.validation_errors,
            result.validation_warnings,
        )
