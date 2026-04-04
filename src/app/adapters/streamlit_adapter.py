"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
This module contains all Streamlit-specific code that wraps the pure business logic services.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.constants import COV_THRESHOLD_DEFAULT

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..models.statistics import StatisticalTestConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from ..services.plotting.box_plot import BoxPlotService
from ..services.plotting.bqc_plotter import BQCPlotterService
from ..services.plotting.correlation import CorrelationPlotterService
from ..services.plotting.pca import PCAPlotterService
from ..services.plotting.retention_time import RetentionTimePlotterService
from ..services.validation import get_matching_concentration_columns
from ..workflows.analysis import AnalysisWorkflow
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
                             pre_filter_df, grade_config, last_quality_config,
                             msdial_quality_config, msdial_use_normalized,
                             msdial_data_type_index
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
                             qc_correlation_plots, qc_pca_plot, qc_samples_removed,
                             _preserved_bqc_filter_choice, _preserved_rt_viewing_mode,
                             _preserved_pca_samples_remove
      analysis.py          → analysis_selection, analysis_bar_chart_fig,
                             analysis_pie_chart_figs, analysis_saturation_figs,
                             analysis_fach_fig, analysis_pathway_fig,
                             analysis_volcano_fig, analysis_volcano_data,
                             analysis_heatmap_fig, analysis_heatmap_clusters,
                             analysis_all_plots
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
    last_quality_config: Any = None
    msdial_quality_config: Optional[QualityFilterConfig] = None
    msdial_use_normalized: bool = False
    msdial_data_type_index: int = 0  # 0 = raw, 1 = pre-normalized

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
    qc_box_plot_fig1: Any = None
    qc_box_plot_fig2: Any = None
    qc_bqc_plot: Any = None
    qc_cov_threshold: int = COV_THRESHOLD_DEFAULT
    qc_retention_time_plot: Any = None
    qc_correlation_plots: Dict[str, Any] = field(default_factory=dict)
    qc_pca_plot: Any = None
    qc_samples_removed: List[str] = field(default_factory=list)
    _preserved_bqc_filter_choice: str = 'No'
    _preserved_rt_viewing_mode: str = 'Comparison Mode'
    _preserved_pca_samples_remove: List[str] = field(default_factory=list)

    # --- Analysis (owner: analysis.py) ---
    analysis_selection: Optional[str] = None
    analysis_bar_chart_fig: Any = None
    analysis_pie_chart_figs: Dict[str, Any] = field(default_factory=dict)
    analysis_saturation_figs: Dict[str, Any] = field(default_factory=dict)
    analysis_fach_fig: Any = None
    analysis_pathway_fig: Any = None
    analysis_pathway_active_classes: Optional[List[str]] = None
    analysis_pathway_added_edges: List = field(default_factory=list)
    analysis_pathway_removed_edges: List = field(default_factory=list)
    analysis_pathway_custom_nodes: Dict[str, Any] = field(default_factory=dict)
    analysis_pathway_position_overrides: Dict[str, Any] = field(default_factory=dict)
    analysis_volcano_fig: Any = None
    analysis_volcano_data: Any = None
    analysis_heatmap_fig: Any = None
    analysis_heatmap_clusters: Optional[pd.DataFrame] = None
    analysis_all_plots: Dict[str, Any] = field(default_factory=dict)


# Keys that should NOT be reset when starting a fresh analysis
# (they control app-level routing, not data-specific state)
_PRESERVE_ON_RESET = {'page', 'module'}

# Widget keys created by Streamlit widgets (not in SessionState dataclass).
# Must be removed on reset to prevent stale widget state.
_WIDGET_KEYS = {
    # Sidebar widgets (file_upload, experiment_config, sample_grouping, confirm_inputs)
    'manual_sample_override', 'grouping_radio',
    'bqc_radio', 'bqc_label_radio', 'confirm_checkbox',
    'sample_data_experiment',
    # Data processing widgets (data_processing.py)
    'grade_filter_mode', 'grade_selections',
    'grade_filter_mode_radio',
    'msdial_quality_level', 'msdial_data_type_radio',
    'msdial_quality_level_radio',
    'msdial_custom_msms', 'msdial_show_custom_threshold', 'msdial_custom_score',
    # Zero filtering widgets (zero_filtering.py)
    'zero_filter_detection_threshold',
    'non_bqc_zero_threshold', 'bqc_zero_threshold',
    # Standards widgets (internal_standards.py)
    'standards_source_radio', 'standards_location_radio', 'standards_conditions_select',
    'standards_file_uploader',
    # Normalization widgets (normalization.py)
    'temp_selected_classes', 'protein_csv_upload',
    'norm_method_selection', 'protein_input_method',
    # Data processing widgets (continued)
    'msdial_msms_only',
    # Download/action button keys (various files)
    'load_sample', 'clear_custom_standards',
    'download_removed_species', 'download_auto_standards',
    'download_custom_standards', 'download_filtered_data',
    'download_normalized_data',
    # Quality Check widgets (quality_check.py)
    'bqc_cov_threshold', 'bqc_filter_choice', 'bqc_lipids_to_keep',
    'bqc_csv_download', 'bqc_filtered_download',
    'rt_viewing_mode', 'rt_class_selection',
    'qc_rt_svg_comparison', 'rt_csv_comparison',
    'corr_condition', 'corr_csv_download', 'qc_corr_svg',
    'pca_samples_remove', 'pca_csv_download',
    'qc_missing_values_svg', 'qc_missing_values_csv',
    'qc_box_plot_svg', 'qc_box_plot_csv',
    'qc_bqc_svg', 'qc_pca_svg',
    # Analysis widgets (analysis.py)
    'analysis_radio',
    'bar_conditions', 'bar_classes', 'bar_scale_radio',
    'bar_stats_mode', 'bar_detailed_stats',
    'pie_classes',
    'sat_conditions', 'sat_classes', 'sat_plot_type',
    'sat_show_significance', 'sat_detailed_stats', 'sat_stats_mode',
    'fach_class', 'fach_conditions',
    'pathway_control', 'pathway_experimental',
    'pathway_class_selector', 'pathway_add_node_name',
    'pathway_add_node_x', 'pathway_add_node_y',
    'pathway_add_edge_source', 'pathway_add_edge_target',
    'volcano_control', 'volcano_experimental', 'volcano_classes',
    'volcano_p_threshold', 'volcano_fc_threshold',
    'volcano_hide_nonsig', 'volcano_top_n',
    'volcano_additional_labels', 'volcano_stats_mode', 'volcano_detailed_stats',
    'heatmap_conditions', 'heatmap_classes', 'heatmap_type',
    'heatmap_n_clusters', 'heatmap_cluster_view',
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
    'volcano_label_x_',       # per-lipid label X position (analysis.py)
    'volcano_label_y_',       # per-lipid label Y position (analysis.py)
    'analysis_svg_',          # analysis SVG download buttons (analysis.py)
    'analysis_csv_',          # analysis CSV download buttons (analysis.py)
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

    Session State Access Patterns
    =============================
    UI files should use exactly one of these three patterns per key:

    1. **Direct access** — for simple reads/writes of data keys owned by the
       current file. Use ``st.session_state['key']`` for reads that must exist
       and ``st.session_state['key'] = value`` for writes.  Use
       ``st.session_state.get('key')`` when the key may be ``None``.

    2. **``restore_widget_value`` / ``save_widget_value``** — for widget keys
       that must survive module navigation (Streamlit removes widget keys for
       non-rendered widgets).  Call ``restore_widget_value`` *before* the widget
       renders and ``save_widget_value`` in an ``on_change`` callback or
       immediately after the widget.  Pair each widget key with a
       ``_preserved_*`` session state key declared in ``SessionState``.

    3. **Widget ``key=`` parameter** — Streamlit widgets automatically read/write
       ``st.session_state[key]``.  Do *not* also write to the same key manually;
       let the widget own it.  Guard ``st.session_state[key] = default`` with
       ``if key not in st.session_state`` to avoid overwriting user selections.

    When to use which:
    - Data flowing between UI files → pattern 1 (direct access)
    - Widget state surviving module switches → pattern 2 (preserve/restore)
    - Widget with ``on_change`` callback → pattern 2 + pattern 3
    - Widget default that only needs to survive within the same page → pattern 3
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

    # ==================== Widget Preservation ====================

    @staticmethod
    def restore_widget_value(widget_key: str, preserved_key: str, default: Any) -> Any:
        """Restore a widget's value from its preserved session state key.

        When navigating between modules, Streamlit removes widget keys for
        non-rendered widgets. This method restores the value from a preserved
        key so the widget renders with the user's last selection.

        Args:
            widget_key: The Streamlit widget key (e.g., 'bqc_filter_choice').
            preserved_key: The session state key storing the preserved value
                          (e.g., '_preserved_bqc_filter_choice').
            default: Default value if neither key exists.

        Returns:
            The value to use for the widget's initial value/index.
        """
        if widget_key in st.session_state:
            return st.session_state[widget_key]
        preserved = st.session_state.get(preserved_key)
        if preserved is not None:
            return preserved
        return default

    @staticmethod
    def save_widget_value(widget_key: str, preserved_key: str) -> None:
        """Save a widget's current value to its preserved session state key.

        Call this after the widget renders (or in an on_change callback)
        to persist the value across module navigation.

        Args:
            widget_key: The Streamlit widget key to read from.
            preserved_key: The session state key to write to.
        """
        value = st.session_state.get(widget_key)
        if value is not None:
            st.session_state[preserved_key] = value

    # ==================== Module State Reset ====================

    @staticmethod
    def reset_module_state(*prefixes: str) -> None:
        """Reset all SessionState fields matching any of the given prefixes.

        Iterates over the SessionState dataclass fields and resets each field
        whose name starts with one of the provided prefixes to its default value.

        Args:
            *prefixes: One or more string prefixes to match against field names.
        """
        import dataclasses
        for f in fields(SessionState):
            if any(f.name.startswith(p) for p in prefixes):
                if f.default is not dataclasses.MISSING:
                    st.session_state[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    st.session_state[f.name] = f.default_factory()

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

    # ==================== Cached QC Wrappers ====================

    @staticmethod
    @st.cache_data(
        show_spinner="Computing box plots...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_box_plots(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> Tuple:
        """Cached box plot computation.

        Computes mean area DataFrame, missing values percentages,
        and both box plot figures in a single cached call.

        Returns:
            Tuple of (missing_values_fig, box_plot_fig, mean_area_df,
                      missing_pct_list, current_samples)
        """
        samples = get_matching_concentration_columns(df, experiment)
        mean_area_df = BoxPlotService.create_mean_area_df(df, samples)
        missing_pct = BoxPlotService.calculate_missing_values_percentage(
            mean_area_df
        )
        fig1 = BoxPlotService.plot_missing_values(
            samples, missing_pct,
            experiment.conditions_list, experiment.individual_samples_list,
        )
        fig2 = BoxPlotService.plot_box_plot(
            mean_area_df, samples,
            experiment.conditions_list, experiment.individual_samples_list,
        )
        return fig1, fig2, mean_area_df, missing_pct, samples

    @staticmethod
    @st.cache_data(
        show_spinner="Computing BQC quality assessment...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_bqc_scatter(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        bqc_sample_index: int,
        cov_threshold: int,
    ) -> Tuple:
        """Cached BQC scatter plot computation.

        Returns:
            Tuple of (scatter_plot, prepared_df, reliable_data_percent, high_cov_info)
        """
        return BQCPlotterService.generate_and_display_cov_plot(
            df, experiment, bqc_sample_index, cov_threshold=cov_threshold,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Computing correlation matrix...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_correlation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        condition_index: int,
        sample_type: str,
    ) -> Tuple:
        """Cached pairwise correlation computation.

        Returns:
            Tuple of (matplotlib_fig, correlation_df)
        """
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            df, experiment.individual_samples_list, condition_index,
        )
        correlation_df, v_min, thresh = (
            CorrelationPlotterService.compute_correlation(
                mean_area_df, sample_type,
            )
        )
        condition_name = experiment.conditions_list[condition_index]
        fig = CorrelationPlotterService.render_correlation_plot(
            correlation_df, v_min, thresh, condition_name,
        )
        return fig, correlation_df

    @staticmethod
    @st.cache_data(
        show_spinner="Computing PCA...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_pca(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> Tuple:
        """Cached PCA computation.

        Returns:
            Tuple of (pca_plot, pca_df)
        """
        return PCAPlotterService.plot_pca(
            df, experiment.full_samples_list,
            experiment.extensive_conditions_list,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Computing retention time plots...",
    )
    def run_retention_time_single(df: pd.DataFrame) -> list:
        """Cached single-class retention time plot computation.

        Returns:
            List of (plot, retention_df) tuples, one per RT group.
        """
        return RetentionTimePlotterService.plot_single_retention(df)

    @staticmethod
    @st.cache_data(
        show_spinner="Computing retention time plots...",
    )
    def run_retention_time_multi(
        df: pd.DataFrame,
        selected_classes: List[str],
    ) -> Tuple:
        """Cached multi-class retention time comparison plot.

        Returns:
            Tuple of (plot, retention_df) or (None, None).
        """
        return RetentionTimePlotterService.plot_multi_retention(
            df, selected_classes,
        )

    # ==================== Cached Analysis Wrappers ====================

    @staticmethod
    @st.cache_data(
        show_spinner="Generating bar chart...",
        hash_funcs={ExperimentConfig: hash, StatisticalTestConfig: hash},
    )
    def run_bar_chart(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        stat_config: Optional[StatisticalTestConfig] = None,
        scale: str = 'linear',
    ):
        """Cached bar chart analysis."""
        return AnalysisWorkflow.run_bar_chart(
            df, experiment, selected_conditions, selected_classes,
            stat_config=stat_config, scale=scale,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating pie charts...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_pie_charts(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ):
        """Cached pie chart analysis."""
        return AnalysisWorkflow.run_pie_charts(
            df, experiment, selected_conditions, selected_classes,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating saturation plots...",
        hash_funcs={ExperimentConfig: hash, StatisticalTestConfig: hash},
    )
    def run_saturation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        stat_config: Optional[StatisticalTestConfig] = None,
        plot_type: str = 'concentration',
        show_significance: bool = False,
    ):
        """Cached saturation analysis."""
        return AnalysisWorkflow.run_saturation(
            df, experiment, selected_conditions, selected_classes,
            stat_config=stat_config, plot_type=plot_type,
            show_significance=show_significance,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating FACH heatmap...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_fach(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_class: str,
        selected_conditions: List[str],
    ):
        """Cached FACH heatmap analysis."""
        return AnalysisWorkflow.run_fach(
            df, experiment, selected_class, selected_conditions,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Computing pathway data...",
        hash_funcs={ExperimentConfig: hash},
    )
    def compute_pathway_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
        saturation_source_df: pd.DataFrame = None,
    ):
        """Cached pathway data computation (fold change + saturation).

        Args:
            df: Full DataFrame for fold-change calculation.
            experiment: Experiment configuration.
            control: Control condition name.
            experimental: Experimental condition name.
            saturation_source_df: Optional filtered DataFrame for
                saturation ratio (excludes consolidated lipids).
                If None, ``df`` is used for both.
        """
        return AnalysisWorkflow.compute_pathway_data(
            df, experiment, control, experimental,
            saturation_source_df=saturation_source_df,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating pathway visualization...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_pathway(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
    ):
        """Cached pathway visualization (default layout)."""
        return AnalysisWorkflow.run_pathway(
            df, experiment, control, experimental,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Performing statistical analysis...",
        hash_funcs={ExperimentConfig: hash, StatisticalTestConfig: hash},
    )
    def run_volcano(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
        selected_classes: List[str],
        stat_config: StatisticalTestConfig,
        p_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        hide_non_sig: bool = False,
        top_n_labels: int = 0,
        custom_label_positions: Optional[Dict] = None,
        additional_labels: Optional[tuple] = None,
    ):
        """Cached volcano plot analysis."""
        return AnalysisWorkflow.run_volcano(
            df, experiment, control, experimental, selected_classes,
            stat_config,
            p_threshold=p_threshold, fc_threshold=fc_threshold,
            hide_non_sig=hide_non_sig, top_n_labels=top_n_labels,
            custom_label_positions=custom_label_positions,
            additional_labels=list(additional_labels) if additional_labels else None,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating heatmap...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_heatmap(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        heatmap_type: str = 'regular',
        n_clusters: int = 3,
    ):
        """Cached lipidomic heatmap analysis."""
        return AnalysisWorkflow.run_heatmap(
            df, experiment, selected_conditions, selected_classes,
            heatmap_type=heatmap_type, n_clusters=n_clusters,
        )
