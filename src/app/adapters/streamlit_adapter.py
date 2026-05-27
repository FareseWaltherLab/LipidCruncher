"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
This module contains all Streamlit-specific code that wraps the pure business logic services.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.constants import COV_THRESHOLD_DEFAULT, PAGE_LANDING, MODULE_DATA_PROCESSING

from ..models.experiment import ExperimentConfig
from ..services.data_standardization import MSDIALFeatures
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
from ..workflows.analysis import (
    AnalysisWorkflow,
    BarChartResult,
    ChainLengthResult,
    FACHResult,
    HeatmapResult,
    PathwayDataResult,
    PathwayResult,
    SaturationResult,
    VolcanoData,
    VolcanoResult,
)
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
                             analysis_chain_length_fig, analysis_fach_fig, analysis_pathway_fig,
                             analysis_volcano_fig, analysis_volcano_data,
                             analysis_heatmap_fig, analysis_heatmap_clusters,
                             analysis_all_plots
      main_app.py          → page, module
    """
    # --- App-level routing (owner: main_app.py) ---
    page: str = PAGE_LANDING
    module: str = MODULE_DATA_PROCESSING

    # --- File upload (owner: file_upload.py) ---
    raw_df: Optional[pd.DataFrame] = None
    using_sample_data: bool = False
    sample_data_file: Optional[str] = None

    # --- Column mapping (owner: column_mapping.py) ---
    standardized_df: Optional[pd.DataFrame] = None
    column_mapping: Optional[pd.DataFrame] = None
    n_intensity_cols: Optional[int] = None
    format_type: Optional[DataFormat] = None
    msdial_features: Optional[MSDIALFeatures] = None
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
    # Survives transient widget garbage collection when the sidebar
    # short-circuits before the BQC section renders (e.g. mid-regroup).
    # Without these, switching grouping to "No" wipes the user's prior
    # BQC answers because Streamlit drops the widget keys whose widgets
    # didn't render this run.
    _preserved_bqc_radio: Optional[str] = None
    _preserved_bqc_label_radio: Optional[str] = None
    # Snapshot of grouping_radio from the previous rerun. Used to
    # detect Yes ↔ No toggles and invalidate the Confirm Inputs
    # checkbox without an on_change callback (which would otherwise
    # require two clicks to commit the toggle in Streamlit 1.50).
    _last_seen_grouping_radio: Optional[str] = None
    # Snapshot of the per-condition sample multiselects (select_*) from
    # the previous rerun. Same purpose as _last_seen_grouping_radio but
    # for the multiselects — toggling samples between conditions also
    # invalidates a prior Confirm Inputs check.
    _last_seen_selects: Optional[Dict[str, Tuple[str, ...]]] = None

    # --- Data processing (owner: data_processing.py) ---
    cleaned_df: Optional[pd.DataFrame] = None
    intsta_df: Optional[pd.DataFrame] = None
    continuation_df: Optional[pd.DataFrame] = None
    pre_filter_df: Optional[pd.DataFrame] = None
    ingestion_result: Optional[IngestionResult] = None
    grade_config: Optional[GradeFilterConfig] = None
    last_quality_config: Optional[QualityFilterConfig] = None
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
    normalization_result: Optional[NormalizationWorkflowResult] = None
    class_standard_map: Optional[Dict[str, str]] = None
    standard_concentrations: Optional[Dict[str, float]] = None
    protein_df: Optional[pd.DataFrame] = None
    protein_input_method: Optional[str] = None
    protein_input_method_prev: Optional[str] = None

    # --- Quality Check (owner: quality_check.py) ---
    qc_continuation_df: Optional[pd.DataFrame] = None
    qc_box_plot_fig1: Optional[go.Figure] = None
    qc_box_plot_fig2: Optional[go.Figure] = None
    qc_bqc_plot: Optional[go.Figure] = None
    qc_cov_threshold: int = COV_THRESHOLD_DEFAULT
    qc_retention_time_plot: Optional[go.Figure] = None
    qc_correlation_plots: Dict[str, matplotlib.figure.Figure] = field(default_factory=dict)
    qc_pca_plot: Optional[go.Figure] = None
    qc_samples_removed: List[str] = field(default_factory=list)
    _preserved_bqc_filter_choice: str = 'No'
    _preserved_rt_viewing_mode: str = 'Comparison Mode'
    _preserved_pca_samples_remove: List[str] = field(default_factory=list)

    # --- Analysis (owner: analysis.py) ---
    analysis_selection: Optional[str] = None
    analysis_bar_chart_fig: Optional[go.Figure] = None
    analysis_pie_chart_figs: Dict[str, go.Figure] = field(default_factory=dict)
    analysis_saturation_figs: Dict[str, go.Figure] = field(default_factory=dict)
    analysis_chain_length_fig: Optional[go.Figure] = None
    analysis_fach_fig: Optional[go.Figure] = None
    analysis_pathway_fig: Optional[go.Figure] = None
    analysis_pathway_active_classes: Optional[List[str]] = None
    analysis_pathway_added_edges: List[Tuple[str, str]] = field(default_factory=list)
    analysis_pathway_removed_edges: List[Tuple[str, str]] = field(default_factory=list)
    analysis_pathway_custom_nodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    analysis_pathway_position_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    analysis_volcano_fig: Optional[go.Figure] = None
    analysis_volcano_data: Optional[VolcanoData] = None
    analysis_heatmap_fig: Optional[go.Figure] = None
    analysis_heatmap_clusters: Optional[pd.DataFrame] = None
    analysis_all_plots: Dict[str, Union[go.Figure, matplotlib.figure.Figure]] = field(default_factory=dict)

    # --- LSI Compliance Report (owner: lsi_report.py) ---
    lsi_manual_fields: Dict[str, str] = field(default_factory=dict)
    lsi_pdf_bytes: Optional[bytes] = None

    # --- AI Chat (owner: ai_chat/chat_panel.py) ---
    chat_transform_session: Optional[Any] = None
    chat_viz_session: Optional[Any] = None
    chat_data_questions_session: Optional[Any] = None
    chat_unified_session: Optional[Any] = None
    chat_result_tables: Dict[str, Any] = field(default_factory=dict)
    chat_mode: Optional[str] = None
    chat_transform_result: Optional[pd.DataFrame] = None
    chat_viz_figures: Dict[str, Any] = field(default_factory=dict)
    chat_message_count: int = 0
    chat_execution_count: int = 0

    # --- AI Studio (owner: ai_studio_page.py) ---
    # `chat_studio_df_original` holds the dataframe with the user's
    # original sample column names (what they uploaded / imported). It is
    # what the data preview, Data Questions, and Transform tabs read.
    # `chat_studio_df` holds a reordered/renamed copy with sample columns
    # normalized to `intensity[s1..sN]` per condition — required by the
    # Custom Visualization tab's plotters because ExperimentConfig
    # generates its sample labels as s1..sN.
    chat_studio_df: Optional[pd.DataFrame] = None
    chat_studio_df_original: Optional[pd.DataFrame] = None
    chat_studio_experiment: Optional[Any] = None
    # Tracks where the AI Studio data came from. None for direct upload,
    # 'normalized' or 'raw' for the two pipeline-import options. Used to
    # hard-lock Custom Visualization when the source is 'raw' regardless
    # of whether the columns happen to satisfy the validation check.
    chat_studio_pipeline_source: Optional[str] = None
    # Detected format string (DataFormat.value) for the uploaded file.
    # Used to block the AI Studio flow when a known standard format
    # (LipidSearch, MS-DIAL) is uploaded directly — those should be
    # processed through the main pipeline first.
    chat_studio_detected_format: Optional[str] = None
    # Names of the columns in `chat_studio_df_original` that the user
    # (or pipeline import) marked as sample-intensity columns. Used by
    # the sandbox to pre-coerce these to numeric before exposing the
    # DataFrame to AI-generated code, preventing TypeError on
    # arithmetic when the raw CSV inferred string dtypes.
    chat_studio_sample_columns_original: Optional[List[str]] = None
    # Names of the same columns AFTER the apply-step rename to
    # `{intensity|concentration}[s1..sN]`. Used when the Custom Viz
    # tab pre-coerces `chat_studio_df` (the renamed copy).
    chat_studio_sample_columns_renamed: Optional[List[str]] = None


# Keys that should NOT be reset when starting a fresh analysis
# (they control app-level routing, not data-specific state)
_PRESERVE_ON_RESET = {'page', 'module'}

# Widget keys created by Streamlit widgets (not in SessionState dataclass).
# Must be removed on reset to prevent stale widget state.
_WIDGET_KEYS = {
    # Sidebar widgets (file_upload, experiment_config, sample_grouping, confirm_inputs)
    'data_format_select', 'n_conditions',
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
    # AI Chat widgets (ai_chat/)
    'chat_input', 'chat_apply_transform',
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
    '_nav_preserved_',  # cross-page widget snapshots (preserve_main_app_widgets)
)


# Sidebar widget keys that need a snapshot when the user navigates to a
# different page (main app ↔ AI Studio). Without this, Streamlit garbage-
# collects keys whose widgets did not render in the current run, and the
# user loses their experiment configuration on every round-trip.
_MAIN_APP_NAV_WIDGET_KEYS = (
    'data_format_select', 'n_conditions',
    'bqc_radio', 'bqc_label_radio', 'confirm_checkbox',
    'grouping_radio', 'manual_sample_override',
)
_MAIN_APP_NAV_WIDGET_PREFIXES = (
    'cond_name_', 'n_samples_', 'select_',
)
_AI_STUDIO_NAV_WIDGET_KEYS = (
    'studio_sample_cols_radio', 'studio_sample_cols_select',
    'studio_n_conditions', 'studio_grouping_radio',
    'studio_bqc_radio', 'studio_bqc_label_radio',
    'studio_confirm_checkbox',
)
_AI_STUDIO_NAV_WIDGET_PREFIXES = (
    'studio_cond_name_', 'studio_n_samples_', 'studio_select_',
)
_NAV_PRESERVE_PREFIX = '_nav_preserved_'


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
        # Pop static widget keys first. Streamlit forbids assigning to a key
        # instantiated as a widget during the current run, but allows deletion.
        # Popped keys are reinitialized to dataclass defaults on the next rerun
        # by initialize_session_state().
        for key in _WIDGET_KEYS:
            st.session_state.pop(key, None)
        # Reset SessionState fields, skipping fields that are also widget keys
        # (already popped above — re-assignment would raise if the widget
        # rendered earlier this run).
        defaults = SessionState()
        for f in fields(SessionState):
            if f.name in _PRESERVE_ON_RESET or f.name in _WIDGET_KEYS:
                continue
            st.session_state[f.name] = getattr(defaults, f.name)
        # Remove dynamic widget keys by prefix (e.g. protein_{sample}, conc_{std}).
        # Runs after the SessionState reset so prefix-matching SessionState
        # fields (e.g. protein_df) are also cleared.
        dynamic_keys = [
            key for key in list(st.session_state.keys())
            if isinstance(key, str) and key.startswith(_DYNAMIC_KEY_PREFIXES)
        ]
        for key in dynamic_keys:
            st.session_state.pop(key, None)

    @staticmethod
    def reset_format_derived_state() -> None:
        """Reset all state derived from format-specific standardization.

        Use when the user switches data format mid-session. Preserves the
        raw uploaded file and sample-data flags, but clears all
        standardization output (column_mapping, msdial_features, etc.) and
        all downstream pipeline state (cleaned_df, normalized_df, qc_*, analysis_*).
        """
        preserve = _PRESERVE_ON_RESET | {'raw_df', 'using_sample_data', 'sample_data_file'}
        defaults = SessionState()
        for f in fields(SessionState):
            if f.name not in preserve:
                st.session_state[f.name] = getattr(defaults, f.name)
        for key in _WIDGET_KEYS:
            st.session_state.pop(key, None)
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

    @staticmethod
    def _snapshot_widgets(static_keys, prefixes) -> None:
        """Copy each named/prefixed widget key into a parallel
        ``_nav_preserved_*`` key. Shared by the per-page preserve helpers
        below. Streamlit garbage-collects widget keys whose widgets don't
        render in the current run, so without this snapshot the user
        would lose their sidebar configuration on every cross-page hop.
        """
        keys = list(static_keys)
        keys += [
            k for k in list(st.session_state.keys())
            if isinstance(k, str) and k.startswith(prefixes)
        ]
        for key in keys:
            if key in st.session_state:
                st.session_state[f'{_NAV_PRESERVE_PREFIX}{key}'] = (
                    st.session_state[key]
                )

    @staticmethod
    def preserve_main_app_widgets() -> None:
        """Snapshot main-app sidebar widget keys before the user navigates
        to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        StreamlitAdapter._snapshot_widgets(
            _MAIN_APP_NAV_WIDGET_KEYS, _MAIN_APP_NAV_WIDGET_PREFIXES,
        )

    @staticmethod
    def preserve_ai_studio_widgets() -> None:
        """Snapshot AI Studio sidebar widget keys before the user
        navigates to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        StreamlitAdapter._snapshot_widgets(
            _AI_STUDIO_NAV_WIDGET_KEYS, _AI_STUDIO_NAV_WIDGET_PREFIXES,
        )

    @staticmethod
    def restore_navigation_widgets() -> None:
        """Restore any widget key that was preserved by either
        ``preserve_main_app_widgets()`` or ``preserve_ai_studio_widgets()``
        and is currently absent from session_state. Generic over both
        pages because each page's widget keys have unique prefixes — keys
        not matching any rendered widget on the current page are simply
        unused. If Streamlit kept the widget alive, the live value wins.
        """
        prefix_len = len(_NAV_PRESERVE_PREFIX)
        for key in list(st.session_state.keys()):
            if not isinstance(key, str):
                continue
            if not key.startswith(_NAV_PRESERVE_PREFIX):
                continue
            target = key[prefix_len:]
            if target not in st.session_state:
                st.session_state[target] = st.session_state[key]

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
    ) -> IngestionResult:
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
            IngestionResult with cleaned data, standards, format, and validation info.
        """
        config = IngestionConfig(
            experiment=experiment,
            data_format=data_format,
            bqc_label=bqc_label,
            apply_zero_filter=apply_zero_filter,
            grade_config=grade_config,
            quality_config=quality_config,
        )

        return DataIngestionWorkflow.run(df, config)

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
    ) -> NormalizationWorkflowResult:
        """
        Cached normalization workflow wrapper.

        Args:
            df: Cleaned DataFrame to normalize
            experiment: Experiment configuration
            normalization: Normalization configuration
            data_format: Data format enum
            intsta_df: Internal standards DataFrame

        Returns:
            NormalizationWorkflowResult with normalized data and metadata.
        """
        workflow_config = NormalizationWorkflowConfig(
            experiment=experiment,
            normalization=normalization,
            data_format=data_format
        )

        return NormalizationWorkflow.run(
            df=df,
            config=workflow_config,
            intsta_df=intsta_df
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
    ) -> Tuple[go.Figure, go.Figure, pd.DataFrame, List[float], List[str]]:
        """Cached box plot computation.

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
    ) -> Tuple[go.Figure, pd.DataFrame, float, pd.DataFrame]:
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
    ) -> Tuple[matplotlib.figure.Figure, pd.DataFrame]:
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
    ) -> Tuple[go.Figure, pd.DataFrame]:
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
    def run_retention_time_single(df: pd.DataFrame) -> List[Tuple[go.Figure, pd.DataFrame]]:
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
    ) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
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
    ) -> BarChartResult:
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
    ) -> Dict[str, Any]:
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
    ) -> SaturationResult:
        """Cached saturation analysis."""
        return AnalysisWorkflow.run_saturation(
            df, experiment, selected_conditions, selected_classes,
            stat_config=stat_config, plot_type=plot_type,
            show_significance=show_significance,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Generating chain length distribution...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_chain_length(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> ChainLengthResult:
        """Cached chain length distribution analysis."""
        return AnalysisWorkflow.run_chain_length(
            df, experiment, selected_conditions, selected_classes,
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
    ) -> FACHResult:
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
    ) -> PathwayDataResult:
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
    ) -> PathwayResult:
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
    ) -> VolcanoResult:
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
    ) -> HeatmapResult:
        """Cached lipidomic heatmap analysis."""
        return AnalysisWorkflow.run_heatmap(
            df, experiment, selected_conditions, selected_classes,
            heatmap_type=heatmap_type, n_clusters=n_clusters,
        )

    # ==================== AI Chat Wrappers (NOT cached — stateful) ====================

    @staticmethod
    def run_ai_transform(
        user_message: str,
        session,
        source_df: pd.DataFrame,
    ):
        """Run one turn of the AI data transformation chat.

        NOT cached because chat is inherently stateful.

        Args:
            user_message: The user's message.
            session: Current ChatSession.
            source_df: The user's uploaded DataFrame.

        Returns:
            TransformStepResult from the workflow.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.handle_transform_message(
            user_message=user_message,
            session=session,
            source_df=source_df,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
        )

    @staticmethod
    def run_ai_viz(
        user_message: str,
        session,
        analysis_df: pd.DataFrame,
        experiment: ExperimentConfig,
    ):
        """Run one turn of the AI visualization chat.

        NOT cached because chat is inherently stateful.

        Args:
            user_message: The user's message.
            session: Current ChatSession.
            analysis_df: The analysis-ready DataFrame.
            experiment: Experiment configuration.

        Returns:
            VizStepResult from the workflow.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.handle_viz_message(
            user_message=user_message,
            session=session,
            analysis_df=analysis_df,
            experiment=experiment,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_renamed'
            ),
        )

    @staticmethod
    def run_ai_data_questions(
        user_message: str,
        session,
        source_df: pd.DataFrame,
    ):
        """Run one turn of the AI data questions chat.

        NOT cached because chat is inherently stateful.

        Args:
            user_message: The user's message.
            session: Current ChatSession.
            source_df: The user's DataFrame.

        Returns:
            DataQuestionsStepResult from the workflow.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.handle_data_questions_message(
            user_message=user_message,
            session=session,
            source_df=source_df,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
        )

    @staticmethod
    def run_ai_lc_questions(
        user_message: str,
        session,
    ):
        """Run one turn of the LipidCruncher questions chat.

        NOT cached because chat is inherently stateful.

        Args:
            user_message: The user's message.
            session: Current ChatSession.

        Returns:
            LCQuestionsStepResult from the workflow.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.handle_lc_questions_message(
            user_message=user_message,
            session=session,
        )

    @staticmethod
    def run_ai_unified(
        user_message: str,
        session,
        source_df: pd.DataFrame,
    ):
        """Run one turn of the unified AI Studio chat.

        NOT cached because chat is inherently stateful.

        Args:
            user_message: The user's message.
            session: Current ChatSession.
            source_df: The user's DataFrame.

        Returns:
            UnifiedStepResult from the workflow.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.handle_unified_message(
            user_message=user_message,
            session=session,
            source_df=source_df,
        )
