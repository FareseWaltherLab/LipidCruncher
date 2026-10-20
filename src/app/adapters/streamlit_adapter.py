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

from app.constants import (
    COV_THRESHOLD_DEFAULT, PAGE_LANDING, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS,
    resolve_format_enum,
)

from ..models.experiment import ExperimentConfig
from ..services.data_standardization import MSDIALFeatures
from ..models.normalization import NormalizationConfig
from ..models.sample_frame import SampleFrame
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
    GROUPED_PAGE_SIZE,
    AnalysisWorkflow,
    BarChartResult,
    ChainLengthResult,
    FACHResult,
    HeatmapResult,
    PathwayDataResult,
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
                             _msdial_override_samples, lipidsearch_sample_names
      experiment_config.py → workbench_conditions, workbench_samples
      sample_grouping.py   → experiment, bqc_label, confirmed, grouping_complete,
                             processing_frame, _preserved_excluded_samples,
                             original_column_order, regroup_origin
      confirm_inputs.py    → _preserved_sidebar_bqc_radio, _preserved_sidebar_bqc_label_radio
      data_processing.py   → cleaned_df, intsta_df, continuation_df, ingestion_result,
                             pre_filter_df, grade_config, last_quality_config,
                             msdial_use_normalized, msdial_data_type_index
      internal_standards.py → original_auto_intsta_df, custom_standards_df,
                              custom_standards_mode, standards_source
      standards_plots.py   → standards_consistency_figs
      zero_filtering.py    → _zero_filter_format, _preserved_zero_filter_*
      normalization.py     → normalized_df, selected_classes,
                             norm_method_selection,
                             _preserved_norm_method_selection, normalization_result,
                             class_standard_map, standard_concentrations, protein_df,
                             protein_input_method, protein_input_method_prev
      quality_check.py     → qc_continuation_df, qc_bqc_plot, qc_cov_threshold,
                             qc_correlation_plots, qc_pca_plot, qc_samples_removed,
                             _preserved_bqc_filter_choice, _preserved_rt_viewing_mode,
                             _preserved_pca_samples_remove, _preserved_pca_groups_remove
      analysis/            → analysis_selection, analysis_volcano_data,
                             analysis_pathway_*, analysis_all_plots (figures
                             for the PDF, written only by _publish)
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
    # LipidSearch 5.2 alignment: s-label → sample name from the raw filenames.
    lipidsearch_sample_names: Optional[Dict[str, str]] = None

    # --- Experiment config (owner: experiment_config.py) ---
    workbench_conditions: Optional[List[str]] = None
    workbench_samples: Optional[Dict] = None

    # --- Sample grouping (owner: sample_grouping.py) ---
    experiment: Optional[ExperimentConfig] = None
    bqc_label: Optional[str] = None
    confirmed: bool = False
    grouping_complete: bool = True
    original_column_order: Optional[List[str]] = None
    # Maps internal sample label (s1, s2, ...) → user-facing display name,
    # auto-seeded from the uploaded column headers and editable in the sidebar.
    sample_names: Optional[Dict[str, str]] = None
    # Module 1's input after the sidebar's optional sample exclusion:
    # standardized_df without the excluded columns (survivors renumbered),
    # with the experiment, sample_names re-keyed the same way and each
    # sample's upload label (so per-sample inputs such as protein amounts
    # can be keyed by sample rather than by a label that regroup/exclusion
    # renumber). The sidebar keeps using the full dataset and experiment.
    processing_frame: Optional[SampleFrame] = None
    # Sidebar exclusions as upload labels, and the regroup they were saved under.
    _preserved_excluded_samples: List[str] = field(default_factory=list)
    _preserved_excluded_regroup_origin: Optional[Dict[str, str]] = None
    # Sidebar label -> upload label (None when not regrouped).
    regroup_origin: Optional[Dict[str, str]] = None
    # BQC answers, kept for runs where the BQC widgets are not drawn (e.g. a
    # manual regroup is mid-edit) and Streamlit drops their keys.
    _preserved_sidebar_bqc_radio: Optional[str] = None
    _preserved_sidebar_bqc_label_radio: Optional[str] = None
    # 'original' shows those names in tables, plots, pickers and downloads;
    # 'standardized' shows s1, s2, ... instead (sidebar switch).
    sample_label_mode: str = 'original'

    # --- Data processing (owner: data_processing.py) ---
    cleaned_df: Optional[pd.DataFrame] = None
    intsta_df: Optional[pd.DataFrame] = None
    continuation_df: Optional[pd.DataFrame] = None
    pre_filter_df: Optional[pd.DataFrame] = None
    ingestion_result: Optional[IngestionResult] = None
    grade_config: Optional[GradeFilterConfig] = None
    last_quality_config: Optional[QualityFilterConfig] = None
    msdial_use_normalized: bool = False
    msdial_data_type_index: int = 0  # 0 = raw, 1 = pre-normalized

    # --- Internal standards (owner: internal_standards.py) ---
    original_auto_intsta_df: Optional[pd.DataFrame] = None
    custom_standards_df: Optional[pd.DataFrame] = None
    custom_standards_mode: Optional[str] = None
    # Which samples a 'complete' table's columns were mapped to (by position);
    # the table is dropped when sidebar regroup/exclusion renumbers them.
    custom_standards_sample_set: Optional[str] = None
    standards_source: Optional[str] = None
    # Consistency bar charts, one per internal-standard class (owner:
    # standards_plots.py). Held so the PDF report can include them —
    # Module 1 plots have no other route into the report.
    standards_consistency_figs: List[go.Figure] = field(default_factory=list)

    # --- Zero filtering (owner: zero_filtering.py) ---
    _zero_filter_format: Optional[str] = None
    _preserved_zero_filter_detection_threshold: Optional[float] = None
    _preserved_non_bqc_zero_threshold: Optional[float] = None
    _preserved_bqc_zero_threshold: Optional[float] = None

    # --- Normalization (owner: normalization.py) ---
    normalized_df: Optional[pd.DataFrame] = None
    selected_classes: List[str] = field(default_factory=list)
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
    # processing_frame (Module 1's input, experiment, names, origins) and the
    # BQC label that produced normalized_df, frozen at Module 1's handoff:
    # the sidebar keeps running on Module 2/3, and a later exclusion or
    # regroup must not be paired with Module 1's already-computed data.
    qc_input_frame: Optional[SampleFrame] = None
    qc_input_bqc_label: Optional[str] = None
    # The samples left after PCA sample exclusion (which renumbers
    # survivors); None when nothing is excluded.
    qc_frame: Optional[SampleFrame] = None
    _preserved_bqc_filter_choice: str = 'No'
    _preserved_rt_viewing_mode: str = 'Comparison Mode'
    _preserved_pca_samples_remove: List[str] = field(default_factory=list)
    _preserved_pca_groups_remove: List[str] = field(default_factory=list)

    # --- Analysis (owner: analysis.py) ---
    analysis_selection: Optional[str] = None
    analysis_pathway_added_edges: List[Tuple[str, str]] = field(default_factory=list)
    analysis_pathway_removed_edges: List[Tuple[str, str]] = field(default_factory=list)
    analysis_pathway_custom_nodes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    analysis_pathway_position_overrides: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    analysis_volcano_data: Optional[VolcanoData] = None
    analysis_all_plots: Dict[str, Union[go.Figure, matplotlib.figure.Figure]] = field(default_factory=dict)

    # --- LSI Compliance Report (owner: lsi_report.py) ---
    lsi_manual_fields: Dict[str, str] = field(default_factory=dict)
    lsi_pdf_bytes: Optional[bytes] = None
    lsi_pdf_source: Optional[str] = None  # the CSV the PDF was made from


@dataclass(frozen=True)
class _Stage:
    """One step of the pipeline and every session key it owns.

    Invalidating a stage invalidates it and every later one (``_STAGES``
    order), so a key belongs to the earliest stage whose change makes it
    stale. Four kinds of key:

    - ``fields``: SessionState fields, reset to their dataclass default.
    - ``widgets``: widget keys, popped (Streamlit forbids assigning to an
      instantiated widget's key, but allows deleting it). A SessionState
      field that a widget also writes goes here, not in ``fields``.
    - ``plain``: other session keys, popped.
    - ``prefixes``: keys built with f-strings (``protein_{sample}``),
      popped. A SessionState field that happens to share the prefix
      (``protein_df``, ``lsi_manual_fields``) is reset like any field, not
      popped, so code later in the same run can still read it.
    """
    name: str
    fields: frozenset = frozenset()
    widgets: frozenset = frozenset()
    plain: frozenset = frozenset()
    prefixes: Tuple[str, ...] = ()


_STAGES: Tuple[_Stage, ...] = (
    # The uploaded file. Only a new source (or Back to Home) clears it.
    _Stage(
        'upload',
        fields=frozenset({'raw_df', 'using_sample_data', 'sample_data_file'}),
        widgets=frozenset({'load_sample'}),
        plain=frozenset({'_dataset_source_id'}),
        prefixes=('lipidsearch_alignment_uploader_',),
    ),
    # The upload read under the selected format. A format switch clears it.
    _Stage(
        'standardize',
        fields=frozenset({
            'standardized_df', 'column_mapping', 'n_intensity_cols',
            'format_type', 'msdial_features', 'msdial_sample_names',
            '_msdial_override_samples', 'lipidsearch_sample_names',
            'workbench_conditions', 'workbench_samples',
            # Kept through "Reset session" with the standardized data they
            # describe (their widgets below are not, and re-seed from them).
            'sample_label_mode', 'msdial_use_normalized',
            'msdial_data_type_index',
        }),
    ),
    # Define Experiment, sample names, grouping, exclusion, BQC, Confirm.
    # "Reset session" clears from here.
    _Stage(
        'experiment',
        fields=frozenset({
            'experiment', 'bqc_label', 'confirmed', 'grouping_complete',
            'original_column_order', 'sample_names', 'processing_frame',
            '_preserved_excluded_samples',
            '_preserved_excluded_regroup_origin', 'regroup_origin',
            '_preserved_sidebar_bqc_radio',
            '_preserved_sidebar_bqc_label_radio',
        }),
        widgets=frozenset({
            'manual_sample_override', 'grouping_radio', 'exclude_samples',
            'bqc_radio', 'bqc_label_radio', 'confirm_checkbox',
            'sample_data_experiment', 'sample_names_editor',
            '_sample_label_mode_radio',
        }),
        # The regroup snapshot (restored by the grouping step's "Yes"), and
        # the marker that stops the LipidSearch alignment pre-filling the
        # experiment twice for the same file.
        plain=frozenset({
            '_pre_regroup_df', '_pre_regroup_sample_names',
            '_ls_align_prefilled_for',
        }),
        prefixes=('cond_name_', 'n_samples_', 'select_'),
    ),
    # Module 1 after the sidebar: cleaning, standards, zero filter,
    # normalization.
    _Stage(
        'processing',
        fields=frozenset({
            'cleaned_df', 'intsta_df', 'continuation_df', 'pre_filter_df',
            'ingestion_result', 'grade_config', 'last_quality_config',
            'original_auto_intsta_df', 'custom_standards_df',
            'custom_standards_mode', 'custom_standards_sample_set',
            'standards_source', 'standards_consistency_figs',
            '_zero_filter_format', '_preserved_zero_filter_detection_threshold',
            '_preserved_non_bqc_zero_threshold', '_preserved_bqc_zero_threshold',
            'normalized_df', 'selected_classes',
            '_preserved_norm_method_selection', 'normalization_result',
            'class_standard_map', 'standard_concentrations', 'protein_df',
            'protein_input_method_prev',
        }),
        widgets=frozenset({
            # data_processing.py
            'grade_filter_mode', 'grade_selections', 'grade_filter_mode_radio',
            'msdial_quality_level', 'msdial_data_type_radio',
            'msdial_quality_level_radio', 'msdial_custom_msms',
            'msdial_show_custom_threshold', 'msdial_custom_score',
            'msdial_msms_only', 'download_removed_species',
            # zero_filtering.py
            'zero_filter_detection_threshold', 'non_bqc_zero_threshold',
            'bqc_zero_threshold',
            # internal_standards.py
            'standards_source_radio', 'standards_location_radio',
            'standards_conditions_select', 'clear_custom_standards',
            'additional_istd_select', 'dataset_standards_select',
            'download_auto_standards', 'download_custom_standards',
            'download_selected_standards', 'download_combined_standards',
            # normalization.py (two of these are also SessionState fields)
            'temp_selected_classes', 'norm_method_selection',
            'protein_input_method', 'download_filtered_data',
            'download_normalized_data',
        }),
        prefixes=('protein_', 'conc_', 'standard_selection_', 'grade_select_',
                  'standards_file_uploader_'),
    ),
    # The LSI compliance report's typed-in fields: kept across Next / Back,
    # cleared with the experiment they describe.
    _Stage(
        'report',
        fields=frozenset({'lsi_manual_fields', 'lsi_pdf_bytes',
                           'lsi_pdf_source'}),
        widgets=frozenset({'lsi_download_csv', 'lsi_generate_pdf',
                           'lsi_download_pdf'}),
        prefixes=('lsi_manual_',),
    ),
    # Module 2. Next and Back to Data Processing clear from here.
    _Stage(
        'qc',
        fields=frozenset({
            'qc_continuation_df', 'qc_box_plot_fig1', 'qc_box_plot_fig2',
            'qc_bqc_plot', 'qc_cov_threshold', 'qc_retention_time_plot',
            'qc_correlation_plots', 'qc_pca_plot', 'qc_samples_removed',
            'qc_input_frame', 'qc_input_bqc_label', 'qc_frame',
            '_preserved_bqc_filter_choice', '_preserved_rt_viewing_mode',
            '_preserved_pca_samples_remove', '_preserved_pca_groups_remove',
        }),
        widgets=frozenset({
            'bqc_cov_threshold', 'bqc_filter_choice', 'bqc_lipids_to_keep',
            'bqc_csv_download', 'bqc_filtered_download', 'rt_viewing_mode',
            'rt_class_selection', 'qc_rt_svg_comparison', 'rt_csv_comparison',
            'corr_condition', 'corr_csv_download', 'qc_corr_svg',
            'pca_samples_remove', 'pca_groups_remove', 'pca_csv_download',
            'qc_missing_values_svg', 'qc_missing_values_csv',
            'qc_box_plot_svg', 'qc_box_plot_csv', 'qc_bqc_svg', 'qc_pca_svg',
        }),
        prefixes=('qc_rt_svg_individual_', 'rt_csv_individual_'),
    ),
    # Module 3.
    _Stage(
        'analysis',
        fields=frozenset({
            'analysis_selection', 
            
            
            'analysis_pathway_added_edges',
            'analysis_pathway_removed_edges', 'analysis_pathway_custom_nodes',
            'analysis_pathway_position_overrides', 
            'analysis_volcano_data', 
            'analysis_all_plots',
        }),
        widgets=frozenset({
            'analysis_radio', 'generate_pdf_report', 'download_pdf_report',
            # bar chart / saturation (keys built from a 'bar' / 'sat' prefix)
            'bar_conditions', 'bar_classes', 'bar_scale_radio',
            'bar_stats_mode', 'bar_detailed_stats', 'bar_auto_transform',
            'bar_test_type', 'bar_correction', 'bar_posthoc',
            'pie_classes',
            'sat_conditions', 'sat_classes', 'sat_plot_type',
            'sat_show_significance', 'sat_detailed_stats', 'sat_stats_mode',
            'sat_auto_transform', 'sat_test_type', 'sat_correction',
            'sat_posthoc', 'sat_exclude_consolidated',
            'clen_conditions', 'clen_classes',
            'fach_class', 'fach_conditions',
            'pathway_control', 'pathway_experimental', 'pathway_class_selector',
            'pathway_add_node_name', 'pathway_add_node_x', 'pathway_add_node_y',
            'pathway_add_node_btn', 'pathway_add_edge_source',
            'pathway_add_edge_target', 'pathway_add_edge_btn',
            'pathway_edge_to_remove_select', 'pathway_remove_edge_btn',
            'pathway_move_node_select', 'pathway_move_node_x',
            'pathway_move_node_y', 'pathway_move_node_btn', 'pathway_show_grid',
            'pathway_start_all', 'pathway_start_default',
            'pathway_start_scratch', 'pathway_upload_config',
            'pathway_download_config', 'pathway_exclude_consolidated',
            'volcano_control', 'volcano_experimental', 'volcano_classes',
            'volcano_p_threshold', 'volcano_fc_threshold', 'volcano_hide_nonsig',
            'volcano_top_n', 'volcano_additional_labels', 'volcano_stats_mode',
            'volcano_detailed_stats', 'volcano_correction',
            'volcano_auto_transform', 'volcano_enable_adjust',
            'volcano_detail_class', 'volcano_detail_lipids',
            'heatmap_conditions', 'heatmap_classes', 'heatmap_type',
            'heatmap_n_clusters', 'heatmap_cluster_view', 'heatmap_color_scale',
            'heatmap_control_condition', 'heatmap_species_page',
            'heatmap_species_page_size', 'heatmap_species_sort',
        }),
        prefixes=('volcano_label_x_', 'volcano_label_y_', 'analysis_svg_',
                  'analysis_csv_'),
    ),
)

_STAGE_NAMES = tuple(stage.name for stage in _STAGES)

# App-level routing and navigation buttons: never invalidated, only set.
_ROUTING_FIELDS = frozenset({'page', 'module'})
_ROUTING_WIDGETS = frozenset({
    'back_home_no_data', 'back_home_module1', 'back_home_module2',
    'back_processing_error', 'back_processing_module2', 'reset_session',
})


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
    def invalidate_from(stage: str) -> None:
        """Clear ``stage`` and every later stage of ``_STAGES``.

        Fields go back to their defaults; widget, plain and prefixed keys are
        popped (re-created by their widgets, or re-initialized from
        ``SessionState`` on the next run).

        Args:
            stage: A ``_STAGES`` name ('upload', 'standardize', 'experiment',
                'processing', 'report', 'qc', 'analysis').
        """
        stages = _STAGES[_STAGE_NAMES.index(stage):]
        # Pop widget and plain keys first: Streamlit forbids assigning to a
        # key instantiated as a widget during the current run, but allows
        # deletion.
        for s in stages:
            for key in s.widgets | s.plain:
                st.session_state.pop(key, None)
        defaults = SessionState()
        for s in stages:
            for name in s.fields:
                st.session_state[name] = getattr(defaults, name)
        prefixes = tuple(p for s in stages for p in s.prefixes)
        if prefixes:
            field_names = {f.name for f in fields(SessionState)}
            for key in [k for k in list(st.session_state.keys())
                        if isinstance(k, str) and k.startswith(prefixes)
                        and k not in field_names]:
                st.session_state.pop(key, None)

    @staticmethod
    def reset_data_state() -> None:
        """Reset all data-related session state for a fresh upload.

        Clears every stage and returns to Module 1; only ``page`` (app-level
        routing) is kept.
        """
        StreamlitAdapter.invalidate_from('upload')
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def reset_for_new_source(source_id: str) -> bool:
        """Reset data state when a different dataset source arrives.

        Uploading another file, or loading sample data, over a loaded dataset
        would otherwise leave every downstream frame (standardized, cleaned,
        normalized, QC, analysis) computed from the previous one. The source
        id is stable across reruns (an upload's file_id, or the sample-data
        format), so the same file surviving a rerun does not reset anything.

        Args:
            source_id: Identifier for the incoming dataset.

        Returns:
            True if a loaded dataset was reset.
        """
        if st.session_state.get('_dataset_source_id') == source_id:
            return False
        had_data = st.session_state.get('raw_df') is not None
        if had_data:
            StreamlitAdapter.reset_data_state()
            st.cache_data.clear()
        st.session_state['_dataset_source_id'] = source_id
        return had_data

    @staticmethod
    def reset_format_derived_state() -> None:
        """Clear everything derived from the upload when the format changes.

        Keeps the uploaded file (and the sample-data flags) so it is
        re-standardized under the newly selected format; standardization
        outputs, experiment setup and all downstream state are cleared, and
        the user returns to Module 1.
        """
        StreamlitAdapter.invalidate_from('standardize')
        # Same reason as in reset_to_experiment_setup(): an explicit False
        # beats the checkbox's cached value on a pre-fill rerun.
        st.session_state['confirm_checkbox'] = False
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def reset_to_experiment_setup() -> None:
        """Soft reset: keep the uploaded/standardized data, clear everything else.

        Clears experiment config, grouping, processing, normalization, QC and
        analysis state plus the ``@st.cache_data`` cache, then returns the user
        to the Define Experiment step — all via a rerun, so no page refresh is
        needed. Unlike ``reset_data_state()`` (a full wipe back to landing),
        this preserves the upload and standardization outputs so the user does
        not have to re-upload the file or re-map columns.
        """
        # A confirmed manual regroup leaves standardized_df renumbered, which no
        # longer matches the column mapping or the seeded sample names. Restore
        # the pre-regroup data, since the grouping itself is being reset.
        pre_regroup = st.session_state.get('_pre_regroup_df')
        StreamlitAdapter.invalidate_from('experiment')
        if pre_regroup is not None:
            st.session_state['standardized_df'] = pre_regroup
        # Force the confirm checkbox off. Popping its key is not enough: the
        # reset button renders before the confirm checkbox, and formats that
        # pre-fill the experiment (LipidSearch 5.2 alignment, Metabolomics
        # Workbench auto-detect) fire an extra st.rerun() in which Streamlit
        # re-creates the checkbox with identical parameters and restores its
        # cached 'checked' value. An explicit False in session state overrides
        # that. Safe to set: the widget is not instantiated yet this run.
        st.session_state['confirm_checkbox'] = False
        # Start from Module 1 so a re-confirmed experiment runs the pipeline
        # from the top rather than jumping into a stale QC/Analysis view.
        st.session_state['module'] = MODULE_DATA_PROCESSING
        # Drop cached computations (box plots, PCA, heatmaps, ...).
        st.cache_data.clear()

    # ==================== Navigation ====================

    @staticmethod
    def go_to_quality_check() -> None:
        """Next: Module 1 -> Quality Check & Analysis.

        QC and analysis start over, so Module 2/3 freeze the current Module 1
        output (see the qc_input_* fields) rather than reuse a previous one.
        """
        StreamlitAdapter.invalidate_from('qc')
        st.session_state['module'] = MODULE_QC_ANALYSIS

    @staticmethod
    def back_to_data_processing() -> None:
        """Back from Quality Check & Analysis to Module 1."""
        StreamlitAdapter.invalidate_from('qc')
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def back_to_home() -> None:
        """Back to the landing page, dropping the dataset."""
        st.session_state['page'] = PAGE_LANDING
        StreamlitAdapter.reset_data_state()

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

    # ==================== Service Wrappers ====================


    @staticmethod
    def detect_format_mismatch(df: pd.DataFrame, data_format: str) -> Optional[DataFormat]:
        """Return the format ``df`` looks like when it is not the selected one.

        None when detection agrees or is inconclusive (UNKNOWN). The dropdown
        alone decides how the file is read; this only feeds a warning.
        """
        detected = FormatDetectionService.detect_format(df)
        if detected in (DataFormat.UNKNOWN, resolve_format_enum(data_format)):
            return None
        return detected

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
        grade_config: Optional[GradeFilterConfig] = None,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> IngestionResult:
        """
        Cached data ingestion workflow wrapper.

        Args:
            df: Raw DataFrame to process
            experiment: Experiment configuration
            data_format: Data format enum
            grade_config: Grade filter config (LipidSearch only)
            quality_config: Quality filter config (MS-DIAL only)

        Returns:
            IngestionResult with cleaned data, standards, format, and validation info.
        """
        config = IngestionConfig(
            experiment=experiment,
            data_format=data_format,
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
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[go.Figure, go.Figure, pd.DataFrame, List[float], List[str]]:
        """Cached box plot computation.

        ``display_names`` (``{s-label -> text shown}``) labels the samples in
        both figures; the returned data stays keyed by s-label.

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
            display_names=display_names,
        )
        fig2 = BoxPlotService.plot_box_plot(
            mean_area_df, samples,
            experiment.conditions_list, experiment.individual_samples_list,
            display_names=display_names,
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
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[matplotlib.figure.Figure, pd.DataFrame]:
        """Cached pairwise correlation computation.

        ``display_names`` (``{s-label -> text shown}``) labels the heatmap's
        ticks; ``correlation_df`` stays keyed by s-label.

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
            display_names=display_names,
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
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Cached PCA computation.

        ``display_names`` (``{s-label -> text shown}``) labels the hover text;
        ``pca_df`` keeps the s-labels.

        Returns:
            Tuple of (pca_plot, pca_df)
        """
        return PCAPlotterService.plot_pca(
            df, experiment.full_samples_list,
            experiment.extensive_conditions_list,
            display_names=display_names,
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
        species_page: int = 0,
        color_scale: str = 'zscore',
        control_condition: Optional[str] = None,
        sort_direction: str = 'desc',
        display_names: Optional[Dict[str, str]] = None,
        species_page_size: int = GROUPED_PAGE_SIZE,
    ) -> HeatmapResult:
        """Cached lipidomic heatmap analysis.

        ``display_names`` is part of the cache key, so switching between
        original names and standardized labels redraws the sample axis.
        """
        return AnalysisWorkflow.run_heatmap(
            df, experiment, selected_conditions, selected_classes,
            heatmap_type=heatmap_type, n_clusters=n_clusters,
            species_page=species_page, color_scale=color_scale,
            control_condition=control_condition,
            sort_direction=sort_direction,
            display_names=display_names,
            species_page_size=species_page_size,
        )
