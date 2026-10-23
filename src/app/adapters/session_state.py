"""
Session state schema: every key the app keeps in ``st.session_state``, the
pipeline stage that owns it, and the helpers that fingerprint a stage's
inputs. ``StreamlitAdapter`` resets and invalidates through this registry.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go

from app.constants import COV_THRESHOLD_DEFAULT, PAGE_LANDING, MODULE_DATA_PROCESSING

from ..models.experiment import ExperimentConfig
from ..models.sample_frame import SampleFrame
from ..services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from ..services.data_standardization import MSDIALFeatures
from ..services.format_detection import DataFormat
from ..services.lsi_report import AnalysisStatistics
from ..workflows.analysis import VolcanoData
from ..workflows.data_ingestion import IngestionResult
from ..workflows.normalization import NormalizationWorkflowResult
from .ai_studio import AI_STUDIO_KEYS, AIStudioState
from .isotope import ISOTOPE_KEYS, IsotopeState


@dataclass
class SessionState(AIStudioState, IsotopeState):
    """
    Type-safe container for all session state variables.

    Defines the complete shape of session state with default values.
    Use this as a reference for what keys exist in session state.

    Ownership model — each key is owned by one UI file:
      file_upload.py       → raw_df, using_sample_data, sample_data_file,
                             upload_kept_across_pages
      column_mapping.py    → standardized_df, column_mapping, n_intensity_cols,
                             format_type, msdial_features, msdial_sample_names,
                             _msdial_override_samples, lipidsearch_sample_names
      experiment_config.py → workbench_conditions, workbench_samples
      sample_grouping.py   → experiment, bqc_label, confirmed, grouping_complete,
                             processing_frame, _preserved_grouping_radio,
                             _preserved_regroup_selections; the regroup
                             snapshot (_pre_regroup_*, original_column_order,
                             regroup_origin) and _preserved_excluded_* are
                             written only through sidebar_state.py
      confirm_inputs.py    → _preserved_sidebar_bqc_radio, _preserved_sidebar_bqc_label_radio
      data_processing.py   → ingestion_result, grade_config, last_quality_config,
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
      quality_check.py     → qc_continuation_df, qc_experiment, qc_bqc_plot,
                             qc_cov_threshold,
                             qc_correlation_plots, qc_pca_plot, qc_samples_removed,
                             _preserved_bqc_filter_choice, _preserved_rt_viewing_mode,
                             _preserved_pca_samples_remove, _preserved_pca_groups_remove
      analysis/            → analysis_selection, analysis_volcano_data,
                             analysis_pathway_*, analysis_all_plots (figures
                             for the PDF, written only by _publish),
                             analysis_statistics (tests run, for the LSI report)
      isotope_tracing_page.py → iso_measurement_df, iso_molecule_df, iso_element_df,
                             iso_result, iso_result_key
      main_app.py          → page, module, cleaned_df and intsta_df (Module 1's
                             final data and standards, written once per run)
    """
    # --- App-level routing (owner: main_app.py) ---
    page: str = PAGE_LANDING
    module: str = MODULE_DATA_PROCESSING

    # --- File upload (owner: file_upload.py) ---
    raw_df: Optional[pd.DataFrame] = None
    using_sample_data: bool = False
    sample_data_file: Optional[str] = None
    # Set when the user leaves for AI Studio with an upload: the uploader
    # comes back empty, and raw_df stands in for it until a new upload.
    upload_kept_across_pages: bool = False

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
    # The regroup answers (radio, picks per condition), kept for runs that
    # stop before the grouping step (an invalid experiment setup) and so drop
    # their widget keys.
    _preserved_grouping_radio: Optional[str] = None
    _preserved_regroup_selections: Dict[str, List[str]] = field(default_factory=dict)
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
    # The experiment as of the end of QC. Differs from `experiment` when the
    # user removed samples in the PCA step, which drops sample columns and
    # renumbers the survivors. Pair it with qc_continuation_df, never the
    # pre-QC `experiment`.
    qc_experiment: Optional[ExperimentConfig] = None
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
    # The tests each analysis ran, for the LSI report (same lifetime as the
    # figures above: cleared when the page re-renders, set when it tests).
    analysis_statistics: Dict[str, AnalysisStatistics] = field(default_factory=dict)

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

    ``derived`` names the ``fields`` computed from the stage's input rather
    than chosen by the user. ``StreamlitAdapter.track_inputs`` resets them
    (and every later stage's) when that input changes, keeping the settings.
    """
    name: str
    fields: frozenset = frozenset()
    widgets: frozenset = frozenset()
    plain: frozenset = frozenset()
    prefixes: Tuple[str, ...] = ()
    derived: frozenset = frozenset()


_STAGES: Tuple[_Stage, ...] = (
    # The uploaded file. Only a new source (or Back to Home) clears it.
    _Stage(
        'upload',
        fields=frozenset({'raw_df', 'using_sample_data', 'sample_data_file',
                          'upload_kept_across_pages'}),
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
            '_preserved_grouping_radio', '_preserved_regroup_selections',
            '_preserved_sidebar_bqc_radio',
            '_preserved_sidebar_bqc_label_radio',
            # What the grouping widgets showed last run (Confirm invalidation)
            '_last_seen_grouping_radio', '_last_seen_selects',
        }),
        widgets=frozenset({
            'n_conditions',
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
        # _nav_preserved_*: sidebar snapshots taken on the way to another
        # page (preserve_main_app_widgets / preserve_ai_studio_widgets).
        prefixes=('cond_name_', 'n_samples_', 'select_', '_nav_preserved_'),
    ),
    # Module 1 after the sidebar: cleaning, standards, zero filter,
    # normalization.
    _Stage(
        'processing',
        fields=frozenset({
            'cleaned_df', 'intsta_df',
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
            'keep_other_adducts', 'keep_other_adducts_checkbox',
            'customize_adducts', 'customize_adducts_checkbox', 'adduct_choices',
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
                  'adduct_select_', 'standards_file_uploader_'),
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
    # The AI Studio and the isotope tracing page. Every main-app reset
    # clears them (they sit before 'qc', so Next / Back do not); their own
    # pages clear just their stage with clear_stage().
    _Stage('ai_studio', **AI_STUDIO_KEYS),
    _Stage('isotope', **ISOTOPE_KEYS),
    # Module 2. Next and Back to Data Processing clear from here.
    _Stage(
        'qc',
        fields=frozenset({
            'qc_continuation_df', 'qc_box_plot_fig1', 'qc_box_plot_fig2',
            'qc_experiment', 'qc_bqc_plot', 'qc_cov_threshold',
            'qc_retention_time_plot',
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
            'analysis_all_plots', 'analysis_statistics',
        }),
        # Every page's results, kept for the reports while another page is
        # shown; computed from Quality Check's output.
        derived=frozenset({
            'analysis_all_plots', 'analysis_statistics', 'analysis_volcano_data',
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


def _inputs_key(stage: str) -> str:
    """Session key holding the fingerprint of ``stage``'s last inputs."""
    return f'_inputs_{stage}'


def _fingerprint(values: Tuple[Any, ...]) -> str:
    """A digest that changes when any of ``values`` does (frames by content)."""
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, pd.DataFrame):
            digest.update(repr(list(value.columns)).encode())
            digest.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b'\0')
    return digest.hexdigest()

# App-level state no reset clears: routing, navigation buttons, and the
# AI Studio's LipidCruncher Questions chat (about the app, not the data).
_ROUTING_FIELDS = frozenset({'page', 'module', 'chat_lc_questions_session'})
_ROUTING_WIDGETS = frozenset({
    'back_home_no_data', 'back_home_module1', 'back_home_module2',
    'back_processing_error', 'back_processing_module2', 'reset_session',
    'open_studio_module2', 'ai_studio_back_home', 'ai_studio_back_main',
    'ai_studio_goto_main_from_advisory', 'iso_back_home',
    # The format dropdown keeps its choice through every reset; a format
    # change itself is what reset_format_derived_state() reacts to.
    'data_format_select',
})
