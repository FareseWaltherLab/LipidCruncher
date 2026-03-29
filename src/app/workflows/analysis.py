"""
Analysis workflow.

Orchestrates the lipid analysis pipeline steps:
Bar chart → Pie chart → Saturation → FACH → Pathway → Volcano → Heatmap

Unlike DataIngestionWorkflow/NormalizationWorkflow, this workflow does NOT
have a single run() method. Analysis is interactive — users select analysis
types and configure parameters. Each method is called individually from the
UI layer.

All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..models.experiment import ExperimentConfig
from ..models.statistics import StatisticalTestConfig
from ..services.format_detection import DataFormat
from ..services.statistical_testing import (
    StatisticalTestingService,
    StatisticalTestSummary,
)
from ..services.validation import get_matching_concentration_columns
from ..services.plotting.abundance_bar_chart import (
    BarChartPlotterService,
    BarChartData,
)
from ..services.plotting.abundance_pie_chart import (
    PieChartPlotterService,
    PieChartData,
)
from ..services.plotting.saturation_plot import (
    SaturationPlotterService,
    SaturationData,
)
from ..services.plotting.fach import (
    FACHPlotterService,
    FACHData,
)
from ..services.plotting.pathway_viz import (
    PathwayVizPlotterService,
    PathwayData,
)
from ..services.plotting.volcano_plot import (
    VolcanoPlotterService,
    VolcanoData,
)
from ..services.plotting.lipidomic_heatmap import (
    LipidomicHeatmapPlotterService,
    ClusteringResult,
)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AnalysisConfig:
    """Configuration for the analysis workflow.

    Attributes:
        format_type: Data format (affects available analyses).
        bqc_label: Label of the BQC condition, or None if no BQC.
    """
    format_type: DataFormat = DataFormat.GENERIC
    bqc_label: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BarChartResult:
    """Result from bar chart analysis.

    Attributes:
        figure: Plotly bar chart figure.
        abundance_df: DataFrame with mean/std per class per condition.
        stat_summary: Statistical test results (if tests were run).
    """
    figure: go.Figure
    abundance_df: pd.DataFrame
    stat_summary: Optional[StatisticalTestSummary] = None


@dataclass
class PieChartResult:
    """Result for a single condition pie chart.

    Attributes:
        figure: Plotly pie chart figure.
        data_df: DataFrame with class proportions for this condition.
        condition: The condition this chart represents.
    """
    figure: go.Figure
    data_df: pd.DataFrame
    condition: str = ""


@dataclass
class SaturationResult:
    """Result from saturation analysis.

    Attributes:
        plots: Dict mapping class name to its Plotly figure.
        stat_summary: Statistical test results (if tests were run).
        consolidated_lipids: Dict mapping class → list of consolidated lipid names.
    """
    plots: Dict[str, go.Figure] = dataclass_field(default_factory=dict)
    stat_summary: Optional[StatisticalTestSummary] = None
    consolidated_lipids: Dict[str, List[str]] = dataclass_field(
        default_factory=dict
    )


@dataclass
class FACHResult:
    """Result from FACH analysis.

    Attributes:
        figure: Plotly FACH heatmap figure (None if no parsable data).
        data_dict: Per-condition DataFrames with Carbon/DB/Proportion.
        unparsable_lipids: Lipid names that could not be parsed.
        weighted_averages: Per-condition (avg_carbon, avg_db) tuples.
    """
    figure: Optional[go.Figure] = None
    data_dict: Dict[str, pd.DataFrame] = dataclass_field(default_factory=dict)
    unparsable_lipids: List[str] = dataclass_field(default_factory=list)
    weighted_averages: Dict[str, Tuple[float, float]] = dataclass_field(
        default_factory=dict
    )


@dataclass
class PathwayDataResult:
    """Result from pathway data computation (no figure).

    Attributes:
        fold_change_df: Per-class fold change DataFrame.
        saturation_df: Per-class saturation ratio DataFrame.
    """
    fold_change_df: pd.DataFrame = dataclass_field(
        default_factory=pd.DataFrame
    )
    saturation_df: pd.DataFrame = dataclass_field(
        default_factory=pd.DataFrame
    )


@dataclass
class PathwayResult:
    """Result from pathway visualization.

    Attributes:
        figure: Matplotlib pathway figure (None if no data).
        pathway_dict: Dict with class, abundance ratio, saturation ratio.
        fold_change_df: Per-class fold change DataFrame.
        saturation_df: Per-class saturation ratio DataFrame.
    """
    figure: Optional[plt.Figure] = None
    pathway_dict: Dict[str, list] = dataclass_field(default_factory=dict)
    fold_change_df: pd.DataFrame = dataclass_field(
        default_factory=pd.DataFrame
    )
    saturation_df: pd.DataFrame = dataclass_field(
        default_factory=pd.DataFrame
    )


@dataclass
class VolcanoResult:
    """Result from volcano analysis.

    Attributes:
        figure: Plotly volcano scatter figure.
        volcano_data: Full VolcanoData with per-lipid stats.
        concentration_plot: Plotly concentration-vs-fold-change figure.
        concentration_df: DataFrame from concentration plot.
        stat_summary: The statistical test summary used.
    """
    figure: Optional[go.Figure] = None
    volcano_data: Optional[VolcanoData] = None
    concentration_plot: Optional[go.Figure] = None
    concentration_df: Optional[pd.DataFrame] = None
    stat_summary: Optional[StatisticalTestSummary] = None


@dataclass
class HeatmapResult:
    """Result from lipidomic heatmap analysis.

    Attributes:
        figure: Plotly heatmap figure.
        z_scores_df: Row-wise Z-score DataFrame.
        cluster_composition: Cluster composition DataFrame (clustered mode only).
    """
    figure: Optional[go.Figure] = None
    z_scores_df: Optional[pd.DataFrame] = None
    cluster_composition: Optional[pd.DataFrame] = None


# ═══════════════════════════════════════════════════════════════════════
# Workflow
# ═══════════════════════════════════════════════════════════════════════


class AnalysisWorkflow:
    """Orchestrates analysis steps for lipidomic data.

    Each public method corresponds to one analysis type in Module 3.
    The UI layer calls them individually, allowing user interaction
    (selecting classes, conditions, statistical options, etc.).

    All methods are static — no instance state is stored.
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_inputs(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Validate that data is ready for analysis.

        Args:
            df: DataFrame with concentration[sample] columns.
            experiment: Experiment configuration.

        Returns:
            List of validation error messages (empty if all valid).
        """
        errors = []

        if df is None or df.empty:
            errors.append(
                "Input DataFrame is empty. Quality checks may have removed all lipids. "
                "Check filtering and quality control steps."
            )
            return errors

        available = ', '.join(list(df.columns)[:15])

        if 'LipidMolec' not in df.columns:
            errors.append(f"Input DataFrame missing 'LipidMolec' column. Available columns: [{available}]")

        if 'ClassKey' not in df.columns:
            errors.append(f"Input DataFrame missing 'ClassKey' column. Available columns: [{available}]")

        conc_cols = [
            col for col in df.columns if col.startswith('concentration[')
        ]
        if not conc_cols:
            errors.append(
                "Input DataFrame has no concentration columns. "
                f"Expected 'concentration[s1]', etc. Available columns: [{available}]. "
                "Run normalization and QC before analysis."
            )
            return errors

        matching = get_matching_concentration_columns(df, experiment)
        if not matching:
            expected = ', '.join([f"concentration[{s}]" for s in experiment.full_samples_list[:5]])
            found = ', '.join(conc_cols[:5])
            errors.append(
                f"No concentration columns match the experiment's sample labels. "
                f"Expected: [{expected}, ...]. Found: [{found}]"
            )

        return errors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_classes(df: pd.DataFrame) -> List[str]:
        """Extract unique ClassKey values from the DataFrame.

        Args:
            df: DataFrame with ClassKey column.

        Returns:
            Sorted list of unique class names.

        Raises:
            ValueError: If ClassKey column is missing.
        """
        if 'ClassKey' not in df.columns:
            raise ValueError("DataFrame missing 'ClassKey' column")
        classes = df['ClassKey'].dropna().unique().tolist()
        return sorted(classes)

    @staticmethod
    def get_eligible_conditions(experiment: ExperimentConfig) -> List[str]:
        """Get conditions with more than 1 sample (suitable for stats).

        Args:
            experiment: Experiment configuration.

        Returns:
            List of condition labels with >1 replicate.
        """
        return [
            cond
            for cond, n in zip(
                experiment.conditions_list,
                experiment.number_of_samples_list,
            )
            if n > 1
        ]

    @staticmethod
    def get_all_conditions(experiment: ExperimentConfig) -> List[str]:
        """Get all condition labels.

        Args:
            experiment: Experiment configuration.

        Returns:
            List of all condition labels.
        """
        return list(experiment.conditions_list)

    @staticmethod
    def _get_samples_for_condition(
        experiment: ExperimentConfig,
        condition: str,
    ) -> List[str]:
        """Get sample labels for a specific condition.

        Args:
            experiment: Experiment configuration.
            condition: Condition name.

        Returns:
            List of sample labels (e.g., ['s1', 's2', 's3']).

        Raises:
            ValueError: If condition not found.
        """
        if condition not in experiment.conditions_list:
            raise ValueError(
                f"Condition '{condition}' not found in experiment. "
                f"Available: {experiment.conditions_list}"
            )
        idx = experiment.conditions_list.index(condition)
        return experiment.individual_samples_list[idx]

    # ------------------------------------------------------------------
    # Bar Chart
    # ------------------------------------------------------------------

    @staticmethod
    def run_bar_chart(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        stat_config: Optional[StatisticalTestConfig] = None,
        scale: str = 'linear',
    ) -> BarChartResult:
        """Run abundance bar chart analysis.

        Computes class-level mean/std, optionally runs statistical tests,
        and creates the bar chart figure.

        Args:
            df: DataFrame with ClassKey and concentration[s*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.
            stat_config: Statistical test config (None to skip tests).
            scale: 'linear' or 'log10'.

        Returns:
            BarChartResult with figure, abundance data, and optional stats.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")

        bar_data = BarChartPlotterService.create_mean_std_data(
            df, experiment, selected_conditions, selected_classes,
        )

        stat_summary = None
        if stat_config is not None and len(selected_conditions) >= 2:
            stat_summary = StatisticalTestingService.run_class_level_tests(
                df, experiment, selected_conditions, selected_classes,
                stat_config,
            )

        mode = 'log10 scale' if scale == 'log10' else 'linear scale'
        figure = BarChartPlotterService.create_bar_chart(
            bar_data, mode, stat_summary,
        )

        return BarChartResult(
            figure=figure,
            abundance_df=bar_data.abundance_df,
            stat_summary=stat_summary,
        )

    # ------------------------------------------------------------------
    # Pie Charts
    # ------------------------------------------------------------------

    @staticmethod
    def run_pie_charts(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> Dict[str, PieChartResult]:
        """Run abundance pie chart analysis for each condition.

        Args:
            df: DataFrame with ClassKey and concentration[s*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to create charts for.
            selected_classes: Lipid classes to include.

        Returns:
            Dict mapping condition name to PieChartResult.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")

        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment, selected_classes,
        )
        color_mapping = PieChartPlotterService.generate_color_mapping(
            selected_classes,
        )

        results = {}
        for condition in selected_conditions:
            if condition not in experiment.conditions_list:
                continue
            idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[idx]
            available_samples = [
                s for s in samples
                if f"concentration[{s}]" in pie_data.abundance_df.columns
            ]
            if not available_samples:
                continue

            figure, data_df = PieChartPlotterService.create_pie_chart(
                pie_data, condition, available_samples, color_mapping,
            )
            results[condition] = PieChartResult(
                figure=figure,
                data_df=data_df,
                condition=condition,
            )

        return results

    # ------------------------------------------------------------------
    # Saturation
    # ------------------------------------------------------------------

    @staticmethod
    def run_saturation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        stat_config: Optional[StatisticalTestConfig] = None,
        plot_type: str = 'concentration',
        show_significance: bool = False,
    ) -> SaturationResult:
        """Run SFA/MUFA/PUFA saturation analysis.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.
            stat_config: Statistical test config (None to skip tests).
            plot_type: 'concentration' or 'percentage'.
            show_significance: Whether to show significance annotations.

        Returns:
            SaturationResult with per-class plots, stats, and consolidated info.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")

        sat_data = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, experiment, selected_conditions, selected_classes,
        )

        consolidated = SaturationPlotterService.identify_consolidated_lipids(
            df, selected_classes,
        )

        stat_summary = None
        if (
            stat_config is not None
            and len(selected_conditions) >= 2
            and plot_type == 'concentration'
        ):
            stat_summary = StatisticalTestingService.run_saturation_tests(
                df, experiment, selected_conditions, selected_classes,
                sat_data.fa_data, stat_config,
            )

        plots = {}
        for lipid_class in sat_data.classes:
            if plot_type == 'percentage':
                fig = SaturationPlotterService.create_percentage_plot(
                    sat_data, lipid_class,
                )
            else:
                fig = SaturationPlotterService.create_concentration_plot(
                    sat_data, lipid_class, stat_summary, show_significance,
                )
            plots[lipid_class] = fig

        return SaturationResult(
            plots=plots,
            stat_summary=stat_summary,
            consolidated_lipids=consolidated,
        )

    # ------------------------------------------------------------------
    # FACH
    # ------------------------------------------------------------------

    @staticmethod
    def run_fach(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_class: str,
        selected_conditions: List[str],
    ) -> FACHResult:
        """Run Fatty Acid Composition Heatmap analysis.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
            experiment: Experiment configuration.
            selected_class: Single lipid class to analyze.
            selected_conditions: Conditions to include.

        Returns:
            FACHResult with heatmap figure, data, and weighted averages.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_class:
            raise ValueError("A lipid class must be selected")
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")

        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment, selected_class, selected_conditions,
        )

        figure = FACHPlotterService.create_fach_heatmap(fach_data)
        weighted_avgs = FACHPlotterService.get_weighted_averages(fach_data)

        return FACHResult(
            figure=figure,
            data_dict=fach_data.data_dict,
            unparsable_lipids=fach_data.unparsable_lipids,
            weighted_averages=weighted_avgs,
        )

    # ------------------------------------------------------------------
    # Pathway Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pathway_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
        saturation_source_df: Optional[pd.DataFrame] = None,
    ) -> PathwayDataResult:
        """Compute pathway fold-change and saturation data (no rendering).

        This method is intended for caching — the expensive computation
        is separated from the cheap figure rendering so that layout
        changes (toggling classes, editing edges) do not invalidate
        the cached data.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
                Used for fold-change calculation (always uses the full
                dataset so abundance is preserved).
            experiment: Experiment configuration.
            control: Name of the control condition.
            experimental: Name of the experimental condition.
            saturation_source_df: Optional DataFrame to use for saturation
                ratio calculation. When consolidated-format lipids have
                been excluded, pass the filtered DataFrame here so that
                saturation ratios are computed only from lipids with
                detailed chain information. If ``None``, ``df`` is used.

        Returns:
            PathwayDataResult with fold_change_df and saturation_df.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not control:
            raise ValueError("Control condition must be specified")
        if not experimental:
            raise ValueError("Experimental condition must be specified")
        if control == experimental:
            raise ValueError(
                "Control and experimental conditions must be different"
            )

        fold_change_df = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment, control, experimental,
        )
        saturation_df = PathwayVizPlotterService.calculate_class_saturation_ratio(
            saturation_source_df if saturation_source_df is not None else df,
        )

        return PathwayDataResult(
            fold_change_df=fold_change_df,
            saturation_df=saturation_df,
        )

    @staticmethod
    def run_pathway(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
    ) -> PathwayResult:
        """Run lipid pathway visualization.

        Computes per-class fold change and saturation ratios, then
        renders the pathway diagram.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
            experiment: Experiment configuration.
            control: Name of the control condition.
            experimental: Name of the experimental condition.

        Returns:
            PathwayResult with figure, pathway dict, and data.

        Raises:
            ValueError: If inputs are invalid.
        """
        data = AnalysisWorkflow.compute_pathway_data(
            df, experiment, control, experimental,
        )

        figure, pathway_dict = PathwayVizPlotterService.create_pathway_viz(
            data.fold_change_df, data.saturation_df,
        )

        return PathwayResult(
            figure=figure,
            pathway_dict=pathway_dict,
            fold_change_df=data.fold_change_df,
            saturation_df=data.saturation_df,
        )

    # ------------------------------------------------------------------
    # Volcano Plot
    # ------------------------------------------------------------------

    @staticmethod
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
        custom_label_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> VolcanoResult:
        """Run volcano plot analysis.

        Performs per-lipid statistical tests, builds volcano data, and
        creates volcano and concentration scatter plots.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
            experiment: Experiment configuration.
            control: Control condition name.
            experimental: Experimental condition name.
            selected_classes: Lipid classes to include.
            stat_config: Statistical test configuration.
            p_threshold: P-value significance threshold.
            fc_threshold: Fold change threshold (log2 scale).
            hide_non_sig: Whether to hide non-significant points.
            top_n_labels: Number of top significant lipids to label.
            custom_label_positions: Manual label position overrides.

        Returns:
            VolcanoResult with figures and data.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not control:
            raise ValueError("Control condition must be specified")
        if not experimental:
            raise ValueError("Experimental condition must be specified")
        if control == experimental:
            raise ValueError(
                "Control and experimental conditions must be different"
            )
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")

        # Filter to selected classes
        filtered_df = df[df['ClassKey'].isin(selected_classes)].copy()
        if filtered_df.empty:
            available_classes = sorted(df['ClassKey'].unique().tolist())
            raise ValueError(
                f"No data found for selected classes: {selected_classes}. "
                f"Available classes in data: {available_classes}"
            )

        control_samples = AnalysisWorkflow._get_samples_for_condition(
            experiment, control,
        )
        exp_samples = AnalysisWorkflow._get_samples_for_condition(
            experiment, experimental,
        )

        # Run per-lipid statistical tests
        stat_summary = StatisticalTestingService.run_species_level_tests(
            filtered_df, control_samples, exp_samples, stat_config,
        )

        # Build volcano data
        volcano_data = VolcanoPlotterService.prepare_volcano_data(
            filtered_df, stat_summary, control_samples, exp_samples,
        )

        # Generate color mapping
        classes_in_data = filtered_df['ClassKey'].unique().tolist()
        color_mapping = VolcanoPlotterService.generate_color_mapping(
            classes_in_data,
        )

        # Determine whether to use adjusted p-values
        use_adjusted = stat_config.correction_method != 'uncorrected'

        # Create volcano plot
        figure = VolcanoPlotterService.create_volcano_plot(
            volcano_data, color_mapping,
            p_threshold=p_threshold,
            fc_threshold=fc_threshold,
            hide_non_sig=hide_non_sig,
            use_adjusted_p=use_adjusted,
            top_n_labels=top_n_labels,
            custom_label_positions=custom_label_positions,
        )

        # Create concentration scatter
        conc_plot, conc_df = VolcanoPlotterService.create_concentration_vs_fc_plot(
            volcano_data, color_mapping,
            p_threshold=p_threshold,
            hide_non_sig=hide_non_sig,
            use_adjusted_p=use_adjusted,
        )

        return VolcanoResult(
            figure=figure,
            volcano_data=volcano_data,
            concentration_plot=conc_plot,
            concentration_df=conc_df,
            stat_summary=stat_summary,
        )

    # ------------------------------------------------------------------
    # Lipidomic Heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def run_heatmap(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
        heatmap_type: str = 'regular',
        n_clusters: int = 3,
    ) -> HeatmapResult:
        """Run lipidomic heatmap analysis.

        Filters data, computes Z-scores, and creates either a regular
        or clustered heatmap with optional cluster composition.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.
            heatmap_type: 'regular' or 'clustered'.
            n_clusters: Number of clusters (only for clustered type).

        Returns:
            HeatmapResult with figure, Z-scores, and optional cluster info.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")
        if heatmap_type not in ('regular', 'clustered'):
            raise ValueError(
                f"Invalid heatmap_type '{heatmap_type}'. "
                "Must be 'regular' or 'clustered'"
            )

        filtered_df, selected_samples = LipidomicHeatmapPlotterService.filter_data(
            df, selected_conditions, selected_classes, experiment,
        )

        z_scores_df = LipidomicHeatmapPlotterService.compute_z_scores(
            filtered_df,
        )

        cluster_composition = None
        if heatmap_type == 'clustered':
            figure = LipidomicHeatmapPlotterService.generate_clustered_heatmap(
                z_scores_df, selected_samples, n_clusters,
            )
            cluster_composition = LipidomicHeatmapPlotterService.get_cluster_composition(
                z_scores_df, n_clusters, mode='species_count',
                filtered_df=filtered_df,
            )
        else:
            figure = LipidomicHeatmapPlotterService.generate_regular_heatmap(
                z_scores_df, selected_samples,
            )

        return HeatmapResult(
            figure=figure,
            z_scores_df=z_scores_df,
            cluster_composition=cluster_composition,
        )
