"""
Lipidomic heatmap plotting service.

Filters lipidomic data by conditions and classes, computes row-wise Z-scores,
performs hierarchical clustering (Ward linkage, Euclidean distance), and
renders regular or clustered Plotly heatmaps.

Pure logic — no Streamlit dependencies.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import generate_condition_color_mapping
from app.services.statistical_testing import (
    MIN_POSITIVE_FALLBACK,
    ZERO_REPLACEMENT_DIVISOR,
)
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist


# ── Constants ──────────────────────────────────────────────────────────

HEATMAP_WIDTH = 900
HEATMAP_HEIGHT = 600
COLORSCALE = 'RdBu_r'
CLUSTER_LINE_STYLE = dict(color='black', width=2, dash='dash')

# The class-grouped and class-aggregated modes draw cells as true squares: the
# plot area is sized from the number of rows and columns rather than stretched
# to the container. The Clustered and Regular modes keep their original
# fixed-canvas layout.
CELL_SIZE_PX = 18

# Solid separators between condition columns and lipid class blocks.
BLOCK_LINE_STYLE = dict(color='black', width=2)

# Condition strip geometry, in px above the plot area. Drawn in pixels rather
# than as a paper fraction: as a fraction the strip thickens with the plot and
# pushes its own labels out of the top margin, which is how a tall heatmap —
# or any heatmap re-exported at the PDF report's larger canvas — ended up
# showing colours but no names.
STRIP_GAP_PX = 6
STRIP_HEIGHT_PX = 18

# A condition name that would run into its neighbour's moves up to the next
# row with room. The title sits at the vertical centre of the top margin, so
# each extra row costs twice its height in margin to stay clear of the title.
STRIP_LABEL_ROW_PX = 16
STRIP_LABEL_GAP_PX = 8
# Condition names are mostly capitals and digits, wider than tick labels.
STRIP_LABEL_PX_PER_CHAR = 8

# Clustered and Regular stretch to the container, so their column width is
# unknown when the figure is built. Name collisions are judged at this plot
# width, roughly what a 700px container leaves beside the species names and
# colour bar; wider renders only gain room. The square-celled class modes are
# judged at no more than this too: Streamlit narrows a figure wider than its
# container, squeezing the columns while the condition names keep their size.
STRETCHED_PLOT_WIDTH_PX = 450

# Margin budget (px). Left/bottom also grow with the longest tick label.
MARGIN_RIGHT = 130
MARGIN_TOP = 90
CLASS_LABEL_WIDTH = 95
PX_PER_CHAR = 7

# Sample labels longer than a standardized label (s1 ... s999) — original
# sample names such as 'CLEANUP_BLANK_01' — stand upright. At 45 degrees,
# adjacent long names in narrow columns run into each other and the last ones
# run off past the plot's right edge.
SHORT_SAMPLE_LABEL_CHARS = 4

# A class-aggregated heatmap can be only a handful of rows tall, where 18px
# cells would leave a sliver of a plot. Cells grow (staying square) until the
# plot area is reasonably tall, up to a ceiling so a 2-class map is not absurd.
MAX_CELL_SIZE_PX = 46
TARGET_PLOT_HEIGHT = 380

# One row per species at a fixed cell size means the figure grows without
# bound: a 3,500-species dataset would be ~64,000px tall. The class-grouped
# mode therefore shows one page of species at a time. Narrowing the class
# selection is not a workaround here — a single class can hold far more than
# this (TG alone has 1,903 species in the bundled LipidSearch dataset).
GROUPED_PAGE_SIZE = 150


@dataclass
class ClusteringResult:
    """Result of hierarchical clustering on Z-score data.

    Attributes:
        linkage_matrix: Scipy linkage matrix from Ward clustering.
        cluster_labels: 1-D array of cluster assignments (1-based).
        dendrogram_order: 1-D array of row indices ordered by dendrogram.
    """
    linkage_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    dendrogram_order: np.ndarray = field(default_factory=lambda: np.array([]))


class LipidomicHeatmapPlotterService:
    """Creates lipidomic heatmaps with optional hierarchical clustering."""

    @staticmethod
    def filter_data(
        df: pd.DataFrame,
        selected_conditions: List[str],
        selected_classes: List[str],
        experiment: ExperimentConfig,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Filter lipidomic data by conditions and lipid classes.

        Args:
            df: DataFrame with LipidMolec, ClassKey, and concentration columns.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.
            experiment: Experiment configuration.

        Returns:
            Tuple of (filtered DataFrame, list of selected sample names).

        Raises:
            ValueError: If inputs are invalid.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")
        if not selected_classes:
            raise ValueError("At least one lipid class must be selected")

        selected_samples = []
        for condition in selected_conditions:
            if condition not in experiment.conditions_list:
                continue
            cond_idx = experiment.conditions_list.index(condition)
            selected_samples.extend(experiment.individual_samples_list[cond_idx])

        if not selected_samples:
            raise ValueError("No valid samples found for selected conditions")

        abundance_cols = [f'concentration[{s}]' for s in selected_samples]

        # Every selected sample must have its column. Dropping the missing ones
        # and still returning the full sample list would leave the column
        # labels shifted against the data — a silently mislabelled heatmap
        # rather than an error.
        missing = [
            sample for sample, col in zip(selected_samples, abundance_cols)
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                "No concentration columns found for selected samples: "
                f"{', '.join(missing)}. The data and the experiment "
                "configuration are out of sync."
            )

        filtered_df = df[df['ClassKey'].isin(selected_classes)][
            ['LipidMolec', 'ClassKey'] + abundance_cols
        ].copy()

        return filtered_df, selected_samples

    @staticmethod
    def sample_condition_labels(
        selected_conditions: List[str],
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Build the condition label of each sample returned by filter_data.

        Index-aligned with ``filter_data``'s ``selected_samples``, so it can be
        used to colour the sample axis by condition.

        Args:
            selected_conditions: Conditions to include, same list passed to
                ``filter_data``.
            experiment: Experiment configuration.

        Returns:
            One condition label per selected sample, in sample order.
        """
        labels: List[str] = []
        for condition in selected_conditions:
            if condition not in experiment.conditions_list:
                continue
            cond_idx = experiment.conditions_list.index(condition)
            labels.extend(
                [condition] * len(experiment.individual_samples_list[cond_idx])
            )
        return labels

    @staticmethod
    def samples_for_condition(
        condition: Optional[str],
        experiment: ExperimentConfig,
    ) -> List[str]:
        """Return the sample names belonging to one condition.

        Args:
            condition: Condition label, or None.
            experiment: Experiment configuration.

        Returns:
            The condition's samples, or an empty list if it is unknown.
        """
        if not condition or condition not in experiment.conditions_list:
            return []
        cond_idx = experiment.conditions_list.index(condition)
        return list(experiment.individual_samples_list[cond_idx])

    @staticmethod
    def count_species(df: pd.DataFrame, selected_classes: List[str]) -> int:
        """Count the lipid species belonging to the selected classes.

        Lets a caller size the class-grouped mode's species pager without
        running the whole heatmap pipeline first.

        Args:
            df: DataFrame with a ClassKey column.
            selected_classes: Lipid classes to count.

        Returns:
            Number of matching species, or 0 if the frame has no ClassKey.
        """
        if df is None or df.empty or 'ClassKey' not in df.columns:
            return 0
        return int(df['ClassKey'].isin(selected_classes).sum())

    @staticmethod
    def page_bounds(total: int, page: int) -> Tuple[int, int]:
        """Resolve a species page to (start, end) row offsets.

        The page index is clamped into range, so a stale selection left over
        from a wider class selection cannot produce an empty heatmap.

        Args:
            total: Total number of species available.
            page: Zero-based page index.

        Returns:
            (start, end) offsets suitable for ``iloc`` slicing.
        """
        if total <= 0:
            return 0, 0
        last_page = max(0, (total - 1) // GROUPED_PAGE_SIZE)
        page = min(max(0, page), last_page)
        start = page * GROUPED_PAGE_SIZE
        return start, min(start + GROUPED_PAGE_SIZE, total)

    @staticmethod
    def order_by_class(
        z_scores_df: pd.DataFrame,
        sort_columns: Optional[List[str]] = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Reorder rows so each lipid class forms one contiguous block.

        Classes keep the order in which they first appear, so the block order
        follows the input data rather than being alphabetised. Within a block,
        species keep their input order unless ``sort_columns`` is given, in
        which case they are ranked by their mean value across those columns —
        the colour gradient then runs down each block in step with the colour
        bar instead of being scattered through it.

        Args:
            z_scores_df: Z-score DataFrame indexed by (LipidMolec, ClassKey).
            sort_columns: Columns to average when ranking species inside a
                class block. None (the default) keeps the input order.
            ascending: Rank from the most negative value down when True,
                from the most positive when False.

        Returns:
            The same DataFrame with rows grouped by class.

        Raises:
            ValueError: If the DataFrame is empty, or if none of
                ``sort_columns`` are present.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        classes = list(z_scores_df.index.get_level_values('ClassKey'))
        rank = {c: i for i, c in enumerate(dict.fromkeys(classes))}
        class_rank = np.array([rank[c] for c in classes])

        if not sort_columns:
            order = np.argsort(class_rank, kind='stable')
            return z_scores_df.iloc[order]

        present = [c for c in sort_columns if c in z_scores_df.columns]
        if not present:
            raise ValueError(
                "None of the requested sort columns are present: "
                f"{', '.join(sort_columns)}"
            )

        means = z_scores_df[present].mean(axis=1).to_numpy(dtype=float)
        key = means if ascending else -means
        # A species with no usable value in any sort column has no place in the
        # ranking; park it at the foot of its block rather than letting NaN
        # land it somewhere arbitrary.
        key = np.where(np.isnan(key), np.inf, key)

        # lexsort takes the primary key last, and is stable, so species that
        # tie on the mean keep their input order.
        order = np.lexsort((key, class_rank))
        return z_scores_df.iloc[order]

    @staticmethod
    def order_by_clusters(
        z_scores_df: pd.DataFrame,
        n_clusters: int,
    ) -> pd.DataFrame:
        """Reorder rows into the clustered heatmap's top-to-bottom order.

        Runs the same clustering as generate_clustered_heatmap, so the rows
        follow its dendrogram order.

        Args:
            z_scores_df: Per-species value DataFrame indexed by
                (LipidMolec, ClassKey).
            n_clusters: Number of clusters.

        Returns:
            The rows in dendrogram order, with a leading 'Cluster' column
            holding each species' 1-based cluster.

        Raises:
            ValueError: If inputs are invalid.
        """
        clustering = LipidomicHeatmapPlotterService.perform_clustering(
            z_scores_df, n_clusters,
        )
        order = clustering.dendrogram_order
        ordered_df = z_scores_df.iloc[order].copy()
        ordered_df.insert(0, 'Cluster', clustering.cluster_labels[order])
        return ordered_df

    @staticmethod
    def compute_log2fc(
        filtered_df: pd.DataFrame,
        control_samples: List[str],
    ) -> pd.DataFrame:
        """Compute per-sample log2 fold change against the control mean.

        Each species' values are expressed relative to the mean of that
        species across the control samples, so control columns sit near zero
        and a colour scale centred on zero reads directly as up or down versus
        control. Zeros are floored using the same adjustment as the statistical
        tests before the ratio is taken.

        Args:
            filtered_df: DataFrame with LipidMolec, ClassKey, and
                concentration columns (output of filter_data).
            control_samples: Sample names forming the control condition.

        Returns:
            DataFrame indexed by (LipidMolec, ClassKey) holding log2 fold
            changes, one column per sample.

        Raises:
            ValueError: If the frame is empty, has no concentration columns,
                or none of the control samples are present.
        """
        if filtered_df is None or filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty")

        working_df = filtered_df.set_index(['LipidMolec', 'ClassKey'])
        return _log2fc_frame(working_df, control_samples)

    @staticmethod
    def compute_class_log2fc(
        filtered_df: pd.DataFrame,
        control_samples: List[str],
    ) -> pd.DataFrame:
        """Aggregate species to class level, then log2 fold change vs control.

        Concentrations are summed within each class per sample — the same
        aggregation as ``compute_class_z_scores`` — and each class total is
        then expressed relative to that class's control mean.

        Args:
            filtered_df: DataFrame with LipidMolec, ClassKey, and
                concentration columns (output of filter_data).
            control_samples: Sample names forming the control condition.

        Returns:
            DataFrame indexed by ClassKey holding log2 fold changes.

        Raises:
            ValueError: If the frame is empty, has no concentration columns,
                or none of the control samples are present.
        """
        if filtered_df is None or filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty")

        abundance_cols = [
            c for c in filtered_df.columns if c.startswith('concentration[')
        ]
        if not abundance_cols:
            raise ValueError("No concentration columns found")

        class_totals = filtered_df.groupby('ClassKey')[abundance_cols].sum()
        return _log2fc_frame(class_totals, control_samples)

    @staticmethod
    def compute_class_z_scores(filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate species to class level and Z-score each class row.

        Concentrations are summed within each lipid class per sample — the same
        class-level aggregation the abundance bar chart, pie chart and pathway
        visualisations use — and each class row is then standardised across
        samples with the same Z-score definition as ``compute_z_scores``.

        Args:
            filtered_df: DataFrame with LipidMolec, ClassKey, and
                concentration columns (output of filter_data).

        Returns:
            DataFrame indexed by ClassKey, one row per class, holding Z-scores.

        Raises:
            ValueError: If the DataFrame is empty or has no concentration columns.
        """
        if filtered_df is None or filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty")

        abundance_cols = [
            c for c in filtered_df.columns if c.startswith('concentration[')
        ]
        if not abundance_cols:
            raise ValueError("No concentration columns found")

        class_totals = filtered_df.groupby('ClassKey')[abundance_cols].sum()

        return class_totals.apply(
            lambda x: (x - x.mean(skipna=True)) / x.std(skipna=True), axis=1,
        )

    @staticmethod
    def compute_z_scores(filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Compute row-wise Z-scores for lipid abundances.

        Each lipid's concentrations are standardized across samples:
        z = (x - mean) / std.

        Args:
            filtered_df: DataFrame with LipidMolec, ClassKey, and
                concentration columns (output of filter_data).

        Returns:
            DataFrame indexed by (LipidMolec, ClassKey) with Z-score values.

        Raises:
            ValueError: If DataFrame is empty or has no concentration columns.
        """
        if filtered_df is None or filtered_df.empty:
            raise ValueError("Filtered DataFrame is empty")

        working_df = filtered_df.copy()
        working_df = working_df.set_index(['LipidMolec', 'ClassKey'])
        abundance_cols = working_df.columns

        if len(abundance_cols) == 0:
            raise ValueError("No concentration columns found")

        z_scores_df = working_df[abundance_cols].apply(
            lambda x: (x - x.mean(skipna=True)) / x.std(skipna=True), axis=1,
        )
        return z_scores_df

    @staticmethod
    def perform_clustering(
        z_scores_df: pd.DataFrame,
        n_clusters: int,
    ) -> ClusteringResult:
        """Perform hierarchical clustering on Z-score data.

        Uses Ward linkage with Euclidean distance.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            n_clusters: Number of clusters to form.

        Returns:
            ClusteringResult with linkage matrix, labels, and dendrogram order.

        Raises:
            ValueError: If inputs are invalid.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")
        if n_clusters < 1:
            raise ValueError("Number of clusters must be at least 1")
        if n_clusters > len(z_scores_df):
            raise ValueError(
                f"Number of clusters ({n_clusters}) cannot exceed "
                f"number of lipids ({len(z_scores_df)})"
            )

        # Replace NaN with 0 for distance computation
        clean_df = z_scores_df.fillna(0)

        linkage_matrix = linkage(pdist(clean_df, 'euclidean'), method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        dendrogram_order = leaves_list(linkage_matrix)

        return ClusteringResult(
            linkage_matrix=linkage_matrix,
            cluster_labels=cluster_labels,
            dendrogram_order=dendrogram_order,
        )

    @staticmethod
    def generate_clustered_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        n_clusters: int,
        sample_conditions: Optional[List[str]] = None,
        value_label: str = 'Z-score',
    ) -> go.Figure:
        """Create a heatmap reordered by hierarchical clustering with cluster boundaries.

        Args:
            z_scores_df: Per-species value DataFrame, either Z-scores
                (compute_z_scores) or log2 fold changes (compute_log2fc).
                Clustering runs on whichever is given.
            selected_samples: Sample names for column labels.
            n_clusters: Number of clusters.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.
            value_label: Name of the plotted quantity, used for the colour bar.

        Returns:
            Plotly Figure with clustered heatmap and dashed cluster boundary lines.

        Raises:
            ValueError: If inputs are invalid.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        clustering = LipidomicHeatmapPlotterService.perform_clustering(
            z_scores_df, n_clusters,
        )

        clustered_df = z_scores_df.iloc[clustering.dendrogram_order].copy()
        clustered_df['Cluster'] = clustering.cluster_labels[clustering.dendrogram_order]

        z_scores_array = clustered_df.drop('Cluster', axis=1).to_numpy()

        if z_scores_array.ndim == 1:
            z_scores_array = z_scores_array.reshape(-1, 1)

        # Symmetric color scale
        vmin = np.nanmin(z_scores_array)
        vmax = np.nanmax(z_scores_array)
        abs_max = max(abs(vmin), abs(vmax))

        fig = go.Figure(data=go.Heatmap(
            z=z_scores_array,
            x=selected_samples,
            y=clustered_df.index.get_level_values('LipidMolec'),
            colorscale=COLORSCALE,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title=value_label),
        ))

        # Add cluster boundary lines
        cluster_sizes = clustered_df['Cluster'].value_counts().sort_index()
        cumulative_sizes = np.cumsum(cluster_sizes.values[:-1])

        for size in cumulative_sizes:
            fig.add_shape(
                type='line',
                x0=-0.5,
                y0=size - 0.5,
                x1=len(selected_samples) - 0.5,
                y1=size - 0.5,
                line=CLUSTER_LINE_STYLE,
            )

        extra_top = _add_condition_strip(
            fig, sample_conditions,
            STRETCHED_PLOT_WIDTH_PX / max(1, len(selected_samples)),
            label_color=None,
        )

        sample_axis, bottom = _sample_axis_layout(selected_samples, bottom=50)
        fig.update_layout(
            title=_titled('Clustered Lipidomic Heatmap', value_label),
            xaxis_title='Samples',
            yaxis_title='Lipid Molecules',
            margin=dict(l=100, r=100, t=MARGIN_TOP + extra_top, b=bottom),
            width=HEATMAP_WIDTH,
            height=HEATMAP_HEIGHT,
        )

        fig.update_xaxes(**sample_axis)
        fig.update_yaxes(tickmode='array', autorange='reversed')

        return fig

    @staticmethod
    def generate_regular_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        sample_conditions: Optional[List[str]] = None,
        value_label: str = 'Z-score',
    ) -> go.Figure:
        """Create a regular heatmap without clustering.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            selected_samples: Sample names for column labels.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.

        Returns:
            Plotly Figure with regular heatmap.

        Raises:
            ValueError: If inputs are invalid.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        z_scores_array = z_scores_df.to_numpy()

        # Symmetric color scale
        vmin = np.nanmin(z_scores_array)
        vmax = np.nanmax(z_scores_array)
        abs_max = max(abs(vmin), abs(vmax))

        fig = go.Figure(data=go.Heatmap(
            z=z_scores_array,
            x=selected_samples,
            y=z_scores_df.index.get_level_values('LipidMolec'),
            colorscale=COLORSCALE,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title=value_label),
        ))

        extra_top = _add_condition_strip(
            fig, sample_conditions,
            STRETCHED_PLOT_WIDTH_PX / max(1, len(selected_samples)),
            label_color=None,
        )

        sample_axis, bottom = _sample_axis_layout(selected_samples, bottom=20)
        fig.update_layout(
            title=_titled('Regular Lipidomic Heatmap', value_label),
            xaxis_title='Samples',
            yaxis_title='Lipid Molecules',
            margin=dict(l=10, r=10, t=MARGIN_TOP + extra_top, b=bottom),
            height=HEATMAP_HEIGHT,
        )

        fig.update_xaxes(**sample_axis)
        fig.update_yaxes(tickmode='array')

        return fig

    @staticmethod
    def generate_class_grouped_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        sample_conditions: Optional[List[str]] = None,
        value_label: str = 'Z-score',
    ) -> go.Figure:
        """Create a heatmap with species grouped into lipid class blocks.

        Rows are reordered so each lipid class is contiguous, and the class name
        is drawn as a group label to the left of the species names with a
        divider between blocks.

        Args:
            z_scores_df: Per-species value DataFrame, either Z-scores
                (compute_z_scores) or log2 fold changes (compute_log2fc).
            selected_samples: Sample names for column labels.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.
            value_label: Name of the plotted quantity, used for the colour bar
                and the figure title.

        Returns:
            Plotly Figure with class-grouped heatmap.

        Raises:
            ValueError: If inputs are invalid.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        ordered_df = LipidomicHeatmapPlotterService.order_by_class(z_scores_df)

        species = list(ordered_df.index.get_level_values('LipidMolec'))
        classes = list(ordered_df.index.get_level_values('ClassKey'))

        # A two-level y axis renders the class as a group label to the left of
        # the species names, with dividers between blocks.
        fig = _build_heatmap_figure(
            ordered_df.to_numpy(), selected_samples, [classes, species],
            value_label=value_label,
        )

        extra_top = _add_condition_strip(
            fig, sample_conditions,
            _square_column_px(len(species), len(selected_samples)),
        )
        _apply_square_layout(
            fig, f'Lipidomic Heatmap Grouped by Class ({value_label})',
            n_rows=len(species), n_cols=len(selected_samples),
            y_labels=species, x_labels=selected_samples,
            grouped=True, extra_top=extra_top,
        )

        fig.update_yaxes(
            autorange='reversed',
            showdividers=True,
            dividercolor=BLOCK_LINE_STYLE['color'],
            dividerwidth=BLOCK_LINE_STYLE['width'],
        )

        return fig

    @staticmethod
    def generate_class_aggregated_heatmap(
        class_z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        sample_conditions: Optional[List[str]] = None,
        value_label: str = 'Z-score',
    ) -> go.Figure:
        """Create a heatmap with one row per lipid class.

        Args:
            class_z_scores_df: Class-level values indexed by ClassKey, either
                Z-scores (compute_class_z_scores) or log2 fold changes
                (compute_class_log2fc).
            selected_samples: Sample names for column labels.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.
            value_label: Name of the plotted quantity, used for the colour bar
                and the figure title.

        Returns:
            Plotly Figure with one row per lipid class.

        Raises:
            ValueError: If inputs are invalid.
        """
        if class_z_scores_df is None or class_z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        classes = list(class_z_scores_df.index)
        fig = _build_heatmap_figure(
            class_z_scores_df.to_numpy(), selected_samples, classes,
            value_label=value_label,
        )

        extra_top = _add_condition_strip(
            fig, sample_conditions,
            _square_column_px(len(classes), len(selected_samples)),
        )
        _apply_square_layout(
            fig, f'Lipidomic Heatmap Aggregated by Class ({value_label})',
            n_rows=len(classes), n_cols=len(selected_samples),
            y_labels=classes, x_labels=selected_samples,
            y_title='Lipid Classes', extra_top=extra_top,
        )

        fig.update_yaxes(tickmode='array', autorange='reversed')

        return fig

    @staticmethod
    def get_cluster_composition(
        z_scores_df: pd.DataFrame,
        n_clusters: int,
        mode: str = 'species_count',
        filtered_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Get lipid class composition per cluster.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            n_clusters: Number of clusters.
            mode: 'species_count' for species percentage, or
                'concentration' for concentration-based percentage.
            filtered_df: Original filtered DataFrame with concentration values.
                Required when mode='concentration'.

        Returns:
            DataFrame with clusters as rows and lipid classes as columns,
            values are percentages.

        Raises:
            ValueError: If inputs are invalid or mode is unrecognized.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")
        if mode not in ('species_count', 'concentration'):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'species_count' or 'concentration'"
            )
        if mode == 'concentration' and (filtered_df is None or filtered_df.empty):
            raise ValueError(
                "filtered_df is required when mode='concentration'"
            )

        clustering = LipidomicHeatmapPlotterService.perform_clustering(
            z_scores_df, n_clusters,
        )

        if mode == 'species_count':
            return _compute_species_percentages(z_scores_df, clustering.cluster_labels)
        else:
            return _compute_concentration_percentages(
                z_scores_df, filtered_df, clustering.cluster_labels,
            )


# ── Private helpers ────────────────────────────────────────────────────


def _titled(base: str, value_label: str) -> str:
    """Figure title, naming the quantity only when it is not the default.

    Keeps the Clustered and Regular titles byte-identical under Z-scores.
    """
    return base if value_label == 'Z-score' else f'{base} ({value_label})'


def _log2fc_frame(
    values_df: pd.DataFrame,
    control_samples: List[str],
) -> pd.DataFrame:
    """Express every value as log2(value / control mean), row by row.

    Zeros are floored at the smallest positive value in the row divided by
    ZERO_REPLACEMENT_DIVISOR, matching how the statistical tests handle zeros
    before taking a ratio, so a single zero cannot send a row to -infinity.
    """
    abundance_cols = [
        c for c in values_df.columns if c.startswith('concentration[')
    ]
    if not abundance_cols:
        raise ValueError("No concentration columns found")

    control_cols = [
        f'concentration[{s}]' for s in control_samples
        if f'concentration[{s}]' in abundance_cols
    ]
    if not control_cols:
        raise ValueError(
            "No concentration columns found for the control condition"
        )

    values = values_df[abundance_cols].to_numpy(dtype=float)

    # Row-wise floor for zeros and negatives, as in the statistical tests.
    with np.errstate(invalid='ignore'):
        positive = np.where(values > 0, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        min_positive = np.nanmin(positive, axis=1)
    min_positive = np.where(
        np.isnan(min_positive), MIN_POSITIVE_FALLBACK, min_positive,
    )
    small = (min_positive / ZERO_REPLACEMENT_DIVISOR).reshape(-1, 1)
    adjusted = np.maximum(values, small)

    control_idx = [abundance_cols.index(c) for c in control_cols]
    control_mean = np.nanmean(adjusted[:, control_idx], axis=1).reshape(-1, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        log2fc = np.log2(adjusted / control_mean)
    log2fc = np.where(np.isfinite(log2fc), log2fc, np.nan)

    return pd.DataFrame(log2fc, index=values_df.index, columns=abundance_cols)


def _build_heatmap_figure(
    z_array: np.ndarray,
    x_labels: List[str],
    y_labels,
    value_label: str = 'Z-score',
) -> go.Figure:
    """Create the base heatmap trace with a symmetric diverging colour scale."""
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)

    abs_max = max(abs(np.nanmin(z_array)), abs(np.nanmax(z_array)))

    return go.Figure(data=go.Heatmap(
        z=z_array,
        x=x_labels,
        y=y_labels,
        colorscale=COLORSCALE,
        zmin=-abs_max,
        zmax=abs_max,
        colorbar=dict(title=value_label),
        xgap=1,
        ygap=1,
    ))


def _condition_blocks(
    sample_conditions: List[str],
) -> List[Tuple[str, int, int]]:
    """Group consecutive samples of the same condition into (name, start, end)."""
    blocks: List[Tuple[str, int, int]] = []
    if not sample_conditions:
        return blocks

    start = 0
    for i in range(1, len(sample_conditions) + 1):
        if (
            i == len(sample_conditions)
            or sample_conditions[i] != sample_conditions[start]
        ):
            blocks.append((sample_conditions[start], start, i - 1))
            start = i
    return blocks


def _add_condition_strip(
    fig: go.Figure,
    sample_conditions: Optional[List[str]],
    column_px: float,
    label_color: Optional[str] = 'black',
) -> int:
    """Draw a colour-coded condition strip above the columns.

    Adds one filled block per condition, the condition name above each block,
    and a solid separator between adjacent conditions. Each block is labelled
    directly rather than through a legend, which keeps the figure readable at
    any row count. Does nothing when no conditions are supplied.

    The strip is anchored to the top of the plot area and sized in pixels, so
    it keeps its thickness however tall the figure is drawn.

    Args:
        fig: Figure to annotate.
        sample_conditions: Condition label per sample, or None for no strip.
        column_px: Width of one sample column in px, used to tell whether
            neighbouring condition names would collide.
        label_color: Colour of the condition names. None leaves it to the
            figure's font, so the names follow the Streamlit theme like the
            title does on a figure without a fixed white background.

    Returns:
        Extra top margin, in px, the caller must add so any rows of
        condition names above the first clear the title; 0 when one row
        suffices.
    """
    blocks = _condition_blocks(sample_conditions or [])
    if not blocks:
        return 0

    strip_top = STRIP_GAP_PX + STRIP_HEIGHT_PX
    rows = _label_rows(blocks, column_px)

    color_map = generate_condition_color_mapping(
        list(dict.fromkeys(cond for cond, _, _ in blocks))
    )

    for (condition, start, end), row in zip(blocks, rows):
        fig.add_shape(
            type='rect',
            xref='x', yref='paper',
            ysizemode='pixel', yanchor=1,
            x0=start - 0.5, x1=end + 0.5,
            y0=STRIP_GAP_PX, y1=strip_top,
            fillcolor=color_map[condition],
            line=dict(width=0),
            layer='above',
        )
        fig.add_annotation(
            xref='x', yref='paper',
            x=(start + end) / 2, y=1,
            yshift=strip_top + row * STRIP_LABEL_ROW_PX,
            text=condition,
            showarrow=False, yanchor='bottom',
            font=dict(size=12, color=label_color),
        )

    # Separator between adjacent condition blocks
    for _, _, end in blocks[:-1]:
        fig.add_shape(
            type='line',
            xref='x', yref='paper',
            x0=end + 0.5, x1=end + 0.5,
            y0=0, y1=1,
            line=BLOCK_LINE_STYLE,
        )

    # Autorange would otherwise widen the axis to fit a name that overhangs
    # the outermost block, leaving an empty sliver beside the columns.
    fig.update_xaxes(range=[-0.5, len(sample_conditions) - 0.5])

    return 2 * STRIP_LABEL_ROW_PX * max(rows)


def _square_column_px(n_rows: int, n_cols: int) -> float:
    """Column width to judge condition-name collisions at in a class mode.

    The drawn cell size, unless the figure is wide enough that Streamlit may
    narrow it to fit the page; then the width the stretched modes assume.
    """
    return min(cell_size(n_rows), STRETCHED_PLOT_WIDTH_PX / max(1, n_cols))


def _label_rows(
    blocks: List[Tuple[str, int, int]],
    column_px: float,
) -> List[int]:
    """Put each condition name on the lowest row where it clears the name
    before it on that row, opening a new row above when none has room."""
    rows: List[int] = []
    right_edge: List[float] = []
    for condition, start, end in blocks:
        centre = (start + end + 1) / 2 * column_px
        half_width = len(str(condition)) * STRIP_LABEL_PX_PER_CHAR / 2
        row = next(
            (r for r, edge in enumerate(right_edge)
             if centre - half_width >= edge + STRIP_LABEL_GAP_PX),
            len(right_edge),
        )
        if row == len(right_edge):
            right_edge.append(centre + half_width)
        else:
            right_edge[row] = centre + half_width
        rows.append(row)
    return rows


def _apply_square_layout(
    fig: go.Figure,
    title: str,
    n_rows: int,
    n_cols: int,
    y_labels: List[str],
    x_labels: List[str],
    grouped: bool = False,
    y_title: str = 'Lipid Molecules',
    extra_top: int = 0,
) -> None:
    """Size the figure so every cell renders as a square.

    The plot area is fixed at n_cols x n_rows cells and the margins are sized
    from the longest tick label, so the caller must render the figure at its
    natural size rather than stretching it to the container width.
    ``extra_top`` adds to the top margin, for extra rows of condition names.
    """
    cell = cell_size(n_rows)
    left = _label_extent(y_labels) + (CLASS_LABEL_WIDTH if grouped else 0)
    bottom = _label_extent(x_labels)
    top = MARGIN_TOP + extra_top

    fig.update_layout(
        title=title,
        xaxis_title='Samples',
        yaxis_title=y_title,
        margin=dict(l=left, r=MARGIN_RIGHT, t=top, b=bottom),
        width=left + MARGIN_RIGHT + n_cols * cell,
        height=top + bottom + n_rows * cell,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    sample_axis, _ = _sample_axis_layout(x_labels, bottom=bottom)
    fig.update_xaxes(**sample_axis, tickfont=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'))


def cell_size(n_rows: int) -> int:
    """Square cell edge, in px, for a heatmap with n_rows rows.

    CELL_SIZE_PX for anything species-sized; larger for the handful of rows a
    class-aggregated heatmap has, so the plot does not collapse to a sliver.
    """
    if n_rows <= 0:
        return CELL_SIZE_PX
    return int(min(MAX_CELL_SIZE_PX, max(CELL_SIZE_PX, TARGET_PLOT_HEIGHT / n_rows)))


def _label_extent(labels: List[str]) -> int:
    """Approximate the margin, in px, needed to fit the longest tick label."""
    longest = max((len(str(label)) for label in labels), default=0)
    return 45 + longest * PX_PER_CHAR


def _sample_axis_layout(x_labels: List[str], bottom: int) -> Tuple[dict, int]:
    """Sample-axis settings and bottom margin, in px, for the x labels.

    Labels as short as a standardized label keep the 45 degree ticks and the
    caller's ``bottom``. Longer ones (original sample names) stand upright,
    with the bottom margin grown to fit the longest, since a fixed margin
    would clip them wherever Plotly does not grow it itself (the PDF report's
    export). Automargin is switched on for them too, which is what drops the
    axis title below the names instead of across them.
    """
    longest = max((len(str(label)) for label in x_labels), default=0)
    if longest <= SHORT_SAMPLE_LABEL_CHARS:
        return dict(tickangle=45), bottom
    return (
        dict(tickangle=90, automargin=True),
        max(bottom, _label_extent(x_labels)),
    )


def _compute_species_percentages(
    z_scores_df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute species count percentages per cluster."""
    clustered_df = z_scores_df.copy()
    clustered_df['Cluster'] = cluster_labels

    records = []
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = clustered_df['Cluster'] == cluster_id
        class_values = clustered_df[cluster_mask].index.get_level_values('ClassKey')
        counts = class_values.value_counts(normalize=True) * 100
        row = counts.to_dict()
        row['Cluster'] = cluster_id
        records.append(row)

    result = pd.DataFrame(records).set_index('Cluster').fillna(0)
    return result


def _compute_concentration_percentages(
    z_scores_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Compute concentration-based percentages per cluster."""
    conc_cols = [col for col in filtered_df.columns if col.startswith('concentration[')]

    clustered_conc_df = filtered_df.set_index(['LipidMolec', 'ClassKey']).copy()
    clustered_conc_df = clustered_conc_df.loc[z_scores_df.index]
    clustered_conc_df['Cluster'] = cluster_labels

    clustered_conc_df['TotalConc'] = clustered_conc_df[conc_cols].sum(axis=1)

    clustered_conc_df = clustered_conc_df.reset_index()

    cluster_class_conc = clustered_conc_df.groupby(
        ['Cluster', 'ClassKey'],
    )['TotalConc'].sum()

    conc_percentages = cluster_class_conc.groupby('Cluster', group_keys=False).apply(
        lambda x: (x / x.sum()) * 100,
    ).unstack(fill_value=0)

    return conc_percentages
