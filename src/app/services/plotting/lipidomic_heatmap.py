"""
Lipidomic heatmap plotting service.

Filters lipidomic data by conditions and classes, computes row-wise Z-scores,
performs hierarchical clustering (Ward linkage, Euclidean distance), and
renders regular or clustered Plotly heatmaps.

Pure logic — no Streamlit dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import generate_condition_color_mapping
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

# Condition strip geometry, in px above the plot area. Held in pixels and
# converted to paper coordinates per figure: as a fixed paper fraction the
# strip would thicken with the plot and push its own labels out of the top
# margin, which is how a tall heatmap ended up showing colours but no names.
STRIP_GAP_PX = 6
STRIP_HEIGHT_PX = 18

# Margin budget (px). Left/bottom also grow with the longest tick label.
MARGIN_RIGHT = 130
MARGIN_TOP = 90
CLASS_LABEL_WIDTH = 95
PX_PER_CHAR = 7

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
        available_cols = [c for c in abundance_cols if c in df.columns]

        if not available_cols:
            raise ValueError("No concentration columns found for selected samples")

        filtered_df = df[df['ClassKey'].isin(selected_classes)][
            ['LipidMolec', 'ClassKey'] + available_cols
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
    def order_by_class(z_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Reorder rows so each lipid class forms one contiguous block.

        Classes keep the order in which they first appear, so the block order
        follows the input data rather than being alphabetised.

        Args:
            z_scores_df: Z-score DataFrame indexed by (LipidMolec, ClassKey).

        Returns:
            The same DataFrame with rows grouped by class.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if z_scores_df is None or z_scores_df.empty:
            raise ValueError("Z-scores DataFrame is empty")

        classes = list(z_scores_df.index.get_level_values('ClassKey'))
        rank = {c: i for i, c in enumerate(dict.fromkeys(classes))}
        order = np.argsort([rank[c] for c in classes], kind='stable')
        return z_scores_df.iloc[order]

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
    ) -> go.Figure:
        """Create a heatmap reordered by hierarchical clustering with cluster boundaries.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            selected_samples: Sample names for column labels.
            n_clusters: Number of clusters.

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
            colorbar=dict(title='Z-score'),
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

        fig.update_layout(
            title='Clustered Lipidomic Heatmap',
            xaxis_title='Samples',
            yaxis_title='Lipid Molecules',
            margin=dict(l=100, r=100, t=50, b=50),
            width=HEATMAP_WIDTH,
            height=HEATMAP_HEIGHT,
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickmode='array', autorange='reversed')

        return fig

    @staticmethod
    def generate_regular_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
    ) -> go.Figure:
        """Create a regular heatmap without clustering.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            selected_samples: Sample names for column labels.

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
            colorbar=dict(title='Z-score'),
        ))

        fig.update_layout(
            title='Regular Lipidomic Heatmap',
            xaxis_title='Samples',
            yaxis_title='Lipid Molecules',
            margin=dict(l=10, r=10, t=25, b=20),
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickmode='array')

        return fig

    @staticmethod
    def generate_class_grouped_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        sample_conditions: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create a heatmap with species grouped into lipid class blocks.

        Rows are reordered so each lipid class is contiguous, and the class name
        is drawn as a group label to the left of the species names with a
        divider between blocks.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            selected_samples: Sample names for column labels.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.

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
        )

        _add_condition_strip(fig, sample_conditions, len(species))
        _apply_square_layout(
            fig, 'Lipidomic Heatmap Grouped by Class',
            n_rows=len(species), n_cols=len(selected_samples),
            y_labels=species, x_labels=selected_samples,
            grouped=True,
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
    ) -> go.Figure:
        """Create a heatmap with one row per lipid class.

        Args:
            class_z_scores_df: Class-level Z-scores indexed by ClassKey
                (output of compute_class_z_scores).
            selected_samples: Sample names for column labels.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.

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
        )

        _add_condition_strip(fig, sample_conditions, len(classes))
        _apply_square_layout(
            fig, 'Lipidomic Heatmap Aggregated by Class',
            n_rows=len(classes), n_cols=len(selected_samples),
            y_labels=classes, x_labels=selected_samples,
            y_title='Lipid Classes',
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


def _build_heatmap_figure(
    z_array: np.ndarray,
    x_labels: List[str],
    y_labels,
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
        colorbar=dict(title='Z-score'),
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
    n_rows: int,
) -> None:
    """Draw a colour-coded condition strip above the columns.

    Adds one filled block per condition, the condition name above each block,
    and a solid separator between adjacent conditions. Each block is labelled
    directly rather than through a legend, which keeps the figure readable at
    any row count. Does nothing when no conditions are supplied.

    Args:
        fig: Figure to annotate.
        sample_conditions: Condition label per sample, or None for no strip.
        n_rows: Heatmap row count, used to convert the strip's pixel geometry
            into paper coordinates so it keeps a constant thickness.
    """
    blocks = _condition_blocks(sample_conditions or [])
    if not blocks:
        return

    plot_height = max(1, n_rows * cell_size(n_rows))
    strip_y0 = 1 + STRIP_GAP_PX / plot_height
    strip_y1 = strip_y0 + STRIP_HEIGHT_PX / plot_height

    color_map = generate_condition_color_mapping(
        list(dict.fromkeys(cond for cond, _, _ in blocks))
    )

    for condition, start, end in blocks:
        fig.add_shape(
            type='rect',
            xref='x', yref='paper',
            x0=start - 0.5, x1=end + 0.5,
            y0=strip_y0, y1=strip_y1,
            fillcolor=color_map[condition],
            line=dict(width=0),
            layer='above',
        )
        fig.add_annotation(
            xref='x', yref='paper',
            x=(start + end) / 2, y=strip_y1,
            text=condition,
            showarrow=False, yanchor='bottom',
            font=dict(size=12, color='black'),
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


def _apply_square_layout(
    fig: go.Figure,
    title: str,
    n_rows: int,
    n_cols: int,
    y_labels: List[str],
    x_labels: List[str],
    grouped: bool = False,
    y_title: str = 'Lipid Molecules',
) -> None:
    """Size the figure so every cell renders as a square.

    The plot area is fixed at n_cols x n_rows cells and the margins are sized
    from the longest tick label, so the caller must render the figure at its
    natural size rather than stretching it to the container width.
    """
    cell = cell_size(n_rows)
    left = _label_extent(y_labels) + (CLASS_LABEL_WIDTH if grouped else 0)
    bottom = _label_extent(x_labels)

    fig.update_layout(
        title=title,
        xaxis_title='Samples',
        yaxis_title=y_title,
        margin=dict(l=left, r=MARGIN_RIGHT, t=MARGIN_TOP, b=bottom),
        width=left + MARGIN_RIGHT + n_cols * cell,
        height=MARGIN_TOP + bottom + n_rows * cell,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(color='black'))
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
