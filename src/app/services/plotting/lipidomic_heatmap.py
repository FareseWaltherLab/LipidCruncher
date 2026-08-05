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

COLORSCALE = 'RdBu_r'
CLUSTER_LINE_STYLE = dict(color='black', width=2, dash='dash')

# Cells are drawn as true squares: the plot area is sized from the number of
# rows and columns rather than stretched to the container.
CELL_SIZE_PX = 18

# Solid separators between condition columns and lipid class blocks.
BLOCK_LINE_STYLE = dict(color='black', width=2)

# Condition strip position, in paper coordinates above the plot area.
STRIP_Y0 = 1.012
STRIP_Y1 = 1.05

# Margin budget (px). Left/bottom also grow with the longest tick label.
MARGIN_RIGHT = 130
MARGIN_TOP = 90
CLASS_LABEL_WIDTH = 95
PX_PER_CHAR = 7


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
    ) -> go.Figure:
        """Create a heatmap reordered by hierarchical clustering with cluster boundaries.

        Args:
            z_scores_df: Z-score DataFrame (output of compute_z_scores).
            selected_samples: Sample names for column labels.
            n_clusters: Number of clusters.
            sample_conditions: Optional condition label per sample, index-aligned
                with selected_samples. When given, a colour-coded condition strip
                is drawn above the columns.

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

        species = list(clustered_df.index.get_level_values('LipidMolec'))
        fig = _build_heatmap_figure(z_scores_array, selected_samples, species)

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

        _add_condition_strip(fig, sample_conditions)
        _apply_square_layout(
            fig, 'Clustered Lipidomic Heatmap',
            n_rows=len(species), n_cols=len(selected_samples),
            y_labels=species, x_labels=selected_samples,
        )

        fig.update_yaxes(tickmode='array', autorange='reversed')

        return fig

    @staticmethod
    def generate_regular_heatmap(
        z_scores_df: pd.DataFrame,
        selected_samples: List[str],
        sample_conditions: Optional[List[str]] = None,
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

        species = list(z_scores_df.index.get_level_values('LipidMolec'))
        fig = _build_heatmap_figure(
            z_scores_df.to_numpy(), selected_samples, species,
        )

        _add_condition_strip(fig, sample_conditions)
        _apply_square_layout(
            fig, 'Regular Lipidomic Heatmap',
            n_rows=len(species), n_cols=len(selected_samples),
            y_labels=species, x_labels=selected_samples,
        )

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

        _add_condition_strip(fig, sample_conditions)
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
        # Anchored to the bottom of the right margin so the condition
        # legend can sit above it without overlapping.
        colorbar=dict(
            title='Z-score', len=0.6,
            x=1.02, xanchor='left', y=0, yanchor='bottom',
        ),
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
) -> None:
    """Draw a colour-coded condition strip above the columns.

    Adds one filled block per condition, a label above each block, a legend
    entry per condition, and a solid separator between adjacent conditions.
    Does nothing when no conditions are supplied.
    """
    blocks = _condition_blocks(sample_conditions or [])
    if not blocks:
        return

    color_map = generate_condition_color_mapping(
        list(dict.fromkeys(cond for cond, _, _ in blocks))
    )

    for condition, start, end in blocks:
        fig.add_shape(
            type='rect',
            xref='x', yref='paper',
            x0=start - 0.5, x1=end + 0.5,
            y0=STRIP_Y0, y1=STRIP_Y1,
            fillcolor=color_map[condition],
            line=dict(width=0),
            layer='above',
        )
        fig.add_annotation(
            xref='x', yref='paper',
            x=(start + end) / 2, y=STRIP_Y1,
            text=condition,
            showarrow=False, yanchor='bottom',
            font=dict(size=12, color='black'),
        )
        # Legend proxy: an empty trace carrying only the condition swatch.
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol='square', color=color_map[condition]),
            name=condition,
            showlegend=True,
            hoverinfo='skip',
        ))

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
) -> None:
    """Size the figure so every cell renders as a square of CELL_SIZE_PX.

    The plot area is fixed at n_cols x n_rows cells and the margins are sized
    from the longest tick label, so the caller must render the figure at its
    natural size rather than stretching it to the container width.
    """
    left = _label_extent(y_labels) + (CLASS_LABEL_WIDTH if grouped else 0)
    bottom = _label_extent(x_labels)

    fig.update_layout(
        title=title,
        xaxis_title='Samples',
        yaxis_title='Lipid Molecules',
        margin=dict(l=left, r=MARGIN_RIGHT, t=MARGIN_TOP, b=bottom),
        width=left + MARGIN_RIGHT + n_cols * CELL_SIZE_PX,
        height=MARGIN_TOP + bottom + n_rows * CELL_SIZE_PX,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title='Condition', font=dict(color='black'),
            x=1.02, xanchor='left', y=1, yanchor='top',
        ),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'))


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
