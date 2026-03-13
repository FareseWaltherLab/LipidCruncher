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
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist


# ── Constants ──────────────────────────────────────────────────────────

HEATMAP_WIDTH = 900
HEATMAP_HEIGHT = 600
COLORSCALE = 'RdBu_r'
CLUSTER_LINE_STYLE = dict(color='black', width=2, dash='dash')


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
