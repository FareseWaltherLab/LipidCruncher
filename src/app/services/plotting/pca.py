"""
PCA plotting service.

Creates PCA scatter plots with confidence ellipses for
sample clustering visualization.

Pure logic — no Streamlit dependencies.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAPlotterService:
    """Creates PCA scatter plots with confidence ellipses."""

    @staticmethod
    def plot_pca(
        df: pd.DataFrame,
        full_samples_list: List[str],
        extensive_conditions_list: List[str],
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Full PCA pipeline: compute PCA and create scatter plot.

        Args:
            df: DataFrame with concentration[sample] columns.
            full_samples_list: All sample names.
            extensive_conditions_list: Condition label per sample
                (same length as full_samples_list, index-aligned).

        Returns:
            Tuple of (Plotly Figure, DataFrame with PC1/PC2/Sample/Condition).
        """
        pc_df, pc_names, available_samples = _run_pca(df, full_samples_list)

        pc_df['Sample'] = available_samples
        pc_df['Condition'] = [
            extensive_conditions_list[full_samples_list.index(s)]
            for s in available_samples
        ]

        color_mapping = _generate_color_mapping(pc_df['Condition'])
        fig = _create_scatter_plot(pc_df, pc_names, color_mapping)

        return fig, pc_df


def _run_pca(
    df: pd.DataFrame,
    full_samples_list: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Run PCA on concentration data.

    Returns:
        Tuple of (pc_df with PC1/PC2, pc_names with variance %, available_samples).
    """
    if isinstance(df, tuple):
        df = df[0]

    available_samples = [
        s for s in full_samples_list
        if f'concentration[{s}]' in df.columns
    ]

    mean_area_df = df[[f'concentration[{s}]' for s in available_samples]].T
    scaled_data = StandardScaler().fit_transform(mean_area_df)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_

    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_names = [f'PC{i+1} ({var:.0%})' for i, var in enumerate(explained_variance)]

    return pc_df, pc_names, available_samples


def _generate_color_mapping(conditions) -> Dict[str, str]:
    """Map each unique condition to a Plotly qualitative color."""
    unique_conditions = sorted(set(conditions))
    palette = px.colors.qualitative.Plotly
    return {
        cond: palette[i % len(palette)]
        for i, cond in enumerate(unique_conditions)
    }


def _create_scatter_plot(
    df: pd.DataFrame,
    pc_names: List[str],
    color_mapping: Dict[str, str],
) -> go.Figure:
    """Create PCA scatter plot with confidence ellipses."""
    fig = go.Figure()

    for condition in df['Condition'].unique():
        cond_df = df[df['Condition'] == condition]
        fig.add_trace(go.Scatter(
            x=cond_df['PC1'], y=cond_df['PC2'],
            mode='markers', name=condition,
            marker=dict(color=color_mapping[condition], size=5),
            text=cond_df['Sample'],
            hovertemplate=(
                '<b>Sample:</b> %{text}<br>'
                '<b>PC1:</b> %{x:.4f}<br>'
                '<b>PC2:</b> %{y:.4f}<br>'
                '<extra></extra>'
            ),
        ))

        _add_confidence_ellipse(
            fig, cond_df['PC1'], cond_df['PC2'],
            color_mapping[condition], condition,
        )

    fig.update_layout(
        title=dict(text='PCA Plot', font=dict(size=24, color='black')),
        xaxis_title=dict(text=pc_names[0], font=dict(size=18, color='black')),
        yaxis_title=dict(text=pc_names[1], font=dict(size=18, color='black')),
        xaxis=dict(tickfont=dict(size=14, color='black')),
        yaxis=dict(tickfont=dict(size=14, color='black')),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(font=dict(size=12, color='black')),
        margin=dict(t=50, r=50, b=50, l=50),
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    return fig


def _add_confidence_ellipse(
    fig: go.Figure,
    x, y,
    color: str,
    name: str,
    n_std: float = 2.0,
) -> None:
    """Add a 95% confidence ellipse to the figure."""
    if len(x) < 2:
        return

    cov = np.cov(x, y)

    # Eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov)

    # Sort eigenvalues descending
    order = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    # Width and height
    width, height = 2 * n_std * np.sqrt(eig_vals)

    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = np.dot(eig_vecs, circle * np.array([[width / 2], [height / 2]]))

    # Center
    center = np.mean(list(zip(x, y)), axis=0)
    ellipse[0] += center[0]
    ellipse[1] += center[1]

    fig.add_trace(go.Scatter(
        x=ellipse[0], y=ellipse[1],
        mode='lines', line=dict(color=color, width=1),
        name=f'{name} Confidence Ellipse',
        showlegend=False,
    ))
