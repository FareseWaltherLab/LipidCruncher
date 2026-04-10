"""
Retention time plotting service.

Creates scatter plots of retention time vs. calculated mass
for lipid class identification quality checks.

Pure logic — no Streamlit dependencies.
"""

import colorsys
import itertools
from typing import List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class RetentionTimePlotterService:
    """Creates retention time plots for lipid identification quality checks."""

    @staticmethod
    def plot_single_retention(
        df: pd.DataFrame,
    ) -> List[Tuple[go.Figure, pd.DataFrame]]:
        """Generate individual retention time plots for each lipid class.

        Args:
            df: DataFrame with ClassKey, BaseRt, CalcMass, LipidMolec columns.

        Returns:
            List of (figure, retention_df) tuples, one per lipid class,
            ordered by class frequency (most frequent first).

        Raises:
            ValueError: If df is empty or missing required columns.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        for col in ('ClassKey', 'BaseRt', 'CalcMass', 'LipidMolec'):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        plots = []
        for lipid_class in df['ClassKey'].value_counts().index:
            class_df = df[df['ClassKey'] == lipid_class]
            retention_df = pd.DataFrame({
                'Mass': class_df['CalcMass'].values,
                'Retention': class_df['BaseRt'].values,
                'Species': class_df['LipidMolec'].values,
            })
            fig = _render_single_plot(retention_df, lipid_class)
            plots.append((fig, retention_df))
        return plots

    @staticmethod
    def plot_multi_retention(
        df: pd.DataFrame,
        selected_classes_list: List[str],
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Generate a retention time comparison plot for selected classes.

        Args:
            df: DataFrame with ClassKey, BaseRt, CalcMass, LipidMolec columns.
            selected_classes_list: Classes to include in the comparison.

        Returns:
            Tuple of (figure, retention_df).

        Raises:
            ValueError: If df is empty, missing required columns,
                or selected_classes_list is empty.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        for col in ('ClassKey', 'BaseRt', 'CalcMass', 'LipidMolec'):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if not selected_classes_list:
            raise ValueError("At least one class must be selected")

        all_classes = df['ClassKey'].value_counts().index.tolist()
        unique_colors = _get_unique_colors(len(all_classes))

        # Build combined data for selected classes
        plot_data = []
        for lipid_class in selected_classes_list:
            class_df = df[df['ClassKey'] == lipid_class]
            # Safe lookup: default to 0 if class not found in all_classes
            color_idx = all_classes.index(lipid_class) if lipid_class in all_classes else 0
            plot_data.append(pd.DataFrame({
                'Mass': class_df['CalcMass'],
                'Retention': class_df['BaseRt'],
                'LipidMolec': class_df['LipidMolec'],
                'Class': lipid_class,
                'Color': unique_colors[color_idx % len(unique_colors)],
            }))

        if not plot_data:
            retention_df = pd.DataFrame(columns=['Mass', 'Retention', 'LipidMolec', 'Class', 'Color'])
        else:
            retention_df = pd.concat(plot_data, ignore_index=True)
        fig = _render_multi_plot(retention_df)
        return fig, retention_df


def _render_single_plot(retention_df: pd.DataFrame, lipid_class: str) -> go.Figure:
    """Create a single retention time scatter plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=retention_df['Mass'], y=retention_df['Retention'],
        mode='markers', marker=dict(size=6),
        text=retention_df['Species'],
        hovertemplate=(
            '<b>Mass</b>: %{x:.4f}<br>'
            '<b>Retention time</b>: %{y:.2f}<br>'
            '<b>Species</b>: %{text}<extra></extra>'
        ),
    ))

    fig.update_layout(
        title=dict(text=lipid_class, font=dict(size=24, color='black')),
        xaxis_title=dict(text='Calculated Mass', font=dict(size=18, color='black')),
        yaxis_title=dict(text='Retention Time (mins)', font=dict(size=18, color='black')),
        xaxis=dict(tickfont=dict(size=14, color='black')),
        yaxis=dict(tickfont=dict(size=14, color='black')),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=50, r=50, b=50, l=50),
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    return fig


def _render_multi_plot(retention_df: pd.DataFrame) -> go.Figure:
    """Create a multi-class retention time comparison plot."""
    fig = go.Figure()

    num_classes = len(retention_df['Class'].unique())
    colors = _get_distinct_colors(num_classes)
    color_palette = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]

    for i, lipid_class in enumerate(retention_df['Class'].unique()):
        class_df = retention_df[retention_df['Class'] == lipid_class]
        fig.add_trace(go.Scatter(
            x=class_df['Mass'], y=class_df['Retention'],
            mode='markers', marker=dict(size=6, color=color_palette[i]),
            name=lipid_class,
            text=class_df['LipidMolec'],
            hovertemplate=(
                '<b>Mass</b>: %{x:.4f}<br>'
                '<b>Retention time</b>: %{y:.2f}<br>'
                '<b>Lipid Molecule</b>: %{text}<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=dict(
            text='Retention Time vs. Mass - Comparison Mode',
            font=dict(size=24, color='black'),
        ),
        xaxis_title=dict(text='Calculated Mass', font=dict(size=18, color='black')),
        yaxis_title=dict(text='Retention Time (mins)', font=dict(size=18, color='black')),
        xaxis=dict(tickfont=dict(size=14, color='black')),
        yaxis=dict(tickfont=dict(size=14, color='black')),
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(font=dict(size=12, color='black')),
        margin=dict(t=50, r=50, b=50, l=50),
    )

    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    return fig


def _get_distinct_colors(n: int) -> List[tuple]:
    """Generate n visually distinct colors using HSV."""
    hue_partition = 1.0 / (n + 1)
    return [colorsys.hsv_to_rgb(hue_partition * i, 1.0, 1.0) for i in range(n)]


def _get_unique_colors(n_classes: int) -> List[str]:
    """Generate unique color hex codes from Plotly Express palette (cycles if needed)."""
    colors = itertools.cycle(px.colors.qualitative.Plotly)
    return [next(colors) for _ in range(n_classes)]
