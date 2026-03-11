"""
Box plot plotting service.

Creates missing values bar charts and concentration box plots
for quality check visualization.

Pure logic — no Streamlit dependencies.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Colorblind-friendly palette for conditions
CONDITION_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def _get_sample_colors(
    full_samples_list: List[str],
    conditions_list: List[str],
    individual_samples_list: List[List[str]],
) -> Tuple[List[str], dict, dict]:
    """Map each sample to a color based on its condition.

    Returns:
        Tuple of (colors_per_sample, sample_to_condition, condition_to_color).
    """
    sample_to_condition = {}
    for cond_idx, samples in enumerate(individual_samples_list):
        for sample in samples:
            sample_to_condition[sample] = conditions_list[cond_idx]

    condition_to_color = {
        cond: CONDITION_COLORS[i % len(CONDITION_COLORS)]
        for i, cond in enumerate(conditions_list)
    }

    colors = [
        condition_to_color[sample_to_condition.get(sample, conditions_list[0])]
        for sample in full_samples_list
    ]

    return colors, sample_to_condition, condition_to_color


class BoxPlotService:
    """Creates missing values bar charts and concentration box plots."""

    @staticmethod
    def create_mean_area_df(
        df: pd.DataFrame,
        full_samples_list: List[str],
    ) -> pd.DataFrame:
        """Extract concentration columns for the given samples.

        Args:
            df: DataFrame with concentration[sample] columns.
            full_samples_list: Sample names.

        Returns:
            DataFrame with only the concentration columns.
        """
        concentration_cols = [f'concentration[{s}]' for s in full_samples_list]
        return df[concentration_cols]

    @staticmethod
    def calculate_missing_values_percentage(
        mean_area_df: pd.DataFrame,
    ) -> List[float]:
        """Calculate percentage of zero values per sample column.

        Args:
            mean_area_df: DataFrame with concentration columns only.

        Returns:
            List of percentages (one per column).
        """
        return [
            len(mean_area_df[mean_area_df[col] == 0]) / len(mean_area_df) * 100
            for col in mean_area_df.columns
        ]

    @staticmethod
    def plot_missing_values(
        full_samples_list: List[str],
        zero_values_percent_list: List[float],
        conditions_list: Optional[List[str]] = None,
        individual_samples_list: Optional[List[List[str]]] = None,
    ) -> go.Figure:
        """Create horizontal bar chart of missing values percentage.

        Bars are colored by condition when condition info is provided.

        Args:
            full_samples_list: Sample names.
            zero_values_percent_list: Missing-value percentages per sample.
            conditions_list: Condition names (optional).
            individual_samples_list: Samples grouped by condition (optional).

        Returns:
            Plotly Figure.
        """
        dynamic_height = max(400, len(full_samples_list) * 25)
        total_height = dynamic_height + 60

        fig = go.Figure()

        if conditions_list and individual_samples_list:
            colors, _, condition_to_color = _get_sample_colors(
                full_samples_list, conditions_list, individual_samples_list
            )

            # Individual bars (no legend)
            for i, sample in enumerate(full_samples_list):
                fig.add_trace(go.Bar(
                    y=[sample],
                    x=[zero_values_percent_list[i]],
                    orientation='h',
                    text=[f'{zero_values_percent_list[i]:.1f}%'],
                    textposition='outside',
                    textfont=dict(size=12, color='black'),
                    marker_color=colors[i],
                    showlegend=False,
                ))

            # Invisible legend traces (one per condition)
            for condition in conditions_list:
                fig.add_trace(go.Bar(
                    y=[None], x=[None], orientation='h',
                    marker_color=condition_to_color[condition],
                    name=condition, showlegend=True,
                ))
        else:
            fig.add_trace(go.Bar(
                y=full_samples_list,
                x=zero_values_percent_list,
                orientation='h',
                text=[f'{val:.1f}%' for val in zero_values_percent_list],
                textposition='outside',
                textfont=dict(size=12, color='black'),
                marker_color='lightblue',
                name='', showlegend=False,
            ))

        fig.update_layout(
            title=dict(
                text='Missing Values Distribution',
                font=dict(size=20, color='black'),
                y=0.99, x=0.5, xanchor='center', yanchor='top',
            ),
            xaxis_title=dict(
                text='Percentage of Missing Values',
                font=dict(size=14, color='black'), standoff=15,
            ),
            yaxis_title=dict(
                text='Sample',
                font=dict(size=14, color='black'), standoff=15,
            ),
            yaxis=dict(tickfont=dict(size=12, color='black')),
            xaxis=dict(tickfont=dict(size=12, color='black')),
            margin=dict(l=100, r=100, t=60, b=120),
            showlegend=bool(conditions_list),
            legend=dict(
                orientation='h', yanchor='top', y=-0.18,
                xanchor='center', x=0.5,
            ),
            height=total_height + 80,
            width=900,
            plot_bgcolor='white',
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        return fig

    @staticmethod
    def plot_box_plot(
        mean_area_df: pd.DataFrame,
        full_samples_list: List[str],
        conditions_list: Optional[List[str]] = None,
        individual_samples_list: Optional[List[List[str]]] = None,
    ) -> go.Figure:
        """Create box plot of log10-transformed non-zero concentrations."""
        log_transformed_data = [
            list(np.log10(mean_area_df[col][mean_area_df[col] > 0]))
            for col in mean_area_df.columns
        ]

        fig = go.Figure()
        BoxPlotService._add_box_traces(
            fig, log_transformed_data, full_samples_list,
            conditions_list, individual_samples_list
        )
        BoxPlotService._apply_box_plot_layout(fig, bool(conditions_list))
        return fig

    @staticmethod
    def _add_box_traces(
        fig: go.Figure,
        data: List[list],
        samples: List[str],
        conditions_list: Optional[List[str]],
        individual_samples_list: Optional[List[List[str]]],
    ) -> None:
        """Add box traces, colored by condition if available."""
        if conditions_list and individual_samples_list:
            colors, _, condition_to_color = _get_sample_colors(
                samples, conditions_list, individual_samples_list
            )
            for i, d in enumerate(data):
                fig.add_trace(go.Box(
                    y=d, name=samples[i], boxpoints='outliers',
                    marker_color=colors[i], line_color=colors[i],
                    showlegend=False,
                ))
            for condition in conditions_list:
                fig.add_trace(go.Box(
                    y=[None], name=condition,
                    marker_color=condition_to_color[condition],
                    line_color=condition_to_color[condition],
                    showlegend=True,
                ))
        else:
            for i, d in enumerate(data):
                fig.add_trace(go.Box(
                    y=d, name=samples[i], boxpoints='outliers',
                    marker_color='lightblue', line_color='darkblue',
                    showlegend=False,
                ))

    @staticmethod
    def _apply_box_plot_layout(fig: go.Figure, show_legend: bool) -> None:
        """Apply standard box plot layout."""
        fig.update_layout(
            title=dict(
                text='Box Plot of Non-Zero Concentrations',
                font=dict(size=20, color='black'),
                y=0.99, x=0.5, xanchor='center', yanchor='top',
            ),
            xaxis_title=dict(text='Sample', font=dict(size=14, color='black'), standoff=15),
            yaxis_title=dict(text='log10(Concentration)', font=dict(size=14, color='black'), standoff=15),
            xaxis=dict(tickangle=45, tickfont=dict(size=12, color='black'), title_standoff=30),
            yaxis=dict(tickfont=dict(size=12, color='black')),
            margin=dict(l=100, r=100, t=60, b=120),
            showlegend=show_legend,
            legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5),
            height=700, width=900,
            plot_bgcolor='white',
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
