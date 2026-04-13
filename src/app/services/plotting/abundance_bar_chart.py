"""
Abundance bar chart plotting service.

Creates grouped horizontal bar charts comparing lipid class abundance
across experimental conditions, with optional significance annotations.

Pure logic — no Streamlit dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.constants import CONDITION_COLORS
from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import (
    generate_condition_color_mapping,
    get_effective_p_value,
    p_value_to_marker,
)
from app.services.statistical_testing import StatisticalTestSummary

ZERO_REPLACEMENT_DIVISOR = 10
MIN_CHART_HEIGHT = 400
HEIGHT_PER_CLASS = 30
CHART_WIDTH = 800


@dataclass
class BarChartData:
    """Computed abundance statistics for bar chart rendering.

    Attributes:
        abundance_df: DataFrame with ClassKey and mean/std columns per condition.
            Columns: ClassKey, mean_AUC_{cond}, std_AUC_{cond},
                     log10_mean_AUC_{cond}, log10_std_AUC_{cond}
        conditions: Conditions included in the data.
        classes: Lipid classes included in the data.
    """
    abundance_df: pd.DataFrame
    conditions: List[str]
    classes: List[str]


class BarChartPlotterService:
    """Creates abundance bar charts comparing lipid class totals across conditions."""

    @staticmethod
    def create_mean_std_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> BarChartData:
        """Compute mean and std of class-level abundance per condition.

        For each (class, condition) pair, sums species concentrations per sample,
        then computes linear mean/std and log10 mean/std.

        Args:
            df: DataFrame with ClassKey and concentration[s*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.

        Returns:
            BarChartData with computed statistics.

        Raises:
            ValueError: If no valid data can be computed.
        """
        if selected_conditions is None or len(selected_conditions) == 0:
            raise ValueError("At least one condition must be selected")
        if selected_classes is None or len(selected_classes) == 0:
            raise ValueError("At least one lipid class must be selected")

        condition_frames: List[pd.DataFrame] = []

        for condition in selected_conditions:
            if condition not in experiment.conditions_list:
                continue

            condition_idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[condition_idx]
            sample_cols = [
                f"concentration[{s}]" for s in samples
                if f"concentration[{s}]" in df.columns
            ]
            if not sample_cols:
                continue

            rows = []
            for class_name in selected_classes:
                class_df = df[df['ClassKey'] == class_name]
                if class_df.empty:
                    continue

                totals = class_df[sample_cols].sum(axis=0)
                linear_mean = totals.mean()
                linear_std = totals.std(ddof=1) if len(totals) > 1 else 0.0

                log10_mean, log10_std = _compute_log10_stats(totals)

                rows.append({
                    'ClassKey': class_name,
                    f'mean_AUC_{condition}': linear_mean,
                    f'std_AUC_{condition}': linear_std,
                    f'log10_mean_AUC_{condition}': log10_mean,
                    f'log10_std_AUC_{condition}': log10_std,
                })

            if rows:
                condition_frames.append(pd.DataFrame(rows))

        if not condition_frames:
            raise ValueError("No valid data available after processing")

        merged = condition_frames[0]
        for frame in condition_frames[1:]:
            merged = merged.merge(frame, on='ClassKey', how='outer')

        # Preserve requested class order
        merged = merged[merged['ClassKey'].isin(selected_classes)]
        class_order = {c: i for i, c in enumerate(selected_classes)}
        merged = merged.sort_values(
            'ClassKey', key=lambda s: s.map(class_order)
        ).reset_index(drop=True)

        actual_conditions = [
            c for c in selected_conditions
            if f'mean_AUC_{c}' in merged.columns
        ]

        return BarChartData(
            abundance_df=merged,
            conditions=actual_conditions,
            classes=merged['ClassKey'].tolist(),
        )

    @staticmethod
    def create_bar_chart(
        bar_data: BarChartData,
        mode: str,
        stat_results: Optional[StatisticalTestSummary] = None,
    ) -> go.Figure:
        """Create a horizontal grouped bar chart with error bars.

        Args:
            bar_data: Pre-computed abundance statistics.
            mode: Scale mode — 'linear scale' or 'log10 scale'.
            stat_results: Optional statistical test results for annotations.

        Returns:
            Plotly Figure.

        Raises:
            ValueError: If mode is invalid or bar_data is empty.
        """
        if mode not in ('linear scale', 'log10 scale'):
            raise ValueError(
                f"Invalid mode: '{mode}'. Must be 'linear scale' or 'log10 scale'"
            )

        abundance_df = bar_data.abundance_df
        conditions = bar_data.conditions

        if abundance_df.empty or not conditions:
            raise ValueError("No data available to create bar chart")

        colors = BarChartPlotterService.generate_color_mapping(conditions)
        fig = go.Figure()

        bar_width = 0.8 / len(conditions)

        for i, condition in enumerate(conditions):
            mean_col, std_col = _get_mode_columns(condition, mode)

            mean_vals = abundance_df.get(mean_col, pd.Series(dtype=float)).fillna(0)
            std_vals = abundance_df.get(std_col, pd.Series(dtype=float)).fillna(0)

            if mean_vals.empty:
                continue

            y_positions = (
                np.arange(len(abundance_df))
                + (i - len(conditions) / 2 + 0.5) * bar_width
            )

            fig.add_trace(go.Bar(
                y=y_positions,
                x=mean_vals,
                name=condition,
                orientation='h',
                error_x=dict(
                    type='data',
                    array=std_vals,
                    visible=True,
                    color='black',
                    thickness=1,
                    width=0,
                ),
                width=bar_width,
                showlegend=True,
                marker_color=colors[condition],
            ))

        # Significance annotations
        if stat_results is not None and fig.data:
            _add_significance_annotations(fig, abundance_df, stat_results)

        scale_label = "Linear" if mode == 'linear scale' else "Log10"
        conditions_str = " vs ".join(conditions)

        fig.update_layout(
            title=dict(
                text=(
                    f"Class Concentration Bar Chart ({scale_label} Scale)"
                    f"<br>{conditions_str}"
                ),
                font=dict(color='black'),
            ),
            xaxis_title=dict(text='Mean Concentration', font=dict(color='black')),
            yaxis_title=dict(text='Lipid Class', font=dict(color='black')),
            xaxis=dict(
                tickfont=dict(color='black'),
                showgrid=True,
                gridcolor='lightgray',
            ),
            yaxis=dict(
                ticktext=abundance_df['ClassKey'].values,
                tickvals=np.arange(len(abundance_df)),
                autorange='reversed',
                tickfont=dict(color='black'),
                showgrid=True,
                gridcolor='lightgray',
            ),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.15,
                font=dict(color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgray',
                borderwidth=1,
            ),
            margin=dict(r=250),
            height=max(MIN_CHART_HEIGHT, len(abundance_df) * HEIGHT_PER_CLASS),
            width=CHART_WIDTH,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        return fig

    @staticmethod
    def generate_color_mapping(conditions: List[str]) -> Dict[str, str]:
        """Map each condition to a consistent color.

        Args:
            conditions: List of condition labels.

        Returns:
            Dict mapping condition name to hex color string.
        """
        return generate_condition_color_mapping(conditions)


# ── Private helpers ────────────────────────────────────────────────────


def _compute_log10_stats(
    totals: pd.Series,
) -> Tuple[float, float]:
    """Compute log10 mean and std from sample totals, replacing zeros.

    Returns:
        (log10_mean, log10_std). Both NaN if all values are zero.
    """
    positive = totals[totals > 0]
    if positive.empty:
        return np.nan, np.nan

    small_value = positive.min() / ZERO_REPLACEMENT_DIVISOR
    adjusted = totals.replace(0, small_value)
    log_vals = np.log10(adjusted)
    log_mean = log_vals.mean()
    log_std = log_vals.std(ddof=1) if len(log_vals) > 1 else 0.0
    return log_mean, log_std


def _get_mode_columns(condition: str, mode: str) -> Tuple[str, str]:
    """Return (mean_column, std_column) names for the given mode."""
    if mode == 'linear scale':
        return f'mean_AUC_{condition}', f'std_AUC_{condition}'
    return f'log10_mean_AUC_{condition}', f'log10_std_AUC_{condition}'


def _add_significance_annotations(
    fig: go.Figure,
    abundance_df: pd.DataFrame,
    stat_results: StatisticalTestSummary,
) -> None:
    """Add *, **, *** annotations to the bar chart for significant classes."""
    for i, class_name in enumerate(abundance_df['ClassKey']):
        if class_name not in stat_results.results:
            continue

        result = stat_results.results[class_name]
        p_val = get_effective_p_value(result)

        marker = p_value_to_marker(p_val)
        if marker:
            fig.add_annotation(
                x=1.0,
                xref='paper',
                xanchor='left',
                y=i,
                text=f"  {marker}",
                showarrow=False,
                font=dict(size=14, color='black'),
            )



# _p_value_to_marker is now in _shared.py — re-export for backward compatibility
_p_value_to_marker = p_value_to_marker
