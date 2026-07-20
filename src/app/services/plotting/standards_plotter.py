"""
Internal standards consistency plotting service.

Creates bar plots showing internal standard intensities across samples,
grouped by lipid class. Consistent bar heights indicate good sample
preparation and instrument performance.

Pure logic — no Streamlit dependencies.
"""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.services.plotting._shared import generate_condition_color_mapping


# Color palette for multiple standards in the same class
_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA',
    '#FFA15A', '#19D3F3', '#FF6692', '#B6E880',
]

_FONT_STYLE = dict(color='black')


class StandardsPlotterService:
    """Creates consistency bar plots for internal standards."""

    @staticmethod
    def create_consistency_plots(
        intsta_df: pd.DataFrame,
        samples: List[str],
        sample_conditions: Optional[List[str]] = None,
    ) -> List[go.Figure]:
        """Create bar plots for each internal standard class.

        Multiple standards in the same class get separate subplot rows
        with different colors.

        Args:
            intsta_df: Internal standards DataFrame with columns
                ``LipidMolec``, ``ClassKey``, and ``intensity[<sample>]``.
            samples: Sample names to display (in desired order).
            sample_conditions: Optional condition/group label per sample,
                index-aligned with ``samples``. When provided, bars are grouped
                and colored by condition with a condition legend; otherwise a
                single color per standard is used.

        Returns:
            List of Plotly figures, one per lipid class.
        """
        if intsta_df.empty:
            return []

        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]

        # Filter to only columns for selected samples
        valid_intensity_cols = [
            f'intensity[{s}]' for s in samples if f'intensity[{s}]' in intensity_cols
        ]

        if not valid_intensity_cols:
            return []

        # When coloring by condition, align samples/conditions to the valid
        # (present) samples so bars, intensities, and colors stay in step.
        plot_samples = samples
        valid_conditions: Optional[List[str]] = None
        if sample_conditions is not None:
            cond_by_sample = dict(zip(samples, sample_conditions))
            plot_samples = [
                s for s in samples if f'intensity[{s}]' in intensity_cols
            ]
            valid_conditions = [cond_by_sample[s] for s in plot_samples]

        figs: List[go.Figure] = []
        classes = sorted(intsta_df['ClassKey'].unique())

        for class_key in classes:
            class_df = intsta_df[intsta_df['ClassKey'] == class_key]
            if class_df.empty:
                continue

            standards = class_df['LipidMolec'].unique()
            n_standards = len(standards)

            if n_standards == 1:
                fig = _build_single_standard_plot(
                    class_df, standards[0], plot_samples, valid_intensity_cols,
                    class_key, valid_conditions,
                )
            else:
                fig = _build_multi_standard_plot(
                    class_df, standards, plot_samples, valid_intensity_cols,
                    class_key, valid_conditions,
                )

            # Style all axes
            fig.update_xaxes(tickfont=_FONT_STYLE, title_font=_FONT_STYLE)
            fig.update_yaxes(tickfont=_FONT_STYLE, title_font=_FONT_STYLE)
            fig.update_layout(legend=dict(font=_FONT_STYLE))

            figs.append(fig)

        return figs


def _condition_groups(
    samples: List[str],
    intensities: List[float],
    sample_conditions: List[str],
):
    """Yield (condition, xs, ys) grouped by condition (first-seen order)."""
    for condition in dict.fromkeys(sample_conditions):
        xs = [s for s, c in zip(samples, sample_conditions) if c == condition]
        ys = [y for y, c in zip(intensities, sample_conditions) if c == condition]
        yield condition, xs, ys


def _build_single_standard_plot(
    class_df: pd.DataFrame,
    standard_name: str,
    samples: List[str],
    intensity_cols: List[str],
    class_key: str,
    sample_conditions: Optional[List[str]] = None,
) -> go.Figure:
    """Build a bar plot for a class with a single internal standard."""
    intensities = [class_df[col].values[0] for col in intensity_cols]

    if sample_conditions is None:
        fig = go.Figure(data=[
            go.Bar(
                x=samples,
                y=intensities,
                marker_color=_COLORS[0],
                name=standard_name,
            )
        ])
        legend_title = 'Internal Standard'
    else:
        color_map = generate_condition_color_mapping(
            list(dict.fromkeys(sample_conditions))
        )
        fig = go.Figure(data=[
            go.Bar(x=xs, y=ys, name=cond, marker_color=color_map[cond])
            for cond, xs, ys in _condition_groups(
                samples, intensities, sample_conditions
            )
        ])
        legend_title = 'Condition'

    fig.update_layout(
        title=f'Internal Standards Intensity for {class_key} ({standard_name})',
        xaxis_title='Samples',
        yaxis_title='Raw Intensity',
        height=400,
        bargap=0.2,
        showlegend=True,
        legend_title=legend_title,
        font=_FONT_STYLE,
        title_font=_FONT_STYLE,
    )
    return fig


def _build_multi_standard_plot(
    class_df: pd.DataFrame,
    standards,
    samples: List[str],
    intensity_cols: List[str],
    class_key: str,
    sample_conditions: Optional[List[str]] = None,
) -> go.Figure:
    """Build a subplot figure for a class with multiple internal standards."""
    n_standards = len(standards)

    fig = make_subplots(
        rows=n_standards,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[str(std) for std in standards],
    )

    color_map = (
        generate_condition_color_mapping(list(dict.fromkeys(sample_conditions)))
        if sample_conditions is not None else None
    )

    for i, standard_name in enumerate(standards):
        std_row = class_df[class_df['LipidMolec'] == standard_name]
        intensities = [
            std_row[col].values[0] if not std_row.empty else 0
            for col in intensity_cols
        ]

        if sample_conditions is None:
            fig.add_trace(
                go.Bar(
                    x=samples,
                    y=intensities,
                    marker_color=_COLORS[i % len(_COLORS)],
                    name=standard_name,
                    showlegend=True,
                ),
                row=i + 1, col=1,
            )
        else:
            # One trace per condition; legend only on the first subplot row.
            for cond, xs, ys in _condition_groups(
                samples, intensities, sample_conditions
            ):
                fig.add_trace(
                    go.Bar(
                        x=xs, y=ys, name=cond,
                        marker_color=color_map[cond],
                        legendgroup=cond,
                        showlegend=(i == 0),
                    ),
                    row=i + 1, col=1,
                )
        fig.update_yaxes(title_text='Intensity', row=i + 1, col=1)

    fig.update_layout(
        title=f'Internal Standards Intensity for {class_key}',
        height=300 * n_standards,
        bargap=0.2,
        showlegend=True,
        legend_title='Internal Standard' if sample_conditions is None else 'Condition',
        font=_FONT_STYLE,
        title_font=_FONT_STYLE,
    )
    # X-axis title only on the bottom subplot
    fig.update_xaxes(title_text='Samples', row=n_standards, col=1)

    return fig
