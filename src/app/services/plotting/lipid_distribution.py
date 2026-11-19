"""
The volcano page's individual-lipid figure: one panel per lipid, with a bar
(mean ± SD) or a box per condition, a dot for each sample, and a bracket
marking the significance of the difference between the two conditions.

Pure logic — no Streamlit dependencies.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.services.plotting._shared import (
    BASE_LAYOUT,
    OUTLINE_WIDTH,
    frame_axes,
    outlined_marker,
    p_value_to_marker,
    tint,
)

STYLES = ('bar', 'box')
MAX_COLUMNS = 3
PANEL_WIDTH_PX = 220
PANEL_HEIGHT_PX = 320
MARGIN = dict(t=90, b=120, l=80, r=30)
ROW_GAP_PX = 90
COLUMN_GAP_PX = 70
BAR_WIDTH = 0.6
# Each condition's dots spread evenly across this much of the x axis.
POINT_SPREAD = 0.3
POINT_SIZE = 7
# The bracket sits this fraction above the panel's tallest mark, its ends
# dropping by BRACKET_TICK of it.
BRACKET_GAP = 0.08
BRACKET_TICK = 0.03
HEADROOM = 1.15
_FONT = dict(color='black')
SIGNIFICANCE_KEY = 'ns p ≥ 0.05    * p < 0.05    ** p < 0.01    *** p < 0.001'


def build_distribution_figure(
    plot_df: pd.DataFrame,
    conditions: List[str],
    colors: Dict[str, str],
    style: str,
    p_values: Optional[Dict[str, float]] = None,
    p_label: str = '',
) -> go.Figure:
    """Draw one panel per lipid of ``plot_df`` (Lipid, Condition, Concentration).

    Args:
        plot_df: One row per (lipid, sample).
        conditions: Conditions in x-axis order.
        colors: Condition -> color.
        style: 'bar' for mean ± SD bars, 'box' for box plots.
        p_values: Lipid -> p-value of the two conditions' difference; a
            bracket is drawn for each lipid listed when there are two
            conditions.
        p_label: Where the p-values come from, for the caption.
    """
    lipids = list(dict.fromkeys(plot_df['Lipid']))
    n_cols = min(len(lipids), MAX_COLUMNS)
    n_rows = math.ceil(len(lipids) / n_cols)
    # The figure is always as wide as a full row, so the title and caption
    # fit; fewer panels sit centred at their usual width.
    plot_width = PANEL_WIDTH_PX * MAX_COLUMNS + COLUMN_GAP_PX * (MAX_COLUMNS - 1)
    plot_height = PANEL_HEIGHT_PX * n_rows + ROW_GAP_PX * (n_rows - 1)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'<b>{lipid}</b>' for lipid in lipids],
        vertical_spacing=ROW_GAP_PX / plot_height if n_rows > 1 else 0,
    )
    _place_columns(fig, n_cols, n_rows, plot_width, len(lipids))

    brackets = bool(p_values) and len(conditions) == 2
    for k, lipid in enumerate(lipids):
        row, col = k // n_cols + 1, k % n_cols + 1
        lipid_df = plot_df[plot_df['Lipid'] == lipid]
        top = 0.0
        for j, condition in enumerate(conditions):
            values = lipid_df.loc[
                lipid_df['Condition'] == condition, 'Concentration',
            ].to_numpy(dtype=float)
            if values.size:
                top = max(top, _add_condition(
                    fig, values, j, condition, colors[condition], style, row, col,
                ))
        if brackets and lipid in p_values:
            top = _add_bracket(fig, p_values[lipid], top, row, col)
        fig.update_yaxes(range=[0, (top or 1) * HEADROOM], row=row, col=col)

    fig.update_xaxes(
        tickvals=list(range(len(conditions))), ticktext=conditions,
        range=[-0.6, len(conditions) - 0.4],
        tickfont=dict(size=13, color='black'),
    )
    fig.update_yaxes(
        ticks='outside', showgrid=False, rangemode='tozero',
        exponentformat='power', tickfont=dict(size=12, color='black'),
    )
    for row in range(1, n_rows + 1):
        fig.update_yaxes(
            title=dict(text='Concentration', font=dict(size=14, color='black')),
            row=row, col=1,
        )
    frame_axes(fig)
    fig.update_annotations(font=dict(size=14, color='black'))

    _add_caption(fig, plot_df, conditions, style, brackets, p_label)
    fig.update_layout(
        title=dict(
            text='Concentration Distribution for Selected Lipids',
            font=dict(size=18, color='black'), x=0.5, xanchor='center',
        ),
        showlegend=False,
        bargap=0,
        width=plot_width + MARGIN['l'] + MARGIN['r'],
        height=plot_height + MARGIN['t'] + MARGIN['b'],
        margin=MARGIN,
        font=_FONT,
        **BASE_LAYOUT,
    )
    return fig


def _place_columns(
    fig: go.Figure, n_cols: int, n_rows: int, plot_width: int, n_panels: int,
) -> None:
    """Give each column its fixed pixel width, the columns centred, and move
    the panel titles (the first ``n_panels`` annotations) over them."""
    used = PANEL_WIDTH_PX * n_cols + COLUMN_GAP_PX * (n_cols - 1)
    left = (plot_width - used) / 2
    centres = []
    for col in range(n_cols):
        x0 = (left + col * (PANEL_WIDTH_PX + COLUMN_GAP_PX)) / plot_width
        x1 = x0 + PANEL_WIDTH_PX / plot_width
        centres.append((x0 + x1) / 2)
        for row in range(1, n_rows + 1):
            fig.update_xaxes(domain=[x0, x1], row=row, col=col + 1)
    for k, title in enumerate(fig.layout.annotations[:n_panels]):
        title.x = centres[k % n_cols]


def _add_condition(
    fig: go.Figure,
    values: np.ndarray,
    x: int,
    condition: str,
    color: str,
    style: str,
    row: int,
    col: int,
) -> float:
    """Draw one condition's bar or box and dots; return its tallest mark."""
    if style == 'box':
        fig.add_trace(go.Box(
            x=[x] * values.size, y=values, name=condition, width=BAR_WIDTH,
            boxpoints='all', jitter=0.5, pointpos=0,
            fillcolor=tint(color),
            line=dict(color=color, width=OUTLINE_WIDTH),
            marker=dict(color=color, size=POINT_SIZE, opacity=0.8),
        ), row=row, col=col)
        return float(values.max())

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    fig.add_trace(go.Bar(
        x=[x], y=[mean], name=condition, width=BAR_WIDTH,
        marker=outlined_marker(color),
        error_y=dict(
            type='data', array=[sd], color=color,
            thickness=OUTLINE_WIDTH, width=8,
        ),
        hovertemplate=f'{condition}<br>mean %{{y:.3g}} ± {sd:.3g} SD<extra></extra>',
    ), row=row, col=col)
    half = POINT_SPREAD / 2
    offsets = np.linspace(-half, half, values.size) if values.size > 1 else [0.0]
    fig.add_trace(go.Scatter(
        x=[x + offset for offset in offsets], y=values, name=condition,
        mode='markers',
        marker=dict(color=color, size=POINT_SIZE, opacity=0.8),
        hovertemplate=f'{condition}: %{{y:.3g}}<extra></extra>',
    ), row=row, col=col)
    return max(mean + sd, float(values.max()))


def _add_bracket(
    fig: go.Figure, p_value: float, top: float, row: int, col: int,
) -> float:
    """Draw the significance bracket above ``top``; return its height."""
    if p_value is None or np.isnan(p_value):
        return top
    y = top * (1 + BRACKET_GAP) if top > 0 else 1.0
    drop = y * BRACKET_TICK
    fig.add_shape(
        type='path', path=f'M 0,{y - drop} L 0,{y} L 1,{y} L 1,{y - drop}',
        line=dict(color='black', width=1.5), row=row, col=col,
    )
    fig.add_annotation(
        x=0.5, y=y, text=p_value_to_marker(p_value) or 'ns',
        showarrow=False, yanchor='bottom', row=row, col=col,
    )
    return y


def _add_caption(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    conditions: List[str],
    style: str,
    brackets: bool,
    p_label: str,
) -> None:
    """Say below the panels what the marks, n and brackets mean."""
    sizes = (
        plot_df.drop_duplicates(['Condition', 'Sample'])
        .groupby('Condition').size().reindex(conditions, fill_value=0)
    )
    if sizes.nunique() == 1:
        n_text = f'n = {sizes.iloc[0]} per group'
    else:
        n_text = 'n = ' + ', '.join(f'{n} ({c})' for c, n in sizes.items())
    marks = 'Mean ± SD' if style == 'bar' else 'Median and IQR, whiskers to 1.5 × IQR'
    lines = [f'{marks}; {n_text}' + (f'; {p_label}' if brackets and p_label else '')]
    if brackets:
        lines.insert(0, SIGNIFICANCE_KEY)
    fig.add_annotation(
        text='<br>'.join(lines),
        xref='paper', yref='paper', x=0.5, y=0, xanchor='center', yanchor='top',
        yshift=-45, showarrow=False, font=dict(size=13, color='black'),
    )
