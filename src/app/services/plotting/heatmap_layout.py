"""
Layout of the lipidomic heatmap figures: the base trace, the colour-coded
condition strip above the columns, the lipid-class column beside the rows,
and the square-cell sizing of the class modes.

Pixel geometry lives here so LipidomicHeatmapPlotterService reads as data
steps. Pure logic — no Streamlit dependencies.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from app.services.plotting._shared import BASE_LAYOUT, generate_condition_color_mapping


# ── Constants ──────────────────────────────────────────────────────────

COLORSCALE = 'RdBu_r'

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

# Inside the class column: room kept at its outer edge for the y-axis title,
# and how far the block lines reach into the gap before the species names.
CLASS_TITLE_PX = 22

CLASS_LINE_CLEARANCE_PX = 30

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


# ── Figure helpers ─────────────────────────────────────────────────────


def _titled(base: str, value_label: str) -> str:
    """Figure title, naming the quantity only when it is not the default.

    Keeps the Clustered and Regular titles byte-identical under Z-scores.
    """
    return base if value_label == 'Z-score' else f'{base} ({value_label})'


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


def _unique_labels(labels: List[str]) -> List[str]:
    """Make repeated labels distinct with invisible zero-width suffixes.

    A categorical axis merges equal labels into one row, which would shift
    every row below it; the suffix keeps them apart without showing.
    """
    seen: Dict[str, int] = {}
    unique = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label + '\u200b' * count)
    return unique


def _add_class_column(
    fig: go.Figure,
    classes: List[str],
    species_px: int,
) -> None:
    """Label each lipid class block in a column left of the species names.

    The class name is centred on its block, and a solid line marks each block
    edge across the class column only, stopping short of the species names so
    no name is ever struck through. Everything is placed in pixels from the
    plot's left edge, so it holds when Streamlit narrows or full-screens the
    figure. The y-axis title moves to the outer edge, clear of the class names.

    Args:
        fig: Class-grouped heatmap figure, one row per species.
        classes: Class of each row, in row order (blocks contiguous).
        species_px: Room the species names take left of the plot, in px.
    """
    blocks = _condition_blocks(classes)
    column_left = -(species_px + CLASS_LABEL_WIDTH) + CLASS_TITLE_PX
    column_right = -(species_px - CLASS_LINE_CLEARANCE_PX)

    edges = [start - 0.5 for _, start, _ in blocks] + [blocks[-1][2] + 0.5]
    for edge in edges:
        fig.add_shape(
            type='line',
            xref='paper', xsizemode='pixel', xanchor=0,
            x0=column_left, x1=column_right,
            yref='y', y0=edge, y1=edge,
            line=BLOCK_LINE_STYLE,
        )
    for name, start, end in blocks:
        fig.add_annotation(
            xref='paper', x=0, xanchor='center',
            xshift=(column_left + column_right) / 2,
            yref='y', y=(start + end) / 2,
            text=name, showarrow=False,
            font=dict(size=12, color='black'),
        )

    fig.update_yaxes(title_text=None)
    fig.add_annotation(
        xref='paper', x=0, xanchor='left',
        xshift=-(species_px + CLASS_LABEL_WIDTH),
        yref='paper', y=0.5, yanchor='middle',
        text='Lipid Molecules', textangle=-90, showarrow=False,
        font=dict(size=14),
    )


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
        **BASE_LAYOUT,
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
