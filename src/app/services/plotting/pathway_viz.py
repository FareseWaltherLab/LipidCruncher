"""
PathwayVizPlotterService — Lipid pathway visualization.

Pure-logic service (no Streamlit). Computes class-level saturation ratios
and fold changes, then renders an interactive Plotly pathway diagram.
Circle size encodes abundance fold change (log2-scaled); color encodes
saturated fatty acid ratio.

The pathway provides 28 curated lipid classes with 18 shown by default.
Users can toggle classes on/off and add/remove nodes and edges.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.constants import HYDROXYL_PATTERN, parse_lipid_name


# ═══════════════════════════════════════════════════════════════════════
# Pathway Layout Constants
# ═══════════════════════════════════════════════════════════════════════

_COS30 = math.cos(math.pi / 6)
_SIN30 = math.sin(math.pi / 6)
_COS45 = math.cos(math.pi / 4)
_SIN45 = math.sin(math.pi / 4)

# Each node: (x, y, display_label, label_x, label_y)
# label_x/label_y are absolute positions for the text label.
ALL_PATHWAY_NODES: Dict[str, Tuple[float, float, str, float, float]] = {
    # --- Default 18 classes (always available) ---
    'TG':      (0, 0, 'TAG', -2.5, -1.5),
    'DG':      (0, 5, 'DAG', -4, 5.5),
    'PA':      (0, 10, 'PA', -3, 9.5),
    'LPA':     (0, 15, 'LPA', -2.5, 14),
    'LCB':     (-10, 15, 'LCBs', -13, 15),
    'Cer':     (-10, 10, 'Cer', -12.5, 10),
    'SM':      (-10, 5, 'SM', -13, 4),
    'PE':      (5 * _COS30, -5 * _SIN30, 'PE', 2.5, -1.5),
    'LPE':     (10 * _COS30, -10 * _SIN30, 'LPE', 9.5, -6),
    'PC':      (-5 * _COS30, -5 * _SIN30, 'PC', -4, -2),
    'LPC':     (-10 * _COS30, -10 * _SIN30, 'LPC', -12.5, -6.5),
    'PI':      (10, 15, 'PI', 11.5, 15.5),
    'LPI':     (10 + 5 * _COS45, 15 + 5 * _SIN45, 'LPI', 14.5, 19),
    'CDP-DAG': (5, 10, 'CDP-DAG', 2, 11.5),
    'PG':      (10, 10, 'PG', 11, 9.5),
    'LPG':     (15, 10, 'LPG', 16, 9.5),
    'PS':      (10, 5, 'PS', 11.5, 4),
    'LPS':     (10 + 5 * _COS45, 5 - 5 * _SIN45, 'LPS', 14.5, 1),
    # --- Additional curated classes (available but not shown by default) ---
    'MAG':     (0, -5, 'MAG', -1.5, -7),
    'dhCer':   (-10, 12.5, 'dhCer', -9, 11.5),
    'CerP':    (-14, 12, 'CerP', -18, 12),
    'HexCer':  (-14, 2, 'HexCer', -20, 3),
    'Hex2Cer': (-18, -1, 'Hex2Cer', -24.5, 0),
    'Hex3Cer': (-22, -4, 'Hex3Cer', -24, -6),
    'ePC':     (-10, 18, 'ePC', -12.5, 18),
    'ePE':     (-2, 19, 'ePE', -4.5, 20.5),
    'CL':      (17, 13, 'CL', 18.5, 13),
    'CE':      (-5, 16, 'CE', -7.5, 16),
}

# All metabolic edges between curated classes.
# During rendering, only edges where BOTH endpoints are active are drawn.
ALL_PATHWAY_EDGES: List[Tuple[str, str]] = [
    # Central glycerolipid axis
    ('TG', 'DG'),
    ('DG', 'PA'),
    ('PA', 'LPA'),
    # Sphingolipid branch
    ('LCB', 'Cer'),
    ('Cer', 'SM'),
    # CDP-DAG phospholipid hub
    ('PA', 'CDP-DAG'),
    ('CDP-DAG', 'PI'),
    ('CDP-DAG', 'PG'),
    ('CDP-DAG', 'PS'),
    # Kennedy pathway (DG → phospholipids)
    ('DG', 'PC'),
    ('DG', 'PE'),
    # Interconversion
    ('PE', 'PS'),
    # Lyso forms (phospholipase activity)
    ('PC', 'LPC'),
    ('PE', 'LPE'),
    ('PI', 'LPI'),
    ('PG', 'LPG'),
    ('PS', 'LPS'),
    # --- Edges involving additional classes ---
    ('DG', 'MAG'),
    ('LCB', 'dhCer'),
    ('dhCer', 'Cer'),
    ('Cer', 'CerP'),
    ('Cer', 'HexCer'),
    ('HexCer', 'Hex2Cer'),
    ('Hex2Cer', 'Hex3Cer'),
    ('PG', 'CL'),
]

# The 18 classes shown by default.
DEFAULT_PATHWAY_CLASSES: List[str] = [
    'TG', 'DG', 'PA', 'LPA', 'LCB', 'Cer', 'SM',
    'PE', 'LPE', 'PC', 'LPC', 'PI', 'LPI',
    'CDP-DAG', 'PG', 'LPG', 'PS', 'LPS',
]

# Backward-compatibility aliases.
PATHWAY_CLASSES: List[str] = list(DEFAULT_PATHWAY_CLASSES)
PATHWAY_COORDS: List[Tuple[float, float]] = [
    (ALL_PATHWAY_NODES[c][0], ALL_PATHWAY_NODES[c][1])
    for c in DEFAULT_PATHWAY_CLASSES
]


# ═══════════════════════════════════════════════════════════════════════
# Visualization Constants
# ═══════════════════════════════════════════════════════════════════════

UNIT_CIRCLE_RADIUS = 0.5
SIZE_SCALE = 50
FIG_SIZE = (10, 10)
PLOT_XLIM = (-25, 25)
PLOT_YLIM = (-20, 30)

# Log2-scaling bounds for fold-change circle sizes.
MIN_LOG2_SIZE = 0.2
MAX_LOG2_SIZE = 5.0


# ═══════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PathwayData:
    """Result container for pathway visualization data.

    Attributes:
        pathway_dict: Dict with keys 'class', 'abundance ratio',
            'saturated fatty acids ratio'.
        fold_change_df: Per-class fold-change DataFrame.
        saturation_ratio_df: Per-class saturation ratio DataFrame.
    """
    pathway_dict: Dict[str, list] = field(default_factory=dict)
    fold_change_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    saturation_ratio_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════


def _count_saturated_unsaturated(mol_structure: str) -> Tuple[int, int]:
    """Count saturated and unsaturated fatty acid chains in a lipid name.

    A chain with 0 double bonds is saturated; any other count is unsaturated.

    Returns:
        (n_saturated, n_unsaturated).  (0, 0) if parsing fails.
    """
    try:
        _, chain_info, _ = parse_lipid_name(mol_structure)
        if not chain_info:
            return 0, 0
        chains = re.split(r'[/_]', chain_info)
        n_sat = 0
        n_unsat = 0
        for chain in chains:
            db_str = chain.split(':')[-1]
            db_str = HYDROXYL_PATTERN.sub('', db_str)
            if db_str == '0':
                n_sat += 1
            else:
                n_unsat += 1
        return n_sat, n_unsat
    except (IndexError, ValueError, AttributeError):
        return 0, 0


def _scale_fold_change(fc: float) -> float:
    """Log2-scale a fold-change value for circle sizing.

    Returns 0 for fc <= 0 (class absent from data).
    Otherwise clamps log2(fc + 1) to [MIN_LOG2_SIZE, MAX_LOG2_SIZE].
    """
    if fc <= 0:
        return 0
    return max(MIN_LOG2_SIZE, min(MAX_LOG2_SIZE, math.log2(fc + 1)))


def _auto_label_pos(
    x: float, y: float, label: str,
) -> Tuple[float, float]:
    """Compute a reasonable label position for a node.

    Places the label offset from the node center so it doesn't overlap
    with the circle.  Left-side nodes get labels to the left; right-side
    nodes get labels to the right.  A vertical offset is added so the
    label sits clearly above/below the node center.
    """
    if x <= 0:
        lx = x - len(label) * 0.6 - 1.5
    else:
        lx = x + 1.5
    ly = y - 1.5
    return lx, ly


def _resolve_nodes(
    active_classes: List[str],
    custom_nodes: Optional[Dict[str, Tuple[float, float]]] = None,
    position_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Tuple[float, float, str, float, float]]:
    """Build the active node dict from curated + custom nodes.

    Args:
        active_classes: Which classes to include.
        custom_nodes: User-added nodes not in ALL_PATHWAY_NODES.
        position_overrides: Overridden (x, y) for curated or custom nodes.

    Returns:
        Ordered dict mapping class name → (x, y, label, label_x, label_y).
        Order follows ``active_classes``.
    """
    nodes: Dict[str, Tuple[float, float, str, float, float]] = {}
    for cls in active_classes:
        if cls in ALL_PATHWAY_NODES:
            x, y, label, lx, ly = ALL_PATHWAY_NODES[cls]
            if position_overrides and cls in position_overrides:
                x, y = position_overrides[cls]
                lx, ly = _auto_label_pos(x, y, label)
            nodes[cls] = (x, y, label, lx, ly)
        elif custom_nodes and cls in custom_nodes:
            x, y = custom_nodes[cls]
            if position_overrides and cls in position_overrides:
                x, y = position_overrides[cls]
            lx, ly = _auto_label_pos(x, y, cls)
            nodes[cls] = (x, y, cls, lx, ly)
    return nodes


def _resolve_edges(
    active_set: Set[str],
    added_edges: Optional[List[Tuple[str, str]]] = None,
    removed_edges: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    """Compute the effective edge list.

    Starts with ``ALL_PATHWAY_EDGES``, adds user-added edges, removes
    user-removed edges, and filters to edges where both endpoints are active.
    """
    removed: Set[Tuple[str, str]] = set()
    if removed_edges:
        for a, b in removed_edges:
            removed.add((a, b))
            removed.add((b, a))

    edges: List[Tuple[str, str]] = []
    for a, b in ALL_PATHWAY_EDGES:
        if (a, b) not in removed and (b, a) not in removed:
            if a in active_set and b in active_set:
                edges.append((a, b))

    if added_edges:
        for a, b in added_edges:
            if a in active_set and b in active_set:
                edges.append((a, b))

    return edges


# ═══════════════════════════════════════════════════════════════════════
# Plotly circle shape helper
# ═══════════════════════════════════════════════════════════════════════

_CIRCLE_POINTS = 64  # number of points to approximate a circle


def _circle_path(cx: float, cy: float, r: float) -> str:
    """Return an SVG path string for a circle centered at (cx, cy)."""
    angles = np.linspace(0, 2 * np.pi, _CIRCLE_POINTS, endpoint=True)
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    parts = [f"M {xs[0]:.4f},{ys[0]:.4f}"]
    for x, y in zip(xs[1:], ys[1:]):
        parts.append(f"L {x:.4f},{y:.4f}")
    parts.append("Z")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════


class PathwayVizPlotterService:
    """Pure-logic service for lipid pathway visualization."""

    @staticmethod
    def calculate_class_saturation_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """Compute the saturated-chain ratio for each lipid class.

        For each lipid species the number of saturated (0 double bonds) and
        unsaturated chains is counted.  These counts are summed per ClassKey
        and the ratio ``n_sat / (n_sat + n_unsat)`` is returned.

        Args:
            df: DataFrame with ``LipidMolec`` and ``ClassKey`` columns.

        Returns:
            DataFrame with columns ``ClassKey`` and ``saturation_ratio``.

        Raises:
            ValueError: If required columns are missing.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=['ClassKey', 'saturation_ratio'])

        for col in ('LipidMolec', 'ClassKey'):
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")

        df_work = df.copy()
        counts = df_work['LipidMolec'].apply(_count_saturated_unsaturated)
        df_work['_n_sat'] = counts.apply(lambda t: t[0])
        df_work['_n_unsat'] = counts.apply(lambda t: t[1])

        grouped = df_work.groupby('ClassKey')[['_n_sat', '_n_unsat']].sum()
        total = grouped['_n_sat'] + grouped['_n_unsat']
        grouped['saturation_ratio'] = np.where(
            total > 0, grouped['_n_sat'] / total, 0.0,
        )
        return grouped[['saturation_ratio']].reset_index()

    @staticmethod
    def calculate_class_fold_change(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        control: str,
        experimental: str,
    ) -> pd.DataFrame:
        """Compute per-class abundance fold change (experimental / control).

        Sums concentration columns per ClassKey for each condition, then
        computes ``mean(experimental) / mean(control)``.

        Args:
            df: DataFrame with ``ClassKey`` and ``concentration[sample]`` cols.
            experiment: Experiment configuration.
            control: Name of the control condition.
            experimental: Name of the experimental condition.

        Returns:
            DataFrame with columns ``ClassKey`` and ``fold_change``.

        Raises:
            ValueError: If conditions are not found in the experiment or
                if no valid concentration columns exist.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=['ClassKey', 'fold_change'])

        if 'ClassKey' not in df.columns:
            raise ValueError("DataFrame must contain 'ClassKey' column")

        if control not in experiment.conditions_list:
            raise ValueError(f"Control condition '{control}' not in experiment")
        if experimental not in experiment.conditions_list:
            raise ValueError(f"Experimental condition '{experimental}' not in experiment")

        ctrl_idx = experiment.conditions_list.index(control)
        exp_idx = experiment.conditions_list.index(experimental)
        ctrl_samples = [
            s for s in experiment.individual_samples_list[ctrl_idx]
            if f'concentration[{s}]' in df.columns
        ]
        exp_samples = [
            s for s in experiment.individual_samples_list[exp_idx]
            if f'concentration[{s}]' in df.columns
        ]

        if not ctrl_samples or not exp_samples:
            raise ValueError("No valid concentration columns for the specified conditions")

        all_samples = ctrl_samples + exp_samples
        conc_cols = [f'concentration[{s}]' for s in all_samples]
        abundance = df.groupby('ClassKey')[conc_cols].sum()

        ctrl_cols = [f'concentration[{s}]' for s in ctrl_samples]
        exp_cols = [f'concentration[{s}]' for s in exp_samples]
        ctrl_mean = abundance[ctrl_cols].mean(axis=1)
        exp_mean = abundance[exp_cols].mean(axis=1)

        fold_change = np.where(ctrl_mean > 0, exp_mean / ctrl_mean, 0.0)
        result = pd.DataFrame({
            'ClassKey': abundance.index,
            'fold_change': fold_change,
        }).reset_index(drop=True)
        return result

    @staticmethod
    def create_pathway_dictionary(
        fold_change_df: pd.DataFrame,
        saturation_ratio_df: pd.DataFrame,
        active_classes: Optional[List[str]] = None,
    ) -> Dict[str, list]:
        """Build the pathway data dictionary for the active classes.

        Classes not present in the data receive 0 for both abundance ratio
        and saturation ratio.

        Args:
            fold_change_df: DataFrame with ``ClassKey`` and ``fold_change``.
            saturation_ratio_df: DataFrame with ``ClassKey`` and
                ``saturation_ratio``.
            active_classes: Which classes to include (default:
                DEFAULT_PATHWAY_CLASSES).

        Returns:
            Dict with keys ``'class'``, ``'abundance ratio'``, and
            ``'saturated fatty acids ratio'``.
        """
        if active_classes is None:
            active_classes = DEFAULT_PATHWAY_CLASSES

        if fold_change_df.empty and saturation_ratio_df.empty:
            return {
                'class': list(active_classes),
                'abundance ratio': [0] * len(active_classes),
                'saturated fatty acids ratio': [0] * len(active_classes),
            }

        fc_lookup: Dict[str, float] = {}
        if not fold_change_df.empty and 'ClassKey' in fold_change_df.columns:
            fc_lookup = dict(zip(
                fold_change_df['ClassKey'], fold_change_df['fold_change'],
            ))

        sat_lookup: Dict[str, float] = {}
        if not saturation_ratio_df.empty and 'ClassKey' in saturation_ratio_df.columns:
            sat_lookup = dict(zip(
                saturation_ratio_df['ClassKey'],
                saturation_ratio_df['saturation_ratio'],
            ))

        return {
            'class': list(active_classes),
            'abundance ratio': [
                fc_lookup.get(cls, 0) for cls in active_classes
            ],
            'saturated fatty acids ratio': [
                sat_lookup.get(cls, 0) for cls in active_classes
            ],
        }

    @staticmethod
    def create_pathway_viz(
        fold_change_df: pd.DataFrame,
        saturation_ratio_df: pd.DataFrame,
        active_classes: Optional[List[str]] = None,
        custom_nodes: Optional[Dict[str, Tuple[float, float]]] = None,
        added_edges: Optional[List[Tuple[str, str]]] = None,
        removed_edges: Optional[List[Tuple[str, str]]] = None,
        position_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
        show_grid: bool = False,
    ) -> Tuple[Optional[go.Figure], Dict[str, list]]:
        """Render the full lipid pathway as an interactive Plotly figure.

        Args:
            fold_change_df: DataFrame with ``ClassKey`` and ``fold_change``.
            saturation_ratio_df: DataFrame with ``ClassKey`` and
                ``saturation_ratio``.
            active_classes: Classes to display (default:
                DEFAULT_PATHWAY_CLASSES).
            custom_nodes: User-added nodes not in ALL_PATHWAY_NODES:
                ``{name: (x, y)}``.
            added_edges: User-added edges: ``[(source, target), ...]``.
            removed_edges: Edges to exclude from ALL_PATHWAY_EDGES.
            position_overrides: Overridden ``(x, y)`` positions for any
                node (curated or custom): ``{name: (x, y)}``.
            show_grid: If True, draw a coordinate grid overlay.

        Returns:
            ``(figure, pathway_dict)`` or ``(None, {})`` when both inputs
            are empty.
        """
        if active_classes is None:
            active_classes = list(DEFAULT_PATHWAY_CLASSES)

        if fold_change_df.empty and saturation_ratio_df.empty:
            return None, {}

        # Resolve layout
        nodes = _resolve_nodes(
            active_classes, custom_nodes, position_overrides,
        )
        resolved_classes = list(nodes.keys())
        active_set = set(resolved_classes)
        edges = _resolve_edges(active_set, added_edges, removed_edges)

        # Build pathway dictionary for resolved classes
        pathway_dict = PathwayVizPlotterService.create_pathway_dictionary(
            fold_change_df, saturation_ratio_df, resolved_classes,
        )

        # Determine which classes have data
        classes_with_data: Set[str] = set()
        if not fold_change_df.empty and 'ClassKey' in fold_change_df.columns:
            classes_with_data = set(
                fold_change_df.loc[
                    fold_change_df['fold_change'] > 0, 'ClassKey'
                ]
            )

        # Compute log2-scaled sizes (in coordinate units for marker radius)
        sat_values = pathway_dict['saturated fatty acids ratio']
        fc_values = pathway_dict['abundance ratio']
        scaled_sizes = [_scale_fold_change(v) for v in fc_values]

        # Dynamic colorbar max: use actual max saturation ratio (floor at 1)
        sat_max = max(sat_values) if sat_values else 1.0
        if sat_max <= 0:
            sat_max = 1.0

        # --- Build Plotly figure ---
        fig = go.Figure()

        # 1) Edges (blue connector lines)
        for a, b in edges:
            if a in nodes and b in nodes:
                fig.add_trace(go.Scatter(
                    x=[nodes[a][0], nodes[b][0]],
                    y=[nodes[a][1], nodes[b][1]],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    hoverinfo='skip',
                    showlegend=False,
                ))

        # 2) Annotation lines and labels (G3P, Fatty Acids)
        #    Arrow points FROM text (ax, ay offset in px) TO (x, y).
        #    Negative ay = text above the anchor → arrow points downward.
        if 'LPA' in active_set:
            fig.add_trace(go.Scatter(
                x=[0, 0], y=[15, 20], mode='lines',
                line=dict(color='blue', width=1),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=0, y=20, text='G3P', showarrow=True,
                arrowhead=2, arrowcolor='black', ax=0, ay=-30,
                font=dict(size=15),
            )
        if 'LCB' in active_set:
            fig.add_trace(go.Scatter(
                x=[-5, -10], y=[20, 15], mode='lines',
                line=dict(color='blue', width=1),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=-5, y=20, text='Fatty Acids', showarrow=True,
                arrowhead=2, arrowcolor='black', ax=0, ay=-30,
                font=dict(size=15),
            )
            # Fatty acid supply edge to LPA
            if 'LPA' in active_set and 'LPA' in nodes:
                lpa_x, lpa_y = nodes['LPA'][0], nodes['LPA'][1]
                fig.add_trace(go.Scatter(
                    x=[-5, lpa_x], y=[20, lpa_y], mode='lines',
                    line=dict(color='blue', width=1),
                    hoverinfo='skip', showlegend=False,
                ))
            # Fatty acid supply edges to ether lipids and cholesteryl ester
            fa_x, fa_y = -5, 20
            for target_cls in ('ePC', 'ePE', 'CE'):
                if target_cls in active_set and target_cls in nodes:
                    tx, ty = nodes[target_cls][0], nodes[target_cls][1]
                    fig.add_trace(go.Scatter(
                        x=[fa_x, tx], y=[fa_y, ty], mode='lines',
                        line=dict(color='blue', width=1),
                        hoverinfo='skip', showlegend=False,
                    ))

        # 3) Unit circles and missing-class circles (as shapes)
        for cls, (x, y, *_rest) in nodes.items():
            # Unit circle (always drawn)
            fig.add_shape(
                type='path',
                path=_circle_path(x, y, UNIT_CIRCLE_RADIUS),
                line=dict(color='black', width=1.2),
                fillcolor='rgba(0,0,0,0)',
            )
            # Missing-class dashed circle
            if cls not in classes_with_data:
                fig.add_shape(
                    type='path',
                    path=_circle_path(x, y, UNIT_CIRCLE_RADIUS * 1.5),
                    line=dict(color='gray', width=1.5, dash='dash'),
                    fillcolor='rgba(0,0,0,0)',
                )

        # 4) Data scatter — main visualization with hover
        xs = [nodes[c][0] for c in resolved_classes]
        ys = [nodes[c][1] for c in resolved_classes]
        # Convert coordinate-space sizes to Plotly marker sizes (pixels).
        # SIZE_SCALE acts as the pixel multiplier.
        marker_sizes = [SIZE_SCALE * s / 5.0 for s in scaled_sizes]

        # Build hover text
        hover_texts = []
        for i, cls in enumerate(resolved_classes):
            fc_val = fc_values[i]
            sat_val = sat_values[i]
            n_species = 0
            if not fold_change_df.empty and 'ClassKey' in fold_change_df.columns:
                n_species = int(
                    (fold_change_df['ClassKey'] == cls).sum()
                )
            hover_texts.append(
                f"<b>{cls}</b><br>"
                f"Fold Change: {fc_val:.3f}<br>"
                f"Saturation Ratio: {sat_val:.3f}<br>"
                f"Species Detected: {n_species}"
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=sat_values,
                colorscale='Plasma',
                cmin=0, cmax=sat_max,
                colorbar=dict(
                    title=dict(text='Saturation Ratio', font=dict(size=15)),
                    tickfont=dict(size=13),
                ),
                line=dict(width=0),
            ),
            text=hover_texts,
            hoverinfo='text',
            showlegend=False,
        ))

        # 5) Text labels
        for cls, (_, _, label, lx, ly) in nodes.items():
            color = 'black' if cls in classes_with_data else 'gray'
            fig.add_annotation(
                x=lx, y=ly, text=label, showarrow=False,
                font=dict(size=14, color=color),
                xanchor='left', yanchor='middle',
            )

        # Auto-fit axis ranges to node positions with padding
        all_xs = [nodes[c][0] for c in resolved_classes]
        all_ys = [nodes[c][1] for c in resolved_classes]
        # Include annotation target points if present
        if 'LPA' in active_set:
            all_ys.append(20)
        if 'LCB' in active_set:
            all_xs.append(-5)
            all_ys.append(20)
        # Include label positions for tighter fitting
        for cls in resolved_classes:
            _, _, _, lx, ly = nodes[cls]
            all_xs.append(lx)
            all_ys.append(ly)

        pad = 4  # coordinate units of padding
        x_min = min(all_xs) - pad
        x_max = max(all_xs) + pad
        y_min = min(all_ys) - pad
        y_max = max(all_ys) + pad

        # Layout
        fig.update_layout(
            title=dict(text='Lipid Pathway Visualization', font=dict(size=20)),
            xaxis=dict(
                range=[x_min, x_max],
                scaleanchor='y', scaleratio=1,
                showgrid=show_grid,
                zeroline=False,
                visible=show_grid,
                dtick=5 if show_grid else None,
                gridcolor='#cccccc' if show_grid else None,
                gridwidth=0.5 if show_grid else None,
            ),
            yaxis=dict(
                range=[y_min, y_max],
                showgrid=show_grid,
                zeroline=False,
                visible=show_grid,
                dtick=5 if show_grid else None,
                gridcolor='#cccccc' if show_grid else None,
                gridwidth=0.5 if show_grid else None,
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=800,
            height=800,
            margin=dict(l=10, r=10, t=60, b=10),
        )

        return fig, pathway_dict
