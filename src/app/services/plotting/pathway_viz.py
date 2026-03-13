"""
PathwayVizPlotterService — Lipid pathway visualization.

Pure-logic service (no Streamlit). Computes class-level saturation ratios
and fold changes, then renders a Matplotlib pathway diagram with 18 fixed
lipid class positions.  Circle size encodes abundance fold change; color
encodes saturated fatty acid ratio.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.models.experiment import ExperimentConfig


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

# The 18 lipid classes displayed on the pathway, in fixed order.
PATHWAY_CLASSES: List[str] = [
    'TG', 'DG', 'PA', 'LPA', 'LCB', 'Cer', 'SM',
    'PE', 'LPE', 'PC', 'LPC', 'PI', 'LPI',
    'CDP-DAG', 'PG', 'LPG', 'PS', 'LPS',
]

# Fixed (x, y) coordinates for each class on the pathway diagram.
_COS30 = math.cos(math.pi / 6)
_SIN30 = math.sin(math.pi / 6)
_COS45 = math.cos(math.pi / 4)
_SIN45 = math.sin(math.pi / 4)

PATHWAY_COORDS: List[Tuple[float, float]] = [
    (0, 0),                                  # TG
    (0, 5),                                  # DG
    (0, 10),                                 # PA
    (0, 15),                                 # LPA
    (-10, 15),                               # LCB
    (-10, 10),                               # Cer
    (-10, 5),                                # SM
    (5 * _COS30, -5 * _SIN30),              # PE
    (10 * _COS30, -10 * _SIN30),            # LPE
    (-5 * _COS30, -5 * _SIN30),             # PC
    (-10 * _COS30, -10 * _SIN30),           # LPC
    (10, 15),                                # PI
    (10 + 5 * _COS45, 15 + 5 * _SIN45),    # LPI
    (5, 10),                                 # CDP-DAG
    (10, 10),                                # PG
    (15, 10),                                # LPG
    (10, 5),                                 # PS
    (10 + 5 * _COS45, 5 - 5 * _SIN45),     # LPS
]

# Unit circle radius (reference for fold-change = 1).
UNIT_CIRCLE_RADIUS = 0.5

# Fold-change size scaling factor.
SIZE_SCALE = 50

# Figure dimensions.
FIG_SIZE = (10, 10)
PLOT_XLIM = (-25, 25)
PLOT_YLIM = (-20, 30)

# Hydroxyl-notation pattern used when counting sat/unsat chains.
_HYDROXYL_PATTERN = re.compile(r';[\dO()]+')


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
        content = mol_structure.split('(')[1].rstrip(')')
        chains = content.split('_')
        n_sat = 0
        n_unsat = 0
        for chain in chains:
            db_str = chain.split(':')[-1]
            db_str = _HYDROXYL_PATTERN.sub('', db_str)
            if db_str == '0':
                n_sat += 1
            else:
                n_unsat += 1
        return n_sat, n_unsat
    except (IndexError, ValueError, AttributeError):
        return 0, 0


def _initiate_plot() -> Tuple[plt.Figure, plt.Axes]:
    """Create the base pathway figure with axes styling."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_title('Lipid Pathway Visualization', fontsize=20)
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    return fig, ax


def _draw_grouping_circles(ax: plt.Axes) -> None:
    """Draw the 6 large blue grouping circles."""
    circle_data = [
        (5, 0, 0),
        (2.5, -7.5 * _COS30, -7.5 * math.cos(math.pi / 3)),
        (2.5, 7.5 * _COS30, -7.5 * math.cos(math.pi / 3)),
        (2.5, 10 + 2.5 * _COS45, 15 + 2.5 * _SIN45),
        (2.5, 12.5, 10),
        (2.5, 10 + 2.5 * _COS45, 5 - 2.5 * _SIN45),
    ]
    for radius, x0, y0 in circle_data:
        ax.add_patch(plt.Circle((x0, y0), radius, color='b', fill=False))


def _draw_unit_circles(ax: plt.Axes) -> None:
    """Draw small black reference circles (fold-change = 1) for each class."""
    for x, y in PATHWAY_COORDS:
        ax.add_patch(plt.Circle((x, y), UNIT_CIRCLE_RADIUS, color='black', fill=False))


def _draw_connecting_lines(ax: plt.Axes) -> None:
    """Draw blue connector lines between pathway nodes."""
    line_data = [
        ([0, 0], [0, 20]),
        ([0, -5], [15, 20]),
        ([-5, -10], [20, 15]),
        ([-10, -10], [15, 5]),
        ([0, 5], [10, 10]),
        ([5, 10], [10, 15]),
        ([5, 10], [10, 10]),
        ([5, 10], [10, 5]),
        ([5 * _COS30, 10], [-5 * _SIN30, 5]),
    ]
    for x, y in line_data:
        ax.plot(x, y, c='b')


def _add_text_labels(ax: plt.Axes) -> None:
    """Add lipid class text labels and annotations."""
    ax.annotate('G3P', xy=(0, 20), xytext=(0, 23),
                arrowprops=dict(facecolor='black'), fontsize=15)
    ax.annotate('Fatty Acids', xy=(-5, 20), xytext=(-5, 25),
                arrowprops=dict(facecolor='black'), fontsize=15)
    text_items = [
        (-3.5, 14, 'LPA'), (-2.5, 9.5, 'PA'), (-4, 5.5, 'DAG'),
        (-4, 0.5, 'TAG'), (-4, -2, 'PC'), (-12, -6.5, 'LPC'),
        (2.5, -2, 'PE'), (9, -6, 'LPE'), (-14, 15, 'LCBs'),
        (-13.5, 10, 'Cer'), (-13, 5, 'SM'), (2, 11, 'CDP-DAG'),
        (10.5, 15.5, 'PI'), (14, 19, 'LPI'), (10.5, 9.5, 'PG'),
        (15.5, 9.5, 'LPG'), (10.5, 4, 'PS'), (14.5, 1, 'LPS'),
    ]
    for x, y, label in text_items:
        ax.text(x, y, label, fontsize=15)


def _render_scatter(ax: plt.Axes, fig: plt.Figure,
                    colors: List[float], sizes: List[float]) -> None:
    """Render the data scatter on the pathway with colorbar."""
    xs = [c[0] for c in PATHWAY_COORDS]
    ys = [c[1] for c in PATHWAY_COORDS]
    points = ax.scatter(xs, ys, c=colors, s=sizes, cmap='plasma')
    cbar = fig.colorbar(points)
    cbar.set_label(label='Saturation Ratio', size=15)
    cbar.ax.tick_params(labelsize=15)


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
    ) -> Dict[str, list]:
        """Build the pathway data dictionary for the 18 fixed classes.

        Classes not present in the data receive 0 for both abundance ratio
        and saturation ratio.

        Args:
            fold_change_df: DataFrame with ``ClassKey`` and ``fold_change``.
            saturation_ratio_df: DataFrame with ``ClassKey`` and
                ``saturation_ratio``.

        Returns:
            Dict with keys ``'class'``, ``'abundance ratio'``, and
            ``'saturated fatty acids ratio'``.
        """
        if fold_change_df.empty and saturation_ratio_df.empty:
            return {
                'class': list(PATHWAY_CLASSES),
                'abundance ratio': [0] * len(PATHWAY_CLASSES),
                'saturated fatty acids ratio': [0] * len(PATHWAY_CLASSES),
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
            'class': list(PATHWAY_CLASSES),
            'abundance ratio': [
                fc_lookup.get(cls, 0) for cls in PATHWAY_CLASSES
            ],
            'saturated fatty acids ratio': [
                sat_lookup.get(cls, 0) for cls in PATHWAY_CLASSES
            ],
        }

    @staticmethod
    def create_pathway_viz(
        fold_change_df: pd.DataFrame,
        saturation_ratio_df: pd.DataFrame,
    ) -> Tuple[Optional[plt.Figure], Dict[str, list]]:
        """Render the full lipid pathway Matplotlib figure.

        Args:
            fold_change_df: DataFrame with ``ClassKey`` and ``fold_change``.
            saturation_ratio_df: DataFrame with ``ClassKey`` and
                ``saturation_ratio``.

        Returns:
            (figure, pathway_dict) or (None, {}) when both inputs are empty.
        """
        if fold_change_df.empty and saturation_ratio_df.empty:
            return None, {}

        pathway_dict = PathwayVizPlotterService.create_pathway_dictionary(
            fold_change_df, saturation_ratio_df,
        )

        colors = pathway_dict['saturated fatty acids ratio']
        sizes = [SIZE_SCALE * v for v in pathway_dict['abundance ratio']]

        fig, ax = _initiate_plot()
        _draw_grouping_circles(ax)
        _draw_connecting_lines(ax)
        _add_text_labels(ax)
        _render_scatter(ax, fig, colors, sizes)
        _draw_unit_circles(ax)

        return fig, pathway_dict
