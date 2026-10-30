"""
Isotopologue distribution plot: IsoCorrectoR's corrected fractions as
stacked bars, one bar per sample, grouped by condition, stacked by
isotopologue (_0, _1, _2, ...).

Conditions come from the sample names: a trailing ``_<number>`` marks a
replicate, so ``WT_13C_1``, ``WT_13C_2`` and ``WT_13C_3`` are three replicates
of ``WT_13C``.

Pure logic — no Streamlit dependencies.
"""

import re
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from app.services.plotting._shared import BASE_LAYOUT, frame_axes

_REPLICATE = re.compile(r'^(.+)_(\d+)$')

# Lightest and darkest point of the Purples scale the isotopologues span:
# _0 lightest, the heaviest darkest.
_LIGHTEST, _DARKEST = 0.15, 1.0


class IsotopologuePlotterService:
    """Stacked isotopologue-distribution bars from corrected fractions."""

    @staticmethod
    def split_id(isotopologue_id: str) -> Tuple[str, str]:
        """``BMP_22:6_22:6_3`` -> (``BMP_22:6_22:6``, ``3``)."""
        molecule, _, label = str(isotopologue_id).rpartition('_')
        return molecule, label

    @staticmethod
    def sample_groups(samples: List[str]) -> Dict[str, List[str]]:
        """Condition -> its samples, read from the names.

        A name ending in ``_<number>`` is that replicate of the condition
        before it; any other name is a condition of its own. Conditions keep
        the order they first appear in; replicates are sorted by number.
        """
        groups: Dict[str, List[Tuple[int, str]]] = {}
        for sample in samples:
            match = _REPLICATE.match(sample)
            condition, replicate = (match.group(1), int(match.group(2))) if match else (sample, 0)
            groups.setdefault(condition, []).append((replicate, sample))
        return {c: [s for _, s in sorted(reps)] for c, reps in groups.items()}

    @staticmethod
    def molecules(fractions: pd.DataFrame) -> List[str]:
        """The molecules in the table, in order."""
        return list(dict.fromkeys(
            IsotopologuePlotterService.split_id(i)[0] for i in fractions.index
        ))

    @staticmethod
    def labels(fractions: pd.DataFrame) -> List[str]:
        """Every isotopologue label in the table (``0``, ``1``, ... or ``C0``,
        ``C1``, ...), in the order they first appear."""
        return list(dict.fromkeys(
            IsotopologuePlotterService.split_id(i)[1] for i in fractions.index
        ))

    @staticmethod
    def create_distribution_plot(
        fractions: pd.DataFrame,
        molecule: str,
        groups: Dict[str, List[str]],
        labels: List[str],
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Stacked bars of one molecule's isotopologue fractions.

        Args:
            fractions: IsoCorrectoR's Corrected Fractions (isotopologue IDs ×
                samples).
            molecule: The molecule to plot.
            groups: Condition -> samples, in plot order.
            labels: The isotopologue labels to stack. Unselected ones are
                left out, not shared out, so the bars then sum to less than 1.

        Returns:
            (figure, the plotted fractions: an ``Isotopologue`` column and
            one column per sample).
        """
        all_labels = IsotopologuePlotterService.labels(fractions)
        colors = _label_colors(all_labels)
        samples = [s for members in groups.values() for s in members]
        rows = {
            label: f'{molecule}_{label}' for label in labels
            if f'{molecule}_{label}' in fractions.index
        }
        data = fractions.loc[list(rows.values()), samples]
        data.index = list(rows)

        # Replicates sit under their condition on a two-level axis; without
        # replicates the bars are just the samples.
        x = samples
        if any(len(members) > 1 for members in groups.values()):
            x = [
                [c for c, members in groups.items() for _ in members],
                [_replicate_label(s, c) for c, members in groups.items() for s in members],
            ]
        fig = go.Figure()
        for label in data.index:
            fig.add_trace(go.Bar(
                x=x, y=data.loc[label].tolist(), name=label,
                marker=dict(color=colors[label], line=dict(color='#555555', width=0.5)),
                customdata=samples,
                hovertemplate=f'%{{customdata}}<br>{label}: %{{y:.4f}}<extra></extra>',
            ))
        fig.update_layout(
            **BASE_LAYOUT,
            barmode='stack',
            bargap=0.15,
            title=dict(text=molecule, x=0.5, xanchor='center', font=dict(size=20, color='black')),
            yaxis=dict(
                title=dict(text='Isotope Distribution', font=dict(size=16, color='black')),
                tickfont=dict(size=14, color='black'),
                # A full stack sums to 1; a partial one gets its own scale.
                range=[0, 1] if len(data.index) == _count_labels(fractions, molecule) else None,
            ),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            legend=dict(traceorder='reversed', font=dict(size=14, color='black')),
            height=500,
            margin=dict(t=60, b=80),
        )
        frame_axes(fig)
        return fig, data.rename_axis('Isotopologue').reset_index()


def _label_colors(labels: List[str]) -> Dict[str, str]:
    """Purples from light (_0) to dark (heaviest), fixed per label so a
    label keeps its colour whichever labels are plotted."""
    if len(labels) == 1:
        return {labels[0]: sample_colorscale('Purples', [_DARKEST])[0]}
    step = (_DARKEST - _LIGHTEST) / (len(labels) - 1)
    points = [_LIGHTEST + i * step for i in range(len(labels))]
    return dict(zip(labels, sample_colorscale('Purples', points)))


def _count_labels(fractions: pd.DataFrame, molecule: str) -> int:
    """How many isotopologues the molecule has in the table."""
    return sum(
        IsotopologuePlotterService.split_id(i)[0] == molecule for i in fractions.index
    )


def _replicate_label(sample: str, condition: str) -> str:
    """The bar's label under its condition: the replicate number, or nothing
    for a sample that is a condition of its own."""
    return sample[len(condition) + 1:] if sample != condition else ''
