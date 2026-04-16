"""
Chain length and double bond distribution bubble chart plotting service.

Creates per-condition bubble charts showing the distribution of total carbon
chain lengths and double bonds per lipid class, with bubble size proportional
to mean concentration.

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.lipid_nomenclature import parse_chain_carbon_db, parse_lipid_name
from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import (
    generate_class_color_mapping,
    validate_dataframe,
)

CHART_HEIGHT_PER_CONDITION = 450
CHART_WIDTH = 900
MIN_MARKER_SIZE = 6
MAX_MARKER_SIZE = 50
MARKER_OPACITY = 0.7


@dataclass
class ChainLengthData:
    """Aggregated chain length and double bond data for bubble charts.

    Attributes:
        records: List of dicts with keys: ClassKey, TotalCarbons,
            TotalDoubleBonds, MeanConcentration.
        classes: Sorted list of lipid classes present.
    """

    records: List[Dict]
    classes: List[str]


class ChainLengthPlotterService:
    """Creates chain length and double bond distribution bubble charts."""

    @staticmethod
    def parse_total_chain_info(lipid_name: str) -> Optional[Tuple[int, int]]:
        """Extract total carbon count and total double bonds from a lipid name.

        Handles both species-level (e.g. 'PC 34:1') and molecular-level
        (e.g. 'PC 16:0_18:1') names by summing across all chains.

        Args:
            lipid_name: Standardized LIPID MAPS name.

        Returns:
            (total_carbons, total_double_bonds) or None if parsing fails.
        """
        try:
            _, chain_info, _ = parse_lipid_name(lipid_name)
            if not chain_info:
                return None

            parts = re.split(r'[/_]', chain_info)

            total_carbons = 0
            total_double_bonds = 0
            for part in parts:
                parsed = parse_chain_carbon_db(part)
                if parsed is None:
                    continue
                total_carbons += parsed[0]
                total_double_bonds += parsed[1]

            if total_carbons == 0:
                return None

            return (total_carbons, total_double_bonds)
        except (ValueError, IndexError, AttributeError):
            return None

    @staticmethod
    def calculate_chain_length_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> ChainLengthData:
        """Compute mean concentration per (class, total_carbons, total_double_bonds).

        For each lipid species, extracts total chain carbons and double bonds,
        computes the mean concentration across all selected condition samples,
        then aggregates by (class, carbons, double_bonds) summing concentrations.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.

        Returns:
            ChainLengthData with per-species aggregated records.
        """
        validate_dataframe(df, ['LipidMolec', 'ClassKey'])

        # Collect concentration columns for selected conditions
        conc_cols = []
        for cond, samples in zip(
            experiment.conditions_list, experiment.individual_samples_list,
        ):
            if cond in selected_conditions:
                conc_cols.extend(f'concentration[{s}]' for s in samples)
        conc_cols = [c for c in conc_cols if c in df.columns]

        if not conc_cols:
            return ChainLengthData(records=[], classes=[])

        # Filter to selected classes
        filtered = df[df['ClassKey'].isin(selected_classes)].copy()
        if filtered.empty:
            return ChainLengthData(records=[], classes=[])

        # Parse chain info and compute mean concentration per species
        rows = []
        for _, row in filtered.iterrows():
            parsed = ChainLengthPlotterService.parse_total_chain_info(
                row['LipidMolec'],
            )
            if parsed is None:
                continue
            total_c, total_db = parsed
            mean_conc = row[conc_cols].mean()
            if np.isnan(mean_conc) or mean_conc <= 0:
                continue
            rows.append({
                'ClassKey': row['ClassKey'],
                'TotalCarbons': total_c,
                'TotalDoubleBonds': total_db,
                'MeanConcentration': mean_conc,
            })

        if not rows:
            return ChainLengthData(records=[], classes=[])

        # Aggregate: sum concentrations for same (class, carbons, db)
        agg_df = pd.DataFrame(rows).groupby(
            ['ClassKey', 'TotalCarbons', 'TotalDoubleBonds'],
            as_index=False,
        ).agg({'MeanConcentration': 'sum'})

        records = agg_df.to_dict('records')
        classes = sorted(agg_df['ClassKey'].unique().tolist())

        return ChainLengthData(records=records, classes=classes)

    @staticmethod
    def calculate_per_condition_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> Dict[str, ChainLengthData]:
        """Compute chain length data separately for each condition.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.

        Returns:
            Dict mapping condition name to its ChainLengthData.
        """
        result = {}
        for cond in selected_conditions:
            data = ChainLengthPlotterService.calculate_chain_length_data(
                df, experiment, [cond], selected_classes,
            )
            result[cond] = data
        return result

    @staticmethod
    def create_per_condition_figure(
        per_condition_data: Dict[str, ChainLengthData],
        color_mapping: Dict[str, str],
    ) -> go.Figure:
        """Create vertically-stacked bubble charts, one row per condition.

        Each row has two panels: carbon chain length (left) and double bonds
        (right). Bubble size is proportional to mean concentration.

        Args:
            per_condition_data: Dict mapping condition name to ChainLengthData.
            color_mapping: Dict mapping class name to hex color.

        Returns:
            Plotly Figure with vertically stacked subplots.
        """
        conditions = list(per_condition_data.keys())
        n_conditions = len(conditions)

        if n_conditions == 0:
            fig = go.Figure()
            fig.update_layout(
                title="No chain length data available",
                height=CHART_HEIGHT_PER_CONDITION,
            )
            return fig

        # Check if any condition has data
        has_any_data = any(
            d.records for d in per_condition_data.values()
        )
        if not has_any_data:
            fig = go.Figure()
            fig.update_layout(
                title="No chain length data available",
                height=CHART_HEIGHT_PER_CONDITION,
            )
            return fig

        # Build subplot titles: two per condition row
        subplot_titles = []
        for cond in conditions:
            subplot_titles.append(f'{cond} — Carbon Chain Length')
            subplot_titles.append(f'{cond} — Double Bonds')

        fig = make_subplots(
            rows=n_conditions, cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.22 if n_conditions > 1 else 0.15,
            horizontal_spacing=0.12,
        )

        # Collect all concentrations across conditions for global marker scaling
        all_conc_values = []
        for data in per_condition_data.values():
            for rec in data.records:
                all_conc_values.append(rec['MeanConcentration'])

        if not all_conc_values:
            fig.update_layout(
                title="No chain length data available",
                height=CHART_HEIGHT_PER_CONDITION,
            )
            return fig

        global_conc = np.array(all_conc_values)

        # Track which classes have been added to the legend
        legend_added = set()

        for row_idx, cond in enumerate(conditions, start=1):
            data = per_condition_data[cond]
            if not data.records:
                continue

            rec_df = pd.DataFrame(data.records)
            # Scale markers using global concentration range
            size_arr = _scale_marker_sizes(
                rec_df['MeanConcentration'].values, global_conc,
            )

            for i, rec_row in rec_df.iterrows():
                cls = rec_row['ClassKey']
                color = color_mapping.get(cls, '#333333')
                show_legend = cls not in legend_added
                if show_legend:
                    legend_added.add(cls)

                shared = dict(
                    mode='markers',
                    name=cls,
                    legendgroup=cls,
                    showlegend=show_legend,
                    marker=dict(
                        color=color,
                        size=size_arr[i],
                        opacity=MARKER_OPACITY,
                        line=dict(width=0.5, color='white'),
                    ),
                    hovertemplate=(
                        f"<b>{cls}</b> ({cond})<br>"
                        f"Carbons: {int(rec_row['TotalCarbons'])}<br>"
                        f"Double bonds: {int(rec_row['TotalDoubleBonds'])}<br>"
                        f"Mean concentration: {rec_row['MeanConcentration']:.2f}"
                        "<extra></extra>"
                    ),
                )

                # Left panel: chain length
                fig.add_trace(
                    go.Scatter(
                        x=[cls], y=[rec_row['TotalCarbons']], **shared,
                    ),
                    row=row_idx, col=1,
                )

                # Right panel: double bonds
                fig.add_trace(
                    go.Scatter(
                        x=[cls], y=[rec_row['TotalDoubleBonds']],
                        **{**shared, 'showlegend': False},
                    ),
                    row=row_idx, col=2,
                )

        total_height = CHART_HEIGHT_PER_CONDITION * n_conditions
        fig.update_layout(
            height=total_height,
            width=CHART_WIDTH,
            plot_bgcolor='white',
            legend=dict(
                title='Lipid Class',
                orientation='h',
                yanchor='top',
                y=-0.12,
                xanchor='center',
                x=0.5,
            ),
            margin=dict(b=max(120, 50 * len(legend_added) // 6)),
        )

        for row_idx in range(1, n_conditions + 1):
            fig.update_xaxes(
                title_text='Lipid Class', showgrid=False,
                row=row_idx, col=1,
            )
            fig.update_xaxes(
                title_text='Lipid Class', showgrid=False,
                row=row_idx, col=2,
            )
            fig.update_yaxes(
                title_text='Total Carbon Chain Length',
                gridcolor='lightgray', row=row_idx, col=1,
            )
            fig.update_yaxes(
                title_text='Total Double Bonds',
                gridcolor='lightgray', row=row_idx, col=2,
            )

        return fig

    @staticmethod
    def generate_color_mapping(classes: List[str]) -> Dict[str, str]:
        """Map each lipid class to a consistent color.

        Args:
            classes: List of lipid class names.

        Returns:
            Dict mapping class name to hex color string.
        """
        return generate_class_color_mapping(classes)


def _scale_marker_sizes(
    values: np.ndarray,
    global_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale concentration values to marker pixel sizes.

    Uses square-root scaling so bubble area (not radius) is proportional
    to the value — matching standard bubble chart conventions.

    Args:
        values: Concentration values to scale.
        global_values: If provided, use this array to determine the
            min/max range for scaling (for consistent sizing across
            conditions). Falls back to values if not provided.
    """
    if len(values) == 0:
        return np.array([])

    ref = global_values if global_values is not None else values
    sqrt_vals = np.sqrt(values)
    sqrt_ref = np.sqrt(ref)
    v_min, v_max = sqrt_ref.min(), sqrt_ref.max()

    if v_max == v_min:
        return np.full_like(sqrt_vals, (MIN_MARKER_SIZE + MAX_MARKER_SIZE) / 2)
    scaled = (sqrt_vals - v_min) / (v_max - v_min)
    # Clamp to [0, 1] in case values is outside global range
    scaled = np.clip(scaled, 0, 1)
    return MIN_MARKER_SIZE + scaled * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)
