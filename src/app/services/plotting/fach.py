"""
Fatty Acid Composition Heatmap (FACH) plotting service.

Parses carbon chain length and double bond count from lipid names,
aggregates proportional abundance per (Carbon, DB) per condition,
and renders side-by-side Plotly heatmaps with weighted average lines.

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.constants import parse_lipid_name
from app.models.experiment import ExperimentConfig


# ── Constants ──────────────────────────────────────────────────────────

HEATMAP_HEIGHT = 600
HEATMAP_WIDTH_PER_CONDITION = 300

CUSTOM_COLORSCALE = [
    [0.0, 'rgb(255, 255, 255)'],
    [0.001, 'rgb(255, 255, 204)'],
    [0.2, 'rgb(255, 204, 153)'],
    [0.4, 'rgb(255, 153, 102)'],
    [0.6, 'rgb(255, 102, 51)'],
    [0.8, 'rgb(204, 51, 0)'],
    [1.0, 'rgb(153, 0, 0)'],
]

DEFAULT_MAX_DB = 9
DEFAULT_MIN_CARBON = 0
DEFAULT_MAX_CARBON = 50

AVG_LINE_STYLE = dict(color='black', dash='dash', width=2)
AVG_ANNOTATION_FONT = dict(size=12, color='black')
AVG_ANNOTATION_BG = 'rgba(255, 255, 255, 0.8)'


@dataclass
class FACHData:
    """Prepared FACH data for heatmap rendering.

    Attributes:
        data_dict: Per-condition DataFrames with Carbon, DB, Proportion columns.
        selected_class: The lipid class this data represents.
        unparsable_lipids: Lipid names that could not be parsed.
    """
    data_dict: Dict[str, pd.DataFrame] = field(default_factory=dict)
    selected_class: str = ''
    unparsable_lipids: List[str] = field(default_factory=list)


class FACHPlotterService:
    """Creates Fatty Acid Composition Heatmaps for lipidomics data."""

    @staticmethod
    def parse_carbon_db(lipid_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse total carbon atoms and double bonds from a lipid name.

        Handles standard, ether, sphingoid, oxidized, and modified formats:
        - Standard: 'PC 16:0_18:1'
        - Ether lipids: 'PC O-38:4', 'PE P-36:2'
        - Sphingoid bases: 'Cer d18:1;O2/24:0', 'SM d18:1/16:0'
        - Oxidized: 'PC 16:0_18:1+O', 'PE 18:0_20:4+2O'
        - With modifications: 'LPC 18:1(d7)'

        Args:
            lipid_name: Standardized lipid name string.

        Returns:
            (total_carbons, total_double_bonds) or (None, None) if unparsable.
        """
        if not lipid_name or not isinstance(lipid_name, str):
            return None, None

        _, chain_info, _ = parse_lipid_name(lipid_name)
        if not chain_info:
            return None, None

        composition = chain_info
        total_c = 0
        total_db = 0

        chains = re.split(r'[/_]', composition)

        for chain in chains:
            # Remove ether/sphingoid prefixes: O-, P-, d, t, m
            chain_cleaned = re.sub(r'^[OPdtm]-?', '', chain)
            # Remove oxidation suffixes: +O, +2O, +3O
            chain_cleaned = re.sub(r'\+\d*O', '', chain_cleaned)
            # Remove chain identifier C (e.g., C24:0)
            chain_cleaned = re.sub(r'^C', '', chain_cleaned)

            chain_match = re.match(r'(\d+):(\d+)', chain_cleaned)
            if chain_match:
                total_c += int(chain_match.group(1))
                total_db += int(chain_match.group(2))

        return (total_c, total_db) if total_c > 0 else (None, None)

    @staticmethod
    def prepare_fach_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_class: str,
        selected_conditions: List[str],
    ) -> FACHData:
        """Aggregate proportional abundance per (Carbon, DB) for each condition.

        For each condition, computes the mean concentration per lipid across
        samples, groups by (Carbon, DB), sums, and converts to percentages.

        Args:
            df: DataFrame with LipidMolec, ClassKey, and concentration[s*] columns.
            experiment: Experiment configuration.
            selected_class: Single lipid class to analyse.
            selected_conditions: Conditions to include.

        Returns:
            FACHData with per-condition proportion DataFrames.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not selected_class:
            raise ValueError("A lipid class must be selected")
        if selected_conditions is None or len(selected_conditions) == 0:
            raise ValueError("At least one condition must be selected")

        class_df = df[df['ClassKey'] == selected_class].copy()

        if class_df.empty:
            return FACHData(selected_class=selected_class)

        # Parse carbon and double bond counts
        unparsable: List[str] = []
        parsed_results = []

        for lipid_name in class_df['LipidMolec']:
            c, db = FACHPlotterService.parse_carbon_db(lipid_name)
            parsed_results.append((c, db))
            if c is None:
                unparsable.append(lipid_name)

        class_df['Carbon'] = [r[0] for r in parsed_results]
        class_df['DB'] = [r[1] for r in parsed_results]

        # Convert None → NaN and drop unparsable rows
        class_df['Carbon'] = pd.to_numeric(class_df['Carbon'], errors='coerce')
        class_df['DB'] = pd.to_numeric(class_df['DB'], errors='coerce')
        class_df = class_df.dropna(subset=['Carbon', 'DB'])

        if class_df.empty:
            return FACHData(
                selected_class=selected_class,
                unparsable_lipids=unparsable,
            )

        data_dict: Dict[str, pd.DataFrame] = {}

        for condition in selected_conditions:
            if condition not in experiment.conditions_list:
                continue

            cond_idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[cond_idx]
            conc_cols = [
                f'concentration[{s}]' for s in samples
                if f'concentration[{s}]' in class_df.columns
            ]

            if not conc_cols:
                continue

            # Mean concentration per lipid across samples in this condition
            mean_conc = class_df[conc_cols].mean(axis=1)

            # Aggregate by (Carbon, DB)
            agg_df = class_df[['Carbon', 'DB']].copy()
            agg_df['Mean_Conc'] = mean_conc.values
            agg_df = agg_df.groupby(['Carbon', 'DB'], as_index=False)['Mean_Conc'].sum()

            # Convert to proportions
            total_conc = agg_df['Mean_Conc'].sum()
            if total_conc > 0:
                agg_df['Proportion'] = (agg_df['Mean_Conc'] / total_conc) * 100
            else:
                agg_df['Proportion'] = 0.0

            data_dict[condition] = agg_df[['Carbon', 'DB', 'Proportion']].copy()

        return FACHData(
            data_dict=data_dict,
            selected_class=selected_class,
            unparsable_lipids=unparsable,
        )

    @staticmethod
    def create_fach_heatmap(fach_data: FACHData) -> Optional[go.Figure]:
        """Create side-by-side heatmaps with weighted average lines.

        Each condition gets its own subplot with consistent axis ranges
        and color scale. Weighted average Carbon and DB lines are drawn
        as dashed black lines with annotations.

        Args:
            fach_data: Prepared FACH data from prepare_fach_data().

        Returns:
            Plotly Figure with side-by-side heatmaps, or None if no data.
        """
        data_dict = fach_data.data_dict
        if not data_dict:
            return None

        # Global axis and color ranges
        all_proportions = pd.concat([d['Proportion'] for d in data_dict.values()])
        vmax = all_proportions.max() if not all_proportions.empty else 1.0

        all_db = pd.concat([d['DB'] for d in data_dict.values()])
        all_carbon = pd.concat([d['Carbon'] for d in data_dict.values()])

        max_db = int(all_db.max()) if not all_db.empty else DEFAULT_MAX_DB
        min_carbon = int(all_carbon.min()) if not all_carbon.empty else DEFAULT_MIN_CARBON
        max_carbon = int(all_carbon.max()) if not all_carbon.empty else DEFAULT_MAX_CARBON

        db_values = list(range(0, max_db + 1))
        carbon_values = list(range(min_carbon, max_carbon + 1))

        n_conditions = len(data_dict)
        fig = make_subplots(
            rows=1,
            cols=n_conditions,
            subplot_titles=list(data_dict.keys()),
            shared_yaxes=True,
        )

        col_idx = 1
        for condition, cond_df in data_dict.items():
            # Weighted averages
            avg_db, avg_carbon = _compute_weighted_averages(cond_df)

            # Build 2D matrix (rows=carbon, cols=db)
            z_matrix = np.zeros((len(carbon_values), len(db_values)))

            for _, row in cond_df.iterrows():
                carbon_idx = int(row['Carbon']) - min_carbon
                db_idx = int(row['DB'])
                if 0 <= carbon_idx < len(carbon_values) and 0 <= db_idx < len(db_values):
                    z_matrix[carbon_idx, db_idx] = row['Proportion']

            heatmap = go.Heatmap(
                x=db_values,
                y=carbon_values,
                z=z_matrix,
                colorscale=CUSTOM_COLORSCALE,
                zmin=0,
                zmax=vmax,
                colorbar=dict(
                    title='Proportion (%)',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black'),
                ),
                hovertemplate=(
                    'Double Bonds: %{x}<br>Carbon: %{y}'
                    '<br>Proportion: %{z:.2f}%<extra></extra>'
                ),
                xgap=1,
                ygap=1,
            )
            fig.add_trace(heatmap, row=1, col=col_idx)

            # Average DB vertical line + annotation
            _add_average_lines(
                fig, col_idx, avg_db, avg_carbon,
                max_db, min_carbon, max_carbon,
            )

            # X-axis: all integer DB values
            fig.update_xaxes(
                tickmode='array',
                tickvals=db_values,
                ticktext=db_values,
                tickangle=0,
                title_text='Double Bonds',
                titlefont=dict(color='black'),
                tickfont=dict(color='black'),
                range=[-0.5, max_db + 1.5],
                row=1,
                col=col_idx,
            )

            # Y-axis: title only on first subplot
            y_kwargs = dict(
                tickfont=dict(color='black'),
                range=[min_carbon - 0.5, max_carbon + 2.5],
            )
            if col_idx == 1:
                y_kwargs['title_text'] = 'Carbon Chain Length'
                y_kwargs['titlefont'] = dict(color='black')
            fig.update_yaxes(row=1, col=col_idx, **y_kwargs)

            col_idx += 1

        # Style subplot titles
        for annotation in fig.layout.annotations:
            if annotation.text in data_dict:
                annotation.font = dict(color='black', size=12)

        fig.update_layout(
            title='Fatty Acid Composition Heatmaps',
            titlefont=dict(color='black'),
            height=HEATMAP_HEIGHT,
            width=HEATMAP_WIDTH_PER_CONDITION * n_conditions,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        return fig

    @staticmethod
    def get_weighted_averages(fach_data: FACHData) -> Dict[str, Tuple[float, float]]:
        """Return weighted average (DB, Carbon) per condition.

        Args:
            fach_data: Prepared FACH data.

        Returns:
            Dict mapping condition name to (avg_db, avg_carbon) tuple.
        """
        result: Dict[str, Tuple[float, float]] = {}
        for condition, cond_df in fach_data.data_dict.items():
            result[condition] = _compute_weighted_averages(cond_df)
        return result


# ── Private helpers ────────────────────────────────────────────────────


def _compute_weighted_averages(
    cond_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Compute weighted average DB and Carbon from proportion data.

    Returns:
        (avg_db, avg_carbon). Both 0.0 if no data or all-zero proportions.
    """
    if cond_df.empty or cond_df['Proportion'].sum() <= 0:
        return 0.0, 0.0

    avg_db = float(np.average(cond_df['DB'], weights=cond_df['Proportion']))
    avg_carbon = float(np.average(cond_df['Carbon'], weights=cond_df['Proportion']))
    return avg_db, avg_carbon


def _add_average_lines(
    fig: go.Figure,
    col_idx: int,
    avg_db: float,
    avg_carbon: float,
    max_db: int,
    min_carbon: int,
    max_carbon: int,
) -> None:
    """Add weighted average lines and annotations to a subplot."""
    # Vertical line for average DB
    fig.add_vline(
        x=avg_db,
        line=AVG_LINE_STYLE,
        row=1,
        col=col_idx,
    )
    fig.add_annotation(
        x=avg_db,
        y=max_carbon + 1.5,
        text=f'Avg DB: {avg_db:.1f}',
        showarrow=False,
        font=AVG_ANNOTATION_FONT,
        bgcolor=AVG_ANNOTATION_BG,
        xref=f'x{col_idx}',
        yref=f'y{col_idx}',
        row=1,
        col=col_idx,
    )

    # Horizontal line for average Carbon
    fig.add_hline(
        y=avg_carbon,
        line=AVG_LINE_STYLE,
        row=1,
        col=col_idx,
    )
    fig.add_annotation(
        x=max_db + 0.5,
        y=avg_carbon,
        text=f'Avg C: {avg_carbon:.1f}',
        showarrow=False,
        font=AVG_ANNOTATION_FONT,
        bgcolor=AVG_ANNOTATION_BG,
        xanchor='left',
        xref=f'x{col_idx}',
        yref=f'y{col_idx}',
        row=1,
        col=col_idx,
    )
