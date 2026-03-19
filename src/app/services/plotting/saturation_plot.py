"""
Saturation plot plotting service.

Computes SFA/MUFA/PUFA fatty acid composition per lipid class and creates
concentration bar charts (with error bars and significance annotations) and
stacked percentage distribution charts.

Pure logic — no Streamlit dependencies.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.constants import HYDROXYL_PATTERN, SINGLE_CHAIN_CLASSES, parse_lipid_name
from app.models.experiment import ExperimentConfig
from app.services.statistical_testing import StatisticalTestSummary


# Fixed FA type colors
FA_COLORS = {'SFA': '#c9d9d3', 'MUFA': '#718dbf', 'PUFA': '#e84d60'}
FA_TYPES = ['SFA', 'MUFA', 'PUFA']

CHART_HEIGHT = 500
CHART_WIDTH = 800
BAR_WIDTH = 0.25
Y_PADDING_FACTOR = 1.1
ANNOTATION_INCREMENT_FACTOR = 0.15
ANNOTATION_TEXT_OFFSET = 0.1


@dataclass
class SaturationData:
    """Computed SFA/MUFA/PUFA statistics per lipid class.

    Attributes:
        fa_data: Nested dict of class -> FA type -> condition -> sample values.
                 e.g. {'PC': {'SFA': {'Control': array([...]), 'KO': array([...])}}}
        plot_data: Dict of class -> DataFrame with Condition, *_AUC, *_std columns.
        conditions: Conditions included in the data.
        classes: Lipid classes included in the data.
    """
    fa_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = field(default_factory=dict)
    plot_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)


class SaturationPlotterService:
    """Creates saturation (SFA/MUFA/PUFA) plots for lipid class analysis."""

    @staticmethod
    def calculate_fa_ratios(mol_structure: str) -> Tuple[float, float, float]:
        """Parse a lipid molecular structure into SFA/MUFA/PUFA chain ratios.

        Extracts individual fatty acid chains from the lipid name, counts
        double bonds for each, and classifies: SFA (0 double bonds),
        MUFA (1 double bond), PUFA (2+ double bonds).

        Args:
            mol_structure: Lipid name, e.g. 'PC 16:0_18:1'.

        Returns:
            (sfa_ratio, mufa_ratio, pufa_ratio) normalized by total chains.
            Returns (0, 0, 0) if parsing fails.
        """
        try:
            _, chain_info, _ = parse_lipid_name(mol_structure)
            if not chain_info:
                return (0.0, 0.0, 0.0)

            parts = re.split(r'[/_]', chain_info)

            # Extract double bond counts, stripping hydroxyl notation
            double_bonds = []
            for fa in parts:
                cleaned = HYDROXYL_PATTERN.sub('', fa.split(':')[-1])
                double_bonds.append(cleaned)

            sfa_count = double_bonds.count('0')
            mufa_count = double_bonds.count('1')
            pufa_count = len(double_bonds) - sfa_count - mufa_count
            total = len(double_bonds)

            if total == 0:
                return (0.0, 0.0, 0.0)

            return (sfa_count / total, mufa_count / total, pufa_count / total)
        except (IndexError, ValueError, AttributeError):
            return (0.0, 0.0, 0.0)

    @staticmethod
    def calculate_sfa_mufa_pufa(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_conditions: List[str],
        selected_classes: List[str],
    ) -> SaturationData:
        """Compute weighted SFA/MUFA/PUFA values per class, condition, and sample.

        For each (class, condition, sample), sums concentration * FA ratio
        across all species, yielding a single SFA/MUFA/PUFA value per sample.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[s*] columns.
            experiment: Experiment configuration.
            selected_conditions: Conditions to include.
            selected_classes: Lipid classes to include.

        Returns:
            SaturationData with per-class per-condition sample values and
            plot-ready DataFrames.

        Raises:
            ValueError: If no valid data can be computed.
        """
        if selected_conditions is None or len(selected_conditions) == 0:
            raise ValueError("At least one condition must be selected")
        if selected_classes is None or len(selected_classes) == 0:
            raise ValueError("At least one lipid class must be selected")

        fa_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        plot_data: Dict[str, pd.DataFrame] = {}
        actual_classes: List[str] = []

        for lipid_class in selected_classes:
            class_df = df[df['ClassKey'] == lipid_class]
            if class_df.empty:
                continue

            # Pre-compute FA ratios for all species in this class
            ratios = class_df['LipidMolec'].apply(
                SaturationPlotterService.calculate_fa_ratios
            )
            sfa_ratios = ratios.apply(lambda r: r[0]).values
            mufa_ratios = ratios.apply(lambda r: r[1]).values
            pufa_ratios = ratios.apply(lambda r: r[2]).values

            class_fa_data: Dict[str, Dict[str, np.ndarray]] = {
                'SFA': {}, 'MUFA': {}, 'PUFA': {}
            }
            rows: List[Dict] = []

            for condition in selected_conditions:
                if condition not in experiment.conditions_list:
                    continue

                cond_idx = experiment.conditions_list.index(condition)
                samples = experiment.individual_samples_list[cond_idx]
                sample_cols = [
                    f"concentration[{s}]" for s in samples
                    if f"concentration[{s}]" in class_df.columns
                ]

                if not sample_cols:
                    continue

                sfa_values = []
                mufa_values = []
                pufa_values = []

                for col in sample_cols:
                    concentrations = class_df[col].values.astype(float)
                    sfa_values.append(np.sum(concentrations * sfa_ratios))
                    mufa_values.append(np.sum(concentrations * mufa_ratios))
                    pufa_values.append(np.sum(concentrations * pufa_ratios))

                sfa_arr = np.array(sfa_values)
                mufa_arr = np.array(mufa_values)
                pufa_arr = np.array(pufa_values)

                class_fa_data['SFA'][condition] = sfa_arr
                class_fa_data['MUFA'][condition] = mufa_arr
                class_fa_data['PUFA'][condition] = pufa_arr

                # Check if we have meaningful data (at least one FA type > 0)
                has_data = (
                    sfa_arr.mean() > 0 or
                    mufa_arr.mean() > 0 or
                    pufa_arr.mean() > 0
                )

                if has_data and len(sample_cols) > 1:
                    n_samples = len(sample_cols)
                    rows.append({
                        'Condition': f"{condition} (n={n_samples})",
                        'SFA_AUC': sfa_arr.mean(),
                        'MUFA_AUC': mufa_arr.mean(),
                        'PUFA_AUC': pufa_arr.mean(),
                        'SFA_std': sfa_arr.std(ddof=1),
                        'MUFA_std': mufa_arr.std(ddof=1),
                        'PUFA_std': pufa_arr.std(ddof=1),
                    })

            if rows:
                fa_data[lipid_class] = class_fa_data
                plot_data[lipid_class] = pd.DataFrame(rows)
                actual_classes.append(lipid_class)

        if not actual_classes:
            raise ValueError("No valid data available after processing")

        actual_conditions = [
            c for c in selected_conditions
            if any(c in fa_data.get(cls, {}).get('SFA', {}) for cls in actual_classes)
        ]

        return SaturationData(
            fa_data=fa_data,
            plot_data=plot_data,
            conditions=actual_conditions,
            classes=actual_classes,
        )

    @staticmethod
    def identify_consolidated_lipids(
        df: pd.DataFrame,
        selected_classes: List[str],
    ) -> Dict[str, List[str]]:
        """Detect lipids in consolidated format (e.g., PC(34:1) instead of PC(16:0_18:1)).

        Consolidated lipids lack individual chain detail (no underscore separator),
        which prevents accurate SFA/MUFA/PUFA calculation. Single-chain classes
        (lysophospholipids, FFA, CE, MAG) are excluded from detection.

        Args:
            df: DataFrame with LipidMolec and ClassKey columns.
            selected_classes: Lipid classes to check.

        Returns:
            Dict mapping class name to list of consolidated lipid names.
            Empty dict if none found.
        """
        if df is None or df.empty:
            return {}

        consolidated_lipids: Dict[str, List[str]] = {}

        for lipid_class in selected_classes:
            if lipid_class in SINGLE_CHAIN_CLASSES:
                continue

            class_df = df[df['ClassKey'] == lipid_class]
            consolidated = []

            for lipid in class_df['LipidMolec']:
                lipid_str = str(lipid)
                _, chain_info, _ = parse_lipid_name(lipid_str)
                if chain_info and '_' not in chain_info and '/' not in chain_info and ':' in chain_info:
                    consolidated.append(lipid_str)

            if consolidated:
                consolidated_lipids[lipid_class] = consolidated

        return consolidated_lipids

    @staticmethod
    def create_concentration_plot(
        sat_data: SaturationData,
        lipid_class: str,
        stat_results: Optional[StatisticalTestSummary] = None,
        show_significance: bool = False,
    ) -> go.Figure:
        """Create a grouped bar chart showing SFA/MUFA/PUFA concentrations.

        Args:
            sat_data: Pre-computed saturation data.
            lipid_class: Class to plot.
            stat_results: Optional statistical test results for annotations.
            show_significance: Whether to display significance annotations.

        Returns:
            Plotly Figure.

        Raises:
            ValueError: If lipid_class not in sat_data or data is empty.
        """
        if lipid_class not in sat_data.plot_data:
            raise ValueError(f"No data available for class '{lipid_class}'")

        plot_df = sat_data.plot_data[lipid_class]
        if plot_df.empty:
            raise ValueError(f"Empty data for class '{lipid_class}'")

        fig = go.Figure()
        offsets = {fa: (i - 1) * BAR_WIDTH for i, fa in enumerate(FA_TYPES)}

        for fa in FA_TYPES:
            fig.add_trace(go.Bar(
                x=[c + offsets[fa] for c in range(len(plot_df))],
                y=plot_df[f'{fa}_AUC'],
                name=fa,
                marker_color=FA_COLORS[fa],
                error_y=dict(
                    type='data',
                    array=plot_df[f'{fa}_std'],
                    visible=True,
                    color='black',
                    thickness=1,
                    width=0,
                ),
                width=BAR_WIDTH,
                offset=0,
            ))

        y_max = _compute_y_max(plot_df)

        fig.update_layout(
            title=dict(
                text=f"Concentration Profile of Fatty Acids in {lipid_class}",
                font=dict(size=24, color='black'),
            ),
            xaxis_title=dict(text="Condition", font=dict(size=18, color='black')),
            yaxis_title=dict(text="Concentration", font=dict(size=18, color='black')),
            xaxis=dict(
                tickfont=dict(size=14, color='black'),
                tickmode='array',
                tickvals=list(range(len(plot_df))),
                ticktext=plot_df['Condition'].tolist(),
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='black'),
                range=[0, y_max * Y_PADDING_FACTOR],
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            barmode='group',
            bargap=0.15,
            bargroupgap=0,
            height=CHART_HEIGHT,
            width=CHART_WIDTH,
            margin=dict(t=50, r=150, b=50, l=50),
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        if show_significance and stat_results is not None:
            _add_significance_annotations(fig, plot_df, stat_results, lipid_class)

        return fig

    @staticmethod
    def create_percentage_plot(
        sat_data: SaturationData,
        lipid_class: str,
    ) -> go.Figure:
        """Create a stacked 100% bar chart showing SFA/MUFA/PUFA composition.

        Args:
            sat_data: Pre-computed saturation data.
            lipid_class: Class to plot.

        Returns:
            Plotly Figure.

        Raises:
            ValueError: If lipid_class not in sat_data or data is empty.
        """
        if lipid_class not in sat_data.plot_data:
            raise ValueError(f"No data available for class '{lipid_class}'")

        plot_df = sat_data.plot_data[lipid_class]
        if plot_df.empty:
            raise ValueError(f"Empty data for class '{lipid_class}'")

        pct_df = _calculate_percentage_df(plot_df)

        fig = go.Figure()

        for fa in FA_TYPES:
            fig.add_trace(go.Bar(
                x=pct_df['Condition'],
                y=pct_df[f'{fa}_AUC'],
                name=fa,
                marker_color=FA_COLORS[fa],
            ))

        fig.update_layout(
            title=dict(
                text=f"Percentage Distribution of Fatty Acids in {lipid_class}",
                font=dict(size=24, color='black'),
            ),
            xaxis_title=dict(text="Condition", font=dict(size=18, color='black')),
            yaxis_title=dict(text="Percentage", font=dict(size=18, color='black')),
            xaxis=dict(tickfont=dict(size=14, color='black')),
            yaxis=dict(tickfont=dict(size=14, color='black'), range=[0, 100]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=12, color='black')),
            barmode='stack',
            height=CHART_HEIGHT,
            width=CHART_WIDTH,
            margin=dict(t=50, r=50, b=50, l=50),
        )

        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig

    @staticmethod
    def generate_color_mapping() -> Dict[str, str]:
        """Return the fixed FA type color mapping.

        Returns:
            Dict mapping 'SFA', 'MUFA', 'PUFA' to hex color strings.
        """
        return dict(FA_COLORS)


# ── Private helpers ────────────────────────────────────────────────────


def _compute_y_max(plot_df: pd.DataFrame) -> float:
    """Compute the maximum y value (mean + std) across all FA types."""
    y_max = 0.0
    for fa in FA_TYPES:
        mean_col = f'{fa}_AUC'
        std_col = f'{fa}_std'
        if mean_col in plot_df.columns and std_col in plot_df.columns:
            local_max = (plot_df[mean_col] + plot_df[std_col]).max()
            y_max = max(y_max, local_max)
    return y_max if y_max > 0 else 1.0


def _calculate_percentage_df(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute FA concentrations to percentage distribution.

    Returns:
        DataFrame with Condition and SFA_AUC, MUFA_AUC, PUFA_AUC as percentages.
    """
    auc_cols = [f'{fa}_AUC' for fa in FA_TYPES]
    total = plot_df[auc_cols].sum(axis=1)

    pct_df = pd.DataFrame()
    pct_df['Condition'] = plot_df['Condition']

    for col in auc_cols:
        pct_df[col] = np.where(total > 0, 100 * plot_df[col] / total, 0.0)

    return pct_df


def _add_significance_annotations(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    stat_results: StatisticalTestSummary,
    lipid_class: str,
) -> None:
    """Add significance lines and *, **, *** symbols to the concentration plot."""
    y_max = _compute_y_max(plot_df)
    y_range = y_max
    y_increment = y_range * ANNOTATION_INCREMENT_FACTOR

    offsets = {fa: (i - 1) * BAR_WIDTH for i, fa in enumerate(FA_TYPES)}
    annotation_count = 0
    n_conditions = len(plot_df)

    for fa in FA_TYPES:
        key = f"{lipid_class}_{fa}"

        if key not in stat_results.results:
            continue

        result = stat_results.results[key]
        p_val = (
            result.adjusted_p_value
            if result.adjusted_p_value is not None
            else result.p_value
        )

        if n_conditions == 2:
            # Two conditions: single comparison line
            if p_val < 0.05:
                x0 = 0 + offsets[fa]
                x1 = 1 + offsets[fa]
                y = y_max + y_increment * (annotation_count + 1)

                fig.add_shape(
                    type="line",
                    x0=x0, y0=y, x1=x1, y1=y,
                    line=dict(color=FA_COLORS[fa], width=2),
                )

                symbol = _p_value_to_marker(p_val)
                fig.add_annotation(
                    x=(x0 + x1) / 2,
                    y=y + y_increment * ANNOTATION_TEXT_OFFSET,
                    text=symbol,
                    showarrow=False,
                    font=dict(size=20, color=FA_COLORS[fa]),
                )

                annotation_count += 1

        elif n_conditions > 2:
            # Multi-condition: post-hoc pairwise comparisons
            if key not in stat_results.posthoc_results:
                continue

            # Build condition-to-index mapping (strip sample count)
            clean_conditions = [
                cond.split(' (n=')[0] for cond in plot_df['Condition']
            ]

            for ph in stat_results.posthoc_results[key]:
                effective_p = (
                    ph.adjusted_p_value
                    if ph.adjusted_p_value is not None
                    else ph.p_value
                )
                if effective_p >= 0.05:
                    continue

                try:
                    x0_idx = clean_conditions.index(ph.group1)
                    x1_idx = clean_conditions.index(ph.group2)
                except ValueError:
                    continue

                x0 = x0_idx + offsets[fa]
                x1 = x1_idx + offsets[fa]
                y = y_max + y_increment * (annotation_count + 1)

                fig.add_shape(
                    type="line",
                    x0=x0, y0=y, x1=x1, y1=y,
                    line=dict(color=FA_COLORS[fa], width=2),
                )

                symbol = _p_value_to_marker(effective_p)
                fig.add_annotation(
                    x=(x0 + x1) / 2,
                    y=y + y_increment * ANNOTATION_TEXT_OFFSET,
                    text=symbol,
                    showarrow=False,
                    font=dict(size=20, color=FA_COLORS[fa]),
                )

                annotation_count += 1

    # Adjust y-axis to accommodate annotations
    if annotation_count > 0:
        new_y_max = y_max + y_increment * (annotation_count + 2)
        fig.update_layout(yaxis_range=[0, new_y_max])


def _p_value_to_marker(p_value: float) -> str:
    """Convert a p-value to a significance marker string."""
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''
