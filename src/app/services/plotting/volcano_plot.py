"""
Volcano plot plotting service.

Prepares per-lipid statistical data (log2 fold change, -log10 p-value),
creates Plotly volcano scatter plots with threshold lines and label placement,
concentration-vs-fold-change scatter plots, and Seaborn distribution box plots.

Pure logic — no Streamlit dependencies.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import generate_class_color_mapping
from app.services.statistical_testing import (
    StatisticalTestResult,
    StatisticalTestSummary,
)


CHART_HEIGHT = 600
CHART_WIDTH = 800
MARKER_SIZE = 5
THRESHOLD_LINE_WIDTH = 2
THRESHOLD_LINE_COLOR = 'red'
THRESHOLD_LINE_DASH = 'dash'

# Label placement parameters
LABEL_CHAR_WIDTH = 0.05
LABEL_HEIGHT = 0.3
LABEL_BUFFER = 0.2
LABEL_FONT_SIZE = 12
ARROW_HEAD = 1
ARROW_WIDTH = 1

# 8 candidate positions: (dx, dy, align)
_LABEL_CANDIDATES = [
    (0.1, 0.3, 'left'),    # above right
    (-0.1, 0.3, 'right'),  # above left
    (0.1, -0.3, 'left'),   # below right
    (-0.1, -0.3, 'right'), # below left
    (0.3, 0.1, 'left'),    # right upper
    (0.3, -0.1, 'left'),   # right lower
    (-0.3, 0.1, 'right'),  # left upper
    (-0.3, -0.1, 'right'), # left lower
]


@dataclass
class VolcanoData:
    """Prepared volcano plot data.

    Attributes:
        volcano_df: DataFrame with columns LipidMolec, ClassKey,
            FoldChange (log2), pValue, adjusted_pValue,
            -log10(pValue), -log10(adjusted_pValue),
            Log10MeanControl, mean_control, mean_experimental,
            test_method, transformation, significant, correction_method.
        removed_lipids_df: DataFrame with LipidMolec, ClassKey, Reason
            for lipids excluded from analysis.
        stat_results: The StatisticalTestSummary used to build the data.
    """
    volcano_df: pd.DataFrame
    removed_lipids_df: pd.DataFrame
    stat_results: StatisticalTestSummary = field(
        default_factory=StatisticalTestSummary
    )


class VolcanoPlotterService:
    """Creates volcano plots for species-level lipidomic analysis."""

    @staticmethod
    def prepare_volcano_data(
        df: pd.DataFrame,
        stat_results: StatisticalTestSummary,
        control_samples: List[str],
        experimental_samples: List[str],
    ) -> VolcanoData:
        """Build volcano-ready DataFrames from statistical test results.

        For each lipid in df, looks up its result in stat_results.
        Tested lipids go into volcano_df; untested lipids go into
        removed_lipids_df with a reason.

        Args:
            df: DataFrame with LipidMolec, ClassKey, concentration[s*] cols.
            stat_results: Results from StatisticalTestingService.run_species_level_tests.
            control_samples: Sample names for the control condition.
            experimental_samples: Sample names for the experimental condition.

        Returns:
            VolcanoData with volcano_df and removed_lipids_df.

        Raises:
            ValueError: If df is empty or missing required columns.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        for col in ('LipidMolec', 'ClassKey'):
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        ctrl_cols = [f'concentration[{s}]' for s in control_samples]
        exp_cols = [f'concentration[{s}]' for s in experimental_samples]

        volcano_rows: List[Dict] = []
        removed_rows: List[Dict] = []

        for _, row in df.iterrows():
            lipid = row['LipidMolec']
            class_key = row['ClassKey']

            if lipid in stat_results.results:
                result = stat_results.results[lipid]
                # effect_size holds log2 fold change
                log2_fc = result.effect_size if result.effect_size is not None else 0.0

                # Compute mean control/experimental from zero-adjusted values
                ctrl_vals = _safe_numeric_values(row, ctrl_cols)
                exp_vals = _safe_numeric_values(row, exp_cols)
                mean_ctrl, mean_exp = _compute_zero_adj_means(
                    ctrl_vals, exp_vals
                )

                raw_p = result.p_value
                adj_p = (
                    result.adjusted_p_value
                    if result.adjusted_p_value is not None
                    and not np.isnan(result.adjusted_p_value)
                    else raw_p
                )

                volcano_rows.append({
                    'LipidMolec': lipid,
                    'ClassKey': class_key,
                    'FoldChange': log2_fc,
                    'pValue': raw_p,
                    'adjusted_pValue': adj_p,
                    '-log10(pValue)': -np.log10(raw_p) if raw_p > 0 else 0.0,
                    '-log10(adjusted_pValue)': -np.log10(adj_p) if adj_p > 0 else 0.0,
                    'Log10MeanControl': (
                        np.log10(mean_ctrl) if mean_ctrl > 0 else 0.0
                    ),
                    'mean_control': mean_ctrl,
                    'mean_experimental': mean_exp,
                    'test_method': result.test_name,
                    'transformation': stat_results.test_info.get('transform', 'none'),
                    'significant': result.significant,
                    'correction_method': stat_results.test_info.get('correction', 'uncorrected'),
                })
            else:
                reason = _determine_exclusion_reason(row, ctrl_cols, exp_cols)
                removed_rows.append({
                    'LipidMolec': lipid,
                    'ClassKey': class_key,
                    'Reason': reason,
                })

        volcano_df = pd.DataFrame(volcano_rows) if volcano_rows else pd.DataFrame()
        removed_df = pd.DataFrame(removed_rows) if removed_rows else pd.DataFrame()

        return VolcanoData(
            volcano_df=volcano_df,
            removed_lipids_df=removed_df,
            stat_results=stat_results,
        )

    @staticmethod
    def create_volcano_plot(
        volcano_data: VolcanoData,
        color_mapping: Dict[str, str],
        p_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        hide_non_sig: bool = False,
        use_adjusted_p: bool = True,
        top_n_labels: int = 0,
        custom_label_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        additional_labels: Optional[List[str]] = None,
    ) -> go.Figure:
        """Create a Plotly volcano plot scatter with threshold lines.

        Args:
            volcano_data: Pre-computed volcano data.
            color_mapping: ClassKey → hex color string.
            p_threshold: P-value threshold for significance line.
            fc_threshold: Log2 fold change threshold for vertical lines.
            hide_non_sig: If True, hide non-significant points.
            use_adjusted_p: Use adjusted p-values for y-axis.
            top_n_labels: Number of top lipids to label (by p-value).
            custom_label_positions: Optional per-lipid (x_offset, y_offset) overrides.

        Returns:
            Plotly Figure.

        Raises:
            ValueError: If volcano_data has no data.
        """
        vdf = volcano_data.volcano_df
        if vdf is None or vdf.empty:
            raise ValueError("No volcano data to plot")

        p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
        p_raw_col = 'adjusted_pValue' if use_adjusted_p else 'pValue'
        q_threshold = -np.log10(p_threshold) if p_threshold > 0 else 0.0

        display_df = _filter_significant(
            vdf, fc_threshold, q_threshold, p_col, hide_non_sig
        )

        fig = go.Figure()

        # One trace per class
        for class_name, color in color_mapping.items():
            class_df = display_df[display_df['ClassKey'] == class_name]
            if class_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=class_df['FoldChange'],
                y=class_df[p_col],
                mode='markers',
                name=class_name,
                marker=dict(color=color, size=MARKER_SIZE),
                text=class_df['LipidMolec'],
                customdata=np.column_stack((
                    class_df[p_raw_col],
                    class_df['test_method'],
                    class_df['transformation'],
                    class_df['correction_method'],
                )) if not class_df.empty else None,
                hovertemplate=(
                    '<b>Lipid:</b> %{text}<br>'
                    '<b>Log2 Fold Change:</b> %{x:.3f}<br>'
                    f'<b>{"-log10(adj. p-value)" if use_adjusted_p else "-log10(p-value)"}:</b> %{{y:.3f}}<br>'
                    f'<b>{"Adj. p-value" if use_adjusted_p else "p-value"}:</b> %{{customdata[0]:.2e}}<br>'
                    '<b>Test:</b> %{customdata[1]}<br>'
                    '<b>Transform:</b> %{customdata[2]}<br>'
                    '<b>Correction:</b> %{customdata[3]}<extra></extra>'
                ),
            ))

        # Threshold lines
        if not display_df.empty:
            _add_threshold_lines(
                fig, display_df, p_col, q_threshold, fc_threshold
            )

        # Combine top N + additional labels for rendering
        lipids_to_label: List[str] = []
        if top_n_labels > 0:
            top_df = vdf.sort_values(p_col, ascending=False).head(top_n_labels)
            lipids_to_label = top_df['LipidMolec'].tolist()
        if additional_labels:
            for lip in additional_labels:
                if lip not in lipids_to_label:
                    lipids_to_label.append(lip)

        if lipids_to_label and not vdf.empty:
            _add_labels(
                fig, vdf, p_col, color_mapping,
                lipids_to_label, custom_label_positions,
            )

        # Layout
        p_label = "Adjusted p-value" if use_adjusted_p else "p-value"
        fig.update_layout(
            title=dict(
                text="Volcano Plot",
                font=dict(size=20, color='black'),
            ),
            xaxis_title=dict(
                text="Log2(Fold Change)",
                font=dict(size=16, color='black'),
            ),
            yaxis_title=dict(
                text=f"-log10({p_label})",
                font=dict(size=16, color='black'),
            ),
            xaxis=dict(
                tickfont=dict(size=12, color='black'),
                showgrid=True, gridcolor='lightgray',
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='black'),
                showgrid=True, gridcolor='lightgray',
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=10, color='black')),
            height=CHART_HEIGHT,
            margin=dict(t=80, r=50, b=50, l=50),
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        return fig

    @staticmethod
    def create_concentration_vs_fc_plot(
        volcano_data: VolcanoData,
        color_mapping: Dict[str, str],
        p_threshold: float = 0.05,
        hide_non_sig: bool = False,
        use_adjusted_p: bool = True,
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Create Log10(Mean Control) vs Log2(Fold Change) scatter plot.

        Args:
            volcano_data: Pre-computed volcano data.
            color_mapping: ClassKey → hex color string.
            p_threshold: P-value threshold for filtering.
            hide_non_sig: If True, hide non-significant points.
            use_adjusted_p: Use adjusted p-values.

        Returns:
            (Plotly Figure, summary DataFrame with LipidMolec,
             Log10MeanControl, FoldChange, ClassKey).

        Raises:
            ValueError: If volcano_data has no data.
        """
        vdf = volcano_data.volcano_df
        if vdf is None or vdf.empty:
            raise ValueError("No volcano data to plot")

        p_col = '-log10(adjusted_pValue)' if use_adjusted_p else '-log10(pValue)'
        q_threshold = -np.log10(p_threshold) if p_threshold > 0 else 0.0

        display_df = _filter_significant(
            vdf, 1.0, q_threshold, p_col, hide_non_sig
        )

        fig = go.Figure()
        for class_name, color in color_mapping.items():
            class_df = display_df[display_df['ClassKey'] == class_name]
            if class_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=class_df['FoldChange'],
                y=class_df['Log10MeanControl'],
                mode='markers',
                name=class_name,
                marker=dict(color=color, size=MARKER_SIZE),
                text=class_df['LipidMolec'],
                hovertemplate=(
                    '<b>Lipid:</b> %{text}<br>'
                    '<b>Log2 Fold Change:</b> %{x:.3f}<br>'
                    '<b>Log10(Mean Control):</b> %{y:.3f}<extra></extra>'
                ),
            ))

        fig.update_layout(
            title=dict(
                text="Fold Change vs. Mean Control Concentration",
                font=dict(size=20, color='black'),
            ),
            xaxis_title=dict(
                text="Log2(Fold Change)",
                font=dict(size=16, color='black'),
            ),
            yaxis_title=dict(
                text="Log10(Mean Control Concentration)",
                font=dict(size=16, color='black'),
            ),
            xaxis=dict(
                tickfont=dict(size=12, color='black'),
                showgrid=True, gridcolor='lightgray',
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='black'),
                showgrid=True, gridcolor='lightgray',
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(font=dict(size=10, color='black')),
            height=CHART_HEIGHT,
            margin=dict(t=50, r=50, b=50, l=50),
        )
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        summary_cols = ['LipidMolec', 'Log10MeanControl', 'FoldChange', 'ClassKey']
        summary_df = display_df[
            [c for c in summary_cols if c in display_df.columns]
        ].copy()

        return fig, summary_df

    @staticmethod
    def create_distribution_plot(
        df: pd.DataFrame,
        selected_lipids: List[str],
        selected_conditions: List[str],
        experiment: ExperimentConfig,
    ) -> plt.Figure:
        """Create a Seaborn box plot of concentration distributions.

        Args:
            df: Full DataFrame with concentration[s*] columns and LipidMolec.
            selected_lipids: Lipid names to plot.
            selected_conditions: Conditions to include.
            experiment: Experiment configuration.

        Returns:
            Matplotlib Figure.

        Raises:
            ValueError: If no valid data for selected lipids/conditions.
        """
        if not selected_lipids:
            raise ValueError("At least one lipid must be selected")
        if not selected_conditions:
            raise ValueError("At least one condition must be selected")

        plot_rows: List[Dict] = []
        for lipid in selected_lipids:
            lipid_rows = df[df['LipidMolec'] == lipid]
            if lipid_rows.empty:
                continue
            for condition in selected_conditions:
                if condition not in experiment.conditions_list:
                    continue
                cond_idx = experiment.conditions_list.index(condition)
                samples = experiment.individual_samples_list[cond_idx]
                for sample in samples:
                    col = f'concentration[{sample}]'
                    if col in df.columns:
                        val = lipid_rows[col].values[0]
                        plot_rows.append({
                            'Lipid': lipid,
                            'Condition': condition,
                            'Concentration': float(val),
                        })

        if not plot_rows:
            raise ValueError("No valid data for selected lipids and conditions")

        plot_df = pd.DataFrame(plot_rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set(style="whitegrid")
        sns.boxplot(
            x="Lipid", y="Concentration", hue="Condition",
            data=plot_df, palette="Set2", ax=ax,
        )
        ax.grid(False)
        ax.set_title(
            "Concentration Distribution for Selected Lipids", fontsize=20
        )
        ax.set_xlabel("Lipid", fontsize=20)
        ax.set_ylabel("Concentration", fontsize=20)
        ax.tick_params(axis='x', labelsize=14, rotation=45)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(title='Condition', loc='upper right')
        fig.tight_layout()

        plt.close(fig)
        return fig

    @staticmethod
    def generate_color_mapping(classes: List[str]) -> Dict[str, str]:
        """Map each lipid class to a consistent color.

        Uses Plotly qualitative color palette, cycling if needed.

        Args:
            classes: List of lipid class names.

        Returns:
            Dict mapping class name to hex color string.
        """
        return generate_class_color_mapping(classes)

    @staticmethod
    def get_most_abundant_lipid(
        df: pd.DataFrame, selected_class: str
    ) -> Optional[str]:
        """Find the most abundant lipid in a class.

        Args:
            df: DataFrame with LipidMolec and concentration columns.
            selected_class: Class to filter by.

        Returns:
            Lipid name with highest total concentration, or None.
        """
        class_df = df[df['ClassKey'] == selected_class]
        if class_df.empty:
            return None

        conc_cols = [c for c in class_df.columns if c.startswith('concentration[')]
        if not conc_cols:
            return None

        totals = class_df[conc_cols].sum(axis=1)
        idx = totals.idxmax()
        return class_df.loc[idx, 'LipidMolec']


# ── Private helpers ────────────────────────────────────────────────────


def _safe_numeric_values(row: pd.Series, cols: List[str]) -> np.ndarray:
    """Extract numeric values from row for given columns, coercing types."""
    vals = []
    for col in cols:
        if col in row.index:
            v = row[col]
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                continue
    return np.array(vals, dtype=float)


def _compute_zero_adj_means(
    ctrl: np.ndarray, exp: np.ndarray
) -> Tuple[float, float]:
    """Compute zero-adjusted means for control and experimental groups."""
    ctrl_clean = ctrl[~np.isnan(ctrl)]
    exp_clean = exp[~np.isnan(exp)]

    all_vals = np.concatenate([ctrl_clean, exp_clean])
    positive = all_vals[all_vals > 0]
    if len(positive) == 0:
        return 0.0, 0.0

    small = float(positive.min()) / 10
    ctrl_adj = np.maximum(ctrl_clean, small)
    exp_adj = np.maximum(exp_clean, small)

    return float(np.mean(ctrl_adj)), float(np.mean(exp_adj))


def _determine_exclusion_reason(
    row: pd.Series, ctrl_cols: List[str], exp_cols: List[str]
) -> str:
    """Determine why a lipid was excluded from statistical testing."""
    ctrl_vals = _safe_numeric_values(row, ctrl_cols)
    exp_vals = _safe_numeric_values(row, exp_cols)

    ctrl_clean = ctrl_vals[~np.isnan(ctrl_vals)]
    exp_clean = exp_vals[~np.isnan(exp_vals)]

    if len(ctrl_clean) == 0:
        return "No valid control values"
    if len(exp_clean) == 0:
        return "No valid experimental values"
    if len(ctrl_clean) < 2:
        return "Insufficient control replicates"
    if len(exp_clean) < 2:
        return "Insufficient experimental replicates"
    if np.all(ctrl_clean == 0) and np.all(exp_clean == 0):
        return "All values are zero"
    if np.all(ctrl_clean == 0):
        return "All control values are zero"
    if np.all(exp_clean == 0):
        return "All experimental values are zero"
    return "Statistical test failed"


def _filter_significant(
    vdf: pd.DataFrame,
    fc_threshold: float,
    q_threshold: float,
    p_col: str,
    hide_non_sig: bool,
) -> pd.DataFrame:
    """Filter volcano DataFrame to significant points if requested."""
    if not hide_non_sig:
        return vdf
    return vdf[
        ((vdf['FoldChange'] < -fc_threshold) |
         (vdf['FoldChange'] > fc_threshold)) &
        (vdf[p_col] >= q_threshold)
    ]


def _add_threshold_lines(
    fig: go.Figure,
    display_df: pd.DataFrame,
    p_col: str,
    q_threshold: float,
    fc_threshold: float,
) -> None:
    """Add horizontal p-value and vertical fold-change threshold lines."""
    x_min = display_df['FoldChange'].min()
    x_max = display_df['FoldChange'].max()
    y_max = display_df[p_col].max()

    # Horizontal p-value threshold
    fig.add_shape(
        type="line",
        x0=x_min, x1=x_max,
        y0=q_threshold, y1=q_threshold,
        line=dict(
            dash=THRESHOLD_LINE_DASH,
            color=THRESHOLD_LINE_COLOR,
            width=THRESHOLD_LINE_WIDTH,
        ),
    )

    # Vertical fold-change thresholds
    for x_val in (-fc_threshold, fc_threshold):
        fig.add_shape(
            type="line",
            x0=x_val, x1=x_val,
            y0=0, y1=y_max,
            line=dict(
                dash=THRESHOLD_LINE_DASH,
                color=THRESHOLD_LINE_COLOR,
                width=THRESHOLD_LINE_WIDTH,
            ),
        )


def _add_labels(
    fig: go.Figure,
    vdf: pd.DataFrame,
    p_col: str,
    color_mapping: Dict[str, str],
    lipids_to_label: List[str],
    custom_positions: Optional[Dict[str, Tuple[float, float]]],
) -> None:
    """Add collision-aware labels for the specified lipids."""
    sorted_df = vdf[vdf['LipidMolec'].isin(lipids_to_label)]
    sorted_df = sorted_df.sort_values(p_col, ascending=False)
    placed_boxes: List[Dict[str, float]] = []
    custom_positions = custom_positions or {}

    for _, row in sorted_df.iterrows():
        text = row['LipidMolec']
        point_x = row['FoldChange']
        point_y = row[p_col]
        color = color_mapping.get(row['ClassKey'], '#000000')

        custom_x, custom_y = custom_positions.get(text, (0.0, 0.0))
        w = len(text) * LABEL_CHAR_WIDTH

        placed = False
        for dx, dy, align in _LABEL_CANDIDATES:
            label_x = point_x + dx + custom_x
            label_y = point_y + dy + custom_y

            if align == 'left':
                left, right = label_x, label_x + w
            else:
                left, right = label_x - w, label_x

            bottom = label_y - LABEL_HEIGHT / 2
            top = label_y + LABEL_HEIGHT / 2

            if not _has_overlap(placed_boxes, left, right, bottom, top):
                _place_label(fig, text, label_x, label_y, point_x, point_y,
                             color, align)
                placed_boxes.append({
                    'left': left - LABEL_BUFFER,
                    'right': right + LABEL_BUFFER,
                    'bottom': bottom - LABEL_BUFFER,
                    'top': top + LABEL_BUFFER,
                })
                placed = True
                break

        if not placed:
            # Fallback: above with side-dependent alignment
            align_fb = 'left' if point_x > 0 else 'right'
            offset = 0.1 if align_fb == 'left' else -0.1
            _place_label(
                fig, text,
                point_x + offset + custom_x,
                point_y + 0.2 + custom_y,
                point_x, point_y, color, align_fb,
            )


def _has_overlap(
    boxes: List[Dict[str, float]],
    left: float, right: float, bottom: float, top: float,
) -> bool:
    """Check if a bounding box overlaps with any placed boxes."""
    for box in boxes:
        if not (right < box['left'] - LABEL_BUFFER or
                left > box['right'] + LABEL_BUFFER or
                top < box['bottom'] - LABEL_BUFFER or
                bottom > box['top'] + LABEL_BUFFER):
            return True
    return False


def _place_label(
    fig: go.Figure,
    text: str,
    label_x: float, label_y: float,
    point_x: float, point_y: float,
    color: str, align: str,
) -> None:
    """Place a text label with an arrow annotation on the figure."""
    fig.add_annotation(
        x=label_x, y=label_y,
        text=text, showarrow=False,
        font=dict(color=color, size=LABEL_FONT_SIZE),
        align=align,
    )
    fig.add_annotation(
        x=point_x, y=point_y,
        ax=label_x, ay=label_y,
        axref='x', ayref='y',
        text='', showarrow=True,
        arrowhead=ARROW_HEAD,
        arrowsize=1,
        arrowwidth=ARROW_WIDTH,
        arrowcolor='black',
    )
