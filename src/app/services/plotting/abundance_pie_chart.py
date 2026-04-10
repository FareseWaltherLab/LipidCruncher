"""
Abundance pie chart plotting service.

Creates pie charts showing lipid class proportions per condition.

Pure logic — no Streamlit dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import generate_class_color_mapping


# Large qualitative palette for lipid classes (many possible classes)
CLASS_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
]

CHART_WIDTH = 450
CHART_HEIGHT = 300

# Threshold for switching to scientific notation
SCIENTIFIC_NOTATION_THRESHOLD = 0.001


@dataclass
class PieChartData:
    """Computed abundance data for pie chart rendering.

    Attributes:
        abundance_df: DataFrame indexed by ClassKey with concentration columns.
        classes: Lipid classes included.
    """
    abundance_df: pd.DataFrame
    classes: List[str]


class PieChartPlotterService:
    """Creates abundance pie charts showing lipid class proportions."""

    @staticmethod
    def calculate_total_abundance(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        selected_classes: List[str],
    ) -> PieChartData:
        """Sum species concentrations per class across all samples.

        Args:
            df: DataFrame with ClassKey and concentration[s*] columns.
            experiment: Experiment configuration.
            selected_classes: Lipid classes to include.

        Returns:
            PieChartData with aggregated abundance per class.

        Raises:
            ValueError: If no valid data can be computed.
        """
        if selected_classes is None or len(selected_classes) == 0:
            raise ValueError("At least one lipid class must be selected")

        sample_cols = [
            f"concentration[{s}]" for s in experiment.full_samples_list
            if f"concentration[{s}]" in df.columns
        ]
        if not sample_cols:
            raise ValueError("No sample columns found in the dataset")

        filtered = df[df['ClassKey'].isin(selected_classes)]
        if filtered.empty:
            raise ValueError("No data found for the selected classes")

        grouped = filtered.groupby('ClassKey')[sample_cols].sum()

        # Preserve requested class order
        ordered_classes = [c for c in selected_classes if c in grouped.index]
        grouped = grouped.loc[ordered_classes]

        return PieChartData(
            abundance_df=grouped,
            classes=ordered_classes,
        )

    @staticmethod
    def create_pie_chart(
        pie_data: PieChartData,
        condition: str,
        samples: List[str],
        color_mapping: Dict[str, str],
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Create a pie chart for one condition.

        Args:
            pie_data: Pre-computed abundance data.
            condition: Condition label for the chart title.
            samples: Sample names belonging to this condition.
            color_mapping: Dict mapping class name to hex color.

        Returns:
            Tuple of (Plotly Figure, DataFrame used for the chart).

        Raises:
            ValueError: If no sample columns are available.
        """
        abundance_df = pie_data.abundance_df
        sample_cols = [
            f"concentration[{s}]" for s in samples
            if f"concentration[{s}]" in abundance_df.columns
        ]
        if not sample_cols:
            raise ValueError(
                f"No sample columns found for condition: {condition}"
            )

        # Sum across samples for this condition
        totals = abundance_df[sample_cols].sum(axis=1)

        # Sort descending by abundance
        sorted_totals = totals.sort_values(ascending=False)
        sorted_labels = sorted_totals.index.tolist()
        sorted_sizes = sorted_totals.values

        # Calculate percentages with meaningful decimal precision
        total_sum = sorted_sizes.sum()
        if total_sum == 0:
            percentages = np.zeros(len(sorted_sizes))
        else:
            percentages = 100 * sorted_sizes / total_sum

        custom_labels = [
            f'{label} - {_format_percentage(pct)}%'
            for label, pct in zip(sorted_labels, percentages)
        ]

        colors = [color_mapping.get(label, '#333333') for label in sorted_labels]

        fig = go.Figure(data=[go.Pie(
            values=sorted_sizes,
            labels=custom_labels,
            marker=dict(colors=colors),
            textinfo='none',
            hovertemplate='%{label}<extra></extra>',
            sort=False,  # Already sorted
        )])

        fig.update_layout(
            title=dict(
                text=f'Total Abundance Pie Chart - {condition}',
                font=dict(color='black'),
            ),
            legend_title="Lipid Classes",
            margin=dict(l=10, r=100, t=40, b=10),
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            paper_bgcolor='white',
        )

        # Build summary DataFrame for export
        summary_df = pd.DataFrame({
            'ClassKey': sorted_labels,
            'Total Abundance': sorted_sizes,
            'Percentage': percentages,
        })

        return fig, summary_df

    @staticmethod
    def generate_color_mapping(classes: List[str]) -> Dict[str, str]:
        """Map each lipid class to a consistent color.

        Uses the shared class color palette with extended colors for
        pie charts that may show many classes.

        Args:
            classes: List of lipid class labels.

        Returns:
            Dict mapping class name to hex color string.
        """
        # Pie charts use an extended palette (20 colors) since they may
        # show many more classes than other chart types.
        return {
            cls: CLASS_COLORS[i % len(CLASS_COLORS)]
            for i, cls in enumerate(classes)
        }


# ── Private helpers ────────────────────────────────────────────────────


def _format_percentage(percentage: float) -> str:
    """Format percentage to show meaningful decimal precision.

    Args:
        percentage: The percentage value to format.

    Returns:
        Formatted percentage string.
    """
    if percentage == 0:
        return "0.0"

    if percentage >= 1:
        return f"{percentage:.1f}"
    if percentage >= 0.1:
        return f"{percentage:.2f}"
    if percentage >= 0.01:
        return f"{percentage:.3f}"
    if percentage >= SCIENTIFIC_NOTATION_THRESHOLD:
        return f"{percentage:.4f}"
    return f"{percentage:.2e}"
