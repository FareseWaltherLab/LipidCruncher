"""
Abundance pie chart plotting service.

Creates pie charts showing lipid class proportions per condition, or the
proportions of the species within one class.

Pure logic — no Streamlit dependencies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.services.plotting._shared import BASE_LAYOUT, CLASS_COLORS

CHART_WIDTH = 450
CHART_HEIGHT = 300

# Threshold for switching to scientific notation
SCIENTIFIC_NOTATION_THRESHOLD = 0.001

LEGEND_FONT_SIZE = 14
# The chart grows to fit this many legend rows (of about LEGEND_ROW_PX each);
# a longer legend scrolls.
MAX_LEGEND_ROWS = 30
LEGEND_ROW_PX = 21
# Slices smaller than this get no border: on a class with many low-abundance
# species the borders would merge into a black wedge.
BORDER_MIN_PERCENT = 1.0


@dataclass
class PieChartData:
    """Computed abundance data for pie chart rendering.

    Attributes:
        abundance_df: DataFrame indexed by ClassKey (or, within a class, by
            LipidMolec) with concentration columns.
        classes: Lipid classes included (or the class's species).
        lipid_class: The class whose species are the slices; None when the
            slices are classes.
    """
    abundance_df: pd.DataFrame
    classes: List[str]
    lipid_class: Optional[str] = None


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
    def calculate_species_abundance(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        lipid_class: str,
    ) -> PieChartData:
        """Concentration of each species of one class across all samples.

        Species are ordered by their total over every sample, most abundant
        first, so a species keeps its color in every condition's chart.

        Raises:
            ValueError: If the class has no species or there are no sample
                columns.
        """
        sample_cols = [
            f"concentration[{s}]" for s in experiment.full_samples_list
            if f"concentration[{s}]" in df.columns
        ]
        if not sample_cols:
            raise ValueError("No sample columns found in the dataset")

        filtered = df[df['ClassKey'] == lipid_class]
        if filtered.empty:
            raise ValueError(f"No species found for class {lipid_class}")

        grouped = filtered.groupby('LipidMolec', sort=False)[sample_cols].sum()
        grouped = grouped.loc[
            grouped.sum(axis=1).sort_values(ascending=False, kind='stable').index
        ]

        return PieChartData(
            abundance_df=grouped,
            classes=grouped.index.tolist(),
            lipid_class=lipid_class,
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
            marker=dict(colors=colors, line=dict(
                color='black',
                width=[1 if pct >= BORDER_MIN_PERCENT else 0 for pct in percentages],
            )),
            textinfo='none',
            hovertemplate='%{label}<extra></extra>',
            sort=False,  # Already sorted
        )])

        if pie_data.lipid_class is None:
            title = f'Total Abundance Pie Chart - {condition}'
            legend_title = "Lipid Classes"
            label_column = 'ClassKey'
        else:
            title = f'{pie_data.lipid_class} Species Distribution - {condition}'
            legend_title = f"{pie_data.lipid_class} Species"
            label_column = 'LipidMolec'

        fig.update_layout(
            title=dict(text=title, font=dict(color='black')),
            legend=dict(
                title=dict(text=legend_title),
                font=dict(size=LEGEND_FONT_SIZE),
            ),
            margin=dict(l=10, r=100, t=40, b=10),
            width=CHART_WIDTH,
            height=max(
                CHART_HEIGHT,
                min(len(sorted_labels), MAX_LEGEND_ROWS) * LEGEND_ROW_PX + 60,
            ),
            **BASE_LAYOUT,
        )

        # Build summary DataFrame for export
        summary_df = pd.DataFrame({
            label_column: sorted_labels,
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
