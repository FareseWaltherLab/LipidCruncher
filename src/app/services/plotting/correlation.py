"""
Correlation heatmap plotting service.

Creates pairwise correlation heatmaps for quality check analysis.

Pure logic — no Streamlit dependencies.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CorrelationPlotterService:
    """Creates pairwise correlation heatmaps."""

    @staticmethod
    def prepare_data_for_correlation(
        df: pd.DataFrame,
        individual_samples_list: List[List[str]],
        condition_index: int,
    ) -> pd.DataFrame:
        """Extract concentration columns for a condition and rename to sample names.

        Args:
            df: DataFrame with concentration[sample] columns.
            individual_samples_list: Samples grouped by condition.
            condition_index: Index of the target condition.

        Returns:
            DataFrame with columns renamed to sample names.
        """
        samples = individual_samples_list[condition_index]
        mean_area_df = df[[f'concentration[{s}]' for s in samples]]
        mean_area_df.columns = samples
        return mean_area_df

    @staticmethod
    def compute_correlation(
        df: pd.DataFrame,
        sample_type: str,
    ) -> Tuple[pd.DataFrame, float, float]:
        """Compute Pearson correlation matrix and thresholds.

        Args:
            df: DataFrame with sample columns.
            sample_type: 'biological replicates' or 'technical replicates'.

        Returns:
            Tuple of (correlation_df, v_min, threshold).
            v_min is the minimum for the heatmap color scale.
            threshold is the quality threshold (0.7 bio, 0.8 tech).
        """
        v_min = 0.5
        thresh = 0.7 if sample_type == 'biological replicates' else 0.8
        correlation_df = df.corr()
        return correlation_df, v_min, thresh

    @staticmethod
    def render_correlation_plot(
        correlation_df: pd.DataFrame,
        v_min: float,
        thresh: float,
        condition: str,
    ) -> plt.Figure:
        """Render a triangle correlation heatmap.

        Args:
            correlation_df: Correlation matrix.
            v_min: Minimum value for color scale.
            thresh: Center value for color scale.
            condition: Condition label for the title.

        Returns:
            Matplotlib Figure with the heatmap.
        """
        fig = plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        sns.set(font_scale=3)
        heatmap = sns.heatmap(
            correlation_df, mask=mask, vmin=v_min, vmax=1,
            center=thresh, annot=False, cmap='RdBu',
            square=False, cbar=True,
        )
        heatmap.set_title(
            f'Triangle Correlation Heatmap - {condition}',
            fontdict={'fontsize': 40},
        )
        plt.close(fig)
        return fig
