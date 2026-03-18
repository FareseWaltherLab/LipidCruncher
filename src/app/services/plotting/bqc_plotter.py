"""
BQC quality check plotting service.

Creates CoV scatter plots for Batch Quality Control assessment.
Handles data preparation (CoV/mean calculation) and plot rendering.

Pure logic — no Streamlit dependencies.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class BQCPlotterService:
    """Creates CoV scatter plots for BQC quality assessment."""

    @staticmethod
    def prepare_dataframe_for_plot(
        dataframe: pd.DataFrame,
        concentration_columns: List[str],
    ) -> pd.DataFrame:
        """Prepare DataFrame for CoV scatter plot.

        Calculates CoV and log10(mean) for each lipid across the given
        concentration columns.

        Args:
            dataframe: DataFrame with concentration columns.
            concentration_columns: Column names to use for CoV calculation.

        Returns:
            Copy of dataframe with added 'cov' and 'mean' columns.
            The 'mean' column contains log10-transformed values where mean > 0.
        """
        df = dataframe.copy()

        df['cov'] = df[concentration_columns].apply(
            _calculate_coefficient_of_variation, axis=1
        )
        df['mean'] = df[concentration_columns].apply(
            _calculate_mean_including_zeros, axis=1
        )

        # Log10-transform valid means
        valid_mask = df['mean'].notnull() & (df['mean'] > 0)
        df.loc[valid_mask, 'mean'] = np.log10(df.loc[valid_mask, 'mean'])

        return df

    @staticmethod
    def generate_cov_plot_data(
        dataframe: pd.DataFrame,
        individual_samples_list: List[List[str]],
        bqc_sample_index: int,
    ) -> pd.DataFrame:
        """Prepare BQC data for CoV scatter plot.

        Extracts BQC samples and calculates CoV/mean metrics.

        Args:
            dataframe: Full lipidomics DataFrame.
            individual_samples_list: Samples grouped by condition.
            bqc_sample_index: Index of BQC condition.

        Returns:
            Prepared DataFrame with 'cov' and 'mean' columns.
        """
        bqc_samples = individual_samples_list[bqc_sample_index]
        conc_cols = [f'concentration[{s}]' for s in bqc_samples]
        return BQCPlotterService.prepare_dataframe_for_plot(
            dataframe.copy(), conc_cols
        )

    @staticmethod
    def generate_and_display_cov_plot(
        dataframe: pd.DataFrame,
        experiment,
        bqc_sample_index: int,
        cov_threshold: float = 30,
    ) -> Tuple[go.Figure, pd.DataFrame, float, pd.DataFrame]:
        """Full pipeline: prepare data and create CoV scatter plot.

        Args:
            dataframe: Full lipidomics DataFrame.
            experiment: Object with individual_samples_list attribute.
            bqc_sample_index: Index of BQC condition.
            cov_threshold: CoV threshold for coloring (default 30%).

        Returns:
            Tuple of (figure, prepared_df, reliable_data_percent, empty_df).
        """
        prepared_df = BQCPlotterService.generate_cov_plot_data(
            dataframe, experiment.individual_samples_list, bqc_sample_index
        )

        mean_vals = prepared_df['mean'].values
        cov_vals = prepared_df['cov'].values
        species = prepared_df['LipidMolec'].values

        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            mean_vals, cov_vals, species, cov_threshold
        )

        reliable_count = len(prepared_df[prepared_df['cov'] < cov_threshold])
        reliable_pct = round(reliable_count / len(prepared_df) * 100, 1)

        # Empty filtered_lipids (unused, kept for interface consistency)
        filtered_lipids = pd.DataFrame()

        return fig, prepared_df, reliable_pct, filtered_lipids

    @staticmethod
    def create_cov_scatter_plot_with_threshold(
        mean_concentrations: np.ndarray,
        coefficients_of_variation: np.ndarray,
        lipid_species: np.ndarray,
        cov_threshold: float,
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Create CoV scatter plot with threshold coloring."""
        cov_df = BQCPlotterService._prepare_cov_data(
            mean_concentrations, coefficients_of_variation, lipid_species
        )

        fig = go.Figure()
        BQCPlotterService._add_cov_scatter_traces(fig, cov_df, cov_threshold)
        BQCPlotterService._apply_cov_layout(fig, cov_threshold)
        return fig, cov_df

    @staticmethod
    def _prepare_cov_data(
        mean_concentrations: np.ndarray,
        coefficients_of_variation: np.ndarray,
        lipid_species: np.ndarray,
    ) -> pd.DataFrame:
        """Filter NaN values and build CoV DataFrame."""
        mean_concentrations = np.array(mean_concentrations)
        coefficients_of_variation = np.array(coefficients_of_variation)
        lipid_species = np.array(lipid_species)

        valid = ~(np.isnan(mean_concentrations) | np.isnan(coefficients_of_variation))
        return pd.DataFrame({
            'Mean_concentration': mean_concentrations[valid],
            'CoV': coefficients_of_variation[valid],
            'Species': lipid_species[valid],
        })

    @staticmethod
    def _add_cov_scatter_traces(
        fig: go.Figure, cov_df: pd.DataFrame, cov_threshold: float
    ) -> None:
        """Add below/above threshold scatter traces and threshold line."""
        hover_tpl = (
            '<b>Species:</b> %{text}<br>'
            '<b>Mean concentration:</b> %{x:.4f}<br>'
            '<b>CoV:</b> %{y:.2f}%<br>'
            '<extra></extra>'
        )

        below = cov_df[cov_df['CoV'] <= cov_threshold]
        above = cov_df[cov_df['CoV'] > cov_threshold]

        if len(below) > 0:
            fig.add_trace(go.Scatter(
                x=below['Mean_concentration'], y=below['CoV'],
                mode='markers', marker=dict(size=5, color='blue'),
                text=below['Species'], hovertemplate=hover_tpl,
            ))
        if len(above) > 0:
            fig.add_trace(go.Scatter(
                x=above['Mean_concentration'], y=above['CoV'],
                mode='markers', marker=dict(size=5, color='red'),
                text=above['Species'], hovertemplate=hover_tpl,
            ))

        fig.add_hline(
            y=cov_threshold, line_dash='solid', line_color='black',
            line_width=2,
            annotation_text=f'Threshold: {cov_threshold}%',
            annotation_position='top right',
            annotation_font=dict(size=14, color='black'),
        )

    @staticmethod
    def _apply_cov_layout(fig: go.Figure, cov_threshold: float) -> None:
        """Apply CoV scatter plot layout."""
        axis_style = dict(
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=14, color='black'),
            tickcolor='black', showline=True, linewidth=2,
            linecolor='black', mirror=True, ticks='outside',
        )
        fig.update_layout(
            title=dict(text='CoV - All lipid Species', font=dict(size=24, color='black')),
            xaxis_title='Log10 of Mean BQC Concentration',
            yaxis_title='CoV(%)',
            xaxis=axis_style,
            yaxis=axis_style,
            plot_bgcolor='white', paper_bgcolor='white',
            showlegend=False,
            margin=dict(t=50, r=50, b=50, l=50),
            font=dict(color='black'),
        )
        fig.update_xaxes(zeroline=False)
        fig.update_yaxes(zeroline=False)


def _calculate_coefficient_of_variation(numbers) -> Optional[float]:
    """Calculate CoV as percentage. Includes zeros. Returns None if < 2 values or mean=0."""
    numbers = np.array(numbers)
    if len(numbers) >= 2:
        mean_val = np.mean(numbers)
        if mean_val == 0:
            return None
        return np.std(numbers, ddof=1) / mean_val * 100
    return None


def _calculate_mean_including_zeros(numbers) -> Optional[float]:
    """Calculate mean including zeros. Returns None if < 2 values."""
    numbers = np.array(numbers)
    if len(numbers) >= 2:
        return np.mean(numbers)
    return None
