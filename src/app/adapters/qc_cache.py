"""Cached Quality Check (Module 2) computations, part of the StreamlitAdapter facade."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..models.experiment import ExperimentConfig
from ..services.plotting.box_plot import BoxPlotService
from ..services.plotting.bqc_plotter import BQCPlotterService
from ..services.plotting.correlation import CorrelationPlotterService
from ..services.plotting.pca import PCAPlotterService
from ..services.plotting.retention_time import RetentionTimePlotterService
from ..services.validation import get_matching_concentration_columns


class QualityCheckCache:
    """Quality Check's cached computations; StreamlitAdapter inherits them."""

    @staticmethod
    @st.cache_data(
        show_spinner="Computing box plots...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_box_plots(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[go.Figure, go.Figure, pd.DataFrame, List[float], List[str]]:
        """Cached box plot computation.

        ``display_names`` (``{s-label -> text shown}``) labels the samples in
        both figures; the returned data stays keyed by s-label.

        Returns:
            Tuple of (missing_values_fig, box_plot_fig, mean_area_df,
                      missing_pct_list, current_samples)
        """
        samples = get_matching_concentration_columns(df, experiment)
        mean_area_df = BoxPlotService.create_mean_area_df(df, samples)
        missing_pct = BoxPlotService.calculate_missing_values_percentage(
            mean_area_df
        )
        fig1 = BoxPlotService.plot_missing_values(
            samples, missing_pct,
            experiment.conditions_list, experiment.individual_samples_list,
            display_names=display_names,
        )
        fig2 = BoxPlotService.plot_box_plot(
            mean_area_df, samples,
            experiment.conditions_list, experiment.individual_samples_list,
            display_names=display_names,
        )
        return fig1, fig2, mean_area_df, missing_pct, samples

    @staticmethod
    @st.cache_data(
        show_spinner="Computing BQC quality assessment...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_bqc_scatter(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        bqc_sample_index: int,
        cov_threshold: int,
    ) -> Tuple[go.Figure, pd.DataFrame, float]:
        """Cached BQC scatter plot computation.

        Returns:
            Tuple of (scatter_plot, prepared_df, reliable_data_percent)
        """
        return BQCPlotterService.generate_and_display_cov_plot(
            df, experiment, bqc_sample_index, cov_threshold=cov_threshold,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Computing correlation matrix...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_correlation(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        condition_index: int,
        sample_type: str,
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[matplotlib.figure.Figure, pd.DataFrame]:
        """Cached pairwise correlation computation.

        ``display_names`` (``{s-label -> text shown}``) labels the heatmap's
        ticks; ``correlation_df`` stays keyed by s-label.

        Returns:
            Tuple of (matplotlib_fig, correlation_df)
        """
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            df, experiment.individual_samples_list, condition_index,
        )
        correlation_df, v_min, thresh = (
            CorrelationPlotterService.compute_correlation(
                mean_area_df, sample_type,
            )
        )
        condition_name = experiment.conditions_list[condition_index]
        fig = CorrelationPlotterService.render_correlation_plot(
            correlation_df, v_min, thresh, condition_name,
            display_names=display_names,
        )
        return fig, correlation_df

    @staticmethod
    @st.cache_data(
        show_spinner="Computing PCA...",
        hash_funcs={ExperimentConfig: hash},
    )
    def run_pca(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        display_names: Optional[Dict[str, str]] = None,
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """Cached PCA computation.

        ``display_names`` (``{s-label -> text shown}``) labels the hover text;
        ``pca_df`` keeps the s-labels.

        Returns:
            Tuple of (pca_plot, pca_df)
        """
        return PCAPlotterService.plot_pca(
            df, experiment.full_samples_list,
            experiment.extensive_conditions_list,
            display_names=display_names,
        )

    @staticmethod
    @st.cache_data(
        show_spinner="Computing retention time plots...",
    )
    def run_retention_time_single(df: pd.DataFrame) -> List[Tuple[go.Figure, pd.DataFrame]]:
        """Cached single-class retention time plot computation.

        Returns:
            List of (plot, retention_df) tuples, one per RT group.
        """
        return RetentionTimePlotterService.plot_single_retention(df)

    @staticmethod
    @st.cache_data(
        show_spinner="Computing retention time plots...",
    )
    def run_retention_time_multi(
        df: pd.DataFrame,
        selected_classes: List[str],
    ) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
        """Cached multi-class retention time comparison plot.

        Returns:
            Tuple of (plot, retention_df) or (None, None).
        """
        return RetentionTimePlotterService.plot_multi_retention(
            df, selected_classes,
        )
