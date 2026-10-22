"""Cached Module 1 workflow wrappers (ingestion, standards, zero filter,
normalization), part of the StreamlitAdapter facade."""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from ..services.format_detection import DataFormat
from ..services.plotting.standards_plotter import StandardsPlotterService
from ..services.standards import StandardsProcessingResult, StandardsService
from ..services.zero_filtering import (
    ZeroFilterConfig, ZeroFilteringResult, ZeroFilteringService,
)
from ..workflows.data_ingestion import DataIngestionWorkflow, IngestionConfig, IngestionResult
from ..workflows.normalization import NormalizationWorkflow, NormalizationWorkflowConfig, NormalizationWorkflowResult


class Module1Cache:
    """Module 1's cached workflow calls; StreamlitAdapter inherits them."""

    @staticmethod
    @st.cache_data(
        show_spinner="Processing data...",
        hash_funcs={ExperimentConfig: hash, GradeFilterConfig: hash, QualityFilterConfig: hash},
    )
    def run_ingestion(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        data_format: Optional[DataFormat] = None,
        grade_config: Optional[GradeFilterConfig] = None,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> IngestionResult:
        """
        Cached data ingestion workflow wrapper.

        Args:
            df: Raw DataFrame to process
            experiment: Experiment configuration
            data_format: Data format enum
            grade_config: Grade filter config (LipidSearch only)
            quality_config: Quality filter config (MS-DIAL only)

        Returns:
            IngestionResult with cleaned data, standards, format, and validation info.
        """
        config = IngestionConfig(
            experiment=experiment,
            data_format=data_format,
            grade_config=grade_config,
            quality_config=quality_config,
        )

        return DataIngestionWorkflow.run(df, config)

    @staticmethod
    @st.cache_data(
        show_spinner="Normalizing data...",
        hash_funcs={ExperimentConfig: hash, NormalizationConfig: hash},
    )
    def run_normalization(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        normalization: NormalizationConfig,
        data_format: DataFormat = DataFormat.GENERIC,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> NormalizationWorkflowResult:
        """
        Cached normalization workflow wrapper.

        Args:
            df: Cleaned DataFrame to normalize
            experiment: Experiment configuration
            normalization: Normalization configuration
            data_format: Data format enum
            intsta_df: Internal standards DataFrame

        Returns:
            NormalizationWorkflowResult with normalized data and metadata.
        """
        workflow_config = NormalizationWorkflowConfig(
            experiment=experiment,
            normalization=normalization,
            data_format=data_format
        )

        return NormalizationWorkflow.run(
            df=df,
            config=workflow_config,
            intsta_df=intsta_df
        )

    @staticmethod
    @st.cache_data(
        show_spinner=False,
        hash_funcs={ExperimentConfig: hash, ZeroFilterConfig: dataclasses.astuple},
    )
    def run_zero_filter(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        config: ZeroFilterConfig,
        bqc_label: Optional[str] = None,
    ) -> ZeroFilteringResult:
        """Cached zero filtering."""
        return ZeroFilteringService.filter_zeros(
            df=df, experiment=experiment, config=config, bqc_label=bqc_label,
        )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def run_process_standards_file(
        uploaded_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        standards_in_dataset: bool,
    ) -> StandardsProcessingResult:
        """Cached processing of an uploaded internal-standards table."""
        return StandardsService.process_standards_file(
            uploaded_df=uploaded_df, cleaned_df=cleaned_df,
            standards_in_dataset=standards_in_dataset,
        )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def run_standards_plots(
        intsta_df: pd.DataFrame,
        samples: List[str],
        sample_conditions: Optional[List[str]] = None,
        display_names: Optional[Dict[str, str]] = None,
    ) -> List[go.Figure]:
        """Cached internal-standards consistency plots."""
        return StandardsPlotterService.create_consistency_plots(
            intsta_df, samples,
            sample_conditions=sample_conditions, display_names=display_names,
        )
