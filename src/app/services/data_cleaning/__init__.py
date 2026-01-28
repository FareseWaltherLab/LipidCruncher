"""
Data cleaning service package.
Provides format-specific cleaners for lipidomic data.
"""
import pandas as pd
from typing import Optional

from ...models.experiment import ExperimentConfig
from ..format_detection import DataFormat

from .configs import GradeFilterConfig, QualityFilterConfig, CleaningResult
from .base import BaseDataCleaner
from .lipidsearch import LipidSearchCleaner
from .msdial import MSDIALCleaner
from .generic import GenericCleaner


class DataCleaningService:
    """
    Main service for cleaning lipidomic data.

    Dispatches to format-specific cleaners based on data format.
    All methods are stateless static methods.

    Supported formats:
    - LipidSearch 5.0 (grade filtering, AUC selection)
    - MS-DIAL (quality filtering, score-based deduplication)
    - Generic (basic cleaning)
    - Metabolomics Workbench (same as Generic)
    """

    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        data_format: DataFormat,
        grade_config: Optional[GradeFilterConfig] = None,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> CleaningResult:
        """
        Clean data using appropriate cleaner based on the format.

        Args:
            df: Input dataframe (already column-standardized).
            experiment: Experiment configuration with sample information.
            data_format: Format type (from FormatDetectionService).
            grade_config: Optional grade filtering config (LipidSearch only).
            quality_config: Optional quality filtering config (MS-DIAL only).

        Returns:
            CleaningResult containing cleaned_df, internal_standards_df, and messages.

        Raises:
            ValueError: If dataset is empty or becomes empty after cleaning.
        """
        # Dispatch to format-specific cleaner
        if data_format == DataFormat.LIPIDSEARCH:
            cleaned_df, messages = LipidSearchCleaner.clean(df, experiment, grade_config)
        elif data_format == DataFormat.MSDIAL:
            cleaned_df, messages = MSDIALCleaner.clean(df, experiment, quality_config)
        else:
            # Generic and Metabolomics Workbench use same cleaner
            cleaned_df, messages = GenericCleaner.clean(df, experiment)

        # Extract internal standards
        cleaned_df, internal_standards_df = BaseDataCleaner.extract_internal_standards(
            cleaned_df
        )

        return CleaningResult(
            cleaned_df=cleaned_df,
            internal_standards_df=internal_standards_df,
            filter_messages=messages
        )

    @staticmethod
    def clean_lipidsearch(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        grade_config: Optional[GradeFilterConfig] = None
    ) -> CleaningResult:
        """
        Clean LipidSearch 5.0 format data.

        Convenience method for direct LipidSearch cleaning.
        """
        return DataCleaningService.clean_data(
            df, experiment, DataFormat.LIPIDSEARCH, grade_config=grade_config
        )

    @staticmethod
    def clean_msdial(
        df: pd.DataFrame,
        experiment: ExperimentConfig,
        quality_config: Optional[QualityFilterConfig] = None
    ) -> CleaningResult:
        """
        Clean MS-DIAL format data.

        Convenience method for direct MS-DIAL cleaning.
        """
        return DataCleaningService.clean_data(
            df, experiment, DataFormat.MSDIAL, quality_config=quality_config
        )

    @staticmethod
    def clean_generic(
        df: pd.DataFrame,
        experiment: ExperimentConfig
    ) -> CleaningResult:
        """
        Clean Generic format data.

        Convenience method for direct Generic cleaning.
        """
        return DataCleaningService.clean_data(df, experiment, DataFormat.GENERIC)

    @staticmethod
    def extract_internal_standards(df: pd.DataFrame):
        """
        Extract internal standards from a cleaned dataframe.

        Delegates to BaseDataCleaner.extract_internal_standards().
        """
        return BaseDataCleaner.extract_internal_standards(df)


# Public API
__all__ = [
    'DataCleaningService',
    'GradeFilterConfig',
    'QualityFilterConfig',
    'CleaningResult',
    'BaseDataCleaner',
    'LipidSearchCleaner',
    'MSDIALCleaner',
    'GenericCleaner',
]
