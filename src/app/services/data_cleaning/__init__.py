"""
Data cleaning service package.
Provides format-specific cleaners for lipidomic data.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from ..format_detection import DataFormat

from .configs import GradeFilterConfig, QualityFilterConfig, CleaningResult
from .exceptions import DataCleaningError, ConfigurationError, EmptyDataError
from .base import BaseDataCleaner
from .lipidsearch import LipidSearchCleaner
from .msdial import MSDIALCleaner
from .generic import GenericCleaner


# Registry mapping DataFormat to cleaning function.
# Each entry is a callable(df, experiment, **kwargs) -> Tuple[DataFrame, List[str]].
# Adding a new format only requires adding an entry here.
_CLEANER_REGISTRY: Dict[DataFormat, type] = {
    DataFormat.LIPIDSEARCH: LipidSearchCleaner,
    DataFormat.MSDIAL: MSDIALCleaner,
    DataFormat.GENERIC: GenericCleaner,
    DataFormat.METABOLOMICS_WORKBENCH: GenericCleaner,
    DataFormat.UNKNOWN: GenericCleaner,
}

# Maps DataFormat to the keyword argument name for its format-specific config.
_FORMAT_CONFIG_KEY: Dict[DataFormat, str] = {
    DataFormat.LIPIDSEARCH: 'grade_config',
    DataFormat.MSDIAL: 'quality_config',
}


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
            DataCleaningError: If dataset is empty or becomes empty after cleaning.
        """
        # Look up cleaner from registry
        cleaner_cls = _CLEANER_REGISTRY.get(data_format, GenericCleaner)

        # Build format-specific kwargs
        config_map = {
            'grade_config': grade_config,
            'quality_config': quality_config,
        }
        config_key = _FORMAT_CONFIG_KEY.get(data_format)
        kwargs = {}
        if config_key and config_map[config_key] is not None:
            kwargs[config_key] = config_map[config_key]

        cleaned_df, messages = cleaner_cls.clean(df, experiment, **kwargs)

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
    'DataCleaningError',
    'ConfigurationError',
    'EmptyDataError',
    'BaseDataCleaner',
    'LipidSearchCleaner',
    'MSDIALCleaner',
    'GenericCleaner',
]