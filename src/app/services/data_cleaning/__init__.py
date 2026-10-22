"""
Data cleaning service package.
Provides format-specific cleaners for lipidomic data.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Type

from ...models.experiment import ExperimentConfig
from ..format_detection import DataFormat

from .configs import GradeFilterConfig, QualityFilterConfig, CleaningResult
from .exceptions import DataCleaningError, ConfigurationError, EmptyDataError
from .base import BaseDataCleaner
from .lipidsearch import LipidSearchCleaner, ADDUCT_NOTE_PREFIX
from .msdial import MSDIALCleaner
from .generic import GenericCleaner


# Registry mapping DataFormat to its cleaner class; dispatch calls
# cleaner_cls.clean(df, experiment, config). A new format also needs its
# DataFormat value and an entry in constants.FORMAT_OPTIONS.
_CLEANER_REGISTRY: Dict[DataFormat, Type[BaseDataCleaner]] = {
    DataFormat.LIPIDSEARCH: LipidSearchCleaner,
    DataFormat.MSDIAL: MSDIALCleaner,
    DataFormat.GENERIC: GenericCleaner,
    DataFormat.METABOLOMICS_WORKBENCH: GenericCleaner,
    DataFormat.UNKNOWN: GenericCleaner,
}


class DataCleaningService:
    """
    Main service for cleaning lipidomic data.

    Dispatches to format-specific cleaners based on data format.
    All methods are stateless static methods.

    Supported formats:
    - LipidSearch (grade filtering, AUC selection)
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

        # Each format's own config; the other is meaningless to its cleaner.
        config = {
            LipidSearchCleaner: grade_config,
            MSDIALCleaner: quality_config,
        }.get(cleaner_cls)
        cleaned_df, messages = cleaner_cls.clean(df, experiment, config)

        # Extract internal standards
        cleaned_df, internal_standards_df = BaseDataCleaner.extract_internal_standards(
            cleaned_df
        )

        return CleaningResult(
            cleaned_df=cleaned_df,
            internal_standards_df=internal_standards_df,
            filter_messages=messages
        )


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
    'ADDUCT_NOTE_PREFIX',
    'MSDIALCleaner',
    'GenericCleaner',
]
