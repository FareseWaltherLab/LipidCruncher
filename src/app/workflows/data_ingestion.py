"""
Data ingestion workflow.

Orchestrates the complete data ingestion pipeline:
File upload → Format detection → Data cleaning → Standards validation

Zero filtering is not part of it: the Data Processing page applies it
afterwards (``ui/zero_filtering.py``), with thresholds the user can adjust.

This workflow handles the business logic flow while leaving UI concerns to the UI layer.
All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..services.exceptions import ServiceError
from ..services.data_cleaning import (
    DataCleaningService,
    GradeFilterConfig,
    QualityFilterConfig,
    DataCleaningError,
)
from ..services.standards import (
    StandardsService
)


@dataclass
class IngestionResult:
    """
    Complete result of data ingestion workflow.

    Contains all intermediate and final results from the ingestion pipeline.
    """
    # Detection results
    detected_format: DataFormat

    # Cleaning results
    cleaned_df: Optional[pd.DataFrame] = None
    internal_standards_df: Optional[pd.DataFrame] = None
    cleaning_messages: List[str] = field(default_factory=list)

    # Validation status
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class IngestionConfig:
    """
    Configuration for the data ingestion workflow.

    Collects all user choices that affect the ingestion process.
    """
    # Required settings
    experiment: ExperimentConfig

    # Format-specific settings
    data_format: Optional[DataFormat] = None  # None = auto-detect
    grade_config: Optional[GradeFilterConfig] = None  # LipidSearch only
    quality_config: Optional[QualityFilterConfig] = None  # MS-DIAL only


class DataIngestionWorkflow:
    """
    Orchestrates the complete data ingestion pipeline.

    This workflow coordinates format detection, data cleaning and internal
    standards validation in the correct order.

    All methods are static - no instance state is stored.
    """

    @staticmethod
    def run(
        df: pd.DataFrame,
        config: IngestionConfig
    ) -> IngestionResult:
        """
        Execute the complete data ingestion workflow.

        Pipeline:
        1. Detect format (if not specified)
        2. Clean data (format-specific)
        3. Validate internal standards

        Args:
            df: Raw input DataFrame
            config: Ingestion configuration with experiment and settings

        Returns:
            IngestionResult with all intermediate and final results. Invalid
            or empty input is reported as is_valid=False with
            validation_errors, not raised.
        """
        result = IngestionResult(detected_format=DataFormat.UNKNOWN)

        # Step 1: Detect format
        data_format = config.data_format
        if data_format is None:
            data_format = FormatDetectionService.detect_format(df)

        result.detected_format = data_format

        if data_format == DataFormat.UNKNOWN:
            result.is_valid = False
            available_cols = ', '.join(list(df.columns)[:15])
            result.validation_errors.append(
                f"Could not detect data format. Found columns: [{available_cols}]. "
                "Please ensure your file matches one of the supported formats "
                "(LipidSearch, MS-DIAL, Generic, or Metabolomics Workbench)."
            )
            return result

        # Step 2: Clean data
        try:
            cleaning_result = DataCleaningService.clean_data(
                df=df,
                experiment=config.experiment,
                data_format=data_format,
                grade_config=config.grade_config,
                quality_config=config.quality_config
            )
            result.cleaned_df = cleaning_result.cleaned_df
            result.internal_standards_df = cleaning_result.internal_standards_df
            result.cleaning_messages = cleaning_result.filter_messages

        except (DataCleaningError, ServiceError, ValueError, KeyError) as e:
            result.is_valid = False
            result.validation_errors.append(str(e))
            return result

        # Step 3: Validate internal standards
        if result.internal_standards_df is not None and not result.internal_standards_df.empty:
            validation = StandardsService.validate_standards(
                result.internal_standards_df,
                config.experiment.full_samples_list
            )
            if not validation.is_valid:
                result.validation_warnings.extend(validation.errors)

        return result

    @staticmethod
    def analysis_frame(
        filtered_df: pd.DataFrame,
        standards_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """The data Module 1 normalizes and shows as its final filtered data.

        The zero-filtered species sorted by class, minus every lipid used as
        an internal standard: a standard is not also analysed as a species,
        whichever way it was chosen (auto-detected standards were already
        taken out by ``run``; ones picked from the dataset or uploaded are
        removed here).

        Args:
            filtered_df: Cleaned, zero-filtered data.
            standards_df: The active internal standards, or None.
        """
        if 'ClassKey' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('ClassKey').reset_index(drop=True)
        analysis_df, _ = StandardsService.remove_standards_from_dataset(
            filtered_df, standards_df,
        )
        return analysis_df

    @staticmethod
    def validate_for_format(
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame is suitable for the specified format.

        Args:
            df: DataFrame to validate
            data_format: Expected format

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if df is None or df.empty:
            errors.append("Dataset is empty")
            return False, errors

        # Check for LipidMolec column (required for all formats)
        if 'LipidMolec' not in df.columns:
            available = ', '.join(list(df.columns)[:15])
            errors.append(f"Missing required column: 'LipidMolec'. Available columns: [{available}]")

        # Format-specific validation
        if data_format == DataFormat.LIPIDSEARCH:
            required = {'ClassKey', 'CalcMass', 'BaseRt', 'TotalGrade', 'FAKey'}
            missing = required - set(df.columns)
            if missing:
                available = ', '.join(list(df.columns)[:15])
                errors.append(
                    f"Missing {len(missing)} required LipidSearch column(s): "
                    f"{', '.join(sorted(missing))}. Available columns: [{available}]"
                )

        elif data_format == DataFormat.MSDIAL:
            if 'Ontology' not in df.columns:
                available = ', '.join(list(df.columns)[:15])
                errors.append(
                    f"Missing MS-DIAL column: 'Ontology' (used as ClassKey). "
                    f"Available columns: [{available}]"
                )

        return len(errors) == 0, errors

    @staticmethod
    def get_sample_columns(
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> List[str]:
        """
        Get the intensity/sample columns from the DataFrame.

        Args:
            df: DataFrame
            data_format: Data format

        Returns:
            List of sample column names
        """
        if data_format == DataFormat.LIPIDSEARCH:
            # LipidSearch uses MeanArea[*] columns
            return [col for col in df.columns if col.startswith('MeanArea[')]

        elif data_format == DataFormat.MSDIAL:
            # MS-DIAL has specific metadata columns; rest are samples
            from ..services.format_detection import FormatDetectionService
            metadata = FormatDetectionService.MSDIAL_METADATA_COLUMNS
            return [col for col in df.columns if col not in metadata]

        else:
            # Generic format: columns after LipidMolec and ClassKey
            metadata = {'LipidMolec', 'ClassKey'}
            return [col for col in df.columns if col not in metadata]
