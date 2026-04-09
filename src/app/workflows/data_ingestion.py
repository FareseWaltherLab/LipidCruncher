"""
Data ingestion workflow.

Orchestrates the complete data ingestion pipeline:
File upload → Format detection → Data cleaning → Zero filtering → Standards extraction

This workflow handles the business logic flow while leaving UI concerns to the UI layer.
All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..services.data_cleaning import (
    DataCleaningService,
    CleaningResult,
    GradeFilterConfig,
    QualityFilterConfig,
    DataCleaningError,
    ConfigurationError,
)
from ..services.zero_filtering import (
    ZeroFilteringService,
    ZeroFilterConfig,
    ZeroFilteringResult
)
from ..services.standards import (
    StandardsService,
    StandardsExtractionResult,
    StandardsValidationResult
)


@dataclass
class IngestionResult:
    """
    Complete result of data ingestion workflow.

    Contains all intermediate and final results from the ingestion pipeline.
    """
    # Detection results
    detected_format: DataFormat
    format_confidence: str = "high"  # high, medium, low

    # Cleaning results
    cleaned_df: Optional[pd.DataFrame] = None
    internal_standards_df: Optional[pd.DataFrame] = None
    cleaning_messages: List[str] = field(default_factory=list)

    # Zero filtering results (optional step)
    zero_filtered: bool = False
    species_before_filter: int = 0
    species_after_filter: int = 0
    removed_species: List[str] = field(default_factory=list)

    # Validation status
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    @property
    def species_removed_count(self) -> int:
        """Number of species removed by zero filtering."""
        return self.species_before_filter - self.species_after_filter

    @property
    def removal_percentage(self) -> float:
        """Percentage of species removed by zero filtering."""
        if self.species_before_filter == 0:
            return 0.0
        return (self.species_removed_count / self.species_before_filter) * 100


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

    # Zero filtering settings
    apply_zero_filter: bool = True
    zero_filter_config: Optional[ZeroFilterConfig] = None  # None = use defaults
    bqc_label: Optional[str] = None

    # Standards settings
    use_external_standards: bool = False
    external_standards_df: Optional[pd.DataFrame] = None


class DataIngestionWorkflow:
    """
    Orchestrates the complete data ingestion pipeline.

    This workflow coordinates format detection, data cleaning, zero filtering,
    and internal standards extraction in the correct order.

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
        3. Apply zero filtering (optional)
        4. Validate internal standards

        Args:
            df: Raw input DataFrame
            config: Ingestion configuration with experiment and settings

        Returns:
            IngestionResult with all intermediate and final results

        Raises:
            ValueError: If input data is invalid or becomes empty after cleaning
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
                "(LipidSearch 5.0, MS-DIAL, Generic, or Metabolomics Workbench)."
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

        except (DataCleaningError, ValueError, KeyError) as e:
            result.is_valid = False
            result.validation_errors.append(str(e))
            return result

        # Step 3: Apply zero filtering (optional)
        if config.apply_zero_filter:
            result = DataIngestionWorkflow._apply_zero_filtering(
                result, config
            )

        # Step 4: Validate internal standards
        if result.internal_standards_df is not None and not result.internal_standards_df.empty:
            validation = StandardsService.validate_standards(
                result.internal_standards_df,
                config.experiment.full_samples_list
            )
            if not validation.is_valid:
                result.validation_warnings.extend(validation.errors)

        # Step 5: Handle external standards (if provided)
        if config.use_external_standards and config.external_standards_df is not None:
            result.internal_standards_df = config.external_standards_df
            result.cleaning_messages.append(
                f"Using external standards file ({len(config.external_standards_df)} standards)"
            )

        return result

    @staticmethod
    def _apply_zero_filtering(
        result: IngestionResult,
        config: IngestionConfig
    ) -> IngestionResult:
        """
        Apply zero filtering to cleaned data.

        Args:
            result: Current ingestion result with cleaned_df
            config: Ingestion configuration

        Returns:
            Updated IngestionResult with zero filtering applied
        """
        if result.cleaned_df is None or result.cleaned_df.empty:
            return result

        # Use default config if not specified
        zero_config = config.zero_filter_config or ZeroFilterConfig()

        # For LipidSearch, use higher detection threshold
        if result.detected_format == DataFormat.LIPIDSEARCH:
            if config.zero_filter_config is None:
                zero_config = ZeroFilterConfig.for_lipidsearch()

        try:
            filter_result = ZeroFilteringService.filter_zeros(
                df=result.cleaned_df,
                experiment=config.experiment,
                config=zero_config,
                bqc_label=config.bqc_label
            )

            result.cleaned_df = filter_result.filtered_df
            result.zero_filtered = True
            result.species_before_filter = filter_result.species_before
            result.species_after_filter = filter_result.species_after
            result.removed_species = filter_result.removed_species

            if filter_result.removed_species:
                result.cleaning_messages.append(
                    f"Zero filter removed {len(filter_result.removed_species)} species "
                    f"({filter_result.species_before} → {filter_result.species_after})"
                )

        except ConfigurationError as e:
            # User-configuration errors are recoverable — show as warning
            result.validation_warnings.append(f"Zero filtering skipped: {e}")
        except (DataCleaningError, ValueError) as e:
            msg = str(e)
            # Only suppress known data-related errors; propagate programmer errors
            if any(kw in msg.lower() for kw in ["threshold", "bqc", "no samples", "empty", "lipidmolec"]):
                result.validation_warnings.append(f"Zero filtering skipped: {msg}")
            else:
                raise

        return result

    @staticmethod
    def detect_format_only(df: pd.DataFrame) -> Tuple[DataFormat, str]:
        """
        Detect format without running the full pipeline.

        Useful for showing format info to user before they configure the experiment.

        Args:
            df: DataFrame to detect format for

        Returns:
            Tuple of (DataFormat, confidence level)
        """
        detected = FormatDetectionService.detect_format(df)

        # Determine confidence based on how many signature columns matched
        if detected == DataFormat.LIPIDSEARCH:
            # LipidSearch has strict requirements, so high confidence
            confidence = "high"
        elif detected == DataFormat.MSDIAL:
            # MS-DIAL detection is also quite reliable
            confidence = "high"
        elif detected == DataFormat.GENERIC:
            # Generic is a fallback, medium confidence
            confidence = "medium"
        else:
            confidence = "low"

        return detected, confidence

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
