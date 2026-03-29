"""
Normalization workflow.

Orchestrates the complete normalization pipeline:
Class selection → Method configuration → Normalization → Column restoration

This workflow handles the business logic flow while leaving UI concerns to the UI layer.
All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..services.format_detection import DataFormat
from ..services.normalization import NormalizationService, NormalizationResult as ServiceResult


@dataclass
class NormalizationWorkflowConfig:
    """
    Configuration for the normalization workflow.

    Collects all user choices that affect the normalization process.
    """
    # Required settings
    experiment: ExperimentConfig
    normalization: NormalizationConfig

    # Data format (used for column restoration)
    data_format: DataFormat = DataFormat.GENERIC

    # Essential columns to restore after normalization (LipidSearch format)
    # These are stored before normalization and restored after
    essential_columns: Optional[Dict[str, pd.Series]] = None


@dataclass
class NormalizationWorkflowResult:
    """
    Complete result of normalization workflow.

    Contains all intermediate and final results from the normalization pipeline.
    """
    # Primary result
    normalized_df: Optional[pd.DataFrame] = None
    success: bool = True

    # Method information
    method_applied: str = "None"
    removed_standards: List[str] = field(default_factory=list)

    # Class information
    classes_in_input: List[str] = field(default_factory=list)
    classes_normalized: List[str] = field(default_factory=list)

    # Validation status
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Statistics
    lipids_before: int = 0
    lipids_after: int = 0
    samples_processed: int = 0

    @property
    def lipids_removed_count(self) -> int:
        """Number of lipids removed (usually standards)."""
        return self.lipids_before - self.lipids_after

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.validation_warnings) > 0


class NormalizationWorkflow:
    """
    Orchestrates the complete normalization pipeline.

    This workflow coordinates class selection, method configuration,
    and normalization service calls in the correct order.

    All methods are static - no instance state is stored.
    """

    @staticmethod
    def run(
        df: pd.DataFrame,
        config: NormalizationWorkflowConfig,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> NormalizationWorkflowResult:
        """
        Execute the complete normalization workflow.

        Pipeline:
        1. Validate inputs
        2. Store essential columns (LipidSearch)
        3. Filter by selected classes
        4. Apply normalization
        5. Restore essential columns

        Args:
            df: Cleaned input DataFrame with intensity columns
            config: Workflow configuration with experiment, normalization settings
            intsta_df: Internal standards DataFrame (required for IS/both methods)

        Returns:
            NormalizationWorkflowResult with normalized data and metadata

        Raises:
            ValueError: If input data is invalid
        """
        result = NormalizationWorkflowResult()

        # Track input stats
        result.lipids_before = len(df) if df is not None else 0
        result.classes_in_input = list(df['ClassKey'].unique()) if df is not None and 'ClassKey' in df.columns else []

        # Step 1: Validate inputs
        validation_errors = NormalizationWorkflow.validate_config(
            df, config, intsta_df
        )
        if validation_errors:
            result.success = False
            result.validation_errors = validation_errors
            return result

        # Step 2: Store essential columns for LipidSearch format
        stored_columns = NormalizationWorkflow._store_essential_columns(
            df, config.data_format
        )

        # Step 3: Apply normalization using the service
        try:
            service_result = NormalizationService.normalize(
                df=df,
                config=config.normalization,
                experiment=config.experiment,
                intsta_df=intsta_df
            )

            result.normalized_df = service_result.normalized_df
            result.method_applied = service_result.method_applied
            result.removed_standards = service_result.removed_standards

        except ValueError as e:
            result.success = False
            result.validation_errors.append(str(e))
            return result

        # Step 4: Restore essential columns
        if result.normalized_df is not None:
            result.normalized_df = NormalizationWorkflow._restore_essential_columns(
                result.normalized_df, stored_columns
            )

        # Step 5: Collect final statistics
        if result.normalized_df is not None:
            result.lipids_after = len(result.normalized_df)
            result.classes_normalized = list(result.normalized_df['ClassKey'].unique())

            # Count samples processed
            concentration_cols = [
                col for col in result.normalized_df.columns
                if col.startswith('concentration[')
            ]
            result.samples_processed = len(concentration_cols)

        return result

    @staticmethod
    def validate_config(
        df: pd.DataFrame,
        config: NormalizationWorkflowConfig,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Validate that all requirements are met for the normalization workflow.

        Args:
            df: Input DataFrame
            config: Workflow configuration
            intsta_df: Internal standards DataFrame

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        # Basic DataFrame validation
        if df is None or df.empty:
            errors.append("Input DataFrame is empty. Please provide data to normalize.")
            return errors

        available = ', '.join(list(df.columns)[:15])

        if 'LipidMolec' not in df.columns:
            errors.append(f"Input DataFrame missing 'LipidMolec' column. Available columns: [{available}]")

        if 'ClassKey' not in df.columns:
            errors.append(f"Input DataFrame missing 'ClassKey' column. Available columns: [{available}]")

        # Check for intensity columns
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            errors.append(
                "Input DataFrame has no intensity columns. "
                f"Expected columns like 'intensity[s1]', 'intensity[s2]', etc. Available columns: [{available}]"
            )

        # Validate experiment configuration
        if config.experiment is None:
            errors.append("Experiment configuration is required.")
            return errors

        # Validate selected classes exist in data
        if config.normalization.selected_classes:
            available_classes = set(df['ClassKey'].unique()) if 'ClassKey' in df.columns else set()
            invalid_classes = set(config.normalization.selected_classes) - available_classes
            if invalid_classes:
                errors.append(
                    f"Selected classes not found in data: {', '.join(sorted(invalid_classes))}. "
                    f"Available classes: {', '.join(sorted(available_classes))}"
                )

        # Delegate method-specific validation to NormalizationService
        service_errors = NormalizationService.validate_normalization_setup(
            df, config.normalization, config.experiment, intsta_df
        )
        errors.extend(service_errors)

        return errors

    @staticmethod
    def get_available_classes(df: pd.DataFrame) -> List[str]:
        """
        Get list of unique lipid classes from the DataFrame.

        Args:
            df: DataFrame with ClassKey column

        Returns:
            Sorted list of unique class names
        """
        if df is None or df.empty:
            return []

        if 'ClassKey' not in df.columns:
            return []

        return sorted(df['ClassKey'].unique().tolist())

    @staticmethod
    def get_available_standards(intsta_df: Optional[pd.DataFrame]) -> List[str]:
        """
        Get list of available internal standards.

        Args:
            intsta_df: Internal standards DataFrame

        Returns:
            List of standard lipid molecule names
        """
        return NormalizationService.get_available_standards(intsta_df)

    @staticmethod
    def get_standards_by_class(intsta_df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Group available internal standards by their lipid class.

        Args:
            intsta_df: Internal standards DataFrame with ClassKey column

        Returns:
            Dictionary mapping ClassKey -> list of standard names
        """
        return NormalizationService.get_standards_by_class(intsta_df)

    @staticmethod
    def suggest_standard_mappings(
        selected_classes: List[str],
        intsta_df: Optional[pd.DataFrame]
    ) -> Dict[str, Optional[str]]:
        """
        Suggest default standard-to-class mappings based on available standards.

        Priority:
        1. Class-specific standard (same ClassKey)
        2. First available standard

        Args:
            selected_classes: List of classes needing standards
            intsta_df: Internal standards DataFrame

        Returns:
            Dictionary mapping class -> suggested standard (or None if no standards)
        """
        if intsta_df is None or intsta_df.empty:
            return {cls: None for cls in selected_classes}

        standards_by_class = NormalizationWorkflow.get_standards_by_class(intsta_df)
        all_standards = NormalizationWorkflow.get_available_standards(intsta_df)

        mappings = {}
        for lipid_class in selected_classes:
            # Try class-specific standard first
            class_standards = standards_by_class.get(lipid_class, [])
            if class_standards:
                mappings[lipid_class] = class_standards[0]
            elif all_standards:
                # Fall back to first available standard
                mappings[lipid_class] = all_standards[0]
            else:
                mappings[lipid_class] = None

        return mappings

    @staticmethod
    def validate_standard_mappings(
        internal_standards: Dict[str, str],
        intsta_df: Optional[pd.DataFrame],
        selected_classes: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all standard mappings are valid.

        Args:
            internal_standards: Mapping of class -> standard name
            intsta_df: Internal standards DataFrame
            selected_classes: Classes that need mappings

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if intsta_df is None or intsta_df.empty:
            errors.append(
                "Internal standards DataFrame is required for validation. "
                "Ensure standards were detected or uploaded during data ingestion."
            )
            return False, errors

        available_standards = set(NormalizationWorkflow.get_available_standards(intsta_df))

        # Check all classes have mappings
        missing_classes = set(selected_classes) - set(internal_standards.keys())
        if missing_classes:
            errors.append(
                f"Missing standard mappings for classes: {', '.join(sorted(missing_classes))}. "
                f"Mapped classes: {', '.join(sorted(internal_standards.keys())) or 'none'}"
            )

        # Check all mapped standards exist
        for lipid_class, standard in internal_standards.items():
            if standard not in available_standards:
                errors.append(
                    f"Standard '{standard}' for class '{lipid_class}' "
                    f"not found in standards DataFrame. "
                    f"Available standards: {', '.join(sorted(available_standards))}"
                )

        return len(errors) == 0, errors

    @staticmethod
    def create_normalization_config(
        method: str,
        selected_classes: List[str],
        internal_standards: Optional[Dict[str, str]] = None,
        intsta_concentrations: Optional[Dict[str, float]] = None,
        protein_concentrations: Optional[Dict[str, float]] = None
    ) -> NormalizationConfig:
        """
        Helper to create a NormalizationConfig with proper validation.

        Args:
            method: Normalization method ('none', 'internal_standard', 'protein', 'both')
            selected_classes: List of classes to normalize
            internal_standards: Class -> standard name mapping
            intsta_concentrations: Standard -> concentration mapping
            protein_concentrations: Sample -> protein concentration mapping

        Returns:
            Validated NormalizationConfig

        Raises:
            ValueError: If configuration is invalid
        """
        return NormalizationConfig(
            method=method,
            selected_classes=selected_classes,
            internal_standards=internal_standards,
            intsta_concentrations=intsta_concentrations,
            protein_concentrations=protein_concentrations
        )

    @staticmethod
    def _store_essential_columns(
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> Dict[str, pd.Series]:
        """
        Store essential columns that need to be restored after normalization.

        For LipidSearch format, CalcMass and BaseRt are needed for downstream
        analysis but may be lost during normalization.

        Args:
            df: Input DataFrame
            data_format: Data format type

        Returns:
            Dictionary of column name -> column data
        """
        stored = {}

        if data_format == DataFormat.LIPIDSEARCH:
            essential = ['CalcMass', 'BaseRt']
            for col in essential:
                if col in df.columns:
                    stored[col] = df[col].copy()

        return stored

    @staticmethod
    def _restore_essential_columns(
        df: pd.DataFrame,
        stored_columns: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Restore essential columns that were stored before normalization.

        Args:
            df: Normalized DataFrame
            stored_columns: Dictionary of columns to restore

        Returns:
            DataFrame with restored columns
        """
        if not stored_columns:
            return df

        result = df.copy()

        for col_name, col_data in stored_columns.items():
            if col_name not in result.columns:
                # Need to align by index if DataFrame was filtered
                if len(col_data) == len(result):
                    result[col_name] = col_data.values
                else:
                    # Try to align by the original index
                    result[col_name] = col_data.reindex(result.index).values

        return result

    @staticmethod
    def get_column_samples(df: pd.DataFrame, prefix: str = 'intensity') -> List[str]:
        """
        Extract sample names from columns with a given prefix.

        Looks for columns matching ``prefix[sample_name]`` and returns the
        sample names.

        Args:
            df: DataFrame with prefixed columns (e.g. intensity[s1], concentration[s1])
            prefix: Column prefix to match (default: 'intensity')

        Returns:
            List of sample names (e.g., ['s1', 's2', 's3'])
        """
        tag = f'{prefix}['
        return [
            col[len(tag):-1]
            for col in df.columns
            if col.startswith(tag) and col.endswith(']')
        ]

    @staticmethod
    def get_intensity_column_samples(df: pd.DataFrame) -> List[str]:
        """Extract sample names from intensity[...] columns."""
        return NormalizationWorkflow.get_column_samples(df, 'intensity')

    @staticmethod
    def get_concentration_column_samples(df: pd.DataFrame) -> List[str]:
        """Extract sample names from concentration[...] columns."""
        return NormalizationWorkflow.get_column_samples(df, 'concentration')

    @staticmethod
    def preview_normalization(
        df: pd.DataFrame,
        config: NormalizationWorkflowConfig,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Preview what normalization will do without actually applying it.

        Useful for showing users what will happen before they commit.

        Args:
            df: Input DataFrame
            config: Workflow configuration
            intsta_df: Internal standards DataFrame

        Returns:
            Dictionary with preview information:
            - method: Normalization method
            - classes_to_process: Classes that will be normalized
            - standards_to_remove: Standards that will be removed
            - samples_to_normalize: Sample count
            - validation_errors: Any errors that would prevent normalization
        """
        preview = {
            'method': config.normalization.method,
            'classes_to_process': [],
            'standards_to_remove': [],
            'samples_to_normalize': 0,
            'validation_errors': [],
            'can_proceed': True
        }

        # Validate configuration
        errors = NormalizationWorkflow.validate_config(df, config, intsta_df)
        if errors:
            preview['validation_errors'] = errors
            preview['can_proceed'] = False
            return preview

        # Determine classes to process
        if config.normalization.selected_classes:
            preview['classes_to_process'] = config.normalization.selected_classes
        else:
            preview['classes_to_process'] = list(df['ClassKey'].unique()) if 'ClassKey' in df.columns else []

        # Determine standards to remove
        if config.normalization.internal_standards:
            preview['standards_to_remove'] = list(set(config.normalization.internal_standards.values()))

        # Count samples
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        preview['samples_to_normalize'] = len(intensity_cols)

        return preview
