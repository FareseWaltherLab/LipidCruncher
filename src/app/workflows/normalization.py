"""
Normalization workflow.

Orchestrates the complete normalization pipeline:
Class selection → Method configuration → Normalization → Column restoration

This workflow handles the business logic flow while leaving UI concerns to the UI layer.
All methods are pure logic (no Streamlit dependencies).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig
from ..services.format_detection import DataFormat
from ..services.normalization import NormalizationService


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
    config: Optional[NormalizationConfig] = None
    method_applied: str = "None"
    removed_standards: List[str] = field(default_factory=list)

    # Validation status
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


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
        result = NormalizationWorkflowResult(config=config.normalization)

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
    def adduct_mismatches(
        species_df: pd.DataFrame,
        intsta_df: Optional[pd.DataFrame],
        class_standard_map: Dict[str, str],
    ) -> List[str]:
        """Warn where a class is normalized with a standard measured in
        another adduct than its species.

        A standard converts a species' signal to a concentration only if
        both ionise alike; a [M+HCOO]- species against a [M+H]+ standard
        compares different scales. Uses LipidSearch's ``MainIon`` column;
        without it (other formats) nothing is checked. Standards with no
        recorded adduct (an uploaded table) are named once, unchecked.
        """
        if 'MainIon' not in species_df.columns:
            return []
        standard_ions = {}
        if intsta_df is not None and 'MainIon' in intsta_df.columns:
            standard_ions = dict(zip(
                intsta_df['LipidMolec'],
                intsta_df['MainIon'].astype(str).str.strip(),
            ))

        warnings, unknown = [], []
        for lipid_class, standard in class_standard_map.items():
            ions = species_df.loc[
                species_df['ClassKey'] == lipid_class, 'MainIon'
            ].astype(str).str.strip()
            if ions.empty:
                continue
            standard_ion = standard_ions.get(standard)
            if standard_ion in (None, '', 'nan'):
                unknown.append(standard)
                continue
            other = ions[ions != standard_ion]
            if len(other):
                tally = other.value_counts().sort_index().sort_values(
                    ascending=False, kind="stable",
                )
                warnings.append(
                    f"**{lipid_class}** is normalized with {standard}, measured as "
                    f"{standard_ion}, but {len(other)} of {len(ions)} {lipid_class} "
                    f"species were measured as other adducts ("
                    + ", ".join(f"{ion} {n}" for ion, n in tally.items())
                    + "): their concentrations are on a different scale from "
                    "the standard's."
                )
        if unknown:
            warnings.append(
                "No adduct is recorded for "
                + ", ".join(dict.fromkeys(unknown))
                + " (e.g. an uploaded standards table), so LipidCruncher cannot "
                "check that it was measured in the same adduct as its species."
            )
        return warnings


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
                    # Key by LipidMolec: the service rebuilds the frame with a
                    # fresh index, so the original row index is meaningless.
                    stored[col] = pd.Series(df[col].values, index=df['LipidMolec'].values)

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
                # Align by LipidMolec, not row position: the normalized frame
                # is re-indexed and reordered class-by-class.
                result[col_name] = col_data.reindex(result['LipidMolec'].values).values

        return result
