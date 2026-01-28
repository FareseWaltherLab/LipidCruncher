"""
Normalization service for lipid data.

Pure business logic - no Streamlit dependencies.
Handles internal standard normalization, protein normalization, and combined normalization.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ..models.experiment import ExperimentConfig
from ..models.normalization import NormalizationConfig


@dataclass
class NormalizationResult:
    """
    Result of a normalization operation.

    Attributes:
        normalized_df: DataFrame with normalized values
        removed_standards: List of standard lipid names removed from the dataset
        method_applied: Description of the normalization method applied
    """
    normalized_df: pd.DataFrame
    removed_standards: List[str]
    method_applied: str


class NormalizationService:
    """
    Service for normalizing lipidomic data.

    All methods are static - no instance state required.
    Supports internal standard normalization, protein normalization, and combined normalization.
    """

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: Optional[pd.DataFrame] = None,
    ) -> NormalizationResult:
        """
        Main normalization method that routes to appropriate normalization type.

        Args:
            df: DataFrame with lipid data (must have LipidMolec, ClassKey, intensity[...] columns)
            config: Normalization configuration specifying method and parameters
            experiment: Experiment configuration with sample information
            intsta_df: DataFrame with internal standards (required for internal_standard/both methods)

        Returns:
            NormalizationResult with normalized data and metadata

        Raises:
            ValueError: If required data is missing for the chosen method
            ValueError: If DataFrame is empty or missing required columns
        """
        # Validate input DataFrame
        NormalizationService._validate_input_dataframe(df)

        # Route to appropriate method
        if config.method == 'none':
            return NormalizationService._apply_none(df, config)

        elif config.method == 'internal_standard':
            return NormalizationService._apply_internal_standard(
                df, config, experiment, intsta_df
            )

        elif config.method == 'protein':
            return NormalizationService._apply_protein(df, config, experiment)

        elif config.method == 'both':
            return NormalizationService._apply_both(
                df, config, experiment, intsta_df
            )

        else:
            raise ValueError(f"Unknown normalization method: {config.method}")

    @staticmethod
    def _validate_input_dataframe(df: pd.DataFrame) -> None:
        """
        Validate that the input DataFrame has required columns and structure.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError(
                "Input DataFrame is empty. Please provide a dataset with lipid data."
            )

        required_cols = ['LipidMolec', 'ClassKey']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing required columns: {', '.join(missing_cols)}"
            )

        # Check for intensity columns
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            raise ValueError(
                "Input DataFrame has no intensity columns. "
                "Expected columns like 'intensity[s1]', 'intensity[s2]', etc."
            )

    @staticmethod
    def _apply_none(
        df: pd.DataFrame,
        config: NormalizationConfig
    ) -> NormalizationResult:
        """
        Apply no normalization - filter by selected classes and rename columns.

        Args:
            df: DataFrame with lipid data
            config: Normalization configuration

        Returns:
            NormalizationResult with filtered data
        """
        result_df = df.copy()

        # Filter by selected classes if specified
        if config.selected_classes:
            result_df = result_df[result_df['ClassKey'].isin(config.selected_classes)]

        # Rename intensity to concentration (even for 'none' method, for consistency)
        result_df = NormalizationService._rename_intensity_to_concentration(result_df)

        return NormalizationResult(
            normalized_df=result_df,
            removed_standards=[],
            method_applied="None (raw data with concentration column naming)"
        )

    @staticmethod
    def _apply_internal_standard(
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: Optional[pd.DataFrame]
    ) -> NormalizationResult:
        """
        Apply internal standard normalization.

        Formula: (Intensity_lipid / Intensity_standard) × Conc_standard

        Args:
            df: DataFrame with lipid data
            config: Normalization configuration with internal_standards and intsta_concentrations
            experiment: Experiment configuration with sample list
            intsta_df: DataFrame with internal standards data

        Returns:
            NormalizationResult with normalized data

        Raises:
            ValueError: If intsta_df is missing or empty
        """
        if intsta_df is None or intsta_df.empty:
            raise ValueError(
                "Internal standards DataFrame is required for internal_standard normalization. "
                "Please provide intsta_df with standard intensity data."
            )

        # Validate intsta_df structure
        NormalizationService._validate_intsta_dataframe(intsta_df)

        # Filter to selected classes
        selected_classes = config.selected_classes or list(df['ClassKey'].unique())
        selected_df = df[df['ClassKey'].isin(selected_classes)].copy()

        # Get standards to remove
        standards_to_remove = set(config.internal_standards.values())

        # Remove standards from the dataset
        selected_df = selected_df[~selected_df['LipidMolec'].isin(standards_to_remove)]

        # Process each lipid class
        normalized_dfs = []
        full_samples_list = experiment.full_samples_list

        for lipid_class in selected_classes:
            standard_name = config.get_standard_for_class(lipid_class)
            if standard_name is None:
                raise ValueError(
                    f"No internal standard mapped for lipid class '{lipid_class}'. "
                    f"Please provide a mapping in internal_standards."
                )

            concentration = config.get_standard_concentration(standard_name)
            if concentration is None:
                raise ValueError(
                    f"No concentration provided for internal standard '{standard_name}'. "
                    f"Please provide concentration in intsta_concentrations."
                )

            # Get AUC values for this standard
            intsta_auc = NormalizationService._compute_intsta_auc(
                intsta_df,
                standard_name,
                full_samples_list
            )

            # Normalize this class
            class_df = NormalizationService._compute_normalized_auc(
                selected_df,
                full_samples_list,
                lipid_class,
                intsta_auc,
                concentration
            )

            if not class_df.empty:
                normalized_dfs.append(class_df)

        # Combine all classes
        if normalized_dfs:
            result_df = pd.concat(normalized_dfs, axis=0, ignore_index=True)
        else:
            # Return empty DataFrame with correct structure
            intensity_cols = [f"concentration[{s}]" for s in full_samples_list]
            result_df = pd.DataFrame(columns=['LipidMolec', 'ClassKey'] + intensity_cols)

        # Clean up the data
        result_df = result_df.fillna(0)
        result_df = result_df.replace([np.inf, -np.inf], 0)

        # Rename columns
        result_df = NormalizationService._rename_intensity_to_concentration(result_df)

        return NormalizationResult(
            normalized_df=result_df,
            removed_standards=list(standards_to_remove),
            method_applied="Internal standards normalization"
        )

    @staticmethod
    def _apply_protein(
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        preserve_prefix: bool = False
    ) -> NormalizationResult:
        """
        Apply protein concentration normalization (BCA assay).

        Formula: Intensity_lipid / Protein_conc

        Args:
            df: DataFrame with lipid data
            config: Normalization configuration with protein_concentrations
            experiment: Experiment configuration with sample list
            preserve_prefix: If True, keep current column prefix (for chaining)

        Returns:
            NormalizationResult with normalized data

        Raises:
            ValueError: If protein_concentrations is missing
        """
        if not config.protein_concentrations:
            raise ValueError(
                "Protein concentrations are required for protein normalization. "
                "Please provide protein_concentrations mapping sample -> concentration."
            )

        # Filter to selected classes if specified
        result_df = df.copy()
        if config.selected_classes:
            result_df = result_df[result_df['ClassKey'].isin(config.selected_classes)]

        # Find columns to normalize - can be either intensity[] or concentration[]
        # (concentration[] appears when chaining with internal standards normalization)
        intensity_cols = [col for col in result_df.columns if col.startswith('intensity[')]
        concentration_cols = [col for col in result_df.columns if col.startswith('concentration[')]

        # Use whichever column type is present
        cols_to_normalize = intensity_cols if intensity_cols else concentration_cols

        if not cols_to_normalize:
            raise ValueError(
                "No intensity or concentration columns found in DataFrame. "
                "Cannot apply protein normalization."
            )

        # Apply normalization
        skipped_samples = []
        for col in cols_to_normalize:
            sample_name = col[col.find('[') + 1:col.find(']')]

            protein_conc = config.get_protein_concentration(sample_name)
            if protein_conc is None:
                skipped_samples.append(sample_name)
                continue

            if protein_conc <= 0:
                # Skip this sample - don't normalize with zero/negative
                skipped_samples.append(sample_name)
                continue

            result_df[col] = result_df[col] / protein_conc

        # Rename columns unless preserving prefix
        if not preserve_prefix and intensity_cols:
            result_df = NormalizationService._rename_intensity_to_concentration(result_df)

        method_desc = "Protein concentration normalization"
        if skipped_samples:
            method_desc += f" (skipped samples: {', '.join(skipped_samples)})"

        return NormalizationResult(
            normalized_df=result_df,
            removed_standards=[],
            method_applied=method_desc
        )

    @staticmethod
    def _apply_both(
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: Optional[pd.DataFrame]
    ) -> NormalizationResult:
        """
        Apply combined normalization: internal standards first, then protein.

        Formula: (Intensity_lipid / Intensity_standard) × (Conc_standard / Protein_conc)

        Args:
            df: DataFrame with lipid data
            config: Normalization configuration
            experiment: Experiment configuration
            intsta_df: DataFrame with internal standards data

        Returns:
            NormalizationResult with normalized data
        """
        # First apply internal standards
        is_result = NormalizationService._apply_internal_standard(
            df, config, experiment, intsta_df
        )

        # Then apply protein normalization (preserve prefix since we're chaining)
        # Need to work with the already-normalized DataFrame
        protein_result = NormalizationService._apply_protein(
            is_result.normalized_df,
            config,
            experiment,
            preserve_prefix=True  # Keep concentration[] prefix
        )

        return NormalizationResult(
            normalized_df=protein_result.normalized_df,
            removed_standards=is_result.removed_standards,
            method_applied="Combined normalization (internal standards + protein)"
        )

    @staticmethod
    def _validate_intsta_dataframe(intsta_df: pd.DataFrame) -> None:
        """
        Validate that the internal standards DataFrame has required structure.

        Args:
            intsta_df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if 'LipidMolec' not in intsta_df.columns:
            raise ValueError(
                "Internal standards DataFrame must have 'LipidMolec' column "
                "containing standard lipid names."
            )

        intensity_cols = [col for col in intsta_df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            raise ValueError(
                "Internal standards DataFrame has no intensity columns. "
                "Expected columns like 'intensity[s1]', 'intensity[s2]', etc."
            )

    @staticmethod
    def _compute_intsta_auc(
        intsta_df: pd.DataFrame,
        intsta_species: str,
        full_samples_list: List[str]
    ) -> np.ndarray:
        """
        Compute AUC values for the selected internal standard.

        Args:
            intsta_df: DataFrame with internal standards
            intsta_species: Name of the standard to use
            full_samples_list: List of sample IDs

        Returns:
            Array of AUC values for each sample

        Raises:
            ValueError: If standard not found in intsta_df
        """
        filtered_intsta = intsta_df[intsta_df['LipidMolec'] == intsta_species]

        if filtered_intsta.empty:
            raise ValueError(
                f"Internal standard '{intsta_species}' not found in standards DataFrame. "
                f"Available standards: {list(intsta_df['LipidMolec'].unique())}"
            )

        cols = [f"intensity[{sample}]" for sample in full_samples_list]

        # Check for missing columns
        missing_cols = [col for col in cols if col not in intsta_df.columns]
        if missing_cols:
            raise ValueError(
                f"Internal standards DataFrame is missing intensity columns: {', '.join(missing_cols)}"
            )

        return filtered_intsta[cols].values.reshape(len(full_samples_list),)

    @staticmethod
    def _compute_normalized_auc(
        selected_df: pd.DataFrame,
        full_samples_list: List[str],
        lipid_class: str,
        intsta_auc: np.ndarray,
        concentration: float
    ) -> pd.DataFrame:
        """
        Normalize AUC values for a specific lipid class using internal standard.

        Args:
            selected_df: DataFrame with lipid data (already filtered to selected classes)
            full_samples_list: List of sample IDs
            lipid_class: Lipid class to normalize
            intsta_auc: Array of standard AUC values
            concentration: Standard concentration

        Returns:
            DataFrame with normalized values for this class
        """
        class_df = selected_df[selected_df['ClassKey'] == lipid_class].copy()

        if class_df.empty:
            return class_df

        intensity_cols = [f"intensity[{sample}]" for sample in full_samples_list]

        # Check for missing columns
        available_cols = [col for col in intensity_cols if col in class_df.columns]
        if not available_cols:
            return pd.DataFrame()

        intensity_df = class_df[available_cols]

        # Normalize: (lipid_intensity / standard_intensity) * standard_concentration
        # Handle division by zero - replace zeros in intsta_auc with nan temporarily
        safe_intsta_auc = np.where(intsta_auc == 0, np.nan, intsta_auc)
        normalized_intensity_df = (intensity_df.divide(safe_intsta_auc, axis='columns') * concentration)

        # Keep essential columns
        essential_cols = ['LipidMolec', 'ClassKey']
        result_df = pd.concat([
            class_df[essential_cols].reset_index(drop=True),
            normalized_intensity_df.reset_index(drop=True)
        ], axis=1)

        return result_df

    @staticmethod
    def _rename_intensity_to_concentration(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename intensity columns to concentration columns.

        Args:
            df: DataFrame with intensity columns

        Returns:
            DataFrame with renamed columns
        """
        renamed_cols = {
            col: col.replace('intensity[', 'concentration[')
            for col in df.columns if 'intensity[' in col
        }
        return df.rename(columns=renamed_cols)

    @staticmethod
    def get_available_standards(intsta_df: pd.DataFrame) -> List[str]:
        """
        Get list of available internal standards from the standards DataFrame.

        Args:
            intsta_df: DataFrame with internal standards

        Returns:
            List of standard lipid molecule names
        """
        if intsta_df is None or intsta_df.empty:
            return []

        if 'LipidMolec' not in intsta_df.columns:
            return []

        return list(intsta_df['LipidMolec'].unique())

    @staticmethod
    def get_standards_by_class(intsta_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group available internal standards by their lipid class.

        Args:
            intsta_df: DataFrame with internal standards (must have ClassKey column)

        Returns:
            Dictionary mapping ClassKey -> list of standard names
        """
        if intsta_df is None or intsta_df.empty:
            return {}

        if 'ClassKey' not in intsta_df.columns or 'LipidMolec' not in intsta_df.columns:
            return {}

        return intsta_df.groupby('ClassKey')['LipidMolec'].apply(list).to_dict()

    @staticmethod
    def validate_normalization_setup(
        df: pd.DataFrame,
        config: NormalizationConfig,
        experiment: ExperimentConfig,
        intsta_df: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Validate that all requirements are met for the specified normalization method.

        Args:
            df: DataFrame with lipid data
            config: Normalization configuration
            experiment: Experiment configuration
            intsta_df: Internal standards DataFrame (optional)

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        # Basic DataFrame validation
        if df is None or df.empty:
            errors.append("Input DataFrame is empty")
            return errors

        if 'ClassKey' not in df.columns:
            errors.append("Input DataFrame missing 'ClassKey' column")

        if 'LipidMolec' not in df.columns:
            errors.append("Input DataFrame missing 'LipidMolec' column")

        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        if not intensity_cols:
            errors.append("Input DataFrame has no intensity columns")

        # Method-specific validation
        if config.method in ('internal_standard', 'both'):
            if intsta_df is None or intsta_df.empty:
                errors.append("Internal standards DataFrame is required but not provided")
            elif 'LipidMolec' not in intsta_df.columns:
                errors.append("Internal standards DataFrame missing 'LipidMolec' column")
            else:
                # Check that all mapped standards exist
                if config.internal_standards:
                    available = set(intsta_df['LipidMolec'].unique())
                    for lipid_class, standard in config.internal_standards.items():
                        if standard not in available:
                            errors.append(
                                f"Standard '{standard}' for class '{lipid_class}' "
                                f"not found in standards DataFrame"
                            )

            # Check sample columns match
            if intsta_df is not None and not intsta_df.empty:
                expected_cols = [f"intensity[{s}]" for s in experiment.full_samples_list]
                missing = [c for c in expected_cols if c not in intsta_df.columns]
                if missing:
                    errors.append(
                        f"Internal standards DataFrame missing sample columns: {', '.join(missing)}"
                    )

        if config.method in ('protein', 'both'):
            if not config.protein_concentrations:
                errors.append("Protein concentrations not provided")
            else:
                # Check that all samples have protein concentrations
                missing_samples = [
                    s for s in experiment.full_samples_list
                    if s not in config.protein_concentrations
                ]
                if missing_samples:
                    errors.append(
                        f"Missing protein concentrations for samples: {', '.join(missing_samples)}"
                    )

        return errors
