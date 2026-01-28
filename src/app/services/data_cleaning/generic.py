"""
Generic format data cleaning.
Also handles Metabolomics Workbench format (same cleaning logic).
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import List, Optional, Tuple

from ...models.experiment import ExperimentConfig
from .base import BaseDataCleaner


class GenericCleaner(BaseDataCleaner):
    """
    Cleaner for Generic format data.

    Also handles Metabolomics Workbench format as the cleaning
    logic is identical after column standardization.

    Handles:
    - Column extraction
    - Invalid lipid row removal
    - Duplicate removal
    - All-zero row removal
    """

    @staticmethod
    def clean(
        df: pd.DataFrame,
        experiment: ExperimentConfig
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Clean Generic format data.

        Args:
            df: Generic DataFrame (already column-standardized).
            experiment: Experiment configuration.

        Returns:
            Tuple of (cleaned_df, messages_list).

        Raises:
            ValueError: If dataset is empty or becomes empty after cleaning.
        """
        messages = []

        # Validate input
        if GenericCleaner.is_effectively_empty(df):
            raise ValueError(
                "Dataset is empty. Please upload a valid data file with lipid species."
            )

        # Step 1: Extract relevant columns
        df = GenericCleaner._extract_columns(df, experiment.full_samples_list)

        # Step 2: Remove invalid lipid rows
        df, msg = GenericCleaner._step_remove_invalid_rows(df)
        if msg:
            messages.append(msg)

        # Step 3: Convert intensity columns to numeric
        df = GenericCleaner.convert_columns_to_numeric(df, experiment.full_samples_list)

        # Step 4: Remove duplicates
        df, msg = GenericCleaner._step_remove_duplicates(df)
        if msg:
            messages.append(msg)

        # Step 5: Remove all-zero rows
        df, msg = GenericCleaner._step_remove_zero_rows(df)
        if msg:
            messages.append(msg)

        if df.empty:
            raise ValueError(
                "No lipid species with non-zero intensity values found."
            )

        return df.reset_index(drop=True), messages

    # ==================== Cleaning Steps ====================

    @staticmethod
    def _step_remove_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove invalid lipid rows step."""
        pre_count = len(df)
        df = GenericCleaner.remove_invalid_lipid_rows(df)

        if df.empty:
            raise ValueError(
                "No valid lipid species found. All rows had invalid or missing lipid names. "
                "Please check that your 'LipidMolec' column contains valid identifiers."
            )

        removed = pre_count - len(df)
        msg = f"Removed {removed} rows with invalid lipid names" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove duplicates step."""
        pre_count = len(df)
        df = df.drop_duplicates(subset=['LipidMolec'])

        removed = pre_count - len(df)
        msg = f"Removed {removed} duplicate entries" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_remove_zero_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove all-zero rows step."""
        pre_count = len(df)
        df = GenericCleaner.remove_all_zero_rows(df)

        removed = pre_count - len(df)
        msg = f"Removed {removed} rows with all-zero intensities" if removed > 0 else None
        return df, msg

    # ==================== Column Extraction ====================

    @staticmethod
    def _extract_columns(
        df: pd.DataFrame,
        full_samples_list: List[str]
    ) -> pd.DataFrame:
        """
        Extract relevant columns for Generic format.

        Keeps: LipidMolec, ClassKey, and intensity columns.
        """
        col_map = {col.lower(): col for col in df.columns}
        columns_to_keep = []
        rename_dict = {}

        # Required columns
        columns_to_keep, rename_dict = GenericCleaner._add_required_columns(
            col_map, columns_to_keep, rename_dict
        )

        # Intensity columns
        columns_to_keep, rename_dict = GenericCleaner._add_intensity_columns(
            df, full_samples_list, columns_to_keep, rename_dict
        )

        result_df = df[columns_to_keep].copy()
        return result_df.rename(columns=rename_dict)

    @staticmethod
    def _add_required_columns(
        col_map: dict,
        columns: List[str],
        rename_dict: dict
    ) -> Tuple[List[str], dict]:
        """Add required columns (LipidMolec, ClassKey)."""
        required = [('lipidmolec', 'LipidMolec'), ('classkey', 'ClassKey')]

        for col_lower, col_standard in required:
            if col_lower not in col_map:
                raise KeyError(f"Missing required column: {col_standard}")
            columns.append(col_map[col_lower])
            rename_dict[col_map[col_lower]] = col_standard

        return columns, rename_dict

    @staticmethod
    def _add_intensity_columns(
        df: pd.DataFrame,
        full_samples_list: List[str],
        columns: List[str],
        rename_dict: dict
    ) -> Tuple[List[str], dict]:
        """Add intensity columns."""
        for sample in full_samples_list:
            expected_col = f'intensity[{sample}]'.lower()
            found = False

            for actual_col in df.columns:
                if actual_col.lower() == expected_col:
                    columns.append(actual_col)
                    rename_dict[actual_col] = f'intensity[{sample}]'
                    found = True
                    break

            if not found:
                raise KeyError(f"Missing intensity column: intensity[{sample}]")

        return columns, rename_dict
