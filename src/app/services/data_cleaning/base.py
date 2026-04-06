"""
Base data cleaning functionality with common methods.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set

from app.constants import (
    INTERNAL_STANDARD_LIPID_PATTERNS,
    INTERNAL_STANDARD_CLASS_PATTERN,
)


class BaseDataCleaner:
    """
    Base class with common data cleaning methods.

    All methods are static for easy testing and stateless operation.
    """

    # Invalid lipid name patterns
    INVALID_LIPID_PATTERNS: List[str] = [
        r'^\s*$',           # Empty strings or pure whitespace
        r'^Unknown$',       # Unknown from failed standardization
        r'^[#@$%&*]+$',     # Strings of only special characters
        r'^[0-9]+$',        # Strings of only numbers
        r'^nan$',           # Literal 'nan'
        r'^null$',          # Literal 'null'
        r'^none$',          # Literal 'none'
    ]

    # Internal standard patterns (imported from app.constants)
    INTERNAL_STANDARD_LIPID_PATTERNS: List[str] = INTERNAL_STANDARD_LIPID_PATTERNS
    INTERNAL_STANDARD_CLASS_PATTERN: str = INTERNAL_STANDARD_CLASS_PATTERN

    # ==================== Validation Methods ====================

    @staticmethod
    def is_effectively_empty(df: pd.DataFrame) -> bool:
        """
        Check if DataFrame is effectively empty.

        Returns True if:
        1. DataFrame has no rows, OR
        2. LipidMolec column contains only NaN/empty values
        """
        if df.empty:
            return True

        if 'LipidMolec' in df.columns:
            lipid_values = df['LipidMolec'].dropna()
            if len(lipid_values) == 0:
                return True
            if (lipid_values.astype(str).str.strip() == '').all():
                return True

        return False

    @staticmethod
    def find_lipid_column(df: pd.DataFrame) -> str:
        """
        Find the LipidMolec column (case-insensitive).

        Returns column name or raises KeyError if not found.
        """
        if 'LipidMolec' in df.columns:
            return 'LipidMolec'

        col_map = {col.lower(): col for col in df.columns}
        if 'lipidmolec' in col_map:
            return col_map['lipidmolec']

        raise KeyError("Missing required column: LipidMolec")

    # ==================== Column Conversion ====================

    @staticmethod
    def convert_columns_to_numeric(
        df: pd.DataFrame,
        full_samples_list: List[str]
    ) -> pd.DataFrame:
        """
        Convert intensity columns to numeric type.

        Non-numeric values become 0, negative values clipped to 0.
        """
        df = df.copy()
        intensity_cols = [f'intensity[{sample}]' for sample in full_samples_list]
        cols_to_convert = [col for col in intensity_cols if col in df.columns]

        df[cols_to_convert] = df[cols_to_convert].apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0).clip(lower=0)

        return df

    @staticmethod
    def get_intensity_columns(df: pd.DataFrame) -> List[str]:
        """Get list of intensity column names from DataFrame."""
        return [col for col in df.columns if col.startswith('intensity[')]

    # ==================== Column Extraction Helpers ====================

    @staticmethod
    def _add_required_columns(
        col_map: Dict[str, str],
        columns: List[str],
        rename_dict: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Add required columns (LipidMolec, ClassKey) via case-insensitive lookup."""
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
        rename_dict: Dict[str, str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Add intensity columns via case-insensitive lookup."""
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

    # ==================== Row Filtering ====================

    @staticmethod
    def remove_invalid_lipid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid lipid names."""
        if df.empty:
            return df

        try:
            lipid_col = BaseDataCleaner.find_lipid_column(df)
        except KeyError:
            available = ', '.join(list(df.columns)[:10])
            raise ValueError(
                f"Cannot validate lipid rows: 'LipidMolec' column not found. "
                f"Available columns: [{available}]"
            )

        valid_mask = BaseDataCleaner._create_valid_lipid_mask(df, lipid_col)
        return df[valid_mask].copy()

    @staticmethod
    def _create_valid_lipid_mask(df: pd.DataFrame, lipid_col: str) -> pd.Series:
        """Create boolean mask for valid lipid names."""
        combined_pattern = '|'.join(BaseDataCleaner.INVALID_LIPID_PATTERNS)

        # Valid if doesn't match invalid pattern and isn't NaN
        valid_mask = ~df[lipid_col].astype(str).str.match(
            combined_pattern, case=False, na=True
        )
        valid_mask = valid_mask & ~df[lipid_col].isna()

        return valid_mask

    @staticmethod
    def remove_all_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where all intensity values are zero or null."""
        if df.empty:
            return df

        intensity_cols = BaseDataCleaner.get_intensity_columns(df)
        if not intensity_cols:
            return df

        # Keep rows with at least one non-zero value
        has_nonzero = ~(df[intensity_cols] == 0).all(axis=1)
        has_nonnull = ~df[intensity_cols].isnull().all(axis=1)

        return df[has_nonzero & has_nonnull].copy()

    # ==================== Cleaning Step Helpers ====================

    @staticmethod
    def _step_remove_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove invalid lipid rows and return count message."""
        pre_count = len(df)
        df = BaseDataCleaner.remove_invalid_lipid_rows(df)

        if df.empty:
            raise ValueError(
                "No valid lipid species found. All rows had invalid or missing lipid names. "
                "Please check that your 'LipidMolec' column contains valid identifiers."
            )

        removed = pre_count - len(df)
        msg = f"Removed {removed} rows with invalid lipid names" if removed > 0 else None
        return df, msg

    @staticmethod
    def _step_remove_zero_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
        """Remove all-zero rows and return count message."""
        pre_count = len(df)
        df = BaseDataCleaner.remove_all_zero_rows(df)

        removed = pre_count - len(df)
        msg = f"Removed {removed} rows with all-zero intensities" if removed > 0 else None
        return df, msg

    @staticmethod
    def _make_summary_message(initial_count: int, final_count: int) -> str:
        """Build a pipeline summary message."""
        removed = initial_count - final_count
        return (
            f"Summary: {initial_count} entries → {final_count} unique lipid species "
            f"({removed} removed total)"
        )

    # ==================== Internal Standards Extraction ====================

    @staticmethod
    def extract_internal_standards(
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract internal standards from cleaned dataframe.

        Returns:
            Tuple of (df_without_standards, standards_df)
        """
        if df.empty or 'LipidMolec' not in df.columns:
            return df, pd.DataFrame(columns=df.columns)

        is_standard = BaseDataCleaner._identify_standards(df)

        standards_df = df[is_standard].copy().reset_index(drop=True)
        cleaned_df = df[~is_standard].copy().reset_index(drop=True)

        return cleaned_df, standards_df

    @staticmethod
    def _identify_standards(df: pd.DataFrame) -> pd.Series:
        """Create boolean mask identifying internal standards."""
        is_standard_lipid = BaseDataCleaner._check_lipid_standard_patterns(df)
        is_standard_class = BaseDataCleaner._check_class_standard_patterns(df)

        return is_standard_lipid | is_standard_class

    @staticmethod
    def _check_lipid_standard_patterns(df: pd.DataFrame) -> pd.Series:
        """Check LipidMolec column for internal standard patterns."""
        combined_pattern = '|'.join(
            f'(?:{p})' for p in BaseDataCleaner.INTERNAL_STANDARD_LIPID_PATTERNS
        )

        return df['LipidMolec'].str.contains(
            combined_pattern,
            regex=True,
            case=False,
            na=False
        )

    @staticmethod
    def _check_class_standard_patterns(df: pd.DataFrame) -> pd.Series:
        """Check ClassKey column for internal standard markers."""
        if 'ClassKey' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        return df['ClassKey'].str.contains(
            BaseDataCleaner.INTERNAL_STANDARD_CLASS_PATTERN,
            regex=True,
            case=False,
            na=False
        )
