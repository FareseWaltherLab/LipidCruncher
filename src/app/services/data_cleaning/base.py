"""
Base data cleaning functionality with common methods.
Pure logic - no Streamlit dependencies.
"""
import pandas as pd
from typing import List, Tuple, Set


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

    # Internal standard patterns in lipid names
    INTERNAL_STANDARD_LIPID_PATTERNS: List[str] = [
        r'\(d\d+\)',        # Deuterium labels: (d5), (d7), (d9)
        r'[+-]D\d+',        # +D7, -D7 patterns
        r'-d\d+[_\)\s]',    # Alternative deuterium: -d7_, -d9)
        r'\[d\d+\]',        # Bracketed deuterium: [d7]
        r'^Ch-D\d+',        # Cholesterol deuterated standards
        r'ISTD',            # ISTD marker
        r'SPLASH',          # SPLASH lipidomix standards
        r':\(s\)',          # :(s) notation
        r'\(IS\)',          # (IS) marker
        r'_IS$',            # Suffix _IS
    ]

    # Internal standard patterns in ClassKey
    INTERNAL_STANDARD_CLASS_PATTERN: str = r'ISTD|Internal'

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

    # ==================== Row Filtering ====================

    @staticmethod
    def remove_invalid_lipid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid lipid names."""
        if df.empty:
            return df

        try:
            lipid_col = BaseDataCleaner.find_lipid_column(df)
        except KeyError:
            return df  # No lipid column, return unchanged

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
