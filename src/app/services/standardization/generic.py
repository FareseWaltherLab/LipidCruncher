"""
Generic format: a lipid name column, an optional ClassKey column, then one
column per sample.
"""

from typing import List, Tuple

import pandas as pd

from app.services import lipid_names
from app.services.standardization.result import StandardizationResult


def is_classkey_column(df: pd.DataFrame, col_idx: int) -> bool:
    """Check if a column looks like a ClassKey column."""
    if col_idx >= len(df.columns):
        return False

    col_name = df.columns[col_idx]
    if str(col_name).lower().strip() == 'classkey':
        return True

    col_values = df.iloc[:, col_idx]
    try:
        numeric_count = pd.to_numeric(col_values, errors='coerce').notna().sum()
        if numeric_count / len(col_values) > 0.5:
            return False

        str_values = col_values.astype(str)
        avg_len = str_values.str.len().mean()
        total_len = str_values.str.len().sum()
        if total_len == 0:
            return False
        alpha_ratio = (
            str_values.str.replace(r'[^A-Za-z]', '', regex=True).str.len().sum()
            / total_len
        )
        return avg_len < 15 and alpha_ratio > 0.7
    except (ValueError, TypeError, ZeroDivisionError):
        return False


def process(df: pd.DataFrame) -> StandardizationResult:
    """Validate and standardize Generic format data."""
    if not isinstance(df, pd.DataFrame):
        return StandardizationResult(False, "Expected a DataFrame for Generic format")

    if len(df.columns) < 2:
        return StandardizationResult(
            False, "Dataset must have at least 2 columns: lipid names and one intensity column"
        )

    # First column should contain letters (lipid names)
    if not df.iloc[:, 0].astype(str).str.contains('[A-Za-z]').any():
        return StandardizationResult(
            False, "First column doesn't appear to contain lipid names (no letters found)."
        )

    standardized_cols, has_classkey, n_intensity = (
        build_column_map(df)
    )

    result_df = df.copy()
    mapping_df = pd.DataFrame({
        'standardized_name': standardized_cols,
        'original_name': df.columns.tolist(),
    })
    result_df.columns = standardized_cols

    result_df['LipidMolec'] = result_df['LipidMolec'].apply(
        lipid_names.standardize_lipid_name
    )
    if not has_classkey:
        result_df['ClassKey'] = result_df['LipidMolec'].apply(
            lipid_names.infer_class_key
        )

    return StandardizationResult(
        success=True,
        message="Data successfully standardized",
        standardized_df=result_df,
        column_mapping=mapping_df,
        n_intensity_cols=n_intensity,
    )


def build_column_map(df: pd.DataFrame) -> Tuple[List[str], bool, int]:
    """Build standardized column names for Generic format.

    Returns:
        Tuple of (standardized_cols, has_classkey, n_intensity_cols).
    """
    original_cols = df.columns.tolist()
    standardized_cols = ['LipidMolec']

    has_classkey = (
        len(original_cols) > 2
        and is_classkey_column(df, 1)
    )

    intensity_start = 2 if has_classkey else 1
    if has_classkey:
        standardized_cols.append('ClassKey')

    for i in range(intensity_start, len(original_cols)):
        standardized_cols.append(f'intensity[s{i - intensity_start + 1}]')

    return standardized_cols, has_classkey, len(original_cols) - intensity_start
