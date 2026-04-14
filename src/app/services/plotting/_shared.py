"""
Shared plotting utilities.

Common helpers used across multiple plotting services: significance markers,
color palette generation, and input validation.

Pure logic — no Streamlit dependencies.
"""

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from app.constants import CONDITION_COLORS


# Extended qualitative palette for lipid classes (tab20-based, 20 distinct colors).
# Ordered so adjacent colors are perceptually far apart, avoiding the confusion
# that arises when a small default palette cycles.
CLASS_COLORS = (
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
    '#aec7e8',  # light blue
    '#ffbb78',  # light orange
    '#98df8a',  # light green
    '#ff9896',  # light red
    '#c5b0d5',  # light purple
    '#c49c94',  # light brown
    '#f7b6d2',  # light pink
    '#c7c7c7',  # light gray
    '#dbdb8d',  # light olive
    '#9edae5',  # light cyan
)


# ── Significance markers ──────────────────────────────────────────────


def p_value_to_marker(p_value: float) -> str:
    """Convert a p-value to a significance marker string.

    Args:
        p_value: The p-value to convert.

    Returns:
        '***' for p < 0.001, '**' for p < 0.01, '*' for p < 0.05,
        '' otherwise.
    """
    if np.isnan(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return ''


def get_effective_p_value(result) -> float:
    """Extract the best available p-value from a StatisticalTestResult.

    Prefers adjusted_p_value when available and non-NaN,
    falls back to raw p_value.

    Args:
        result: A StatisticalTestResult instance.

    Returns:
        The effective p-value as a float.
    """
    if (
        result.adjusted_p_value is not None
        and not np.isnan(result.adjusted_p_value)
    ):
        return result.adjusted_p_value
    return result.p_value


# ── Color utilities ───────────────────────────────────────────────────


def generate_condition_color_mapping(conditions: List[str]) -> Dict[str, str]:
    """Map each condition to a consistent color from CONDITION_COLORS.

    Cycles through CONDITION_COLORS if there are more conditions than colors.

    Args:
        conditions: List of condition labels.

    Returns:
        Dict mapping condition name to hex color string.
    """
    return {
        cond: CONDITION_COLORS[i % len(CONDITION_COLORS)]
        for i, cond in enumerate(conditions)
    }


def generate_class_color_mapping(classes: List[str]) -> Dict[str, str]:
    """Map each lipid class to a consistent color from Plotly qualitative palette.

    Cycles through the palette if there are more classes than colors.

    Args:
        classes: List of lipid class names.

    Returns:
        Dict mapping class name to hex color string.
    """
    return {
        cls: CLASS_COLORS[i % len(CLASS_COLORS)]
        for i, cls in enumerate(classes)
    }


# ── Input validation ──────────────────────────────────────────────────


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame",
) -> None:
    """Validate that a DataFrame is non-empty and has required columns.

    Args:
        df: The DataFrame to validate.
        required_columns: Column names that must be present.
        name: Human-readable name for error messages.

    Raises:
        ValueError: If df is None, empty, or missing required columns.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a non-null DataFrame")
    if df.empty:
        raise ValueError(f"{name} is empty")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )