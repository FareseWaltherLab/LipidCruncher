"""
Shared validation utilities for DataFrames with concentration columns.

Used by QualityCheckService, QualityCheckWorkflow, and UI layer to validate
that DataFrames have the expected concentration columns for an experiment.
"""
from typing import List

import pandas as pd

from app.models.experiment import ExperimentConfig
from app.services.exceptions import EmptyDataError, ValidationError


def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    """Validate that the DataFrame is not None or empty.

    Raises:
        EmptyDataError: If DataFrame is None or empty.
    """
    if df is None or df.empty:
        raise EmptyDataError(
            "Input DataFrame is empty. Cannot perform analysis."
        )


def get_matching_concentration_columns(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
) -> List[str]:
    """Return sample labels whose concentration columns exist in the DataFrame.

    Args:
        df: DataFrame to check.
        experiment: Experiment configuration with sample labels.

    Returns:
        List of sample labels (e.g., ['s1', 's2']) that have matching
        concentration[sN] columns in df.
    """
    return [
        s for s in experiment.full_samples_list
        if f'concentration[{s}]' in df.columns
    ]


def validate_concentration_columns(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
) -> List[str]:
    """Validate that concentration columns exist and return available samples.

    Args:
        df: DataFrame with concentration[sample] columns.
        experiment: Experiment configuration with sample labels.

    Returns:
        List of sample labels with matching concentration columns.

    Raises:
        ValueError: If no concentration columns match the experiment.
    """
    available = get_matching_concentration_columns(df, experiment)
    if not available:
        raise ValidationError(
            "DataFrame has no concentration columns matching the experiment. "
            "Expected columns like 'concentration[s1]', 'concentration[s2]', etc."
        )
    return available
