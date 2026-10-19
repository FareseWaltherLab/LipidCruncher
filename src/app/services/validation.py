"""
Shared validation utilities for DataFrames with concentration columns.

Used by QualityCheckWorkflow and the adapter to find the concentration
columns a DataFrame has for an experiment.
"""
from typing import List

import pandas as pd

from app.models.experiment import ExperimentConfig


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
