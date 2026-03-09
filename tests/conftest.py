"""
Shared test fixtures and factory functions.

Provides reusable experiment and DataFrame builders to reduce
duplication across unit and integration test files.
"""

import numpy as np
import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig


# =============================================================================
# Factory Functions
# =============================================================================


def make_experiment(
    n_conditions: int = 2,
    samples_per_condition: int = 3,
    conditions_list: list = None,
    number_of_samples_list: list = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig with sensible defaults.

    Args:
        n_conditions: Number of experimental conditions.
        samples_per_condition: Samples per condition (used when
            number_of_samples_list is not provided).
        conditions_list: Explicit condition names. Defaults to
            ['Control', 'Treatment', 'Vehicle', ...].
        number_of_samples_list: Explicit per-condition sample counts.
            Defaults to [samples_per_condition] * n_conditions.
    """
    default_names = ['Control', 'Treatment', 'Vehicle', 'Reference', 'Baseline']
    if conditions_list is None:
        conditions_list = default_names[:n_conditions]
    if number_of_samples_list is None:
        number_of_samples_list = [samples_per_condition] * n_conditions
    return ExperimentConfig(
        n_conditions=n_conditions,
        conditions_list=conditions_list,
        number_of_samples_list=number_of_samples_list,
    )


def make_dataframe(
    n_lipids: int = 5,
    n_samples: int = 6,
    classes: list = None,
    lipids: list = None,
    with_classkey: bool = True,
    prefix: str = 'intensity',
    value_fn=None,
) -> pd.DataFrame:
    """Build a minimal standardized DataFrame for testing.

    Args:
        n_lipids: Number of lipid rows.
        n_samples: Number of sample columns.
        classes: List of class names (length must equal n_lipids).
            Defaults to all 'PC'.
        lipids: Explicit lipid names. Defaults to 'PC(0:0)'..'PC(n:0)'.
        with_classkey: Whether to include a ClassKey column.
        prefix: Column prefix — 'intensity' or 'concentration'.
        value_fn: Callable(n_lipids) returning a 1-D array of values
            for each sample column. Defaults to random [0, 1000).
    """
    if lipids is None:
        lipids = [f'PC({i}:0)' for i in range(n_lipids)]
    data: dict = {'LipidMolec': lipids}
    if with_classkey:
        data['ClassKey'] = classes if classes else ['PC'] * n_lipids
    if value_fn is None:
        value_fn = lambda n: np.random.rand(n) * 1000
    for i in range(1, n_samples + 1):
        data[f'{prefix}[s{i}]'] = value_fn(n_lipids)
    return pd.DataFrame(data)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def simple_experiment_2x3():
    """2 conditions x 3 samples each = 6 total."""
    return make_experiment(2, 3)


@pytest.fixture
def simple_experiment_2x2():
    """2 conditions x 2 samples each = 4 total."""
    return make_experiment(2, 2)


@pytest.fixture
def three_condition_experiment():
    """3 conditions x 2 samples each = 6 total."""
    return make_experiment(3, 2)
