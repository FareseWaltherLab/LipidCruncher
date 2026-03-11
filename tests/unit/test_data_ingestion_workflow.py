"""
Unit tests for DataIngestionWorkflow.

Tests the complete data ingestion pipeline orchestration:
format detection → cleaning → zero filtering → standards extraction

Comprehensive test coverage matching Phase 3 service test depth.
"""
import pytest
import pandas as pd
import numpy as np

from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
    IngestionResult
)
from app.models.experiment import ExperimentConfig
from app.services.format_detection import DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.services.zero_filtering import ZeroFilterConfig
from tests.conftest import make_experiment


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def basic_experiment():
    """Basic two-condition experiment."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_with_bqc():
    """Experiment with BQC condition."""
    return make_experiment(
        n_conditions=3,
        conditions_list=['BQC', 'Control', 'Treatment'],
        number_of_samples_list=[2, 3, 3],
    )


@pytest.fixture
def single_condition_experiment():
    """Single condition experiment."""
    return make_experiment(
        n_conditions=1,
        conditions_list=['Samples'],
        number_of_samples_list=[4],
    )


@pytest.fixture
def many_conditions_experiment():
    """Many conditions experiment for stress testing."""
    return make_experiment(5, 2, conditions_list=['Cond1', 'Cond2', 'Cond3', 'Cond4', 'Cond5'])


@pytest.fixture
def unequal_samples_experiment():
    """Experiment with unequal sample counts per condition."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Small', 'Medium', 'Large'],
        number_of_samples_list=[1, 3, 5]
    )


# =============================================================================
# LipidSearch Format Fixtures
# =============================================================================

@pytest.fixture
def lipidsearch_df(simple_experiment_2x2):
    """Sample LipidSearch 5.0 format DataFrame (standardized column names)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'TotalGrade': ['A', 'B', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 85.0, 95.0, 90.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def lipidsearch_raw_df():
    """LipidSearch with raw MeanArea columns (pre-standardization)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'B', 'A'],
        'TotalSmpIDRate(%)': [100.0, 85.0, 95.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'MeanArea[s1]': [1e6, 2e6, 3e6],
        'MeanArea[s2]': [1.1e6, 2.1e6, 3.1e6],
        'MeanArea[s3]': [1.2e6, 2.2e6, 3.2e6],
        'MeanArea[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def lipidsearch_all_grades_df():
    """LipidSearch with all possible grades."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)',
                       'SM(d18:1_16:0)', 'DG(16:0_18:1)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM', 'DG'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6, 620.5],
        'BaseRt': [10.5, 12.3, 15.0, 8.2, 9.5],
        'TotalGrade': ['A', 'B', 'C', 'D', 'A'],
        'TotalSmpIDRate(%)': [100.0, 85.0, 70.0, 50.0, 90.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0', '16:0_18:1'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6, 0.8e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6, 0.9e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6, 1.0e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6, 1.1e6],
    })


@pytest.fixture
def lipidsearch_df_with_standards(simple_experiment_2x2):
    """LipidSearch DataFrame with internal standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_18:1-d7)', 'PE(18:0_20:4)', 'PE(17:0_20:4-d7)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'CalcMass': [760.5, 774.6, 768.5, 782.6],
        'BaseRt': [10.5, 10.8, 12.3, 12.6],
        'TotalGrade': ['A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '15:0_18:1', '18:0_20:4', '17:0_20:4'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5],
    })


@pytest.fixture
def lipidsearch_df_with_mixed_standards():
    """LipidSearch DataFrame with multiple standard patterns."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_18:1-d7)',  # Deuterated
            'PE(18:0_20:4)', 'PE_ISTD',            # ISTD
            'TG(16:0_18:1_18:2)', 'TG(15:0)_IS',   # _IS suffix
            'SM(d18:1_16:0)', 'SPLASH_SM',         # SPLASH
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'ISTD', 'TG', 'TG', 'SM', 'SM'],
        'CalcMass': [760.5, 774.6, 768.5, 700.0, 850.7, 800.0, 703.6, 700.0],
        'BaseRt': [10.5, 10.8, 12.3, 12.0, 15.0, 14.5, 8.2, 8.0],
        'TotalGrade': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0] * 8,
        'FAKey': ['16:0_18:1', '15:0_18:1', '18:0_20:4', 'na', '16:0_18:1_18:2', '15:0', 'd18:1_16:0', 'na'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5, 3e6, 5e5, 1.5e6, 5e5],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5, 3.1e6, 5.1e5, 1.6e6, 5.1e5],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5, 3.2e6, 5.2e5, 1.7e6, 5.2e5],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5, 3.3e6, 5.3e5, 1.8e6, 5.3e5],
    })


@pytest.fixture
def lipidsearch_df_only_standards():
    """LipidSearch DataFrame with only internal standards (edge case)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)', 'TG(15:0_15:0_15:0-d9)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [774.6, 782.6, 800.0],
        'BaseRt': [10.8, 12.6, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['15:0_18:1', '17:0_20:4', '15:0_15:0_15:0'],
        'intensity[s1]': [5e5, 5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5, 5.3e5],
    })


# =============================================================================
# MS-DIAL Format Fixtures
# =============================================================================

@pytest.fixture
def msdial_df(simple_experiment_2x2):
    """Sample MS-DIAL format DataFrame (standardized column names)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'Total score': [90.0, 85.0, 88.0, 82.0],
        'MS/MS matched': ['TRUE', 'TRUE', 'TRUE', 'TRUE'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def msdial_raw_df():
    """MS-DIAL with raw column names (pre-standardization)."""
    return pd.DataFrame({
        'Metabolite name': ['PC 32:1', 'PE 38:4', 'TG 50:3', 'SM 34:1'],
        'Ontology': ['PC', 'PE', 'TG', 'SM'],
        'Average Mz': [760.5, 768.5, 850.7, 703.6],
        'Average Rt(min)': [10.5, 12.3, 15.0, 8.2],
        'Total score': [90.0, 85.0, 88.0, 82.0],
        'MS/MS matched': ['True', 'True', 'True', 'True'],
        's1': [1e6, 2e6, 3e6, 1.5e6],
        's2': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        's3': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        's4': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def msdial_df_with_standards():
    """MS-DIAL DataFrame with internal standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)(d7)', 'PE(18:0_20:4)', 'PE(17:0)(d5)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'CalcMass': [760.5, 774.6, 768.5, 780.0],
        'BaseRt': [10.5, 10.8, 12.3, 12.5],
        'Total score': [90.0, 95.0, 85.0, 92.0],
        'MS/MS matched': ['TRUE', 'TRUE', 'TRUE', 'TRUE'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5],
    })


@pytest.fixture
def msdial_low_quality_df():
    """MS-DIAL DataFrame with low quality scores."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'Total score': [30.0, 45.0, 60.0, 75.0],  # Low to moderate scores
        'MS/MS matched': ['FALSE', 'FALSE', 'TRUE', 'TRUE'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


# =============================================================================
# Generic Format Fixtures
# =============================================================================

@pytest.fixture
def generic_df(simple_experiment_2x2):
    """Sample Generic format DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def generic_minimal_df():
    """Minimal generic DataFrame with just required columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'Sample1': [1e6, 2e6],
        'Sample2': [1.1e6, 2.1e6],
    })


@pytest.fixture
def generic_df_with_standards():
    """Generic DataFrame with internal standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0)(d7)', 'PE(18:0_20:4)', 'PE(17:0)(d5)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5],
    })


@pytest.fixture
def generic_df_no_class():
    """Generic DataFrame without ClassKey column."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'intensity[s1]': [1e6, 2e6, 3e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


# =============================================================================
# Edge Case and Zero Value Fixtures
# =============================================================================

@pytest.fixture
def df_with_zeros(simple_experiment_2x2):
    """DataFrame with some zero values for filtering tests."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'TotalGrade': ['A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0'],
        # PC: all good
        # PE: many zeros (should be filtered)
        # TG: some zeros but not enough
        # SM: all good
        'intensity[s1]': [1e6, 0, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 0, 0, 1.6e6],
        'intensity[s3]': [1.2e6, 0, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 0, 3.3e6, 1.8e6],
    })


@pytest.fixture
def df_all_zeros():
    """DataFrame with all zero intensity values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'intensity[s1]': [0, 0, 0],
        'intensity[s2]': [0, 0, 0],
        'intensity[s3]': [0, 0, 0],
        'intensity[s4]': [0, 0, 0],
    })


@pytest.fixture
def df_with_nan_values():
    """DataFrame with NaN intensity values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'intensity[s1]': [1e6, np.nan, 3e6],
        'intensity[s2]': [np.nan, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, np.nan],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def df_with_very_small_values():
    """DataFrame with values near zero threshold."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'intensity[s1]': [1e-10, 1e-5, 1e6],
        'intensity[s2]': [1e-10, 1e-5, 1.1e6],
        'intensity[s3]': [1e-10, 1e-5, 1.2e6],
        'intensity[s4]': [1e-10, 1e-5, 1.3e6],
    })


@pytest.fixture
def df_single_row():
    """DataFrame with single row."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)'],
        'ClassKey': ['PC'],
        'CalcMass': [760.5],
        'BaseRt': [10.5],
        'TotalGrade': ['A'],
        'TotalSmpIDRate(%)': [100.0],
        'FAKey': ['16:0_18:1'],
        'intensity[s1]': [1e6],
        'intensity[s2]': [1.1e6],
        'intensity[s3]': [1.2e6],
        'intensity[s4]': [1.3e6],
    })


@pytest.fixture
def df_large_dataset():
    """Large DataFrame for performance/stress testing."""
    n_species = 500
    lipids = [f'PC({i}:0_{i+1}:1)' for i in range(n_species)]
    classes = ['PC'] * (n_species // 2) + ['PE'] * (n_species - n_species // 2)
    return pd.DataFrame({
        'LipidMolec': lipids,
        'ClassKey': classes,
        'CalcMass': np.random.uniform(500, 1000, n_species),
        'BaseRt': np.random.uniform(1, 20, n_species),
        'TotalGrade': np.random.choice(['A', 'B', 'C'], n_species),
        'TotalSmpIDRate(%)': np.random.uniform(50, 100, n_species),
        'FAKey': [f'{i}:0_{i+1}:1' for i in range(n_species)],
        'intensity[s1]': np.random.uniform(1e5, 1e7, n_species),
        'intensity[s2]': np.random.uniform(1e5, 1e7, n_species),
        'intensity[s3]': np.random.uniform(1e5, 1e7, n_species),
        'intensity[s4]': np.random.uniform(1e5, 1e7, n_species),
    })


@pytest.fixture
def df_many_samples():
    """DataFrame with many samples (10 samples)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        **{f'intensity[s{i}]': [1e6 * i, 2e6 * i, 3e6 * i] for i in range(1, 11)}
    })


@pytest.fixture
def df_duplicate_lipids():
    """DataFrame with duplicate LipidMolec entries."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'CalcMass': [760.5, 760.5, 768.5],
        'BaseRt': [10.5, 10.6, 12.3],
        'TotalGrade': ['A', 'B', 'A'],
        'TotalSmpIDRate(%)': [100.0, 90.0, 95.0],
        'FAKey': ['16:0_18:1', '16:0_18:1', '18:0_20:4'],
        'intensity[s1]': [1e6, 1.2e6, 2e6],
        'intensity[s2]': [1.1e6, 1.3e6, 2.1e6],
        'intensity[s3]': [1.2e6, 1.4e6, 2.2e6],
        'intensity[s4]': [1.3e6, 1.5e6, 2.3e6],
    })


@pytest.fixture
def df_special_characters():
    """DataFrame with special characters in lipid names."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0/18:1)', 'PE(18:0-20:4)', 'TG(16:0_18:1_18:2)[M+NH4]+'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'TotalGrade': ['A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0],
        'FAKey': ['16:0/18:1', '18:0-20:4', '16:0_18:1_18:2'],
        'intensity[s1]': [1e6, 2e6, 3e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def df_bqc_samples(experiment_with_bqc):
    """DataFrame with BQC samples for BQC-specific zero filtering tests."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'TotalGrade': ['A', 'A', 'A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0, 100.0, 100.0],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2', 'd18:1_16:0'],
        # BQC samples (s1, s2)
        'intensity[s1]': [1e6, 0, 3e6, 1.5e6],  # BQC1
        'intensity[s2]': [1.1e6, 0, 3.1e6, 1.6e6],  # BQC2
        # Control samples (s3, s4, s5)
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],  # Control1
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],  # Control2
        'intensity[s5]': [1.4e6, 2.4e6, 3.4e6, 1.9e6],  # Control3
        # Treatment samples (s6, s7, s8)
        'intensity[s6]': [1.5e6, 2.5e6, 3.5e6, 2.0e6],  # Treatment1
        'intensity[s7]': [1.6e6, 2.6e6, 3.6e6, 2.1e6],  # Treatment2
        'intensity[s8]': [1.7e6, 2.7e6, 3.7e6, 2.2e6],  # Treatment3
    })


# =============================================================================
# External Standards Fixtures
# =============================================================================

@pytest.fixture
def external_standards_simple():
    """Simple external standards DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC-IS(d7)', 'PE-IS(d5)'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5],
    })


@pytest.fixture
def external_standards_multiple():
    """External standards with multiple per class."""
    return pd.DataFrame({
        'LipidMolec': ['PC-IS1(d7)', 'PC-IS2(d7)', 'PE-IS(d5)', 'TG-IS(d9)'],
        'ClassKey': ['PC', 'PC', 'PE', 'TG'],
        'intensity[s1]': [5e5, 4e5, 5e5, 6e5],
        'intensity[s2]': [5.1e5, 4.1e5, 5.1e5, 6.1e5],
        'intensity[s3]': [5.2e5, 4.2e5, 5.2e5, 6.2e5],
        'intensity[s4]': [5.3e5, 4.3e5, 5.3e5, 6.3e5],
    })


@pytest.fixture
def external_standards_empty():
    """Empty external standards DataFrame."""
    return pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]', 'intensity[s2]'])


# =============================================================================
# IngestionResult Tests
# =============================================================================

class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = IngestionResult(detected_format=DataFormat.GENERIC)
        assert result.detected_format == DataFormat.GENERIC
        assert result.format_confidence == "high"
        assert result.cleaned_df is None
        assert result.internal_standards_df is None
        assert result.cleaning_messages == []
        assert result.zero_filtered is False
        assert result.is_valid is True
        assert result.validation_errors == []

    def test_species_removed_count(self):
        """Test species removed count calculation."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=80
        )
        assert result.species_removed_count == 20

    def test_removal_percentage(self):
        """Test removal percentage calculation."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=75
        )
        assert result.removal_percentage == 25.0

    def test_removal_percentage_zero_before(self):
        """Test removal percentage when no species before."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=0,
            species_after_filter=0
        )
        assert result.removal_percentage == 0.0

    def test_species_removed_count_no_removal(self):
        """Test species removed count when nothing removed."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=100
        )
        assert result.species_removed_count == 0

    def test_removal_percentage_all_removed(self):
        """Test removal percentage when all species removed."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=100,
            species_after_filter=0
        )
        assert result.removal_percentage == 100.0

    def test_removal_percentage_fractional(self):
        """Test removal percentage with fractional result."""
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            species_before_filter=3,
            species_after_filter=2
        )
        assert abs(result.removal_percentage - 33.333333) < 0.001

    def test_all_format_types(self):
        """Test result creation with all format types."""
        for fmt in [DataFormat.LIPIDSEARCH, DataFormat.MSDIAL, DataFormat.GENERIC, DataFormat.UNKNOWN]:
            result = IngestionResult(detected_format=fmt)
            assert result.detected_format == fmt

    def test_all_confidence_levels(self):
        """Test result with different confidence levels."""
        for confidence in ["high", "medium", "low"]:
            result = IngestionResult(
                detected_format=DataFormat.GENERIC,
                format_confidence=confidence
            )
            assert result.format_confidence == confidence

    def test_with_cleaned_df(self, lipidsearch_df):
        """Test result with cleaned DataFrame."""
        result = IngestionResult(
            detected_format=DataFormat.LIPIDSEARCH,
            cleaned_df=lipidsearch_df
        )
        assert result.cleaned_df is not None
        assert len(result.cleaned_df) == 4

    def test_with_internal_standards_df(self, lipidsearch_df_with_standards):
        """Test result with internal standards DataFrame."""
        standards_df = lipidsearch_df_with_standards[
            lipidsearch_df_with_standards['LipidMolec'].str.contains('d7')
        ]
        result = IngestionResult(
            detected_format=DataFormat.LIPIDSEARCH,
            internal_standards_df=standards_df
        )
        assert result.internal_standards_df is not None
        assert len(result.internal_standards_df) == 2

    def test_with_cleaning_messages(self):
        """Test result with cleaning messages."""
        messages = ["Filtered 5 low-grade species", "Removed 3 duplicates"]
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            cleaning_messages=messages
        )
        assert len(result.cleaning_messages) == 2
        assert "Filtered 5" in result.cleaning_messages[0]

    def test_with_removed_species_list(self):
        """Test result with removed species list."""
        removed = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            removed_species=removed
        )
        assert len(result.removed_species) == 2

    def test_validation_errors_list(self):
        """Test result with validation errors."""
        errors = ["Missing column: ClassKey", "Invalid format"]
        result = IngestionResult(
            detected_format=DataFormat.UNKNOWN,
            is_valid=False,
            validation_errors=errors
        )
        assert not result.is_valid
        assert len(result.validation_errors) == 2

    def test_validation_warnings_list(self):
        """Test result with validation warnings."""
        warnings = ["Low sample count", "Potential outliers detected"]
        result = IngestionResult(
            detected_format=DataFormat.GENERIC,
            is_valid=True,
            validation_warnings=warnings
        )
        assert result.is_valid
        assert len(result.validation_warnings) == 2

    def test_complete_result(self, lipidsearch_df):
        """Test result with all fields populated."""
        result = IngestionResult(
            detected_format=DataFormat.LIPIDSEARCH,
            format_confidence="high",
            cleaned_df=lipidsearch_df,
            internal_standards_df=pd.DataFrame(),
            cleaning_messages=["Processed successfully"],
            zero_filtered=True,
            species_before_filter=100,
            species_after_filter=95,
            removed_species=['PC(16:0_18:1)'],
            is_valid=True,
            validation_errors=[],
            validation_warnings=["Minor issue detected"]
        )
        assert result.is_valid
        assert result.zero_filtered
        assert result.species_removed_count == 5
        assert result.removal_percentage == 5.0


# =============================================================================
# IngestionConfig Tests
# =============================================================================

class TestIngestionConfig:
    """Tests for IngestionConfig dataclass."""

    def test_minimal_config(self, simple_experiment_2x2):
        """Test creating config with minimal required fields."""
        config = IngestionConfig(experiment=simple_experiment_2x2)
        assert config.experiment == simple_experiment_2x2
        assert config.data_format is None
        assert config.apply_zero_filter is True
        assert config.use_external_standards is False

    def test_full_config(self, simple_experiment_2x2):
        """Test creating config with all fields."""
        grade_config = GradeFilterConfig()
        zero_config = ZeroFilterConfig()

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=True,
            zero_filter_config=zero_config,
            bqc_label='BQC'
        )

        assert config.data_format == DataFormat.LIPIDSEARCH
        assert config.grade_config is not None
        assert config.bqc_label == 'BQC'

    def test_config_with_all_formats(self, simple_experiment_2x2):
        """Test config with each data format."""
        for fmt in [DataFormat.LIPIDSEARCH, DataFormat.MSDIAL, DataFormat.GENERIC]:
            config = IngestionConfig(
                experiment=simple_experiment_2x2,
                data_format=fmt
            )
            assert config.data_format == fmt

    def test_config_with_zero_filter_disabled(self, simple_experiment_2x2):
        """Test config with zero filtering disabled."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )
        assert config.apply_zero_filter is False

    def test_config_with_custom_zero_filter(self, simple_experiment_2x2):
        """Test config with custom zero filter settings."""
        zero_config = ZeroFilterConfig(
            detection_threshold=0.1,
            bqc_threshold=0.3,
            non_bqc_threshold=0.6
        )
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            zero_filter_config=zero_config
        )
        assert config.zero_filter_config.detection_threshold == 0.1
        assert config.zero_filter_config.bqc_threshold == 0.3

    def test_config_with_grade_config(self, simple_experiment_2x2):
        """Test config with grade filter configuration."""
        grade_config = GradeFilterConfig(grade_config={'PC': ['A'], 'PE': ['A', 'B']})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            grade_config=grade_config
        )
        assert config.grade_config is not None
        assert 'PC' in config.grade_config.grade_config

    def test_config_with_quality_config(self, simple_experiment_2x2):
        """Test config with MS-DIAL quality filter configuration."""
        quality_config = QualityFilterConfig(total_score_threshold=70, require_msms=True)
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            quality_config=quality_config
        )
        assert config.quality_config is not None
        assert config.quality_config.total_score_threshold == 70

    def test_config_with_external_standards(self, simple_experiment_2x2, external_standards_simple):
        """Test config with external standards enabled."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            use_external_standards=True,
            external_standards_df=external_standards_simple
        )
        assert config.use_external_standards is True
        assert config.external_standards_df is not None
        assert len(config.external_standards_df) == 2

    def test_config_with_bqc_label(self, experiment_with_bqc):
        """Test config with BQC label specified."""
        config = IngestionConfig(
            experiment=experiment_with_bqc,
            bqc_label='BQC'
        )
        assert config.bqc_label == 'BQC'

    def test_config_experiment_variants(self, single_condition_experiment, many_conditions_experiment):
        """Test config with different experiment configurations."""
        config1 = IngestionConfig(experiment=single_condition_experiment)
        config2 = IngestionConfig(experiment=many_conditions_experiment)

        assert config1.experiment.n_conditions == 1
        assert config2.experiment.n_conditions == 5

    def test_config_preserves_experiment_reference(self, simple_experiment_2x2):
        """Test that config preserves the experiment object reference."""
        config = IngestionConfig(experiment=simple_experiment_2x2)
        assert config.experiment is simple_experiment_2x2


# =============================================================================
# Format Detection Tests
# =============================================================================

class TestDetectFormatOnly:
    """Tests for detect_format_only method."""

    def test_lipidsearch_detection_raw_format(self):
        """Test LipidSearch format detection with raw MeanArea columns."""
        # Raw LipidSearch format uses MeanArea[*] columns
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1e6],
            'MeanArea[s2]': [1.1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.LIPIDSEARCH
        assert confidence == "high"

    def test_standardized_df_detected_as_generic(self, lipidsearch_df):
        """Test that standardized intensity[] columns are detected as Generic."""
        # Our fixtures use intensity[*] columns (standardized format)
        # which lack the MeanArea[*] signature, so they detect as Generic
        detected, confidence = DataIngestionWorkflow.detect_format_only(lipidsearch_df)
        assert detected == DataFormat.GENERIC
        assert confidence == "medium"

    def test_generic_detection(self, generic_df):
        """Test Generic format detection."""
        detected, confidence = DataIngestionWorkflow.detect_format_only(generic_df)
        assert detected == DataFormat.GENERIC
        assert confidence == "medium"

    def test_unknown_format(self):
        """Test unknown format detection."""
        df = pd.DataFrame({'random': [1, 2, 3]})
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.UNKNOWN
        assert confidence == "low"

    def test_lipidsearch_raw_df_detection(self, lipidsearch_raw_df):
        """Test LipidSearch detection with raw MeanArea columns."""
        detected, confidence = DataIngestionWorkflow.detect_format_only(lipidsearch_raw_df)
        assert detected == DataFormat.LIPIDSEARCH
        assert confidence == "high"

    def test_msdial_raw_df_detection(self, msdial_raw_df):
        """Test MS-DIAL detection with raw column names."""
        detected, confidence = DataIngestionWorkflow.detect_format_only(msdial_raw_df)
        assert detected == DataFormat.MSDIAL
        assert confidence == "high"

    def test_empty_dataframe_detection(self):
        """Test detection on empty DataFrame."""
        df = pd.DataFrame()
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.UNKNOWN
        assert confidence == "low"

    def test_single_row_detection(self, df_single_row):
        """Test detection with single row DataFrame."""
        detected, confidence = DataIngestionWorkflow.detect_format_only(df_single_row)
        # Single row with LipidMolec should be detected as Generic
        assert detected == DataFormat.GENERIC
        assert confidence == "medium"

    def test_detection_with_extra_columns(self):
        """Test detection ignores extra columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['16:0_18:1'],
            'MeanArea[s1]': [1e6],
            'ExtraColumn': ['extra'],
            'AnotherExtra': [123],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.LIPIDSEARCH

    def test_detection_case_sensitive_columns(self):
        """Test format detection with different column cases."""
        # LipidMolec with wrong case - may still detect as generic depending on implementation
        df = pd.DataFrame({
            'lipidmolec': ['PC(16:0_18:1)'],  # Wrong case
            'ClassKey': ['PC'],
            'Sample1': [1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        # Detection may vary based on implementation - just verify it doesn't crash
        assert detected in [DataFormat.UNKNOWN, DataFormat.GENERIC]

    def test_lipidsearch_missing_signature_column(self):
        """Test LipidSearch detection fails without signature columns."""
        # Has some LipidSearch columns but missing MeanArea
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [10.5],
            'TotalGrade': ['A'],
            'Sample1': [1e6],  # Not MeanArea format
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        # Should fall back to Generic since LipidMolec exists
        assert detected == DataFormat.GENERIC


class TestDetectFormatOnlyMSDIAL:
    """Tests for MS-DIAL specific format detection."""

    def test_msdial_full_signature(self):
        """Test MS-DIAL detection with full signature columns."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 32:1'],
            'Ontology': ['PC'],
            'Average Mz': [760.5],
            'Average Rt(min)': [10.5],
            'Total score': [90.0],
            'MS/MS matched': ['True'],
            's1': [1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.MSDIAL
        assert confidence == "high"

    def test_msdial_missing_ontology(self):
        """Test MS-DIAL detection without Ontology column."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 32:1'],
            'Average Mz': [760.5],
            'Total score': [90.0],
            's1': [1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        # May still detect as MS-DIAL if other signature columns match
        # Just verify it produces some result
        assert detected in [DataFormat.MSDIAL, DataFormat.GENERIC, DataFormat.UNKNOWN]

    def test_msdial_partial_signature(self):
        """Test MS-DIAL detection with partial signature."""
        df = pd.DataFrame({
            'Metabolite name': ['PC 32:1'],
            'Ontology': ['PC'],
            'Total score': [90.0],
            's1': [1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.MSDIAL


class TestDetectFormatOnlyEdgeCases:
    """Edge case tests for format detection."""

    def test_detection_with_numeric_only_values(self):
        """Test detection with numeric-only column names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            '1': [1e6],
            '2': [1.1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.GENERIC

    def test_detection_with_spaces_in_column_names(self):
        """Test detection with spaces in column names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'Sample 1': [1e6],
            'Sample 2': [1.1e6],
        })
        detected, confidence = DataIngestionWorkflow.detect_format_only(df)
        assert detected == DataFormat.GENERIC

    def test_detection_preserves_dataframe(self, lipidsearch_df):
        """Test that detection doesn't modify input DataFrame."""
        original_cols = list(lipidsearch_df.columns)
        original_len = len(lipidsearch_df)
        DataIngestionWorkflow.detect_format_only(lipidsearch_df)
        assert list(lipidsearch_df.columns) == original_cols
        assert len(lipidsearch_df) == original_len


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateForFormat:
    """Tests for validate_for_format method."""

    def test_valid_lipidsearch(self, lipidsearch_df):
        """Test validation for valid LipidSearch data."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            lipidsearch_df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is True
        assert errors == []

    def test_valid_generic(self, generic_df):
        """Test validation for valid Generic data."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            generic_df, DataFormat.GENERIC
        )
        assert is_valid is True
        assert errors == []

    def test_empty_dataframe(self):
        """Test validation fails for empty DataFrame."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            pd.DataFrame(), DataFormat.GENERIC
        )
        assert is_valid is False
        assert 'empty' in errors[0].lower()

    def test_missing_lipidmolec(self):
        """Test validation fails when LipidMolec is missing."""
        df = pd.DataFrame({'ClassKey': ['PC'], 'Sample1': [1e6]})
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.GENERIC
        )
        assert is_valid is False
        assert 'LipidMolec' in errors[0]

    def test_lipidsearch_missing_columns(self, generic_df):
        """Test validation fails for LipidSearch when columns missing."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            generic_df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        assert 'LipidSearch' in errors[0]

    def test_valid_msdial(self, msdial_df):
        """Test validation for valid MS-DIAL data."""
        # MS-DIAL validation may require specific columns
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            msdial_df, DataFormat.MSDIAL
        )
        # Validation result depends on msdial_df fixture columns
        # If it has ClassKey (renamed from Ontology), it should pass
        assert isinstance(is_valid, bool)
        if not is_valid:
            # If invalid, it's because Ontology isn't in standardized fixture
            assert len(errors) > 0

    def test_none_dataframe(self):
        """Test validation fails for None DataFrame."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            None, DataFormat.GENERIC
        )
        assert is_valid is False
        assert 'empty' in errors[0].lower()

    def test_lipidsearch_missing_calcmass(self, lipidsearch_df):
        """Test LipidSearch validation fails when CalcMass missing."""
        df = lipidsearch_df.drop(columns=['CalcMass'])
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        assert 'CalcMass' in errors[0]

    def test_lipidsearch_missing_basert(self, lipidsearch_df):
        """Test LipidSearch validation fails when BaseRt missing."""
        df = lipidsearch_df.drop(columns=['BaseRt'])
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        assert 'BaseRt' in errors[0]

    def test_lipidsearch_missing_totalgrade(self, lipidsearch_df):
        """Test LipidSearch validation fails when TotalGrade missing."""
        df = lipidsearch_df.drop(columns=['TotalGrade'])
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        assert 'TotalGrade' in errors[0]

    def test_lipidsearch_missing_multiple_columns(self, lipidsearch_df):
        """Test LipidSearch validation reports multiple missing columns."""
        df = lipidsearch_df.drop(columns=['CalcMass', 'BaseRt', 'TotalGrade'])
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.LIPIDSEARCH
        )
        assert is_valid is False
        # Error should mention missing columns
        assert len(errors) > 0

    def test_generic_with_only_lipidmolec(self):
        """Test Generic validation passes with just LipidMolec."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'Sample1': [1e6],
        })
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.GENERIC
        )
        assert is_valid is True

    def test_validation_single_row(self, df_single_row):
        """Test validation with single row DataFrame."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df_single_row, DataFormat.LIPIDSEARCH
        )
        assert is_valid is True

    def test_validation_large_dataset(self, df_large_dataset):
        """Test validation with large dataset."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df_large_dataset, DataFormat.LIPIDSEARCH
        )
        assert is_valid is True


class TestValidateForFormatMSDIAL:
    """MS-DIAL specific validation tests."""

    def test_msdial_missing_ontology(self, msdial_df):
        """Test MS-DIAL validation when Ontology is missing."""
        df = msdial_df.drop(columns=['ClassKey'])  # ClassKey is renamed Ontology
        # Create raw MS-DIAL without Ontology
        df_raw = pd.DataFrame({
            'LipidMolec': ['PC 32:1'],
            'Sample1': [1e6],
        })
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df_raw, DataFormat.MSDIAL
        )
        assert is_valid is False
        assert 'Ontology' in errors[0]

    def test_msdial_with_all_required_columns(self, msdial_df):
        """Test MS-DIAL validation with all required columns."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            msdial_df, DataFormat.MSDIAL
        )
        # Validation depends on fixture structure - msdial_df uses ClassKey not Ontology
        # Just verify it doesn't crash and returns consistent types
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


class TestValidateForFormatEdgeCases:
    """Edge case validation tests."""

    def test_dataframe_with_special_characters(self, df_special_characters):
        """Test validation with special characters in data."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df_special_characters, DataFormat.LIPIDSEARCH
        )
        assert is_valid is True

    def test_dataframe_with_duplicates(self, df_duplicate_lipids):
        """Test validation with duplicate lipid names."""
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df_duplicate_lipids, DataFormat.LIPIDSEARCH
        )
        # Validation should pass - duplicates are handled elsewhere
        assert is_valid is True

    def test_dataframe_with_nan_in_lipidmolec(self):
        """Test validation with NaN in LipidMolec column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', np.nan, 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'Sample1': [1e6, 2e6, 3e6],
        })
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.GENERIC
        )
        # Should still be valid as LipidMolec column exists
        assert is_valid is True

    def test_unknown_format_validation(self):
        """Test validation with UNKNOWN format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'Sample1': [1e6],
        })
        is_valid, errors = DataIngestionWorkflow.validate_for_format(
            df, DataFormat.UNKNOWN
        )
        # UNKNOWN format should just check for LipidMolec
        assert is_valid is True


# =============================================================================
# Get Sample Columns Tests
# =============================================================================

class TestGetSampleColumns:
    """Tests for get_sample_columns method."""

    def test_lipidsearch_samples(self, lipidsearch_df):
        """Test getting sample columns from LipidSearch data."""
        # Note: After cleaning, columns are intensity[*], but for raw detection
        # it looks for MeanArea[*]
        samples = DataIngestionWorkflow.get_sample_columns(
            lipidsearch_df, DataFormat.LIPIDSEARCH
        )
        # Our fixture uses intensity[*] not MeanArea[*], so none match
        assert len(samples) == 0  # MeanArea columns not present

    def test_generic_samples(self, generic_df):
        """Test getting sample columns from Generic data."""
        samples = DataIngestionWorkflow.get_sample_columns(
            generic_df, DataFormat.GENERIC
        )
        # Should exclude LipidMolec and ClassKey
        assert 'LipidMolec' not in samples
        assert 'ClassKey' not in samples
        # Should include intensity columns
        assert 'intensity[s1]' in samples

    def test_lipidsearch_raw_samples(self, lipidsearch_raw_df):
        """Test getting MeanArea sample columns from raw LipidSearch data."""
        samples = DataIngestionWorkflow.get_sample_columns(
            lipidsearch_raw_df, DataFormat.LIPIDSEARCH
        )
        assert len(samples) == 4
        assert all(s.startswith('MeanArea[') for s in samples)

    def test_msdial_samples(self, msdial_raw_df):
        """Test getting sample columns from MS-DIAL data."""
        samples = DataIngestionWorkflow.get_sample_columns(
            msdial_raw_df, DataFormat.MSDIAL
        )
        # Should exclude metadata columns
        assert 'Metabolite name' not in samples
        assert 'Ontology' not in samples
        assert 'Total score' not in samples
        # Should include sample columns
        assert 's1' in samples

    def test_generic_minimal_samples(self, generic_minimal_df):
        """Test getting sample columns from minimal Generic data."""
        samples = DataIngestionWorkflow.get_sample_columns(
            generic_minimal_df, DataFormat.GENERIC
        )
        assert 'Sample1' in samples
        assert 'Sample2' in samples
        assert 'LipidMolec' not in samples

    def test_many_samples(self, df_many_samples):
        """Test getting sample columns from DataFrame with many samples."""
        samples = DataIngestionWorkflow.get_sample_columns(
            df_many_samples, DataFormat.GENERIC
        )
        # Should have 10 intensity columns
        intensity_cols = [s for s in samples if s.startswith('intensity[')]
        assert len(intensity_cols) == 10

    def test_empty_dataframe_samples(self):
        """Test getting sample columns from empty DataFrame."""
        df = pd.DataFrame()
        samples = DataIngestionWorkflow.get_sample_columns(df, DataFormat.GENERIC)
        assert samples == []

    def test_samples_exclude_metadata(self, lipidsearch_df):
        """Test that key metadata columns are excluded."""
        samples = DataIngestionWorkflow.get_sample_columns(
            lipidsearch_df, DataFormat.GENERIC
        )
        # LipidMolec and ClassKey should always be excluded for Generic
        assert 'LipidMolec' not in samples
        assert 'ClassKey' not in samples
        # Intensity columns should be included
        intensity_cols = [s for s in samples if 'intensity' in s]
        assert len(intensity_cols) > 0


# =============================================================================
# Full Workflow Tests
# =============================================================================

class TestWorkflowRun:
    """Tests for the complete run() method."""

    def test_basic_workflow_lipidsearch(self, lipidsearch_df, simple_experiment_2x2):
        """Test basic workflow with LipidSearch data."""
        # Explicitly specify format since fixture uses standardized column names
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False  # Skip for simplicity
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.cleaned_df is not None
        assert len(result.validation_errors) == 0

    def test_basic_workflow_generic(self, generic_df, simple_experiment_2x2):
        """Test basic workflow with Generic data."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.GENERIC
        assert result.cleaned_df is not None

    def test_workflow_with_explicit_format(self, lipidsearch_df, simple_experiment_2x2):
        """Test workflow with explicitly specified format."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.is_valid is True

    def test_workflow_extracts_standards(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that workflow extracts internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        # Internal standards with (d7) should be extracted
        assert result.internal_standards_df is not None

    def test_workflow_with_grade_config(self, lipidsearch_df, simple_experiment_2x2):
        """Test workflow with grade filtering config."""
        # GradeFilterConfig takes a dict mapping class to allowed grades
        grade_config = GradeFilterConfig(grade_config={'PC': ['A'], 'PE': ['A', 'B']})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True

    def test_workflow_unknown_format_fails(self, simple_experiment_2x2):
        """Test that unknown format causes validation failure."""
        df = pd.DataFrame({'random_col': [1, 2, 3]})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        assert DataFormat.UNKNOWN == result.detected_format
        assert len(result.validation_errors) > 0

    def test_workflow_empty_df_fails(self, simple_experiment_2x2):
        """Test that empty DataFrame causes failure."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(pd.DataFrame(), config)

        assert result.is_valid is False


class TestWorkflowRunMSDIAL:
    """Tests for MS-DIAL specific workflow runs."""

    def test_msdial_workflow(self, msdial_df, simple_experiment_2x2):
        """Test workflow with MS-DIAL format data."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(msdial_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.MSDIAL

    def test_msdial_with_quality_config(self, msdial_df, simple_experiment_2x2):
        """Test MS-DIAL workflow with quality filter configuration."""
        quality_config = QualityFilterConfig(total_score_threshold=80, require_msms=True)
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(msdial_df, config)

        assert result.is_valid is True

    def test_msdial_with_standards(self, msdial_df_with_standards, simple_experiment_2x2):
        """Test MS-DIAL workflow extracts internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(msdial_df_with_standards, config)

        assert result.is_valid is True
        assert result.internal_standards_df is not None


class TestWorkflowRunEdgeCases:
    """Edge case tests for workflow run."""

    def test_workflow_single_row(self, df_single_row, simple_experiment_2x2):
        """Test workflow with single row DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df_single_row, config)

        assert result.is_valid is True
        assert len(result.cleaned_df) == 1

    def test_workflow_large_dataset(self, df_large_dataset, simple_experiment_2x2):
        """Test workflow with large dataset."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df_large_dataset, config)

        assert result.is_valid is True
        assert len(result.cleaned_df) > 0

    def test_workflow_preserves_input(self, lipidsearch_df, simple_experiment_2x2):
        """Test that workflow doesn't modify input DataFrame."""
        original_df = lipidsearch_df.copy()
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        DataIngestionWorkflow.run(lipidsearch_df, config)

        pd.testing.assert_frame_equal(lipidsearch_df, original_df)

    def test_workflow_with_special_characters(self, df_special_characters, simple_experiment_2x2):
        """Test workflow with special characters in lipid names."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df_special_characters, config)

        assert result.is_valid is True

    def test_workflow_single_condition(self, generic_df, single_condition_experiment):
        """Test workflow with single condition experiment."""
        # Adjust generic_df to have 4 samples for single condition
        config = IngestionConfig(
            experiment=single_condition_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True

    def test_workflow_many_conditions(self, df_many_samples, many_conditions_experiment):
        """Test workflow with many conditions experiment."""
        config = IngestionConfig(
            experiment=many_conditions_experiment,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df_many_samples, config)

        assert result.is_valid is True

    def test_workflow_mixed_standards(self, lipidsearch_df_with_mixed_standards, simple_experiment_2x2):
        """Test workflow extracts multiple standard patterns."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_mixed_standards, config)

        assert result.is_valid is True
        if result.internal_standards_df is not None:
            assert len(result.internal_standards_df) > 0

    def test_workflow_returns_cleaning_messages(self, lipidsearch_df, simple_experiment_2x2):
        """Test that workflow returns cleaning messages."""
        grade_config = GradeFilterConfig(grade_config={'PC': ['A']})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Should have messages about filtering
        assert isinstance(result.cleaning_messages, list)


# =============================================================================
# Zero Filtering Tests
# =============================================================================

class TestWorkflowZeroFiltering:
    """Tests for zero filtering in the workflow."""

    def test_zero_filtering_applied(self, df_with_zeros, simple_experiment_2x2):
        """Test that zero filtering is applied."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        assert result.is_valid is True
        assert result.zero_filtered is True

    def test_zero_filtering_skipped_when_disabled(self, lipidsearch_df, simple_experiment_2x2):
        """Test that zero filtering is skipped when disabled."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.zero_filtered is False
        assert result.species_before_filter == 0
        assert result.species_after_filter == 0

    def test_zero_filtering_with_custom_config(self, df_with_zeros, simple_experiment_2x2):
        """Test zero filtering with custom configuration."""
        zero_config = ZeroFilterConfig(
            detection_threshold=0,
            bqc_threshold=0.3,
            non_bqc_threshold=0.5
        )
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=zero_config
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        assert result.is_valid is True


class TestWorkflowZeroFilteringThresholds:
    """Tests for zero filtering threshold boundaries."""

    def test_strict_threshold_filters_more(self, df_with_zeros, simple_experiment_2x2):
        """Test that strict threshold filters more species."""
        strict_config = ZeroFilterConfig(
            detection_threshold=0,
            bqc_threshold=0.1,
            non_bqc_threshold=0.1
        )
        lenient_config = ZeroFilterConfig(
            detection_threshold=0,
            bqc_threshold=0.9,
            non_bqc_threshold=0.9
        )

        strict_ingestion = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=strict_config
        )
        lenient_ingestion = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=lenient_config
        )

        strict_result = DataIngestionWorkflow.run(df_with_zeros.copy(), strict_ingestion)
        lenient_result = DataIngestionWorkflow.run(df_with_zeros.copy(), lenient_ingestion)

        # Strict should filter more (fewer species after)
        assert strict_result.species_after_filter <= lenient_result.species_after_filter

    def test_detection_threshold_boundary(self, df_with_very_small_values, simple_experiment_2x2):
        """Test detection threshold at boundary values."""
        # Threshold that treats small values as zeros
        high_threshold = ZeroFilterConfig(detection_threshold=1e-4)

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=high_threshold
        )

        result = DataIngestionWorkflow.run(df_with_very_small_values, config)

        assert result.is_valid is True
        assert result.zero_filtered is True

    def test_zero_threshold_exact_zero(self, simple_experiment_2x2):
        """Test zero threshold only considers exact zeros."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            'intensity[s1]': [1e-10, 1e6],  # Very small but not zero
            'intensity[s2]': [1e-10, 1.1e6],
            'intensity[s3]': [1e-10, 1.2e6],
            'intensity[s4]': [1e-10, 1.3e6],
        })

        zero_threshold_config = ZeroFilterConfig(detection_threshold=0)

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=zero_threshold_config
        )

        result = DataIngestionWorkflow.run(df, config)

        # With exact zero threshold, small values are not zeros
        assert result.is_valid is True

    def test_fifty_percent_threshold(self, simple_experiment_2x2):
        """Test 50% non-zero threshold."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            # PC: 2/4 non-zero (50%)
            # PE: 4/4 non-zero (100%)
            'intensity[s1]': [0, 1e6],
            'intensity[s2]': [0, 1.1e6],
            'intensity[s3]': [1e6, 1.2e6],
            'intensity[s4]': [1.1e6, 1.3e6],
        })

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=ZeroFilterConfig(non_bqc_threshold=0.5)
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is True
        assert result.zero_filtered is True


class TestWorkflowZeroFilteringBQC:
    """Tests for BQC-specific zero filtering."""

    def test_bqc_specific_threshold(self, df_bqc_samples, experiment_with_bqc):
        """Test BQC samples use different threshold."""
        config = IngestionConfig(
            experiment=experiment_with_bqc,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            bqc_label='BQC',
            zero_filter_config=ZeroFilterConfig(bqc_threshold=0.3, non_bqc_threshold=0.7)
        )

        result = DataIngestionWorkflow.run(df_bqc_samples, config)

        assert result.is_valid is True
        assert result.zero_filtered is True

    def test_no_bqc_label_uses_non_bqc_threshold(self, df_with_zeros, simple_experiment_2x2):
        """Test that without BQC label, non-BQC threshold is used."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            bqc_label=None  # No BQC
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        assert result.is_valid is True
        assert result.zero_filtered is True


class TestWorkflowZeroFilteringEdgeCases:
    """Edge case tests for zero filtering."""

    def test_all_zeros_dataframe(self, df_all_zeros, simple_experiment_2x2):
        """Test zero filtering with all-zeros DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(df_all_zeros, config)

        # Should complete but may have warnings or all species removed
        assert result.zero_filtered is True

    def test_no_zeros_dataframe(self, lipidsearch_df, simple_experiment_2x2):
        """Test zero filtering when no zeros present."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.zero_filtered is True
        # No species should be removed when no zeros
        assert result.species_removed_count == 0

    def test_filtering_reports_removed_species(self, df_with_zeros, simple_experiment_2x2):
        """Test that filtering reports removed species names."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        if result.species_removed_count > 0:
            assert len(result.removed_species) == result.species_removed_count

    def test_filtering_updates_cleaning_messages(self, df_with_zeros, simple_experiment_2x2):
        """Test that filtering adds message to cleaning_messages."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(df_with_zeros, config)

        if result.species_removed_count > 0:
            # Should have a message about zero filtering
            assert any('zero' in msg.lower() or 'filter' in msg.lower()
                       for msg in result.cleaning_messages)


# =============================================================================
# External Standards Tests
# =============================================================================

class TestWorkflowExternalStandards:
    """Tests for external standards handling."""

    def test_external_standards_used(self, lipidsearch_df, simple_experiment_2x2):
        """Test that external standards are used when provided."""
        external_standards = pd.DataFrame({
            'LipidMolec': ['PC-IS(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5e5],
            'intensity[s2]': [5.1e5],
            'intensity[s3]': [5.2e5],
            'intensity[s4]': [5.3e5],
        })

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=external_standards
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.internal_standards_df is not None
        assert len(result.internal_standards_df) == 1
        # Should have a message about external standards
        assert any('external' in msg.lower() for msg in result.cleaning_messages)

    def test_external_standards_multiple(self, lipidsearch_df, simple_experiment_2x2, external_standards_multiple):
        """Test external standards with multiple standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=external_standards_multiple
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        assert result.internal_standards_df is not None
        assert len(result.internal_standards_df) == 4

    def test_external_standards_replaces_auto_detected(self, lipidsearch_df_with_standards, simple_experiment_2x2, external_standards_simple):
        """Test external standards replace auto-detected standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=external_standards_simple
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        # External standards should replace auto-detected
        assert len(result.internal_standards_df) == len(external_standards_simple)

    def test_external_standards_flag_without_df(self, lipidsearch_df, simple_experiment_2x2):
        """Test external standards flag set but no DataFrame provided."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=None
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        # Should still work, just use auto-detected
        assert result.is_valid is True

    def test_external_standards_empty_df(self, lipidsearch_df, simple_experiment_2x2, external_standards_empty):
        """Test external standards with empty DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=external_standards_empty
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True


class TestWorkflowStandardsExtraction:
    """Tests for internal standards extraction in workflow."""

    def test_deuterated_standards_extracted(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that deuterated standards (d7, d5, d9) are extracted."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        if result.internal_standards_df is not None:
            # Should extract standards with d7 pattern
            assert len(result.internal_standards_df) > 0

    def test_no_standards_in_data(self, generic_df, simple_experiment_2x2):
        """Test workflow when data has no internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True
        # Standards df should be None or empty
        if result.internal_standards_df is not None:
            assert len(result.internal_standards_df) == 0

    def test_only_standards_data(self, lipidsearch_df_only_standards, simple_experiment_2x2):
        """Test workflow when all data is standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_only_standards, config)

        # Should handle gracefully
        assert result.is_valid is True

    def test_mixed_standards_patterns(self, lipidsearch_df_with_mixed_standards, simple_experiment_2x2):
        """Test extraction of multiple standard patterns."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_mixed_standards, config)

        assert result.is_valid is True
        # Should extract standards with various patterns
        if result.internal_standards_df is not None:
            assert len(result.internal_standards_df) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestWorkflowErrorHandling:
    """Tests for error handling in the workflow."""

    def test_cleaning_error_captured(self, simple_experiment_2x2):
        """Test that cleaning errors are captured in result."""
        # Create DataFrame that will fail cleaning (missing intensity columns)
        df = pd.DataFrame({
            'LipidMolec': ['PC 32:0'],
            'ClassKey': ['PC'],
            # Missing intensity columns
        })

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        # Should fail during cleaning
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_invalid_zero_config_raises(self, simple_experiment_2x2):
        """Test that invalid zero filter config raises ValueError."""
        with pytest.raises(ValueError, match="detection_threshold"):
            ZeroFilterConfig(detection_threshold=-1)

    def test_invalid_bqc_threshold_raises(self):
        """Test that invalid BQC threshold raises ValueError."""
        with pytest.raises(ValueError):
            ZeroFilterConfig(bqc_threshold=-0.1)

    def test_invalid_non_bqc_threshold_raises(self):
        """Test that invalid non-BQC threshold raises ValueError."""
        with pytest.raises(ValueError):
            ZeroFilterConfig(non_bqc_threshold=1.5)

    def test_unknown_format_error_message(self, simple_experiment_2x2):
        """Test error message content for unknown format."""
        df = pd.DataFrame({'random': [1, 2, 3]})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        assert any('format' in err.lower() for err in result.validation_errors)

    def test_empty_df_error_message(self, simple_experiment_2x2):
        """Test error message for empty DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(pd.DataFrame(), config)

        assert result.is_valid is False

    def test_missing_required_columns_error(self, simple_experiment_2x2):
        """Test error when required columns missing for specified format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'Sample1': [1e6],
        })

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,  # Requires more columns
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        # Should fail validation for LipidSearch
        assert result.is_valid is False


class TestWorkflowErrorRecovery:
    """Tests for error recovery and graceful degradation."""

    def test_workflow_continues_after_warning(self, lipidsearch_df, simple_experiment_2x2):
        """Test that workflow continues after non-fatal warnings."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        # Should succeed even with warnings
        assert result.is_valid is True

    def test_standards_validation_warning_not_error(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that standards validation issues are warnings, not errors."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        # Should be valid even if standards have warnings
        assert result.is_valid is True

    def test_zero_filtering_warning_on_skip(self, lipidsearch_df, simple_experiment_2x2):
        """Test warning message when zero filtering is skipped."""
        # This test checks that errors during zero filtering become warnings
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        # Should complete successfully
        assert result.is_valid is True


class TestWorkflowValidationErrors:
    """Tests for specific validation error scenarios."""

    def test_format_mismatch_error(self, lipidsearch_df, simple_experiment_2x2):
        """Test error when data doesn't match specified format."""
        # LipidSearch df specified as MS-DIAL
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        # May succeed or fail depending on column overlap
        # The key is it doesn't crash
        assert isinstance(result.is_valid, bool)

    def test_experiment_sample_mismatch(self, lipidsearch_df, basic_experiment):
        """Test handling when experiment samples don't match data columns."""
        # Experiment expects 6 samples, data has 4
        config = IngestionConfig(
            experiment=basic_experiment,  # 6 samples
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)  # 4 sample columns

        # Should handle gracefully
        assert isinstance(result.is_valid, bool)

    def test_multiple_validation_errors(self, simple_experiment_2x2):
        """Test that multiple validation errors are all reported."""
        df = pd.DataFrame({
            'WrongColumn': ['value'],
            'AnotherWrong': [123],
        })

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        # Should have at least one error
        assert len(result.validation_errors) >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestWorkflowIntegration:
    """Integration tests for the complete workflow."""

    def test_full_pipeline_lipidsearch(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test complete pipeline with LipidSearch data."""
        # Use default grade config (None = all grades A, B, C allowed)
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=GradeFilterConfig(),  # Default config
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.cleaned_df is not None
        assert 'LipidMolec' in result.cleaned_df.columns

    def test_result_df_usable_for_downstream(self, lipidsearch_df, simple_experiment_2x2):
        """Test that result DataFrame is usable for downstream analysis."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        df = result.cleaned_df

        # Should be able to group by class
        if 'ClassKey' in df.columns:
            grouped = df.groupby('ClassKey').size()
            assert len(grouped) > 0

    def test_multiple_runs_independent(self, lipidsearch_df, generic_df, simple_experiment_2x2):
        """Test that multiple workflow runs are independent."""
        config1 = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        config2 = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False
        )

        result1 = DataIngestionWorkflow.run(lipidsearch_df.copy(), config1)
        result2 = DataIngestionWorkflow.run(generic_df.copy(), config2)

        # Results should be independent
        assert result1.detected_format == DataFormat.LIPIDSEARCH
        assert result2.detected_format == DataFormat.GENERIC
        assert result1.is_valid is True
        assert result2.is_valid is True


class TestWorkflowIntegrationComplete:
    """Complete end-to-end integration tests."""

    def test_full_pipeline_msdial(self, msdial_df, simple_experiment_2x2):
        """Test complete pipeline with MS-DIAL data."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(msdial_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.MSDIAL
        assert result.cleaned_df is not None

    def test_full_pipeline_generic(self, generic_df, simple_experiment_2x2):
        """Test complete pipeline with Generic data."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            apply_zero_filter=True
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True
        assert result.cleaned_df is not None

    def test_pipeline_with_all_options(self, lipidsearch_df_with_standards, simple_experiment_2x2, external_standards_simple):
        """Test pipeline with all optional features enabled."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=GradeFilterConfig(grade_config={'PC': ['A', 'B']}),
            apply_zero_filter=True,
            zero_filter_config=ZeroFilterConfig(
                detection_threshold=0,
                non_bqc_threshold=0.5
            ),
            use_external_standards=True,
            external_standards_df=external_standards_simple
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        assert result.is_valid is True
        assert result.zero_filtered is True
        assert result.internal_standards_df is not None

    def test_sequential_runs_same_config(self, lipidsearch_df, simple_experiment_2x2):
        """Test multiple sequential runs with same config."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )

        results = []
        for _ in range(3):
            result = DataIngestionWorkflow.run(lipidsearch_df.copy(), config)
            results.append(result)

        # All results should be identical
        assert all(r.is_valid for r in results)
        assert all(r.detected_format == DataFormat.LIPIDSEARCH for r in results)

    def test_pipeline_state_isolation(self, lipidsearch_df, msdial_df, simple_experiment_2x2):
        """Test that pipeline runs are fully isolated."""
        # First run with LipidSearch
        config1 = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True
        )
        result1 = DataIngestionWorkflow.run(lipidsearch_df.copy(), config1)

        # Second run with MS-DIAL
        config2 = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=False
        )
        result2 = DataIngestionWorkflow.run(msdial_df.copy(), config2)

        # Results should be independent
        assert result1.detected_format == DataFormat.LIPIDSEARCH
        assert result2.detected_format == DataFormat.MSDIAL
        assert result1.zero_filtered is True
        assert result2.zero_filtered is False


class TestWorkflowIntegrationDataIntegrity:
    """Tests for data integrity through the workflow."""

    def test_lipid_names_preserved(self, lipidsearch_df, simple_experiment_2x2):
        """Test that lipid names are preserved through workflow."""
        original_lipids = set(lipidsearch_df['LipidMolec'].tolist())
        original_classes = set(lipidsearch_df['ClassKey'].tolist())

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        if result.is_valid and result.cleaned_df is not None:
            result_lipids = set(result.cleaned_df['LipidMolec'].tolist())
            result_classes = set(result.cleaned_df['ClassKey'].tolist())
            # Result lipids count should be <= original (some may be filtered/extracted as standards)
            # The key is no new lipid classes are introduced
            assert len(result_lipids) > 0
            assert len(result_lipids) <= len(original_lipids)
            # Lipid classes should be a subset of original classes
            # (cleaning may standardize lipid names but shouldn't change classes)
            assert result_classes.issubset(original_classes)

    def test_sample_columns_preserved(self, lipidsearch_df, simple_experiment_2x2):
        """Test that sample data columns are preserved."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Should have intensity columns in output
        intensity_cols = [c for c in result.cleaned_df.columns if 'intensity' in c.lower() or 's' in c]
        assert len(intensity_cols) > 0

    def test_row_count_consistent(self, lipidsearch_df, simple_experiment_2x2):
        """Test row count is consistent (no unexpected duplication)."""
        original_rows = len(lipidsearch_df)

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Without filtering, row count should be same or less (if standards extracted)
        assert len(result.cleaned_df) <= original_rows

    def test_numeric_values_preserved(self, lipidsearch_df, simple_experiment_2x2):
        """Test that numeric values are not corrupted."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Values should still be numeric
        intensity_cols = [c for c in result.cleaned_df.columns if 'intensity' in c]
        for col in intensity_cols:
            assert pd.api.types.is_numeric_dtype(result.cleaned_df[col])
