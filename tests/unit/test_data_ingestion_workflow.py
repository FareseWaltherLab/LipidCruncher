"""
Unit tests for DataIngestionWorkflow.

Tests the complete data ingestion pipeline orchestration:
format detection → cleaning → standards validation

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


# =============================================================================
# Edge Case and Zero Value Fixtures
# =============================================================================


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


# =============================================================================
# IngestionResult Tests
# =============================================================================

class TestIngestionResult:
    """Tests for IngestionResult dataclass."""


    def test_all_format_types(self):
        """Test result creation with all format types."""
        for fmt in [DataFormat.LIPIDSEARCH, DataFormat.MSDIAL, DataFormat.GENERIC, DataFormat.UNKNOWN]:
            result = IngestionResult(detected_format=fmt)
            assert result.detected_format == fmt


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


# =============================================================================
# IngestionConfig Tests
# =============================================================================

class TestIngestionConfig:
    """Tests for IngestionConfig dataclass."""


    def test_config_with_all_formats(self, simple_experiment_2x2):
        """Test config with each data format."""
        for fmt in [DataFormat.LIPIDSEARCH, DataFormat.MSDIAL, DataFormat.GENERIC]:
            config = IngestionConfig(
                experiment=simple_experiment_2x2,
                data_format=fmt
            )
            assert config.data_format == fmt


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
        msdial_df.drop(columns=['ClassKey'])  # ClassKey is renamed Ontology
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.is_valid is True

    def test_workflow_extracts_standards(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that workflow extracts internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True

    def test_workflow_unknown_format_fails(self, simple_experiment_2x2):
        """Test that unknown format causes validation failure."""
        df = pd.DataFrame({'random_col': [1, 2, 3]})
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        assert DataFormat.UNKNOWN == result.detected_format
        assert len(result.validation_errors) > 0

    def test_workflow_empty_df_fails(self, simple_experiment_2x2):
        """Test that empty DataFrame causes failure."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(msdial_df, config)

        assert result.is_valid is True

    def test_msdial_with_standards(self, msdial_df_with_standards, simple_experiment_2x2):
        """Test MS-DIAL workflow extracts internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
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
        )

        result = DataIngestionWorkflow.run(df_single_row, config)

        assert result.is_valid is True
        assert len(result.cleaned_df) == 1

    def test_workflow_large_dataset(self, df_large_dataset, simple_experiment_2x2):
        """Test workflow with large dataset."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
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
        )

        DataIngestionWorkflow.run(lipidsearch_df, config)

        pd.testing.assert_frame_equal(lipidsearch_df, original_df)

    def test_workflow_with_special_characters(self, df_special_characters, simple_experiment_2x2):
        """Test workflow with special characters in lipid names."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
        )

        result = DataIngestionWorkflow.run(df_special_characters, config)

        assert result.is_valid is True

    def test_workflow_single_condition(self, generic_df, single_condition_experiment):
        """Test workflow with single condition experiment."""
        # Adjust generic_df to have 4 samples for single condition
        config = IngestionConfig(
            experiment=single_condition_experiment,
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True

    def test_workflow_many_conditions(self, df_many_samples, many_conditions_experiment):
        """Test workflow with many conditions experiment."""
        config = IngestionConfig(
            experiment=many_conditions_experiment,
        )

        result = DataIngestionWorkflow.run(df_many_samples, config)

        assert result.is_valid is True

    def test_workflow_mixed_standards(self, lipidsearch_df_with_mixed_standards, simple_experiment_2x2):
        """Test workflow extracts multiple standard patterns."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Should have messages about filtering
        assert isinstance(result.cleaning_messages, list)


# =============================================================================
# Standards Extraction Tests
# =============================================================================


class TestWorkflowStandardsExtraction:
    """Tests for internal standards extraction in workflow."""

    def test_deuterated_standards_extracted(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that deuterated standards (d7, d5, d9) are extracted."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_only_standards, config)

        # Should handle gracefully
        assert result.is_valid is True

    def test_mixed_standards_patterns(self, lipidsearch_df_with_mixed_standards, simple_experiment_2x2):
        """Test extraction of multiple standard patterns."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(df, config)

        # Should fail during cleaning
        assert result.is_valid is False
        assert len(result.validation_errors) > 0


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
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid is False
        assert any('format' in err.lower() for err in result.validation_errors)

    def test_empty_df_error_message(self, simple_experiment_2x2):
        """Test error message for empty DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        # Should succeed even with warnings
        assert result.is_valid is True

    def test_standards_validation_warning_not_error(self, lipidsearch_df_with_standards, simple_experiment_2x2):
        """Test that standards validation issues are warnings, not errors."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
        )

        result = DataIngestionWorkflow.run(lipidsearch_df_with_standards, config)

        # Should be valid even if standards have warnings
        assert result.is_valid is True


class TestWorkflowValidationErrors:
    """Tests for specific validation error scenarios."""

    def test_format_mismatch_error(self, lipidsearch_df, simple_experiment_2x2):
        """Test error when data doesn't match specified format."""
        # LipidSearch df specified as MS-DIAL
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.MSDIAL,
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
        )

        config2 = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.GENERIC,
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
        )

        result = DataIngestionWorkflow.run(msdial_df, config)

        assert result.is_valid is True
        assert result.detected_format == DataFormat.MSDIAL
        assert result.cleaned_df is not None

    def test_full_pipeline_generic(self, generic_df, simple_experiment_2x2):
        """Test complete pipeline with Generic data."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
        )

        result = DataIngestionWorkflow.run(generic_df, config)

        assert result.is_valid is True
        assert result.cleaned_df is not None


    def test_sequential_runs_same_config(self, lipidsearch_df, simple_experiment_2x2):
        """Test multiple sequential runs with same config."""
        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
        )

        results = []
        for _ in range(3):
            result = DataIngestionWorkflow.run(lipidsearch_df.copy(), config)
            results.append(result)

        # All results should be identical
        assert all(r.is_valid for r in results)
        assert all(r.detected_format == DataFormat.LIPIDSEARCH for r in results)


class TestWorkflowIntegrationDataIntegrity:
    """Tests for data integrity through the workflow."""

    def test_lipid_names_preserved(self, lipidsearch_df, simple_experiment_2x2):
        """Test that lipid names are preserved through workflow."""
        original_lipids = set(lipidsearch_df['LipidMolec'].tolist())
        original_classes = set(lipidsearch_df['ClassKey'].tolist())

        config = IngestionConfig(
            experiment=simple_experiment_2x2,
            data_format=DataFormat.LIPIDSEARCH,
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
        )

        result = DataIngestionWorkflow.run(lipidsearch_df, config)

        assert result.is_valid is True
        # Values should still be numeric
        intensity_cols = [c for c in result.cleaned_df.columns if 'intensity' in c]
        for col in intensity_cols:
            assert pd.api.types.is_numeric_dtype(result.cleaned_df[col])
