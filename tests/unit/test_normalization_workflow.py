"""
Unit tests for NormalizationWorkflow.

Tests the complete normalization pipeline orchestration:
class selection → method configuration → normalization → column restoration

Comprehensive test coverage matching Phase 3/4 service test depth.
"""
import pytest
import pandas as pd
import numpy as np

from app.workflows.normalization import (
    NormalizationWorkflow,
    NormalizationWorkflowConfig,
    NormalizationWorkflowResult
)
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.services.format_detection import DataFormat
from tests.conftest import make_experiment


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def single_condition_experiment():
    """Single condition experiment."""
    return make_experiment(
        n_conditions=1,
        conditions_list=['Samples'],
        number_of_samples_list=[4],
    )


@pytest.fixture
def unequal_samples_experiment():
    """Experiment with unequal sample counts."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Small', 'Medium', 'Large'],
        number_of_samples_list=[1, 3, 5]
    )


@pytest.fixture
def large_experiment():
    """Large experiment for stress testing."""
    return ExperimentConfig(
        n_conditions=5,
        conditions_list=['C1', 'C2', 'C3', 'C4', 'C5'],
        number_of_samples_list=[4, 4, 4, 4, 4]
    )


# =============================================================================
# Basic Data Fixtures
# =============================================================================

@pytest.fixture
def basic_df(simple_experiment_2x2):
    """Basic DataFrame with intensity columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def lipidsearch_df(simple_experiment_2x2):
    """LipidSearch format DataFrame with essential columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'CalcMass': [760.5, 768.5, 850.7, 703.6],
        'BaseRt': [10.5, 12.3, 15.0, 8.2],
        'intensity[s1]': [1e6, 2e6, 3e6, 1.5e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6, 1.6e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6, 1.7e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6, 1.8e6],
    })


@pytest.fixture
def single_class_df():
    """DataFrame with single lipid class."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(18:0_20:4)', 'PC(16:0_16:0)'],
        'ClassKey': ['PC', 'PC', 'PC'],
        'intensity[s1]': [1e6, 2e6, 3e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def many_classes_df():
    """DataFrame with many lipid classes."""
    classes = ['PC', 'PE', 'TG', 'DG', 'SM', 'Cer', 'PA', 'PG', 'PI', 'PS']
    return pd.DataFrame({
        'LipidMolec': [f'{cls}(16:0_18:1)' for cls in classes],
        'ClassKey': classes,
        'intensity[s1]': [1e6 + i*1e5 for i in range(len(classes))],
        'intensity[s2]': [1.1e6 + i*1e5 for i in range(len(classes))],
        'intensity[s3]': [1.2e6 + i*1e5 for i in range(len(classes))],
        'intensity[s4]': [1.3e6 + i*1e5 for i in range(len(classes))],
    })


# =============================================================================
# Internal Standards Fixtures
# =============================================================================

@pytest.fixture
def basic_intsta_df():
    """Basic internal standards DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)', 'TG(15:0_15:0_15:0-d9)', 'SM(d18:1_12:0-d7)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [5e5, 5e5, 5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5, 5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5, 5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5, 5.3e5, 5.3e5],
    })


@pytest.fixture
def single_standard_df():
    """Single internal standard DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)'],
        'ClassKey': ['PC'],
        'intensity[s1]': [5e5],
        'intensity[s2]': [5.1e5],
        'intensity[s3]': [5.2e5],
        'intensity[s4]': [5.3e5],
    })


@pytest.fixture
def multiple_standards_per_class_df():
    """Multiple standards per class."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PC(17:0_17:0-d5)', 'PE(17:0_20:4-d7)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'intensity[s1]': [5e5, 4e5, 5e5],
        'intensity[s2]': [5.1e5, 4.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 4.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 4.3e5, 5.3e5],
    })


@pytest.fixture
def intsta_with_zeros_df():
    """Internal standards with zero values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [5e5, 0],  # Zero for PE in s1
        'intensity[s2]': [5.1e5, 5.1e5],
        'intensity[s3]': [0, 5.2e5],  # Zero for PC in s3
        'intensity[s4]': [5.3e5, 5.3e5],
    })


# =============================================================================
# Edge Case Data Fixtures
# =============================================================================

@pytest.fixture
def empty_df():
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]', 'intensity[s2]'])


@pytest.fixture
def single_row_df():
    """Single row DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)'],
        'ClassKey': ['PC'],
        'intensity[s1]': [1e6],
        'intensity[s2]': [1.1e6],
        'intensity[s3]': [1.2e6],
        'intensity[s4]': [1.3e6],
    })


@pytest.fixture
def df_with_nan():
    """DataFrame with NaN values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1e6, np.nan, 3e6],
        'intensity[s2]': [np.nan, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, np.nan],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def df_with_zeros():
    """DataFrame with zero values."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [0, 2e6, 3e6],
        'intensity[s2]': [1.1e6, 0, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, 0],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def df_missing_classkey():
    """DataFrame missing ClassKey column."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'intensity[s1]': [1e6, 2e6],
        'intensity[s2]': [1.1e6, 2.1e6],
    })


@pytest.fixture
def df_missing_lipidmolec():
    """DataFrame missing LipidMolec column."""
    return pd.DataFrame({
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [1e6, 2e6],
        'intensity[s2]': [1.1e6, 2.1e6],
    })


@pytest.fixture
def df_no_intensity_columns():
    """DataFrame without intensity columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PE'],
        'SomeOtherColumn': [1e6, 2e6],
    })


@pytest.fixture
def df_with_special_chars():
    """DataFrame with special characters in names."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0/18:1)', 'PE[18:0_20:4]', 'TG{16:0_18:1_18:2}'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1e6, 2e6, 3e6],
        'intensity[s2]': [1.1e6, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, 3.2e6],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


# =============================================================================
# Protein Concentration Fixtures
# =============================================================================

@pytest.fixture
def protein_concentrations():
    """Basic protein concentrations."""
    return {'s1': 10.0, 's2': 12.0, 's3': 11.0, 's4': 13.0}


@pytest.fixture
def protein_concentrations_unequal():
    """Varying protein concentrations."""
    return {'s1': 5.0, 's2': 50.0, 's3': 15.0, 's4': 25.0}


@pytest.fixture
def protein_concentrations_partial():
    """Partial protein concentrations (missing some samples)."""
    return {'s1': 10.0, 's2': 12.0}  # Missing s3, s4


# =============================================================================
# Normalization Config Fixtures
# =============================================================================

@pytest.fixture
def none_config():
    """No normalization config."""
    return NormalizationConfig(method='none')


@pytest.fixture
def none_config_filtered():
    """No normalization with class filter."""
    return NormalizationConfig(method='none', selected_classes=['PC', 'PE'])


@pytest.fixture
def protein_config(protein_concentrations):
    """Protein normalization config."""
    return NormalizationConfig(
        method='protein',
        protein_concentrations=protein_concentrations
    )


@pytest.fixture
def protein_config_filtered(protein_concentrations):
    """Protein normalization with class filter."""
    return NormalizationConfig(
        method='protein',
        selected_classes=['PC', 'PE'],
        protein_concentrations=protein_concentrations
    )


@pytest.fixture
def internal_standard_config():
    """Internal standard normalization config."""
    return NormalizationConfig(
        method='internal_standard',
        selected_classes=['PC', 'PE', 'TG', 'SM'],
        internal_standards={
            'PC': 'PC(15:0_18:1-d7)',
            'PE': 'PE(17:0_20:4-d7)',
            'TG': 'TG(15:0_15:0_15:0-d9)',
            'SM': 'SM(d18:1_12:0-d7)'
        },
        intsta_concentrations={
            'PC(15:0_18:1-d7)': 1.0,
            'PE(17:0_20:4-d7)': 1.0,
            'TG(15:0_15:0_15:0-d9)': 1.0,
            'SM(d18:1_12:0-d7)': 1.0
        }
    )


@pytest.fixture
def internal_standard_config_filtered():
    """Internal standard normalization with class filter."""
    return NormalizationConfig(
        method='internal_standard',
        selected_classes=['PC', 'PE'],
        internal_standards={
            'PC': 'PC(15:0_18:1-d7)',
            'PE': 'PE(17:0_20:4-d7)'
        },
        intsta_concentrations={
            'PC(15:0_18:1-d7)': 1.0,
            'PE(17:0_20:4-d7)': 1.0
        }
    )


@pytest.fixture
def both_config(protein_concentrations):
    """Combined normalization config."""
    return NormalizationConfig(
        method='both',
        selected_classes=['PC', 'PE', 'TG', 'SM'],
        internal_standards={
            'PC': 'PC(15:0_18:1-d7)',
            'PE': 'PE(17:0_20:4-d7)',
            'TG': 'TG(15:0_15:0_15:0-d9)',
            'SM': 'SM(d18:1_12:0-d7)'
        },
        intsta_concentrations={
            'PC(15:0_18:1-d7)': 1.0,
            'PE(17:0_20:4-d7)': 1.0,
            'TG(15:0_15:0_15:0-d9)': 1.0,
            'SM(d18:1_12:0-d7)': 1.0
        },
        protein_concentrations=protein_concentrations
    )


# =============================================================================
# Workflow Config Fixtures
# =============================================================================

@pytest.fixture
def basic_workflow_config(simple_experiment_2x2, none_config):
    """Basic workflow config with no normalization."""
    return NormalizationWorkflowConfig(
        experiment=simple_experiment_2x2,
        normalization=none_config,
        data_format=DataFormat.GENERIC
    )


@pytest.fixture
def lipidsearch_workflow_config(simple_experiment_2x2, none_config):
    """LipidSearch workflow config."""
    return NormalizationWorkflowConfig(
        experiment=simple_experiment_2x2,
        normalization=none_config,
        data_format=DataFormat.LIPIDSEARCH
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestNormalizationWorkflowResult:
    """Tests for NormalizationWorkflowResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = NormalizationWorkflowResult()
        assert result.normalized_df is None
        assert result.success is True
        assert result.method_applied == "None"
        assert result.removed_standards == []
        assert result.classes_in_input == []
        assert result.classes_normalized == []
        assert result.validation_errors == []
        assert result.validation_warnings == []
        assert result.lipids_before == 0
        assert result.lipids_after == 0
        assert result.samples_processed == 0

    def test_lipids_removed_count(self):
        """Test lipids removed count property."""
        result = NormalizationWorkflowResult(lipids_before=10, lipids_after=8)
        assert result.lipids_removed_count == 2

    def test_lipids_removed_count_zero(self):
        """Test lipids removed when none removed."""
        result = NormalizationWorkflowResult(lipids_before=10, lipids_after=10)
        assert result.lipids_removed_count == 0

    def test_has_warnings_true(self):
        """Test has_warnings when warnings exist."""
        result = NormalizationWorkflowResult(validation_warnings=["Warning 1"])
        assert result.has_warnings is True

    def test_has_warnings_false(self):
        """Test has_warnings when no warnings."""
        result = NormalizationWorkflowResult()
        assert result.has_warnings is False


class TestNormalizationWorkflowConfig:
    """Tests for NormalizationWorkflowConfig dataclass."""

    def test_basic_creation(self, simple_experiment_2x2, none_config):
        """Test basic config creation."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        assert config.experiment == simple_experiment_2x2
        assert config.normalization == none_config
        assert config.data_format == DataFormat.GENERIC
        assert config.essential_columns is None

    def test_with_data_format(self, simple_experiment_2x2, none_config):
        """Test config with specific data format."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            data_format=DataFormat.LIPIDSEARCH
        )
        assert config.data_format == DataFormat.LIPIDSEARCH

    def test_with_essential_columns(self, simple_experiment_2x2, none_config):
        """Test config with essential columns."""
        essential = {'CalcMass': pd.Series([760.5, 768.5])}
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            essential_columns=essential
        )
        assert config.essential_columns is not None
        assert 'CalcMass' in config.essential_columns


class TestValidateConfig:
    """Tests for validate_config method."""

    def test_valid_basic_config(self, basic_df, simple_experiment_2x2, none_config):
        """Test validation of valid basic config."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(basic_df, config)
        assert errors == []

    def test_empty_dataframe(self, empty_df, simple_experiment_2x2, none_config):
        """Test validation with empty DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(empty_df, config)
        assert len(errors) > 0
        assert any('empty' in e.lower() for e in errors)

    def test_none_dataframe(self, simple_experiment_2x2, none_config):
        """Test validation with None DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(None, config)
        assert len(errors) > 0
        assert any('empty' in e.lower() for e in errors)

    def test_missing_lipidmolec(self, df_missing_lipidmolec, simple_experiment_2x2, none_config):
        """Test validation with missing LipidMolec column."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(df_missing_lipidmolec, config)
        assert any('LipidMolec' in e for e in errors)

    def test_missing_classkey(self, df_missing_classkey, simple_experiment_2x2, none_config):
        """Test validation with missing ClassKey column."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(df_missing_classkey, config)
        assert any('ClassKey' in e for e in errors)

    def test_no_intensity_columns(self, df_no_intensity_columns, simple_experiment_2x2, none_config):
        """Test validation with no intensity columns."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        errors = NormalizationWorkflow.validate_config(df_no_intensity_columns, config)
        assert any('intensity' in e.lower() for e in errors)

    def test_invalid_selected_classes(self, basic_df, simple_experiment_2x2):
        """Test validation with invalid selected classes."""
        norm_config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'INVALID_CLASS']
        )
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )
        errors = NormalizationWorkflow.validate_config(basic_df, config)
        assert any('INVALID_CLASS' in e for e in errors)

    def test_internal_standard_without_intsta_df(
        self, basic_df, simple_experiment_2x2, internal_standard_config
    ):
        """Test validation for IS normalization without standards DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        errors = NormalizationWorkflow.validate_config(basic_df, config, intsta_df=None)
        assert any('standards' in e.lower() for e in errors)

    def test_protein_without_concentrations(self, basic_df, simple_experiment_2x2):
        """Test validation for protein normalization without concentrations."""
        # This should raise during NormalizationConfig creation
        with pytest.raises(ValueError):
            NormalizationConfig(method='protein')

    def test_valid_internal_standard_config(
        self, basic_df, simple_experiment_2x2, internal_standard_config, basic_intsta_df
    ):
        """Test validation with valid IS config."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        errors = NormalizationWorkflow.validate_config(basic_df, config, basic_intsta_df)
        assert errors == []


class TestGetAvailableClasses:
    """Tests for get_available_classes method."""

    def test_basic_classes(self, basic_df):
        """Test getting classes from basic DataFrame."""
        classes = NormalizationWorkflow.get_available_classes(basic_df)
        assert set(classes) == {'PC', 'PE', 'TG', 'SM'}

    def test_sorted_output(self, basic_df):
        """Test that output is sorted."""
        classes = NormalizationWorkflow.get_available_classes(basic_df)
        assert classes == sorted(classes)

    def test_single_class(self, single_class_df):
        """Test with single class."""
        classes = NormalizationWorkflow.get_available_classes(single_class_df)
        assert classes == ['PC']

    def test_many_classes(self, many_classes_df):
        """Test with many classes."""
        classes = NormalizationWorkflow.get_available_classes(many_classes_df)
        assert len(classes) == 10

    def test_empty_dataframe(self, empty_df):
        """Test with empty DataFrame."""
        classes = NormalizationWorkflow.get_available_classes(empty_df)
        assert classes == []

    def test_none_dataframe(self):
        """Test with None DataFrame."""
        classes = NormalizationWorkflow.get_available_classes(None)
        assert classes == []

    def test_missing_classkey(self, df_missing_classkey):
        """Test with missing ClassKey column."""
        classes = NormalizationWorkflow.get_available_classes(df_missing_classkey)
        assert classes == []


class TestGetAvailableStandards:
    """Tests for get_available_standards method."""

    def test_basic_standards(self, basic_intsta_df):
        """Test getting standards from basic DataFrame."""
        standards = NormalizationWorkflow.get_available_standards(basic_intsta_df)
        assert len(standards) == 4
        assert 'PC(15:0_18:1-d7)' in standards

    def test_single_standard(self, single_standard_df):
        """Test with single standard."""
        standards = NormalizationWorkflow.get_available_standards(single_standard_df)
        assert standards == ['PC(15:0_18:1-d7)']

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        standards = NormalizationWorkflow.get_available_standards(pd.DataFrame())
        assert standards == []

    def test_none_dataframe(self):
        """Test with None DataFrame."""
        standards = NormalizationWorkflow.get_available_standards(None)
        assert standards == []


class TestGetStandardsByClass:
    """Tests for get_standards_by_class method."""

    def test_basic_grouping(self, basic_intsta_df):
        """Test basic class grouping."""
        by_class = NormalizationWorkflow.get_standards_by_class(basic_intsta_df)
        assert 'PC' in by_class
        assert 'PE' in by_class
        assert by_class['PC'] == ['PC(15:0_18:1-d7)']

    def test_multiple_standards_per_class(self, multiple_standards_per_class_df):
        """Test with multiple standards per class."""
        by_class = NormalizationWorkflow.get_standards_by_class(multiple_standards_per_class_df)
        assert len(by_class['PC']) == 2
        assert len(by_class['PE']) == 1

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        by_class = NormalizationWorkflow.get_standards_by_class(pd.DataFrame())
        assert by_class == {}

    def test_none_dataframe(self):
        """Test with None DataFrame."""
        by_class = NormalizationWorkflow.get_standards_by_class(None)
        assert by_class == {}


class TestSuggestStandardMappings:
    """Tests for suggest_standard_mappings method."""

    def test_class_specific_suggestions(self, basic_intsta_df):
        """Test suggestions use class-specific standards."""
        mappings = NormalizationWorkflow.suggest_standard_mappings(
            ['PC', 'PE'], basic_intsta_df
        )
        assert mappings['PC'] == 'PC(15:0_18:1-d7)'
        assert mappings['PE'] == 'PE(17:0_20:4-d7)'

    def test_fallback_to_first_available(self, single_standard_df):
        """Test fallback when no class-specific standard."""
        mappings = NormalizationWorkflow.suggest_standard_mappings(
            ['PE', 'TG'], single_standard_df  # Only PC standard available
        )
        # Both should fall back to the only available standard
        assert mappings['PE'] == 'PC(15:0_18:1-d7)'
        assert mappings['TG'] == 'PC(15:0_18:1-d7)'

    def test_no_standards_available(self):
        """Test when no standards available."""
        mappings = NormalizationWorkflow.suggest_standard_mappings(
            ['PC', 'PE'], pd.DataFrame()
        )
        assert mappings['PC'] is None
        assert mappings['PE'] is None

    def test_none_intsta_df(self):
        """Test with None intsta_df."""
        mappings = NormalizationWorkflow.suggest_standard_mappings(
            ['PC', 'PE'], None
        )
        assert mappings['PC'] is None
        assert mappings['PE'] is None


class TestValidateStandardMappings:
    """Tests for validate_standard_mappings method."""

    def test_valid_mappings(self, basic_intsta_df):
        """Test validation of valid mappings."""
        mappings = {'PC': 'PC(15:0_18:1-d7)', 'PE': 'PE(17:0_20:4-d7)'}
        is_valid, errors = NormalizationWorkflow.validate_standard_mappings(
            mappings, basic_intsta_df, ['PC', 'PE']
        )
        assert is_valid is True
        assert errors == []

    def test_missing_class_mapping(self, basic_intsta_df):
        """Test validation with missing class mapping."""
        mappings = {'PC': 'PC(15:0_18:1-d7)'}  # Missing PE
        is_valid, errors = NormalizationWorkflow.validate_standard_mappings(
            mappings, basic_intsta_df, ['PC', 'PE']
        )
        assert is_valid is False
        assert any('PE' in e for e in errors)

    def test_invalid_standard_name(self, basic_intsta_df):
        """Test validation with invalid standard name."""
        mappings = {'PC': 'INVALID_STANDARD', 'PE': 'PE(17:0_20:4-d7)'}
        is_valid, errors = NormalizationWorkflow.validate_standard_mappings(
            mappings, basic_intsta_df, ['PC', 'PE']
        )
        assert is_valid is False
        assert any('INVALID_STANDARD' in e for e in errors)

    def test_none_intsta_df(self):
        """Test validation with None intsta_df."""
        mappings = {'PC': 'PC(15:0_18:1-d7)'}
        is_valid, errors = NormalizationWorkflow.validate_standard_mappings(
            mappings, None, ['PC']
        )
        assert is_valid is False


class TestCreateNormalizationConfig:
    """Tests for create_normalization_config helper."""

    def test_create_none_config(self):
        """Test creating none config."""
        config = NormalizationWorkflow.create_normalization_config(
            method='none',
            selected_classes=['PC', 'PE']
        )
        assert config.method == 'none'
        assert config.selected_classes == ['PC', 'PE']

    def test_create_protein_config(self, protein_concentrations):
        """Test creating protein config."""
        config = NormalizationWorkflow.create_normalization_config(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations=protein_concentrations
        )
        assert config.method == 'protein'
        assert config.protein_concentrations == protein_concentrations

    def test_create_is_config(self):
        """Test creating internal standard config."""
        config = NormalizationWorkflow.create_normalization_config(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC_d7'},
            intsta_concentrations={'PC_d7': 1.0}
        )
        assert config.method == 'internal_standard'
        assert config.internal_standards == {'PC': 'PC_d7'}

    def test_create_both_config(self, protein_concentrations):
        """Test creating combined config."""
        config = NormalizationWorkflow.create_normalization_config(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC_d7'},
            intsta_concentrations={'PC_d7': 1.0},
            protein_concentrations=protein_concentrations
        )
        assert config.method == 'both'

    def test_invalid_config_raises(self):
        """Test that invalid config raises ValueError."""
        with pytest.raises(ValueError):
            NormalizationWorkflow.create_normalization_config(
                method='internal_standard',
                selected_classes=['PC']
                # Missing internal_standards and intsta_concentrations
            )


class TestGetIntensityColumnSamples:
    """Tests for get_intensity_column_samples method."""

    def test_basic_extraction(self, basic_df):
        """Test basic sample extraction."""
        samples = NormalizationWorkflow.get_intensity_column_samples(basic_df)
        assert set(samples) == {'s1', 's2', 's3', 's4'}

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        samples = NormalizationWorkflow.get_intensity_column_samples(pd.DataFrame())
        assert samples == []

    def test_no_intensity_columns(self, df_no_intensity_columns):
        """Test with no intensity columns."""
        samples = NormalizationWorkflow.get_intensity_column_samples(df_no_intensity_columns)
        assert samples == []


class TestGetConcentrationColumnSamples:
    """Tests for get_concentration_column_samples method."""

    def test_basic_extraction(self):
        """Test basic sample extraction from concentration columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC'],
            'concentration[s1]': [1.0],
            'concentration[s2]': [2.0]
        })
        samples = NormalizationWorkflow.get_concentration_column_samples(df)
        assert set(samples) == {'s1', 's2'}

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        samples = NormalizationWorkflow.get_concentration_column_samples(pd.DataFrame())
        assert samples == []


class TestStoreAndRestoreEssentialColumns:
    """Tests for _store_essential_columns and _restore_essential_columns methods."""

    def test_store_lipidsearch_columns(self, lipidsearch_df):
        """Test storing LipidSearch essential columns."""
        stored = NormalizationWorkflow._store_essential_columns(
            lipidsearch_df, DataFormat.LIPIDSEARCH
        )
        assert 'CalcMass' in stored
        assert 'BaseRt' in stored
        assert len(stored['CalcMass']) == 4

    def test_store_generic_format_no_columns(self, basic_df):
        """Test no columns stored for generic format."""
        stored = NormalizationWorkflow._store_essential_columns(
            basic_df, DataFormat.GENERIC
        )
        assert stored == {}

    def test_store_msdial_format_no_columns(self, basic_df):
        """Test no columns stored for MS-DIAL format."""
        stored = NormalizationWorkflow._store_essential_columns(
            basic_df, DataFormat.MSDIAL
        )
        assert stored == {}

    def test_restore_essential_columns(self, lipidsearch_df):
        """Test restoring essential columns."""
        # Store columns
        stored = {'CalcMass': lipidsearch_df['CalcMass'].copy()}

        # Create a DataFrame without CalcMass
        df_without = lipidsearch_df.drop(columns=['CalcMass'])

        # Restore
        restored = NormalizationWorkflow._restore_essential_columns(
            df_without, stored
        )
        assert 'CalcMass' in restored.columns
        assert list(restored['CalcMass']) == list(lipidsearch_df['CalcMass'])

    def test_restore_empty_stored(self, basic_df):
        """Test restore with no stored columns."""
        restored = NormalizationWorkflow._restore_essential_columns(basic_df, {})
        pd.testing.assert_frame_equal(restored, basic_df)


class TestPreviewNormalization:
    """Tests for preview_normalization method."""

    def test_preview_none_method(self, basic_df, simple_experiment_2x2, none_config):
        """Test preview for none method."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        preview = NormalizationWorkflow.preview_normalization(basic_df, config)

        assert preview.method == 'none'
        assert preview.can_proceed is True
        assert set(preview.classes_to_process) == {'PC', 'PE', 'TG', 'SM'}
        assert preview.samples_to_normalize == 4

    def test_preview_with_class_filter(self, basic_df, simple_experiment_2x2):
        """Test preview with class filter."""
        norm_config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE']
        )
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )
        preview = NormalizationWorkflow.preview_normalization(basic_df, config)

        assert preview.classes_to_process == ['PC', 'PE']

    def test_preview_with_standards_removal(
        self, basic_df, simple_experiment_2x2, internal_standard_config, basic_intsta_df
    ):
        """Test preview shows standards to be removed."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        preview = NormalizationWorkflow.preview_normalization(
            basic_df, config, basic_intsta_df
        )

        assert len(preview.standards_to_remove) > 0

    def test_preview_with_validation_errors(self, empty_df, simple_experiment_2x2, none_config):
        """Test preview with validation errors."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        preview = NormalizationWorkflow.preview_normalization(empty_df, config)

        assert preview.can_proceed is False
        assert len(preview.validation_errors) > 0


class TestRunWorkflowNoneMethod:
    """Tests for run method with 'none' normalization."""

    def test_run_none_basic(self, basic_df, simple_experiment_2x2, none_config):
        """Test basic run with no normalization."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        assert result.normalized_df is not None
        assert result.method_applied == "None (raw data with concentration column naming)"
        assert len(result.normalized_df) == 4

    def test_run_none_renames_columns(self, basic_df, simple_experiment_2x2, none_config):
        """Test that intensity columns are renamed to concentration."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        concentration_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
        intensity_cols = [c for c in result.normalized_df.columns if c.startswith('intensity[')]

        assert len(concentration_cols) == 4
        assert len(intensity_cols) == 0

    def test_run_none_with_class_filter(self, basic_df, simple_experiment_2x2, none_config_filtered):
        """Test run with class filter."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config_filtered
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        # Should only have PC and PE
        assert set(result.normalized_df['ClassKey'].unique()) == {'PC', 'PE'}
        assert len(result.normalized_df) == 2

    def test_run_none_preserves_lipidsearch_columns(self, lipidsearch_df, simple_experiment_2x2, none_config):
        """Test that LipidSearch essential columns are preserved."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            data_format=DataFormat.LIPIDSEARCH
        )
        result = NormalizationWorkflow.run(lipidsearch_df, config)

        assert 'CalcMass' in result.normalized_df.columns
        assert 'BaseRt' in result.normalized_df.columns

    def test_run_none_single_row(self, single_row_df, simple_experiment_2x2, none_config):
        """Test run with single row DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(single_row_df, config)

        assert result.success is True
        assert len(result.normalized_df) == 1


class TestRunWorkflowProteinMethod:
    """Tests for run method with 'protein' normalization."""

    def test_run_protein_basic(self, basic_df, simple_experiment_2x2, protein_config):
        """Test basic protein normalization."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=protein_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        assert result.normalized_df is not None
        assert 'Protein' in result.method_applied

    def test_run_protein_divides_by_concentration(self, basic_df, simple_experiment_2x2):
        """Test that protein normalization divides by concentration."""
        protein_conc = {'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}
        norm_config = NormalizationConfig(
            method='protein',
            protein_concentrations=protein_conc
        )
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        # Original intensity was 1e6, divided by 10 = 1e5
        assert result.normalized_df['concentration[s1]'].iloc[0] == pytest.approx(1e5, rel=0.01)

    def test_run_protein_with_class_filter(self, basic_df, simple_experiment_2x2, protein_config_filtered):
        """Test protein normalization with class filter."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=protein_config_filtered
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        assert set(result.normalized_df['ClassKey'].unique()) == {'PC', 'PE'}


class TestRunWorkflowInternalStandardMethod:
    """Tests for run method with 'internal_standard' normalization."""

    def test_run_is_basic(self, basic_df, simple_experiment_2x2, internal_standard_config, basic_intsta_df):
        """Test basic internal standard normalization."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        result = NormalizationWorkflow.run(basic_df, config, basic_intsta_df)

        assert result.success is True
        assert result.normalized_df is not None
        assert 'Internal standards' in result.method_applied

    def test_run_is_removes_standards(self, basic_df, simple_experiment_2x2, internal_standard_config, basic_intsta_df):
        """Test that standards are removed from result."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        result = NormalizationWorkflow.run(basic_df, config, basic_intsta_df)

        assert len(result.removed_standards) > 0

    def test_run_is_without_intsta_df_fails(self, basic_df, simple_experiment_2x2, internal_standard_config):
        """Test that IS normalization fails without standards DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config
        )
        result = NormalizationWorkflow.run(basic_df, config, intsta_df=None)

        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_run_is_with_class_filter(
        self, basic_df, simple_experiment_2x2, internal_standard_config_filtered, basic_intsta_df
    ):
        """Test IS normalization with class filter."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=internal_standard_config_filtered
        )
        result = NormalizationWorkflow.run(basic_df, config, basic_intsta_df)

        assert result.success is True
        assert set(result.classes_normalized) == {'PC', 'PE'}


class TestRunWorkflowBothMethod:
    """Tests for run method with 'both' normalization."""

    def test_run_both_basic(self, basic_df, simple_experiment_2x2, both_config, basic_intsta_df):
        """Test combined normalization."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=both_config
        )
        result = NormalizationWorkflow.run(basic_df, config, basic_intsta_df)

        assert result.success is True
        assert result.normalized_df is not None
        assert 'Combined' in result.method_applied

    def test_run_both_removes_standards(self, basic_df, simple_experiment_2x2, both_config, basic_intsta_df):
        """Test that combined normalization removes standards."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=both_config
        )
        result = NormalizationWorkflow.run(basic_df, config, basic_intsta_df)

        assert len(result.removed_standards) > 0


class TestRunWorkflowTotalIntensity:
    """Tests for total intensity normalization through the workflow."""

    @pytest.fixture
    def total_intensity_config(self):
        return NormalizationConfig(method='total_intensity')

    def test_basic_total_intensity_workflow(self, basic_df, simple_experiment_2x2, total_intensity_config):
        """Total intensity normalization runs successfully through workflow."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        assert result.method_applied == "Total intensity normalization"
        assert result.normalized_df is not None

    def test_total_intensity_equalizes_totals(self, basic_df, simple_experiment_2x2, total_intensity_config):
        """Workflow output has equal sample totals."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        conc_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
        totals = result.normalized_df[conc_cols].sum(axis=0)
        assert totals.std() < 1e-6

    def test_total_intensity_renames_columns(self, basic_df, simple_experiment_2x2, total_intensity_config):
        """Workflow output has concentration columns, not intensity."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert not any(c.startswith('intensity[') for c in result.normalized_df.columns)
        assert any(c.startswith('concentration[') for c in result.normalized_df.columns)

    def test_total_intensity_no_standards_removed(self, basic_df, simple_experiment_2x2, total_intensity_config):
        """Total intensity normalization does not remove internal standards."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.removed_standards == []

    def test_total_intensity_preserves_essential_columns(self, simple_experiment_2x2, total_intensity_config):
        """LipidSearch essential columns (CalcMass, BaseRt) are preserved."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'intensity[s1]': [1e6, 2e6],
            'intensity[s2]': [1.1e6, 2.1e6],
            'intensity[s3]': [1.2e6, 2.2e6],
            'intensity[s4]': [1.3e6, 2.3e6],
        })
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config,
            data_format=DataFormat.LIPIDSEARCH,
        )
        result = NormalizationWorkflow.run(df, config)

        assert result.success is True
        assert 'CalcMass' in result.normalized_df.columns
        assert 'BaseRt' in result.normalized_df.columns

    def test_total_intensity_with_class_filter(self, basic_df, simple_experiment_2x2):
        """Class filtering works through the workflow."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=NormalizationConfig(
                method='total_intensity',
                selected_classes=['PC', 'PE'],
            )
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        assert 'TG' not in result.normalized_df['ClassKey'].values
        assert 'SM' not in result.normalized_df['ClassKey'].values

    def test_total_intensity_tracks_statistics(self, basic_df, simple_experiment_2x2, total_intensity_config):
        """Workflow tracks lipid counts and classes."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=total_intensity_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.lipids_before == 4
        assert result.lipids_after == 4
        assert result.samples_processed == 4


class TestRunWorkflowEdgeCases:
    """Tests for run method edge cases."""

    def test_run_empty_dataframe(self, empty_df, simple_experiment_2x2, none_config):
        """Test run with empty DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(empty_df, config)

        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_run_none_dataframe(self, simple_experiment_2x2, none_config):
        """Test run with None DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(None, config)

        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_run_df_with_nan(self, df_with_nan, simple_experiment_2x2, none_config):
        """Test run with DataFrame containing NaN values."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df_with_nan, config)

        assert result.success is True
        # NaN values should still be present
        assert result.normalized_df['concentration[s1]'].isna().any()

    def test_run_df_with_zeros(self, df_with_zeros, simple_experiment_2x2, none_config):
        """Test run with DataFrame containing zero values."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df_with_zeros, config)

        assert result.success is True
        assert (result.normalized_df['concentration[s1]'] == 0).any()

    def test_run_single_class(self, single_class_df, simple_experiment_2x2, none_config):
        """Test run with single class DataFrame."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(single_class_df, config)

        assert result.success is True
        assert result.classes_normalized == ['PC']

    def test_run_many_classes(self, many_classes_df, simple_experiment_2x2, none_config):
        """Test run with many classes."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(many_classes_df, config)

        assert result.success is True
        assert len(result.classes_normalized) == 10

    def test_run_with_special_chars(self, df_with_special_chars, simple_experiment_2x2, none_config):
        """Test run with special characters in names."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df_with_special_chars, config)

        assert result.success is True
        assert len(result.normalized_df) == 3


class TestRunWorkflowStatistics:
    """Tests for run method statistics tracking."""

    def test_tracks_lipids_before_after(self, basic_df, simple_experiment_2x2, none_config):
        """Test that lipid counts are tracked."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.lipids_before == 4
        assert result.lipids_after == 4

    def test_tracks_classes_in_input(self, basic_df, simple_experiment_2x2, none_config):
        """Test that input classes are tracked."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert set(result.classes_in_input) == {'PC', 'PE', 'TG', 'SM'}

    def test_tracks_classes_normalized(self, basic_df, simple_experiment_2x2, none_config_filtered):
        """Test that normalized classes are tracked."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config_filtered
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert set(result.classes_normalized) == {'PC', 'PE'}

    def test_tracks_samples_processed(self, basic_df, simple_experiment_2x2, none_config):
        """Test that sample count is tracked."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.samples_processed == 4


class TestRunWorkflowDataFormats:
    """Tests for run method with different data formats."""

    def test_lipidsearch_format_preserves_columns(self, lipidsearch_df, simple_experiment_2x2, none_config):
        """Test LipidSearch format preserves CalcMass and BaseRt."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            data_format=DataFormat.LIPIDSEARCH
        )
        result = NormalizationWorkflow.run(lipidsearch_df, config)

        assert 'CalcMass' in result.normalized_df.columns
        assert 'BaseRt' in result.normalized_df.columns

    def test_msdial_format(self, basic_df, simple_experiment_2x2, none_config):
        """Test MS-DIAL format."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            data_format=DataFormat.MSDIAL
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True

    def test_generic_format(self, basic_df, simple_experiment_2x2, none_config):
        """Test Generic format."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config,
            data_format=DataFormat.GENERIC
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True


class TestRunWorkflowIntegration:
    """Integration tests for complete workflow scenarios."""

    def test_full_is_normalization_workflow(self, simple_experiment_2x2, basic_intsta_df):
        """Test complete internal standard normalization workflow."""
        # Create data with standards present
        df = pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)', 'PC(15:0_18:1-d7)',
                'PE(18:0_20:4)', 'PE(17:0_20:4-d7)'
            ],
            'ClassKey': ['PC', 'PC', 'PE', 'PE'],
            'intensity[s1]': [1e6, 5e5, 2e6, 5e5],
            'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5],
            'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5],
            'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5],
        })

        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_18:1-d7)',
                'PE': 'PE(17:0_20:4-d7)'
            },
            intsta_concentrations={
                'PC(15:0_18:1-d7)': 1.0,
                'PE(17:0_20:4-d7)': 1.0
            }
        )

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )

        result = NormalizationWorkflow.run(df, config, basic_intsta_df)

        assert result.success is True
        # Standards should be removed from result
        assert 'PC(15:0_18:1-d7)' in result.removed_standards
        # Only non-standard lipids should remain
        assert 'PC(15:0_18:1-d7)' not in result.normalized_df['LipidMolec'].values

    def test_full_protein_normalization_workflow(self, basic_df, simple_experiment_2x2):
        """Test complete protein normalization workflow."""
        protein_conc = {'s1': 10.0, 's2': 20.0, 's3': 15.0, 's4': 25.0}
        norm_config = NormalizationConfig(
            method='protein',
            protein_concentrations=protein_conc
        )

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )

        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True
        # Values should be divided by protein concentration
        # s2 has 2x protein of s1, so normalized value should be ~half relative
        s1_val = result.normalized_df['concentration[s1]'].iloc[0]
        s2_val = result.normalized_df['concentration[s2]'].iloc[0]
        # Original s2 was 1.1e6 / 20 = 55000, s1 was 1e6 / 10 = 100000
        assert s1_val == pytest.approx(1e5, rel=0.01)
        assert s2_val == pytest.approx(5.5e4, rel=0.01)

    def test_full_combined_normalization_workflow(self, simple_experiment_2x2, basic_intsta_df):
        """Test complete combined normalization workflow."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1e6, 2e6],
            'intensity[s2]': [1.1e6, 2.1e6],
            'intensity[s3]': [1.2e6, 2.2e6],
            'intensity[s4]': [1.3e6, 2.3e6],
        })

        norm_config = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_18:1-d7)',
                'PE': 'PE(17:0_20:4-d7)'
            },
            intsta_concentrations={
                'PC(15:0_18:1-d7)': 1.0,
                'PE(17:0_20:4-d7)': 1.0
            },
            protein_concentrations={'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}
        )

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )

        result = NormalizationWorkflow.run(df, config, basic_intsta_df)

        assert result.success is True
        assert 'Combined' in result.method_applied

    def test_workflow_with_lipidsearch_format(self, simple_experiment_2x2, basic_intsta_df):
        """Test complete workflow with LipidSearch format."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [10.5, 12.3],
            'intensity[s1]': [1e6, 2e6],
            'intensity[s2]': [1.1e6, 2.1e6],
            'intensity[s3]': [1.2e6, 2.2e6],
            'intensity[s4]': [1.3e6, 2.3e6],
        })

        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(15:0_18:1-d7)',
                'PE': 'PE(17:0_20:4-d7)'
            },
            intsta_concentrations={
                'PC(15:0_18:1-d7)': 1.0,
                'PE(17:0_20:4-d7)': 1.0
            }
        )

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config,
            data_format=DataFormat.LIPIDSEARCH
        )

        result = NormalizationWorkflow.run(df, config, basic_intsta_df)

        assert result.success is True
        # Essential columns should be preserved
        assert 'CalcMass' in result.normalized_df.columns
        assert 'BaseRt' in result.normalized_df.columns


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    def test_graceful_handling_of_service_error(self, basic_df, simple_experiment_2x2):
        """Test graceful handling when service throws error."""
        # Create config that will cause service to fail
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'NONEXISTENT_STANDARD'},
            intsta_concentrations={'NONEXISTENT_STANDARD': 1.0}
        )

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=norm_config
        )

        # Empty intsta_df will cause failure
        result = NormalizationWorkflow.run(basic_df, config, pd.DataFrame())

        assert result.success is False
        assert len(result.validation_errors) > 0

    def test_validation_errors_prevent_normalization(
        self, df_missing_classkey, simple_experiment_2x2, none_config
    ):
        """Test that validation errors prevent normalization attempt."""
        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )

        result = NormalizationWorkflow.run(df_missing_classkey, config)

        assert result.success is False
        assert result.normalized_df is None


class TestWorkflowWithVariousExperiments:
    """Tests for workflow with different experiment configurations."""

    def test_single_condition_experiment(self, basic_df, single_condition_experiment, none_config):
        """Test workflow with single condition."""
        config = NormalizationWorkflowConfig(
            experiment=single_condition_experiment,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(basic_df, config)

        assert result.success is True

    def test_three_condition_experiment(self, basic_df, three_condition_experiment, none_config):
        """Test workflow with three conditions."""
        # Create matching DataFrame
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1e6, 2e6],
            'intensity[s2]': [1.1e6, 2.1e6],
            'intensity[s3]': [1.2e6, 2.2e6],
            'intensity[s4]': [1.3e6, 2.3e6],
            'intensity[s5]': [1.4e6, 2.4e6],
            'intensity[s6]': [1.5e6, 2.5e6],
        })

        config = NormalizationWorkflowConfig(
            experiment=three_condition_experiment,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df, config)

        assert result.success is True
        assert result.samples_processed == 6

    def test_unequal_samples_experiment(self, simple_experiment_2x2, none_config):
        """Test workflow with unequal sample counts."""
        # Create matching DataFrame for unequal experiment
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
            'intensity[s4]': [1.3e6],
        })

        config = NormalizationWorkflowConfig(
            experiment=simple_experiment_2x2,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df, config)

        assert result.success is True

    def test_large_experiment(self, large_experiment, none_config):
        """Test workflow with large experiment."""
        # Create large DataFrame
        n_samples = 20
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            **{f'intensity[s{i}]': [1e6 + i*1e4, 2e6 + i*1e4] for i in range(1, n_samples + 1)}
        })

        config = NormalizationWorkflowConfig(
            experiment=large_experiment,
            normalization=none_config
        )
        result = NormalizationWorkflow.run(df, config)

        assert result.success is True
        assert result.samples_processed == n_samples
