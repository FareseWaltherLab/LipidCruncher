"""
Unit tests for QualityCheckWorkflow.

Tests the quality check orchestration layer the QC page calls:
validation → BQC filter → retention time → correlation eligibility →
sample removal

Comprehensive test coverage matching Phase 3/4 service test depth.
"""
import pytest
import pandas as pd
import numpy as np

from app.workflows.quality_check import (
    QualityCheckWorkflow,
    QualityCheckConfig,
)
from app.services.quality_check import (
    BQCFilterResult,
    RetentionTimeDataResult,
    SampleRemovalResult,
)
from app.services.format_detection import DataFormat
from app.models.experiment import ExperimentConfig

# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def single_replicate_experiment():
    """2 conditions x 1 sample each = 2 samples."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1],
    )


@pytest.fixture
def mixed_replicate_experiment():
    """3 conditions with mixed replicate counts."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Single', 'Pair', 'Triple'],
        number_of_samples_list=[1, 2, 3],
    )


# =============================================================================
# DataFrame Fixtures
# =============================================================================

@pytest.fixture
def basic_conc_df(simple_experiment_2x3):
    """Basic DataFrame with concentration columns for 6 samples."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'concentration[s1]': [1000.0, 2000.0, 3000.0],
        'concentration[s2]': [1100.0, 2100.0, 3100.0],
        'concentration[s3]': [1200.0, 2200.0, 3200.0],
        'concentration[s4]': [1300.0, 2300.0, 3300.0],
        'concentration[s5]': [1400.0, 2400.0, 3400.0],
        'concentration[s6]': [1500.0, 2500.0, 3500.0],
    })


@pytest.fixture
def lipidsearch_conc_df(simple_experiment_2x3):
    """LipidSearch-format DataFrame with BaseRt and CalcMass."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'CalcMass': [760.5, 768.5, 850.7],
        'BaseRt': [10.5, 12.3, 15.0],
        'concentration[s1]': [1000.0, 2000.0, 3000.0],
        'concentration[s2]': [1100.0, 2100.0, 3100.0],
        'concentration[s3]': [1200.0, 2200.0, 3200.0],
        'concentration[s4]': [1300.0, 2300.0, 3300.0],
        'concentration[s5]': [1400.0, 2400.0, 3400.0],
        'concentration[s6]': [1500.0, 2500.0, 3500.0],
    })


@pytest.fixture
def large_conc_df(large_experiment):
    """Large DataFrame with 20 samples and 50 lipids."""
    n_lipids = 50
    data = {
        'LipidMolec': [f'Lipid_{i}' for i in range(n_lipids)],
        'ClassKey': [f'Class_{i % 5}' for i in range(n_lipids)],
    }
    np.random.seed(42)
    for i in range(1, 21):
        data[f'concentration[s{i}]'] = np.random.uniform(100, 10000, n_lipids)
    return pd.DataFrame(data)


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def lipidsearch_config():
    """Config for LipidSearch format."""
    return QualityCheckConfig(format_type=DataFormat.LIPIDSEARCH)


@pytest.fixture
def msdial_config():
    """Config for MS-DIAL format."""
    return QualityCheckConfig(format_type=DataFormat.MSDIAL)


# =============================================================================
# QualityCheckConfig Tests
# =============================================================================

class TestQualityCheckConfig:
    """Tests for QualityCheckConfig dataclass."""

    def test_default_values(self):
        assert QualityCheckConfig().format_type == DataFormat.GENERIC

    def test_all_format_types(self):
        for fmt in [DataFormat.GENERIC, DataFormat.LIPIDSEARCH,
                    DataFormat.MSDIAL, DataFormat.METABOLOMICS_WORKBENCH]:
            config = QualityCheckConfig(format_type=fmt)
            assert config.format_type == fmt


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidateInputs:
    """Tests for QualityCheckWorkflow.validate_inputs()."""

    def test_valid_inputs(self, basic_conc_df, simple_experiment_2x3):
        errors = QualityCheckWorkflow.validate_inputs(basic_conc_df, simple_experiment_2x3)
        assert errors == []

    def test_none_dataframe(self, simple_experiment_2x3):
        errors = QualityCheckWorkflow.validate_inputs(None, simple_experiment_2x3)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_empty_dataframe(self, simple_experiment_2x3):
        errors = QualityCheckWorkflow.validate_inputs(pd.DataFrame(), simple_experiment_2x3)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_missing_lipidmolec_column(self, simple_experiment_2x3):
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment_2x3)
        assert any('LipidMolec' in e for e in errors)

    def test_no_concentration_columns(self, simple_experiment_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment_2x3)
        assert any('concentration' in e.lower() for e in errors)

    def test_no_matching_samples(self, simple_experiment_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[x1]': [100.0],
            'concentration[x2]': [200.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment_2x3)
        assert any('match' in e.lower() for e in errors)

    def test_only_one_matching_sample(self):
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[x2]': [200.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, experiment)
        assert any('2 samples' in e.lower() or 'at least 2' in e.lower() for e in errors)

    def test_partial_matching_samples_valid(self, simple_experiment_2x3):
        """Some samples missing columns is OK if >=2 match."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            # s3-s6 missing — still OK because 2 >= 2
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment_2x3)
        assert errors == []

    def test_returns_early_on_empty_df(self, simple_experiment_2x3):
        """Empty df should return immediately, not check other conditions."""
        errors = QualityCheckWorkflow.validate_inputs(pd.DataFrame(), simple_experiment_2x3)
        assert len(errors) == 1

    def test_returns_early_on_no_conc_columns(self, simple_experiment_2x3):
        """No concentration columns returns early before sample matching."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'other_col': [100.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment_2x3)
        assert any('concentration' in e.lower() for e in errors)

    def test_valid_with_extra_columns(self, basic_conc_df, simple_experiment_2x3):
        """Extra non-concentration columns don't cause errors."""
        basic_conc_df['extra_col'] = 'test'
        errors = QualityCheckWorkflow.validate_inputs(basic_conc_df, simple_experiment_2x3)
        assert errors == []

    def test_valid_large_dataset(self, large_conc_df, large_experiment):
        errors = QualityCheckWorkflow.validate_inputs(large_conc_df, large_experiment)
        assert errors == []


# =============================================================================
# BQC Filter Tests
# =============================================================================

class TestApplyBQCFilter:
    """Tests for QualityCheckWorkflow.apply_bqc_filter()."""

    def test_returns_bqc_filter_result(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert isinstance(result, BQCFilterResult)

    def test_removes_high_cov_lipids(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert 'PC(16:0_18:1)' in result.removed_lipids
        assert result.lipids_before == 3
        assert result.lipids_after == 2

    def test_keeps_specified_lipids(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df,
            ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            lipids_to_keep=['PC(16:0_18:1)'],
        )
        assert 'PC(16:0_18:1)' not in result.removed_lipids
        assert 'PE(18:0_20:4)' in result.removed_lipids
        assert result.lipids_after == 2

    def test_keeps_all_lipids(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df,
            ['PC(16:0_18:1)'],
            lipids_to_keep=['PC(16:0_18:1)'],
        )
        assert result.removed_lipids == []
        assert result.lipids_after == 3

    def test_empty_high_cov_list(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(basic_conc_df, [])
        assert result.removed_lipids == []
        assert result.lipids_after == result.lipids_before

    def test_removed_count_property(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        )
        assert result.removed_count == 2

    def test_removed_percentage_property(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        expected = 1 / 3 * 100
        assert abs(result.removed_percentage - expected) < 0.1

    def test_filtered_df_sorted_by_classkey(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(basic_conc_df, [])
        classes = result.filtered_df['ClassKey'].tolist()
        assert classes == sorted(classes)

    def test_filtered_df_index_reset(self, basic_conc_df):
        result = QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PE(18:0_20:4)']
        )
        assert list(result.filtered_df.index) == list(range(len(result.filtered_df)))


# =============================================================================
# Retention Time Tests
# =============================================================================

class TestCheckRetentionTimeAvailability:
    """Tests for QualityCheckWorkflow.check_retention_time_availability()."""

    @pytest.mark.parametrize("format_type", [
        DataFormat.GENERIC,
        DataFormat.METABOLOMICS_WORKBENCH,
    ])
    def test_non_rt_formats_return_unavailable(self, basic_conc_df, format_type):
        """Generic and MW formats always return unavailable for retention time."""
        config = QualityCheckConfig(format_type=format_type)
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, config
        )
        assert isinstance(result, RetentionTimeDataResult)
        assert result.available is False

    def test_lipidsearch_with_rt_columns(
        self, lipidsearch_conc_df, lipidsearch_config
    ):
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_conc_df, lipidsearch_config
        )
        assert result.available is True
        assert len(result.lipid_classes) > 0

    @pytest.mark.parametrize("format_type", [
        DataFormat.LIPIDSEARCH,
        DataFormat.MSDIAL,
    ])
    def test_rt_formats_without_columns_return_unavailable(self, basic_conc_df, format_type):
        """LipidSearch/MS-DIAL without BaseRt/CalcMass return unavailable."""
        config = QualityCheckConfig(format_type=format_type)
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, config
        )
        assert result.available is False

    def test_msdial_with_rt_columns(self, lipidsearch_conc_df, msdial_config):
        """MS-DIAL format with BaseRt and CalcMass columns."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_conc_df, msdial_config
        )
        assert result.available is True

    def test_lipid_classes_sorted_by_frequency(self, lipidsearch_config):
        """Classes should be sorted by frequency (most common first)."""
        df = pd.DataFrame({
            'LipidMolec': ['A', 'B', 'C', 'D', 'E'],
            'ClassKey': ['PC', 'PE', 'PC', 'PC', 'PE'],
            'BaseRt': [1.0, 2.0, 3.0, 4.0, 5.0],
            'CalcMass': [100.0, 200.0, 300.0, 400.0, 500.0],
            'concentration[s1]': [1.0] * 5,
        })
        result = QualityCheckWorkflow.check_retention_time_availability(
            df, lipidsearch_config
        )
        assert result.lipid_classes[0] == 'PC'  # 3 occurrences
        assert result.lipid_classes[1] == 'PE'  # 2 occurrences


# =============================================================================
# Correlation Tests
# =============================================================================

class TestGetEligibleCorrelationConditions:
    """Tests for QualityCheckWorkflow.get_eligible_correlation_conditions()."""

    def test_all_conditions_eligible(self, simple_experiment_2x3):
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            simple_experiment_2x3
        )
        assert eligible == ['Control', 'Treatment']

    def test_single_replicate_not_eligible(self, single_replicate_experiment):
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            single_replicate_experiment
        )
        assert eligible == []

    def test_mixed_replicates(self, mixed_replicate_experiment):
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            mixed_replicate_experiment
        )
        assert 'Single' not in eligible
        assert 'Pair' in eligible
        assert 'Triple' in eligible

    def test_large_experiment(self, large_experiment):
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            large_experiment
        )
        assert len(eligible) == 4


# =============================================================================
# Sample Removal Tests
# =============================================================================

class TestRemoveSamples:
    """Tests for QualityCheckWorkflow.remove_samples()."""

    def test_returns_sample_removal_result(self, basic_conc_df, simple_experiment_2x3):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert isinstance(result, SampleRemovalResult)

    def test_single_sample_removal(self, basic_conc_df, simple_experiment_2x3):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert result.samples_before == 6
        assert result.samples_after == 5
        assert len(result.removed_samples) == 1

    def test_multiple_sample_removal(self, basic_conc_df, simple_experiment_2x3):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1', 's4']
        )
        assert result.samples_before == 6
        assert result.samples_after == 4

    def test_updated_experiment(self, basic_conc_df, simple_experiment_2x3):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert result.updated_experiment.n_conditions == 2
        total_samples = sum(result.updated_experiment.number_of_samples_list)
        assert total_samples == 5

    def test_updated_df_columns(self, basic_conc_df, simple_experiment_2x3):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        conc_cols = [c for c in result.updated_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) == 5

    def test_raises_on_empty_list(self, basic_conc_df, simple_experiment_2x3):
        with pytest.raises(ValueError, match='empty'):
            QualityCheckWorkflow.remove_samples(
                basic_conc_df, simple_experiment_2x3, []
            )

    def test_raises_if_too_few_remaining(self, basic_conc_df, simple_experiment_2x3):
        with pytest.raises(ValueError, match='at least 2'):
            QualityCheckWorkflow.remove_samples(
                basic_conc_df, simple_experiment_2x3,
                ['s1', 's2', 's3', 's4', 's5']
            )

    def test_does_not_modify_original_df(self, basic_conc_df, simple_experiment_2x3):
        original_shape = basic_conc_df.shape
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert basic_conc_df.shape == original_shape

    def test_does_not_modify_original_experiment(self, basic_conc_df, simple_experiment_2x3):
        original_n = simple_experiment_2x3.n_conditions
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert simple_experiment_2x3.n_conditions == original_n


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""


    def test_retention_time_without_classkey(self, lipidsearch_config):
        """RT availability check without ClassKey column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'BaseRt': [10.5],
            'CalcMass': [760.5],
            'concentration[s1]': [100.0],
        })
        result = QualityCheckWorkflow.check_retention_time_availability(
            df, lipidsearch_config
        )
        assert result.available is True
        assert result.lipid_classes == []


# =============================================================================
# Type Coercion Tests
# =============================================================================

class TestTypeCoercion:
    """Tests for handling various data types."""


    def test_int_concentrations_remove_samples(self, simple_experiment_2x3):
        """remove_samples handles integer concentration columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [120, 220],
            'concentration[s4]': [130, 230],
            'concentration[s5]': [140, 240],
            'concentration[s6]': [150, 250],
        })
        result = QualityCheckWorkflow.remove_samples(
            df, simple_experiment_2x3, ['s1']
        )
        assert isinstance(result, SampleRemovalResult)
        assert result.samples_after == 5


    def test_int_concentrations_filter_by_bqc(self):
        """apply_bqc_filter handles integer concentration columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
        })
        result = QualityCheckWorkflow.apply_bqc_filter(df, ['PE(18:0)'])
        assert isinstance(result, BQCFilterResult)
        assert len(result.filtered_df) == 2


# =============================================================================
# Immutability Tests
# =============================================================================

class TestImmutability:
    """Tests to ensure workflow methods don't modify input data."""


    def test_bqc_filter_immutable(self, basic_conc_df):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        pd.testing.assert_frame_equal(basic_conc_df, original)


    def test_remove_samples_immutable(self, basic_conc_df, simple_experiment_2x3):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        pd.testing.assert_frame_equal(basic_conc_df, original)
