"""
Unit tests for QualityCheckWorkflow.

Tests the quality check orchestration layer:
validation → box plots → BQC assessment → retention time →
correlation → PCA → sample removal → non-interactive pipeline

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
    BoxPlotResult,
    BQCPrepareResult,
    BQCFilterResult,
    RetentionTimeDataResult,
    CorrelationResult,
    PCAResult,
    SampleRemovalResult,
)
from app.services.format_detection import DataFormat
from app.models.experiment import ExperimentConfig


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def simple_experiment():
    """2 conditions x 3 samples each = 6 samples (s1..s6)."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


@pytest.fixture
def three_condition_experiment():
    """3 conditions x 2 samples each = 6 samples (s1..s6)."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'Vehicle'],
        number_of_samples_list=[2, 2, 2],
    )


@pytest.fixture
def bqc_experiment():
    """3 conditions including BQC: Control(3), Treatment(3), BQC(2) = 8 samples."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'BQC'],
        number_of_samples_list=[3, 3, 2],
    )


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


@pytest.fixture
def large_experiment():
    """4 conditions x 5 samples each = 20 samples."""
    return ExperimentConfig(
        n_conditions=4,
        conditions_list=['A', 'B', 'C', 'D'],
        number_of_samples_list=[5, 5, 5, 5],
    )


# =============================================================================
# DataFrame Fixtures
# =============================================================================

@pytest.fixture
def basic_conc_df(simple_experiment):
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
def conc_df_with_zeros(simple_experiment):
    """DataFrame with some zero values in concentration columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'concentration[s1]': [1000.0, 0.0, 3000.0, 0.0],
        'concentration[s2]': [1100.0, 2100.0, 0.0, 0.0],
        'concentration[s3]': [0.0, 2200.0, 3200.0, 0.0],
        'concentration[s4]': [1300.0, 2300.0, 3300.0, 4300.0],
        'concentration[s5]': [0.0, 0.0, 3400.0, 4400.0],
        'concentration[s6]': [1500.0, 2500.0, 3500.0, 0.0],
    })


@pytest.fixture
def bqc_conc_df(bqc_experiment):
    """DataFrame for BQC experiment with 8 samples."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'concentration[s1]': [100.0, 200.0, 300.0, 400.0],
        'concentration[s2]': [110.0, 210.0, 310.0, 410.0],
        'concentration[s3]': [120.0, 220.0, 320.0, 420.0],
        'concentration[s4]': [130.0, 230.0, 330.0, 430.0],
        'concentration[s5]': [140.0, 240.0, 340.0, 440.0],
        'concentration[s6]': [150.0, 250.0, 350.0, 450.0],
        # BQC samples: s7, s8 — low variability
        'concentration[s7]': [100.0, 200.0, 300.0, 400.0],
        'concentration[s8]': [102.0, 198.0, 305.0, 395.0],
    })


@pytest.fixture
def bqc_conc_df_high_cov(bqc_experiment):
    """DataFrame where some lipids have high CoV in BQC samples."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'concentration[s1]': [100.0, 200.0, 300.0, 400.0],
        'concentration[s2]': [110.0, 210.0, 310.0, 410.0],
        'concentration[s3]': [120.0, 220.0, 320.0, 420.0],
        'concentration[s4]': [130.0, 230.0, 330.0, 430.0],
        'concentration[s5]': [140.0, 240.0, 340.0, 440.0],
        'concentration[s6]': [150.0, 250.0, 350.0, 450.0],
        # BQC samples: s7=low, s8=very different → high CoV for PC, SM
        'concentration[s7]': [50.0, 200.0, 300.0, 100.0],
        'concentration[s8]': [200.0, 198.0, 305.0, 500.0],
    })


@pytest.fixture
def lipidsearch_conc_df(simple_experiment):
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


@pytest.fixture
def three_cond_conc_df(three_condition_experiment):
    """DataFrame with concentration columns for 3-condition experiment."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'concentration[s1]': [100.0, 200.0, 300.0, 400.0],
        'concentration[s2]': [110.0, 210.0, 310.0, 410.0],
        'concentration[s3]': [120.0, 220.0, 320.0, 420.0],
        'concentration[s4]': [130.0, 230.0, 330.0, 430.0],
        'concentration[s5]': [140.0, 240.0, 340.0, 440.0],
        'concentration[s6]': [150.0, 250.0, 350.0, 450.0],
    })


@pytest.fixture
def mixed_conc_df(mixed_replicate_experiment):
    """DataFrame for mixed replicate experiment (1+2+3=6 samples)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'concentration[s1]': [100.0, 200.0, 300.0],
        'concentration[s2]': [110.0, 210.0, 310.0],
        'concentration[s3]': [120.0, 220.0, 320.0],
        'concentration[s4]': [130.0, 230.0, 330.0],
        'concentration[s5]': [140.0, 240.0, 340.0],
        'concentration[s6]': [150.0, 250.0, 350.0],
    })


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default QualityCheckConfig (no BQC, Generic format)."""
    return QualityCheckConfig()


@pytest.fixture
def bqc_config():
    """Config with BQC label."""
    return QualityCheckConfig(bqc_label='BQC', cov_threshold=30.0)


@pytest.fixture
def lipidsearch_config():
    """Config for LipidSearch format."""
    return QualityCheckConfig(format_type=DataFormat.LIPIDSEARCH)


@pytest.fixture
def msdial_config():
    """Config for MS-DIAL format."""
    return QualityCheckConfig(format_type=DataFormat.MSDIAL)


@pytest.fixture
def lipidsearch_bqc_config():
    """Config for LipidSearch with BQC."""
    return QualityCheckConfig(
        bqc_label='BQC',
        format_type=DataFormat.LIPIDSEARCH,
        cov_threshold=30.0,
    )


# =============================================================================
# QualityCheckConfig Tests
# =============================================================================

class TestQualityCheckConfig:
    """Tests for QualityCheckConfig dataclass."""

    def test_default_values(self):
        config = QualityCheckConfig()
        assert config.bqc_label is None
        assert config.format_type == DataFormat.GENERIC
        assert config.cov_threshold == 30.0

    def test_custom_values(self):
        config = QualityCheckConfig(
            bqc_label='BQC',
            format_type=DataFormat.LIPIDSEARCH,
            cov_threshold=25.0,
        )
        assert config.bqc_label == 'BQC'
        assert config.format_type == DataFormat.LIPIDSEARCH
        assert config.cov_threshold == 25.0

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

    def test_valid_inputs(self, basic_conc_df, simple_experiment):
        errors = QualityCheckWorkflow.validate_inputs(basic_conc_df, simple_experiment)
        assert errors == []

    def test_none_dataframe(self, simple_experiment):
        errors = QualityCheckWorkflow.validate_inputs(None, simple_experiment)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_empty_dataframe(self, simple_experiment):
        errors = QualityCheckWorkflow.validate_inputs(pd.DataFrame(), simple_experiment)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_missing_lipidmolec_column(self, simple_experiment):
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment)
        assert any('LipidMolec' in e for e in errors)

    def test_no_concentration_columns(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment)
        assert any('concentration' in e.lower() for e in errors)

    def test_no_matching_samples(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[x1]': [100.0],
            'concentration[x2]': [200.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment)
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

    def test_partial_matching_samples_valid(self, simple_experiment):
        """Some samples missing columns is OK if >=2 match."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            # s3-s6 missing — still OK because 2 >= 2
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment)
        assert errors == []

    def test_returns_early_on_empty_df(self, simple_experiment):
        """Empty df should return immediately, not check other conditions."""
        errors = QualityCheckWorkflow.validate_inputs(pd.DataFrame(), simple_experiment)
        assert len(errors) == 1

    def test_returns_early_on_no_conc_columns(self, simple_experiment):
        """No concentration columns returns early before sample matching."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'other_col': [100.0],
        })
        errors = QualityCheckWorkflow.validate_inputs(df, simple_experiment)
        assert any('concentration' in e.lower() for e in errors)

    def test_valid_with_extra_columns(self, basic_conc_df, simple_experiment):
        """Extra non-concentration columns don't cause errors."""
        basic_conc_df['extra_col'] = 'test'
        errors = QualityCheckWorkflow.validate_inputs(basic_conc_df, simple_experiment)
        assert errors == []

    def test_valid_large_dataset(self, large_conc_df, large_experiment):
        errors = QualityCheckWorkflow.validate_inputs(large_conc_df, large_experiment)
        assert errors == []


# =============================================================================
# Box Plot Tests
# =============================================================================

class TestRunBoxPlots:
    """Tests for QualityCheckWorkflow.run_box_plots()."""

    def test_returns_box_plot_result(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        assert isinstance(result, BoxPlotResult)

    def test_mean_area_df_shape(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        assert result.mean_area_df.shape == (3, 6)

    def test_available_samples(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        assert result.available_samples == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_missing_values_length(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        assert len(result.missing_values_percent) == 6

    def test_no_missing_values(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        assert all(pct == 0.0 for pct in result.missing_values_percent)

    def test_with_zeros(self, conc_df_with_zeros, simple_experiment):
        result = QualityCheckWorkflow.run_box_plots(conc_df_with_zeros, simple_experiment)
        assert any(pct > 0.0 for pct in result.missing_values_percent)

    def test_raises_on_empty_df(self, simple_experiment):
        with pytest.raises(ValueError, match='empty'):
            QualityCheckWorkflow.run_box_plots(pd.DataFrame(), simple_experiment)

    def test_raises_on_none_df(self, simple_experiment):
        with pytest.raises(ValueError):
            QualityCheckWorkflow.run_box_plots(None, simple_experiment)

    def test_with_large_dataset(self, large_conc_df, large_experiment):
        result = QualityCheckWorkflow.run_box_plots(large_conc_df, large_experiment)
        assert len(result.available_samples) == 20
        assert result.mean_area_df.shape[0] == 50


# =============================================================================
# BQC Assessment Tests
# =============================================================================

class TestRunBQCAssessment:
    """Tests for QualityCheckWorkflow.run_bqc_assessment()."""

    def test_returns_none_when_no_bqc_label(self, basic_conc_df, simple_experiment, default_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            basic_conc_df, simple_experiment, default_config
        )
        assert result is None

    def test_returns_none_when_bqc_label_not_in_experiment(
        self, basic_conc_df, simple_experiment
    ):
        config = QualityCheckConfig(bqc_label='NonExistent')
        result = QualityCheckWorkflow.run_bqc_assessment(
            basic_conc_df, simple_experiment, config
        )
        assert result is None

    def test_returns_bqc_result_when_valid(self, bqc_conc_df, bqc_experiment, bqc_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert isinstance(result, BQCPrepareResult)

    def test_bqc_samples_identified(self, bqc_conc_df, bqc_experiment, bqc_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert result.bqc_samples == ['s7', 's8']

    def test_reliable_data_percent(self, bqc_conc_df, bqc_experiment, bqc_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert 0.0 <= result.reliable_data_percent <= 100.0

    def test_high_cov_lipids_list(self, bqc_conc_df, bqc_experiment, bqc_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        # With low variability BQC samples, all lipids should be reliable
        assert isinstance(result.high_cov_lipids, list)

    def test_prepared_df_has_cov_mean_columns(
        self, bqc_conc_df, bqc_experiment, bqc_config
    ):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert 'cov' in result.prepared_df.columns
        assert 'mean' in result.prepared_df.columns

    def test_custom_cov_threshold(self, bqc_conc_df_high_cov, bqc_experiment):
        # Strict threshold → more lipids flagged
        strict_config = QualityCheckConfig(bqc_label='BQC', cov_threshold=10.0)
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df_high_cov, bqc_experiment, strict_config
        )
        strict_count = len(result.high_cov_lipids)

        # Lenient threshold → fewer lipids flagged
        lenient_config = QualityCheckConfig(bqc_label='BQC', cov_threshold=80.0)
        result2 = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df_high_cov, bqc_experiment, lenient_config
        )
        lenient_count = len(result2.high_cov_lipids)

        assert strict_count >= lenient_count

    def test_bqc_index_correct(self, bqc_conc_df, bqc_experiment, bqc_config):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert result.bqc_sample_index == 2  # BQC is 3rd condition

    def test_high_cov_details_dataframe(
        self, bqc_conc_df_high_cov, bqc_experiment, bqc_config
    ):
        result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df_high_cov, bqc_experiment, bqc_config
        )
        assert isinstance(result.high_cov_details, pd.DataFrame)
        if not result.high_cov_details.empty:
            assert 'LipidMolec' in result.high_cov_details.columns

    def test_does_not_modify_input_df(self, bqc_conc_df, bqc_experiment, bqc_config):
        original_cols = list(bqc_conc_df.columns)
        QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert list(bqc_conc_df.columns) == original_cols
        assert 'cov' not in bqc_conc_df.columns


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

    def test_generic_format_returns_unavailable(
        self, basic_conc_df, default_config
    ):
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, default_config
        )
        assert isinstance(result, RetentionTimeDataResult)
        assert result.available is False
        assert result.lipid_classes == []

    def test_metabolomics_workbench_returns_unavailable(self, basic_conc_df):
        config = QualityCheckConfig(format_type=DataFormat.METABOLOMICS_WORKBENCH)
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, config
        )
        assert result.available is False

    def test_lipidsearch_with_rt_columns(
        self, lipidsearch_conc_df, lipidsearch_config
    ):
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_conc_df, lipidsearch_config
        )
        assert result.available is True
        assert len(result.lipid_classes) > 0

    def test_lipidsearch_without_rt_columns(self, basic_conc_df, lipidsearch_config):
        """LipidSearch format but missing BaseRt/CalcMass columns."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, lipidsearch_config
        )
        assert result.available is False

    def test_msdial_with_rt_columns(self, lipidsearch_conc_df, msdial_config):
        """MS-DIAL format with BaseRt and CalcMass columns."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_conc_df, msdial_config
        )
        assert result.available is True

    def test_msdial_without_rt_columns(self, basic_conc_df, msdial_config):
        result = QualityCheckWorkflow.check_retention_time_availability(
            basic_conc_df, msdial_config
        )
        assert result.available is False

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

    def test_all_conditions_eligible(self, simple_experiment):
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            simple_experiment
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


class TestRunCorrelation:
    """Tests for QualityCheckWorkflow.run_correlation()."""

    def test_returns_correlation_result(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert isinstance(result, CorrelationResult)

    def test_correlation_matrix_shape(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert result.correlation_df.shape == (3, 3)  # 3 samples per condition

    def test_biological_replicates_threshold(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert result.sample_type == 'biological replicates'
        assert result.threshold == 0.7

    def test_technical_replicates_threshold(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control', bqc_label='BQC'
        )
        assert result.sample_type == 'technical replicates'
        assert result.threshold == 0.8

    def test_condition_samples_correct(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert result.condition_samples == ['s1', 's2', 's3']

    def test_diagonal_is_one(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        for i in range(len(result.correlation_df)):
            assert abs(result.correlation_df.iloc[i, i] - 1.0) < 1e-10

    def test_correlation_values_in_range(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert (result.correlation_df >= -1.0).all().all()
        assert (result.correlation_df <= 1.0).all().all()

    def test_raises_on_invalid_condition(self, basic_conc_df, simple_experiment):
        with pytest.raises(ValueError, match='not found'):
            QualityCheckWorkflow.run_correlation(
                basic_conc_df, simple_experiment, 'NonExistent'
            )

    def test_raises_on_single_replicate(self, mixed_conc_df, mixed_replicate_experiment):
        with pytest.raises(ValueError, match='at least 2'):
            QualityCheckWorkflow.run_correlation(
                mixed_conc_df, mixed_replicate_experiment, 'Single'
            )

    def test_v_min_is_half(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        assert result.v_min == 0.5


class TestRunAllCorrelations:
    """Tests for QualityCheckWorkflow.run_all_correlations()."""

    def test_returns_dict_of_results(self, basic_conc_df, simple_experiment):
        results = QualityCheckWorkflow.run_all_correlations(
            basic_conc_df, simple_experiment
        )
        assert isinstance(results, dict)
        assert len(results) == 2
        assert 'Control' in results
        assert 'Treatment' in results

    def test_each_result_is_correlation(self, basic_conc_df, simple_experiment):
        results = QualityCheckWorkflow.run_all_correlations(
            basic_conc_df, simple_experiment
        )
        for cond, result in results.items():
            assert isinstance(result, CorrelationResult)

    def test_skips_single_replicate_conditions(
        self, mixed_conc_df, mixed_replicate_experiment
    ):
        results = QualityCheckWorkflow.run_all_correlations(
            mixed_conc_df, mixed_replicate_experiment
        )
        assert 'Single' not in results
        assert 'Pair' in results
        assert 'Triple' in results

    def test_empty_when_all_single_replicate(
        self, single_replicate_experiment
    ):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
        })
        results = QualityCheckWorkflow.run_all_correlations(
            df, single_replicate_experiment
        )
        assert results == {}

    def test_bqc_label_passed_through(self, basic_conc_df, simple_experiment):
        results = QualityCheckWorkflow.run_all_correlations(
            basic_conc_df, simple_experiment, bqc_label='SomeBQC'
        )
        for result in results.values():
            assert result.sample_type == 'technical replicates'

    def test_three_conditions(self, three_cond_conc_df, three_condition_experiment):
        results = QualityCheckWorkflow.run_all_correlations(
            three_cond_conc_df, three_condition_experiment
        )
        assert len(results) == 3


# =============================================================================
# PCA Tests
# =============================================================================

class TestRunPCA:
    """Tests for QualityCheckWorkflow.run_pca()."""

    def test_returns_pca_result(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert isinstance(result, PCAResult)

    def test_pc_df_shape(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert result.pc_df.shape == (6, 2)  # 6 samples, 2 components

    def test_pc_df_columns(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert list(result.pc_df.columns) == ['PC1', 'PC2']

    def test_pc_labels_format(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert len(result.pc_labels) == 2
        assert 'PC1' in result.pc_labels[0]
        assert 'PC2' in result.pc_labels[1]
        assert '%' in result.pc_labels[0]

    def test_available_samples(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert result.available_samples == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_conditions_mapping(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert result.conditions == [
            'Control', 'Control', 'Control',
            'Treatment', 'Treatment', 'Treatment',
        ]

    def test_raises_on_single_sample(self):
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        with pytest.raises(ValueError, match='at least 2'):
            QualityCheckWorkflow.run_pca(df, experiment)

    def test_large_dataset(self, large_conc_df, large_experiment):
        result = QualityCheckWorkflow.run_pca(large_conc_df, large_experiment)
        assert result.pc_df.shape == (20, 2)
        assert len(result.conditions) == 20


# =============================================================================
# Sample Removal Tests
# =============================================================================

class TestRemoveSamples:
    """Tests for QualityCheckWorkflow.remove_samples()."""

    def test_returns_sample_removal_result(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        assert isinstance(result, SampleRemovalResult)

    def test_single_sample_removal(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        assert result.samples_before == 6
        assert result.samples_after == 5
        assert len(result.removed_samples) == 1

    def test_multiple_sample_removal(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1', 's4']
        )
        assert result.samples_before == 6
        assert result.samples_after == 4

    def test_updated_experiment(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        assert result.updated_experiment.n_conditions == 2
        total_samples = sum(result.updated_experiment.number_of_samples_list)
        assert total_samples == 5

    def test_updated_df_columns(self, basic_conc_df, simple_experiment):
        result = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        conc_cols = [c for c in result.updated_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) == 5

    def test_raises_on_empty_list(self, basic_conc_df, simple_experiment):
        with pytest.raises(ValueError, match='empty'):
            QualityCheckWorkflow.remove_samples(
                basic_conc_df, simple_experiment, []
            )

    def test_raises_if_too_few_remaining(self, basic_conc_df, simple_experiment):
        with pytest.raises(ValueError, match='at least 2'):
            QualityCheckWorkflow.remove_samples(
                basic_conc_df, simple_experiment,
                ['s1', 's2', 's3', 's4', 's5']
            )

    def test_does_not_modify_original_df(self, basic_conc_df, simple_experiment):
        original_shape = basic_conc_df.shape
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        assert basic_conc_df.shape == original_shape

    def test_does_not_modify_original_experiment(self, basic_conc_df, simple_experiment):
        original_n = simple_experiment.n_conditions
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        assert simple_experiment.n_conditions == original_n


# =============================================================================
# Non-Interactive Pipeline Tests
# =============================================================================

class TestRunNonInteractive:
    """Tests for QualityCheckWorkflow.run_non_interactive()."""

    def test_returns_all_expected_keys(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert 'box_plot' in results
        assert 'bqc' in results
        assert 'retention_time' in results
        assert 'correlations' in results
        assert 'pca' in results
        assert 'validation_errors' in results

    def test_box_plot_populated(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert isinstance(results['box_plot'], BoxPlotResult)

    def test_bqc_none_without_label(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert results['bqc'] is None

    def test_bqc_populated_with_label(
        self, bqc_conc_df, bqc_experiment, bqc_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert isinstance(results['bqc'], BQCPrepareResult)

    def test_retention_time_generic_format(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert results['retention_time'].available is False

    def test_retention_time_lipidsearch(
        self, lipidsearch_conc_df, simple_experiment, lipidsearch_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            lipidsearch_conc_df, simple_experiment, lipidsearch_config
        )
        assert results['retention_time'].available is True

    def test_correlations_dict(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert isinstance(results['correlations'], dict)
        assert len(results['correlations']) == 2

    def test_pca_populated(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert isinstance(results['pca'], PCAResult)

    def test_no_validation_errors_on_valid_input(
        self, basic_conc_df, simple_experiment, default_config
    ):
        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert results['validation_errors'] == []

    def test_validation_errors_on_invalid_input(self, simple_experiment, default_config):
        results = QualityCheckWorkflow.run_non_interactive(
            pd.DataFrame(), simple_experiment, default_config
        )
        assert len(results['validation_errors']) > 0
        assert results['box_plot'] is None
        assert results['pca'] is None

    def test_none_df_returns_errors(self, simple_experiment, default_config):
        results = QualityCheckWorkflow.run_non_interactive(
            None, simple_experiment, default_config
        )
        assert len(results['validation_errors']) > 0

    def test_large_dataset(self, large_conc_df, large_experiment, default_config):
        results = QualityCheckWorkflow.run_non_interactive(
            large_conc_df, large_experiment, default_config
        )
        assert results['validation_errors'] == []
        assert results['box_plot'] is not None
        assert results['pca'] is not None
        assert len(results['correlations']) == 4


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_lipid_df(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert result.mean_area_df.shape[0] == 1

    def test_all_zeros_df(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [0.0, 0.0],
            'concentration[s2]': [0.0, 0.0],
            'concentration[s3]': [0.0, 0.0],
            'concentration[s4]': [0.0, 0.0],
            'concentration[s5]': [0.0, 0.0],
            'concentration[s6]': [0.0, 0.0],
        })
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert all(pct == 100.0 for pct in result.missing_values_percent)

    def test_nan_values_in_concentrations(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, np.nan],
            'concentration[s2]': [np.nan, 200.0],
            'concentration[s3]': [300.0, 300.0],
            'concentration[s4]': [400.0, 400.0],
            'concentration[s5]': [500.0, 500.0],
            'concentration[s6]': [600.0, 600.0],
        })
        # Should not raise — NaN is handled
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert isinstance(result, BoxPlotResult)

    def test_special_characters_in_lipid_names(self, simple_experiment):
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)+D7:(s)', 'PE(p-18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [110.0, 210.0],
            'concentration[s3]': [120.0, 220.0],
            'concentration[s4]': [130.0, 230.0],
            'concentration[s5]': [140.0, 240.0],
            'concentration[s6]': [150.0, 250.0],
        })
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert result.mean_area_df.shape[0] == 2

    def test_two_samples_minimum_for_pca(self):
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100.0, 200.0, 300.0],
            'concentration[s2]': [150.0, 250.0, 350.0],
        })
        result = QualityCheckWorkflow.run_pca(df, experiment)
        assert result.pc_df.shape == (2, 2)

    def test_identical_samples_pca(self, simple_experiment):
        """PCA with identical samples — should handle gracefully."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [100.0, 200.0],
            'concentration[s3]': [100.0, 200.0],
            'concentration[s4]': [100.0, 200.0],
            'concentration[s5]': [100.0, 200.0],
            'concentration[s6]': [100.0, 200.0],
        })
        # Should not raise
        result = QualityCheckWorkflow.run_pca(df, simple_experiment)
        assert isinstance(result, PCAResult)

    def test_bqc_filter_then_pca(self, bqc_conc_df_high_cov, bqc_experiment, bqc_config):
        """BQC filter followed by PCA — multi-step pipeline."""
        # Step 1: BQC assessment
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df_high_cov, bqc_experiment, bqc_config
        )

        # Step 2: Filter (remove all high-CoV lipids)
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            bqc_conc_df_high_cov, bqc_result.high_cov_lipids
        )
        filtered_df = filter_result.filtered_df

        # Step 3: PCA on filtered data (if enough data)
        if len(filtered_df) > 0:
            result = QualityCheckWorkflow.run_pca(filtered_df, bqc_experiment)
            assert isinstance(result, PCAResult)

    def test_pca_then_remove_then_pca(self, basic_conc_df, simple_experiment):
        """PCA → remove samples → re-run PCA pipeline."""
        # First PCA
        pca1 = QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        assert pca1.pc_df.shape[0] == 6

        # Remove a sample
        removal = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )

        # Second PCA with updated data
        pca2 = QualityCheckWorkflow.run_pca(
            removal.updated_df, removal.updated_experiment
        )
        assert pca2.pc_df.shape[0] == 5

    def test_correlation_after_sample_removal(
        self, basic_conc_df, simple_experiment
    ):
        """Correlation works after removing samples."""
        removal = QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        # Control now has 2 samples, Treatment still has 3
        result = QualityCheckWorkflow.run_correlation(
            removal.updated_df, removal.updated_experiment, 'Control'
        )
        assert result.correlation_df.shape == (2, 2)

    def test_validate_then_run(self, basic_conc_df, simple_experiment, default_config):
        """Validate inputs before running pipeline."""
        errors = QualityCheckWorkflow.validate_inputs(
            basic_conc_df, simple_experiment
        )
        assert errors == []

        results = QualityCheckWorkflow.run_non_interactive(
            basic_conc_df, simple_experiment, default_config
        )
        assert results['validation_errors'] == []

    def test_no_classkey_column(self, simple_experiment):
        """DataFrame without ClassKey column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [110.0, 210.0],
            'concentration[s3]': [120.0, 220.0],
            'concentration[s4]': [130.0, 230.0],
            'concentration[s5]': [140.0, 240.0],
            'concentration[s6]': [150.0, 250.0],
        })
        # Box plots should work without ClassKey
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert isinstance(result, BoxPlotResult)

    def test_partial_samples_box_plots(self):
        """DataFrame missing some sample columns."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[3, 3],
        )
        # Only s1, s2, s4, s5 present (missing s3, s6)
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
        })
        result = QualityCheckWorkflow.run_box_plots(df, experiment)
        assert len(result.available_samples) == 4

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

    def test_full_pipeline_with_bqc(
        self, bqc_conc_df, bqc_experiment, bqc_config
    ):
        """Full non-interactive pipeline with BQC."""
        results = QualityCheckWorkflow.run_non_interactive(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        assert results['validation_errors'] == []
        assert results['box_plot'] is not None
        assert results['bqc'] is not None
        assert results['pca'] is not None


# =============================================================================
# Type Coercion Tests
# =============================================================================

class TestTypeCoercion:
    """Tests for handling various data types."""

    def test_string_numbers_in_concentration(self, simple_experiment):
        """String numbers in concentration columns (object dtype)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': ['100', '200'],
            'concentration[s2]': ['110', '210'],
            'concentration[s3]': ['120', '220'],
            'concentration[s4]': ['130', '230'],
            'concentration[s5]': ['140', '240'],
            'concentration[s6]': ['150', '250'],
        })
        # Convert to numeric as would happen in real pipeline
        for col in [c for c in df.columns if c.startswith('concentration[')]:
            df[col] = pd.to_numeric(df[col])
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert isinstance(result, BoxPlotResult)

    def test_int_concentrations(self, simple_experiment):
        """Integer concentration values."""
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
        result = QualityCheckWorkflow.run_box_plots(df, simple_experiment)
        assert isinstance(result, BoxPlotResult)

    def test_mixed_numeric_types(self, simple_experiment):
        """Mixed int and float concentrations."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200.5],
            'concentration[s2]': [110.5, 210],
            'concentration[s3]': [120, 220.5],
            'concentration[s4]': [130.5, 230],
            'concentration[s5]': [140, 240.5],
            'concentration[s6]': [150.5, 250],
        })
        result = QualityCheckWorkflow.run_pca(df, simple_experiment)
        assert isinstance(result, PCAResult)


# =============================================================================
# Immutability Tests
# =============================================================================

class TestImmutability:
    """Tests to ensure workflow methods don't modify input data."""

    def test_box_plots_immutable(self, basic_conc_df, simple_experiment):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.run_box_plots(basic_conc_df, simple_experiment)
        pd.testing.assert_frame_equal(basic_conc_df, original)

    def test_bqc_assessment_immutable(
        self, bqc_conc_df, bqc_experiment, bqc_config
    ):
        original = bqc_conc_df.copy()
        QualityCheckWorkflow.run_bqc_assessment(
            bqc_conc_df, bqc_experiment, bqc_config
        )
        pd.testing.assert_frame_equal(bqc_conc_df, original)

    def test_bqc_filter_immutable(self, basic_conc_df):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.apply_bqc_filter(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        pd.testing.assert_frame_equal(basic_conc_df, original)

    def test_correlation_immutable(self, basic_conc_df, simple_experiment):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.run_correlation(
            basic_conc_df, simple_experiment, 'Control'
        )
        pd.testing.assert_frame_equal(basic_conc_df, original)

    def test_pca_immutable(self, basic_conc_df, simple_experiment):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.run_pca(basic_conc_df, simple_experiment)
        pd.testing.assert_frame_equal(basic_conc_df, original)

    def test_remove_samples_immutable(self, basic_conc_df, simple_experiment):
        original = basic_conc_df.copy()
        QualityCheckWorkflow.remove_samples(
            basic_conc_df, simple_experiment, ['s1']
        )
        pd.testing.assert_frame_equal(basic_conc_df, original)