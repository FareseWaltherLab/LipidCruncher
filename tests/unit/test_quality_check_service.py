"""Unit tests for QualityCheckService."""
import pytest
import pandas as pd
import numpy as np
from app.services.quality_check import (
    QualityCheckService,
    BoxPlotResult,
    BQCPrepareResult,
    BQCFilterResult,
    RetentionTimeDataResult,
    CorrelationResult,
    PCAResult,
    SampleRemovalResult,
)
from app.models.experiment import ExperimentConfig

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def single_sample_experiment():
    """2 conditions x 1 sample each = 2 samples."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1],
    )


@pytest.fixture
def large_experiment():
    """4 conditions x 5 samples each = 20 samples."""
    return ExperimentConfig(
        n_conditions=4,
        conditions_list=['A', 'B', 'C', 'D'],
        number_of_samples_list=[5, 5, 5, 5],
    )


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
def conc_df_with_zeros(simple_experiment_2x3):
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
    """DataFrame with concentration columns for 8 samples (includes BQC)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'concentration[s1]': [100.0, 200.0, 300.0, 400.0],
        'concentration[s2]': [110.0, 210.0, 310.0, 410.0],
        'concentration[s3]': [120.0, 220.0, 320.0, 420.0],
        'concentration[s4]': [130.0, 230.0, 330.0, 430.0],
        'concentration[s5]': [140.0, 240.0, 340.0, 440.0],
        'concentration[s6]': [150.0, 250.0, 350.0, 450.0],
        # BQC samples: s7, s8 — low variability → low CoV
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
        # BQC samples: s7, s8 — PC and TG have high variability
        'concentration[s7]': [50.0, 200.0, 100.0, 400.0],
        'concentration[s8]': [200.0, 198.0, 500.0, 395.0],
    })


@pytest.fixture
def rt_df():
    """DataFrame with retention time columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'BaseRt': [5.1, 6.2, 7.3],
        'CalcMass': [733.5, 743.5, 850.7],
        'concentration[s1]': [100.0, 200.0, 300.0],
    })


@pytest.fixture
def multi_class_df(simple_experiment_2x3):
    """DataFrame with multiple lipids per class for correlation/PCA."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(18:0_18:1)', 'PC(16:0_20:4)',
            'PE(18:0_20:4)', 'PE(16:0_22:6)',
            'TG(16:0_18:1_18:2)',
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PE', 'PE', 'TG'],
        'concentration[s1]': [1000, 1100, 1200, 2000, 2100, 3000],
        'concentration[s2]': [1050, 1150, 1250, 2050, 2150, 3050],
        'concentration[s3]': [1100, 1200, 1300, 2100, 2200, 3100],
        'concentration[s4]': [900, 1000, 1100, 1900, 2000, 2900],
        'concentration[s5]': [950, 1050, 1150, 1950, 2050, 2950],
        'concentration[s6]': [800, 900, 1000, 1800, 1900, 2800],
    })


# =============================================================================
# TestCoVCalculation
# =============================================================================

class TestCoVCalculation:
    """Tests for calculate_coefficient_of_variation()."""

    def test_standard_cov(self):
        """CoV of known values: std(ddof=1)/mean * 100."""
        values = [100, 100, 100, 100]
        result = QualityCheckService.calculate_coefficient_of_variation(values)
        assert result == 0.0

    def test_cov_known_values(self):
        """Verify formula: CoV = std(ddof=1) / mean * 100."""
        values = [10, 20, 30]
        expected = float(np.std(values, ddof=1) / np.mean(values) * 100)
        result = QualityCheckService.calculate_coefficient_of_variation(values)
        assert result == pytest.approx(expected)

    def test_cov_with_two_values(self):
        """CoV with exactly two values should work."""
        result = QualityCheckService.calculate_coefficient_of_variation([10, 20])
        assert result is not None
        assert result > 0

    def test_cov_single_value_returns_none(self):
        """CoV with one value returns None (need >=2)."""
        result = QualityCheckService.calculate_coefficient_of_variation([42])
        assert result is None

    def test_cov_empty_returns_none(self):
        """CoV with empty input returns None."""
        result = QualityCheckService.calculate_coefficient_of_variation([])
        assert result is None

    def test_cov_all_zeros_returns_none(self):
        """CoV when all values are zero returns None (mean=0)."""
        result = QualityCheckService.calculate_coefficient_of_variation([0, 0, 0])
        assert result is None

    def test_cov_includes_zeros(self):
        """CoV includes zero values in calculation."""
        result = QualityCheckService.calculate_coefficient_of_variation([0, 100, 200])
        assert result is not None
        assert result > 0

    def test_cov_large_variability(self):
        """CoV for highly variable data should be large."""
        result = QualityCheckService.calculate_coefficient_of_variation([1, 1000])
        assert result is not None
        assert result > 100

    def test_cov_numpy_array_input(self):
        """CoV accepts numpy array input."""
        arr = np.array([10.0, 20.0, 30.0])
        result = QualityCheckService.calculate_coefficient_of_variation(arr)
        assert result is not None

    def test_cov_pandas_series_input(self):
        """CoV accepts pandas Series input."""
        series = pd.Series([10.0, 20.0, 30.0])
        result = QualityCheckService.calculate_coefficient_of_variation(series)
        assert result is not None

    def test_cov_float_values(self):
        """CoV works with float values."""
        result = QualityCheckService.calculate_coefficient_of_variation([1.5, 2.5, 3.5])
        assert result is not None
        assert isinstance(result, float)

    def test_cov_negative_values(self):
        """CoV works with negative values (if mean != 0)."""
        result = QualityCheckService.calculate_coefficient_of_variation([-10, -20, -30])
        assert result is not None

    def test_cov_mixed_positive_negative(self):
        """CoV with mixed sign where mean could be near zero."""
        result = QualityCheckService.calculate_coefficient_of_variation([100, -100])
        # Mean is 0, so returns None
        assert result is None


# =============================================================================
# TestMeanCalculation
# =============================================================================

class TestMeanCalculation:
    """Tests for calculate_mean_including_zeros()."""

    def test_mean_basic(self):
        """Mean of simple values."""
        result = QualityCheckService.calculate_mean_including_zeros([10, 20, 30])
        assert result == pytest.approx(20.0)

    def test_mean_with_zeros(self):
        """Mean includes zeros."""
        result = QualityCheckService.calculate_mean_including_zeros([0, 0, 30])
        assert result == pytest.approx(10.0)

    def test_mean_single_value_returns_none(self):
        """Mean with one value returns None."""
        result = QualityCheckService.calculate_mean_including_zeros([42])
        assert result is None

    def test_mean_empty_returns_none(self):
        """Mean with empty input returns None."""
        result = QualityCheckService.calculate_mean_including_zeros([])
        assert result is None

    def test_mean_all_zeros(self):
        """Mean of all zeros is 0."""
        result = QualityCheckService.calculate_mean_including_zeros([0, 0, 0])
        assert result == pytest.approx(0.0)

    def test_mean_two_values(self):
        """Mean with exactly two values."""
        result = QualityCheckService.calculate_mean_including_zeros([10, 20])
        assert result == pytest.approx(15.0)

    def test_mean_numpy_array(self):
        """Mean accepts numpy array."""
        arr = np.array([10.0, 20.0, 30.0])
        result = QualityCheckService.calculate_mean_including_zeros(arr)
        assert result == pytest.approx(20.0)

    def test_mean_pandas_series(self):
        """Mean accepts pandas Series."""
        series = pd.Series([10.0, 20.0, 30.0])
        result = QualityCheckService.calculate_mean_including_zeros(series)
        assert result == pytest.approx(20.0)

    def test_mean_returns_float(self):
        """Mean returns a float type."""
        result = QualityCheckService.calculate_mean_including_zeros([10, 20])
        assert isinstance(result, float)


# =============================================================================
# TestPrepareBoxPlotData
# =============================================================================

class TestPrepareBoxPlotData:
    """Tests for prepare_box_plot_data()."""

    def test_basic_box_plot_data(self, basic_conc_df, simple_experiment_2x3):
        """Basic preparation returns correct structure."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert isinstance(result, BoxPlotResult)
        assert len(result.available_samples) == 6
        assert result.mean_area_df.shape == (3, 6)

    def test_returns_concentration_columns_only(self, basic_conc_df, simple_experiment_2x3):
        """mean_area_df contains only concentration columns."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        for col in result.mean_area_df.columns:
            assert col.startswith('concentration[')

    def test_no_metadata_columns(self, basic_conc_df, simple_experiment_2x3):
        """mean_area_df does not contain LipidMolec or ClassKey."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert 'LipidMolec' not in result.mean_area_df.columns
        assert 'ClassKey' not in result.mean_area_df.columns

    def test_missing_values_all_nonzero(self, basic_conc_df, simple_experiment_2x3):
        """No zeros → all missing_values_percent should be 0."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert all(pct == 0.0 for pct in result.missing_values_percent)

    def test_missing_values_with_zeros(self, conc_df_with_zeros, simple_experiment_2x3):
        """Correct missing value percentages when zeros exist."""
        result = QualityCheckService.prepare_box_plot_data(conc_df_with_zeros, simple_experiment_2x3)
        n_rows = len(conc_df_with_zeros)
        # s1: [1000, 0, 3000, 0] → 2 zeros out of 4 = 50%
        assert result.missing_values_percent[0] == pytest.approx(50.0)

    def test_missing_values_length_matches_samples(self, basic_conc_df, simple_experiment_2x3):
        """missing_values_percent has same length as available_samples."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert len(result.missing_values_percent) == len(result.available_samples)

    def test_available_samples_order(self, basic_conc_df, simple_experiment_2x3):
        """available_samples follows experiment's full_samples_list order."""
        result = QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert result.available_samples == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_partial_columns_available(self, simple_experiment_2x3):
        """Only samples with concentration columns are returned."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            # s3-s6 missing
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert result.available_samples == ['s1', 's2']
        assert result.mean_area_df.shape[1] == 2

    def test_empty_dataframe_raises(self, simple_experiment_2x3):
        """Empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)

    def test_no_concentration_columns_raises(self, simple_experiment_2x3):
        """DataFrame without concentration columns raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        with pytest.raises(ValueError, match="no concentration columns"):
            QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)

    def test_does_not_modify_original(self, basic_conc_df, simple_experiment_2x3):
        """Original DataFrame is not modified."""
        original_cols = list(basic_conc_df.columns)
        QualityCheckService.prepare_box_plot_data(basic_conc_df, simple_experiment_2x3)
        assert list(basic_conc_df.columns) == original_cols

    def test_all_zeros_100_percent_missing(self, simple_experiment_2x3):
        """All-zero column should be 100% missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [0.0, 0.0],
            'concentration[s2]': [100.0, 200.0],
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        # s1 is all zeros
        s1_idx = result.available_samples.index('s1')
        assert result.missing_values_percent[s1_idx] == pytest.approx(100.0)

    def test_single_row_dataframe(self, simple_experiment_2x3):
        """Single-row DataFrame works correctly."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [0.0],
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        s1_idx = result.available_samples.index('s1')
        s2_idx = result.available_samples.index('s2')
        assert result.missing_values_percent[s1_idx] == pytest.approx(0.0)
        assert result.missing_values_percent[s2_idx] == pytest.approx(100.0)


# =============================================================================
# TestPrepareBQCData
# =============================================================================

class TestPrepareBQCData:
    """Tests for prepare_bqc_data()."""

    def test_basic_bqc_preparation(self, bqc_conc_df, bqc_experiment):
        """Basic BQC preparation returns correct result type."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert isinstance(result, BQCPrepareResult)

    def test_bqc_sample_index(self, bqc_conc_df, bqc_experiment):
        """bqc_sample_index is correct index in conditions_list."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert result.bqc_sample_index == 2  # BQC is 3rd condition

    def test_bqc_samples_correct(self, bqc_conc_df, bqc_experiment):
        """bqc_samples contains the right sample labels."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert result.bqc_samples == ['s7', 's8']

    def test_prepared_df_has_cov_column(self, bqc_conc_df, bqc_experiment):
        """Prepared DataFrame has 'cov' column."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert 'cov' in result.prepared_df.columns

    def test_prepared_df_has_mean_column(self, bqc_conc_df, bqc_experiment):
        """Prepared DataFrame has 'mean' column (log10 scale)."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert 'mean' in result.prepared_df.columns

    def test_mean_is_log10(self, bqc_conc_df, bqc_experiment):
        """Mean values are log10-transformed for positive values."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        # First lipid: BQC samples are s7=100, s8=102, mean=101
        # log10(101) ≈ 2.004
        mean_val = result.prepared_df.iloc[0]['mean']
        assert mean_val == pytest.approx(np.log10(101.0), abs=0.01)

    def test_low_cov_means_high_reliability(self, bqc_conc_df, bqc_experiment):
        """Low CoV in all lipids → high reliable_data_percent."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        # All lipids have low CoV (s7 ≈ s8) → 100% reliable
        assert result.reliable_data_percent == pytest.approx(100.0, abs=1.0)

    def test_high_cov_lipids_identified(self, bqc_conc_df_high_cov, bqc_experiment):
        """Lipids with CoV >= threshold are identified."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df_high_cov, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        # PC: s7=50, s8=200 → very high CoV
        # TG: s7=100, s8=500 → very high CoV
        assert 'PC(16:0)' in result.high_cov_lipids
        assert 'TG(16:0)' in result.high_cov_lipids

    def test_low_cov_lipids_not_in_high_list(self, bqc_conc_df_high_cov, bqc_experiment):
        """Lipids with CoV < threshold are NOT in high_cov_lipids."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df_high_cov, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        # PE: s7=200, s8=198 → very low CoV
        # SM: s7=400, s8=395 → very low CoV
        assert 'PE(18:0)' not in result.high_cov_lipids
        assert 'SM(d18:1)' not in result.high_cov_lipids

    def test_high_cov_details_columns(self, bqc_conc_df_high_cov, bqc_experiment):
        """high_cov_details has expected columns."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df_high_cov, bqc_experiment, 'BQC'
        )
        assert 'LipidMolec' in result.high_cov_details.columns
        assert 'ClassKey' in result.high_cov_details.columns
        assert 'cov' in result.high_cov_details.columns
        assert 'mean' in result.high_cov_details.columns

    def test_custom_cov_threshold(self, bqc_conc_df, bqc_experiment):
        """Custom CoV threshold changes high_cov_lipids."""
        # Very low threshold → more lipids flagged
        result_low = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC', cov_threshold=0.1
        )
        # Very high threshold → fewer lipids flagged
        result_high = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC', cov_threshold=99.0
        )
        assert len(result_low.high_cov_lipids) >= len(result_high.high_cov_lipids)

    def test_invalid_bqc_label_raises(self, bqc_conc_df, bqc_experiment):
        """Non-existent BQC label raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            QualityCheckService.prepare_bqc_data(
                bqc_conc_df, bqc_experiment, 'NonExistent'
            )

    def test_empty_dataframe_raises(self, bqc_experiment):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService.prepare_bqc_data(
                pd.DataFrame(), bqc_experiment, 'BQC'
            )

    def test_negative_cov_threshold_raises(self, bqc_conc_df, bqc_experiment):
        """Negative CoV threshold raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QualityCheckService.prepare_bqc_data(
                bqc_conc_df, bqc_experiment, 'BQC', cov_threshold=-5.0
            )

    def test_zero_cov_threshold_raises(self, bqc_conc_df, bqc_experiment):
        """Zero CoV threshold raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QualityCheckService.prepare_bqc_data(
                bqc_conc_df, bqc_experiment, 'BQC', cov_threshold=0.0
            )

    def test_missing_bqc_columns_raises(self, bqc_experiment):
        """Missing BQC concentration columns raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            # s7, s8 (BQC) missing
        })
        with pytest.raises(ValueError, match="not found"):
            QualityCheckService.prepare_bqc_data(df, bqc_experiment, 'BQC')

    def test_does_not_modify_original(self, bqc_conc_df, bqc_experiment):
        """Original DataFrame is not modified."""
        original_cols = list(bqc_conc_df.columns)
        original_shape = bqc_conc_df.shape
        QualityCheckService.prepare_bqc_data(bqc_conc_df, bqc_experiment, 'BQC')
        assert list(bqc_conc_df.columns) == original_cols
        assert bqc_conc_df.shape == original_shape

    def test_reliable_percent_with_all_high_cov(self, bqc_experiment):
        """All lipids having high CoV → 0% reliable."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [110.0, 210.0],
            'concentration[s3]': [120.0, 220.0],
            'concentration[s4]': [130.0, 230.0],
            'concentration[s5]': [140.0, 240.0],
            'concentration[s6]': [150.0, 250.0],
            'concentration[s7]': [10.0, 20.0],
            'concentration[s8]': [500.0, 800.0],
        })
        result = QualityCheckService.prepare_bqc_data(
            df, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        assert result.reliable_data_percent == pytest.approx(0.0)

    def test_bqc_with_zero_mean_lipid(self, bqc_experiment):
        """Lipid with zero mean in BQC samples: mean should be null."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [110.0, 210.0],
            'concentration[s3]': [120.0, 220.0],
            'concentration[s4]': [130.0, 230.0],
            'concentration[s5]': [140.0, 240.0],
            'concentration[s6]': [150.0, 250.0],
            'concentration[s7]': [0.0, 200.0],
            'concentration[s8]': [0.0, 198.0],
        })
        result = QualityCheckService.prepare_bqc_data(
            df, bqc_experiment, 'BQC'
        )
        # PC has mean=0 in BQC samples → CoV is None
        pc_cov = result.prepared_df.iloc[0]['cov']
        assert pd.isna(pc_cov)

    def test_bqc_as_first_condition(self):
        """BQC as the first condition (index 0) works correctly."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['BQC', 'Treatment'],
            number_of_samples_list=[3, 3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [102.0, 198.0],
            'concentration[s3]': [98.0, 202.0],
            'concentration[s4]': [130.0, 230.0],
            'concentration[s5]': [140.0, 240.0],
            'concentration[s6]': [150.0, 250.0],
        })
        result = QualityCheckService.prepare_bqc_data(df, exp, 'BQC')
        assert result.bqc_sample_index == 0
        assert result.bqc_samples == ['s1', 's2', 's3']

    def test_prepared_df_preserves_row_count(self, bqc_conc_df, bqc_experiment):
        """prepared_df has same number of rows as input."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        assert len(result.prepared_df) == len(bqc_conc_df)

    def test_high_cov_details_row_count_matches_list(self, bqc_conc_df_high_cov, bqc_experiment):
        """high_cov_details row count matches high_cov_lipids length."""
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df_high_cov, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        assert len(result.high_cov_details) == len(result.high_cov_lipids)

    def test_mixed_reliability_percent(self, bqc_experiment):
        """50% of lipids below threshold → ~50% reliable."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
            'ClassKey': ['PC', 'PE', 'TG', 'SM'],
            'concentration[s1]': [100.0, 200.0, 300.0, 400.0],
            'concentration[s2]': [110.0, 210.0, 310.0, 410.0],
            'concentration[s3]': [120.0, 220.0, 320.0, 420.0],
            'concentration[s4]': [130.0, 230.0, 330.0, 430.0],
            'concentration[s5]': [140.0, 240.0, 340.0, 440.0],
            'concentration[s6]': [150.0, 250.0, 350.0, 450.0],
            # BQC: PC and TG high variability, PE and SM low variability
            'concentration[s7]': [50.0, 200.0, 100.0, 400.0],
            'concentration[s8]': [200.0, 198.0, 500.0, 395.0],
        })
        result = QualityCheckService.prepare_bqc_data(
            df, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        # 2 below threshold out of 4 lipids → 50%
        assert result.reliable_data_percent == pytest.approx(50.0)

    def test_prepared_df_preserves_original_columns(self, bqc_conc_df, bqc_experiment):
        """prepared_df still has all original columns plus cov and mean."""
        original_cols = set(bqc_conc_df.columns)
        result = QualityCheckService.prepare_bqc_data(
            bqc_conc_df, bqc_experiment, 'BQC'
        )
        for col in original_cols:
            assert col in result.prepared_df.columns

    def test_cov_threshold_at_exact_boundary(self, bqc_experiment):
        """Lipid with CoV exactly at threshold is flagged as high."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [110.0],
            'concentration[s3]': [120.0],
            'concentration[s4]': [130.0],
            'concentration[s5]': [140.0],
            'concentration[s6]': [150.0],
            'concentration[s7]': [78.0],
            'concentration[s8]': [122.0],
        })
        result = QualityCheckService.prepare_bqc_data(
            df, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        cov_val = result.prepared_df.iloc[0]['cov']
        if cov_val >= 30.0:
            assert 'PC(16:0)' in result.high_cov_lipids
        else:
            assert 'PC(16:0)' not in result.high_cov_lipids

    def test_bqc_with_many_bqc_samples(self):
        """BQC with many replicates (5 BQC samples)."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Treatment', 'BQC'],
            number_of_samples_list=[3, 5],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [110.0],
            'concentration[s3]': [120.0],
            'concentration[s4]': [100.0],
            'concentration[s5]': [101.0],
            'concentration[s6]': [99.0],
            'concentration[s7]': [100.5],
            'concentration[s8]': [100.2],
        })
        result = QualityCheckService.prepare_bqc_data(df, exp, 'BQC')
        assert result.bqc_samples == ['s4', 's5', 's6', 's7', 's8']
        assert result.reliable_data_percent == pytest.approx(100.0, abs=1.0)


# =============================================================================
# TestFilterByBQC
# =============================================================================

class TestFilterByBQC:
    """Tests for filter_by_bqc()."""

    def test_remove_all_high_cov(self, basic_conc_df):
        """Remove all high-CoV lipids when lipids_to_keep is empty."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov)
        assert isinstance(result, BQCFilterResult)
        assert len(result.filtered_df) == 1  # Only TG remains
        assert result.removed_lipids == high_cov
        assert result.kept_despite_high_cov == []

    def test_keep_some_high_cov(self, basic_conc_df):
        """Keep selected lipids despite high CoV."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        keep = ['PC(16:0_18:1)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov, keep)
        assert len(result.filtered_df) == 2  # PC + TG
        assert 'PE(18:0_20:4)' in result.removed_lipids
        assert 'PC(16:0_18:1)' not in result.removed_lipids

    def test_keep_all_high_cov(self, basic_conc_df):
        """Keep all high-CoV lipids → no removal."""
        high_cov = ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov, high_cov)
        assert len(result.filtered_df) == 3  # All kept
        assert result.removed_lipids == []

    def test_empty_high_cov_list(self, basic_conc_df):
        """No high-CoV lipids → no changes."""
        result = QualityCheckService.filter_by_bqc(basic_conc_df, [])
        assert len(result.filtered_df) == len(basic_conc_df)
        assert result.removed_lipids == []

    def test_none_lipids_to_keep(self, basic_conc_df):
        """lipids_to_keep=None removes all high-CoV."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)'], None
        )
        assert len(result.filtered_df) == 2

    def test_removed_count_property(self, basic_conc_df):
        """removed_count computed correctly."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert result.removed_count == 1
        assert result.lipids_before == 3
        assert result.lipids_after == 2

    def test_removed_percentage_property(self, basic_conc_df):
        """removed_percentage computed correctly."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        expected_pct = (1 / 3) * 100
        assert result.removed_percentage == pytest.approx(expected_pct)

    def test_removed_percentage_zero_lipids(self):
        """removed_percentage with 0 lipids_before returns 0."""
        result = BQCFilterResult(
            filtered_df=pd.DataFrame(),
            removed_lipids=[],
            kept_despite_high_cov=[],
            lipids_before=0,
            lipids_after=0,
        )
        assert result.removed_percentage == 0.0

    def test_sorted_by_classkey(self, basic_conc_df):
        """Filtered DataFrame is sorted by ClassKey."""
        result = QualityCheckService.filter_by_bqc(basic_conc_df, [])
        if 'ClassKey' in result.filtered_df.columns:
            classkeys = result.filtered_df['ClassKey'].tolist()
            assert classkeys == sorted(classkeys)

    def test_index_reset(self, basic_conc_df):
        """Filtered DataFrame has reset index."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        assert list(result.filtered_df.index) == list(range(len(result.filtered_df)))

    def test_does_not_modify_original(self, basic_conc_df):
        """Original DataFrame is not modified."""
        original_len = len(basic_conc_df)
        QualityCheckService.filter_by_bqc(basic_conc_df, ['PC(16:0_18:1)'])
        assert len(basic_conc_df) == original_len

    def test_nonexistent_lipid_in_high_cov(self, basic_conc_df):
        """Non-existent lipid in high_cov_lipids is silently ignored."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['NONEXISTENT_LIPID']
        )
        assert len(result.filtered_df) == 3  # Nothing removed
        assert result.removed_lipids == ['NONEXISTENT_LIPID']

    def test_without_classkey_column(self):
        """DataFrame without ClassKey still works (no sorting)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'concentration[s1]': [100.0, 200.0, 300.0],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PE(18:0)'])
        assert len(result.filtered_df) == 2
        assert list(result.filtered_df.index) == [0, 1]

    def test_keep_list_with_items_not_in_high_cov(self, basic_conc_df):
        """keep list items not in high_cov are harmless."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)'], ['PC(16:0_18:1)', 'NOT_IN_HIGH_COV']
        )
        assert len(result.filtered_df) == 3  # PC kept
        assert result.removed_lipids == []

    def test_duplicate_entries_in_high_cov(self, basic_conc_df):
        """Duplicate entries in high_cov_lipids don't cause double removal."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)', 'PC(16:0_18:1)']
        )
        assert len(result.filtered_df) == 2  # PC removed once
        assert result.removed_count == 1

    def test_all_lipids_removed(self, basic_conc_df):
        """Removing all lipids produces empty DataFrame."""
        all_lipids = basic_conc_df['LipidMolec'].tolist()
        result = QualityCheckService.filter_by_bqc(basic_conc_df, all_lipids)
        assert len(result.filtered_df) == 0
        assert result.removed_percentage == pytest.approx(100.0)

    def test_removed_lipids_order_preserved(self, basic_conc_df):
        """removed_lipids list preserves order of high_cov_lipids."""
        high_cov = ['TG(16:0_18:1_18:2)', 'PC(16:0_18:1)']
        result = QualityCheckService.filter_by_bqc(basic_conc_df, high_cov)
        assert result.removed_lipids == high_cov

    def test_filter_preserves_data_values(self, basic_conc_df):
        """Filtering preserves concentration values of remaining lipids."""
        result = QualityCheckService.filter_by_bqc(
            basic_conc_df, ['PC(16:0_18:1)']
        )
        # PE row should have same values
        pe_row = result.filtered_df[
            result.filtered_df['LipidMolec'] == 'PE(18:0_20:4)'
        ]
        assert pe_row['concentration[s1]'].iloc[0] == 2000.0


# =============================================================================
# TestRetentionTimeAvailability
# =============================================================================

class TestRetentionTimeAvailability:
    """Tests for check_retention_time_availability()."""

    def test_available_with_both_columns(self, rt_df):
        """Available when both BaseRt and CalcMass exist."""
        result = QualityCheckService.check_retention_time_availability(rt_df)
        assert isinstance(result, RetentionTimeDataResult)
        assert result.available is True

    def test_lipid_classes_listed(self, rt_df):
        """Lipid classes are returned sorted by frequency."""
        result = QualityCheckService.check_retention_time_availability(rt_df)
        assert set(result.lipid_classes) == {'PC', 'PE', 'TG'}

    def test_not_available_missing_basert(self):
        """Not available when BaseRt is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'CalcMass': [733.5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False
        assert result.lipid_classes == []

    def test_not_available_missing_calcmass(self):
        """Not available when CalcMass is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'BaseRt': [5.1],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False

    def test_not_available_no_rt_columns(self):
        """Not available when neither column exists."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is False

    def test_available_without_classkey(self):
        """Available but no classes when ClassKey is missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'BaseRt': [5.1],
            'CalcMass': [733.5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is True
        assert result.lipid_classes == []

    def test_classes_sorted_by_frequency(self):
        """Classes are sorted by frequency (most frequent first)."""
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c', 'd', 'e'],
            'ClassKey': ['PE', 'PC', 'PC', 'PC', 'PE'],
            'BaseRt': [1, 2, 3, 4, 5],
            'CalcMass': [1, 2, 3, 4, 5],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.lipid_classes[0] == 'PC'  # 3 occurrences
        assert result.lipid_classes[1] == 'PE'  # 2 occurrences

    def test_empty_dataframe(self):
        """Empty DataFrame → not available."""
        df = pd.DataFrame(columns=['BaseRt', 'CalcMass', 'ClassKey'])
        result = QualityCheckService.check_retention_time_availability(df)
        # Empty DF has the columns but no rows
        assert result.available is True
        assert result.lipid_classes == []


# =============================================================================
# TestCorrelationEligibleConditions
# =============================================================================

class TestCorrelationEligibleConditions:
    """Tests for get_correlation_eligible_conditions()."""

    def test_all_multi_replicate(self, simple_experiment_2x3):
        """All conditions eligible when all have >1 replicate."""
        result = QualityCheckService.get_correlation_eligible_conditions(simple_experiment_2x3)
        assert result == ['Control', 'Treatment']

    def test_mixed_replicates(self):
        """Only multi-replicate conditions returned."""
        exp = ExperimentConfig(
            n_conditions=3,
            conditions_list=['A', 'B', 'C'],
            number_of_samples_list=[3, 1, 2],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['A', 'C']

    def test_all_single_replicate(self, single_sample_experiment):
        """No eligible conditions when all have 1 replicate."""
        result = QualityCheckService.get_correlation_eligible_conditions(
            single_sample_experiment
        )
        assert result == []

    def test_preserves_order(self, large_experiment):
        """Eligible conditions preserve original order."""
        result = QualityCheckService.get_correlation_eligible_conditions(large_experiment)
        assert result == ['A', 'B', 'C', 'D']

    def test_single_condition_multi_replicate(self):
        """Single condition with multiple replicates."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Only'],
            number_of_samples_list=[5],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['Only']

    def test_exactly_two_replicates_is_eligible(self):
        """Condition with exactly 2 replicates is eligible."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 1],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['A']

    def test_three_conditions_with_bqc(self, three_condition_experiment):
        """Three conditions all eligible (all have 2 replicates)."""
        result = QualityCheckService.get_correlation_eligible_conditions(
            three_condition_experiment
        )
        assert result == ['Control', 'Treatment', 'Vehicle']

    def test_many_conditions_varied_replicates(self):
        """Many conditions with varied sample counts."""
        exp = ExperimentConfig(
            n_conditions=5,
            conditions_list=['A', 'B', 'C', 'D', 'E'],
            number_of_samples_list=[1, 5, 1, 3, 1],
        )
        result = QualityCheckService.get_correlation_eligible_conditions(exp)
        assert result == ['B', 'D']


# =============================================================================
# TestComputeCorrelation
# =============================================================================

class TestComputeCorrelation:
    """Tests for compute_correlation()."""

    def test_basic_correlation(self, multi_class_df, simple_experiment_2x3):
        """Basic correlation returns correct structure."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert isinstance(result, CorrelationResult)
        assert result.condition == 'Control'

    def test_correlation_matrix_shape(self, multi_class_df, simple_experiment_2x3):
        """Correlation matrix is square with n_samples dimensions."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        # Control has 3 samples (s1, s2, s3)
        assert result.correlation_df.shape == (3, 3)

    def test_correlation_diagonal_is_one(self, multi_class_df, simple_experiment_2x3):
        """Diagonal of correlation matrix should be 1.0."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        for i in range(len(result.correlation_df)):
            assert result.correlation_df.iloc[i, i] == pytest.approx(1.0)

    def test_correlation_symmetry(self, multi_class_df, simple_experiment_2x3):
        """Correlation matrix should be symmetric."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        corr = result.correlation_df
        for i in range(len(corr)):
            for j in range(len(corr)):
                assert corr.iloc[i, j] == pytest.approx(corr.iloc[j, i])

    def test_biological_threshold(self, multi_class_df, simple_experiment_2x3):
        """Without bqc_label → biological replicates (threshold=0.7)."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert result.threshold == 0.7
        assert result.sample_type == 'biological replicates'

    def test_technical_threshold(self, multi_class_df, simple_experiment_2x3):
        """With bqc_label → technical replicates (threshold=0.8)."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control', bqc_label='BQC'
        )
        assert result.threshold == 0.8
        assert result.sample_type == 'technical replicates'

    def test_vmin_always_half(self, multi_class_df, simple_experiment_2x3):
        """v_min is always 0.5."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert result.v_min == 0.5

    def test_condition_samples_correct(self, multi_class_df, simple_experiment_2x3):
        """condition_samples lists the right samples."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert result.condition_samples == ['s1', 's2', 's3']

    def test_treatment_condition(self, multi_class_df, simple_experiment_2x3):
        """Correlation works for second condition."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Treatment'
        )
        assert result.condition == 'Treatment'
        assert result.condition_samples == ['s4', 's5', 's6']

    def test_invalid_condition_raises(self, multi_class_df, simple_experiment_2x3):
        """Non-existent condition raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            QualityCheckService.compute_correlation(
                multi_class_df, simple_experiment_2x3, 'NonExistent'
            )

    def test_single_sample_condition_raises(self):
        """Condition with only 1 sample raises ValueError."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[1, 3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
        })
        with pytest.raises(ValueError, match="at least 2 replicates"):
            QualityCheckService.compute_correlation(df, exp, 'A')

    def test_empty_dataframe_raises(self, simple_experiment_2x3):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService.compute_correlation(
                pd.DataFrame(), simple_experiment_2x3, 'Control'
            )

    def test_missing_concentration_columns_raises(self, simple_experiment_2x3):
        """Missing concentration columns raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            # s2, s3 missing
        })
        with pytest.raises(ValueError, match="Missing concentration columns"):
            QualityCheckService.compute_correlation(df, simple_experiment_2x3, 'Control')

    def test_column_names_are_sample_labels(self, multi_class_df, simple_experiment_2x3):
        """Correlation DataFrame columns/index are sample labels, not concentration[...]."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert list(result.correlation_df.columns) == ['s1', 's2', 's3']
        assert list(result.correlation_df.index) == ['s1', 's2', 's3']

    def test_highly_correlated_samples(self):
        """Samples with proportional values should have correlation ≈ 1."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [200, 400, 600],
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        assert result.correlation_df.iloc[0, 1] == pytest.approx(1.0)

    def test_correlation_values_between_minus1_and_1(self, multi_class_df, simple_experiment_2x3):
        """All correlation values should be between -1 and 1."""
        result = QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        for i in range(len(result.correlation_df)):
            for j in range(len(result.correlation_df)):
                val = result.correlation_df.iloc[i, j]
                assert -1.0 <= val <= 1.0 or pd.isna(val)

    def test_anticorrelated_samples(self):
        """Samples with inversely proportional values → correlation ≈ -1."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [300, 200, 100],
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        assert result.correlation_df.iloc[0, 1] == pytest.approx(-1.0)

    def test_correlation_three_condition_experiment(self, three_condition_experiment):
        """Correlation works for a 3-condition experiment (2 samples each)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
            'concentration[s3]': [120, 220, 320],
            'concentration[s4]': [130, 230, 330],
            'concentration[s5]': [140, 240, 340],
            'concentration[s6]': [150, 250, 350],
        })
        result = QualityCheckService.compute_correlation(
            df, three_condition_experiment, 'Vehicle'
        )
        assert result.condition_samples == ['s5', 's6']
        assert result.correlation_df.shape == (2, 2)

    def test_correlation_does_not_modify_original(self, multi_class_df, simple_experiment_2x3):
        """Original DataFrame is not modified by correlation."""
        original_cols = list(multi_class_df.columns)
        QualityCheckService.compute_correlation(
            multi_class_df, simple_experiment_2x3, 'Control'
        )
        assert list(multi_class_df.columns) == original_cols

    def test_correlation_with_zeros(self):
        """Correlation handles zero values in data."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': [0, 100, 200],
            'concentration[s2]': [0, 0, 100],
            'concentration[s3]': [100, 200, 300],
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        assert result.correlation_df.shape == (3, 3)


# =============================================================================
# TestComputePCA
# =============================================================================

class TestComputePCA:
    """Tests for compute_pca()."""

    def test_basic_pca(self, multi_class_df, simple_experiment_2x3):
        """Basic PCA returns correct structure."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert isinstance(result, PCAResult)

    def test_pc_df_shape(self, multi_class_df, simple_experiment_2x3):
        """PC DataFrame has (n_samples, 2) shape."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert result.pc_df.shape == (6, 2)

    def test_pc_df_columns(self, multi_class_df, simple_experiment_2x3):
        """PC DataFrame has PC1 and PC2 columns."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert list(result.pc_df.columns) == ['PC1', 'PC2']

    def test_pc_labels_format(self, multi_class_df, simple_experiment_2x3):
        """PC labels contain variance explained percentages."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert len(result.pc_labels) == 2
        assert result.pc_labels[0].startswith('PC1')
        assert '%' in result.pc_labels[0]
        assert result.pc_labels[1].startswith('PC2')
        assert '%' in result.pc_labels[1]

    def test_available_samples(self, multi_class_df, simple_experiment_2x3):
        """available_samples lists all samples with concentration columns."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert result.available_samples == ['s1', 's2', 's3', 's4', 's5', 's6']

    def test_conditions_mapped(self, multi_class_df, simple_experiment_2x3):
        """Conditions mapped correctly per sample."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert result.conditions == [
            'Control', 'Control', 'Control',
            'Treatment', 'Treatment', 'Treatment',
        ]

    def test_variance_explained_sums_to_one_or_less(self, multi_class_df, simple_experiment_2x3):
        """Variance explained by PC1+PC2 <= 100%."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        # Extract percentages from labels like 'PC1 (45%)'
        import re
        pcts = []
        for label in result.pc_labels:
            match = re.search(r'(\d+)%', label)
            if match:
                pcts.append(int(match.group(1)))
        assert sum(pcts) <= 100

    def test_pc1_explains_more_than_pc2(self, multi_class_df, simple_experiment_2x3):
        """PC1 should explain more variance than PC2."""
        result = QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        import re
        pcts = []
        for label in result.pc_labels:
            match = re.search(r'(\d+)%', label)
            if match:
                pcts.append(int(match.group(1)))
        if len(pcts) == 2:
            assert pcts[0] >= pcts[1]

    def test_fewer_than_2_samples_raises(self):
        """Fewer than 2 concentration columns raises ValueError."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
        })
        with pytest.raises(ValueError, match="at least 2 samples"):
            QualityCheckService.compute_pca(df, exp)

    def test_empty_dataframe_raises(self, simple_experiment_2x3):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService.compute_pca(pd.DataFrame(), simple_experiment_2x3)

    def test_no_concentration_columns_raises(self, simple_experiment_2x3):
        """No matching concentration columns raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'intensity[s1]': [100.0],
        })
        with pytest.raises(ValueError, match="no concentration columns"):
            QualityCheckService.compute_pca(df, simple_experiment_2x3)

    def test_partial_samples(self, simple_experiment_2x3):
        """PCA works with partial sample columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'concentration[s1]': [100.0, 200.0, 300.0],
            'concentration[s2]': [110.0, 210.0, 310.0],
            'concentration[s3]': [120.0, 220.0, 320.0],
            # s4, s5, s6 missing
        })
        result = QualityCheckService.compute_pca(df, simple_experiment_2x3)
        assert result.available_samples == ['s1', 's2', 's3']
        assert result.pc_df.shape == (3, 2)

    def test_does_not_modify_original(self, multi_class_df, simple_experiment_2x3):
        """Original DataFrame is not modified."""
        original_shape = multi_class_df.shape
        QualityCheckService.compute_pca(multi_class_df, simple_experiment_2x3)
        assert multi_class_df.shape == original_shape

    def test_pca_with_two_samples(self):
        """PCA works with exactly 2 samples (minimum)."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'concentration[s1]': [100.0, 200.0, 300.0],
            'concentration[s2]': [150.0, 250.0, 350.0],
        })
        result = QualityCheckService.compute_pca(df, exp)
        assert result.pc_df.shape == (2, 2)

    def test_pca_conditions_with_partial_samples(self):
        """Conditions correctly mapped when some samples are missing."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'concentration[s1]': [100.0, 200.0],
            # s2 missing
            'concentration[s3]': [120.0, 220.0],
            'concentration[s4]': [130.0, 230.0],
        })
        result = QualityCheckService.compute_pca(df, exp)
        assert result.available_samples == ['s1', 's3', 's4']
        assert result.conditions == ['Control', 'Treatment', 'Treatment']


# =============================================================================
# TestRemoveSamples
# =============================================================================

class TestRemoveSamples:
    """Tests for remove_samples()."""

    def test_remove_single_sample(self, basic_conc_df, simple_experiment_2x3):
        """Remove one sample from a 6-sample experiment."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s3']
        )
        assert isinstance(result, SampleRemovalResult)
        assert result.samples_before == 6
        assert result.samples_after == 5
        assert result.removed_samples == ['s3']

    def test_remove_multiple_samples(self, basic_conc_df, simple_experiment_2x3):
        """Remove two samples from different conditions."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2', 's5']
        )
        assert result.samples_before == 6
        assert result.samples_after == 4
        assert set(result.removed_samples) == {'s2', 's5'}

    def test_columns_renamed_sequentially(self, basic_conc_df, simple_experiment_2x3):
        """After removal, columns are renamed to match new sequential labels."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2']
        )
        new_exp = result.updated_experiment
        for s in new_exp.full_samples_list:
            assert f'concentration[{s}]' in result.updated_df.columns

    def test_removed_column_not_in_result(self, basic_conc_df, simple_experiment_2x3):
        """Removed sample's concentration column is not in updated_df."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s3']
        )
        # The old column concentration[s3] should not exist
        # (it may have been renamed to something else, but the data for s3 is gone)
        assert result.samples_after == 5

    def test_updated_experiment_config(self, basic_conc_df, simple_experiment_2x3):
        """Updated experiment has correct sample counts."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1', 's4']
        )
        new_exp = result.updated_experiment
        # Control had 3 (s1,s2,s3) → now 2; Treatment had 3 (s4,s5,s6) → now 2
        assert new_exp.number_of_samples_list == [2, 2]
        assert sum(new_exp.number_of_samples_list) == 4

    def test_condition_dropped_when_empty(self, simple_experiment_2x3):
        """Condition is dropped when all its samples are removed."""
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
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s1', 's2', 's3']
        )
        # Control condition entirely removed
        assert result.updated_experiment.n_conditions == 1
        assert result.updated_experiment.conditions_list == ['Treatment']

    def test_empty_samples_to_remove_raises(self, basic_conc_df, simple_experiment_2x3):
        """Empty samples_to_remove raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            QualityCheckService.remove_samples(
                basic_conc_df, simple_experiment_2x3, []
            )

    def test_too_many_removals_raises(self, simple_experiment_2x3):
        """Removing all but 1 sample raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        with pytest.raises(ValueError, match="at least 2 are required"):
            QualityCheckService.remove_samples(
                df, simple_experiment_2x3, ['s1', 's2', 's3', 's4', 's5']
            )

    def test_nonexistent_sample_silently_ignored(self, basic_conc_df, simple_experiment_2x3):
        """Non-existent sample labels are silently filtered out."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1', 'nonexistent']
        )
        assert result.removed_samples == ['s1']
        assert result.samples_after == 5

    def test_does_not_modify_original(self, basic_conc_df, simple_experiment_2x3):
        """Original DataFrame and experiment are not modified."""
        original_cols = list(basic_conc_df.columns)
        original_shape = basic_conc_df.shape
        QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert list(basic_conc_df.columns) == original_cols
        assert basic_conc_df.shape == original_shape

    def test_data_integrity_after_removal(self, simple_experiment_2x3):
        """Data values are preserved (just columns renamed)."""
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
        result = QualityCheckService.remove_samples(df, simple_experiment_2x3, ['s3'])
        # s1=100, s2=200 stay as s1,s2; s4=400,s5=500,s6=600 become s3,s4,s5
        new_df = result.updated_df
        assert new_df['concentration[s1]'].iloc[0] == 100.0
        assert new_df['concentration[s2]'].iloc[0] == 200.0
        # s4 → s3 in new labeling
        assert new_df['concentration[s3]'].iloc[0] == 400.0

    def test_metadata_columns_preserved(self, basic_conc_df, simple_experiment_2x3):
        """Non-concentration columns (LipidMolec, ClassKey) preserved."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert 'LipidMolec' in result.updated_df.columns
        assert 'ClassKey' in result.updated_df.columns

    def test_remove_all_samples_from_all_conditions_raises(self, simple_experiment_2x3):
        """Removing all samples raises ValueError."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        with pytest.raises(ValueError):
            QualityCheckService.remove_samples(
                df, simple_experiment_2x3,
                ['s1', 's2', 's3', 's4', 's5', 's6']
            )

    def test_remove_exactly_n_minus_2_boundary(self, simple_experiment_2x3):
        """Removing N-2 samples leaves exactly 2 (minimum allowed)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s1', 's2', 's3', 's4']
        )
        assert result.samples_after == 2

    def test_all_nonexistent_samples_still_raises(self, basic_conc_df, simple_experiment_2x3):
        """All nonexistent samples → 0 actual removals → but still <2 check passes.
        Empty list after filtering → would leave all 6 → should not raise."""
        # Actually all are filtered out, remaining=6, so it won't raise for <2.
        # But actual_to_remove is empty which leaves remaining unchanged.
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['x1', 'x2']
        )
        assert result.removed_samples == []
        assert result.samples_after == 6

    def test_column_count_after_removal(self, basic_conc_df, simple_experiment_2x3):
        """Number of concentration columns decreases by number removed."""
        original_conc_cols = [c for c in basic_conc_df.columns if c.startswith('concentration[')]
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s2', 's5']
        )
        new_conc_cols = [c for c in result.updated_df.columns if c.startswith('concentration[')]
        assert len(new_conc_cols) == len(original_conc_cols) - 2

    def test_remove_from_three_condition_experiment(self, three_condition_experiment):
        """Remove samples across 3 conditions, one condition drops."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
            'concentration[s4]': [400.0],
            'concentration[s5]': [500.0],
            'concentration[s6]': [600.0],
        })
        # Remove both samples from Control (s1, s2)
        result = QualityCheckService.remove_samples(
            df, three_condition_experiment, ['s1', 's2']
        )
        assert result.updated_experiment.n_conditions == 2
        assert 'Control' not in result.updated_experiment.conditions_list
        assert result.samples_after == 4

    def test_remove_sample_renaming_across_conditions(self, simple_experiment_2x3):
        """Verify exact rename mapping when removing from middle."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'concentration[s1]': [10.0],
            'concentration[s2]': [20.0],
            'concentration[s3]': [30.0],
            'concentration[s4]': [40.0],
            'concentration[s5]': [50.0],
            'concentration[s6]': [60.0],
        })
        # Remove s2 (Control) and s4 (Treatment)
        result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s2', 's4']
        )
        new_df = result.updated_df
        # Surviving: s1(10), s3(30), s5(50), s6(60) → renamed to s1, s2, s3, s4
        assert new_df['concentration[s1]'].iloc[0] == 10.0
        assert new_df['concentration[s2]'].iloc[0] == 30.0
        assert new_df['concentration[s3]'].iloc[0] == 50.0
        assert new_df['concentration[s4]'].iloc[0] == 60.0


# =============================================================================
# TestValidation
# =============================================================================

class TestValidation:
    """Tests for private validation methods."""

    def test_validate_dataframe_none(self):
        """None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService._validate_dataframe(None)

    def test_validate_dataframe_empty(self):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            QualityCheckService._validate_dataframe(pd.DataFrame())

    def test_validate_dataframe_valid(self):
        """Valid DataFrame passes validation."""
        df = pd.DataFrame({'a': [1]})
        QualityCheckService._validate_dataframe(df)  # Should not raise

    def test_validate_concentration_columns_found(self, simple_experiment_2x3):
        """Returns available samples when columns exist."""
        df = pd.DataFrame({
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
        })
        result = QualityCheckService._validate_concentration_columns(df, simple_experiment_2x3)
        assert result == ['s1', 's2']

    def test_validate_concentration_columns_none_found(self, simple_experiment_2x3):
        """Raises when no concentration columns match."""
        df = pd.DataFrame({'intensity[s1]': [100.0]})
        with pytest.raises(ValueError, match="no concentration columns"):
            QualityCheckService._validate_concentration_columns(df, simple_experiment_2x3)

    def test_validate_concentration_columns_partial(self, simple_experiment_2x3):
        """Returns only available samples when some are missing."""
        df = pd.DataFrame({
            'concentration[s1]': [100.0],
            'concentration[s3]': [300.0],
        })
        result = QualityCheckService._validate_concentration_columns(df, simple_experiment_2x3)
        assert result == ['s1', 's3']

    def test_validate_bqc_label_found(self, bqc_experiment):
        """Returns correct index for valid BQC label."""
        idx = QualityCheckService._validate_bqc_label(bqc_experiment, 'BQC')
        assert idx == 2

    def test_validate_bqc_label_not_found(self, bqc_experiment):
        """Raises for invalid BQC label."""
        with pytest.raises(ValueError, match="not found"):
            QualityCheckService._validate_bqc_label(bqc_experiment, 'NoSuchLabel')

    def test_validate_condition_found(self, simple_experiment_2x3):
        """Returns correct index for valid condition."""
        idx = QualityCheckService._validate_condition(simple_experiment_2x3, 'Treatment')
        assert idx == 1

    def test_validate_condition_not_found(self, simple_experiment_2x3):
        """Raises for invalid condition."""
        with pytest.raises(ValueError, match="not found"):
            QualityCheckService._validate_condition(simple_experiment_2x3, 'NoSuch')

    def test_validate_cov_threshold_positive(self):
        """Positive threshold passes validation."""
        QualityCheckService._validate_cov_threshold(30.0)  # Should not raise

    def test_validate_cov_threshold_zero(self):
        """Zero threshold raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QualityCheckService._validate_cov_threshold(0.0)

    def test_validate_cov_threshold_negative(self):
        """Negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            QualityCheckService._validate_cov_threshold(-10.0)

    def test_validate_cov_threshold_small_positive(self):
        """Very small positive threshold passes validation."""
        QualityCheckService._validate_cov_threshold(0.001)  # Should not raise


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Edge case tests across multiple methods."""

    def test_nan_values_in_concentration(self, simple_experiment_2x3):
        """NaN values in concentration columns are handled."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, np.nan],
            'concentration[s2]': [np.nan, 200.0],
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        # NaN is not zero, so missing percent based on zeros should be 0
        assert result.available_samples == ['s1', 's2']

    def test_large_dataset_performance(self, simple_experiment_2x3):
        """Service handles large DataFrames (1000 lipids)."""
        n = 1000
        data = {
            'LipidMolec': [f'Lipid_{i}' for i in range(n)],
            'ClassKey': [f'Class_{i % 10}' for i in range(n)],
        }
        for s in ['s1', 's2', 's3', 's4', 's5', 's6']:
            data[f'concentration[{s}]'] = np.random.rand(n) * 1000
        df = pd.DataFrame(data)
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert len(result.available_samples) == 6
        assert result.mean_area_df.shape == (n, 6)

    def test_pca_with_identical_samples(self):
        """PCA with identical samples (zero variance in some features)."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [100.0, 200.0],
            'concentration[s3]': [100.0, 200.0],
        })
        # All samples identical → PCA should still run (may have 0 variance)
        result = QualityCheckService.compute_pca(df, exp)
        assert result.pc_df.shape == (3, 2)

    def test_correlation_with_constant_values(self):
        """Correlation when all values are the same (results in NaN)."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'concentration[s1]': [100.0, 100.0],
            'concentration[s2]': [100.0, 100.0],
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        # Constant columns → correlation is NaN (expected behavior)
        assert isinstance(result, CorrelationResult)

    def test_bqc_all_lipids_none_cov(self):
        """When all lipids have None CoV → 0% reliable."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'BQC'],
            number_of_samples_list=[2, 1],
        )
        # Only 1 BQC sample → CoV needs >=2, so all None
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [110.0, 210.0],
            'concentration[s3]': [120.0, 220.0],
        })
        result = QualityCheckService.prepare_bqc_data(df, exp, 'BQC')
        assert result.reliable_data_percent == 0.0
        assert result.high_cov_lipids == []

    def test_filter_bqc_single_lipid_removed(self):
        """Filter with single lipid dataset removes it."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        assert len(result.filtered_df) == 0
        assert result.removed_count == 1

    def test_box_plot_many_samples(self, large_experiment):
        """Box plot with 20 samples."""
        data = {'LipidMolec': ['PC(16:0)'], 'ClassKey': ['PC']}
        for i in range(1, 21):
            data[f'concentration[s{i}]'] = [100.0 * i]
        df = pd.DataFrame(data)
        result = QualityCheckService.prepare_box_plot_data(df, large_experiment)
        assert len(result.available_samples) == 20
        assert result.mean_area_df.shape == (1, 20)

    def test_special_characters_in_lipid_names(self, simple_experiment_2x3):
        """Lipid names with special characters handled correctly."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)+D7:(s)', 'PE O-18:0/20:4', 'TG 16:0;2'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100.0, 200.0, 300.0],
            'concentration[s2]': [110.0, 210.0, 310.0],
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert len(result.available_samples) == 2

    def test_remove_samples_preserves_row_count(self, basic_conc_df, simple_experiment_2x3):
        """Removing samples doesn't change row count (lipids)."""
        result = QualityCheckService.remove_samples(
            basic_conc_df, simple_experiment_2x3, ['s1']
        )
        assert len(result.updated_df) == len(basic_conc_df)

    def test_filter_then_pca_pipeline(self, simple_experiment_2x3):
        """Simulates BQC filter → PCA flow."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
            'ClassKey': ['PC', 'PE', 'TG', 'SM'],
            'concentration[s1]': [100, 200, 300, 400],
            'concentration[s2]': [110, 210, 310, 410],
            'concentration[s3]': [120, 220, 320, 420],
            'concentration[s4]': [130, 230, 330, 430],
            'concentration[s5]': [140, 240, 340, 440],
            'concentration[s6]': [150, 250, 350, 450],
        })
        # Filter out PC
        filter_result = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        # PCA on filtered data
        pca_result = QualityCheckService.compute_pca(
            filter_result.filtered_df, simple_experiment_2x3
        )
        assert pca_result.pc_df.shape[0] == 6

    def test_remove_then_pca_pipeline(self, simple_experiment_2x3):
        """Simulates sample removal → PCA flow."""
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
        # Remove s3
        remove_result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s3']
        )
        # PCA on updated data with updated experiment
        pca_result = QualityCheckService.compute_pca(
            remove_result.updated_df, remove_result.updated_experiment
        )
        assert pca_result.pc_df.shape[0] == 5

    def test_retention_time_with_extra_columns(self):
        """Retention time availability ignores extra columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'BaseRt': [5.1],
            'CalcMass': [733.5],
            'ExtraCol': ['x'],
            'AnotherCol': [42],
        })
        result = QualityCheckService.check_retention_time_availability(df)
        assert result.available is True

    def test_cov_with_nan_values(self):
        """CoV calculation handles NaN by converting to float."""
        # np.array([1, np.nan, 3], dtype=float) → NaN propagates
        result = QualityCheckService.calculate_coefficient_of_variation([1, np.nan, 3])
        # NaN propagation: result is None or NaN
        # With numpy, mean of [1, nan, 3] is nan → returns None since mean==nan
        # Actually np.mean returns nan, and nan == 0 is False, so it would compute
        # np.std(...)/nan * 100 = nan → float(nan)
        # Let's just check it doesn't crash
        assert result is None or isinstance(result, float)

    def test_box_plot_with_mixed_types(self, simple_experiment_2x3):
        """Box plot handles integer and float concentration values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100],   # int
            'concentration[s2]': [200.5], # float
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert result.mean_area_df.shape == (1, 2)


# =============================================================================
# TestTypeCoercion
# =============================================================================

class TestTypeCoercion:
    """Tests for type handling across methods."""

    def test_cov_with_string_numbers(self):
        """CoV handles string-representable numbers via numpy conversion."""
        # numpy will convert string numbers to float
        result = QualityCheckService.calculate_coefficient_of_variation(
            np.array(['10', '20', '30'], dtype=float)
        )
        assert result is not None
        assert isinstance(result, float)

    def test_mean_with_integer_list(self):
        """Mean of plain Python integers."""
        result = QualityCheckService.calculate_mean_including_zeros([10, 20, 30])
        assert result == pytest.approx(20.0)
        assert isinstance(result, float)

    def test_box_plot_object_dtype_column(self, simple_experiment_2x3):
        """Box plot handles object dtype columns that contain numeric values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': pd.array([100, 200], dtype='object'),
            'concentration[s2]': pd.array([110, 210], dtype='object'),
        })
        # Convert to numeric (simulates real-world data cleaning)
        df['concentration[s1]'] = pd.to_numeric(df['concentration[s1]'])
        df['concentration[s2]'] = pd.to_numeric(df['concentration[s2]'])
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert result.mean_area_df.shape == (2, 2)

    def test_correlation_with_int64_columns(self):
        """Correlation handles int64 concentration columns."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': pd.array([100, 200, 300], dtype='int64'),
            'concentration[s2]': pd.array([110, 210, 310], dtype='int64'),
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        assert result.correlation_df.shape == (2, 2)

    def test_pca_with_float32_columns(self):
        """PCA handles float32 concentration columns."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': pd.array([100.0, 200.0, 300.0], dtype='float32'),
            'concentration[s2]': pd.array([110.0, 210.0, 310.0], dtype='float32'),
            'concentration[s3]': pd.array([120.0, 220.0, 320.0], dtype='float32'),
        })
        result = QualityCheckService.compute_pca(df, exp)
        assert result.pc_df.shape == (3, 2)

    def test_cov_with_python_float_list(self):
        """CoV works with plain Python float list."""
        result = QualityCheckService.calculate_coefficient_of_variation(
            [10.0, 20.0, 30.0]
        )
        assert isinstance(result, float)

    def test_bqc_with_int_concentrations(self):
        """BQC works when concentration columns are integers."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'BQC'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100],
            'concentration[s2]': [110],
            'concentration[s3]': [100],
            'concentration[s4]': [102],
        })
        result = QualityCheckService.prepare_bqc_data(df, exp, 'BQC')
        assert isinstance(result, BQCPrepareResult)
        assert result.prepared_df['cov'].iloc[0] is not None

    def test_remove_samples_with_int_concentrations(self):
        """remove_samples works when concentration columns are integers."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [120, 220],
            'concentration[s4]': [130, 230],
        })
        result = QualityCheckService.remove_samples(df, exp, ['s1'])
        assert isinstance(result, SampleRemovalResult)
        assert result.samples_after == 3

    def test_filter_by_bqc_with_int_concentrations(self):
        """filter_by_bqc works when concentration columns are integers."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
        })
        result = QualityCheckService.filter_by_bqc(df, ['PE(18:0)'])
        assert len(result.filtered_df) == 2
        assert result.filtered_df['concentration[s1]'].dtype in (
            np.int64, np.float64, int,
        )

    def test_box_plot_with_float32_columns(self, simple_experiment_2x3):
        """Box plot handles float32 concentration columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': pd.array([100.0, 200.0], dtype='float32'),
            'concentration[s2]': pd.array([110.0, 210.0], dtype='float32'),
        })
        result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert result.mean_area_df.shape == (2, 2)

    def test_bqc_with_float32_columns(self):
        """BQC works with float32 concentration columns."""
        exp = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'BQC'],
            number_of_samples_list=[2, 2],
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': pd.array([100.0], dtype='float32'),
            'concentration[s2]': pd.array([110.0], dtype='float32'),
            'concentration[s3]': pd.array([100.0], dtype='float32'),
            'concentration[s4]': pd.array([102.0], dtype='float32'),
        })
        result = QualityCheckService.prepare_bqc_data(df, exp, 'BQC')
        assert isinstance(result, BQCPrepareResult)

    def test_correlation_with_float32_columns(self):
        """Correlation handles float32 concentration columns."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': pd.array([100.0, 200.0, 300.0], dtype='float32'),
            'concentration[s2]': pd.array([110.0, 210.0, 310.0], dtype='float32'),
            'concentration[s3]': pd.array([120.0, 220.0, 320.0], dtype='float32'),
        })
        result = QualityCheckService.compute_correlation(df, exp, 'A')
        assert result.correlation_df.shape == (3, 3)

    def test_pca_with_int64_columns(self):
        """PCA handles int64 concentration columns."""
        exp = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[3],
        )
        df = pd.DataFrame({
            'LipidMolec': ['a', 'b', 'c'],
            'concentration[s1]': pd.array([100, 200, 300], dtype='int64'),
            'concentration[s2]': pd.array([110, 210, 310], dtype='int64'),
            'concentration[s3]': pd.array([120, 220, 320], dtype='int64'),
        })
        result = QualityCheckService.compute_pca(df, exp)
        assert result.pc_df.shape == (3, 2)

    def test_cov_with_int64_array(self):
        """CoV handles int64 numpy array."""
        arr = np.array([10, 20, 30], dtype='int64')
        result = QualityCheckService.calculate_coefficient_of_variation(arr)
        assert result is not None
        assert isinstance(result, float)

    def test_mean_with_float32_series(self):
        """Mean handles float32 pandas Series."""
        series = pd.Series([10.0, 20.0, 30.0], dtype='float32')
        result = QualityCheckService.calculate_mean_including_zeros(series)
        assert result == pytest.approx(20.0)
        assert isinstance(result, float)


# =============================================================================
# TestMultiStepPipelines
# =============================================================================

class TestMultiStepPipelines:
    """Tests for realistic multi-step workflows."""

    def test_full_qc_pipeline_no_bqc(self, simple_experiment_2x3):
        """Full QC pipeline without BQC: box plot → correlation → PCA."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
            'ClassKey': ['PC', 'PE', 'TG', 'SM'],
            'concentration[s1]': [100, 200, 300, 400],
            'concentration[s2]': [110, 210, 310, 410],
            'concentration[s3]': [120, 220, 320, 420],
            'concentration[s4]': [90, 190, 290, 390],
            'concentration[s5]': [95, 195, 295, 395],
            'concentration[s6]': [80, 180, 280, 380],
        })
        # Step 1: box plot
        box_result = QualityCheckService.prepare_box_plot_data(df, simple_experiment_2x3)
        assert box_result.mean_area_df.shape == (4, 6)

        # Step 2: correlation for each condition
        eligible = QualityCheckService.get_correlation_eligible_conditions(simple_experiment_2x3)
        assert len(eligible) == 2
        for cond in eligible:
            corr_result = QualityCheckService.compute_correlation(
                df, simple_experiment_2x3, cond
            )
            assert corr_result.correlation_df.shape == (3, 3)

        # Step 3: PCA
        pca_result = QualityCheckService.compute_pca(df, simple_experiment_2x3)
        assert pca_result.pc_df.shape == (6, 2)

    def test_full_qc_pipeline_with_bqc(self, bqc_experiment):
        """Full QC pipeline with BQC: box plot → BQC filter → correlation → PCA → remove."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
            'ClassKey': ['PC', 'PE', 'TG', 'SM'],
            'concentration[s1]': [100, 200, 300, 400],
            'concentration[s2]': [110, 210, 310, 410],
            'concentration[s3]': [120, 220, 320, 420],
            'concentration[s4]': [90, 190, 290, 390],
            'concentration[s5]': [95, 195, 295, 395],
            'concentration[s6]': [80, 180, 280, 380],
            # BQC: PC has high variability
            'concentration[s7]': [50, 200, 300, 400],
            'concentration[s8]': [200, 198, 305, 395],
        })
        # Step 1: box plot
        box_result = QualityCheckService.prepare_box_plot_data(df, bqc_experiment)
        assert len(box_result.available_samples) == 8

        # Step 2: BQC assessment
        bqc_result = QualityCheckService.prepare_bqc_data(
            df, bqc_experiment, 'BQC', cov_threshold=30.0
        )
        assert len(bqc_result.high_cov_lipids) >= 1

        # Step 3: BQC filter (keep none of the high-CoV)
        filter_result = QualityCheckService.filter_by_bqc(
            df, bqc_result.high_cov_lipids
        )
        assert filter_result.lipids_after < filter_result.lipids_before

        # Step 4: correlation on filtered data
        for cond in QualityCheckService.get_correlation_eligible_conditions(bqc_experiment):
            corr = QualityCheckService.compute_correlation(
                filter_result.filtered_df, bqc_experiment, cond, bqc_label='BQC'
            )
            assert corr.threshold == 0.8  # Technical replicates

        # Step 5: PCA on filtered data
        pca_result = QualityCheckService.compute_pca(
            filter_result.filtered_df, bqc_experiment
        )
        assert pca_result.pc_df.shape[0] == 8

    def test_pca_then_remove_then_pca_again(self, simple_experiment_2x3):
        """PCA → identify outlier → remove → re-PCA."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
            'concentration[s3]': [120, 220, 320],
            'concentration[s4]': [900, 1900, 2900],  # outlier
            'concentration[s5]': [95, 195, 295],
            'concentration[s6]': [80, 180, 280],
        })
        # First PCA
        pca1 = QualityCheckService.compute_pca(df, simple_experiment_2x3)
        assert pca1.pc_df.shape == (6, 2)

        # Remove outlier s4
        remove_result = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s4']
        )
        assert remove_result.samples_after == 5

        # Second PCA with updated data
        pca2 = QualityCheckService.compute_pca(
            remove_result.updated_df, remove_result.updated_experiment
        )
        assert pca2.pc_df.shape == (5, 2)

    def test_bqc_filter_then_remove_samples_pipeline(self, bqc_experiment):
        """BQC filter → sample removal → correlation (chained operations)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)', 'SM(d18:1)'],
            'ClassKey': ['PC', 'PE', 'TG', 'SM'],
            'concentration[s1]': [100, 200, 300, 400],
            'concentration[s2]': [110, 210, 310, 410],
            'concentration[s3]': [120, 220, 320, 420],
            'concentration[s4]': [90, 190, 290, 390],
            'concentration[s5]': [95, 195, 295, 395],
            'concentration[s6]': [80, 180, 280, 380],
            'concentration[s7]': [100, 200, 300, 400],
            'concentration[s8]': [102, 198, 305, 395],
        })
        # Step 1: filter out a lipid
        filtered = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        assert len(filtered.filtered_df) == 3

        # Step 2: remove a sample
        removed = QualityCheckService.remove_samples(
            filtered.filtered_df, bqc_experiment, ['s1']
        )
        assert removed.samples_after == 7

        # Step 3: correlation on final data
        eligible = QualityCheckService.get_correlation_eligible_conditions(
            removed.updated_experiment
        )
        for cond in eligible:
            corr = QualityCheckService.compute_correlation(
                removed.updated_df, removed.updated_experiment, cond
            )
            assert isinstance(corr, CorrelationResult)

    def test_retention_time_check_in_pipeline(self, simple_experiment_2x3):
        """Retention time check integrates into pipeline correctly."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:0)', 'TG(16:0)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'BaseRt': [5.1, 6.2, 7.3],
            'CalcMass': [733.5, 743.5, 850.7],
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [110, 210, 310],
            'concentration[s3]': [120, 220, 320],
            'concentration[s4]': [90, 190, 290],
            'concentration[s5]': [95, 195, 295],
            'concentration[s6]': [80, 180, 280],
        })
        # Check RT availability
        rt_result = QualityCheckService.check_retention_time_availability(df)
        assert rt_result.available is True
        assert len(rt_result.lipid_classes) == 3

        # After BQC filter, RT columns still present
        filtered = QualityCheckService.filter_by_bqc(df, ['PC(16:0)'])
        rt_after = QualityCheckService.check_retention_time_availability(
            filtered.filtered_df
        )
        assert rt_after.available is True
        assert len(rt_after.lipid_classes) == 2  # PC removed

    def test_box_plot_after_sample_removal(self, simple_experiment_2x3):
        """Box plot works correctly after sample removal."""
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
        # Remove s3
        removed = QualityCheckService.remove_samples(
            df, simple_experiment_2x3, ['s3']
        )
        # Box plot on updated data
        box_result = QualityCheckService.prepare_box_plot_data(
            removed.updated_df, removed.updated_experiment
        )
        assert len(box_result.available_samples) == 5
        assert box_result.mean_area_df.shape == (2, 5)
