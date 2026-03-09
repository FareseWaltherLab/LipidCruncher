"""Tests for BQCPlotterService — CoV scatter plots for BQC quality assessment."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.bqc_plotter import (
    BQCPlotterService,
    _calculate_coefficient_of_variation,
    _calculate_mean_including_zeros,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def bqc_df():
    """DataFrame with concentration columns for BQC testing."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:1)', 'SM(18:0)', 'TG(16:0)', 'CE(18:2)'],
        'ClassKey': ['PC', 'PE', 'SM', 'TG', 'CE'],
        'concentration[bqc1]': [100.0, 200.0, 300.0, 0.0, 500.0],
        'concentration[bqc2]': [110.0, 190.0, 0.0, 420.0, 490.0],
        'concentration[bqc3]': [105.0, 210.0, 310.0, 410.0, 510.0],
    })


@pytest.fixture
def simple_experiment():
    """Experiment-like object with individual_samples_list."""
    class Exp:
        individual_samples_list = [['s1', 's2', 's3'], ['bqc1', 'bqc2', 'bqc3']]
    return Exp()


# =============================================================================
# TestCoVCalculation
# =============================================================================

class TestCoVCalculation:
    def test_known_values(self):
        # [100, 200, 300]: mean=200, std=100 → CoV=50%
        result = _calculate_coefficient_of_variation([100, 200, 300])
        assert result == pytest.approx(50.0)

    def test_identical_values(self):
        result = _calculate_coefficient_of_variation([5, 5, 5])
        assert result == pytest.approx(0.0)

    def test_includes_zeros(self):
        result = _calculate_coefficient_of_variation([0, 100, 200])
        assert result is not None
        assert result > 0

    def test_all_zeros(self):
        result = _calculate_coefficient_of_variation([0, 0, 0])
        assert result is None  # mean=0, undefined

    def test_single_value(self):
        result = _calculate_coefficient_of_variation([100])
        assert result is None

    def test_empty(self):
        result = _calculate_coefficient_of_variation([])
        assert result is None

    def test_two_values(self):
        result = _calculate_coefficient_of_variation([100, 200])
        assert result is not None

    def test_uses_sample_std(self):
        # ddof=1: [100, 200] → mean=150, std=70.71 → CoV=47.14%
        result = _calculate_coefficient_of_variation([100, 200])
        expected = np.std([100, 200], ddof=1) / np.mean([100, 200]) * 100
        assert result == pytest.approx(expected)

    def test_numpy_input(self):
        result = _calculate_coefficient_of_variation(np.array([100, 200, 300]))
        assert result == pytest.approx(50.0)

    def test_negative_values(self):
        result = _calculate_coefficient_of_variation([-100, 100])
        # mean=0 → None
        assert result is None


# =============================================================================
# TestMeanCalculation
# =============================================================================

class TestMeanCalculation:
    def test_known_values(self):
        result = _calculate_mean_including_zeros([100, 200, 300])
        assert result == pytest.approx(200.0)

    def test_includes_zeros(self):
        result = _calculate_mean_including_zeros([0, 100, 200])
        assert result == pytest.approx(100.0)

    def test_single_value(self):
        result = _calculate_mean_including_zeros([100])
        assert result is None

    def test_empty(self):
        result = _calculate_mean_including_zeros([])
        assert result is None

    def test_all_zeros(self):
        result = _calculate_mean_including_zeros([0, 0, 0])
        assert result == pytest.approx(0.0)

    def test_two_values(self):
        result = _calculate_mean_including_zeros([100, 200])
        assert result == pytest.approx(150.0)


# =============================================================================
# TestPrepareDataframeForPlot
# =============================================================================

class TestPrepareDataframeForPlot:
    def test_adds_cov_column(self, bqc_df):
        result = BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert 'cov' in result.columns

    def test_adds_mean_column(self, bqc_df):
        result = BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert 'mean' in result.columns

    def test_mean_is_log10_transformed(self, bqc_df):
        result = BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        # PC: mean of [100, 110, 105] = 105 → log10(105) ≈ 2.021
        pc_mean = result.iloc[0]['mean']
        assert pc_mean == pytest.approx(np.log10(105), abs=0.01)

    def test_preserves_original_columns(self, bqc_df):
        result = BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns

    def test_does_not_modify_input(self, bqc_df):
        original = bqc_df.copy()
        BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        pd.testing.assert_frame_equal(bqc_df, original)

    def test_row_count_preserved(self, bqc_df):
        result = BQCPlotterService.prepare_dataframe_for_plot(
            bqc_df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert len(result) == len(bqc_df)


# =============================================================================
# TestGenerateCovPlotData
# =============================================================================

class TestGenerateCovPlotData:
    def test_returns_dataframe(self, bqc_df):
        ind_samples = [['s1'], ['bqc1', 'bqc2', 'bqc3']]
        result = BQCPlotterService.generate_cov_plot_data(bqc_df, ind_samples, 1)
        assert isinstance(result, pd.DataFrame)
        assert 'cov' in result.columns
        assert 'mean' in result.columns

    def test_uses_correct_bqc_index(self, bqc_df):
        ind_samples = [['bqc1', 'bqc2', 'bqc3'], ['other1']]
        result = BQCPlotterService.generate_cov_plot_data(bqc_df, ind_samples, 0)
        assert 'cov' in result.columns


# =============================================================================
# TestGenerateAndDisplayCovPlot
# =============================================================================

class TestGenerateAndDisplayCovPlot:
    def test_returns_tuple_of_four(self, bqc_df, simple_experiment):
        result = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1, cov_threshold=30
        )
        assert len(result) == 4

    def test_returns_figure(self, bqc_df, simple_experiment):
        fig, _, _, _ = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1
        )
        assert isinstance(fig, go.Figure)

    def test_returns_prepared_df(self, bqc_df, simple_experiment):
        _, prepared_df, _, _ = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1
        )
        assert isinstance(prepared_df, pd.DataFrame)
        assert 'cov' in prepared_df.columns

    def test_returns_reliable_percentage(self, bqc_df, simple_experiment):
        _, _, pct, _ = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1, cov_threshold=200
        )
        assert isinstance(pct, float)
        assert 0 <= pct <= 100

    def test_high_threshold_all_reliable(self, bqc_df, simple_experiment):
        _, _, pct, _ = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1, cov_threshold=9999
        )
        assert pct == 100.0

    def test_low_threshold_low_reliability(self, bqc_df, simple_experiment):
        _, _, pct, _ = BQCPlotterService.generate_and_display_cov_plot(
            bqc_df, simple_experiment, 1, cov_threshold=0.001
        )
        assert pct < 50


# =============================================================================
# TestCreateCovScatterPlotWithThreshold
# =============================================================================

class TestCreateCovScatterPlotWithThreshold:
    def test_returns_figure_and_dataframe(self):
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 50.0, 90.0]),
            np.array(['A', 'B', 'C']),
            cov_threshold=30,
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(df, pd.DataFrame)

    def test_filters_nan_values(self):
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, np.nan, 3.0]),
            np.array([10.0, 50.0, np.nan]),
            np.array(['A', 'B', 'C']),
            cov_threshold=30,
        )
        assert len(df) == 1  # Only first point valid

    def test_below_threshold_blue(self):
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),  # All below 30
            np.array(['A', 'B']),
            cov_threshold=30,
        )
        # Should have 1 trace (all below) + no above trace
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1
        assert scatter_traces[0].marker.color == 'blue'

    def test_above_threshold_red(self):
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0]),
            np.array([50.0, 60.0]),  # All above 30
            np.array(['A', 'B']),
            cov_threshold=30,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1
        assert scatter_traces[0].marker.color == 'red'

    def test_mixed_threshold(self):
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 50.0, 90.0]),
            np.array(['A', 'B', 'C']),
            cov_threshold=30,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 2  # One blue, one red

    def test_has_threshold_line(self):
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0]), np.array([10.0]), np.array(['A']),
            cov_threshold=30,
        )
        # Check horizontal line exists in layout shapes
        shapes = fig.layout.shapes
        assert len(shapes) >= 1

    def test_white_background(self):
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0]), np.array([10.0]), np.array(['A']),
            cov_threshold=30,
        )
        assert fig.layout.plot_bgcolor == 'white'

    def test_empty_arrays(self):
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([]), np.array([]), np.array([]),
            cov_threshold=30,
        )
        assert isinstance(fig, go.Figure)
        assert len(df) == 0


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    def test_single_lipid(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [100.0],
            'concentration[bqc2]': [110.0],
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        assert len(result) == 1
        assert result.iloc[0]['cov'] is not None

    def test_large_dataset(self):
        n = 1000
        df = pd.DataFrame({
            'LipidMolec': [f'L{i}' for i in range(n)],
            'ClassKey': ['PC'] * n,
            'concentration[bqc1]': np.random.rand(n) * 1000,
            'concentration[bqc2]': np.random.rand(n) * 1000,
            'concentration[bqc3]': np.random.rand(n) * 1000,
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert len(result) == n

    def test_all_zeros_in_lipid(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [0.0],
            'concentration[bqc2]': [0.0],
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        # CoV undefined (mean=0), mean should not be log-transformed
        assert result.iloc[0]['cov'] is None

    def test_special_characters_in_lipid_names(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)+D7:(s)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [100.0],
            'concentration[bqc2]': [110.0],
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        assert result.iloc[0]['LipidMolec'] == 'PC(15:0_18:1)+D7:(s)'


# =============================================================================
# TestErrorHandling
# =============================================================================

class TestErrorHandling:
    def test_missing_concentration_column_raises(self, bqc_df):
        with pytest.raises(KeyError):
            BQCPlotterService.prepare_dataframe_for_plot(
                bqc_df, ['concentration[nonexistent]']
            )

    def test_generate_cov_plot_data_invalid_index(self, bqc_df):
        ind_samples = [['bqc1', 'bqc2', 'bqc3']]
        with pytest.raises(IndexError):
            BQCPlotterService.generate_cov_plot_data(bqc_df, ind_samples, 5)

    def test_generate_and_display_cov_plot_invalid_index(self, bqc_df, simple_experiment):
        with pytest.raises(IndexError):
            BQCPlotterService.generate_and_display_cov_plot(bqc_df, simple_experiment, 99)

    def test_all_nan_arrays_produce_empty_plot(self):
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            np.array(['A', 'B']),
            cov_threshold=30,
        )
        assert isinstance(fig, go.Figure)
        assert len(df) == 0


# =============================================================================
# TestTypeCoercion
# =============================================================================

class TestTypeCoercion:
    def test_integer_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(18:1)'],
            'ClassKey': ['PC', 'PE'],
            'concentration[bqc1]': [100, 200],
            'concentration[bqc2]': [110, 190],
        })
        assert df['concentration[bqc1]'].dtype == np.int64
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        assert result.iloc[0]['cov'] is not None

    def test_float32_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': np.array([100], dtype=np.float32),
            'concentration[bqc2]': np.array([110], dtype=np.float32),
            'concentration[bqc3]': np.array([105], dtype=np.float32),
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        assert result.iloc[0]['cov'] is not None
        assert result.iloc[0]['mean'] is not None

    def test_mixed_int_float_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [100],          # int
            'concentration[bqc2]': [110.5],         # float
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        assert result.iloc[0]['cov'] is not None

    def test_cov_with_list_input(self):
        result = _calculate_coefficient_of_variation([100.0, 200.0, 300.0])
        assert result == pytest.approx(50.0)

    def test_cov_with_pandas_series(self):
        result = _calculate_coefficient_of_variation(pd.Series([100.0, 200.0, 300.0]))
        assert result == pytest.approx(50.0)

    def test_mean_with_pandas_series(self):
        result = _calculate_mean_including_zeros(pd.Series([100.0, 200.0, 300.0]))
        assert result == pytest.approx(200.0)

    def test_scatter_plot_with_list_inputs(self):
        """Lists (not numpy arrays) should be converted internally."""
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            [1.0, 2.0, 3.0],
            [10.0, 50.0, 90.0],
            ['A', 'B', 'C'],
            cov_threshold=30,
        )
        assert isinstance(fig, go.Figure)
        assert len(df) == 3


# =============================================================================
# TestNaNHandling
# =============================================================================

class TestNaNHandling:
    def test_nan_in_concentration_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [np.nan],
            'concentration[bqc2]': [110.0],
            'concentration[bqc3]': [105.0],
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]', 'concentration[bqc3]']
        )
        # Row still present even if CoV/mean may be NaN
        assert len(result) == 1

    def test_scatter_plot_filters_partial_nan(self):
        """Only points where both mean and cov are valid should appear."""
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, np.nan, 3.0]),
            np.array([10.0, 20.0, np.nan]),
            np.array(['A', 'B', 'C']),
            cov_threshold=30,
        )
        assert len(df) == 1  # Only first point has both valid


# =============================================================================
# TestBoundary
# =============================================================================

class TestBoundary:
    def test_cov_threshold_zero(self):
        """Threshold of 0 means everything is above."""
        fig, df = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            np.array(['A', 'B']),
            cov_threshold=0,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        # All above threshold → only red trace
        assert len(scatter_traces) == 1
        assert scatter_traces[0].marker.color == 'red'

    def test_cov_threshold_very_high(self):
        """Very high threshold means everything is below."""
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            np.array(['A', 'B']),
            cov_threshold=99999,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1
        assert scatter_traces[0].marker.color == 'blue'

    def test_cov_exactly_at_threshold(self):
        """Value exactly at threshold should be <= (blue)."""
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            np.array([1.0]),
            np.array([30.0]),
            np.array(['A']),
            cov_threshold=30,
        )
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 1
        assert scatter_traces[0].marker.color == 'blue'

    def test_two_bqc_samples_minimum(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[bqc1]': [100.0],
            'concentration[bqc2]': [200.0],
        })
        result = BQCPlotterService.prepare_dataframe_for_plot(
            df, ['concentration[bqc1]', 'concentration[bqc2]']
        )
        assert result.iloc[0]['cov'] is not None
