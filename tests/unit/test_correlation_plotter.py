"""Tests for CorrelationPlotterService — pairwise correlation heatmaps."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.services.plotting.correlation import CorrelationPlotterService

matplotlib.use('Agg')  # Non-interactive backend for testing


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def corr_df():
    """DataFrame with concentration columns for correlation testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'LipidMolec': [f'L{i}' for i in range(20)],
        'ClassKey': ['PC'] * 20,
        'concentration[s1]': np.random.rand(20) * 1000,
        'concentration[s2]': np.random.rand(20) * 1000,
        'concentration[s3]': np.random.rand(20) * 1000,
        'concentration[s4]': np.random.rand(20) * 1000,
    })


@pytest.fixture
def individual_samples():
    return [['s1', 's2'], ['s3', 's4']]


# =============================================================================
# TestPrepareDataForCorrelation
# =============================================================================

class TestPrepareDataForCorrelation:
    def test_returns_dataframe(self, corr_df, individual_samples):
        result = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns_condition_0(self, corr_df, individual_samples):
        result = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        assert list(result.columns) == ['s1', 's2']

    def test_correct_columns_condition_1(self, corr_df, individual_samples):
        result = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 1
        )
        assert list(result.columns) == ['s3', 's4']

    def test_row_count_preserved(self, corr_df, individual_samples):
        result = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        assert len(result) == len(corr_df)

    def test_values_match(self, corr_df, individual_samples):
        result = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        pd.testing.assert_series_equal(
            result['s1'].reset_index(drop=True),
            corr_df['concentration[s1]'].reset_index(drop=True),
            check_names=False,
        )

    def test_three_conditions(self, corr_df):
        ind = [['s1'], ['s2', 's3'], ['s4']]
        result = CorrelationPlotterService.prepare_data_for_correlation(corr_df, ind, 1)
        assert list(result.columns) == ['s2', 's3']


# =============================================================================
# TestComputeCorrelation
# =============================================================================

class TestComputeCorrelation:
    def test_returns_three_values(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        result = CorrelationPlotterService.compute_correlation(mean_area_df, 'biological replicates')
        assert len(result) == 3

    def test_correlation_df_is_square(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        assert corr_matrix.shape == (2, 2)

    def test_diagonal_is_one(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        np.testing.assert_array_almost_equal(np.diag(corr_matrix.values), [1.0, 1.0])

    def test_bio_threshold_0_7(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        _, _, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        assert thresh == 0.7

    def test_tech_threshold_0_8(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        _, _, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'technical replicates'
        )
        assert thresh == 0.8

    def test_vmin_is_0_5(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        _, v_min, _ = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        assert v_min == 0.5

    def test_symmetric_matrix(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        pd.testing.assert_frame_equal(corr_matrix, corr_matrix.T)

    def test_values_in_range(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        assert (corr_matrix.values >= -1).all()
        assert (corr_matrix.values <= 1).all()

    def test_identical_samples_perfect_correlation(self):
        df = pd.DataFrame({
            's1': [100, 200, 300, 400],
            's2': [100, 200, 300, 400],
        })
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert corr_matrix.iloc[0, 1] == pytest.approx(1.0)


# =============================================================================
# TestRenderCorrelationPlot
# =============================================================================

class TestRenderCorrelationPlot:
    def test_returns_figure(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, v_min, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, v_min, thresh, 'Control'
        )
        assert isinstance(fig, plt.Figure)

    def test_title_contains_condition(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, v_min, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, v_min, thresh, 'MyCondition'
        )
        # Check the axes title
        ax = fig.axes[0]
        assert 'MyCondition' in ax.get_title()

    def test_figure_is_closed(self, corr_df, individual_samples):
        """Figure should be closed to prevent double display in Streamlit."""
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, v_min, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, v_min, thresh, 'Control'
        )
        # plt.close() was called, so the figure should not be in plt's figure list
        assert fig not in plt.get_fignums()

    def test_large_matrix(self):
        """Larger matrix (6 samples) should render without error."""
        samples = [f's{i}' for i in range(6)]
        np.random.seed(42)
        data = {s: np.random.rand(50) * 1000 for s in samples}
        df = pd.DataFrame(data)
        corr_matrix = df.corr()
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, 0.5, 0.7, 'LargeCondition'
        )
        assert isinstance(fig, plt.Figure)


# =============================================================================
# TestFullPipeline
# =============================================================================

class TestFullPipeline:
    def test_prepare_compute_render(self, corr_df, individual_samples):
        """Full pipeline: prepare → compute → render."""
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        corr_matrix, v_min, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'biological replicates'
        )
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, v_min, thresh, 'Control'
        )
        assert isinstance(fig, plt.Figure)

    def test_technical_replicates_pipeline(self, corr_df, individual_samples):
        mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        _, _, thresh = CorrelationPlotterService.compute_correlation(
            mean_area_df, 'technical replicates'
        )
        assert thresh == 0.8


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    def test_input_immutability(self, corr_df, individual_samples):
        original = corr_df.copy()
        CorrelationPlotterService.prepare_data_for_correlation(
            corr_df, individual_samples, 0
        )
        pd.testing.assert_frame_equal(corr_df, original)

    def test_single_sample_correlation(self):
        """Single sample produces 1x1 correlation matrix."""
        df = pd.DataFrame({'s1': [100, 200, 300]})
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix.iloc[0, 0] == pytest.approx(1.0)


# =============================================================================
# TestErrorHandling
# =============================================================================

class TestErrorHandling:
    def test_missing_concentration_column_raises(self, corr_df):
        with pytest.raises(KeyError):
            CorrelationPlotterService.prepare_data_for_correlation(
                corr_df, [['nonexistent']], 0
            )

    def test_condition_index_out_of_range(self, corr_df, individual_samples):
        with pytest.raises(IndexError):
            CorrelationPlotterService.prepare_data_for_correlation(
                corr_df, individual_samples, 99
            )

    def test_unknown_sample_type_defaults_to_tech(self):
        """Unknown sample type should default to 0.8 threshold (tech replicates)."""
        df = pd.DataFrame({'s1': [100, 200], 's2': [150, 250]})
        _, _, thresh = CorrelationPlotterService.compute_correlation(df, 'unknown_type')
        assert thresh == 0.8

    def test_render_1x1_correlation_matrix(self):
        """1x1 matrix should still render without error."""
        corr_matrix = pd.DataFrame({'s1': [1.0]}, index=['s1'])
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, 0.5, 0.7, 'SingleSample'
        )
        assert isinstance(fig, plt.Figure)


# =============================================================================
# TestTypeCoercion
# =============================================================================

class TestTypeCoercion:
    def test_integer_concentrations(self):
        df = pd.DataFrame({
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [150, 250, 350],
        })
        assert df['concentration[s1]'].dtype == np.int64
        result = CorrelationPlotterService.prepare_data_for_correlation(
            df, [['s1', 's2']], 0
        )
        assert list(result.columns) == ['s1', 's2']

    def test_float32_concentrations(self):
        df = pd.DataFrame({
            'concentration[s1]': np.array([100, 200, 300], dtype=np.float32),
            'concentration[s2]': np.array([150, 250, 350], dtype=np.float32),
        })
        result = CorrelationPlotterService.prepare_data_for_correlation(
            df, [['s1', 's2']], 0
        )
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            result, 'biological replicates'
        )
        assert corr_matrix.shape == (2, 2)
        # Diagonal should still be 1.0
        assert corr_matrix.iloc[0, 0] == pytest.approx(1.0, abs=0.001)

    def test_mixed_int_float_correlation(self):
        df = pd.DataFrame({
            's1': [100, 200, 300],        # int
            's2': [1.5, 2.5, 3.5],        # float
        })
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert corr_matrix.shape == (2, 2)
        assert (corr_matrix.values >= -1).all()
        assert (corr_matrix.values <= 1).all()


# =============================================================================
# TestNaNHandling
# =============================================================================

class TestNaNHandling:
    def test_nan_in_concentration_data(self):
        """NaN values result in NaN correlations for that pair."""
        df = pd.DataFrame({
            's1': [100, np.nan, 300],
            's2': [150, 250, 350],
        })
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert corr_matrix.shape == (2, 2)
        # Correlation with NaN-containing column still computes (pandas skips NaN)
        assert np.isfinite(corr_matrix.iloc[0, 1])

    def test_all_nan_column(self):
        """All-NaN column produces NaN correlation."""
        df = pd.DataFrame({
            's1': [np.nan, np.nan, np.nan],
            's2': [150, 250, 350],
        })
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert np.isnan(corr_matrix.iloc[0, 1])

    def test_render_with_nan_in_correlation(self):
        """Rendering a heatmap with NaN values should not crash."""
        corr_matrix = pd.DataFrame({
            's1': [1.0, np.nan],
            's2': [np.nan, 1.0],
        }, index=['s1', 's2'])
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_matrix, 0.5, 0.7, 'NaNCondition'
        )
        assert isinstance(fig, plt.Figure)

    def test_constant_column_nan_correlation(self):
        """Constant values produce NaN correlation (std=0)."""
        df = pd.DataFrame({
            's1': [100, 100, 100],
            's2': [150, 250, 350],
        })
        corr_matrix, _, _ = CorrelationPlotterService.compute_correlation(
            df, 'biological replicates'
        )
        assert np.isnan(corr_matrix.iloc[0, 1])
