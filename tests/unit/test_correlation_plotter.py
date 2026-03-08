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
