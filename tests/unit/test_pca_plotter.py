"""Tests for PCAPlotterService — PCA scatter plots with confidence ellipses."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.pca import (
    PCAPlotterService,
    _run_pca,
    _generate_color_mapping,
    _add_confidence_ellipse,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def pca_df():
    """DataFrame with 20 lipids x 6 samples (2 conditions x 3 replicates)."""
    np.random.seed(42)
    n_lipids = 20
    return pd.DataFrame({
        'LipidMolec': [f'L{i}' for i in range(n_lipids)],
        'ClassKey': ['PC'] * n_lipids,
        'concentration[s1]': np.random.rand(n_lipids) * 1000,
        'concentration[s2]': np.random.rand(n_lipids) * 1000,
        'concentration[s3]': np.random.rand(n_lipids) * 1000,
        'concentration[s4]': np.random.rand(n_lipids) * 1000 + 500,
        'concentration[s5]': np.random.rand(n_lipids) * 1000 + 500,
        'concentration[s6]': np.random.rand(n_lipids) * 1000 + 500,
    })


@pytest.fixture
def samples():
    return ['s1', 's2', 's3', 's4', 's5', 's6']


@pytest.fixture
def conditions_list():
    return ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment']


# =============================================================================
# TestRunPCA
# =============================================================================

class TestRunPCA:
    def test_returns_three_items(self, pca_df, samples):
        pc_df, pc_names, avail = _run_pca(pca_df, samples)
        assert isinstance(pc_df, pd.DataFrame)
        assert isinstance(pc_names, list)
        assert isinstance(avail, list)

    def test_pc_df_has_two_columns(self, pca_df, samples):
        pc_df, _, _ = _run_pca(pca_df, samples)
        assert list(pc_df.columns) == ['PC1', 'PC2']

    def test_pc_df_rows_equal_samples(self, pca_df, samples):
        pc_df, _, avail = _run_pca(pca_df, samples)
        assert len(pc_df) == len(avail)
        assert len(avail) == len(samples)

    def test_pc_names_format(self, pca_df, samples):
        _, pc_names, _ = _run_pca(pca_df, samples)
        assert len(pc_names) == 2
        assert pc_names[0].startswith('PC1')
        assert '%' in pc_names[0]

    def test_variance_explained_sums_to_less_than_1(self, pca_df, samples):
        _, pc_names, _ = _run_pca(pca_df, samples)
        # Extract percentages
        import re
        pcts = [float(re.search(r'(\d+)%', name).group(1)) for name in pc_names]
        assert sum(pcts) <= 100

    def test_filters_missing_samples(self, pca_df):
        # Ask for sample that doesn't exist
        pc_df, _, avail = _run_pca(pca_df, ['s1', 's2', 'nonexistent'])
        assert len(avail) == 2
        assert 'nonexistent' not in avail

    def test_handles_tuple_input(self, pca_df, samples):
        """Legacy compatibility: accepts tuple of (df,)."""
        pc_df, _, _ = _run_pca((pca_df,), samples)
        assert len(pc_df) == len(samples)

    def test_two_samples_minimum(self):
        """PCA with exactly 2 samples should work."""
        df = pd.DataFrame({
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [150, 250, 350],
        })
        pc_df, _, _ = _run_pca(df, ['s1', 's2'])
        assert len(pc_df) == 2


# =============================================================================
# TestGenerateColorMapping
# =============================================================================

class TestGenerateColorMapping:
    def test_returns_dict(self, conditions_list):
        result = _generate_color_mapping(conditions_list)
        assert isinstance(result, dict)

    def test_one_color_per_unique_condition(self, conditions_list):
        result = _generate_color_mapping(conditions_list)
        assert len(result) == 2  # Control, Treatment

    def test_colors_are_strings(self, conditions_list):
        result = _generate_color_mapping(conditions_list)
        for color in result.values():
            assert isinstance(color, str)

    def test_many_conditions_cycles(self):
        conditions = [f'C{i}' for i in range(20)]
        result = _generate_color_mapping(conditions)
        assert len(result) == 20

    def test_single_condition(self):
        result = _generate_color_mapping(['Control', 'Control'])
        assert len(result) == 1


# =============================================================================
# TestAddConfidenceEllipse
# =============================================================================

class TestAddConfidenceEllipse:
    def test_adds_trace_to_figure(self):
        fig = go.Figure()
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 3, 2, 4, 5])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert len(fig.data) == 1

    def test_ellipse_is_line_trace(self):
        fig = go.Figure()
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 3, 2, 4, 5])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert fig.data[0].mode == 'lines'

    def test_ellipse_not_in_legend(self):
        fig = go.Figure()
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 3, 2, 4, 5])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert fig.data[0].showlegend is False

    def test_skips_single_point(self):
        fig = go.Figure()
        x = pd.Series([1])
        y = pd.Series([1])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert len(fig.data) == 0

    def test_two_points_works(self):
        fig = go.Figure()
        x = pd.Series([1, 5])
        y = pd.Series([1, 5])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert len(fig.data) == 1

    def test_ellipse_has_100_points(self):
        fig = go.Figure()
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 3, 2, 4, 5])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        assert len(fig.data[0].x) == 100

    def test_ellipse_centered_on_mean(self):
        fig = go.Figure()
        x = pd.Series([0, 10])
        y = pd.Series([0, 10])
        _add_confidence_ellipse(fig, x, y, 'blue', 'Test')
        # Ellipse should be roughly centered around (5, 5)
        center_x = np.mean(fig.data[0].x)
        center_y = np.mean(fig.data[0].y)
        assert center_x == pytest.approx(5.0, abs=0.5)
        assert center_y == pytest.approx(5.0, abs=0.5)


# =============================================================================
# TestPlotPCA
# =============================================================================

class TestPlotPCA:
    def test_returns_figure_and_df(self, pca_df, samples, conditions_list):
        fig, pc_df = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert isinstance(fig, go.Figure)
        assert isinstance(pc_df, pd.DataFrame)

    def test_pc_df_has_expected_columns(self, pca_df, samples, conditions_list):
        _, pc_df = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert 'PC1' in pc_df.columns
        assert 'PC2' in pc_df.columns
        assert 'Sample' in pc_df.columns
        assert 'Condition' in pc_df.columns

    def test_pc_df_row_count(self, pca_df, samples, conditions_list):
        _, pc_df = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert len(pc_df) == len(samples)

    def test_condition_labels_assigned(self, pca_df, samples, conditions_list):
        _, pc_df = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert pc_df.iloc[0]['Condition'] == 'Control'
        assert pc_df.iloc[3]['Condition'] == 'Treatment'

    def test_sample_names_assigned(self, pca_df, samples, conditions_list):
        _, pc_df = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert pc_df.iloc[0]['Sample'] == 's1'
        assert pc_df.iloc[5]['Sample'] == 's6'

    def test_figure_has_traces(self, pca_df, samples, conditions_list):
        fig, _ = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        # 2 conditions × (1 scatter + 1 ellipse) = 4 traces
        assert len(fig.data) >= 2

    def test_title(self, pca_df, samples, conditions_list):
        fig, _ = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert 'PCA' in fig.layout.title.text

    def test_white_background(self, pca_df, samples, conditions_list):
        fig, _ = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert fig.layout.plot_bgcolor == 'white'

    def test_axis_labels_contain_variance(self, pca_df, samples, conditions_list):
        fig, _ = PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        assert 'PC1' in fig.layout.xaxis.title.text
        assert '%' in fig.layout.xaxis.title.text


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    def test_input_immutability(self, pca_df, samples, conditions_list):
        original = pca_df.copy()
        PCAPlotterService.plot_pca(pca_df, samples, conditions_list)
        pd.testing.assert_frame_equal(pca_df, original)

    def test_three_conditions(self):
        np.random.seed(42)
        n = 10
        df = pd.DataFrame({
            'concentration[s1]': np.random.rand(n) * 100,
            'concentration[s2]': np.random.rand(n) * 100,
            'concentration[s3]': np.random.rand(n) * 100,
            'concentration[s4]': np.random.rand(n) * 100,
            'concentration[s5]': np.random.rand(n) * 100,
            'concentration[s6]': np.random.rand(n) * 100,
        })
        samples = ['s1', 's2', 's3', 's4', 's5', 's6']
        conditions = ['A', 'A', 'B', 'B', 'C', 'C']
        fig, pc_df = PCAPlotterService.plot_pca(df, samples, conditions)
        assert len(pc_df['Condition'].unique()) == 3

    def test_single_condition(self):
        np.random.seed(42)
        n = 10
        df = pd.DataFrame({
            'concentration[s1]': np.random.rand(n) * 100,
            'concentration[s2]': np.random.rand(n) * 100,
        })
        fig, pc_df = PCAPlotterService.plot_pca(
            df, ['s1', 's2'], ['Control', 'Control']
        )
        assert len(pc_df) == 2

    def test_missing_sample_columns(self):
        """Only existing samples should appear in PCA."""
        df = pd.DataFrame({
            'concentration[s1]': [100, 200, 300],
            'concentration[s2]': [150, 250, 350],
        })
        fig, pc_df = PCAPlotterService.plot_pca(
            df, ['s1', 's2', 's3'], ['A', 'A', 'B']
        )
        assert len(pc_df) == 2
        assert 's3' not in pc_df['Sample'].values

    def test_large_dataset(self):
        np.random.seed(42)
        n_lipids = 500
        n_samples = 20
        data = {f'concentration[s{i}]': np.random.rand(n_lipids) * 1000
                for i in range(n_samples)}
        df = pd.DataFrame(data)
        samples = [f's{i}' for i in range(n_samples)]
        conditions = ['A'] * 10 + ['B'] * 10
        fig, pc_df = PCAPlotterService.plot_pca(df, samples, conditions)
        assert len(pc_df) == n_samples
