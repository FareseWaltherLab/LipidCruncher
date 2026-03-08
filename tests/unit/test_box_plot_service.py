"""Tests for BoxPlotService — missing values bar charts and concentration box plots."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.box_plot import BoxPlotService, _get_sample_colors, CONDITION_COLORS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_df():
    """DataFrame with 5 lipids x 4 samples (concentration columns)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PE(18:1)', 'SM(18:0)', 'TG(16:0)', 'CE(18:2)'],
        'ClassKey': ['PC', 'PE', 'SM', 'TG', 'CE'],
        'concentration[s1]': [100, 200, 0, 400, 500],
        'concentration[s2]': [150, 0, 300, 450, 0],
        'concentration[s3]': [0, 250, 350, 0, 550],
        'concentration[s4]': [200, 300, 400, 500, 600],
    })


@pytest.fixture
def samples():
    return ['s1', 's2', 's3', 's4']


@pytest.fixture
def conditions():
    return ['Control', 'Treatment']


@pytest.fixture
def individual_samples():
    return [['s1', 's2'], ['s3', 's4']]


# =============================================================================
# TestCreateMeanAreaDf
# =============================================================================

class TestCreateMeanAreaDf:
    def test_extracts_concentration_columns(self, simple_df, samples):
        result = BoxPlotService.create_mean_area_df(simple_df, samples)
        assert list(result.columns) == [f'concentration[{s}]' for s in samples]

    def test_preserves_row_count(self, simple_df, samples):
        result = BoxPlotService.create_mean_area_df(simple_df, samples)
        assert len(result) == len(simple_df)

    def test_preserves_values(self, simple_df, samples):
        result = BoxPlotService.create_mean_area_df(simple_df, samples)
        assert result['concentration[s1]'].iloc[0] == 100

    def test_subset_of_samples(self, simple_df):
        result = BoxPlotService.create_mean_area_df(simple_df, ['s1', 's3'])
        assert len(result.columns) == 2

    def test_single_sample(self, simple_df):
        result = BoxPlotService.create_mean_area_df(simple_df, ['s2'])
        assert len(result.columns) == 1

    def test_missing_sample_raises(self, simple_df):
        with pytest.raises(KeyError):
            BoxPlotService.create_mean_area_df(simple_df, ['nonexistent'])


# =============================================================================
# TestCalculateMissingValuesPercentage
# =============================================================================

class TestCalculateMissingValuesPercentage:
    def test_returns_list_of_floats(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        result = BoxPlotService.calculate_missing_values_percentage(mean_area_df)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_correct_length(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        result = BoxPlotService.calculate_missing_values_percentage(mean_area_df)
        assert len(result) == len(samples)

    def test_known_values(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        result = BoxPlotService.calculate_missing_values_percentage(mean_area_df)
        # s1: 1 zero out of 5 = 20%, s2: 2 zeros = 40%, s3: 2 zeros = 40%, s4: 0 zeros = 0%
        assert result[0] == pytest.approx(20.0)
        assert result[1] == pytest.approx(40.0)
        assert result[2] == pytest.approx(40.0)
        assert result[3] == pytest.approx(0.0)

    def test_all_zeros(self):
        df = pd.DataFrame({'concentration[s1]': [0, 0, 0]})
        result = BoxPlotService.calculate_missing_values_percentage(df)
        assert result[0] == pytest.approx(100.0)

    def test_no_zeros(self):
        df = pd.DataFrame({'concentration[s1]': [1, 2, 3]})
        result = BoxPlotService.calculate_missing_values_percentage(df)
        assert result[0] == pytest.approx(0.0)

    def test_single_row(self):
        df = pd.DataFrame({'concentration[s1]': [0], 'concentration[s2]': [5]})
        result = BoxPlotService.calculate_missing_values_percentage(df)
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(0.0)


# =============================================================================
# TestGetSampleColors
# =============================================================================

class TestGetSampleColors:
    def test_returns_three_items(self, samples, conditions, individual_samples):
        colors, s2c, c2c = _get_sample_colors(samples, conditions, individual_samples)
        assert len(colors) == len(samples)
        assert isinstance(s2c, dict)
        assert isinstance(c2c, dict)

    def test_sample_to_condition_mapping(self, samples, conditions, individual_samples):
        _, s2c, _ = _get_sample_colors(samples, conditions, individual_samples)
        assert s2c['s1'] == 'Control'
        assert s2c['s2'] == 'Control'
        assert s2c['s3'] == 'Treatment'
        assert s2c['s4'] == 'Treatment'

    def test_condition_to_color_mapping(self, samples, conditions, individual_samples):
        _, _, c2c = _get_sample_colors(samples, conditions, individual_samples)
        assert c2c['Control'] == CONDITION_COLORS[0]
        assert c2c['Treatment'] == CONDITION_COLORS[1]

    def test_colors_match_conditions(self, samples, conditions, individual_samples):
        colors, _, c2c = _get_sample_colors(samples, conditions, individual_samples)
        assert colors[0] == c2c['Control']  # s1
        assert colors[2] == c2c['Treatment']  # s3

    def test_many_conditions_cycle_colors(self):
        conds = [f'C{i}' for i in range(15)]
        samples = [f's{i}' for i in range(15)]
        ind_samples = [[s] for s in samples]
        colors, _, c2c = _get_sample_colors(samples, conds, ind_samples)
        # Color should cycle after 10
        assert c2c['C10'] == CONDITION_COLORS[0]


# =============================================================================
# TestPlotMissingValues
# =============================================================================

class TestPlotMissingValues:
    def test_returns_figure(self, samples):
        fig = BoxPlotService.plot_missing_values(samples, [10, 20, 30, 40])
        assert isinstance(fig, go.Figure)

    def test_with_conditions(self, samples, conditions, individual_samples):
        fig = BoxPlotService.plot_missing_values(
            samples, [10, 20, 30, 40], conditions, individual_samples
        )
        assert isinstance(fig, go.Figure)

    def test_without_conditions_single_color(self, samples):
        fig = BoxPlotService.plot_missing_values(samples, [10, 20, 30, 40])
        # Should have one trace (single bar trace)
        assert len(fig.data) == 1

    def test_with_conditions_has_legend_traces(self, samples, conditions, individual_samples):
        fig = BoxPlotService.plot_missing_values(
            samples, [10, 20, 30, 40], conditions, individual_samples
        )
        # 4 sample traces + 2 legend traces = 6
        assert len(fig.data) == 6

    def test_title(self, samples):
        fig = BoxPlotService.plot_missing_values(samples, [10, 20, 30, 40])
        assert 'Missing Values' in fig.layout.title.text

    def test_dynamic_height_small(self):
        fig = BoxPlotService.plot_missing_values(['s1', 's2'], [10, 20])
        assert fig.layout.height >= 400

    def test_dynamic_height_large(self):
        samples = [f's{i}' for i in range(50)]
        percentages = [i for i in range(50)]
        fig = BoxPlotService.plot_missing_values(samples, percentages)
        assert fig.layout.height > 500

    def test_single_sample(self):
        fig = BoxPlotService.plot_missing_values(['s1'], [50.0])
        assert isinstance(fig, go.Figure)

    def test_empty_lists(self):
        fig = BoxPlotService.plot_missing_values([], [])
        assert isinstance(fig, go.Figure)


# =============================================================================
# TestPlotBoxPlot
# =============================================================================

class TestPlotBoxPlot:
    def test_returns_figure(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples)
        assert isinstance(fig, go.Figure)

    def test_with_conditions(self, simple_df, samples, conditions, individual_samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples, conditions, individual_samples)
        assert isinstance(fig, go.Figure)

    def test_without_conditions_box_traces(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples)
        # One box per sample
        assert len(fig.data) == len(samples)

    def test_with_conditions_has_legend(self, simple_df, samples, conditions, individual_samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples, conditions, individual_samples)
        # 4 sample traces + 2 legend traces = 6
        assert len(fig.data) == 6

    def test_title(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples)
        assert 'Box Plot' in fig.layout.title.text

    def test_log_transformation(self, simple_df, samples):
        """Box plot uses log10-transformed non-zero values."""
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples)
        # First trace should have log10-transformed values
        y_data = fig.data[0].y
        # s1 non-zero values: 100, 200, 400, 500 → log10: 2, 2.301, 2.602, 2.699
        assert any(v < 3 for v in y_data)  # log10(500) ≈ 2.7

    def test_all_zeros_sample(self):
        """Sample with all zeros should have empty box."""
        df = pd.DataFrame({
            'concentration[s1]': [0, 0, 0],
            'concentration[s2]': [100, 200, 300],
        })
        fig = BoxPlotService.plot_box_plot(df, ['s1', 's2'])
        # s1 trace should have empty y data
        assert len(fig.data[0].y) == 0

    def test_white_background(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        fig = BoxPlotService.plot_box_plot(mean_area_df, samples)
        assert fig.layout.plot_bgcolor == 'white'


# =============================================================================
# TestInputImmutability
# =============================================================================

class TestInputImmutability:
    def test_create_mean_area_df_preserves_input(self, simple_df, samples):
        original = simple_df.copy()
        BoxPlotService.create_mean_area_df(simple_df, samples)
        pd.testing.assert_frame_equal(simple_df, original)

    def test_plot_missing_values_preserves_input(self, samples):
        percentages = [10, 20, 30, 40]
        original = percentages.copy()
        BoxPlotService.plot_missing_values(samples, percentages)
        assert percentages == original

    def test_plot_box_plot_preserves_input(self, simple_df, samples):
        mean_area_df = BoxPlotService.create_mean_area_df(simple_df, samples)
        original = mean_area_df.copy()
        BoxPlotService.plot_box_plot(mean_area_df, samples)
        pd.testing.assert_frame_equal(mean_area_df, original)


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    def test_large_values(self):
        df = pd.DataFrame({
            'concentration[s1]': [1e12, 2e12],
            'concentration[s2]': [3e12, 4e12],
        })
        fig = BoxPlotService.plot_box_plot(df, ['s1', 's2'])
        assert isinstance(fig, go.Figure)

    def test_very_small_values(self):
        df = pd.DataFrame({
            'concentration[s1]': [1e-10, 2e-10],
            'concentration[s2]': [3e-10, 4e-10],
        })
        fig = BoxPlotService.plot_box_plot(df, ['s1', 's2'])
        assert isinstance(fig, go.Figure)

    def test_mixed_types_coerced(self):
        df = pd.DataFrame({
            'concentration[s1]': ['100', '200', '300'],
            'concentration[s2]': ['400', '500', '600'],
        })
        df = df.astype(float)
        result = BoxPlotService.calculate_missing_values_percentage(df)
        assert result[0] == pytest.approx(0.0)

    def test_three_conditions(self):
        samples = ['s1', 's2', 's3', 's4', 's5', 's6']
        conditions = ['A', 'B', 'C']
        individual = [['s1', 's2'], ['s3', 's4'], ['s5', 's6']]
        fig = BoxPlotService.plot_missing_values(
            samples, [10, 20, 30, 40, 50, 60], conditions, individual
        )
        # 6 sample traces + 3 legend traces
        assert len(fig.data) == 9
