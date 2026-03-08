"""Tests for RetentionTimePlotterService — retention time scatter plots."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.retention_time import (
    RetentionTimePlotterService,
    _get_distinct_colors,
    _get_unique_colors,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def rt_df():
    """DataFrame with retention time data across 3 lipid classes."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PC(18:1)', 'PE(18:0)', 'PE(20:4)', 'TG(16:0)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TG'],
        'BaseRt': [3.5, 4.2, 5.1, 6.0, 12.3],
        'CalcMass': [733.5, 759.6, 717.5, 767.5, 807.7],
        'concentration[s1]': [100, 200, 300, 400, 500],
    })


# =============================================================================
# TestPlotSingleRetention
# =============================================================================

class TestPlotSingleRetention:
    def test_returns_list(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        assert isinstance(result, list)

    def test_one_plot_per_class(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        # 3 classes: PC, PE, TG
        assert len(result) == 3

    def test_each_item_is_tuple(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_figure_type(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        fig, df = result[0]
        assert isinstance(fig, go.Figure)
        assert isinstance(df, pd.DataFrame)

    def test_retention_df_columns(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        _, df = result[0]
        assert 'Mass' in df.columns
        assert 'Retention' in df.columns
        assert 'Species' in df.columns

    def test_ordered_by_frequency(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        # PC has 2, PE has 2, TG has 1 → PC and PE first
        _, first_df = result[0]
        _, second_df = result[1]
        assert len(first_df) >= len(second_df)

    def test_single_class(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(18:1)'],
            'ClassKey': ['PC', 'PC'],
            'BaseRt': [3.5, 4.2],
            'CalcMass': [733.5, 759.6],
        })
        result = RetentionTimePlotterService.plot_single_retention(df)
        assert len(result) == 1

    def test_single_lipid(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'BaseRt': [3.5],
            'CalcMass': [733.5],
        })
        result = RetentionTimePlotterService.plot_single_retention(df)
        assert len(result) == 1
        _, rdf = result[0]
        assert len(rdf) == 1

    def test_plot_title_is_class_name(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        fig, _ = result[0]
        # Title should be the class name
        assert fig.layout.title.text in ['PC', 'PE', 'TG']

    def test_white_background(self, rt_df):
        result = RetentionTimePlotterService.plot_single_retention(rt_df)
        fig, _ = result[0]
        assert fig.layout.plot_bgcolor == 'white'


# =============================================================================
# TestPlotMultiRetention
# =============================================================================

class TestPlotMultiRetention:
    def test_returns_tuple(self, rt_df):
        fig, df = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE'])
        assert isinstance(fig, go.Figure)
        assert isinstance(df, pd.DataFrame)

    def test_retention_df_has_class_column(self, rt_df):
        _, df = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE'])
        assert 'Class' in df.columns

    def test_filters_to_selected_classes(self, rt_df):
        _, df = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC'])
        assert set(df['Class'].unique()) == {'PC'}

    def test_all_classes(self, rt_df):
        _, df = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE', 'TG'])
        assert len(df['Class'].unique()) == 3

    def test_one_trace_per_class(self, rt_df):
        fig, _ = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE', 'TG'])
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 3

    def test_comparison_title(self, rt_df):
        fig, _ = RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE'])
        assert 'Comparison' in fig.layout.title.text

    def test_single_class_comparison(self, rt_df):
        fig, df = RetentionTimePlotterService.plot_multi_retention(rt_df, ['TG'])
        assert isinstance(fig, go.Figure)
        assert len(df) == 1

    def test_empty_class_list_empty_df(self, rt_df):
        fig, df = RetentionTimePlotterService.plot_multi_retention(rt_df, [])
        assert len(df) == 0


# =============================================================================
# TestColorGeneration
# =============================================================================

class TestColorGeneration:
    def test_distinct_colors_returns_n(self):
        colors = _get_distinct_colors(5)
        assert len(colors) == 5

    def test_distinct_colors_are_tuples(self):
        colors = _get_distinct_colors(3)
        for c in colors:
            assert isinstance(c, tuple)
            assert len(c) == 3

    def test_distinct_colors_rgb_range(self):
        colors = _get_distinct_colors(5)
        for r, g, b in colors:
            assert 0 <= r <= 1
            assert 0 <= g <= 1
            assert 0 <= b <= 1

    def test_unique_colors_returns_n(self):
        colors = _get_unique_colors(5)
        assert len(colors) == 5

    def test_unique_colors_are_strings(self):
        colors = _get_unique_colors(3)
        for c in colors:
            assert isinstance(c, str)

    def test_unique_colors_cycles(self):
        colors = _get_unique_colors(20)
        assert len(colors) == 20

    def test_distinct_colors_single(self):
        colors = _get_distinct_colors(1)
        assert len(colors) == 1

    def test_unique_colors_zero(self):
        colors = _get_unique_colors(0)
        assert len(colors) == 0


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    def test_many_classes(self):
        """15 classes should still work (color cycling)."""
        lipids = [f'L{i}' for i in range(15)]
        classes = [f'C{i}' for i in range(15)]
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
            'BaseRt': [i * 0.5 for i in range(15)],
            'CalcMass': [500 + i * 10 for i in range(15)],
        })
        result = RetentionTimePlotterService.plot_single_retention(df)
        assert len(result) == 15

    def test_input_immutability(self, rt_df):
        original = rt_df.copy()
        RetentionTimePlotterService.plot_single_retention(rt_df)
        pd.testing.assert_frame_equal(rt_df, original)

    def test_input_immutability_multi(self, rt_df):
        original = rt_df.copy()
        RetentionTimePlotterService.plot_multi_retention(rt_df, ['PC', 'PE'])
        pd.testing.assert_frame_equal(rt_df, original)

    def test_special_characters_in_class(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0/18:1)'],
            'ClassKey': ['PC/PE'],
            'BaseRt': [3.5],
            'CalcMass': [733.5],
        })
        result = RetentionTimePlotterService.plot_single_retention(df)
        assert len(result) == 1
