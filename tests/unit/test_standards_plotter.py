"""
Unit tests for StandardsPlotterService.

Tests cover: empty inputs, single standard per class, multiple standards
per class, sample filtering, color cycling, layout properties, and
multi-class datasets.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.services.plotting.standards_plotter import StandardsPlotterService, _COLORS


# =============================================================================
# Helpers
# =============================================================================


def make_standards_df(standards_spec, samples):
    """Build an internal standards DataFrame.

    Args:
        standards_spec: List of (lipid_name, class_key, intensity_values) tuples.
            intensity_values is a list of floats, one per sample.
        samples: List of sample names.

    Returns:
        DataFrame with LipidMolec, ClassKey, and intensity[<sample>] columns.
    """
    rows = []
    for lipid, class_key, intensities in standards_spec:
        row = {'LipidMolec': lipid, 'ClassKey': class_key}
        for sample, val in zip(samples, intensities):
            row[f'intensity[{sample}]'] = val
        rows.append(row)
    return pd.DataFrame(rows)


SAMPLES_2 = ['s1', 's2']
SAMPLES_3 = ['s1', 's2', 's3']
SAMPLES_5 = ['s1', 's2', 's3', 's4', 's5']


# =============================================================================
# TestEmptyInputs
# =============================================================================


class TestEmptyInputs:
    """Edge cases that should return an empty list."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]'])
        result = StandardsPlotterService.create_consistency_plots(df, ['s1'])
        assert result == []

    def test_empty_samples_list(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, 200])],
            SAMPLES_2,
        )
        result = StandardsPlotterService.create_consistency_plots(df, [])
        assert result == []

    def test_no_matching_samples(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, 200])],
            SAMPLES_2,
        )
        result = StandardsPlotterService.create_consistency_plots(df, ['nonexistent'])
        assert result == []

    def test_samples_partially_matching(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, 200])],
            SAMPLES_2,
        )
        result = StandardsPlotterService.create_consistency_plots(df, ['s1', 'nonexistent'])
        assert len(result) == 1
        # Should only have data for s1
        bar = result[0].data[0]
        assert list(bar.x) == ['s1', 'nonexistent']
        # s1 has data, nonexistent is filtered out of valid_intensity_cols
        # Actually, nonexistent won't be in valid_intensity_cols so only s1 intensity is used

    def test_dataframe_no_intensity_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0)+D7:(s)'],
            'ClassKey': ['PC'],
        })
        result = StandardsPlotterService.create_consistency_plots(df, ['s1'])
        assert result == []


# =============================================================================
# TestSingleStandardPerClass
# =============================================================================


class TestSingleStandardPerClass:
    """Tests for classes with a single internal standard."""

    @pytest.fixture
    def single_pc(self):
        return make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [1000, 2000, 3000])],
            SAMPLES_3,
        )

    def test_returns_one_figure(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert len(figs) == 1

    def test_figure_is_plotly_figure(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert isinstance(figs[0], go.Figure)

    def test_bar_trace_data(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        bar = figs[0].data[0]
        assert list(bar.x) == SAMPLES_3
        assert list(bar.y) == [1000, 2000, 3000]

    def test_bar_color_is_first_color(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        bar = figs[0].data[0]
        assert bar.marker.color == _COLORS[0]

    def test_bar_name_is_standard_name(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        bar = figs[0].data[0]
        assert bar.name == 'PC(15:0)+D7:(s)'

    def test_layout_title_contains_class(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert 'PC' in figs[0].layout.title.text

    def test_layout_height(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert figs[0].layout.height == 400

    def test_layout_xaxis_title(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert figs[0].layout.xaxis.title.text == 'Samples'

    def test_layout_yaxis_title(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert figs[0].layout.yaxis.title.text == 'Raw Intensity'

    def test_legend_shown(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert figs[0].layout.showlegend is True

    def test_legend_title(self, single_pc):
        figs = StandardsPlotterService.create_consistency_plots(single_pc, SAMPLES_3)
        assert figs[0].layout.legend.title.text == 'Internal Standard'


# =============================================================================
# TestMultipleStandardsPerClass
# =============================================================================


class TestMultipleStandardsPerClass:
    """Tests for classes with multiple internal standards (subplots)."""

    @pytest.fixture
    def multi_pe(self):
        return make_standards_df(
            [
                ('PE(15:0)+D7:(s)', 'PE', [500, 600, 700]),
                ('PE(17:0)+D7:(s)', 'PE', [800, 900, 1000]),
            ],
            SAMPLES_3,
        )

    def test_returns_one_figure(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        assert len(figs) == 1

    def test_two_traces(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        assert len(figs[0].data) == 2

    def test_trace_names(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        names = [t.name for t in figs[0].data]
        assert names == ['PE(15:0)+D7:(s)', 'PE(17:0)+D7:(s)']

    def test_trace_colors_differ(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        colors = [t.marker.color for t in figs[0].data]
        assert colors[0] == _COLORS[0]
        assert colors[1] == _COLORS[1]
        assert colors[0] != colors[1]

    def test_trace_intensities(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        assert list(figs[0].data[0].y) == [500, 600, 700]
        assert list(figs[0].data[1].y) == [800, 900, 1000]

    def test_layout_height_scales_with_standards(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        assert figs[0].layout.height == 600  # 300 * 2

    def test_layout_title_contains_class(self, multi_pe):
        figs = StandardsPlotterService.create_consistency_plots(multi_pe, SAMPLES_3)
        assert 'PE' in figs[0].layout.title.text

    def test_three_standards_height(self):
        df = make_standards_df(
            [
                ('SM(d18:1)+D7:(s)', 'SM', [100, 200]),
                ('SM(d18:0)+D7:(s)', 'SM', [300, 400]),
                ('SM(d16:1)+D7:(s)', 'SM', [500, 600]),
            ],
            SAMPLES_2,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_2)
        assert figs[0].layout.height == 900  # 300 * 3
        assert len(figs[0].data) == 3


# =============================================================================
# TestMultipleClasses
# =============================================================================


class TestMultipleClasses:
    """Tests for datasets with standards from multiple lipid classes."""

    @pytest.fixture
    def multi_class(self):
        return make_standards_df(
            [
                ('PC(15:0)+D7:(s)', 'PC', [1000, 2000]),
                ('PE(15:0)+D7:(s)', 'PE', [3000, 4000]),
                ('SM(d18:1)+D7:(s)', 'SM', [5000, 6000]),
            ],
            SAMPLES_2,
        )

    def test_returns_one_figure_per_class(self, multi_class):
        figs = StandardsPlotterService.create_consistency_plots(multi_class, SAMPLES_2)
        assert len(figs) == 3

    def test_figures_sorted_by_class(self, multi_class):
        figs = StandardsPlotterService.create_consistency_plots(multi_class, SAMPLES_2)
        titles = [fig.layout.title.text for fig in figs]
        assert 'PC' in titles[0]
        assert 'PE' in titles[1]
        assert 'SM' in titles[2]

    def test_each_figure_has_correct_data(self, multi_class):
        figs = StandardsPlotterService.create_consistency_plots(multi_class, SAMPLES_2)
        assert list(figs[0].data[0].y) == [1000, 2000]
        assert list(figs[1].data[0].y) == [3000, 4000]
        assert list(figs[2].data[0].y) == [5000, 6000]

    def test_mixed_single_and_multi(self):
        """One class with 1 standard, another with 2."""
        df = make_standards_df(
            [
                ('PC(15:0)+D7:(s)', 'PC', [100, 200]),
                ('PE(15:0)+D7:(s)', 'PE', [300, 400]),
                ('PE(17:0)+D7:(s)', 'PE', [500, 600]),
            ],
            SAMPLES_2,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_2)
        assert len(figs) == 2
        # PC: single standard, simple bar
        assert len(figs[0].data) == 1
        assert figs[0].layout.height == 400
        # PE: two standards, subplots
        assert len(figs[1].data) == 2
        assert figs[1].layout.height == 600


# =============================================================================
# TestSampleFiltering
# =============================================================================


class TestSampleFiltering:
    """Tests for filtering to a subset of samples."""

    @pytest.fixture
    def full_df(self):
        return make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, 200, 300, 400, 500])],
            SAMPLES_5,
        )

    def test_subset_of_samples(self, full_df):
        figs = StandardsPlotterService.create_consistency_plots(full_df, ['s2', 's4'])
        bar = figs[0].data[0]
        assert list(bar.x) == ['s2', 's4']
        assert list(bar.y) == [200, 400]

    def test_sample_order_preserved(self, full_df):
        figs = StandardsPlotterService.create_consistency_plots(full_df, ['s5', 's1', 's3'])
        bar = figs[0].data[0]
        assert list(bar.x) == ['s5', 's1', 's3']
        assert list(bar.y) == [500, 100, 300]

    def test_single_sample(self, full_df):
        figs = StandardsPlotterService.create_consistency_plots(full_df, ['s3'])
        bar = figs[0].data[0]
        assert list(bar.x) == ['s3']
        assert list(bar.y) == [300]


# =============================================================================
# TestColorCycling
# =============================================================================


class TestColorCycling:
    """Tests that colors cycle when there are more standards than colors."""

    def test_colors_cycle_after_palette_exhausted(self):
        n_colors = len(_COLORS)
        n_standards = n_colors + 2
        specs = [
            (f'TG({i}:0)+D7:(s)', 'TG', [float(i * 100)])
            for i in range(n_standards)
        ]
        df = make_standards_df(specs, ['s1'])
        figs = StandardsPlotterService.create_consistency_plots(df, ['s1'])
        colors = [t.marker.color for t in figs[0].data]
        # First n_colors should be the palette
        assert colors[:n_colors] == _COLORS
        # Next should wrap around
        assert colors[n_colors] == _COLORS[0]
        assert colors[n_colors + 1] == _COLORS[1]


# =============================================================================
# TestFontStyling
# =============================================================================


class TestFontStyling:
    """Tests that font styling is applied consistently."""

    @pytest.fixture
    def single_fig(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, 200])],
            SAMPLES_2,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_2)
        return figs[0]

    def test_font_color(self, single_fig):
        assert single_fig.layout.font.color == 'black'

    def test_title_font_color(self, single_fig):
        assert single_fig.layout.title.font.color == 'black'

    def test_legend_font_color(self, single_fig):
        assert single_fig.layout.legend.font.color == 'black'

    def test_xaxis_tickfont_color(self, single_fig):
        assert single_fig.layout.xaxis.tickfont.color == 'black'

    def test_yaxis_tickfont_color(self, single_fig):
        assert single_fig.layout.yaxis.tickfont.color == 'black'


# =============================================================================
# TestIntensityValues
# =============================================================================


class TestIntensityValues:
    """Tests with various intensity value types."""

    def test_zero_intensities(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [0, 0, 0])],
            SAMPLES_3,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_3)
        assert list(figs[0].data[0].y) == [0, 0, 0]

    def test_very_large_intensities(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [1e12, 2e12])],
            SAMPLES_2,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_2)
        assert list(figs[0].data[0].y) == [1e12, 2e12]

    def test_float_intensities(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [1.5, 2.7, 3.14])],
            SAMPLES_3,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_3)
        assert list(figs[0].data[0].y) == [1.5, 2.7, 3.14]

    def test_nan_intensities(self):
        df = make_standards_df(
            [('PC(15:0)+D7:(s)', 'PC', [100, float('nan'), 300])],
            SAMPLES_3,
        )
        figs = StandardsPlotterService.create_consistency_plots(df, SAMPLES_3)
        y_vals = list(figs[0].data[0].y)
        assert y_vals[0] == 100
        assert np.isnan(y_vals[1])
        assert y_vals[2] == 300
