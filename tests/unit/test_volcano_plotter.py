"""
Unit tests for VolcanoPlotterService.

Covers: data preparation, volcano plot rendering, concentration-vs-FC plot,
distribution plot, color mapping, label placement, edge cases, type coercion,
immutability, and large dataset stress tests.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt

from tests.conftest import make_experiment

from app.services.plotting.volcano_plot import (
    VolcanoData,
    VolcanoPlotterService,
    _compute_zero_adj_means,
    _determine_exclusion_reason,
    _filter_significant,
    _has_overlap,
    _safe_numeric_values,
)
from app.services.statistical_testing import (
    StatisticalTestResult,
    StatisticalTestSummary,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_df(
    lipids=None,
    classes=None,
    n_samples=6,
    values_per_sample=None,
    prefix='concentration',
):
    """Build a minimal volcano-style DataFrame."""
    if lipids is None:
        lipids = ['PC(16:0)', 'PE(18:1)']
    if classes is None:
        classes = ['PC', 'PE'] if len(lipids) == 2 else ['PC'] * len(lipids)
    data = {'LipidMolec': lipids, 'ClassKey': classes}
    for s in range(1, n_samples + 1):
        if values_per_sample is not None:
            data[f'{prefix}[s{s}]'] = values_per_sample[s - 1]
        else:
            data[f'{prefix}[s{s}]'] = [100.0 * s] * len(lipids)
    return pd.DataFrame(data)


def _make_stat_results(
    lipids,
    p_values=None,
    adjusted_p_values=None,
    effect_sizes=None,
    significant=None,
    test_name="Welch's t-test",
    correction='fdr_bh',
    transform='log10',
):
    """Build a StatisticalTestSummary for testing."""
    n = len(lipids)
    if p_values is None:
        p_values = [0.01] * n
    if adjusted_p_values is None:
        adjusted_p_values = p_values
    if effect_sizes is None:
        effect_sizes = [1.0] * n
    if significant is None:
        significant = [True] * n

    results = {}
    for i, lipid in enumerate(lipids):
        results[lipid] = StatisticalTestResult(
            test_name=test_name,
            statistic=2.5,
            p_value=p_values[i],
            adjusted_p_value=adjusted_p_values[i],
            significant=significant[i],
            effect_size=effect_sizes[i],
            group_key=lipid,
        )

    return StatisticalTestSummary(
        results=results,
        test_info={
            'test_type': 'parametric',
            'correction': correction,
            'transform': transform,
        },
        parameters={
            'n_lipids_tested': n,
            'n_lipids_total': n,
            'alpha': 0.05,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """2 lipids, 6 samples (3 ctrl + 3 exp)."""
    return _make_df(
        lipids=['PC(16:0)', 'PE(18:1)'],
        classes=['PC', 'PE'],
        n_samples=6,
        values_per_sample=[
            [100, 200],  # s1 (ctrl)
            [110, 210],  # s2 (ctrl)
            [105, 205],  # s3 (ctrl)
            [500, 400],  # s4 (exp)
            [510, 410],  # s5 (exp)
            [505, 405],  # s6 (exp)
        ],
    )


@pytest.fixture
def simple_stat_results():
    """Stat results for simple_df lipids."""
    return _make_stat_results(
        lipids=['PC(16:0)', 'PE(18:1)'],
        p_values=[0.001, 0.03],
        adjusted_p_values=[0.002, 0.04],
        effect_sizes=[2.25, 0.98],
        significant=[True, True],
    )


@pytest.fixture
def simple_volcano_data(simple_df, simple_stat_results):
    """Pre-built VolcanoData for convenience."""
    return VolcanoPlotterService.prepare_volcano_data(
        simple_df, simple_stat_results,
        control_samples=['s1', 's2', 's3'],
        experimental_samples=['s4', 's5', 's6'],
    )


@pytest.fixture
def simple_color_mapping():
    return {'PC': '#636EFA', 'PE': '#EF553B'}


# ═══════════════════════════════════════════════════════════════════════
# VolcanoData Dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestVolcanoDataDataclass:
    """Tests for the VolcanoData dataclass."""

    def test_default_construction(self):
        vd = VolcanoData(
            volcano_df=pd.DataFrame(),
            removed_lipids_df=pd.DataFrame(),
        )
        assert isinstance(vd.volcano_df, pd.DataFrame)
        assert isinstance(vd.removed_lipids_df, pd.DataFrame)
        assert isinstance(vd.stat_results, StatisticalTestSummary)

    def test_with_stat_results(self, simple_stat_results):
        vd = VolcanoData(
            volcano_df=pd.DataFrame({'a': [1]}),
            removed_lipids_df=pd.DataFrame(),
            stat_results=simple_stat_results,
        )
        assert len(vd.stat_results.results) == 2


# ═══════════════════════════════════════════════════════════════════════
# prepare_volcano_data
# ═══════════════════════════════════════════════════════════════════════


class TestPrepareVolcanoData:
    """Tests for VolcanoPlotterService.prepare_volcano_data."""

    def test_returns_volcano_data(self, simple_df, simple_stat_results):
        result = VolcanoPlotterService.prepare_volcano_data(
            simple_df, simple_stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert isinstance(result, VolcanoData)

    def test_volcano_df_has_required_columns(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        expected_cols = [
            'LipidMolec', 'ClassKey', 'FoldChange', 'pValue',
            'adjusted_pValue', '-log10(pValue)', '-log10(adjusted_pValue)',
            'Log10MeanControl', 'mean_control', 'mean_experimental',
            'test_method', 'transformation', 'significant',
            'correction_method',
        ]
        for col in expected_cols:
            assert col in vdf.columns, f"Missing column: {col}"

    def test_all_tested_lipids_in_volcano_df(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert set(vdf['LipidMolec'].tolist()) == {'PC(16:0)', 'PE(18:1)'}

    def test_fold_change_from_effect_size(self, simple_df, simple_stat_results):
        result = VolcanoPlotterService.prepare_volcano_data(
            simple_df, simple_stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        vdf = result.volcano_df
        pc_row = vdf[vdf['LipidMolec'] == 'PC(16:0)'].iloc[0]
        assert pc_row['FoldChange'] == pytest.approx(2.25)

    def test_p_values_populated(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        pc_row = vdf[vdf['LipidMolec'] == 'PC(16:0)'].iloc[0]
        assert pc_row['pValue'] == pytest.approx(0.001)
        assert pc_row['adjusted_pValue'] == pytest.approx(0.002)

    def test_neg_log10_p_values(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        pc_row = vdf[vdf['LipidMolec'] == 'PC(16:0)'].iloc[0]
        assert pc_row['-log10(pValue)'] == pytest.approx(-np.log10(0.001))
        assert pc_row['-log10(adjusted_pValue)'] == pytest.approx(-np.log10(0.002))

    def test_mean_control_positive(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['mean_control'] > 0)

    def test_mean_experimental_positive(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['mean_experimental'] > 0)

    def test_log10_mean_control(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        for _, row in vdf.iterrows():
            if row['mean_control'] > 0:
                expected = np.log10(row['mean_control'])
                assert row['Log10MeanControl'] == pytest.approx(expected, rel=0.01)

    def test_test_method_populated(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['test_method'] == "Welch's t-test")

    def test_transformation_from_test_info(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['transformation'] == 'log10')

    def test_correction_method_from_test_info(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['correction_method'] == 'fdr_bh')

    def test_significant_flag(self, simple_volcano_data):
        vdf = simple_volcano_data.volcano_df
        assert all(vdf['significant'])

    def test_removed_lipids_df_empty_when_all_tested(self, simple_volcano_data):
        assert simple_volcano_data.removed_lipids_df.empty

    def test_removed_lipids_has_untested_lipids(self):
        """Lipids in df but not in stat_results go to removed_lipids_df."""
        df = _make_df(
            lipids=['PC(16:0)', 'PE(18:1)', 'SM(18:0)'],
            classes=['PC', 'PE', 'SM'],
            n_samples=6,
        )
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == 1
        assert len(result.removed_lipids_df) == 2
        assert 'PE(18:1)' in result.removed_lipids_df['LipidMolec'].values
        assert 'SM(18:0)' in result.removed_lipids_df['LipidMolec'].values

    def test_removed_lipids_has_reason_column(self):
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=6)
        stat_results = _make_stat_results(lipids=[])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert 'Reason' in result.removed_lipids_df.columns

    def test_all_zeros_excluded_with_reason(self):
        """Lipid with all zero values should be excluded with proper reason."""
        df = _make_df(
            lipids=['PC(16:0)'],
            classes=['PC'],
            n_samples=6,
            values_per_sample=[[0]] * 6,
        )
        stat_results = _make_stat_results(lipids=[])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.removed_lipids_df) == 1
        assert 'zero' in result.removed_lipids_df['Reason'].iloc[0].lower()

    def test_stat_results_stored(self, simple_df, simple_stat_results):
        result = VolcanoPlotterService.prepare_volcano_data(
            simple_df, simple_stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert result.stat_results is simple_stat_results

    def test_no_adjusted_p_falls_back_to_raw(self):
        """When adjusted_p_value is None, use raw p_value."""
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=6)
        results = {
            'PC(16:0)': StatisticalTestResult(
                test_name="Welch's t-test",
                statistic=2.5,
                p_value=0.03,
                adjusted_p_value=None,
                significant=True,
                effect_size=1.5,
                group_key='PC(16:0)',
            ),
        }
        stat_results = StatisticalTestSummary(
            results=results,
            test_info={'correction': 'uncorrected', 'transform': 'none'},
        )
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        vdf = result.volcano_df
        assert vdf['adjusted_pValue'].iloc[0] == pytest.approx(0.03)

    def test_effect_size_none_defaults_to_zero(self):
        """When effect_size is None, FoldChange should be 0."""
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=6)
        results = {
            'PC(16:0)': StatisticalTestResult(
                test_name="Welch's t-test",
                statistic=2.5,
                p_value=0.03,
                effect_size=None,
                group_key='PC(16:0)',
            ),
        }
        stat_results = StatisticalTestSummary(results=results, test_info={})
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert result.volcano_df['FoldChange'].iloc[0] == 0.0


class TestPrepareVolcanoDataEdgeCases:
    """Edge cases for prepare_volcano_data."""

    def test_empty_df_raises(self, simple_stat_results):
        with pytest.raises(ValueError, match="empty"):
            VolcanoPlotterService.prepare_volcano_data(
                pd.DataFrame(), simple_stat_results,
                control_samples=['s1'], experimental_samples=['s4'],
            )

    def test_none_df_raises(self, simple_stat_results):
        with pytest.raises(ValueError, match="empty"):
            VolcanoPlotterService.prepare_volcano_data(
                None, simple_stat_results,
                control_samples=['s1'], experimental_samples=['s4'],
            )

    def test_missing_lipidmolec_column_raises(self, simple_stat_results):
        df = pd.DataFrame({'ClassKey': ['PC'], 'concentration[s1]': [100]})
        with pytest.raises(ValueError, match="LipidMolec"):
            VolcanoPlotterService.prepare_volcano_data(
                df, simple_stat_results,
                control_samples=['s1'], experimental_samples=['s4'],
            )

    def test_missing_classkey_column_raises(self, simple_stat_results):
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'concentration[s1]': [100]})
        with pytest.raises(ValueError, match="ClassKey"):
            VolcanoPlotterService.prepare_volcano_data(
                df, simple_stat_results,
                control_samples=['s1'], experimental_samples=['s4'],
            )

    def test_no_matching_lipids_returns_empty_volcano_df(self):
        """All lipids excluded → empty volcano_df, populated removed_df."""
        df = _make_df(lipids=['X1'], classes=['X'], n_samples=6)
        stat_results = _make_stat_results(lipids=[])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert result.volcano_df.empty
        assert len(result.removed_lipids_df) == 1

    def test_single_lipid(self):
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=6)
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == 1
        assert result.removed_lipids_df.empty

    def test_nan_concentrations_handled(self):
        """NaN values in concentration columns should not crash."""
        df = _make_df(
            lipids=['PC(16:0)'],
            classes=['PC'],
            n_samples=6,
            values_per_sample=[[np.nan], [100], [100], [200], [np.nan], [200]],
        )
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == 1

    def test_many_classes(self):
        """Multiple classes in same DataFrame."""
        lipids = ['PC(16:0)', 'PE(18:1)', 'SM(18:0)', 'PI(20:4)']
        classes = ['PC', 'PE', 'SM', 'PI']
        df = _make_df(lipids=lipids, classes=classes, n_samples=6)
        stat_results = _make_stat_results(lipids=lipids)
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == 4
        assert set(result.volcano_df['ClassKey'].tolist()) == set(classes)


# ═══════════════════════════════════════════════════════════════════════
# create_volcano_plot
# ═══════════════════════════════════════════════════════════════════════


class TestCreateVolcanoPlot:
    """Tests for VolcanoPlotterService.create_volcano_plot."""

    def test_returns_figure(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_class(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        trace_names = [t.name for t in fig.data]
        assert 'PC' in trace_names
        assert 'PE' in trace_names

    def test_scatter_mode_markers(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        for trace in fig.data:
            assert trace.mode == 'markers'

    def test_marker_colors_match_mapping(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        for trace in fig.data:
            assert trace.marker.color == simple_color_mapping[trace.name]

    def test_title_is_volcano_plot(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert 'Volcano Plot' in fig.layout.title.text

    def test_xaxis_label(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert 'Fold Change' in fig.layout.xaxis.title.text

    def test_yaxis_label_adjusted(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            use_adjusted_p=True,
        )
        assert 'Adjusted' in fig.layout.yaxis.title.text

    def test_yaxis_label_raw(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            use_adjusted_p=False,
        )
        assert 'Adjusted' not in fig.layout.yaxis.title.text

    def test_chart_height(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert fig.layout.height == 600

    def test_white_background(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert fig.layout.plot_bgcolor == 'white'

    def test_black_axes(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert fig.layout.xaxis.linecolor == 'black'
        assert fig.layout.yaxis.linecolor == 'black'

    def test_mirrored_axes(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert fig.layout.xaxis.mirror is True
        assert fig.layout.yaxis.mirror is True


class TestThresholdLines:
    """Tests for threshold lines on volcano plot."""

    def test_threshold_lines_present(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            p_threshold=0.05, fc_threshold=1.0,
        )
        # 3 shapes: 1 horizontal + 2 vertical
        assert len(fig.layout.shapes) == 3

    def test_horizontal_line_at_p_threshold(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            p_threshold=0.05,
        )
        q_val = -np.log10(0.05)
        h_lines = [s for s in fig.layout.shapes if s.y0 == s.y1]
        assert any(
            abs(s.y0 - q_val) < 0.01 for s in h_lines
        ), f"No horizontal line at q={q_val}"

    def test_vertical_lines_at_fc_threshold(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            fc_threshold=1.5,
        )
        v_lines = [s for s in fig.layout.shapes if s.x0 == s.x1]
        x_vals = [s.x0 for s in v_lines]
        assert pytest.approx(-1.5) in x_vals
        assert pytest.approx(1.5) in x_vals

    def test_threshold_line_style(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        for shape in fig.layout.shapes:
            assert shape.line.dash == 'dash'
            assert shape.line.color == 'red'
            assert shape.line.width == 2


class TestHideNonSignificant:
    """Tests for hide_non_sig parameter."""

    def test_hide_non_sig_filters_points(self):
        """Non-significant lipids should be hidden."""
        df = _make_df(
            lipids=['L1', 'L2', 'L3'],
            classes=['PC', 'PC', 'PE'],
            n_samples=6,
        )
        stat_results = _make_stat_results(
            lipids=['L1', 'L2', 'L3'],
            p_values=[0.001, 0.5, 0.001],
            adjusted_p_values=[0.002, 0.6, 0.002],
            effect_sizes=[2.0, 0.1, -2.0],  # L2 has small FC
            significant=[True, False, True],
        )
        volcano_data = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        color_map = VolcanoPlotterService.generate_color_mapping(['PC', 'PE'])

        fig_all = VolcanoPlotterService.create_volcano_plot(
            volcano_data, color_map, hide_non_sig=False,
        )
        fig_sig = VolcanoPlotterService.create_volcano_plot(
            volcano_data, color_map,
            hide_non_sig=True, p_threshold=0.05, fc_threshold=0.5,
        )

        total_pts_all = sum(len(t.x) for t in fig_all.data)
        total_pts_sig = sum(len(t.x) for t in fig_sig.data)
        assert total_pts_sig < total_pts_all

    def test_hide_non_sig_false_shows_all(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, hide_non_sig=False,
        )
        total_pts = sum(len(t.x) for t in fig.data)
        assert total_pts == 2  # Both lipids shown


class TestTopNLabels:
    """Tests for top N label placement."""

    def test_no_labels_by_default(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, top_n_labels=0,
        )
        # Only threshold line annotations, no text labels
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 0

    def test_top_1_label(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, top_n_labels=1,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 1
        # Most significant is PC(16:0) (p=0.001 vs 0.03)
        assert text_annotations[0].text == 'PC(16:0)'

    def test_top_2_labels(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, top_n_labels=2,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 2
        label_texts = {a.text for a in text_annotations}
        assert label_texts == {'PC(16:0)', 'PE(18:1)'}

    def test_labels_have_arrows(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, top_n_labels=1,
        )
        arrow_annotations = [a for a in fig.layout.annotations if a.showarrow]
        assert len(arrow_annotations) >= 1

    def test_custom_label_positions(self, simple_volcano_data, simple_color_mapping):
        custom = {'PC(16:0)': (0.5, 0.5)}
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
            top_n_labels=1, custom_label_positions=custom,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 1

    def test_top_n_exceeds_lipid_count(self, simple_volcano_data, simple_color_mapping):
        """Requesting more labels than lipids should label all."""
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, top_n_labels=100,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 2


class TestCreateVolcanoPlotEdgeCases:
    """Edge cases for create_volcano_plot."""

    def test_empty_volcano_data_raises(self, simple_color_mapping):
        vd = VolcanoData(
            volcano_df=pd.DataFrame(),
            removed_lipids_df=pd.DataFrame(),
        )
        with pytest.raises(ValueError, match="No volcano data"):
            VolcanoPlotterService.create_volcano_plot(vd, simple_color_mapping)

    def test_single_point(self, simple_color_mapping):
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=6)
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        fig = VolcanoPlotterService.create_volcano_plot(vd, {'PC': '#636EFA'})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_class_not_in_color_mapping_skipped(self, simple_volcano_data):
        """Classes not in color_mapping get no trace."""
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, {'PC': '#636EFA'},  # PE missing
        )
        trace_names = [t.name for t in fig.data]
        assert 'PC' in trace_names
        assert 'PE' not in trace_names

    def test_use_raw_p_values(self, simple_volcano_data, simple_color_mapping):
        fig = VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping, use_adjusted_p=False,
        )
        # Should use -log10(pValue) for y-axis
        for trace in fig.data:
            # y values should be -log10(raw p)
            assert all(y >= 0 for y in trace.y)


# ═══════════════════════════════════════════════════════════════════════
# create_concentration_vs_fc_plot
# ═══════════════════════════════════════════════════════════════════════


class TestConcentrationVsFcPlot:
    """Tests for concentration vs fold change plot."""

    def test_returns_figure_and_df(self, simple_volcano_data, simple_color_mapping):
        fig, summary_df = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(summary_df, pd.DataFrame)

    def test_summary_df_columns(self, simple_volcano_data, simple_color_mapping):
        _, summary_df = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        for col in ['LipidMolec', 'Log10MeanControl', 'FoldChange', 'ClassKey']:
            assert col in summary_df.columns

    def test_summary_df_contains_all_lipids(self, simple_volcano_data, simple_color_mapping):
        _, summary_df = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert len(summary_df) == 2

    def test_title(self, simple_volcano_data, simple_color_mapping):
        fig, _ = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert 'Fold Change' in fig.layout.title.text
        assert 'Control' in fig.layout.title.text

    def test_axes_labels(self, simple_volcano_data, simple_color_mapping):
        fig, _ = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        assert 'Fold Change' in fig.layout.xaxis.title.text
        assert 'Control' in fig.layout.yaxis.title.text

    def test_one_trace_per_class(self, simple_volcano_data, simple_color_mapping):
        fig, _ = VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        trace_names = {t.name for t in fig.data}
        assert trace_names == {'PC', 'PE'}

    def test_empty_raises(self, simple_color_mapping):
        vd = VolcanoData(
            volcano_df=pd.DataFrame(),
            removed_lipids_df=pd.DataFrame(),
        )
        with pytest.raises(ValueError, match="No volcano data"):
            VolcanoPlotterService.create_concentration_vs_fc_plot(
                vd, simple_color_mapping,
            )

    def test_hide_non_sig(self, simple_color_mapping):
        """With hide_non_sig, only significant points shown."""
        df = _make_df(
            lipids=['L1', 'L2'],
            classes=['PC', 'PE'],
            n_samples=6,
        )
        stat_results = _make_stat_results(
            lipids=['L1', 'L2'],
            p_values=[0.001, 0.5],
            adjusted_p_values=[0.002, 0.6],
            effect_sizes=[2.0, 0.1],
            significant=[True, False],
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        fig_all, _ = VolcanoPlotterService.create_concentration_vs_fc_plot(
            vd, simple_color_mapping, hide_non_sig=False,
        )
        fig_sig, _ = VolcanoPlotterService.create_concentration_vs_fc_plot(
            vd, simple_color_mapping,
            hide_non_sig=True, p_threshold=0.05,
        )
        pts_all = sum(len(t.x) for t in fig_all.data)
        pts_sig = sum(len(t.x) for t in fig_sig.data)
        assert pts_sig <= pts_all


# ═══════════════════════════════════════════════════════════════════════
# create_distribution_plot
# ═══════════════════════════════════════════════════════════════════════


class TestCreateDistributionPlot:
    """Tests for concentration distribution box plot."""

    def test_returns_matplotlib_figure(self, simple_df, experiment_2x3):
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control'],
            experiment=experiment_2x3,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title(self, simple_df, experiment_2x3):
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control'],
            experiment=experiment_2x3,
        )
        ax = fig.axes[0]
        assert 'Distribution' in ax.get_title()
        plt.close(fig)

    def test_axis_labels(self, simple_df, experiment_2x3):
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control'],
            experiment=experiment_2x3,
        )
        ax = fig.axes[0]
        assert 'Lipid' in ax.get_xlabel()
        assert 'Concentration' in ax.get_ylabel()
        plt.close(fig)

    def test_multiple_lipids(self, simple_df, experiment_2x3):
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)', 'PE(18:1)'],
            selected_conditions=['Control', 'Treatment'],
            experiment=experiment_2x3,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_lipids_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="lipid"):
            VolcanoPlotterService.create_distribution_plot(
                simple_df,
                selected_lipids=[],
                selected_conditions=['Control'],
                experiment=experiment_2x3,
            )

    def test_empty_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="condition"):
            VolcanoPlotterService.create_distribution_plot(
                simple_df,
                selected_lipids=['PC(16:0)'],
                selected_conditions=[],
                experiment=experiment_2x3,
            )

    def test_nonexistent_lipid_skipped(self, simple_df, experiment_2x3):
        """Non-existent lipid should not crash, just be omitted."""
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)', 'NONEXISTENT'],
            selected_conditions=['Control'],
            experiment=experiment_2x3,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_nonexistent_lipids_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="No valid data"):
            VolcanoPlotterService.create_distribution_plot(
                simple_df,
                selected_lipids=['NONEXISTENT'],
                selected_conditions=['Control'],
                experiment=experiment_2x3,
            )

    def test_nonexistent_condition_skipped(self, simple_df, experiment_2x3):
        """Non-existent condition should be skipped gracefully."""
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control', 'FAKE'],
            experiment=experiment_2x3,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# generate_color_mapping
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateColorMapping:
    """Tests for generate_color_mapping."""

    def test_returns_dict(self):
        result = VolcanoPlotterService.generate_color_mapping(['PC', 'PE'])
        assert isinstance(result, dict)

    def test_one_color_per_class(self):
        classes = ['PC', 'PE', 'SM']
        result = VolcanoPlotterService.generate_color_mapping(classes)
        assert len(result) == 3

    def test_consistent_order(self):
        classes = ['PC', 'PE', 'SM']
        r1 = VolcanoPlotterService.generate_color_mapping(classes)
        r2 = VolcanoPlotterService.generate_color_mapping(classes)
        assert r1 == r2

    def test_empty_classes(self):
        result = VolcanoPlotterService.generate_color_mapping([])
        assert result == {}

    def test_wraps_around_colors(self):
        """More classes than colors should cycle."""
        from app.services.plotting.volcano_plot import CLASS_COLORS
        n = len(CLASS_COLORS) + 3
        classes = [f'C{i}' for i in range(n)]
        result = VolcanoPlotterService.generate_color_mapping(classes)
        assert len(result) == n
        # Verify cycling: class at index len(CLASS_COLORS) gets same color as index 0
        assert result[f'C{len(CLASS_COLORS)}'] == result['C0']

    def test_colors_are_hex_strings(self):
        result = VolcanoPlotterService.generate_color_mapping(['PC'])
        assert result['PC'].startswith('#')


# ═══════════════════════════════════════════════════════════════════════
# get_most_abundant_lipid
# ═══════════════════════════════════════════════════════════════════════


class TestGetMostAbundantLipid:
    """Tests for get_most_abundant_lipid."""

    def test_returns_most_abundant(self):
        df = _make_df(
            lipids=['PC(14:0)', 'PC(16:0)', 'PC(18:0)'],
            classes=['PC', 'PC', 'PC'],
            n_samples=3,
            values_per_sample=[
                [10, 100, 50],   # s1
                [20, 200, 60],   # s2
                [30, 300, 70],   # s3
            ],
        )
        result = VolcanoPlotterService.get_most_abundant_lipid(df, 'PC')
        assert result == 'PC(16:0)'

    def test_empty_class_returns_none(self):
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=3)
        result = VolcanoPlotterService.get_most_abundant_lipid(df, 'PE')
        assert result is None

    def test_no_concentration_columns_returns_none(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
        })
        result = VolcanoPlotterService.get_most_abundant_lipid(df, 'PC')
        assert result is None

    def test_single_lipid(self):
        df = _make_df(lipids=['PC(16:0)'], classes=['PC'], n_samples=3)
        result = VolcanoPlotterService.get_most_abundant_lipid(df, 'PC')
        assert result == 'PC(16:0)'


# ═══════════════════════════════════════════════════════════════════════
# Private Helpers
# ═══════════════════════════════════════════════════════════════════════


class TestPrivateHelpers:
    """Tests for private helper functions."""

    def test_safe_numeric_values_basic(self):
        row = pd.Series({'concentration[s1]': 100.0, 'concentration[s2]': 200.0})
        result = _safe_numeric_values(row, ['concentration[s1]', 'concentration[s2]'])
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(200.0)

    def test_safe_numeric_values_missing_col(self):
        row = pd.Series({'concentration[s1]': 100.0})
        result = _safe_numeric_values(row, ['concentration[s1]', 'concentration[s9]'])
        assert len(result) == 1

    def test_safe_numeric_values_string_number(self):
        row = pd.Series({'concentration[s1]': '100.5'})
        result = _safe_numeric_values(row, ['concentration[s1]'])
        assert result[0] == pytest.approx(100.5)

    def test_safe_numeric_values_non_numeric(self):
        row = pd.Series({'concentration[s1]': 'abc'})
        result = _safe_numeric_values(row, ['concentration[s1]'])
        assert len(result) == 0

    def test_compute_zero_adj_means_basic(self):
        ctrl = np.array([100.0, 110.0, 105.0])
        exp = np.array([200.0, 210.0, 205.0])
        mean_c, mean_e = _compute_zero_adj_means(ctrl, exp)
        assert mean_c == pytest.approx(105.0)
        assert mean_e == pytest.approx(205.0)

    def test_compute_zero_adj_means_with_zeros(self):
        ctrl = np.array([0.0, 100.0, 100.0])
        exp = np.array([200.0, 200.0, 200.0])
        mean_c, mean_e = _compute_zero_adj_means(ctrl, exp)
        # Zero replaced with small value, so mean_c > 0
        assert mean_c > 0
        assert mean_c < 100  # less than without replacement

    def test_compute_zero_adj_means_all_zeros(self):
        ctrl = np.array([0.0, 0.0])
        exp = np.array([0.0, 0.0])
        mean_c, mean_e = _compute_zero_adj_means(ctrl, exp)
        assert mean_c == 0.0
        assert mean_e == 0.0

    def test_compute_zero_adj_means_with_nan(self):
        ctrl = np.array([np.nan, 100.0])
        exp = np.array([200.0, np.nan])
        mean_c, mean_e = _compute_zero_adj_means(ctrl, exp)
        assert mean_c == pytest.approx(100.0)
        assert mean_e == pytest.approx(200.0)

    def test_determine_exclusion_reason_no_control(self):
        row = pd.Series({'concentration[s1]': np.nan, 'concentration[s4]': 100.0})
        reason = _determine_exclusion_reason(
            row, ['concentration[s1]'], ['concentration[s4]']
        )
        assert 'control' in reason.lower() or 'replicate' in reason.lower()

    def test_determine_exclusion_reason_all_zeros(self):
        row = pd.Series({
            'concentration[s1]': 0, 'concentration[s2]': 0,
            'concentration[s4]': 0, 'concentration[s5]': 0,
        })
        reason = _determine_exclusion_reason(
            row, ['concentration[s1]', 'concentration[s2]'],
            ['concentration[s4]', 'concentration[s5]'],
        )
        assert 'zero' in reason.lower()

    def test_determine_exclusion_reason_insufficient_replicates(self):
        row = pd.Series({
            'concentration[s1]': 100.0,
            'concentration[s4]': 200.0, 'concentration[s5]': 210.0,
        })
        reason = _determine_exclusion_reason(
            row, ['concentration[s1]'],
            ['concentration[s4]', 'concentration[s5]'],
        )
        assert 'replicate' in reason.lower() or 'Insufficient' in reason

    def test_filter_significant_hide_false(self):
        vdf = pd.DataFrame({
            'FoldChange': [2.0, 0.1],
            '-log10(adjusted_pValue)': [3.0, 0.5],
        })
        result = _filter_significant(vdf, 1.0, 1.3, '-log10(adjusted_pValue)', False)
        assert len(result) == 2

    def test_filter_significant_hide_true(self):
        vdf = pd.DataFrame({
            'FoldChange': [2.0, 0.1, -2.0],
            '-log10(adjusted_pValue)': [3.0, 0.5, 3.0],
        })
        result = _filter_significant(vdf, 1.0, 1.3, '-log10(adjusted_pValue)', True)
        # Only FoldChange > 1.0 or < -1.0 AND -log10(p) >= 1.3
        assert len(result) == 2

    def test_has_overlap_no_overlap(self):
        boxes = [{'left': 0, 'right': 1, 'bottom': 0, 'top': 1}]
        assert not _has_overlap(boxes, 5, 6, 5, 6)

    def test_has_overlap_with_overlap(self):
        boxes = [{'left': 0, 'right': 1, 'bottom': 0, 'top': 1}]
        assert _has_overlap(boxes, 0.5, 1.5, 0.5, 1.5)

    def test_has_overlap_empty_boxes(self):
        assert not _has_overlap([], 0, 1, 0, 1)


# ═══════════════════════════════════════════════════════════════════════
# Type Coercion
# ═══════════════════════════════════════════════════════════════════════


class TestTypeCoercion:
    """Verify service handles various numeric dtypes correctly."""

    def test_integer_concentrations(self):
        df = _make_df(
            lipids=['PC(16:0)'],
            classes=['PC'],
            n_samples=6,
            values_per_sample=[[100], [110], [105], [500], [510], [505]],
        )
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert isinstance(result, VolcanoData)
        assert len(result.volcano_df) == 1

    def test_float32_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': np.array([100], dtype=np.float32),
            'concentration[s2]': np.array([110], dtype=np.float32),
            'concentration[s3]': np.array([105], dtype=np.float32),
            'concentration[s4]': np.array([500], dtype=np.float32),
            'concentration[s5]': np.array([510], dtype=np.float32),
            'concentration[s6]': np.array([505], dtype=np.float32),
        })
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert isinstance(result, VolcanoData)
        assert result.volcano_df['mean_control'].iloc[0] > 0

    def test_int64_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': np.array([100], dtype=np.int64),
            'concentration[s2]': np.array([110], dtype=np.int64),
            'concentration[s3]': np.array([105], dtype=np.int64),
            'concentration[s4]': np.array([500], dtype=np.int64),
            'concentration[s5]': np.array([510], dtype=np.int64),
            'concentration[s6]': np.array([505], dtype=np.int64),
        })
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert isinstance(result, VolcanoData)

    def test_object_dtype_coerced(self):
        """Object dtype columns with numeric strings should be coerced."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': pd.array(['100.0'], dtype='object'),
            'concentration[s2]': pd.array(['110.0'], dtype='object'),
            'concentration[s3]': pd.array(['105.0'], dtype='object'),
            'concentration[s4]': pd.array(['500.0'], dtype='object'),
            'concentration[s5]': pd.array(['510.0'], dtype='object'),
            'concentration[s6]': pd.array(['505.0'], dtype='object'),
        })
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        # Should work because _safe_numeric_values converts via float()
        assert isinstance(result, VolcanoData)
        assert result.volcano_df['mean_control'].iloc[0] > 0

    def test_mixed_int_float_columns(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100],
            'concentration[s2]': [110.5],
            'concentration[s3]': [105],
            'concentration[s4]': [500.5],
            'concentration[s5]': [510],
            'concentration[s6]': [505.5],
        })
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert isinstance(result, VolcanoData)

    def test_full_pipeline_with_int_data(self):
        """End-to-end: int data → prepare → plot."""
        df = _make_df(
            lipids=['PC(16:0)', 'PE(18:1)'],
            classes=['PC', 'PE'],
            n_samples=6,
            values_per_sample=[
                [100, 200], [110, 210], [105, 205],
                [500, 400], [510, 410], [505, 405],
            ],
        )
        stat_results = _make_stat_results(
            lipids=['PC(16:0)', 'PE(18:1)'],
            effect_sizes=[2.0, 1.0],
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        colors = VolcanoPlotterService.generate_color_mapping(['PC', 'PE'])
        fig = VolcanoPlotterService.create_volcano_plot(vd, colors)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


# ═══════════════════════════════════════════════════════════════════════
# Immutability
# ═══════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Verify that input DataFrames are not modified by service methods."""

    def test_prepare_preserves_input_df(self, simple_df, simple_stat_results):
        df_copy = simple_df.copy()
        VolcanoPlotterService.prepare_volcano_data(
            simple_df, simple_stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        pd.testing.assert_frame_equal(simple_df, df_copy)

    def test_prepare_with_zeros_preserves_input(self):
        df = _make_df(
            lipids=['PC(16:0)'],
            classes=['PC'],
            n_samples=6,
            values_per_sample=[[0], [100], [100], [200], [200], [200]],
        )
        df_copy = df.copy()
        stat_results = _make_stat_results(lipids=['PC(16:0)'])
        VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        pd.testing.assert_frame_equal(df, df_copy)

    def test_create_volcano_plot_preserves_data(self, simple_volcano_data, simple_color_mapping):
        vdf_copy = simple_volcano_data.volcano_df.copy()
        VolcanoPlotterService.create_volcano_plot(
            simple_volcano_data, simple_color_mapping,
        )
        pd.testing.assert_frame_equal(simple_volcano_data.volcano_df, vdf_copy)

    def test_conc_vs_fc_preserves_data(self, simple_volcano_data, simple_color_mapping):
        vdf_copy = simple_volcano_data.volcano_df.copy()
        VolcanoPlotterService.create_concentration_vs_fc_plot(
            simple_volcano_data, simple_color_mapping,
        )
        pd.testing.assert_frame_equal(simple_volcano_data.volcano_df, vdf_copy)

    def test_distribution_plot_preserves_input(self, simple_df, experiment_2x3):
        df_copy = simple_df.copy()
        fig = VolcanoPlotterService.create_distribution_plot(
            simple_df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control'],
            experiment=experiment_2x3,
        )
        pd.testing.assert_frame_equal(simple_df, df_copy)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Large Dataset
# ═══════════════════════════════════════════════════════════════════════


class TestLargeDataset:
    """Stress tests with many lipids and classes."""

    def test_100_lipids_prepare(self):
        n = 100
        lipids = [f'Lipid_{i}' for i in range(n)]
        classes = [f'Class{i % 10}' for i in range(n)]
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
        })
        for s in range(1, 7):
            df[f'concentration[s{s}]'] = rng.uniform(10, 1000, n)

        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=rng.uniform(0.001, 0.1, n).tolist(),
            effect_sizes=rng.uniform(-3, 3, n).tolist(),
        )
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == n
        assert result.removed_lipids_df.empty

    def test_100_lipids_chart_renders(self):
        n = 100
        lipids = [f'Lipid_{i}' for i in range(n)]
        classes = [f'Class{i % 10}' for i in range(n)]
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
        })
        for s in range(1, 7):
            df[f'concentration[s{s}]'] = rng.uniform(10, 1000, n)

        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=rng.uniform(0.001, 0.1, n).tolist(),
            effect_sizes=rng.uniform(-3, 3, n).tolist(),
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        unique_classes = list(set(classes))
        colors = VolcanoPlotterService.generate_color_mapping(unique_classes)
        fig = VolcanoPlotterService.create_volcano_plot(vd, colors)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(unique_classes)

    def test_100_lipids_with_labels(self):
        n = 100
        lipids = [f'Lipid_{i}' for i in range(n)]
        classes = [f'Class{i % 5}' for i in range(n)]
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
        })
        for s in range(1, 7):
            df[f'concentration[s{s}]'] = rng.uniform(10, 1000, n)

        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=rng.uniform(0.001, 0.1, n).tolist(),
            effect_sizes=rng.uniform(-3, 3, n).tolist(),
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        colors = VolcanoPlotterService.generate_color_mapping(list(set(classes)))
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors, top_n_labels=10,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        assert len(text_annotations) == 10

    def test_500_lipids_prepare(self):
        """500 lipids should not crash."""
        n = 500
        lipids = [f'L{i}' for i in range(n)]
        classes = [f'C{i % 20}' for i in range(n)]
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
        })
        for s in range(1, 7):
            df[f'concentration[s{s}]'] = rng.uniform(1, 5000, n)

        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=rng.uniform(0.0001, 0.5, n).tolist(),
            effect_sizes=rng.uniform(-5, 5, n).tolist(),
        )
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert len(result.volcano_df) == n

    def test_concentration_vs_fc_large(self):
        n = 100
        lipids = [f'Lipid_{i}' for i in range(n)]
        classes = [f'C{i % 5}' for i in range(n)]
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'LipidMolec': lipids,
            'ClassKey': classes,
        })
        for s in range(1, 7):
            df[f'concentration[s{s}]'] = rng.uniform(10, 1000, n)

        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=rng.uniform(0.001, 0.1, n).tolist(),
            effect_sizes=rng.uniform(-3, 3, n).tolist(),
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        colors = VolcanoPlotterService.generate_color_mapping(list(set(classes)))
        fig, summary_df = VolcanoPlotterService.create_concentration_vs_fc_plot(
            vd, colors,
        )
        assert isinstance(fig, go.Figure)
        assert len(summary_df) == n


# ═══════════════════════════════════════════════════════════════════════
# Multi-condition scenarios
# ═══════════════════════════════════════════════════════════════════════


class TestMultiCondition:
    """Tests with 3+ conditions (volcano still uses 2 at a time)."""

    def test_three_condition_experiment_uses_two(self, experiment_3x2):
        """Volcano plot uses only 2 conditions at a time."""
        df = _make_df(
            lipids=['PC(16:0)', 'PE(18:1)'],
            classes=['PC', 'PE'],
            n_samples=6,
            values_per_sample=[
                [100, 200], [110, 210],  # s1-s2: Control
                [500, 400], [510, 410],  # s3-s4: Treatment
                [300, 350], [310, 360],  # s5-s6: Vehicle
            ],
        )
        stat_results = _make_stat_results(lipids=['PC(16:0)', 'PE(18:1)'])
        # Use Control vs Treatment
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2'],
            experimental_samples=['s3', 's4'],
        )
        assert len(result.volcano_df) == 2

    def test_distribution_plot_three_conditions(self, experiment_3x2):
        df = _make_df(
            lipids=['PC(16:0)'],
            classes=['PC'],
            n_samples=6,
        )
        fig = VolcanoPlotterService.create_distribution_plot(
            df,
            selected_lipids=['PC(16:0)'],
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            experiment=experiment_3x2,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Significance levels (*, **, ***)
# ═══════════════════════════════════════════════════════════════════════


class TestSignificanceLevels:
    """Verify significance is correctly reflected in volcano_df."""

    @pytest.mark.parametrize("p_val,expected_sig", [
        (0.001, True),
        (0.01, True),
        (0.03, True),
        (0.049, True),
        (0.051, False),
        (0.5, False),
    ])
    def test_significance_flag_by_p_value(self, p_val, expected_sig):
        df = _make_df(lipids=['L1'], classes=['PC'], n_samples=6)
        stat_results = _make_stat_results(
            lipids=['L1'],
            p_values=[p_val],
            adjusted_p_values=[p_val],
            significant=[expected_sig],
        )
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert result.volcano_df['significant'].iloc[0] == expected_sig

    @pytest.mark.parametrize("fc,expected_fc", [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (2.5, 2.5),
        (-3.0, -3.0),
    ])
    def test_fold_change_values(self, fc, expected_fc):
        df = _make_df(lipids=['L1'], classes=['PC'], n_samples=6)
        stat_results = _make_stat_results(
            lipids=['L1'],
            effect_sizes=[fc],
        )
        result = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        assert result.volcano_df['FoldChange'].iloc[0] == pytest.approx(expected_fc)


# ═══════════════════════════════════════════════════════════════════════
# Additional Labels
# ═══════════════════════════════════════════════════════════════════════


class TestAdditionalLabels:
    """Tests for additional_labels parameter in create_volcano_plot.

    These tests verify that user-selected lipids are labeled on the plot
    independently of the top_n_labels ranking, and that the combined set
    (top N + additional) is rendered correctly.
    """

    @pytest.fixture
    def five_lipid_data(self):
        """5 lipids with distinct raw and adjusted p-values."""
        lipids = ['L1', 'L2', 'L3', 'L4', 'L5']
        classes = ['PC', 'PC', 'PE', 'PE', 'SM']
        df = _make_df(lipids=lipids, classes=classes, n_samples=6)
        # Raw p-values rank: L1 < L2 < L3 < L4 < L5
        # Adjusted p-values rank: L3 < L1 < L5 < L2 < L4  (different order)
        stat_results = _make_stat_results(
            lipids=lipids,
            p_values=[0.001, 0.005, 0.01, 0.05, 0.10],
            adjusted_p_values=[0.005, 0.08, 0.002, 0.20, 0.03],
            effect_sizes=[2.0, -1.5, 1.0, -0.5, 0.3],
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        colors = VolcanoPlotterService.generate_color_mapping(
            list(set(classes)),
        )
        return vd, colors

    def test_additional_labels_only(self, five_lipid_data):
        """additional_labels alone (no top_n) should label specified lipids."""
        vd, colors = five_lipid_data
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=0,
            additional_labels=['L4', 'L5'],
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = {a.text for a in text_annotations}
        assert labeled == {'L4', 'L5'}

    def test_additional_labels_combined_with_top_n(self, five_lipid_data):
        """top_n + additional_labels should produce the union (no duplicates)."""
        vd, colors = five_lipid_data
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=2,
            use_adjusted_p=True,
            additional_labels=['L1', 'L4'],
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = {a.text for a in text_annotations}
        # Top 2 by adjusted p: L3 (0.002), L1 (0.005)
        # Additional: L1 (already in top 2), L4
        assert labeled == {'L3', 'L1', 'L4'}

    def test_additional_labels_deduplicates(self, five_lipid_data):
        """A lipid in both top N and additional should appear only once."""
        vd, colors = five_lipid_data
        # Top 1 by adjusted p is L3
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=1,
            use_adjusted_p=True,
            additional_labels=['L3'],  # duplicate
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = [a.text for a in text_annotations]
        assert labeled.count('L3') == 1

    def test_additional_labels_nonexistent_ignored(self, five_lipid_data):
        """Lipids not in the data should be silently skipped."""
        vd, colors = five_lipid_data
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=0,
            additional_labels=['L1', 'NONEXISTENT'],
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = {a.text for a in text_annotations}
        assert labeled == {'L1'}

    def test_additional_labels_custom_positions_applied(self, five_lipid_data):
        """Custom positions should work for additional_labels lipids."""
        vd, colors = five_lipid_data
        custom = {'L4': (2.0, 3.0)}
        fig_with = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=0,
            additional_labels=['L4'],
            custom_label_positions=custom,
        )
        fig_without = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=0,
            additional_labels=['L4'],
        )
        ann_with = [a for a in fig_with.layout.annotations if a.text == 'L4'][0]
        ann_without = [a for a in fig_without.layout.annotations if a.text == 'L4'][0]
        # The custom offset should change the label position
        assert ann_with.x != ann_without.x or ann_with.y != ann_without.y

    def test_top_n_uses_adjusted_p_when_specified(self, five_lipid_data):
        """Top N should rank by adjusted p-values when use_adjusted_p=True."""
        vd, colors = five_lipid_data
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=2,
            use_adjusted_p=True,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = {a.text for a in text_annotations}
        # Top 2 by adjusted p: L3 (0.002), L1 (0.005)
        assert labeled == {'L3', 'L1'}

    def test_top_n_uses_raw_p_when_not_adjusted(self, five_lipid_data):
        """Top N should rank by raw p-values when use_adjusted_p=False."""
        vd, colors = five_lipid_data
        fig = VolcanoPlotterService.create_volcano_plot(
            vd, colors,
            top_n_labels=2,
            use_adjusted_p=False,
        )
        text_annotations = [a for a in fig.layout.annotations if a.text]
        labeled = {a.text for a in text_annotations}
        # Top 2 by raw p: L1 (0.001), L2 (0.005)
        assert labeled == {'L1', 'L2'}
