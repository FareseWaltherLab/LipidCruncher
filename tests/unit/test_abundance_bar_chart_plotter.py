"""
Tests for BarChartPlotterService.

Covers data preparation (create_mean_std_data), chart rendering
(create_bar_chart), color mapping, significance annotations,
and edge cases.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.models.experiment import ExperimentConfig
from app.services.plotting.abundance_bar_chart import (
    BarChartData,
    BarChartPlotterService,
    CONDITION_COLORS,
    _compute_log10_stats,
    _get_mode_columns,
    _p_value_to_marker,
)
from app.services.statistical_testing import (
    StatisticalTestResult,
    StatisticalTestSummary,
)
from tests.conftest import make_experiment


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_df(classes, values_per_sample, n_samples=6):
    """Build a DataFrame with ClassKey and concentration columns.

    Args:
        classes: List of ClassKey values (one per row).
        values_per_sample: List of lists, each inner list has one value per row.
            Length must equal n_samples.
        n_samples: Number of sample columns.
    """
    data = {
        'LipidMolec': [f'Lipid_{i}' for i in range(len(classes))],
        'ClassKey': classes,
    }
    for i, vals in enumerate(values_per_sample):
        data[f'concentration[s{i + 1}]'] = vals
    return pd.DataFrame(data)


def _make_stat_results(class_results):
    """Build a StatisticalTestSummary from {class_name: p_value} dict."""
    results = {}
    for name, p_val in class_results.items():
        results[name] = StatisticalTestResult(
            test_name="Welch's t-test",
            statistic=5.0,
            p_value=p_val,
            adjusted_p_value=p_val,
            significant=p_val < 0.05,
            group_key=name,
        )
    return StatisticalTestSummary(results=results)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    """2 conditions × 3 samples each → s1-s6."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    """3 conditions × 2 samples each → s1-s6."""
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """2 classes, 6 samples (2 conditions × 3 each).

    PC: all values = 100
    PE: all values = 200
    """
    return _make_df(
        classes=['PC', 'PE'],
        values_per_sample=[
            [100, 200],   # s1
            [100, 200],   # s2
            [100, 200],   # s3
            [100, 200],   # s4
            [100, 200],   # s5
            [100, 200],   # s6
        ],
        n_samples=6,
    )


@pytest.fixture
def multi_species_df():
    """2 classes with multiple species per class, 6 samples."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)', 'PC(18:1)', 'PE(16:0)', 'PE(18:1)'],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'concentration[s1]': [50, 50, 100, 100],
        'concentration[s2]': [60, 40, 110, 90],
        'concentration[s3]': [55, 45, 105, 95],
        'concentration[s4]': [200, 200, 300, 300],
        'concentration[s5]': [210, 190, 310, 290],
        'concentration[s6]': [205, 195, 305, 295],
    })


@pytest.fixture
def different_means_df():
    """Two classes where conditions have clearly different means.

    PC: Control (s1-s3) ≈ 100, Treatment (s4-s6) ≈ 1000
    PE: Control (s1-s3) ≈ 50,  Treatment (s4-s6) ≈ 500
    """
    return _make_df(
        classes=['PC', 'PE'],
        values_per_sample=[
            [100, 50],     # s1 (Control)
            [110, 55],     # s2
            [90, 45],      # s3
            [1000, 500],   # s4 (Treatment)
            [1100, 550],   # s5
            [900, 450],    # s6
        ],
        n_samples=6,
    )


# ═══════════════════════════════════════════════════════════════════════
# create_mean_std_data — basic functionality
# ═══════════════════════════════════════════════════════════════════════


class TestCreateMeanStdData:

    def test_basic_two_conditions(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        assert isinstance(result, BarChartData)
        assert result.conditions == ['Control', 'Treatment']
        assert result.classes == ['PC', 'PE']

    def test_abundance_df_has_expected_columns(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        df = result.abundance_df
        assert 'ClassKey' in df.columns
        assert 'mean_AUC_Control' in df.columns
        assert 'std_AUC_Control' in df.columns
        assert 'log10_mean_AUC_Control' in df.columns
        assert 'log10_std_AUC_Control' in df.columns
        assert 'mean_AUC_Treatment' in df.columns

    def test_linear_mean_computation(self, simple_df, experiment_2x3):
        """With constant values, mean should equal the value and std should be 0."""
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC']
        )
        df = result.abundance_df
        assert df['mean_AUC_Control'].iloc[0] == pytest.approx(100.0)
        assert df['std_AUC_Control'].iloc[0] == pytest.approx(0.0)

    def test_linear_mean_with_different_values(self, different_means_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            different_means_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC']
        )
        df = result.abundance_df
        assert df['mean_AUC_Control'].iloc[0] == pytest.approx(100.0)
        assert df['mean_AUC_Treatment'].iloc[0] == pytest.approx(1000.0)

    def test_multi_species_sums_per_class(self, multi_species_df, experiment_2x3):
        """Species within a class should be summed per sample before stats."""
        result = BarChartPlotterService.create_mean_std_data(
            multi_species_df, experiment_2x3, ['Control'], ['PC']
        )
        df = result.abundance_df
        # PC: s1=50+50=100, s2=60+40=100, s3=55+45=100
        assert df['mean_AUC_Control'].iloc[0] == pytest.approx(100.0)

    def test_log10_stats_computed(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC']
        )
        df = result.abundance_df
        expected_log_mean = np.log10(100.0)
        assert df['log10_mean_AUC_Control'].iloc[0] == pytest.approx(expected_log_mean)

    def test_class_order_preserved(self, experiment_2x3):
        """Requested class order should be maintained in output."""
        df = _make_df(
            classes=['PE', 'PC', 'SM'],
            values_per_sample=[[10, 20, 30]] * 6,
            n_samples=6,
        )
        result = BarChartPlotterService.create_mean_std_data(
            df, experiment_2x3, ['Control'], ['SM', 'PC', 'PE']
        )
        assert result.classes == ['SM', 'PC', 'PE']

    def test_single_condition(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC', 'PE']
        )
        assert result.conditions == ['Control']
        assert 'mean_AUC_Treatment' not in result.abundance_df.columns

    def test_single_class(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC']
        )
        assert result.classes == ['PC']
        assert len(result.abundance_df) == 1


class TestCreateMeanStdDataEdgeCases:

    def test_empty_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one condition"):
            BarChartPlotterService.create_mean_std_data(
                simple_df, experiment_2x3, [], ['PC']
            )

    def test_empty_classes_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one lipid class"):
            BarChartPlotterService.create_mean_std_data(
                simple_df, experiment_2x3, ['Control'], []
            )

    def test_nonexistent_condition_skipped(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Nonexistent'], ['PC']
        )
        assert result.conditions == ['Control']

    def test_nonexistent_class_skipped(self, simple_df, experiment_2x3):
        result = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC', 'MISSING']
        )
        assert result.classes == ['PC']

    def test_all_conditions_invalid_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="No valid data"):
            BarChartPlotterService.create_mean_std_data(
                simple_df, experiment_2x3, ['Nonexistent'], ['PC']
            )

    def test_all_classes_invalid_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="No valid data"):
            BarChartPlotterService.create_mean_std_data(
                simple_df, experiment_2x3, ['Control'], ['MISSING']
            )

    def test_zero_values_log10_handled(self, experiment_2x3):
        """Zeros replaced with min_positive/10 for log10 stats."""
        df = _make_df(
            classes=['PC'],
            values_per_sample=[
                [0],       # s1: zero
                [100],     # s2
                [100],     # s3
                [100],     # s4
                [100],     # s5
                [100],     # s6
            ],
            n_samples=6,
        )
        result = BarChartPlotterService.create_mean_std_data(
            df, experiment_2x3, ['Control'], ['PC']
        )
        log_mean = result.abundance_df['log10_mean_AUC_Control'].iloc[0]
        assert not np.isnan(log_mean)

    def test_all_zeros_log10_is_nan(self, experiment_2x3):
        """All-zero class should yield NaN for log10 stats."""
        df = _make_df(
            classes=['PC'],
            values_per_sample=[[0]] * 6,
            n_samples=6,
        )
        result = BarChartPlotterService.create_mean_std_data(
            df, experiment_2x3, ['Control'], ['PC']
        )
        assert np.isnan(result.abundance_df['log10_mean_AUC_Control'].iloc[0])

    def test_single_sample_per_condition(self):
        """With 1 sample, std should be 0."""
        exp = make_experiment(2, 1)
        df = _make_df(
            classes=['PC'],
            values_per_sample=[[100], [200]],
            n_samples=2,
        )
        result = BarChartPlotterService.create_mean_std_data(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert result.abundance_df['std_AUC_Control'].iloc[0] == pytest.approx(0.0)
        assert result.abundance_df['mean_AUC_Control'].iloc[0] == pytest.approx(100.0)

    def test_three_conditions(self, experiment_3x2):
        df = _make_df(
            classes=['PC'],
            values_per_sample=[[100], [110], [200], [210], [300], [310]],
            n_samples=6,
        )
        result = BarChartPlotterService.create_mean_std_data(
            df, experiment_3x2, ['Control', 'Treatment', 'Vehicle'], ['PC']
        )
        assert len(result.conditions) == 3


# ═══════════════════════════════════════════════════════════════════════
# create_bar_chart — rendering
# ═══════════════════════════════════════════════════════════════════════


class TestCreateBarChart:

    def test_returns_figure(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_condition(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert len(fig.data) == 2  # One trace per condition

    def test_trace_names_match_conditions(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        trace_names = [t.name for t in fig.data]
        assert 'Control' in trace_names
        assert 'Treatment' in trace_names

    def test_horizontal_bars(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert fig.data[0].orientation == 'h'

    def test_linear_scale_values(self, different_means_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            different_means_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        control_trace = next(t for t in fig.data if t.name == 'Control')
        assert control_trace.x[0] == pytest.approx(100.0)

    def test_log10_scale_values(self, different_means_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            different_means_df, experiment_2x3,
            ['Control', 'Treatment'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'log10 scale')
        control_trace = next(t for t in fig.data if t.name == 'Control')
        assert control_trace.x[0] == pytest.approx(np.log10(100.0), abs=0.1)

    def test_error_bars_present(self, multi_species_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            multi_species_df, experiment_2x3, ['Control'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert fig.data[0].error_x is not None
        assert fig.data[0].error_x.visible is True

    def test_invalid_mode_raises(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC']
        )
        with pytest.raises(ValueError, match="Invalid mode"):
            BarChartPlotterService.create_bar_chart(bar_data, 'bad_mode')

    def test_title_contains_scale(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert 'Linear' in fig.layout.title.text

        fig2 = BarChartPlotterService.create_bar_chart(bar_data, 'log10 scale')
        assert 'Log10' in fig2.layout.title.text

    def test_title_contains_conditions(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert 'Control' in fig.layout.title.text
        assert 'Treatment' in fig.layout.title.text

    def test_dynamic_height(self, experiment_2x3):
        """More classes → taller chart."""
        classes = [f'Class{i}' for i in range(20)]
        df = _make_df(
            classes=classes,
            values_per_sample=[[100] * 20] * 6,
            n_samples=6,
        )
        bar_data = BarChartPlotterService.create_mean_std_data(
            df, experiment_2x3, ['Control'], classes
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert fig.layout.height >= 400
        assert fig.layout.height >= 20 * 30

    def test_yaxis_reversed(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC', 'PE']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert fig.layout.yaxis.autorange == 'reversed'


# ═══════════════════════════════════════════════════════════════════════
# Significance annotations
# ═══════════════════════════════════════════════════════════════════════


class TestSignificanceAnnotations:

    def test_significant_class_gets_annotation(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC', 'PE']
        )
        stats = _make_stat_results({'PC': 0.0001, 'PE': 0.8})
        fig = BarChartPlotterService.create_bar_chart(
            bar_data, 'linear scale', stat_results=stats
        )
        annotations = fig.layout.annotations
        texts = [a.text for a in annotations]
        assert '***' in texts  # PC p=0.0001 < 0.001
        # PE should NOT have annotation
        assert texts.count('***') + texts.count('**') + texts.count('*') == 1

    def test_three_star_levels(self):
        assert _p_value_to_marker(0.0001) == '***'
        assert _p_value_to_marker(0.005) == '**'
        assert _p_value_to_marker(0.03) == '*'
        assert _p_value_to_marker(0.1) == ''

    def test_no_annotations_without_stats(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control'], ['PC']
        )
        fig = BarChartPlotterService.create_bar_chart(bar_data, 'linear scale')
        assert len(fig.layout.annotations) == 0

    def test_annotations_use_adjusted_p(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC']
        )
        result = StatisticalTestResult(
            test_name="test", statistic=5.0,
            p_value=0.001, adjusted_p_value=0.1,
            significant=False, group_key='PC',
        )
        stats = StatisticalTestSummary(results={'PC': result})
        fig = BarChartPlotterService.create_bar_chart(
            bar_data, 'linear scale', stat_results=stats
        )
        # adjusted p=0.1 → no marker
        assert len(fig.layout.annotations) == 0

    def test_annotations_fall_back_to_raw_p(self, simple_df, experiment_2x3):
        bar_data = BarChartPlotterService.create_mean_std_data(
            simple_df, experiment_2x3, ['Control', 'Treatment'], ['PC']
        )
        result = StatisticalTestResult(
            test_name="test", statistic=5.0,
            p_value=0.0001, adjusted_p_value=None,
            significant=True, group_key='PC',
        )
        stats = StatisticalTestSummary(results={'PC': result})
        fig = BarChartPlotterService.create_bar_chart(
            bar_data, 'linear scale', stat_results=stats
        )
        assert any(a.text == '***' for a in fig.layout.annotations)


# ═══════════════════════════════════════════════════════════════════════
# generate_color_mapping
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateColorMapping:

    def test_returns_dict(self):
        result = BarChartPlotterService.generate_color_mapping(['A', 'B'])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_consistent_order(self):
        result = BarChartPlotterService.generate_color_mapping(['A', 'B', 'C'])
        assert result['A'] == CONDITION_COLORS[0]
        assert result['B'] == CONDITION_COLORS[1]
        assert result['C'] == CONDITION_COLORS[2]

    def test_wraps_around(self):
        conditions = [f'Cond{i}' for i in range(len(CONDITION_COLORS) + 2)]
        result = BarChartPlotterService.generate_color_mapping(conditions)
        assert result[conditions[-1]] == CONDITION_COLORS[1]

    def test_single_condition(self):
        result = BarChartPlotterService.generate_color_mapping(['Only'])
        assert result['Only'] == CONDITION_COLORS[0]

    def test_empty_list(self):
        result = BarChartPlotterService.generate_color_mapping([])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════


class TestComputeLog10Stats:

    def test_positive_values(self):
        totals = pd.Series([100.0, 100.0, 100.0])
        mean, std = _compute_log10_stats(totals)
        assert mean == pytest.approx(np.log10(100.0))
        assert std == pytest.approx(0.0)

    def test_with_zero_replacement(self):
        totals = pd.Series([0.0, 100.0, 100.0])
        mean, std = _compute_log10_stats(totals)
        assert not np.isnan(mean)
        assert mean < np.log10(100.0)  # Zero replacement pulls mean down

    def test_all_zeros(self):
        totals = pd.Series([0.0, 0.0])
        mean, std = _compute_log10_stats(totals)
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_single_value(self):
        totals = pd.Series([50.0])
        mean, std = _compute_log10_stats(totals)
        assert mean == pytest.approx(np.log10(50.0))
        assert std == pytest.approx(0.0)


class TestGetModeColumns:

    def test_linear(self):
        m, s = _get_mode_columns('Control', 'linear scale')
        assert m == 'mean_AUC_Control'
        assert s == 'std_AUC_Control'

    def test_log10(self):
        m, s = _get_mode_columns('Treatment', 'log10 scale')
        assert m == 'log10_mean_AUC_Treatment'
        assert s == 'log10_std_AUC_Treatment'
