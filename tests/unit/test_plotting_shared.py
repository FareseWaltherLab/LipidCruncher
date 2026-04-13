"""
Tests for shared plotting utilities (_shared.py) and cross-cutting concerns:
- p_value_to_marker (Fix #2 / #7: parametrize)
- Color mapping utilities (Fix #10)
- Input validation (Fix #4)
- Figure cleanup for matplotlib plotters (Fix #9)
- Hover text / annotation content validation (Fix #8)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tests.conftest import make_experiment

from app.services.plotting._shared import (
    generate_class_color_mapping,
    generate_condition_color_mapping,
    get_effective_p_value,
    p_value_to_marker,
    validate_dataframe,
)
from app.services.plotting.box_plot import BoxPlotService
from app.services.plotting.bqc_plotter import BQCPlotterService
from app.services.plotting.retention_time import RetentionTimePlotterService
from app.services.plotting.volcano_plot import (
    VolcanoData,
    VolcanoPlotterService,
)
from app.services.plotting.abundance_bar_chart import (
    BarChartData,
    BarChartPlotterService,
    _p_value_to_marker,
)
from app.services.plotting.correlation import CorrelationPlotterService
from app.services.statistical_testing import (
    StatisticalTestResult,
    StatisticalTestSummary,
)


# ═══════════════════════════════════════════════════════════════════════
# Fix #7: Parametrized p_value_to_marker tests
# ═══════════════════════════════════════════════════════════════════════


class TestPValueToMarker:
    """Parametrized tests for p_value_to_marker — shared utility."""

    @pytest.mark.parametrize("p_value,expected", [
        (0.0001, '***'),
        (0.0005, '***'),
        (0.001, '**'),   # boundary: 0.001 is NOT < 0.001
        (0.005, '**'),
        (0.01, '*'),      # boundary: 0.01 is NOT < 0.01
        (0.03, '*'),
        (0.049, '*'),
        (0.05, ''),       # boundary: 0.05 is NOT < 0.05
        (0.1, ''),
        (0.5, ''),
        (1.0, ''),
    ])
    def test_p_value_thresholds(self, p_value, expected):
        assert p_value_to_marker(p_value) == expected

    def test_nan_returns_empty(self):
        assert p_value_to_marker(float('nan')) == ''

    def test_backward_compat_alias(self):
        """The re-exported _p_value_to_marker should behave identically."""
        assert _p_value_to_marker(0.001) == p_value_to_marker(0.001)
        assert _p_value_to_marker(0.05) == p_value_to_marker(0.05)
        assert _p_value_to_marker(0.0001) == p_value_to_marker(0.0001)


class TestGetEffectivePValue:
    """Tests for get_effective_p_value helper."""

    def test_prefers_adjusted_when_available(self):
        result = StatisticalTestResult(
            test_name='t-test', statistic=2.0,
            p_value=0.05, adjusted_p_value=0.02,
            significant=True, group_key='test',
        )
        assert get_effective_p_value(result) == 0.02

    def test_falls_back_to_raw_when_adjusted_is_none(self):
        result = StatisticalTestResult(
            test_name='t-test', statistic=2.0,
            p_value=0.05, adjusted_p_value=None,
            significant=True, group_key='test',
        )
        assert get_effective_p_value(result) == 0.05

    def test_falls_back_to_raw_when_adjusted_is_nan(self):
        result = StatisticalTestResult(
            test_name='t-test', statistic=2.0,
            p_value=0.05, adjusted_p_value=float('nan'),
            significant=True, group_key='test',
        )
        assert get_effective_p_value(result) == 0.05


# ═══════════════════════════════════════════════════════════════════════
# Fix #10: Color mapping tests
# ═══════════════════════════════════════════════════════════════════════


class TestColorMappings:
    """Tests for centralized color utilities."""

    def test_condition_colors_returns_dict(self):
        result = generate_condition_color_mapping(['Control', 'Treatment'])
        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'Control' in result
        assert 'Treatment' in result

    def test_class_colors_returns_dict(self):
        result = generate_class_color_mapping(['PC', 'PE', 'SM'])
        assert isinstance(result, dict)
        assert len(result) == 3

    @pytest.mark.parametrize("n_items", [1, 5, 15, 25])
    def test_color_cycling_handles_many_items(self, n_items):
        """Colors should cycle without error for any count."""
        items = [f'Item_{i}' for i in range(n_items)]
        result = generate_class_color_mapping(items)
        assert len(result) == n_items
        # All values are hex color strings
        for color in result.values():
            assert isinstance(color, str)

    def test_empty_list_returns_empty_dict(self):
        assert generate_condition_color_mapping([]) == {}
        assert generate_class_color_mapping([]) == {}

    def test_consistency_across_calls(self):
        """Same input should produce same output."""
        items = ['PC', 'PE', 'SM']
        assert generate_class_color_mapping(items) == generate_class_color_mapping(items)


# ═══════════════════════════════════════════════════════════════════════
# Fix #4: Input validation tests
# ═══════════════════════════════════════════════════════════════════════


class TestValidateDataframe:
    """Tests for the validate_dataframe utility."""

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-null"):
            validate_dataframe(None, ['col'])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(pd.DataFrame(), ['col'])

    def test_missing_columns_raises(self):
        df = pd.DataFrame({'A': [1]})
        with pytest.raises(ValueError, match="missing required columns.*B"):
            validate_dataframe(df, ['A', 'B'])

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame({'A': [1], 'B': [2]})
        validate_dataframe(df, ['A', 'B'])  # should not raise


class TestBoxPlotInputValidation:
    """Tests for input validation added to BoxPlotService."""

    def test_create_mean_area_df_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BoxPlotService.create_mean_area_df(pd.DataFrame(), ['s1'])

    def test_create_mean_area_df_empty_samples_raises(self):
        df = pd.DataFrame({'concentration[s1]': [1]})
        with pytest.raises(ValueError, match="empty"):
            BoxPlotService.create_mean_area_df(df, [])

    def test_create_mean_area_df_missing_columns_raises(self):
        df = pd.DataFrame({'concentration[s1]': [1]})
        with pytest.raises(ValueError, match="Missing concentration"):
            BoxPlotService.create_mean_area_df(df, ['s1', 's2'])

    def test_calculate_missing_values_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BoxPlotService.calculate_missing_values_percentage(pd.DataFrame())


class TestBQCPlotterInputValidation:
    """Tests for input validation added to BQCPlotterService."""

    def test_prepare_dataframe_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BQCPlotterService.prepare_dataframe_for_plot(
                pd.DataFrame(), ['col1']
            )

    def test_prepare_dataframe_missing_columns_raises(self):
        df = pd.DataFrame({'A': [1, 2]})
        with pytest.raises(ValueError, match="Missing concentration"):
            BQCPlotterService.prepare_dataframe_for_plot(df, ['B'])

    def test_generate_cov_data_invalid_index_raises(self):
        df = pd.DataFrame({'concentration[s1]': [1, 2], 'LipidMolec': ['A', 'B']})
        with pytest.raises(ValueError, match="out of range"):
            BQCPlotterService.generate_cov_plot_data(
                df, [['s1']], bqc_sample_index=5
            )


class TestRetentionTimeInputValidation:
    """Tests for input validation added to RetentionTimePlotterService."""

    def test_plot_single_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RetentionTimePlotterService.plot_single_retention(pd.DataFrame())

    def test_plot_single_missing_column_raises(self):
        df = pd.DataFrame({'ClassKey': ['PC'], 'BaseRt': [1.0]})
        with pytest.raises(ValueError, match="Missing required column.*CalcMass"):
            RetentionTimePlotterService.plot_single_retention(df)

    def test_plot_multi_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RetentionTimePlotterService.plot_multi_retention(
                pd.DataFrame(), ['PC']
            )

    def test_plot_multi_empty_classes_raises(self):
        df = pd.DataFrame({
            'ClassKey': ['PC'], 'BaseRt': [1.0],
            'CalcMass': [500.0], 'LipidMolec': ['PC 16:0'],
        })
        with pytest.raises(ValueError, match="At least one class"):
            RetentionTimePlotterService.plot_multi_retention(df, [])


# ═══════════════════════════════════════════════════════════════════════
# Fix #9: Figure cleanup tests for matplotlib plotters
# ═══════════════════════════════════════════════════════════════════════


class TestMatplotlibFigureCleanup:
    """Verify matplotlib figures are closed after creation to prevent leaks."""

    def test_correlation_figure_is_closed(self):
        """CorrelationPlotterService.render_correlation_plot closes its figure."""
        corr_df = pd.DataFrame(
            [[1.0, 0.9], [0.9, 1.0]],
            columns=['s1', 's2'], index=['s1', 's2'],
        )
        fig = CorrelationPlotterService.render_correlation_plot(
            corr_df, 0.5, 0.7, 'Control'
        )
        assert isinstance(fig, plt.Figure)
        # Figure should be closed (not in active figure list)
        assert fig.number not in plt.get_fignums()

    def test_volcano_distribution_figure_is_closed(self):
        """VolcanoPlotterService.create_distribution_plot closes its figure."""
        experiment = make_experiment(2, 3)
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0', 'PE 18:1'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [105, 205],
            'concentration[s4]': [500, 400],
            'concentration[s5]': [510, 410],
            'concentration[s6]': [505, 405],
        })
        fig = VolcanoPlotterService.create_distribution_plot(
            df, ['PC 16:0'], ['Control'], experiment
        )
        assert isinstance(fig, plt.Figure)
        # Figure should be closed (not in active figure list)
        assert fig.number not in plt.get_fignums()


# ═══════════════════════════════════════════════════════════════════════
# Fix #8: Hover text and annotation content validation
# ═══════════════════════════════════════════════════════════════════════


class TestHoverTextContent:
    """Verify hover templates contain expected content."""

    def test_bqc_scatter_hover_contains_species(self):
        """BQC scatter hover should mention Species, Mean, CoV."""
        mean_vals = np.array([100.0, 200.0])
        cov_vals = np.array([10.0, 50.0])
        species = np.array(['PC 16:0', 'PE 18:1'])
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            mean_vals, cov_vals, species, 30.0
        )
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                assert 'Species' in trace.hovertemplate
                assert 'Mean concentration' in trace.hovertemplate
                assert 'CoV' in trace.hovertemplate

    def test_retention_time_hover_contains_mass_and_rt(self):
        """RT plot hover should mention Mass, Retention time, Species."""
        df = pd.DataFrame({
            'ClassKey': ['PC', 'PC'],
            'BaseRt': [1.5, 2.0],
            'CalcMass': [500.0, 600.0],
            'LipidMolec': ['PC 16:0', 'PC 18:1'],
        })
        plots = RetentionTimePlotterService.plot_single_retention(df)
        fig, _ = plots[0]
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                assert 'Mass' in trace.hovertemplate
                assert 'Retention time' in trace.hovertemplate

    def test_volcano_hover_contains_lipid_and_fc(self):
        """Volcano plot hover should mention Lipid, Log2 Fold Change, p-value."""
        experiment = make_experiment(2, 3)
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100], 'concentration[s2]': [110],
            'concentration[s3]': [105],
            'concentration[s4]': [500], 'concentration[s5]': [510],
            'concentration[s6]': [505],
        })
        stat_results = StatisticalTestSummary(
            results={
                'PC 16:0': StatisticalTestResult(
                    test_name="Welch's t-test", statistic=10.0,
                    p_value=0.001, adjusted_p_value=0.002,
                    significant=True, effect_size=2.3, group_key='PC 16:0',
                ),
            },
            test_info={'correction': 'fdr_bh', 'transform': 'log10'},
        )
        vd = VolcanoPlotterService.prepare_volcano_data(
            df, stat_results,
            control_samples=['s1', 's2', 's3'],
            experimental_samples=['s4', 's5', 's6'],
        )
        colors = {'PC': '#1f77b4'}
        fig = VolcanoPlotterService.create_volcano_plot(vd, colors)
        for trace in fig.data:
            if hasattr(trace, 'hovertemplate') and trace.hovertemplate:
                assert 'Lipid' in trace.hovertemplate
                assert 'Fold Change' in trace.hovertemplate


class TestSignificanceAnnotationContent:
    """Verify significance annotations render correct markers."""

    def test_bar_chart_significance_markers(self):
        """Bar chart should show * or ** or *** for significant classes."""
        experiment = make_experiment(2, 3)
        df = pd.DataFrame({
            'LipidMolec': ['L1', 'L2'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [100, 200],
            'concentration[s2]': [110, 210],
            'concentration[s3]': [105, 205],
            'concentration[s4]': [500, 400],
            'concentration[s5]': [510, 410],
            'concentration[s6]': [505, 405],
        })
        bar_data = BarChartPlotterService.create_mean_std_data(
            df, experiment, ['Control', 'Treatment'], ['PC', 'PE']
        )
        stat_results = StatisticalTestSummary(
            results={
                'PC': StatisticalTestResult(
                    test_name='t-test', statistic=5.0,
                    p_value=0.0001, adjusted_p_value=0.0001,
                    significant=True, group_key='PC',
                ),
                'PE': StatisticalTestResult(
                    test_name='t-test', statistic=3.0,
                    p_value=0.03, adjusted_p_value=0.03,
                    significant=True, group_key='PE',
                ),
            },
        )
        fig = BarChartPlotterService.create_bar_chart(
            bar_data, 'linear scale', stat_results
        )
        annotation_texts = [a.text.strip() for a in fig.layout.annotations]
        assert '***' in annotation_texts  # PC: p=0.0001
        assert '*' in annotation_texts    # PE: p=0.03


# ═══════════════════════════════════════════════════════════════════════
# Fix #7: Additional parametrized tests for type coercion
# ═══════════════════════════════════════════════════════════════════════


class TestParametrizedTypeCoercion:
    """Parametrized type coercion tests to replace repetitive test classes."""

    @pytest.mark.parametrize("dtype", [
        np.int32, np.int64, np.float32, np.float64,
    ])
    def test_box_plot_handles_dtypes(self, dtype):
        """BoxPlotService should work with various numeric dtypes."""
        df = pd.DataFrame({
            'concentration[s1]': np.array([100, 200, 300], dtype=dtype),
            'concentration[s2]': np.array([150, 250, 350], dtype=dtype),
        })
        result = BoxPlotService.create_mean_area_df(df, ['s1', 's2'])
        assert len(result.columns) == 2

    @pytest.mark.parametrize("dtype", [
        np.int32, np.int64, np.float32, np.float64,
    ])
    def test_missing_values_handles_dtypes(self, dtype):
        """Missing values percentage should work with various dtypes."""
        df = pd.DataFrame({
            'concentration[s1]': np.array([0, 200, 0], dtype=dtype),
        })
        result = BoxPlotService.calculate_missing_values_percentage(df)
        assert pytest.approx(result[0], abs=0.1) == 66.7

    @pytest.mark.parametrize("dtype", [
        np.int32, np.int64, np.float32, np.float64,
    ])
    def test_bqc_cov_handles_dtypes(self, dtype):
        """BQC CoV calculation should work with various dtypes."""
        mean_vals = np.array([100.0, 200.0], dtype=dtype)
        cov_vals = np.array([10.0, 50.0], dtype=dtype)
        species = np.array(['PC 16:0', 'PE 18:1'])
        fig, _ = BQCPlotterService.create_cov_scatter_plot_with_threshold(
            mean_vals, cov_vals, species, 30.0
        )
        assert isinstance(fig, go.Figure)