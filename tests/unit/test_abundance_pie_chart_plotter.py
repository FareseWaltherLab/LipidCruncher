"""
Tests for PieChartPlotterService.

Covers data preparation (calculate_total_abundance), chart rendering
(create_pie_chart), color mapping, percentage formatting, and edge cases.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.abundance_pie_chart import (
    CLASS_COLORS,
    PieChartData,
    PieChartPlotterService,
    _format_percentage,
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


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    """2 conditions x 3 samples each -> s1-s6."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    """3 conditions x 2 samples each -> s1-s6."""
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """2 classes, 6 samples. PC=100, PE=200 per sample."""
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
def three_class_df():
    """3 classes, 6 samples. PC=100, PE=200, SM=50 per sample."""
    return _make_df(
        classes=['PC', 'PE', 'SM'],
        values_per_sample=[
            [100, 200, 50],
            [100, 200, 50],
            [100, 200, 50],
            [100, 200, 50],
            [100, 200, 50],
            [100, 200, 50],
        ],
        n_samples=6,
    )


# ═══════════════════════════════════════════════════════════════════════
# calculate_total_abundance — basic functionality
# ═══════════════════════════════════════════════════════════════════════


class TestCalculateTotalAbundance:

    def test_returns_pie_chart_data(self, simple_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        assert isinstance(result, PieChartData)
        assert result.classes == ['PC', 'PE']

    def test_abundance_df_indexed_by_class(self, simple_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        assert list(result.abundance_df.index) == ['PC', 'PE']

    def test_sums_species_per_class(self, multi_species_df, experiment_2x3):
        """Multiple species in a class should be summed."""
        result = PieChartPlotterService.calculate_total_abundance(
            multi_species_df, experiment_2x3, ['PC']
        )
        # PC s1: 50+50=100
        assert result.abundance_df.loc['PC', 'concentration[s1]'] == 100

    def test_class_order_preserved(self, three_class_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            three_class_df, experiment_2x3, ['SM', 'PE', 'PC']
        )
        assert result.classes == ['SM', 'PE', 'PC']
        assert list(result.abundance_df.index) == ['SM', 'PE', 'PC']

    def test_single_class(self, simple_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        assert result.classes == ['PC']
        assert len(result.abundance_df) == 1

    def test_all_sample_columns_present(self, simple_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        for s in experiment_2x3.full_samples_list:
            assert f'concentration[{s}]' in result.abundance_df.columns


class TestCalculateTotalAbundanceEdgeCases:

    def test_empty_classes_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one lipid class"):
            PieChartPlotterService.calculate_total_abundance(
                simple_df, experiment_2x3, []
            )

    def test_none_classes_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one lipid class"):
            PieChartPlotterService.calculate_total_abundance(
                simple_df, experiment_2x3, None
            )

    def test_nonexistent_class_skipped(self, simple_df, experiment_2x3):
        result = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'MISSING']
        )
        assert result.classes == ['PC']

    def test_all_classes_invalid_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="No data found"):
            PieChartPlotterService.calculate_total_abundance(
                simple_df, experiment_2x3, ['MISSING']
            )

    def test_no_sample_columns_raises(self, experiment_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)'],
            'ClassKey': ['PC'],
        })
        with pytest.raises(ValueError, match="No sample columns"):
            PieChartPlotterService.calculate_total_abundance(
                df, experiment_2x3, ['PC']
            )


# ═══════════════════════════════════════════════════════════════════════
# create_pie_chart — rendering
# ═══════════════════════════════════════════════════════════════════════


class TestCreatePieChart:

    def test_returns_figure_and_dataframe(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert isinstance(fig, go.Figure)
        assert isinstance(summary, pd.DataFrame)

    def test_summary_has_expected_columns(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert 'ClassKey' in summary.columns
        assert 'Total Abundance' in summary.columns
        assert 'Percentage' in summary.columns

    def test_percentages_sum_to_100(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert summary['Percentage'].sum() == pytest.approx(100.0)

    def test_sorted_by_abundance_descending(self, simple_df, experiment_2x3):
        """PE (200) should come before PC (100) in sorted output."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert summary['ClassKey'].iloc[0] == 'PE'
        assert summary['ClassKey'].iloc[1] == 'PC'

    def test_title_contains_condition(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert 'Control' in fig.layout.title.text

    def test_uses_color_mapping(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = {'PC': '#ff0000', 'PE': '#00ff00'}
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        # PE sorted first (higher abundance), PC second
        marker_colors = fig.data[0].marker.colors
        assert marker_colors[0] == '#00ff00'  # PE
        assert marker_colors[1] == '#ff0000'  # PC

    def test_missing_color_gets_fallback(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = {'PC': '#ff0000'}  # Missing PE
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        marker_colors = fig.data[0].marker.colors
        assert marker_colors[0] == '#333333'  # PE fallback

    def test_only_uses_condition_samples(self, simple_df, experiment_2x3):
        """Only samples from the specified condition should be summed."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        # PC: 100*3 = 300 for Control (s1-s3)
        assert summary['Total Abundance'].iloc[0] == pytest.approx(300.0)

    def test_no_sample_columns_raises(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        with pytest.raises(ValueError, match="No sample columns"):
            PieChartPlotterService.create_pie_chart(
                pie_data, 'Control', ['nonexistent_sample'], colors
            )

    def test_multi_species_summed(self, multi_species_df, experiment_2x3):
        """Species within a class should already be summed in pie_data."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            multi_species_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        # PC: s1(50+50)+s2(60+40)+s3(55+45) = 100+100+100 = 300
        pc_row = summary[summary['ClassKey'] == 'PC']
        assert pc_row['Total Abundance'].iloc[0] == pytest.approx(300.0)

    def test_three_classes(self, three_class_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            three_class_df, experiment_2x3, ['PC', 'PE', 'SM']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE', 'SM'])
        fig, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert len(summary) == 3
        assert len(fig.data[0].values) == 3

    def test_chart_dimensions(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert fig.layout.width == 450
        assert fig.layout.height == 300


    def test_single_pie_trace(self, simple_df, experiment_2x3):
        """Pie chart should always have exactly one trace."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)

    def test_textinfo_is_none(self, simple_df, experiment_2x3):
        """Text on pie slices should be hidden (legend shows labels)."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert fig.data[0].textinfo == 'none'

    def test_legend_title(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert fig.layout.legend.title.text == 'Lipid Classes'

    def test_different_conditions_different_totals(self, experiment_2x3):
        """Two conditions with different values should produce different totals."""
        df = _make_df(
            classes=['PC', 'PE'],
            values_per_sample=[
                [100, 200],   # s1 (Control)
                [100, 200],   # s2
                [100, 200],   # s3
                [500, 800],   # s4 (Treatment)
                [500, 800],   # s5
                [500, 800],   # s6
            ],
            n_samples=6,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, ctrl_summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        _, treat_summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Treatment', ['s4', 's5', 's6'], colors
        )
        assert ctrl_summary['Total Abundance'].sum() < treat_summary['Total Abundance'].sum()

    def test_three_conditions(self, three_class_df, experiment_3x2):
        """Should work with 3 conditions."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            three_class_df, experiment_3x2, ['PC', 'PE', 'SM']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE', 'SM'])
        for cond, samples in [
            ('Control', ['s1', 's2']),
            ('Treatment', ['s3', 's4']),
            ('Vehicle', ['s5', 's6']),
        ]:
            fig, summary = PieChartPlotterService.create_pie_chart(
                pie_data, cond, samples, colors
            )
            assert isinstance(fig, go.Figure)
            assert len(summary) == 3

    def test_labels_include_class_name_and_percentage(self, simple_df, experiment_2x3):
        """Pie labels should contain both class name and percentage."""
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, _ = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        labels = list(fig.data[0].labels)
        assert any('PE' in label and '%' in label for label in labels)
        assert any('PC' in label and '%' in label for label in labels)

    def test_many_classes(self, experiment_2x3):
        """Should handle a large number of classes."""
        n_classes = 15
        classes = [f'Class{i}' for i in range(n_classes)]
        df = _make_df(
            classes=classes,
            values_per_sample=[list(range(1, n_classes + 1))] * 6,
            n_samples=6,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment_2x3, classes
        )
        colors = PieChartPlotterService.generate_color_mapping(classes)
        fig, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert len(summary) == n_classes
        assert summary['Percentage'].sum() == pytest.approx(100.0)


class TestCreatePieChartEdgeCases:

    def test_all_zeros(self, experiment_2x3):
        """All-zero data should produce a chart with 0% everywhere."""
        df = _make_df(
            classes=['PC', 'PE'],
            values_per_sample=[[0, 0]] * 6,
            n_samples=6,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert summary['Percentage'].sum() == pytest.approx(0.0)

    def test_single_class_is_100_percent(self, simple_df, experiment_2x3):
        pie_data = PieChartPlotterService.calculate_total_abundance(
            simple_df, experiment_2x3, ['PC']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        assert summary['Percentage'].iloc[0] == pytest.approx(100.0)

    def test_single_sample(self):
        """Works with 1 sample per condition."""
        exp = make_experiment(2, 1)
        df = _make_df(
            classes=['PC', 'PE'],
            values_per_sample=[[100, 200], [150, 250]],
            n_samples=2,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, exp, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1'], colors
        )
        assert isinstance(fig, go.Figure)
        assert summary['Total Abundance'].sum() == pytest.approx(300.0)

    def test_dominant_class(self, experiment_2x3):
        """One class with vastly larger abundance than others."""
        df = _make_df(
            classes=['PC', 'PE', 'SM'],
            values_per_sample=[[10000, 1, 1]] * 6,
            n_samples=6,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment_2x3, ['PC', 'PE', 'SM']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE', 'SM'])
        _, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        pc_pct = summary[summary['ClassKey'] == 'PC']['Percentage'].iloc[0]
        assert pc_pct > 99.0
        assert summary['Percentage'].sum() == pytest.approx(100.0)

    def test_negative_values(self, experiment_2x3):
        """Negative concentrations (possible after normalization) should not crash."""
        df = _make_df(
            classes=['PC', 'PE'],
            values_per_sample=[[-10, 200]] * 6,
            n_samples=6,
        )
        pie_data = PieChartPlotterService.calculate_total_abundance(
            df, experiment_2x3, ['PC', 'PE']
        )
        colors = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        fig, summary = PieChartPlotterService.create_pie_chart(
            pie_data, 'Control', ['s1', 's2', 's3'], colors
        )
        # Should not crash; Plotly handles negative pie values
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════════════
# generate_color_mapping
# ═══════════════════════════════════════════════════════════════════════


class TestGenerateColorMapping:

    def test_returns_dict(self):
        result = PieChartPlotterService.generate_color_mapping(['PC', 'PE'])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_consistent_order(self):
        result = PieChartPlotterService.generate_color_mapping(['PC', 'PE', 'SM'])
        assert result['PC'] == CLASS_COLORS[0]
        assert result['PE'] == CLASS_COLORS[1]
        assert result['SM'] == CLASS_COLORS[2]

    def test_wraps_around(self):
        classes = [f'Class{i}' for i in range(len(CLASS_COLORS) + 2)]
        result = PieChartPlotterService.generate_color_mapping(classes)
        assert result[classes[-1]] == CLASS_COLORS[1]

    def test_single_class(self):
        result = PieChartPlotterService.generate_color_mapping(['PC'])
        assert result['PC'] == CLASS_COLORS[0]

    def test_empty_list(self):
        result = PieChartPlotterService.generate_color_mapping([])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════
# _format_percentage
# ═══════════════════════════════════════════════════════════════════════


class TestFormatPercentage:

    def test_zero(self):
        assert _format_percentage(0) == "0.0"

    def test_large_percentage(self):
        assert _format_percentage(55.3) == "55.3"

    def test_medium_percentage(self):
        assert _format_percentage(5.67) == "5.7"

    def test_small_percentage(self):
        assert _format_percentage(0.56) == "0.56"

    def test_very_small_percentage(self):
        assert _format_percentage(0.056) == "0.056"

    def test_tiny_percentage(self):
        assert _format_percentage(0.0056) == "0.0056"

    def test_scientific_notation(self):
        result = _format_percentage(0.00005)
        assert 'e' in result

    def test_one_percent(self):
        assert _format_percentage(1.0) == "1.0"

    def test_hundred_percent(self):
        assert _format_percentage(100.0) == "100.0"
