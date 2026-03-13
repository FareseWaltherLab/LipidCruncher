"""
Tests for SaturationPlotterService.

Covers FA ratio parsing, SFA/MUFA/PUFA calculation, consolidated lipid
detection, concentration and percentage plot rendering, significance
annotations, and edge cases.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.saturation_plot import (
    FA_COLORS,
    FA_TYPES,
    SINGLE_CHAIN_CLASSES,
    SaturationData,
    SaturationPlotterService,
    _calculate_percentage_df,
    _compute_y_max,
    _p_value_to_marker,
)
from app.services.statistical_testing import (
    PostHocResult,
    StatisticalTestResult,
    StatisticalTestSummary,
)
from tests.conftest import make_experiment


# =============================================================================
# Helper functions
# =============================================================================


def _make_df(
    lipids=None,
    classes=None,
    n_samples=6,
    values=None,
):
    """Build a concentration DataFrame for saturation tests."""
    if lipids is None:
        lipids = [
            'PC(16:0_18:1)',  # 1 SFA, 1 MUFA
            'PC(16:0_20:4)',  # 1 SFA, 1 PUFA
            'PE(18:0_18:2)',  # 1 SFA, 1 PUFA
        ]
    if classes is None:
        classes = ['PC', 'PC', 'PE']

    n_lipids = len(lipids)
    sample_cols = [f'concentration[s{i+1}]' for i in range(n_samples)]

    data = {'LipidMolec': lipids, 'ClassKey': classes}
    if values is not None:
        for i, col in enumerate(sample_cols):
            data[col] = values[i] if i < len(values) else [100.0] * n_lipids
    else:
        rng = np.random.RandomState(42)
        for col in sample_cols:
            data[col] = rng.uniform(50, 200, n_lipids)

    return pd.DataFrame(data)


def _make_experiment_2x3():
    """2 conditions x 3 samples = 6 samples named s1-s6."""
    return make_experiment(
        n_conditions=2,
        samples_per_condition=3,
        conditions_list=['Control', 'Treatment'],
    )


def _make_experiment_3x2():
    """3 conditions x 2 samples = 6 samples named s1-s6."""
    return make_experiment(
        n_conditions=3,
        samples_per_condition=2,
        conditions_list=['Control', 'Treatment', 'Vehicle'],
    )


def _make_stat_results(lipid_class, significant_fa=None, posthoc=None):
    """Build a StatisticalTestSummary for saturation tests."""
    results = {}
    posthoc_results = {}

    for fa in FA_TYPES:
        key = f"{lipid_class}_{fa}"
        is_sig = significant_fa and fa in significant_fa
        p_val = 0.001 if is_sig else 0.5
        results[key] = StatisticalTestResult(
            test_name="Welch's t-test",
            statistic=3.5 if is_sig else 0.5,
            p_value=p_val,
            adjusted_p_value=p_val,
            significant=is_sig,
            group_key=key,
        )

    if posthoc:
        posthoc_results = posthoc

    return StatisticalTestSummary(
        results=results,
        posthoc_results=posthoc_results,
    )


# =============================================================================
# TestCalculateFaRatios
# =============================================================================


class TestCalculateFaRatios:
    """Tests for calculate_fa_ratios() — lipid name parsing."""

    def test_two_chain_sfa_mufa(self):
        """PC(16:0_18:1) → 0.5 SFA, 0.5 MUFA, 0 PUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('PC(16:0_18:1)')
        assert result == pytest.approx((0.5, 0.5, 0.0))

    def test_two_chain_sfa_pufa(self):
        """PA(16:0_20:4) → 0.5 SFA, 0 MUFA, 0.5 PUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('PA(16:0_20:4)')
        assert result == pytest.approx((0.5, 0.0, 0.5))

    def test_three_chains(self):
        """TAG(16:0_18:1_20:4) → 1/3 each."""
        result = SaturationPlotterService.calculate_fa_ratios('TAG(16:0_18:1_20:4)')
        assert result == pytest.approx((1/3, 1/3, 1/3))

    def test_all_saturated(self):
        """PC(16:0_18:0) → 1.0 SFA."""
        result = SaturationPlotterService.calculate_fa_ratios('PC(16:0_18:0)')
        assert result == pytest.approx((1.0, 0.0, 0.0))

    def test_all_pufa(self):
        """PC(20:4_22:6) → 0 SFA, 0 MUFA, 1.0 PUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('PC(20:4_22:6)')
        assert result == pytest.approx((0.0, 0.0, 1.0))

    def test_hydroxyl_notation_stripped(self):
        """Cer(18:1;2O_16:0) → hydroxyl stripped, 0.5 MUFA, 0.5 SFA."""
        result = SaturationPlotterService.calculate_fa_ratios('Cer(18:1;2O_16:0)')
        assert result == pytest.approx((0.5, 0.5, 0.0))

    def test_complex_hydroxyl(self):
        """SM(18:0;3O_24:1) → 0.5 SFA, 0.5 MUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('SM(18:0;3O_24:1)')
        assert result == pytest.approx((0.5, 0.5, 0.0))

    def test_consolidated_format_single_chain(self):
        """PC(34:1) → single chain counted as MUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('PC(34:1)')
        assert result == pytest.approx((0.0, 1.0, 0.0))

    def test_invalid_string_returns_zeros(self):
        result = SaturationPlotterService.calculate_fa_ratios('invalid')
        assert result == (0.0, 0.0, 0.0)

    def test_empty_string_returns_zeros(self):
        result = SaturationPlotterService.calculate_fa_ratios('')
        assert result == (0.0, 0.0, 0.0)

    def test_none_returns_zeros(self):
        result = SaturationPlotterService.calculate_fa_ratios(None)
        assert result == (0.0, 0.0, 0.0)

    def test_no_parentheses_returns_zeros(self):
        result = SaturationPlotterService.calculate_fa_ratios('PC 16:0_18:1')
        assert result == (0.0, 0.0, 0.0)

    def test_four_chains(self):
        """CL(18:1_18:2_16:0_20:4) → 1 SFA, 1 MUFA, 2 PUFA."""
        result = SaturationPlotterService.calculate_fa_ratios('CL(18:1_18:2_16:0_20:4)')
        assert result == pytest.approx((0.25, 0.25, 0.5))


# =============================================================================
# TestCalculateSfaMufaPufa
# =============================================================================


class TestCalculateSfaMufaPufa:
    """Tests for calculate_sfa_mufa_pufa() — data preparation."""

    def test_returns_saturation_data(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC', 'PE']
        )
        assert isinstance(result, SaturationData)

    def test_fa_data_structure(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'PC' in result.fa_data
        for fa in FA_TYPES:
            assert fa in result.fa_data['PC']
            assert 'Control' in result.fa_data['PC'][fa]
            assert 'Treatment' in result.fa_data['PC'][fa]

    def test_sample_values_count_matches(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control'], ['PC']
        )
        # 3 samples per condition
        assert len(result.fa_data['PC']['SFA']['Control']) == 3

    def test_plot_data_has_correct_columns(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        plot_df = result.plot_data['PC']
        expected_cols = ['Condition', 'SFA_AUC', 'MUFA_AUC', 'PUFA_AUC',
                         'SFA_std', 'MUFA_std', 'PUFA_std']
        for col in expected_cols:
            assert col in plot_df.columns

    def test_condition_label_includes_sample_count(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control'], ['PC']
        )
        cond_label = result.plot_data['PC']['Condition'].iloc[0]
        assert '(n=3)' in cond_label

    def test_classes_list_populated(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC', 'PE']
        )
        assert 'PC' in result.classes
        assert 'PE' in result.classes

    def test_conditions_list_populated(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'Control' in result.conditions
        assert 'Treatment' in result.conditions

    def test_weighted_values_calculation(self):
        """Verify SFA value = sum(concentration * sfa_ratio) per sample."""
        # PC(16:0_18:0): 100% SFA → ratio (1.0, 0.0, 0.0)
        # Concentration 100 for s1 → SFA = 100*1.0 = 100
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:0)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [300.0],
        })
        exp = make_experiment(n_conditions=1, samples_per_condition=3,
                              conditions_list=['Control'])
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control'], ['PC']
        )
        sfa_values = result.fa_data['PC']['SFA']['Control']
        np.testing.assert_array_almost_equal(sfa_values, [100.0, 200.0, 300.0])
        # MUFA and PUFA should be 0
        np.testing.assert_array_almost_equal(
            result.fa_data['PC']['MUFA']['Control'], [0.0, 0.0, 0.0]
        )

    def test_multiple_species_summed(self):
        """Two species in same class: values are summed per sample."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:0)', 'PC(16:0_18:0)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [100.0, 50.0],
            'concentration[s2]': [200.0, 100.0],
            'concentration[s3]': [300.0, 150.0],
        })
        exp = make_experiment(n_conditions=1, samples_per_condition=3,
                              conditions_list=['Control'])
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control'], ['PC']
        )
        sfa_values = result.fa_data['PC']['SFA']['Control']
        np.testing.assert_array_almost_equal(sfa_values, [150.0, 300.0, 450.0])

    def test_empty_conditions_raises(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        with pytest.raises(ValueError, match="condition"):
            SaturationPlotterService.calculate_sfa_mufa_pufa(df, exp, [], ['PC'])

    def test_empty_classes_raises(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        with pytest.raises(ValueError, match="class"):
            SaturationPlotterService.calculate_sfa_mufa_pufa(
                df, exp, ['Control'], []
            )

    def test_nonexistent_class_raises(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        with pytest.raises(ValueError, match="No valid data"):
            SaturationPlotterService.calculate_sfa_mufa_pufa(
                df, exp, ['Control'], ['FAKE']
            )

    def test_single_sample_excluded_from_plot_data(self):
        """Conditions with only 1 sample should not appear in plot_data."""
        df = _make_df(n_samples=4)
        exp = make_experiment(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[3, 1],
        )
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        # Treatment has 1 sample → excluded from plot_data
        plot_df = result.plot_data['PC']
        assert len(plot_df) == 1
        assert 'Control' in plot_df['Condition'].iloc[0]


# =============================================================================
# TestIdentifyConsolidatedLipids
# =============================================================================


class TestIdentifyConsolidatedLipids:
    """Tests for identify_consolidated_lipids() — consolidated format detection."""

    def test_detects_consolidated(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(34:1)', 'PC(16:0_18:1)'],
            'ClassKey': ['PC', 'PC'],
        })
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['PC'])
        assert 'PC' in result
        assert 'PC(34:1)' in result['PC']
        assert 'PC(16:0_18:1)' not in result['PC']

    def test_detailed_format_not_flagged(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
        })
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['PC', 'PE'])
        assert result == {}

    def test_single_chain_classes_skipped(self):
        """CE, LPC, etc. are truly single-chain and should not be flagged."""
        for cls in ['CE', 'LPC', 'LPE', 'MAG', 'FFA']:
            df = pd.DataFrame({
                'LipidMolec': [f'{cls}(18:1)'],
                'ClassKey': [cls],
            })
            result = SaturationPlotterService.identify_consolidated_lipids(df, [cls])
            assert result == {}, f"{cls} should be skipped"

    def test_sphingolipid_consolidated_detected(self):
        """Cer, SM are two-chain — consolidated format should be detected."""
        df = pd.DataFrame({
            'LipidMolec': ['Cer(42:1)', 'SM(34:1)'],
            'ClassKey': ['Cer', 'SM'],
        })
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['Cer', 'SM'])
        assert 'Cer' in result
        assert 'SM' in result

    def test_empty_dataframe(self):
        df = pd.DataFrame({'LipidMolec': [], 'ClassKey': []})
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['PC'])
        assert result == {}

    def test_none_dataframe(self):
        result = SaturationPlotterService.identify_consolidated_lipids(None, ['PC'])
        assert result == {}

    def test_mixed_classes(self):
        """Only classes with consolidated lipids appear in result."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(34:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
        })
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['PC', 'PE'])
        assert 'PC' in result
        assert 'PE' not in result

    def test_no_colon_not_flagged(self):
        """Lipid without colon is not consolidated (no chain notation)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(unknown)'],
            'ClassKey': ['PC'],
        })
        result = SaturationPlotterService.identify_consolidated_lipids(df, ['PC'])
        assert result == {}


# =============================================================================
# TestConcentrationPlot
# =============================================================================


class TestConcentrationPlot:
    """Tests for create_concentration_plot() — chart rendering."""

    @pytest.fixture
    def sat_data(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        return SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC', 'PE']
        )

    def test_returns_figure(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert isinstance(fig, go.Figure)

    def test_three_traces(self, sat_data):
        """One bar trace per FA type (SFA, MUFA, PUFA)."""
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert len(fig.data) == 3

    def test_trace_names(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        trace_names = [t.name for t in fig.data]
        assert trace_names == ['SFA', 'MUFA', 'PUFA']

    def test_trace_colors(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        for trace in fig.data:
            assert trace.marker.color == FA_COLORS[trace.name]

    def test_error_bars_present(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        for trace in fig.data:
            assert trace.error_y.visible is True

    def test_title_contains_class_name(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert 'PC' in fig.layout.title.text

    def test_y_axis_starts_at_zero(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert fig.layout.yaxis.range[0] == 0

    def test_invalid_class_raises(self, sat_data):
        with pytest.raises(ValueError, match="No data"):
            SaturationPlotterService.create_concentration_plot(sat_data, 'FAKE')

    def test_barmode_is_group(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert fig.layout.barmode == 'group'

    def test_white_background(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(sat_data, 'PC')
        assert fig.layout.plot_bgcolor == 'white'


# =============================================================================
# TestConcentrationPlotWithSignificance
# =============================================================================


class TestConcentrationPlotWithSignificance:
    """Tests for significance annotations on concentration plots."""

    @pytest.fixture
    def sat_data(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        return SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )

    def test_no_annotations_without_flag(self, sat_data):
        stats = _make_stat_results('PC', significant_fa=['SFA'])
        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=False
        )
        assert len(fig.layout.shapes) == 0

    def test_annotations_with_flag(self, sat_data):
        stats = _make_stat_results('PC', significant_fa=['SFA'])
        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=True
        )
        assert len(fig.layout.shapes) > 0
        assert len(fig.layout.annotations) > 0

    def test_only_significant_fa_annotated(self, sat_data):
        stats = _make_stat_results('PC', significant_fa=['PUFA'])
        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=True
        )
        # Only 1 significant FA type → 1 annotation line
        assert len(fig.layout.shapes) == 1

    def test_all_three_fa_significant(self, sat_data):
        stats = _make_stat_results('PC', significant_fa=['SFA', 'MUFA', 'PUFA'])
        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=True
        )
        assert len(fig.layout.shapes) == 3

    def test_no_stats_provided(self, sat_data):
        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=None, show_significance=True
        )
        assert len(fig.layout.shapes) == 0


# =============================================================================
# TestPercentagePlot
# =============================================================================


class TestPercentagePlot:
    """Tests for create_percentage_plot() — stacked bar chart."""

    @pytest.fixture
    def sat_data(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        return SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )

    def test_returns_figure(self, sat_data):
        fig = SaturationPlotterService.create_percentage_plot(sat_data, 'PC')
        assert isinstance(fig, go.Figure)

    def test_three_traces(self, sat_data):
        fig = SaturationPlotterService.create_percentage_plot(sat_data, 'PC')
        assert len(fig.data) == 3

    def test_stacked_mode(self, sat_data):
        fig = SaturationPlotterService.create_percentage_plot(sat_data, 'PC')
        assert fig.layout.barmode == 'stack'

    def test_y_axis_range_0_to_100(self, sat_data):
        fig = SaturationPlotterService.create_percentage_plot(sat_data, 'PC')
        assert tuple(fig.layout.yaxis.range) == (0, 100)

    def test_title_contains_percentage(self, sat_data):
        fig = SaturationPlotterService.create_percentage_plot(sat_data, 'PC')
        assert 'Percentage' in fig.layout.title.text

    def test_invalid_class_raises(self, sat_data):
        with pytest.raises(ValueError, match="No data"):
            SaturationPlotterService.create_percentage_plot(sat_data, 'FAKE')


# =============================================================================
# TestPrivateHelpers
# =============================================================================


class TestPrivateHelpers:
    """Tests for private helper functions."""

    def test_compute_y_max(self):
        df = pd.DataFrame({
            'SFA_AUC': [100.0], 'SFA_std': [10.0],
            'MUFA_AUC': [200.0], 'MUFA_std': [20.0],
            'PUFA_AUC': [50.0], 'PUFA_std': [5.0],
        })
        assert _compute_y_max(df) == 220.0  # 200 + 20

    def test_compute_y_max_empty(self):
        df = pd.DataFrame({
            'SFA_AUC': [0.0], 'SFA_std': [0.0],
            'MUFA_AUC': [0.0], 'MUFA_std': [0.0],
            'PUFA_AUC': [0.0], 'PUFA_std': [0.0],
        })
        assert _compute_y_max(df) == 1.0  # fallback

    def test_calculate_percentage_df_sums_to_100(self):
        df = pd.DataFrame({
            'Condition': ['A'],
            'SFA_AUC': [50.0], 'MUFA_AUC': [30.0], 'PUFA_AUC': [20.0],
        })
        pct = _calculate_percentage_df(df)
        total = pct['SFA_AUC'].iloc[0] + pct['MUFA_AUC'].iloc[0] + pct['PUFA_AUC'].iloc[0]
        assert total == pytest.approx(100.0)

    def test_calculate_percentage_df_preserves_condition(self):
        df = pd.DataFrame({
            'Condition': ['Ctrl', 'Trt'],
            'SFA_AUC': [100.0, 200.0],
            'MUFA_AUC': [100.0, 100.0],
            'PUFA_AUC': [100.0, 100.0],
        })
        pct = _calculate_percentage_df(df)
        assert list(pct['Condition']) == ['Ctrl', 'Trt']

    def test_calculate_percentage_df_zero_total(self):
        df = pd.DataFrame({
            'Condition': ['A'],
            'SFA_AUC': [0.0], 'MUFA_AUC': [0.0], 'PUFA_AUC': [0.0],
        })
        pct = _calculate_percentage_df(df)
        assert pct['SFA_AUC'].iloc[0] == 0.0

    def test_p_value_to_marker_triple(self):
        assert _p_value_to_marker(0.0001) == '***'

    def test_p_value_to_marker_double(self):
        assert _p_value_to_marker(0.005) == '**'

    def test_p_value_to_marker_single(self):
        assert _p_value_to_marker(0.03) == '*'

    def test_p_value_to_marker_none(self):
        assert _p_value_to_marker(0.1) == ''


# =============================================================================
# TestColorMapping
# =============================================================================


class TestColorMapping:
    """Tests for generate_color_mapping()."""

    def test_returns_three_keys(self):
        mapping = SaturationPlotterService.generate_color_mapping()
        assert set(mapping.keys()) == {'SFA', 'MUFA', 'PUFA'}

    def test_matches_fa_colors(self):
        mapping = SaturationPlotterService.generate_color_mapping()
        assert mapping == FA_COLORS


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for the saturation plotter service."""

    def test_all_zeros_excluded(self):
        """If all concentration values are 0, class should be excluded."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [0.0],
            'concentration[s2]': [0.0],
            'concentration[s3]': [0.0],
            'concentration[s4]': [0.0],
            'concentration[s5]': [0.0],
            'concentration[s6]': [0.0],
        })
        exp = _make_experiment_2x3()
        with pytest.raises(ValueError, match="No valid data"):
            SaturationPlotterService.calculate_sfa_mufa_pufa(
                df, exp, ['Control', 'Treatment'], ['PC']
            )

    def test_single_lipid_per_class(self):
        """Works with just one species in a class."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            'concentration[s2]': [200.0],
            'concentration[s3]': [150.0],
            'concentration[s4]': [120.0],
            'concentration[s5]': [180.0],
            'concentration[s6]': [160.0],
        })
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'PC' in result.classes

    def test_three_conditions(self):
        """Works with 3 conditions."""
        df = _make_df()
        exp = _make_experiment_3x2()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment', 'Vehicle'], ['PC']
        )
        plot_df = result.plot_data['PC']
        assert len(plot_df) == 3

    def test_nonexistent_condition_ignored(self):
        df = _make_df()
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'FakeCondition'], ['PC']
        )
        # Only Control should appear
        assert len(result.plot_data['PC']) == 1

    def test_many_classes(self):
        """Handles 5+ classes without error."""
        lipids = [
            'PC(16:0_18:1)', 'PE(16:0_18:1)', 'PS(16:0_18:1)',
            'PI(16:0_18:1)', 'PA(16:0_18:1)',
        ]
        classes = ['PC', 'PE', 'PS', 'PI', 'PA']
        df = _make_df(lipids=lipids, classes=classes)
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], classes
        )
        assert len(result.classes) == 5

    def test_immutability_of_input(self):
        """Input DataFrame should not be modified."""
        df = _make_df()
        df_copy = df.copy()
        exp = _make_experiment_2x3()
        SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        pd.testing.assert_frame_equal(df, df_copy)


# =============================================================================
# TestPostHocAnnotations
# =============================================================================


class TestPostHocAnnotations:
    """Tests for multi-condition post-hoc significance annotations."""

    def test_posthoc_annotations_rendered(self):
        """3 conditions with significant posthoc → lines on plot."""
        df = _make_df()
        exp = _make_experiment_3x2()
        sat_data = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment', 'Vehicle'], ['PC']
        )

        posthoc = {
            'PC_SFA': [
                PostHocResult(
                    group1='Control', group2='Treatment',
                    p_value=0.01, adjusted_p_value=0.01,
                    significant=True, test_name="Tukey's HSD",
                ),
            ],
        }
        stats = _make_stat_results('PC', significant_fa=['SFA'], posthoc=posthoc)

        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=True
        )
        assert len(fig.layout.shapes) >= 1

    def test_nonsignificant_posthoc_not_rendered(self):
        df = _make_df()
        exp = _make_experiment_3x2()
        sat_data = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment', 'Vehicle'], ['PC']
        )

        posthoc = {
            'PC_SFA': [
                PostHocResult(
                    group1='Control', group2='Treatment',
                    p_value=0.5, adjusted_p_value=0.5,
                    significant=False, test_name="Tukey's HSD",
                ),
            ],
        }
        stats = _make_stat_results('PC', significant_fa=['SFA'], posthoc=posthoc)

        fig = SaturationPlotterService.create_concentration_plot(
            sat_data, 'PC', stat_results=stats, show_significance=True
        )
        assert len(fig.layout.shapes) == 0


# =============================================================================
# TestTypeCoercion
# =============================================================================


class TestTypeCoercion:
    """Tests for handling different numeric types."""

    def test_integer_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100],
            'concentration[s2]': [200],
            'concentration[s3]': [150],
            'concentration[s4]': [120],
            'concentration[s5]': [180],
            'concentration[s6]': [160],
        })
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'PC' in result.classes

    def test_string_concentrations(self):
        """String numbers should be coerced to float."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': ['100.0'],
            'concentration[s2]': ['200.0'],
            'concentration[s3]': ['150.0'],
            'concentration[s4]': ['120.0'],
            'concentration[s5]': ['180.0'],
            'concentration[s6]': ['160.0'],
        })
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'PC' in result.classes

    def test_numpy_int64_concentrations(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': np.array([100], dtype=np.int64),
            'concentration[s2]': np.array([200], dtype=np.int64),
            'concentration[s3]': np.array([150], dtype=np.int64),
            'concentration[s4]': np.array([120], dtype=np.int64),
            'concentration[s5]': np.array([180], dtype=np.int64),
            'concentration[s6]': np.array([160], dtype=np.int64),
        })
        exp = _make_experiment_2x3()
        result = SaturationPlotterService.calculate_sfa_mufa_pufa(
            df, exp, ['Control', 'Treatment'], ['PC']
        )
        assert 'PC' in result.classes
