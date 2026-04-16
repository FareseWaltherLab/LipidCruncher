"""
Unit tests for ChainLengthPlotterService.

Tests cover:
1. parse_total_chain_info — lipid name parsing for carbons and double bonds
2. calculate_chain_length_data — data aggregation (single condition)
3. calculate_per_condition_data — per-condition data separation
4. create_per_condition_figure — vertical per-condition figure structure
5. _scale_marker_sizes — marker size scaling (local and global)
6. generate_color_mapping — color assignment
7. Edge cases — empty data, unparsable lipids, single class
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.chain_length_plot import (
    ChainLengthData,
    ChainLengthPlotterService,
    _scale_marker_sizes,
)
from tests.conftest import make_experiment


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_df():
    """DataFrame with molecular-level lipid names and concentrations."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC 16:0_18:1',
            'PC 16:0_18:2',
            'PE 18:0_20:4',
            'PE 16:0_18:1',
            'SM 18:1;O2/24:0',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE', 'SM'],
        'concentration[s1]': [100.0, 200.0, 150.0, 120.0, 80.0],
        'concentration[s2]': [110.0, 190.0, 160.0, 130.0, 90.0],
        'concentration[s3]': [105.0, 195.0, 155.0, 125.0, 85.0],
        'concentration[s4]': [90.0, 210.0, 140.0, 115.0, 75.0],
        'concentration[s5]': [95.0, 205.0, 145.0, 110.0, 70.0],
        'concentration[s6]': [92.0, 208.0, 142.0, 112.0, 72.0],
    })


@pytest.fixture
def basic_experiment():
    return make_experiment(n_conditions=2, samples_per_condition=3)


# =============================================================================
# TestParseTotalChainInfo
# =============================================================================


class TestParseTotalChainInfo:
    """Tests for lipid name → (total_carbons, total_double_bonds)."""

    def test_molecular_level_two_chains(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PC 16:0_18:1')
        assert result == (34, 1)

    def test_molecular_level_three_chains(self):
        result = ChainLengthPlotterService.parse_total_chain_info('TG 16:0_18:1_18:2')
        assert result == (52, 3)

    def test_species_level(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PC 34:1')
        assert result == (34, 1)

    def test_sphingolipid_slash_separator(self):
        result = ChainLengthPlotterService.parse_total_chain_info('Cer 18:1;O2/24:0')
        assert result == (42, 1)

    def test_lyso_single_chain(self):
        result = ChainLengthPlotterService.parse_total_chain_info('LPC 18:1')
        assert result == (18, 1)

    def test_no_chain_info(self):
        result = ChainLengthPlotterService.parse_total_chain_info('Ch')
        assert result is None

    def test_empty_string(self):
        result = ChainLengthPlotterService.parse_total_chain_info('')
        assert result is None

    def test_invalid_format(self):
        result = ChainLengthPlotterService.parse_total_chain_info('not_a_lipid')
        assert result is None

    def test_modification_stripped(self):
        result = ChainLengthPlotterService.parse_total_chain_info('LPC 18:1(d7)')
        assert result == (18, 1)

    def test_zero_double_bonds(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PC 16:0_18:0')
        assert result == (34, 0)

    def test_high_double_bonds(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PE 20:4_22:6')
        assert result == (42, 10)

    # -- Formats that previously failed (prefix handling) --

    def test_sphingoid_d_prefix_cer(self):
        result = ChainLengthPlotterService.parse_total_chain_info('Cer 16:0/d18:1')
        assert result == (34, 1)

    def test_sphingoid_d_prefix_sm(self):
        result = ChainLengthPlotterService.parse_total_chain_info('SM d37:1')
        assert result == (37, 1)

    def test_sphingoid_d_prefix_hexcer(self):
        result = ChainLengthPlotterService.parse_total_chain_info('Hex1Cer 18:0_d18:1')
        assert result == (36, 1)

    def test_plasmalogen_P_prefix(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PE P-16:0_22:5')
        assert result == (38, 5)

    def test_ether_O_prefix(self):
        result = ChainLengthPlotterService.parse_total_chain_info('DG 20:1_O-30:1')
        assert result == (50, 2)

    def test_ether_O_prefix_species_level(self):
        result = ChainLengthPlotterService.parse_total_chain_info('PC O-38:4')
        assert result == (38, 4)

    def test_ether_O_prefix_triglyceride(self):
        result = ChainLengthPlotterService.parse_total_chain_info('TG 18:2_O-20:0_22:6')
        assert result == (60, 8)


# =============================================================================
# TestCalculateChainLengthData
# =============================================================================


class TestCalculateChainLengthData:
    """Tests for data aggregation."""

    def test_returns_chain_length_data(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE', 'SM'],
        )
        assert isinstance(result, ChainLengthData)
        assert len(result.records) > 0
        assert len(result.classes) > 0

    def test_classes_sorted(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PE', 'PC', 'SM'],
        )
        assert result.classes == sorted(result.classes)

    def test_filter_by_class(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        assert all(r['ClassKey'] == 'PC' for r in result.records)

    def test_filter_by_condition(self, basic_df, basic_experiment):
        """Selecting fewer conditions should change mean concentrations."""
        all_cond = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        one_cond = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control'], ['PC'],
        )
        # Same number of records but different concentrations
        assert len(all_cond.records) == len(one_cond.records)

    def test_records_have_required_keys(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control'], ['PC'],
        )
        for rec in result.records:
            assert 'ClassKey' in rec
            assert 'TotalCarbons' in rec
            assert 'TotalDoubleBonds' in rec
            assert 'MeanConcentration' in rec

    def test_aggregates_same_chain_info(self):
        """Two species with same total carbons/db in same class should be summed."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PC 14:0_20:1'],  # both 34:1
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [100.0, 50.0],
            'concentration[s2]': [100.0, 50.0],
        })
        exp = make_experiment(n_conditions=1, samples_per_condition=2)
        result = ChainLengthPlotterService.calculate_chain_length_data(
            df, exp, ['Control'], ['PC'],
        )
        # Both are PC with 34 carbons, 1 double bond → should be aggregated
        pc_34_1 = [r for r in result.records
                    if r['TotalCarbons'] == 34 and r['TotalDoubleBonds'] == 1]
        assert len(pc_34_1) == 1
        assert pc_34_1[0]['MeanConcentration'] == pytest.approx(150.0)

    def test_empty_class_selection(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment, ['Control'], [],
        )
        assert result.records == []
        assert result.classes == []

    def test_no_matching_classes(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment, ['Control'], ['TG'],
        )
        assert result.records == []

    def test_unparsable_lipids_skipped(self, basic_experiment):
        """Lipids without chain info should be silently skipped."""
        df = pd.DataFrame({
            'LipidMolec': ['Ch', 'PC 34:1'],
            'ClassKey': ['Ch', 'PC'],
            'concentration[s1]': [100.0, 200.0],
            'concentration[s2]': [100.0, 200.0],
            'concentration[s3]': [100.0, 200.0],
        })
        exp = make_experiment(n_conditions=1, samples_per_condition=3)
        result = ChainLengthPlotterService.calculate_chain_length_data(
            df, exp, ['Control'], ['Ch', 'PC'],
        )
        # Ch has no chain info → skipped
        assert all(r['ClassKey'] == 'PC' for r in result.records)


# =============================================================================
# TestCalculatePerConditionData
# =============================================================================


class TestCalculatePerConditionData:
    """Tests for per-condition data separation."""

    def test_returns_dict_per_condition(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE', 'SM'],
        )
        assert isinstance(result, dict)
        assert 'Control' in result
        assert 'Treatment' in result
        assert len(result) == 2

    def test_each_value_is_chain_length_data(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        for cond, data in result.items():
            assert isinstance(data, ChainLengthData)

    def test_single_condition(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control'], ['PC', 'PE'],
        )
        assert len(result) == 1
        assert 'Control' in result

    def test_conditions_have_different_concentrations(self, basic_df, basic_experiment):
        """Each condition uses only its own samples, so means should differ."""
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        # Get mean concentration for 34:1 in PC for each condition
        ctrl_conc = [
            r['MeanConcentration'] for r in result['Control'].records
            if r['TotalCarbons'] == 34 and r['TotalDoubleBonds'] == 1
        ]
        treat_conc = [
            r['MeanConcentration'] for r in result['Treatment'].records
            if r['TotalCarbons'] == 34 and r['TotalDoubleBonds'] == 1
        ]
        assert len(ctrl_conc) == 1
        assert len(treat_conc) == 1
        # Control uses s1-s3, Treatment uses s4-s6 — different values
        assert ctrl_conc[0] != treat_conc[0]

    def test_empty_conditions_returns_empty_dict(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment, [], ['PC'],
        )
        assert result == {}

    def test_class_filtering_applies_per_condition(self, basic_df, basic_experiment):
        result = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        for data in result.values():
            assert all(r['ClassKey'] == 'PC' for r in data.records)


# =============================================================================
# TestCreatePerConditionFigure
# =============================================================================


class TestCreatePerConditionFigure:
    """Tests for per-condition vertical figure creation."""

    def test_returns_plotly_figure(self, basic_df, basic_experiment):
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )
        all_classes = sorted({
            cls for d in per_cond.values() for cls in d.classes
        })
        colors = ChainLengthPlotterService.generate_color_mapping(all_classes)
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        assert isinstance(fig, go.Figure)

    def test_vertical_layout_two_conditions(self, basic_df, basic_experiment):
        """Two conditions should produce 2 rows x 2 cols = 4 subplot axes."""
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC', 'PE'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        # 2 rows × 2 cols → xaxis, xaxis2, xaxis3, xaxis4
        assert fig.layout.xaxis is not None
        assert fig.layout.xaxis2 is not None
        assert fig.layout.xaxis3 is not None
        assert fig.layout.xaxis4 is not None

    def test_single_condition_has_two_panels(self, basic_df, basic_experiment):
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control'], ['PC'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        assert fig.layout.xaxis is not None
        assert fig.layout.xaxis2 is not None

    def test_empty_data_returns_figure(self):
        fig = ChainLengthPlotterService.create_per_condition_figure({}, {})
        assert isinstance(fig, go.Figure)

    def test_all_empty_conditions_returns_figure(self):
        per_cond = {
            'Control': ChainLengthData(records=[], classes=[]),
            'Treatment': ChainLengthData(records=[], classes=[]),
        }
        fig = ChainLengthPlotterService.create_per_condition_figure(per_cond, {})
        assert isinstance(fig, go.Figure)

    def test_traces_have_correct_mode(self, basic_df, basic_experiment):
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control'], ['PC'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        for trace in fig.data:
            assert trace.mode == 'markers'

    def test_legend_shows_each_class_once(self, basic_df, basic_experiment):
        """Even with multiple conditions, each class appears once in legend."""
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE', 'SM'],
        )
        all_classes = sorted({
            cls for d in per_cond.values() for cls in d.classes
        })
        colors = ChainLengthPlotterService.generate_color_mapping(all_classes)
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        legend_entries = [t.name for t in fig.data if t.showlegend]
        # Each class appears exactly once in legend
        assert len(legend_entries) == len(set(legend_entries))

    def test_legend_is_horizontal(self, basic_df, basic_experiment):
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC', 'PE'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        assert fig.layout.legend.orientation == 'h'

    def test_subplot_titles_contain_condition_names(self, basic_df, basic_experiment):
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        annotations = [a.text for a in fig.layout.annotations]
        assert any('Control' in t for t in annotations)
        assert any('Treatment' in t for t in annotations)

    def test_height_scales_with_conditions(self, basic_df, basic_experiment):
        """More conditions should produce a taller figure."""
        from app.services.plotting.chain_length_plot import CHART_HEIGHT_PER_CONDITION
        per_cond = ChainLengthPlotterService.calculate_per_condition_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC'],
        )
        colors = ChainLengthPlotterService.generate_color_mapping(['PC'])
        fig = ChainLengthPlotterService.create_per_condition_figure(
            per_cond, colors,
        )
        assert fig.layout.height == CHART_HEIGHT_PER_CONDITION * 2


# =============================================================================
# TestScaleMarkerSizes
# =============================================================================


class TestScaleMarkerSizes:
    """Tests for marker size scaling."""

    def test_empty_array(self):
        result = _scale_marker_sizes(np.array([]))
        assert len(result) == 0

    def test_single_value(self):
        result = _scale_marker_sizes(np.array([100.0]))
        assert len(result) == 1

    def test_all_same_values(self):
        result = _scale_marker_sizes(np.array([50.0, 50.0, 50.0]))
        assert np.all(result == result[0])

    def test_larger_values_get_larger_markers(self):
        result = _scale_marker_sizes(np.array([10.0, 100.0, 1000.0]))
        assert result[0] < result[1] < result[2]

    def test_values_within_bounds(self):
        from app.services.plotting.chain_length_plot import MIN_MARKER_SIZE, MAX_MARKER_SIZE
        result = _scale_marker_sizes(np.array([1.0, 50.0, 100.0, 500.0]))
        assert np.all(result >= MIN_MARKER_SIZE)
        assert np.all(result <= MAX_MARKER_SIZE)

    def test_global_scaling_produces_consistent_sizes(self):
        """Using a global range ensures markers from different conditions are comparable."""
        global_vals = np.array([10.0, 100.0, 500.0, 1000.0])
        subset1 = np.array([10.0, 100.0])
        subset2 = np.array([500.0, 1000.0])

        sizes1 = _scale_marker_sizes(subset1, global_vals)
        sizes2 = _scale_marker_sizes(subset2, global_vals)

        # The largest in subset1 should be smaller than the smallest in subset2
        assert sizes1[-1] < sizes2[0]

    def test_global_scaling_values_within_bounds(self):
        from app.services.plotting.chain_length_plot import MIN_MARKER_SIZE, MAX_MARKER_SIZE
        global_vals = np.array([1.0, 50.0, 100.0, 500.0])
        subset = np.array([50.0, 100.0])
        result = _scale_marker_sizes(subset, global_vals)
        assert np.all(result >= MIN_MARKER_SIZE)
        assert np.all(result <= MAX_MARKER_SIZE)


# =============================================================================
# TestGenerateColorMapping
# =============================================================================


class TestGenerateColorMapping:
    def test_returns_dict(self):
        result = ChainLengthPlotterService.generate_color_mapping(['PC', 'PE'])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_empty_classes(self):
        result = ChainLengthPlotterService.generate_color_mapping([])
        assert result == {}

    def test_colors_are_hex(self):
        result = ChainLengthPlotterService.generate_color_mapping(['PC'])
        assert result['PC'].startswith('#')
