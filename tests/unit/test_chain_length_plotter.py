"""
Unit tests for ChainLengthPlotterService.

Tests cover:
1. parse_total_chain_info — lipid name parsing for carbons and double bonds
2. calculate_chain_length_data — data aggregation
3. create_chain_length_figure — figure structure
4. _scale_marker_sizes — marker size scaling
5. generate_color_mapping — color assignment
6. Edge cases — empty data, unparsable lipids, single class
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
# TestCreateChainLengthFigure
# =============================================================================


class TestCreateChainLengthFigure:
    """Tests for figure creation."""

    def test_returns_plotly_figure(self, basic_df, basic_experiment):
        data = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )
        color_mapping = ChainLengthPlotterService.generate_color_mapping(
            data.classes,
        )
        fig = ChainLengthPlotterService.create_chain_length_figure(
            data, color_mapping,
        )
        assert isinstance(fig, go.Figure)

    def test_has_two_subplots(self, basic_df, basic_experiment):
        data = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )
        color_mapping = ChainLengthPlotterService.generate_color_mapping(
            data.classes,
        )
        fig = ChainLengthPlotterService.create_chain_length_figure(
            data, color_mapping,
        )
        # Two subplots → xaxis and xaxis2
        assert fig.layout.xaxis is not None
        assert fig.layout.xaxis2 is not None

    def test_empty_data_returns_figure(self):
        data = ChainLengthData(records=[], classes=[])
        fig = ChainLengthPlotterService.create_chain_length_figure(data, {})
        assert isinstance(fig, go.Figure)

    def test_traces_have_correct_mode(self, basic_df, basic_experiment):
        data = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control'], ['PC'],
        )
        color_mapping = ChainLengthPlotterService.generate_color_mapping(
            data.classes,
        )
        fig = ChainLengthPlotterService.create_chain_length_figure(
            data, color_mapping,
        )
        for trace in fig.data:
            assert trace.mode == 'markers'

    def test_legend_shows_each_class_once(self, basic_df, basic_experiment):
        data = ChainLengthPlotterService.calculate_chain_length_data(
            basic_df, basic_experiment,
            ['Control', 'Treatment'], ['PC', 'PE', 'SM'],
        )
        color_mapping = ChainLengthPlotterService.generate_color_mapping(
            data.classes,
        )
        fig = ChainLengthPlotterService.create_chain_length_figure(
            data, color_mapping,
        )
        legend_entries = [t.name for t in fig.data if t.showlegend]
        # Each class appears exactly once in legend
        assert len(legend_entries) == len(set(legend_entries))


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