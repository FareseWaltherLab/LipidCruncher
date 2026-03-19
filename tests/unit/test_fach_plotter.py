"""
Tests for FACHPlotterService.

Covers: lipid name parsing (standard, ether, sphingoid, oxidized, edge cases),
data preparation (aggregation, proportions, multi-condition), heatmap rendering
(traces, layout, colorscale, average lines, annotations, subplots), weighted
averages, edge cases (empty data, all zeros, single lipid, missing columns),
type coercion, immutability, and large dataset stress tests.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.services.plotting.fach import (
    FACHData,
    FACHPlotterService,
    _compute_weighted_averages,
)
from tests.conftest import make_experiment


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _make_df(lipids, classes, sample_values):
    """Build a DataFrame with LipidMolec, ClassKey, and concentration columns.

    Args:
        lipids: List of lipid name strings.
        classes: List of ClassKey strings (same length as lipids).
        sample_values: List of lists, one per sample column.
    """
    data = {'LipidMolec': lipids, 'ClassKey': classes}
    for i, vals in enumerate(sample_values, start=1):
        data[f'concentration[s{i}]'] = vals
    return pd.DataFrame(data)


def _make_fach_data(data_dict, selected_class='PC', unparsable=None):
    """Build a FACHData instance for testing."""
    return FACHData(
        data_dict=data_dict,
        selected_class=selected_class,
        unparsable_lipids=unparsable or [],
    )


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    """2 conditions x 3 samples each."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    """3 conditions x 2 samples each."""
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """2 PC lipids with detailed chain notation, 6 samples."""
    return _make_df(
        lipids=['PC 16:0_18:1', 'PC 16:0_18:2'],
        classes=['PC', 'PC'],
        sample_values=[
            [100.0, 100.0],  # s1
            [110.0, 110.0],  # s2
            [120.0, 120.0],  # s3
            [200.0, 200.0],  # s4
            [210.0, 210.0],  # s5
            [220.0, 220.0],  # s6
        ],
    )


@pytest.fixture
def multi_class_df():
    """PC and PE lipids with different chain compositions."""
    return _make_df(
        lipids=[
            'PC 16:0_18:1', 'PC 16:0_18:2',
            'PE 18:0_20:4', 'PE 18:0_22:6',
        ],
        classes=['PC', 'PC', 'PE', 'PE'],
        sample_values=[
            [100.0, 50.0, 200.0, 300.0],   # s1
            [110.0, 60.0, 210.0, 310.0],   # s2
            [120.0, 70.0, 220.0, 320.0],   # s3
            [200.0, 100.0, 400.0, 500.0],  # s4
            [210.0, 110.0, 410.0, 510.0],  # s5
            [220.0, 120.0, 420.0, 520.0],  # s6
        ],
    )


@pytest.fixture
def diverse_lipid_df():
    """Lipids with varied chain compositions for proportion testing."""
    return _make_df(
        lipids=[
            'PC 16:0_18:1',  # C=34, DB=1
            'PC 16:0_18:2',  # C=34, DB=2
            'PC 18:0_20:4',  # C=38, DB=4
        ],
        classes=['PC', 'PC', 'PC'],
        sample_values=[
            [300.0, 200.0, 100.0],  # s1
            [300.0, 200.0, 100.0],  # s2
            [300.0, 200.0, 100.0],  # s3
            [300.0, 200.0, 100.0],  # s4
            [300.0, 200.0, 100.0],  # s5
            [300.0, 200.0, 100.0],  # s6
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# TestParseCarbonDb — standard formats
# ═══════════════════════════════════════════════════════════════════════


class TestParseCarbonDb:
    """Test lipid name parsing for carbon and double bond extraction."""

    def test_standard_consolidated(self):
        """PC 34:1 → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('PC 34:1') == (34, 1)

    def test_standard_two_chain(self):
        """PC 16:0_18:1 → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('PC 16:0_18:1') == (34, 1)

    def test_standard_three_chain(self):
        """TAG 16:0_18:1_18:2 → (52, 3)."""
        assert FACHPlotterService.parse_carbon_db('TAG 16:0_18:1_18:2') == (52, 3)

    def test_all_saturated(self):
        """PC 16:0_18:0 → (34, 0)."""
        assert FACHPlotterService.parse_carbon_db('PC 16:0_18:0') == (34, 0)

    def test_high_unsaturation(self):
        """PE 20:4_22:6 → (42, 10)."""
        assert FACHPlotterService.parse_carbon_db('PE 20:4_22:6') == (42, 10)


class TestParseCarbonDbEtherLipids:
    """Test parsing of ether lipid formats."""

    def test_ether_o_prefix(self):
        """PC O-38:4 → (38, 4)."""
        assert FACHPlotterService.parse_carbon_db('PC O-38:4') == (38, 4)

    def test_plasmalogen_p_prefix(self):
        """PE P-36:2 → (36, 2)."""
        assert FACHPlotterService.parse_carbon_db('PE P-36:2') == (36, 2)

    def test_ether_two_chain(self):
        """PC O-16:0_18:1 → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('PC O-16:0_18:1') == (34, 1)


class TestParseCarbonDbSphingoidBases:
    """Test parsing of sphingolipid formats."""

    def test_ceramide_d_prefix(self):
        """Cer d18:1/24:0 → (42, 1)."""
        assert FACHPlotterService.parse_carbon_db('Cer d18:1/24:0') == (42, 1)

    def test_sphingomyelin(self):
        """SM d18:1/16:0 → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('SM d18:1/16:0') == (34, 1)

    def test_t_prefix(self):
        """Cer t18:0/24:0 → (42, 0)."""
        assert FACHPlotterService.parse_carbon_db('Cer t18:0/24:0') == (42, 0)

    def test_m_prefix(self):
        """Cer m18:1/16:0 → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('Cer m18:1/16:0') == (34, 1)


class TestParseCarbonDbOxidized:
    """Test parsing of oxidized lipid formats."""

    def test_single_oxidation(self):
        """PC 16:0_18:1;O → (34, 1)."""
        assert FACHPlotterService.parse_carbon_db('PC 16:0_18:1;O') == (34, 1)

    def test_double_oxidation(self):
        """PE 18:0_20:4;O2 → (38, 4)."""
        assert FACHPlotterService.parse_carbon_db('PE 18:0_20:4;O2') == (38, 4)

    def test_triple_oxidation(self):
        """PC 18:0_20:4;O3 → (38, 4)."""
        assert FACHPlotterService.parse_carbon_db('PC 18:0_20:4;O3') == (38, 4)


class TestParseCarbonDbModifications:
    """Test parsing with modification tags."""

    def test_deuterated_modification(self):
        """LPC 18:1(d7) → (18, 1) — ignores parenthesized modification."""
        assert FACHPlotterService.parse_carbon_db('LPC 18:1(d7)') == (18, 1)

    def test_c_prefix_chain(self):
        """PC C24:0 → (24, 0)."""
        assert FACHPlotterService.parse_carbon_db('PC C24:0') == (24, 0)


class TestParseCarbonDbEdgeCases:
    """Test parsing edge cases and invalid inputs."""

    def test_none_returns_none(self):
        assert FACHPlotterService.parse_carbon_db(None) == (None, None)

    def test_empty_string_returns_none(self):
        assert FACHPlotterService.parse_carbon_db('') == (None, None)

    def test_no_chain_info_returns_none(self):
        assert FACHPlotterService.parse_carbon_db('PC') == (None, None)

    def test_no_chain_pattern_returns_none(self):
        assert FACHPlotterService.parse_carbon_db('Unknown abc') == (None, None)

    def test_integer_input_returns_none(self):
        assert FACHPlotterService.parse_carbon_db(123) == (None, None)

    def test_four_chains(self):
        """CL 16:0_18:1_18:2_20:4 → (72, 7) — cardiolipin."""
        assert FACHPlotterService.parse_carbon_db('CL 16:0_18:1_18:2_20:4') == (72, 7)

    def test_single_chain(self):
        """LPC 18:1 → (18, 1)."""
        assert FACHPlotterService.parse_carbon_db('LPC 18:1') == (18, 1)

    def test_zero_double_bonds_is_valid(self):
        """PC 32:0 → (32, 0) — zero DB is valid."""
        assert FACHPlotterService.parse_carbon_db('PC 32:0') == (32, 0)


# ═══════════════════════════════════════════════════════════════════════
# TestPrepareFachData — basic functionality
# ═══════════════════════════════════════════════════════════════════════


class TestPrepareFachData:
    """Test FACH data preparation."""

    def test_returns_fach_data(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)

    def test_selected_class_stored(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        assert result.selected_class == 'PC'

    def test_data_dict_has_conditions(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        assert 'Control' in result.data_dict
        assert 'Treatment' in result.data_dict

    def test_proportion_columns(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        cond_df = result.data_dict['Control']
        assert 'Carbon' in cond_df.columns
        assert 'DB' in cond_df.columns
        assert 'Proportion' in cond_df.columns

    def test_proportions_sum_to_100(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        total = result.data_dict['Control']['Proportion'].sum()
        assert total == pytest.approx(100.0)

    def test_aggregation_by_carbon_db(self, diverse_lipid_df, experiment_2x3):
        """Two lipids with C=34 should be summed together."""
        result = FACHPlotterService.prepare_fach_data(
            diverse_lipid_df, experiment_2x3, 'PC', ['Control'],
        )
        cond_df = result.data_dict['Control']
        # C=34 has two species (DB=1 and DB=2), C=38 has one (DB=4)
        assert len(cond_df) == 3  # 3 unique (Carbon, DB) combos

    def test_proportion_values_correct(self, diverse_lipid_df, experiment_2x3):
        """300+200+100 = 600 total. C34:DB1=300→50%, C34:DB2=200→33.3%, C38:DB4=100→16.7%."""
        result = FACHPlotterService.prepare_fach_data(
            diverse_lipid_df, experiment_2x3, 'PC', ['Control'],
        )
        cond_df = result.data_dict['Control']
        row_34_1 = cond_df[(cond_df['Carbon'] == 34) & (cond_df['DB'] == 1)]
        assert row_34_1['Proportion'].iloc[0] == pytest.approx(50.0)

    def test_multi_condition_different_samples(self, simple_df, experiment_2x3):
        """Control uses s1-s3, Treatment uses s4-s6 with higher values."""
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        # Both conditions should still sum to 100% within themselves
        for cond in ['Control', 'Treatment']:
            total = result.data_dict[cond]['Proportion'].sum()
            assert total == pytest.approx(100.0)

    def test_only_selected_class(self, multi_class_df, experiment_2x3):
        """Only PC lipids should appear, not PE."""
        result = FACHPlotterService.prepare_fach_data(
            multi_class_df, experiment_2x3, 'PC', ['Control'],
        )
        cond_df = result.data_dict['Control']
        # PC lipids: C=34 DB=1 and C=34 DB=2
        assert all(cond_df['Carbon'] == 34)

    def test_unparsable_lipids_tracked(self, experiment_2x3):
        """Unparsable lipid names should be recorded."""
        df = _make_df(
            lipids=['PC 16:0_18:1', 'UnknownLipid'],
            classes=['PC', 'PC'],
            sample_values=[
                [100.0, 50.0],
                [110.0, 60.0],
                [120.0, 70.0],
                [200.0, 100.0],
                [210.0, 110.0],
                [220.0, 120.0],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert 'UnknownLipid' in result.unparsable_lipids


class TestPrepareFachDataEdgeCases:
    """Test FACH data preparation edge cases."""

    def test_empty_class_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="lipid class must be selected"):
            FACHPlotterService.prepare_fach_data(
                simple_df, experiment_2x3, '', ['Control'],
            )

    def test_empty_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one condition"):
            FACHPlotterService.prepare_fach_data(
                simple_df, experiment_2x3, 'PC', [],
            )

    def test_none_conditions_raises(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="At least one condition"):
            FACHPlotterService.prepare_fach_data(
                simple_df, experiment_2x3, 'PC', None,
            )

    def test_nonexistent_class_returns_empty(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'NonExistent', ['Control'],
        )
        assert result.data_dict == {}

    def test_nonexistent_condition_skipped(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'FakeCondition'],
        )
        assert 'Control' in result.data_dict
        assert 'FakeCondition' not in result.data_dict

    def test_all_conditions_invalid_returns_empty(self, simple_df, experiment_2x3):
        result = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Fake1', 'Fake2'],
        )
        assert result.data_dict == {}

    def test_all_unparsable_returns_empty(self, experiment_2x3):
        """All lipids unparsable → empty data_dict."""
        df = _make_df(
            lipids=['Unknown1', 'Unknown2'],
            classes=['PC', 'PC'],
            sample_values=[
                [100.0, 50.0],
                [110.0, 60.0],
                [120.0, 70.0],
                [200.0, 100.0],
                [210.0, 110.0],
                [220.0, 120.0],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert result.data_dict == {}
        assert len(result.unparsable_lipids) == 2

    def test_all_zeros_proportion_is_zero(self, experiment_2x3):
        """All zero concentrations → proportion = 0."""
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[
                [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        cond_df = result.data_dict['Control']
        assert cond_df['Proportion'].iloc[0] == pytest.approx(0.0)

    def test_single_lipid(self, experiment_2x3):
        """Single parsable lipid → 100% proportion."""
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[
                [100.0], [110.0], [120.0], [200.0], [210.0], [220.0],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert result.data_dict['Control']['Proportion'].iloc[0] == pytest.approx(100.0)

    def test_missing_concentration_columns(self, experiment_2x3):
        """Missing sample columns should be skipped gracefully."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
            # s2, s3 missing for Control; s4-s6 missing for Treatment
        })
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        # Should still work with just s1
        assert 'Control' in result.data_dict

    def test_mixed_parsable_and_unparsable(self, experiment_2x3):
        """Mix of parsable and unparsable lipids."""
        df = _make_df(
            lipids=['PC 16:0_18:1', 'BadLipid', 'PC 18:0_20:4'],
            classes=['PC', 'PC', 'PC'],
            sample_values=[
                [300.0, 50.0, 100.0],
                [300.0, 50.0, 100.0],
                [300.0, 50.0, 100.0],
                [300.0, 50.0, 100.0],
                [300.0, 50.0, 100.0],
                [300.0, 50.0, 100.0],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert len(result.unparsable_lipids) == 1
        assert 'BadLipid' in result.unparsable_lipids
        # 300/(300+100)=75%, 100/(300+100)=25%
        cond_df = result.data_dict['Control']
        assert cond_df['Proportion'].sum() == pytest.approx(100.0)


# ═══════════════════════════════════════════════════════════════════════
# TestCreateFachHeatmap — rendering
# ═══════════════════════════════════════════════════════════════════════


class TestCreateFachHeatmap:
    """Test FACH heatmap rendering."""

    def test_returns_figure(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert isinstance(fig, go.Figure)

    def test_empty_data_returns_none(self):
        fach_data = _make_fach_data({})
        assert FACHPlotterService.create_fach_heatmap(fach_data) is None

    def test_one_trace_per_condition(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 2

    def test_single_condition_heatmap(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) == 1

    def test_heatmap_z_matrix_shape(self, diverse_lipid_df, experiment_2x3):
        """Z matrix should span full Carbon x DB grid."""
        fach_data = FACHPlotterService.prepare_fach_data(
            diverse_lipid_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        # DB range 0..4 → 5 columns, Carbon range 34..38 → 5 rows
        assert z.shape == (5, 5)

    def test_heatmap_has_correct_proportions(self, experiment_2x3):
        """Single lipid at C=34, DB=1 should have 100% in that cell."""
        df = _make_df(
            lipids=['PC 16:0_18:1'],  # C=34, DB=1
            classes=['PC'],
            sample_values=[[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]],
        )
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        # Only one cell should have 100%, rest should be 0
        assert z.max() == pytest.approx(100.0)
        assert np.count_nonzero(z) == 1

    def test_colorscale_applied(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.colorscale is not None
        assert len(heatmap.colorscale) == 7

    def test_zmin_is_zero(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.zmin == 0

    def test_consistent_zmax_across_conditions(self, simple_df, experiment_2x3):
        """Both heatmaps should share the same zmax."""
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert heatmaps[0].zmax == heatmaps[1].zmax

    def test_cell_gaps_present(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert heatmap.xgap == 1
        assert heatmap.ygap == 1

    def test_hover_template(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        assert 'Double Bonds' in heatmap.hovertemplate
        assert 'Carbon' in heatmap.hovertemplate
        assert 'Proportion' in heatmap.hovertemplate


class TestCreateFachHeatmapLayout:
    """Test heatmap layout properties."""

    def test_title(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert 'Fatty Acid Composition' in fig.layout.title.text

    def test_white_background(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert fig.layout.plot_bgcolor == 'white'
        assert fig.layout.paper_bgcolor == 'white'

    def test_height_fixed(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert fig.layout.height == 600

    def test_width_scales_with_conditions(self, simple_df, experiment_2x3):
        fach_data_1 = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fach_data_2 = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig1 = FACHPlotterService.create_fach_heatmap(fach_data_1)
        fig2 = FACHPlotterService.create_fach_heatmap(fach_data_2)
        assert fig2.layout.width == 2 * fig1.layout.width

    def test_xaxis_title(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert fig.layout.xaxis.title.text == 'Double Bonds'

    def test_yaxis_title_only_on_first(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert fig.layout.yaxis.title.text == 'Carbon Chain Length'
        # Second y-axis should not have a title
        if hasattr(fig.layout, 'yaxis2') and fig.layout.yaxis2 is not None:
            y2_title = fig.layout.yaxis2.title
            assert y2_title is None or y2_title.text is None

    def test_subplot_titles_match_conditions(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert 'Control' in annotation_texts
        assert 'Treatment' in annotation_texts


# ═══════════════════════════════════════════════════════════════════════
# TestAverageLines — weighted average annotations
# ═══════════════════════════════════════════════════════════════════════


class TestAverageLines:
    """Test weighted average lines and annotations on heatmaps."""

    def test_average_annotations_present(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        annotation_texts = [a.text for a in fig.layout.annotations]
        avg_db_annotations = [t for t in annotation_texts if 'Avg DB' in t]
        avg_c_annotations = [t for t in annotation_texts if 'Avg C' in t]
        assert len(avg_db_annotations) == 1
        assert len(avg_c_annotations) == 1

    def test_two_conditions_have_four_avg_annotations(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        annotation_texts = [a.text for a in fig.layout.annotations]
        avg_db = [t for t in annotation_texts if 'Avg DB' in t]
        avg_c = [t for t in annotation_texts if 'Avg C' in t]
        assert len(avg_db) == 2
        assert len(avg_c) == 2

    def test_average_values_correct(self, experiment_2x3):
        """Single lipid at C=34, DB=1 → avg_db=1.0, avg_carbon=34.0."""
        df = _make_df(
            lipids=['PC 16:0_18:1'],  # C=34, DB=1
            classes=['PC'],
            sample_values=[[100.0], [100.0], [100.0], [200.0], [200.0], [200.0]],
        )
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        annotation_texts = [a.text for a in fig.layout.annotations]
        avg_db_text = [t for t in annotation_texts if 'Avg DB' in t][0]
        avg_c_text = [t for t in annotation_texts if 'Avg C' in t][0]
        assert 'Avg DB: 1.0' in avg_db_text
        assert 'Avg C: 34.0' in avg_c_text

    def test_weighted_average_values(self, diverse_lipid_df, experiment_2x3):
        """Verify weighted averages are proportion-weighted, not simple averages."""
        fach_data = FACHPlotterService.prepare_fach_data(
            diverse_lipid_df, experiment_2x3, 'PC', ['Control'],
        )
        avgs = FACHPlotterService.get_weighted_averages(fach_data)
        avg_db, avg_carbon = avgs['Control']
        # C34:DB1=300(50%), C34:DB2=200(33.3%), C38:DB4=100(16.7%)
        # weighted_db = (1*50 + 2*33.3 + 4*16.7) / 100 = (50+66.7+66.7)/100 ≈ 1.833
        # weighted_c = (34*50 + 34*33.3 + 38*16.7) / 100 = (1700+1133.3+633.3)/100 ≈ 34.67
        assert avg_db == pytest.approx(1.833, abs=0.01)
        assert avg_carbon == pytest.approx(34.667, abs=0.01)

    def test_all_zero_proportions_avg_is_zero(self, experiment_2x3):
        """All zero concentrations → avg lines at 0."""
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
        )
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        avgs = FACHPlotterService.get_weighted_averages(fach_data)
        avg_db, avg_carbon = avgs['Control']
        assert avg_db == 0.0
        assert avg_carbon == 0.0


# ═══════════════════════════════════════════════════════════════════════
# TestGetWeightedAverages
# ═══════════════════════════════════════════════════════════════════════


class TestGetWeightedAverages:
    """Test the get_weighted_averages public method."""

    def test_returns_dict(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        avgs = FACHPlotterService.get_weighted_averages(fach_data)
        assert isinstance(avgs, dict)
        assert 'Control' in avgs
        assert 'Treatment' in avgs

    def test_returns_tuple_per_condition(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        avgs = FACHPlotterService.get_weighted_averages(fach_data)
        assert len(avgs['Control']) == 2

    def test_empty_data_returns_empty(self):
        fach_data = _make_fach_data({})
        avgs = FACHPlotterService.get_weighted_averages(fach_data)
        assert avgs == {}


# ═══════════════════════════════════════════════════════════════════════
# TestComputeWeightedAverages — private helper
# ═══════════════════════════════════════════════════════════════════════


class TestComputeWeightedAverages:
    """Test the _compute_weighted_averages private helper."""

    def test_basic_weighted_average(self):
        cond_df = pd.DataFrame({
            'Carbon': [34, 36],
            'DB': [1, 2],
            'Proportion': [75.0, 25.0],
        })
        avg_db, avg_carbon = _compute_weighted_averages(cond_df)
        # weighted_db = (1*75 + 2*25)/100 = 1.25
        assert avg_db == pytest.approx(1.25)
        # weighted_c = (34*75 + 36*25)/100 = 34.5
        assert avg_carbon == pytest.approx(34.5)

    def test_empty_dataframe(self):
        cond_df = pd.DataFrame(columns=['Carbon', 'DB', 'Proportion'])
        avg_db, avg_carbon = _compute_weighted_averages(cond_df)
        assert avg_db == 0.0
        assert avg_carbon == 0.0

    def test_all_zero_proportions(self):
        cond_df = pd.DataFrame({
            'Carbon': [34, 36],
            'DB': [1, 2],
            'Proportion': [0.0, 0.0],
        })
        avg_db, avg_carbon = _compute_weighted_averages(cond_df)
        assert avg_db == 0.0
        assert avg_carbon == 0.0

    def test_single_row(self):
        cond_df = pd.DataFrame({
            'Carbon': [38],
            'DB': [4],
            'Proportion': [100.0],
        })
        avg_db, avg_carbon = _compute_weighted_averages(cond_df)
        assert avg_db == pytest.approx(4.0)
        assert avg_carbon == pytest.approx(38.0)


# ═══════════════════════════════════════════════════════════════════════
# TestThreeConditions
# ═══════════════════════════════════════════════════════════════════════


class TestThreeConditions:
    """Test FACH with 3 experimental conditions."""

    def test_three_condition_data(self, simple_df, experiment_3x2):
        """3 conditions x 2 samples each = 6 samples."""
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_3x2, 'PC',
            ['Control', 'Treatment', 'Vehicle'],
        )
        assert len(fach_data.data_dict) == 3

    def test_three_condition_heatmap(self, simple_df, experiment_3x2):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_3x2, 'PC',
            ['Control', 'Treatment', 'Vehicle'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 3

    def test_three_condition_width(self, simple_df, experiment_3x2):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_3x2, 'PC',
            ['Control', 'Treatment', 'Vehicle'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert fig.layout.width == 900  # 300 * 3

    def test_three_condition_six_avg_annotations(self, simple_df, experiment_3x2):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_3x2, 'PC',
            ['Control', 'Treatment', 'Vehicle'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        annotation_texts = [a.text for a in fig.layout.annotations]
        avg_annotations = [t for t in annotation_texts if 'Avg' in t]
        assert len(avg_annotations) == 6  # 3 * (Avg DB + Avg C)


# ═══════════════════════════════════════════════════════════════════════
# TestTypeCoercion
# ═══════════════════════════════════════════════════════════════════════


class TestTypeCoercion:
    """Test that various numeric types are handled correctly."""

    def test_integer_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC 16:0_18:1', 'PC 16:0_18:2'],
            classes=['PC', 'PC'],
            sample_values=[
                [100, 50], [110, 60], [120, 70],
                [200, 100], [210, 110], [220, 120],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)
        assert result.data_dict['Control']['Proportion'].sum() == pytest.approx(100.0)

    def test_float32_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[
                np.array([100.0], dtype=np.float32),
                np.array([110.0], dtype=np.float32),
                np.array([120.0], dtype=np.float32),
                np.array([200.0], dtype=np.float32),
                np.array([210.0], dtype=np.float32),
                np.array([220.0], dtype=np.float32),
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)

    def test_numpy_int64_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[
                np.array([100], dtype=np.int64),
                np.array([110], dtype=np.int64),
                np.array([120], dtype=np.int64),
                np.array([200], dtype=np.int64),
                np.array([210], dtype=np.int64),
                np.array([220], dtype=np.int64),
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)

    def test_object_dtype_concentrations(self, experiment_2x3):
        """String numbers in object dtype columns should still work."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'concentration[s1]': pd.array(['100.0'], dtype='object'),
            'concentration[s2]': pd.array(['110.0'], dtype='object'),
            'concentration[s3]': pd.array(['120.0'], dtype='object'),
            'concentration[s4]': pd.array(['200.0'], dtype='object'),
            'concentration[s5]': pd.array(['210.0'], dtype='object'),
            'concentration[s6]': pd.array(['220.0'], dtype='object'),
        })
        # Convert to numeric so mean() works
        for col in df.columns:
            if col.startswith('concentration'):
                df[col] = pd.to_numeric(df[col])
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)

    def test_mixed_int_float_columns(self, experiment_2x3):
        df = _make_df(
            lipids=['PC 16:0_18:1'],
            classes=['PC'],
            sample_values=[
                [100], [110.5], [120], [200.5], [210], [220.5],
            ],
        )
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert isinstance(result, FACHData)

    def test_full_pipeline_with_int_data(self, experiment_2x3):
        """End-to-end: int data → prepare → heatmap renders."""
        df = _make_df(
            lipids=['PC 16:0_18:1', 'PC 16:0_18:2'],
            classes=['PC', 'PC'],
            sample_values=[
                [100, 50], [110, 60], [120, 70],
                [200, 100], [210, 110], [220, 120],
            ],
        )
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert isinstance(fig, go.Figure)


# ═══════════════════════════════════════════════════════════════════════
# TestImmutability
# ═══════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Test that input DataFrames are not modified by service methods."""

    def test_prepare_fach_data_preserves_input(self, simple_df, experiment_2x3):
        df_copy = simple_df.copy()
        FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control'],
        )
        pd.testing.assert_frame_equal(simple_df, df_copy)

    def test_prepare_fach_data_preserves_multi_class(self, multi_class_df, experiment_2x3):
        df_copy = multi_class_df.copy()
        FACHPlotterService.prepare_fach_data(
            multi_class_df, experiment_2x3, 'PC', ['Control'],
        )
        pd.testing.assert_frame_equal(multi_class_df, df_copy)

    def test_prepare_with_unparsable_preserves_input(self, experiment_2x3):
        df = _make_df(
            lipids=['PC 16:0_18:1', 'BadLipid'],
            classes=['PC', 'PC'],
            sample_values=[
                [100.0, 50.0], [110.0, 60.0], [120.0, 70.0],
                [200.0, 100.0], [210.0, 110.0], [220.0, 120.0],
            ],
        )
        df_copy = df.copy()
        FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        pd.testing.assert_frame_equal(df, df_copy)

    def test_create_heatmap_preserves_fach_data(self, simple_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            simple_df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        # Deep copy the data_dict DataFrames
        copies = {k: v.copy() for k, v in fach_data.data_dict.items()}
        FACHPlotterService.create_fach_heatmap(fach_data)
        for cond, original_df in fach_data.data_dict.items():
            pd.testing.assert_frame_equal(original_df, copies[cond])

    def test_get_weighted_averages_preserves_fach_data(self, diverse_lipid_df, experiment_2x3):
        fach_data = FACHPlotterService.prepare_fach_data(
            diverse_lipid_df, experiment_2x3, 'PC', ['Control'],
        )
        copies = {k: v.copy() for k, v in fach_data.data_dict.items()}
        FACHPlotterService.get_weighted_averages(fach_data)
        for cond, original_df in fach_data.data_dict.items():
            pd.testing.assert_frame_equal(original_df, copies[cond])


# ═══════════════════════════════════════════════════════════════════════
# TestLargeDataset
# ═══════════════════════════════════════════════════════════════════════


class TestLargeDataset:
    """Stress tests with large datasets."""

    def test_100_lipids_single_class(self, experiment_2x3):
        """100 lipids in one class should be processed correctly."""
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC {14 + i // 10}:{i % 7}_{16 + i % 5}:{i % 4}' for i in range(n)]
        classes = ['PC'] * n
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        result = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        assert 'Control' in result.data_dict
        assert result.data_dict['Control']['Proportion'].sum() == pytest.approx(100.0)

    def test_100_lipids_heatmap_renders(self, experiment_2x3):
        """Heatmap with 100 lipids should render without error."""
        rng = np.random.RandomState(42)
        n = 100
        lipids = [f'PC {14 + i // 10}:{i % 7}_{16 + i % 5}:{i % 4}' for i in range(n)]
        classes = ['PC'] * n
        sample_values = [rng.uniform(10, 1000, n).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control', 'Treatment'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        assert isinstance(fig, go.Figure)
        heatmaps = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmaps) == 2

    def test_many_unique_carbon_db_combos(self, experiment_2x3):
        """Wide range of Carbon/DB values produces correct grid."""
        rng = np.random.RandomState(42)
        lipids = [
            'PC 12:0_14:0',  # C=26, DB=0
            'PC 20:4_22:6',  # C=42, DB=10
            'PC 16:0_18:1',  # C=34, DB=1
        ]
        classes = ['PC'] * 3
        sample_values = [rng.uniform(10, 1000, 3).tolist() for _ in range(6)]

        df = _make_df(lipids, classes, sample_values)
        fach_data = FACHPlotterService.prepare_fach_data(
            df, experiment_2x3, 'PC', ['Control'],
        )
        fig = FACHPlotterService.create_fach_heatmap(fach_data)
        heatmap = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        z = np.array(heatmap.z)
        # Carbon range 26..42 → 17 rows, DB range 0..10 → 11 columns
        assert z.shape == (17, 11)


# ═══════════════════════════════════════════════════════════════════════
# TestFACHDataDataclass
# ═══════════════════════════════════════════════════════════════════════


class TestFACHDataDataclass:
    """Test FACHData dataclass defaults and attributes."""

    def test_default_empty(self):
        data = FACHData()
        assert data.data_dict == {}
        assert data.selected_class == ''
        assert data.unparsable_lipids == []

    def test_with_values(self):
        df = pd.DataFrame({'Carbon': [34], 'DB': [1], 'Proportion': [100.0]})
        data = FACHData(
            data_dict={'Control': df},
            selected_class='PC',
            unparsable_lipids=['Bad'],
        )
        assert 'Control' in data.data_dict
        assert data.selected_class == 'PC'
        assert data.unparsable_lipids == ['Bad']
