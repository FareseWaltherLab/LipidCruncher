"""
Tests for PathwayVizPlotterService.

Covers: chain saturation parsing (standard, ether, sphingoid, hydroxyl, edge
cases), class-level saturation ratio computation, fold-change computation,
pathway dictionary building, pathway visualization rendering (figure, scatter,
colorbar, circles, lines, labels), edge cases (empty data, single class, all
zeros, missing columns, NaN), type coercion (int, float32, int64, object
dtype, mixed), immutability, and large dataset stress tests.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.services.plotting.pathway_viz import (
    PATHWAY_CLASSES,
    PATHWAY_COORDS,
    PathwayData,
    PathwayVizPlotterService,
    SIZE_SCALE,
    UNIT_CIRCLE_RADIUS,
    _count_saturated_unsaturated,
)
from tests.conftest import make_experiment


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _make_df(lipids, classes, sample_values):
    """Build a DataFrame with LipidMolec, ClassKey, and concentration columns."""
    data = {'LipidMolec': lipids, 'ClassKey': classes}
    for i, vals in enumerate(sample_values, start=1):
        data[f'concentration[s{i}]'] = vals
    return pd.DataFrame(data)


def _make_fold_change_df(classes, values):
    """Build a fold-change DataFrame."""
    return pd.DataFrame({'ClassKey': classes, 'fold_change': values})


def _make_saturation_df(classes, values):
    """Build a saturation-ratio DataFrame."""
    return pd.DataFrame({'ClassKey': classes, 'saturation_ratio': values})


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def experiment_2x3():
    """2 conditions × 3 samples each."""
    return make_experiment(2, 3)


@pytest.fixture
def experiment_3x2():
    """3 conditions × 2 samples each."""
    return make_experiment(3, 2)


@pytest.fixture
def simple_df():
    """DataFrame with 4 lipids across 2 classes, 6 concentration columns."""
    return _make_df(
        lipids=['PC(16:0_18:1)', 'PC(16:0_16:0)', 'PE(18:1_18:2)', 'PE(18:0_20:4)'],
        classes=['PC', 'PC', 'PE', 'PE'],
        sample_values=[
            [100, 200, 150, 50],   # s1
            [110, 210, 160, 60],   # s2
            [120, 220, 170, 70],   # s3
            [200, 300, 250, 90],   # s4
            [210, 310, 260, 100],  # s5
            [220, 320, 270, 110],  # s6
        ],
    )


@pytest.fixture
def multi_class_df():
    """DataFrame with 6 lipids across 4 classes."""
    return _make_df(
        lipids=[
            'TG(16:0_18:0_18:1)', 'TG(16:0_16:0_16:0)',
            'PC(16:0_18:1)', 'PE(18:0_20:4)',
            'SM(d18:1_16:0)', 'Cer(d18:1_24:0)',
        ],
        classes=['TG', 'TG', 'PC', 'PE', 'SM', 'Cer'],
        sample_values=[
            [100, 200, 50, 30, 80, 60],   # s1
            [110, 210, 55, 35, 85, 65],   # s2
            [120, 220, 60, 40, 90, 70],   # s3
            [200, 400, 100, 60, 160, 120], # s4
            [210, 410, 105, 65, 165, 125], # s5
            [220, 420, 110, 70, 170, 130], # s6
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# TestCountSaturatedUnsaturated — private helper
# ═══════════════════════════════════════════════════════════════════════


class TestCountSaturatedUnsaturated:
    """Tests for the _count_saturated_unsaturated helper."""

    def test_two_chain_one_sat_one_unsat(self):
        assert _count_saturated_unsaturated('PC(16:0_18:1)') == (1, 1)

    def test_two_chain_both_saturated(self):
        assert _count_saturated_unsaturated('PC(16:0_16:0)') == (2, 0)

    def test_two_chain_both_unsaturated(self):
        assert _count_saturated_unsaturated('PC(18:1_18:2)') == (0, 2)

    def test_three_chain_mixed(self):
        assert _count_saturated_unsaturated('TG(16:0_18:1_18:0)') == (2, 1)

    def test_three_chain_all_saturated(self):
        assert _count_saturated_unsaturated('TG(16:0_16:0_16:0)') == (3, 0)

    def test_single_chain(self):
        assert _count_saturated_unsaturated('LPC(16:0)') == (1, 0)

    def test_single_chain_unsaturated(self):
        assert _count_saturated_unsaturated('LPC(18:1)') == (0, 1)

    def test_pufa_chain(self):
        """Chain with 4 double bonds is unsaturated."""
        assert _count_saturated_unsaturated('PE(20:4_18:0)') == (1, 1)


class TestCountSaturatedUnsaturatedHydroxyl:
    """Hydroxyl-notation lipids (e.g., ;2O suffixes)."""

    def test_hydroxyl_saturated(self):
        assert _count_saturated_unsaturated('Cer(d18:0;2O_16:0)') == (2, 0)

    def test_hydroxyl_unsaturated(self):
        assert _count_saturated_unsaturated('Cer(d18:1;2O_16:0)') == (1, 1)

    def test_hydroxyl_complex(self):
        assert _count_saturated_unsaturated('SM(d18:1;2O_24:0)') == (1, 1)


class TestCountSaturatedUnsaturatedEdgeCases:
    """Edge cases for chain parsing."""

    def test_no_parentheses(self):
        assert _count_saturated_unsaturated('InvalidLipid') == (0, 0)

    def test_empty_string(self):
        assert _count_saturated_unsaturated('') == (0, 0)

    def test_none_input(self):
        assert _count_saturated_unsaturated(None) == (0, 0)

    def test_numeric_input(self):
        assert _count_saturated_unsaturated(123) == (0, 0)

    def test_empty_parentheses(self):
        """Empty parentheses: one 'chain' with empty string, classified as unsaturated."""
        assert _count_saturated_unsaturated('PC()') == (0, 1)

    def test_malformed_chain(self):
        """No colon separator — parsing should not crash."""
        result = _count_saturated_unsaturated('PC(abc_def)')
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_consolidated_format(self):
        """Consolidated lipids like 'PC 34:1' have no parenthesised chains."""
        assert _count_saturated_unsaturated('PC 34:1') == (0, 0)


# ═══════════════════════════════════════════════════════════════════════
# TestCalculateClassSaturationRatio
# ═══════════════════════════════════════════════════════════════════════


class TestCalculateClassSaturationRatio:
    """Tests for calculate_class_saturation_ratio."""

    def test_basic_result_structure(self, simple_df):
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        assert 'ClassKey' in result.columns
        assert 'saturation_ratio' in result.columns

    def test_ratio_values_between_0_and_1(self, simple_df):
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        assert (result['saturation_ratio'] >= 0).all()
        assert (result['saturation_ratio'] <= 1).all()

    def test_pc_class_ratio(self, simple_df):
        """PC(16:0_18:1) has 1 sat + 1 unsat; PC(16:0_16:0) has 2 sat.
        Total: 3 sat + 1 unsat → ratio = 3/4 = 0.75."""
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        pc_ratio = result.loc[result['ClassKey'] == 'PC', 'saturation_ratio'].iloc[0]
        assert pc_ratio == pytest.approx(0.75)

    def test_pe_class_ratio(self, simple_df):
        """PE(18:1_18:2) has 0 sat + 2 unsat; PE(18:0_20:4) has 1 sat + 1 unsat.
        Total: 1 sat + 3 unsat → ratio = 1/4 = 0.25."""
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        pe_ratio = result.loc[result['ClassKey'] == 'PE', 'saturation_ratio'].iloc[0]
        assert pe_ratio == pytest.approx(0.25)

    def test_all_saturated_class(self):
        df = _make_df(
            lipids=['PC(16:0_16:0)', 'PC(18:0_16:0)'],
            classes=['PC', 'PC'],
            sample_values=[[100, 200], [110, 210]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert result.loc[result['ClassKey'] == 'PC', 'saturation_ratio'].iloc[0] == pytest.approx(1.0)

    def test_all_unsaturated_class(self):
        df = _make_df(
            lipids=['PC(18:1_18:2)', 'PC(20:4_22:6)'],
            classes=['PC', 'PC'],
            sample_values=[[100, 200], [110, 210]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert result.loc[result['ClassKey'] == 'PC', 'saturation_ratio'].iloc[0] == pytest.approx(0.0)

    def test_multiple_classes(self, multi_class_df):
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(multi_class_df)
        assert len(result) == 5  # TG, PC, PE, SM, Cer

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey'])
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert result.empty

    def test_none_dataframe(self):
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(None)
        assert result.empty

    def test_missing_lipidmolec_column(self):
        df = pd.DataFrame({'ClassKey': ['PC'], 'concentration[s1]': [100]})
        with pytest.raises(ValueError, match="LipidMolec"):
            PathwayVizPlotterService.calculate_class_saturation_ratio(df)

    def test_missing_classkey_column(self):
        df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)'], 'concentration[s1]': [100]})
        with pytest.raises(ValueError, match="ClassKey"):
            PathwayVizPlotterService.calculate_class_saturation_ratio(df)

    def test_unparsable_lipids_get_zero_counts(self):
        """Lipids that can't be parsed contribute (0, 0) — they shouldn't
        crash or bias the ratio."""
        df = _make_df(
            lipids=['PC 34:1', 'PC(16:0_18:1)'],
            classes=['PC', 'PC'],
            sample_values=[[100, 200], [110, 210]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        # Only the parsable lipid contributes: 1 sat + 1 unsat → 0.5
        pc_ratio = result.loc[result['ClassKey'] == 'PC', 'saturation_ratio'].iloc[0]
        assert pc_ratio == pytest.approx(0.5)

    def test_single_lipid(self):
        df = _make_df(
            lipids=['TG(16:0_18:0_18:1)'],
            classes=['TG'],
            sample_values=[[500]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        # 2 sat + 1 unsat → 2/3
        assert result.loc[result['ClassKey'] == 'TG', 'saturation_ratio'].iloc[0] == pytest.approx(2 / 3)


# ═══════════════════════════════════════════════════════════════════════
# TestCalculateClassFoldChange
# ═══════════════════════════════════════════════════════════════════════


class TestCalculateClassFoldChange:
    """Tests for calculate_class_fold_change."""

    def test_basic_result_structure(self, simple_df, experiment_2x3):
        result = PathwayVizPlotterService.calculate_class_fold_change(
            simple_df, experiment_2x3, 'Control', 'Treatment',
        )
        assert 'ClassKey' in result.columns
        assert 'fold_change' in result.columns

    def test_fold_change_values_positive(self, simple_df, experiment_2x3):
        result = PathwayVizPlotterService.calculate_class_fold_change(
            simple_df, experiment_2x3, 'Control', 'Treatment',
        )
        assert (result['fold_change'] >= 0).all()

    def test_fold_change_ratio_correct(self, experiment_2x3):
        """Control samples all 100, experimental samples all 200 → FC = 2.0."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[
                [100], [100], [100],  # s1-s3 (Control)
                [200], [200], [200],  # s4-s6 (Treatment)
            ],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == pytest.approx(2.0)

    def test_fold_change_less_than_one(self, experiment_2x3):
        """Experimental lower than control → FC < 1."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[
                [200], [200], [200],  # Control
                [100], [100], [100],  # Treatment
            ],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == pytest.approx(0.5)

    def test_equal_abundances_fc_one(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[100]] * 6,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == pytest.approx(1.0)

    def test_multiple_species_summed(self, experiment_2x3):
        """Two PC species should be summed per class before fold change."""
        df = _make_df(
            lipids=['PC(16:0_18:1)', 'PC(16:0_16:0)'],
            classes=['PC', 'PC'],
            sample_values=[
                [100, 100], [100, 100], [100, 100],  # Control
                [200, 200], [200, 200], [200, 200],  # Treatment
            ],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == pytest.approx(2.0)

    def test_multiple_classes(self, simple_df, experiment_2x3):
        result = PathwayVizPlotterService.calculate_class_fold_change(
            simple_df, experiment_2x3, 'Control', 'Treatment',
        )
        assert len(result) == 2  # PC and PE

    def test_zero_control_abundance(self, experiment_2x3):
        """Zero control mean → fold change = 0 (not inf)."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[0], [0], [0], [100], [100], [100]],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == 0.0

    def test_empty_dataframe(self, experiment_2x3):
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey'])
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.empty

    def test_none_dataframe(self, experiment_2x3):
        result = PathwayVizPlotterService.calculate_class_fold_change(
            None, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.empty

    def test_missing_classkey_column(self, experiment_2x3):
        df = pd.DataFrame({'LipidMolec': ['PC(16:0)'], 'concentration[s1]': [100]})
        with pytest.raises(ValueError, match="ClassKey"):
            PathwayVizPlotterService.calculate_class_fold_change(
                df, experiment_2x3, 'Control', 'Treatment',
            )

    def test_invalid_control_condition(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="Control condition"):
            PathwayVizPlotterService.calculate_class_fold_change(
                simple_df, experiment_2x3, 'NonExistent', 'Treatment',
            )

    def test_invalid_experimental_condition(self, simple_df, experiment_2x3):
        with pytest.raises(ValueError, match="Experimental condition"):
            PathwayVizPlotterService.calculate_class_fold_change(
                simple_df, experiment_2x3, 'Control', 'NonExistent',
            )

    def test_no_matching_concentration_columns(self, experiment_2x3):
        """DataFrame exists but has no concentration columns matching samples."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100],
        })
        with pytest.raises(ValueError, match="No valid concentration columns"):
            PathwayVizPlotterService.calculate_class_fold_change(
                df, experiment_2x3, 'Control', 'Treatment',
            )

    def test_three_conditions(self, experiment_3x2):
        """Fold change between two of three conditions."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[
                [100], [100],  # Control (s1-s2)
                [200], [200],  # Treatment (s3-s4)
                [300], [300],  # Vehicle (s5-s6)
            ],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_3x2, 'Control', 'Vehicle',
        )
        fc = result.loc[result['ClassKey'] == 'PC', 'fold_change'].iloc[0]
        assert fc == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════
# TestCreatePathwayDictionary
# ═══════════════════════════════════════════════════════════════════════


class TestCreatePathwayDictionary:
    """Tests for create_pathway_dictionary."""

    def test_has_all_three_keys(self):
        fc_df = _make_fold_change_df(['PC'], [1.5])
        sat_df = _make_saturation_df(['PC'], [0.6])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        assert 'class' in result
        assert 'abundance ratio' in result
        assert 'saturated fatty acids ratio' in result

    def test_always_18_classes(self):
        fc_df = _make_fold_change_df(['PC', 'PE'], [1.5, 2.0])
        sat_df = _make_saturation_df(['PC', 'PE'], [0.6, 0.3])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        assert len(result['class']) == 18
        assert result['class'] == list(PATHWAY_CLASSES)

    def test_missing_classes_get_zero(self):
        """Classes not in the data should get 0."""
        fc_df = _make_fold_change_df(['PC'], [2.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        tg_idx = PATHWAY_CLASSES.index('TG')
        assert result['abundance ratio'][tg_idx] == 0
        assert result['saturated fatty acids ratio'][tg_idx] == 0

    def test_present_class_values_correct(self):
        fc_df = _make_fold_change_df(['TG', 'PC'], [3.0, 1.5])
        sat_df = _make_saturation_df(['TG', 'PC'], [0.8, 0.5])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        tg_idx = PATHWAY_CLASSES.index('TG')
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert result['abundance ratio'][tg_idx] == pytest.approx(3.0)
        assert result['abundance ratio'][pc_idx] == pytest.approx(1.5)
        assert result['saturated fatty acids ratio'][tg_idx] == pytest.approx(0.8)
        assert result['saturated fatty acids ratio'][pc_idx] == pytest.approx(0.5)

    def test_both_empty_dataframes(self):
        fc_df = pd.DataFrame(columns=['ClassKey', 'fold_change'])
        sat_df = pd.DataFrame(columns=['ClassKey', 'saturation_ratio'])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        assert len(result['class']) == 18
        assert all(v == 0 for v in result['abundance ratio'])
        assert all(v == 0 for v in result['saturated fatty acids ratio'])

    def test_only_fold_change_provided(self):
        fc_df = _make_fold_change_df(['PC'], [2.0])
        sat_df = pd.DataFrame(columns=['ClassKey', 'saturation_ratio'])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert result['abundance ratio'][pc_idx] == pytest.approx(2.0)
        assert result['saturated fatty acids ratio'][pc_idx] == 0

    def test_only_saturation_provided(self):
        fc_df = pd.DataFrame(columns=['ClassKey', 'fold_change'])
        sat_df = _make_saturation_df(['PE'], [0.4])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        pe_idx = PATHWAY_CLASSES.index('PE')
        assert result['abundance ratio'][pe_idx] == 0
        assert result['saturated fatty acids ratio'][pe_idx] == pytest.approx(0.4)

    def test_all_18_classes_present(self):
        fc_df = _make_fold_change_df(PATHWAY_CLASSES, [1.0] * 18)
        sat_df = _make_saturation_df(PATHWAY_CLASSES, [0.5] * 18)
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        assert all(v == pytest.approx(1.0) for v in result['abundance ratio'])
        assert all(v == pytest.approx(0.5) for v in result['saturated fatty acids ratio'])

    def test_extra_classes_ignored(self):
        """Classes not in the pathway list are silently ignored."""
        fc_df = _make_fold_change_df(['PC', 'UnknownClass'], [2.0, 99.0])
        sat_df = _make_saturation_df(['PC', 'UnknownClass'], [0.5, 0.9])
        result = PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        assert 'UnknownClass' not in result['class']
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert result['abundance ratio'][pc_idx] == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════
# TestCreatePathwayViz — figure rendering
# ═══════════════════════════════════════════════════════════════════════


class TestCreatePathwayViz:
    """Tests for create_pathway_viz figure rendering."""

    def test_returns_figure_and_dict(self):
        fc_df = _make_fold_change_df(['PC'], [1.5])
        sat_df = _make_saturation_df(['PC'], [0.6])
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        assert isinstance(pathway_dict, dict)
        plt.close(fig)

    def test_both_empty_returns_none(self):
        fc_df = pd.DataFrame(columns=['ClassKey', 'fold_change'])
        sat_df = pd.DataFrame(columns=['ClassKey', 'saturation_ratio'])
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert fig is None
        assert pathway_dict == {}

    def test_figure_has_axes(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert len(fig.axes) >= 1  # main axes + colorbar axes
        plt.close(fig)

    def test_figure_has_title(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        assert 'Lipid Pathway' in ax.get_title()
        plt.close(fig)

    def test_figure_has_colorbar(self):
        fc_df = _make_fold_change_df(['PC', 'PE'], [1.5, 2.0])
        sat_df = _make_saturation_df(['PC', 'PE'], [0.6, 0.3])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        # Colorbar adds a second axes
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_axes_hidden(self):
        fc_df = _make_fold_change_df(['TG'], [1.0])
        sat_df = _make_saturation_df(['TG'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        assert not ax.xaxis.get_visible()
        assert not ax.yaxis.get_visible()
        plt.close(fig)

    def test_scatter_has_18_points(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        scatter_collections = [c for c in ax.collections
                               if isinstance(c, matplotlib.collections.PathCollection)]
        assert len(scatter_collections) >= 1
        offsets = scatter_collections[0].get_offsets()
        assert len(offsets) == 18
        plt.close(fig)

    def test_unit_circles_drawn(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        circles = [p for p in ax.patches if isinstance(p, plt.Circle)]
        # 6 grouping circles + 18 unit circles = 24
        assert len(circles) == 24
        plt.close(fig)

    def test_connecting_lines_drawn(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 9
        plt.close(fig)

    def test_text_labels_present(self):
        fc_df = _make_fold_change_df(['PC'], [1.0])
        sat_df = _make_saturation_df(['PC'], [0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        text_content = [t.get_text() for t in ax.texts]
        for label in ['TAG', 'DAG', 'PA', 'LPA', 'PC', 'PE', 'SM', 'Cer',
                       'CDP-DAG', 'PI', 'PG', 'PS']:
            assert label in text_content
        plt.close(fig)

    def test_pathway_dict_in_result(self):
        fc_df = _make_fold_change_df(['PC', 'TG'], [1.5, 2.0])
        sat_df = _make_saturation_df(['PC', 'TG'], [0.5, 0.8])
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert len(pathway_dict['class']) == 18
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert pathway_dict['abundance ratio'][pc_idx] == pytest.approx(1.5)
        plt.close(fig)

    def test_scatter_sizes_scale_with_fold_change(self):
        fc_df = _make_fold_change_df(['PC', 'PE'], [2.0, 4.0])
        sat_df = _make_saturation_df(['PC', 'PE'], [0.5, 0.5])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        ax = fig.axes[0]
        scatter = [c for c in ax.collections
                   if isinstance(c, matplotlib.collections.PathCollection)][0]
        sizes = scatter.get_sizes()
        pc_idx = PATHWAY_CLASSES.index('PC')
        pe_idx = PATHWAY_CLASSES.index('PE')
        assert sizes[pc_idx] == pytest.approx(SIZE_SCALE * 2.0)
        assert sizes[pe_idx] == pytest.approx(SIZE_SCALE * 4.0)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TestPathwayData dataclass
# ═══════════════════════════════════════════════════════════════════════


class TestPathwayData:
    """Tests for the PathwayData result dataclass."""

    def test_default_values(self):
        data = PathwayData()
        assert data.pathway_dict == {}
        assert data.fold_change_df.empty
        assert data.saturation_ratio_df.empty

    def test_custom_values(self):
        fc_df = _make_fold_change_df(['PC'], [1.5])
        sat_df = _make_saturation_df(['PC'], [0.6])
        pathway_dict = {'class': ['PC'], 'abundance ratio': [1.5],
                        'saturated fatty acids ratio': [0.6]}
        data = PathwayData(
            pathway_dict=pathway_dict,
            fold_change_df=fc_df,
            saturation_ratio_df=sat_df,
        )
        assert data.pathway_dict['class'] == ['PC']
        assert len(data.fold_change_df) == 1
        assert len(data.saturation_ratio_df) == 1


# ═══════════════════════════════════════════════════════════════════════
# TestConstants
# ═══════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify pathway constants are consistent."""

    def test_pathway_classes_count(self):
        assert len(PATHWAY_CLASSES) == 18

    def test_pathway_coords_count(self):
        assert len(PATHWAY_COORDS) == 18

    def test_coords_are_tuples_of_two(self):
        for coord in PATHWAY_COORDS:
            assert len(coord) == 2

    def test_classes_unique(self):
        assert len(set(PATHWAY_CLASSES)) == 18

    def test_known_classes_present(self):
        for cls in ['TG', 'DG', 'PA', 'PC', 'PE', 'SM', 'Cer', 'PI', 'PG', 'PS']:
            assert cls in PATHWAY_CLASSES


# ═══════════════════════════════════════════════════════════════════════
# TestEndToEnd — full pipeline from raw data
# ═══════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """End-to-end tests: raw DataFrame → saturation + fold change → viz."""

    def test_full_pipeline(self, simple_df, experiment_2x3):
        sat_df = PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        fc_df = PathwayVizPlotterService.calculate_class_fold_change(
            simple_df, experiment_2x3, 'Control', 'Treatment',
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        assert len(pathway_dict['class']) == 18
        pc_idx = PATHWAY_CLASSES.index('PC')
        pe_idx = PATHWAY_CLASSES.index('PE')
        assert pathway_dict['abundance ratio'][pc_idx] > 0
        assert pathway_dict['abundance ratio'][pe_idx] > 0
        plt.close(fig)

    def test_full_pipeline_multi_class(self, multi_class_df, experiment_2x3):
        sat_df = PathwayVizPlotterService.calculate_class_saturation_ratio(multi_class_df)
        fc_df = PathwayVizPlotterService.calculate_class_fold_change(
            multi_class_df, experiment_2x3, 'Control', 'Treatment',
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        # All 4 present classes should have non-zero values
        for cls in ['TG', 'PC', 'PE', 'SM']:
            idx = PATHWAY_CLASSES.index(cls)
            assert pathway_dict['abundance ratio'][idx] > 0
        plt.close(fig)

    def test_pipeline_three_conditions(self, experiment_3x2):
        df = _make_df(
            lipids=['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            classes=['PC', 'PE'],
            sample_values=[
                [100, 50], [100, 50],    # Control
                [200, 100], [200, 100],  # Treatment
                [300, 150], [300, 150],  # Vehicle
            ],
        )
        sat_df = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        fc_df = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_3x2, 'Control', 'Vehicle',
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert pathway_dict['abundance ratio'][pc_idx] == pytest.approx(3.0)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ═══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for the full service."""

    def test_all_zeros_saturation(self):
        """All-zero concentration still computes saturation from lipid names."""
        df = _make_df(
            lipids=['PC(16:0_18:1)', 'PC(16:0_16:0)'],
            classes=['PC', 'PC'],
            sample_values=[[0, 0], [0, 0]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert not result.empty

    def test_all_zeros_fold_change(self, experiment_2x3):
        """All-zero data → fold change = 0."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[0]] * 6,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == 0.0

    def test_nan_in_concentration_columns(self, experiment_2x3):
        """NaN values in concentrations should not crash."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[np.nan], [100], [100], [200], [200], [np.nan]],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert 'fold_change' in result.columns

    def test_single_sample_per_condition(self):
        """1 sample per condition should work."""
        exp = make_experiment(2, 1)
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[100], [200]],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, exp, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == pytest.approx(2.0)

    def test_very_large_fold_change(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[1], [1], [1], [1000000], [1000000], [1000000]],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fc = result.loc[0, 'fold_change']
        assert fc == pytest.approx(1000000.0)

    def test_negative_values_in_data(self, experiment_2x3):
        """Negative concentration values should not crash."""
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[-100], [-100], [-100], [200], [200], [200]],
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert 'fold_change' in result.columns

    def test_special_characters_in_lipid_names(self):
        """Lipid names with special characters in saturation calc."""
        df = _make_df(
            lipids=['PC(O-16:0_18:1)', 'PE(P-18:0_20:4)'],
            classes=['PC', 'PE'],
            sample_values=[[100, 50], [200, 100]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert not result.empty

    def test_pathway_viz_with_single_class(self):
        fc_df = _make_fold_change_df(['TG'], [2.5])
        sat_df = _make_saturation_df(['TG'], [0.7])
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        tg_idx = PATHWAY_CLASSES.index('TG')
        assert pathway_dict['abundance ratio'][tg_idx] == pytest.approx(2.5)
        # Other classes should be 0
        pc_idx = PATHWAY_CLASSES.index('PC')
        assert pathway_dict['abundance ratio'][pc_idx] == 0
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TestTypeCoercion
# ═══════════════════════════════════════════════════════════════════════


class TestTypeCoercion:
    """Verify numeric type handling across methods."""

    def test_int_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[int(100)]] * 6,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == pytest.approx(1.0)

    def test_float32_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[np.float32(100)]] * 3 + [[np.float32(200)]] * 3,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == pytest.approx(2.0)

    def test_int64_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[np.int64(100)]] * 3 + [[np.int64(200)]] * 3,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == pytest.approx(2.0)

    def test_object_dtype_concentrations(self, experiment_2x3):
        """String numbers stored as object dtype."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'concentration[s1]': pd.array([100], dtype='object'),
            'concentration[s2]': pd.array([100], dtype='object'),
            'concentration[s3]': pd.array([100], dtype='object'),
            'concentration[s4]': pd.array([200], dtype='object'),
            'concentration[s5]': pd.array([200], dtype='object'),
            'concentration[s6]': pd.array([200], dtype='object'),
        })
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert result.loc[0, 'fold_change'] == pytest.approx(2.0)

    def test_mixed_types_in_saturation(self):
        """Mixed numeric types should not crash saturation calculation."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_16:0)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [np.int64(100), np.float32(200)],
        })
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert not result.empty

    def test_full_pipeline_with_int_data(self, experiment_2x3):
        """Full pipeline with integer-only data."""
        df = _make_df(
            lipids=['PC(16:0_18:1)', 'PE(18:0_18:0)'],
            classes=['PC', 'PE'],
            sample_values=[
                [int(100), int(50)], [int(100), int(50)], [int(100), int(50)],
                [int(200), int(100)], [int(200), int(100)], [int(200), int(100)],
            ],
        )
        sat_df = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        fc_df = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TestImmutability
# ═══════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Verify input DataFrames are not modified."""

    def test_saturation_does_not_modify_input(self, simple_df):
        df_copy = simple_df.copy()
        PathwayVizPlotterService.calculate_class_saturation_ratio(simple_df)
        pd.testing.assert_frame_equal(simple_df, df_copy)

    def test_fold_change_does_not_modify_input(self, simple_df, experiment_2x3):
        df_copy = simple_df.copy()
        PathwayVizPlotterService.calculate_class_fold_change(
            simple_df, experiment_2x3, 'Control', 'Treatment',
        )
        pd.testing.assert_frame_equal(simple_df, df_copy)

    def test_viz_does_not_modify_fold_change_df(self):
        fc_df = _make_fold_change_df(['PC', 'PE'], [1.5, 2.0])
        fc_copy = fc_df.copy()
        sat_df = _make_saturation_df(['PC', 'PE'], [0.5, 0.3])
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        pd.testing.assert_frame_equal(fc_df, fc_copy)
        if fig:
            plt.close(fig)

    def test_viz_does_not_modify_saturation_df(self):
        fc_df = _make_fold_change_df(['PC'], [1.5])
        sat_df = _make_saturation_df(['PC'], [0.6])
        sat_copy = sat_df.copy()
        fig, _ = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        pd.testing.assert_frame_equal(sat_df, sat_copy)
        if fig:
            plt.close(fig)

    def test_dictionary_does_not_modify_inputs(self):
        fc_df = _make_fold_change_df(['PC'], [1.5])
        fc_copy = fc_df.copy()
        sat_df = _make_saturation_df(['PC'], [0.6])
        sat_copy = sat_df.copy()
        PathwayVizPlotterService.create_pathway_dictionary(fc_df, sat_df)
        pd.testing.assert_frame_equal(fc_df, fc_copy)
        pd.testing.assert_frame_equal(sat_df, sat_copy)


# ═══════════════════════════════════════════════════════════════════════
# TestLargeDataset
# ═══════════════════════════════════════════════════════════════════════


class TestLargeDataset:
    """Stress tests with large datasets."""

    def test_saturation_100_lipids(self):
        lipids = [f'PC({i}:0_{i+1}:1)' for i in range(100)]
        classes = ['PC'] * 100
        values = [list(np.random.rand(100) * 1000)]
        df = _make_df(lipids, classes, values)
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert len(result) == 1
        assert 0 <= result.iloc[0]['saturation_ratio'] <= 1

    def test_fold_change_many_classes(self):
        """20 different classes."""
        exp = make_experiment(2, 3)
        lipids = [f'{cls}(16:0_18:1)' for cls in PATHWAY_CLASSES[:18]]
        classes = list(PATHWAY_CLASSES[:18])
        values = [list(np.random.rand(18) * 1000) for _ in range(6)]
        df = _make_df(lipids, classes, values)
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, exp, 'Control', 'Treatment',
        )
        assert len(result) == 18

    def test_viz_with_all_18_classes(self):
        fc_df = _make_fold_change_df(
            list(PATHWAY_CLASSES),
            list(np.random.rand(18) * 5),
        )
        sat_df = _make_saturation_df(
            list(PATHWAY_CLASSES),
            list(np.random.rand(18)),
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        assert len(pathway_dict['class']) == 18
        assert all(v > 0 for v in pathway_dict['abundance ratio'])
        plt.close(fig)

    def test_large_lipid_count_full_pipeline(self):
        """500 lipids across 10 classes."""
        exp = make_experiment(2, 3)
        classes_pool = PATHWAY_CLASSES[:10]
        lipids = [f'{classes_pool[i % 10]}({16 + i % 5}:0_{18 + i % 3}:{i % 4})'
                  for i in range(500)]
        classes = [classes_pool[i % 10] for i in range(500)]
        values = [list(np.random.rand(500) * 10000) for _ in range(6)]
        df = _make_df(lipids, classes, values)

        sat_df = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        fc_df = PathwayVizPlotterService.calculate_class_fold_change(
            df, exp, 'Control', 'Treatment',
        )
        fig, pathway_dict = PathwayVizPlotterService.create_pathway_viz(fc_df, sat_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# TestNaNHandling
# ═══════════════════════════════════════════════════════════════════════


class TestNaNHandling:
    """NaN-specific edge cases."""

    def test_nan_lipid_name_in_saturation(self):
        """NaN in LipidMolec should not crash."""
        df = pd.DataFrame({
            'LipidMolec': [np.nan, 'PC(16:0_18:1)'],
            'ClassKey': ['PC', 'PC'],
            'concentration[s1]': [100, 200],
        })
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert not result.empty

    def test_nan_classkey(self):
        """NaN ClassKey should be handled (grouped as NaN)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_18:0)'],
            'ClassKey': [np.nan, 'PE'],
            'concentration[s1]': [100, 200],
        })
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert 'PE' in result['ClassKey'].values

    def test_all_nan_concentrations(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[np.nan]] * 6,
        )
        result = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert 'fold_change' in result.columns


# ═══════════════════════════════════════════════════════════════════════
# TestBoundary
# ═══════════════════════════════════════════════════════════════════════


class TestBoundary:
    """Boundary condition tests."""

    def test_single_lipid_single_sample(self):
        exp = make_experiment(2, 1)
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[100], [200]],
        )
        fc = PathwayVizPlotterService.calculate_class_fold_change(
            df, exp, 'Control', 'Treatment',
        )
        assert fc.loc[0, 'fold_change'] == pytest.approx(2.0)

    def test_fold_change_exactly_one(self):
        exp = make_experiment(2, 2)
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[500], [500], [500], [500]],
        )
        fc = PathwayVizPlotterService.calculate_class_fold_change(
            df, exp, 'Control', 'Treatment',
        )
        assert fc.loc[0, 'fold_change'] == pytest.approx(1.0)

    def test_saturation_ratio_boundary_all_sat(self):
        df = _make_df(
            lipids=['PC(16:0_16:0)'],
            classes=['PC'],
            sample_values=[[100]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert result.iloc[0]['saturation_ratio'] == pytest.approx(1.0)

    def test_saturation_ratio_boundary_all_unsat(self):
        df = _make_df(
            lipids=['PC(18:1_20:4)'],
            classes=['PC'],
            sample_values=[[100]],
        )
        result = PathwayVizPlotterService.calculate_class_saturation_ratio(df)
        assert result.iloc[0]['saturation_ratio'] == pytest.approx(0.0)

    def test_very_small_fold_change(self, experiment_2x3):
        df = _make_df(
            lipids=['PC(16:0_18:1)'],
            classes=['PC'],
            sample_values=[[1000000]] * 3 + [[1]] * 3,
        )
        fc = PathwayVizPlotterService.calculate_class_fold_change(
            df, experiment_2x3, 'Control', 'Treatment',
        )
        assert fc.loc[0, 'fold_change'] < 0.001
