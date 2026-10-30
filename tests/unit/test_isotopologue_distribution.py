"""Tests for IsotopologuePlotterService: stacked corrected-fraction bars."""

import pandas as pd
import pytest

from app.services.plotting.isotopologue_distribution import IsotopologuePlotterService

SAMPLES = ['GRN_13C_3', 'GRN_13C_1', 'GRN_13C_2', 'WT_neg_2', 'WT_neg_1']


@pytest.fixture
def fractions():
    """Two molecules, three isotopologues, fractions summing to 1."""
    rows = {
        'BMP_22:6_22:6_0': [0.8, 0.7, 0.6, 1.0, 0.9],
        'BMP_22:6_22:6_1': [0.1, 0.2, 0.3, 0.0, 0.1],
        'BMP_22:6_22:6_2': [0.1, 0.1, 0.1, 0.0, 0.0],
        'PC_34:1_0': [0.5] * 5,
        'PC_34:1_1': [0.5] * 5,
    }
    return pd.DataFrame.from_dict(rows, orient='index', columns=SAMPLES)


def _plot(fractions, labels=('0', '1', '2'), molecule='BMP_22:6_22:6'):
    groups = IsotopologuePlotterService.sample_groups(list(fractions.columns))
    return IsotopologuePlotterService.create_distribution_plot(
        fractions, molecule, groups, list(labels),
    )


class TestParsing:

    @pytest.mark.parametrize('isotopologue_id,expected', [
        ('BMP_22:6_22:6_3', ('BMP_22:6_22:6', '3')),
        ('PC_34:1_C12', ('PC_34:1', 'C12')),
    ])
    def test_split_id(self, isotopologue_id, expected):
        assert IsotopologuePlotterService.split_id(isotopologue_id) == expected

    def test_molecules_and_labels(self, fractions):
        assert IsotopologuePlotterService.molecules(fractions) == ['BMP_22:6_22:6', 'PC_34:1']
        assert IsotopologuePlotterService.labels(fractions) == ['0', '1', '2']


class TestSampleGroups:

    def test_trailing_number_marks_replicates(self):
        assert IsotopologuePlotterService.sample_groups(SAMPLES) == {
            'GRN_13C': ['GRN_13C_1', 'GRN_13C_2', 'GRN_13C_3'],
            'WT_neg': ['WT_neg_1', 'WT_neg_2'],
        }

    def test_replicates_sorted_numerically(self):
        groups = IsotopologuePlotterService.sample_groups(['A_10', 'A_2', 'A_1'])
        assert groups == {'A': ['A_1', 'A_2', 'A_10']}

    def test_case_is_ignored_and_first_spelling_kept(self):
        groups = IsotopologuePlotterService.sample_groups(['WT_Neg_1', 'WT_neg_2', 'WT_NEG_3'])
        assert groups == {'WT_Neg': ['WT_Neg_1', 'WT_neg_2', 'WT_NEG_3']}

    def test_mixed_case_replicates_keep_their_number_labels(self):
        groups = IsotopologuePlotterService.sample_groups(['WT_Neg_1', 'WT_neg_2'])
        fractions = pd.DataFrame({'WT_Neg_1': [1.0], 'WT_neg_2': [1.0]}, index=['PC_34:1_0'])
        fig, _ = IsotopologuePlotterService.create_distribution_plot(
            fractions, 'PC_34:1', groups, ['0'],
        )
        assert list(fig.data[0].x[1]) == ['1', '2']

    def test_name_without_replicate_number_is_its_own_condition(self):
        groups = IsotopologuePlotterService.sample_groups(
            ['HepG2_neg', 'HMC3_PTPMT1-1_glucose', 'HepG2_0_5mM-glycerol'],
        )
        assert groups == {
            'HepG2_neg': ['HepG2_neg'],
            'HMC3_PTPMT1-1_glucose': ['HMC3_PTPMT1-1_glucose'],
            'HepG2_0_5mM-glycerol': ['HepG2_0_5mM-glycerol'],
        }


class TestDistributionPlot:

    def test_one_stacked_trace_per_isotopologue(self, fractions):
        fig, _ = _plot(fractions)
        assert fig.layout.barmode == 'stack'
        assert [t.name for t in fig.data] == ['0', '1', '2']

    def test_title_axis_and_legend(self, fractions):
        fig, _ = _plot(fractions)
        assert fig.layout.title.text == 'BMP_22:6_22:6'
        assert fig.layout.yaxis.title.text == 'Isotope Distribution'
        assert tuple(fig.layout.yaxis.range) == (0, 1)
        assert fig.layout.legend.traceorder == 'reversed'  # heaviest on top

    def test_bars_grouped_by_condition_in_replicate_order(self, fractions):
        fig, data = _plot(fractions)
        conditions, replicates = fig.data[0].x
        assert list(conditions) == ['GRN_13C'] * 3 + ['WT_neg'] * 2
        assert list(replicates) == ['1', '2', '3', '1', '2']
        assert list(fig.data[0].y) == [0.7, 0.6, 0.8, 0.9, 1.0]
        assert list(data.columns) == [
            'Isotopologue', 'GRN_13C_1', 'GRN_13C_2', 'GRN_13C_3', 'WT_neg_1', 'WT_neg_2',
        ]

    def test_condition_order_follows_groups(self, fractions):
        groups = {'WT_neg': ['WT_neg_1'], 'GRN_13C': ['GRN_13C_1', 'GRN_13C_2']}
        fig, _ = IsotopologuePlotterService.create_distribution_plot(
            fractions, 'BMP_22:6_22:6', groups, ['0'],
        )
        assert list(fig.data[0].x[0]) == ['WT_neg', 'GRN_13C', 'GRN_13C']

    def test_without_replicates_bars_are_the_samples(self, fractions):
        groups = {'GRN_13C_1': ['GRN_13C_1'], 'WT_neg_1': ['WT_neg_1']}
        fig, _ = IsotopologuePlotterService.create_distribution_plot(
            fractions, 'BMP_22:6_22:6', groups, ['0'],
        )
        assert list(fig.data[0].x) == ['GRN_13C_1', 'WT_neg_1']

    def test_chosen_isotopologues_only_and_not_rescaled(self, fractions):
        fig, data = _plot(fractions, labels=['1', '2'])
        assert [t.name for t in fig.data] == ['1', '2']
        assert list(data['Isotopologue']) == ['1', '2']
        assert data['GRN_13C_1'].sum() == pytest.approx(0.3)
        assert fig.layout.yaxis.range is None

    def test_label_keeps_its_colour_when_others_are_left_out(self, fractions):
        all_fig, _ = _plot(fractions)
        some_fig, _ = _plot(fractions, labels=['2'])
        assert some_fig.data[0].marker.color == all_fig.data[2].marker.color

    def test_heavier_isotopologues_are_darker(self, fractions):
        fig, _ = _plot(fractions)
        def brightness(color):
            r, g, b = (int(v) for v in color[4:-1].split(','))
            return r + g + b
        shades = [brightness(t.marker.color) for t in fig.data]
        assert shades == sorted(shades, reverse=True)

    def test_label_missing_for_a_molecule_is_skipped(self, fractions):
        fig, _ = _plot(fractions, molecule='PC_34:1')
        assert [t.name for t in fig.data] == ['0', '1']
        assert tuple(fig.layout.yaxis.range) == (0, 1)


class TestGroupsFromTable:

    def test_same_condition_makes_replicates_in_table_order(self):
        groups = IsotopologuePlotterService.groups_from_table(
            ['HepG2_neg', 'HMC3_neg', 'HepG2_glucose'], ['Neg', 'Neg', 'Glucose'],
        )
        assert groups == {'Neg': ['HepG2_neg', 'HMC3_neg'], 'Glucose': ['HepG2_glucose']}

    def test_typed_conditions_ignore_case(self):
        groups = IsotopologuePlotterService.groups_from_table(
            ['A_1', 'B_1', 'C_1'], ['Control', 'control', 'CONTROL'],
        )
        assert groups == {'Control': ['A_1', 'B_1', 'C_1']}

    @pytest.mark.parametrize('blank', ['', '  ', None])
    def test_blank_condition_leaves_the_sample_on_its_own(self, blank):
        groups = IsotopologuePlotterService.groups_from_table(['A_1', 'B_1'], ['A', blank])
        assert groups == {'A': ['A_1'], 'B_1': ['B_1']}

    def test_regrouped_sample_is_labelled_by_its_name(self, fractions):
        groups = {'Pooled': ['GRN_13C_1', 'WT_neg_1']}
        fig, _ = IsotopologuePlotterService.create_distribution_plot(
            fractions, 'BMP_22:6_22:6', groups, ['0'],
        )
        assert list(fig.data[0].x[1]) == ['GRN_13C_1', 'WT_neg_1']


class TestRescale:

    def test_shown_isotopologues_fill_each_bar(self, fractions):
        fig, data = _plot_rescaled(fractions, ['1', '2'])
        assert data['GRN_13C_1'].tolist() == pytest.approx([2 / 3, 1 / 3])
        assert tuple(fig.layout.yaxis.range) == (0, 1)
        assert fig.layout.yaxis.title.text == 'Isotope Distribution (of shown)'

    def test_sample_with_none_of_them_gets_an_empty_bar(self, fractions):
        _, data = _plot_rescaled(fractions, ['2'])
        assert data['WT_neg_1'].tolist() == [0.0]
        assert data['GRN_13C_1'].tolist() == [1.0]

    def test_off_by_default(self, fractions):
        fig, data = _plot(fractions, labels=['1', '2'])
        assert data['GRN_13C_1'].sum() == pytest.approx(0.3)
        assert fig.layout.yaxis.title.text == 'Isotope Distribution'


def _plot_rescaled(fractions, labels):
    groups = IsotopologuePlotterService.sample_groups(list(fractions.columns))
    return IsotopologuePlotterService.create_distribution_plot(
        fractions, 'BMP_22:6_22:6', groups, labels, rescale=True,
    )
