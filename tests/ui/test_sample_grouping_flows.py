"""
Sidebar experiment setup and sample grouping, driven through the full app.

Starts from the Generic sample data confirmed (WT s1-s4, ADGAT-DKO s5-s8,
BQC s9-s12), makes the setup invalid in each way a user can, and checks the
sidebar says why and Module 1 stops until it is fixed.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
STOPPED = "Configure your experiment in the sidebar and check 'Confirm Inputs' to proceed."


def _confirmed():
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception, at.exception
    return at


def _sidebar_errors(at):
    assert not at.exception, at.exception
    return [e.value for e in at.sidebar.error]


def _module1_stopped(at):
    return [i.value for i in at.main.info] == [STOPPED]


def _reconfirm(at):
    """Check Confirm Inputs again: a run that stopped before it (an invalid
    setup) does not render it, so it comes back unchecked."""
    checkbox = at.sidebar.checkbox(key='confirm_checkbox')
    if not checkbox.value:
        checkbox.check().run()
    assert not at.exception, at.exception


class TestConditionLabels:

    def test_duplicate_label_is_named(self):
        """Regression: a repeated condition label got the generic "ensure all
        condition names are filled in and each condition has at least one
        sample", which was true; the repeat was only in the log."""
        at = _confirmed()
        at.sidebar.text_input(key='cond_name_1').input('WT').run()
        assert _sidebar_errors(at) == [
            "Condition labels must be unique: WT is used more than once."
        ]
        assert _module1_stopped(at)

    def test_blank_label(self):
        at = _confirmed()
        at.sidebar.text_input(key='cond_name_1').input('  ').run()
        assert _sidebar_errors(at) == ["All condition labels must be non-empty."]
        assert _module1_stopped(at)

    def test_fixed_label_resumes(self):
        at = _confirmed()
        at.sidebar.text_input(key='cond_name_1').input('WT').run()
        at.sidebar.text_input(key='cond_name_1').input('KO').run()
        _reconfirm(at)
        assert _sidebar_errors(at) == []
        assert at.session_state['experiment'].conditions_list == ['WT', 'KO', 'BQC']
        assert not _module1_stopped(at)


class TestSampleCounts:

    def test_counts_not_matching_the_data(self):
        at = _confirmed()
        at.sidebar.number_input(key='n_samples_0').set_value(3).run()
        assert any(
            'Assigned 11 samples but dataset has 12' in w.value
            for w in at.sidebar.warning
        )
        assert _module1_stopped(at)


class TestSampleExclusion:

    def test_fewer_than_two_left(self):
        at = _confirmed()
        at.sidebar.multiselect(key='exclude_samples').set_value(
            [f's{i}' for i in range(1, 12)]
        ).run()
        assert _sidebar_errors(at) == [
            'At least two samples must remain after exclusion.',
            'Fix the sample exclusion above before confirming.',
        ]
        assert _module1_stopped(at)

    def test_fixed_exclusion_resumes(self):
        at = _confirmed()
        at.sidebar.multiselect(key='exclude_samples').set_value(
            [f's{i}' for i in range(1, 12)]
        ).run()
        at.sidebar.multiselect(key='exclude_samples').set_value(['s1']).run()
        assert _sidebar_errors(at) == []
        assert at.session_state['experiment'].number_of_samples_list == [3, 4, 4]
        assert not _module1_stopped(at)


# Swap s1 and s5 between WT and ADGAT-DKO.
SWAP = {
    'WT': ['s5', 's2', 's3', 's4'],
    'ADGAT-DKO': ['s1', 's6', 's7', 's8'],
    'BQC': ['s9', 's10', 's11', 's12'],
}


def _break_and_fix_name(at):
    at.sidebar.text_input(key='cond_name_1').input('WT').run()
    at.sidebar.text_input(key='cond_name_1').input('ADGAT-DKO').run()


def _break_and_fix_count(at):
    at.sidebar.number_input(key='n_samples_0').set_value(3).run()
    at.sidebar.number_input(key='n_samples_0').set_value(4).run()


class TestRegroupSurvivesInvalidSetup:
    """Regression: a run that stopped at an invalid experiment setup (a
    repeated label, a count briefly off while editing) dropped the regroup
    widgets, so fixing the setup silently restored the uploaded grouping
    and the analysis ran with s1 and s5 back in their old conditions."""

    @pytest.mark.parametrize('break_and_fix', [_break_and_fix_name, _break_and_fix_count])
    def test_swap_kept(self, break_and_fix):
        at = _confirmed()
        upload = at.session_state['standardized_df'].copy()
        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        for condition, samples in SWAP.items():
            at.sidebar.multiselect(key=f'select_{condition}').set_value(samples).run()
        _reconfirm(at)
        swapped = at.session_state['regroup_origin']
        assert swapped['s1'] == 's5'

        break_and_fix(at)
        _reconfirm(at)

        assert _sidebar_errors(at) == []
        assert at.sidebar.radio(key='grouping_radio').value == 'No'
        assert at.session_state['regroup_origin'] == swapped
        # Module 1's WT s1 holds the uploaded s5's intensities.
        frame = at.session_state['processing_frame']
        assert frame.df['intensity[s1]'].tolist() == upload['intensity[s5]'].tolist()
