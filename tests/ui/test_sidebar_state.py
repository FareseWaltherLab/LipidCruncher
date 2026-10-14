"""Full-app regression tests for sidebar state that must survive (or be
invalidated by) regrouping, exclusion and other sidebar changes.

Each test drives the real ``src/main_app.py`` with a bundled sample dataset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')


def _load_sample(format_prefix=None):
    """Open the app, optionally pick a format, load its sample data and confirm."""
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    if format_prefix is not None:
        select = at.sidebar.selectbox[0]
        option = [o for o in select.options if o.startswith(format_prefix)][0]
        select.set_value(option).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception
    return at


def _swap_first_samples(at):
    """Regroup by swapping the first sample of the first two conditions."""
    experiment = at.session_state['experiment']
    selections = [list(s) for s in experiment.individual_samples_list]
    selections[0][0], selections[1][0] = selections[1][0], selections[0][0]
    at.sidebar.radio(key='grouping_radio').set_value('No').run()
    for condition, samples in zip(experiment.conditions_list, selections):
        at.sidebar.multiselect(key=f'select_{condition}').set_value(samples).run()
    return experiment, selections


def _values(df, column):
    return pd.to_numeric(df[column], errors='coerce').fillna(0).to_numpy(float)


class TestRegroupKeepsBqcAnswer:

    def test_bqc_label_survives_manual_regroup(self):
        """Regression: while a manual regroup is mid-edit the BQC section is
        skipped, Streamlit drops its widget keys, and re-confirming silently
        set the BQC label to the first condition (WT) — so the BQC CoV filter
        ran on WT samples."""
        at = _load_sample('LipidSearch')
        assert at.session_state['bqc_label'] == 'BQC'

        _swap_first_samples(at)
        checkbox = at.sidebar.checkbox(key='confirm_checkbox')
        if not checkbox.value:
            checkbox.check().run()

        assert not at.exception
        assert at.session_state['confirmed']
        assert at.session_state['bqc_label'] == 'BQC'


class TestRegroupUndoRestoresData:

    def test_switching_back_to_yes_mid_edit_restores_upload_order(self):
        """Regression: after a confirmed regroup, starting a new edit and then
        answering "Yes" restored the names but re-committed the regrouped
        data, so s1 carried another sample's intensities under s1's name."""
        at = _load_sample('LipidSearch')
        original = at.session_state['standardized_df'].copy()
        at.session_state['sample_names'] = {f's{i}': f'N{i}' for i in range(1, 13)}
        at.run()

        experiment, selections = _swap_first_samples(at)
        checkbox = at.sidebar.checkbox(key='confirm_checkbox')
        if not checkbox.value:
            checkbox.check().run()
        # Start another edit (leaves the first condition incomplete) ...
        first = experiment.conditions_list[0]
        at.sidebar.multiselect(key=f'select_{first}').set_value(selections[0][1:]).run()
        # ... then decide the original grouping was right after all.
        at.sidebar.radio(key='grouping_radio').set_value('Yes').run()
        checkbox = at.sidebar.checkbox(key='confirm_checkbox')
        if not checkbox.value:
            checkbox.check().run()

        assert not at.exception
        restored = at.session_state['standardized_df']
        np.testing.assert_allclose(
            _values(restored, 'intensity[s1]'), _values(original, 'intensity[s1]')
        )
        assert at.session_state['sample_names']['s1'] == 'N1'


def _switch_format(at, format_prefix):
    select = at.sidebar.selectbox[0]
    option = [o for o in select.options if o.startswith(format_prefix)][0]
    select.set_value(option).run()


class TestFormatSwitchWithDataLoaded:
    """Regression for M11: changing the format dropdown with data loaded kept
    the previous format's standardized data and ran the new format's cleaner
    on it (or crashed)."""

    def test_switch_to_msdial_does_not_clean_generic_data(self):
        at = _load_sample()
        assert at.session_state['format_type'] == 'Generic Format'
        assert at.session_state['cleaned_df'] is not None

        _switch_format(at, 'MS-DIAL')

        assert not at.exception
        assert at.session_state['format_type'] != 'Generic Format'
        assert at.session_state['cleaned_df'] is None
        assert at.sidebar.error, "the Generic file should fail MS-DIAL standardization visibly"

    def test_switch_to_workbench_does_not_crash(self):
        at = _load_sample()
        _switch_format(at, 'Metabolomics Workbench')
        assert not at.exception

    def test_switching_back_restandardizes_the_upload(self):
        at = _load_sample()
        standardized = at.session_state['standardized_df'].copy()

        _switch_format(at, 'MS-DIAL')
        _switch_format(at, 'Generic')

        assert not at.exception
        assert at.session_state['format_type'] == 'Generic Format'
        pd.testing.assert_frame_equal(at.session_state['standardized_df'], standardized)


def _go_to_quality_check(at):
    [b for b in at.button if b.label.startswith('Next')][0].click().run()
    assert not at.exception
    assert at.session_state['normalized_df'] is not None


def _pca_points_per_condition(at):
    figure = at.session_state['qc_pca_plot']
    return {t.name: len(t.x) for t in figure.data if t.mode == 'markers'}


class TestSidebarChangeAfterModule1:
    """Module 2/3 analyse Module 1's frozen output, so they must keep the
    experiment that produced it while the sidebar keeps running."""

    def test_exclusion_in_quality_check_keeps_module1_grouping(self):
        """Regression: excluding s1 in the sidebar while on Quality Check
        paired the live 11-sample experiment with the frozen 12-sample
        normalized data — the excluded sample stayed in WT and later samples
        shifted into the wrong conditions, with no message."""
        at = _load_sample()
        _go_to_quality_check(at)
        assert _pca_points_per_condition(at) == {'WT': 4, 'ADGAT-DKO': 4, 'BQC': 4}

        at.sidebar.multiselect(key='exclude_samples').set_value(['s1']).run()

        assert not at.exception
        assert _pca_points_per_condition(at) == {'WT': 4, 'ADGAT-DKO': 4, 'BQC': 4}
        assert any('Back to Data Processing' in w.value for w in at.warning)

    def test_no_warning_without_sidebar_change(self):
        at = _load_sample()
        _go_to_quality_check(at)
        at.run()
        assert not any('Back to Data Processing' in w.value for w in at.warning)


def _enter_protein_amounts(at, n_samples):
    at.radio(key='norm_method_selection').set_value('Protein-based').run()
    for i in range(1, n_samples + 1):
        at.number_input(key=f'protein_s{i}').set_value(float(i * 10))
    at.run()
    assert not at.exception


def _confirm(at):
    checkbox = at.sidebar.checkbox(key='confirm_checkbox')
    if not checkbox.value:
        checkbox.check().run()
    assert not at.exception


def _concentrations(at, sample):
    df = at.session_state['normalized_df'].set_index('LipidMolec')
    return df[f'concentration[{sample}]']


class TestProteinAmountsFollowTheirSample:
    """Protein amounts are entered per sample; renumbering samples in the
    sidebar must not hand one sample's amount to another."""

    def test_amounts_survive_sidebar_exclusion(self):
        """Regression: excluding s1 renumbered old s2 to s1, which then read
        protein_s1 (old s1's amount, 10 instead of 20): its normalized values
        came out exactly 2x."""
        at = _load_sample()
        _enter_protein_amounts(at, 12)
        before = _concentrations(at, 's2')

        at.sidebar.multiselect(key='exclude_samples').set_value(['s1']).run()
        assert not at.exception

        after = _concentrations(at, 's1')  # the same sample, renumbered
        pd.testing.assert_series_equal(after, before, check_names=False)

    def test_amounts_survive_sidebar_regroup(self):
        at = _load_sample()
        _enter_protein_amounts(at, 12)
        before = _concentrations(at, 's1')

        _swap_first_samples(at)  # old s1 becomes s5
        _confirm(at)

        after = _concentrations(at, 's5')
        pd.testing.assert_series_equal(after, before, check_names=False)


class TestExclusionFollowsItsSample:

    def test_exclusion_survives_regroup(self):
        """Regression: an exclusion was kept by label, so after a regroup
        that swapped s1 and s5 it dropped the sample now labelled s1 (the old
        s5) and kept the one the user had excluded."""
        at = _load_sample()
        original = at.session_state['standardized_df'].copy()
        at.sidebar.multiselect(key='exclude_samples').set_value(['s1']).run()
        assert not at.exception

        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        selections = {
            'WT': ['s5', 's2', 's3', 's4'],
            'ADGAT-DKO': ['s1', 's6', 's7', 's8'],
            'BQC': ['s9', 's10', 's11', 's12'],
        }
        for condition, samples in selections.items():
            at.sidebar.multiselect(key=f'select_{condition}').set_value(samples).run()
        _confirm(at)

        processing = at.session_state['processing_df']
        excluded_data = _values(original, 'intensity[s1]')
        kept = [c for c in processing.columns if c.startswith('intensity[')]
        assert len(kept) == 11
        assert not any(np.allclose(_values(processing, c), excluded_data) for c in kept)
