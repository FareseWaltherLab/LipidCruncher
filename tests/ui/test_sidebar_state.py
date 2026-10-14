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
