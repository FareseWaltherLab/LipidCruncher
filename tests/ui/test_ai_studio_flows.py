"""AI Studio driven through the full app: the main pipeline's data reaches
the Studio with the experiment it belongs to.

Each test loads the Generic sample (WT s1-s4, ADGAT-DKO s5-s8, BQC s9-s12),
runs Module 1 with normalization off, moves on to Quality Check, opens the
AI Studio and imports from the pipeline, then checks which uploaded samples
each Studio condition holds.
"""

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_AI_STUDIO, PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')

UPLOADED = {
    'WT': ['s1', 's2', 's3', 's4'],
    'ADGAT-DKO': ['s5', 's6', 's7', 's8'],
    'BQC': ['s9', 's10', 's11', 's12'],
}


def _ok(at):
    assert not at.exception, at.exception


def _to_quality_check():
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    at.radio(key='norm_method_selection').set_value('None (pre-normalized data)').run()
    [b for b in at.button if b.label.startswith('Next')][0].click().run()
    _ok(at)
    return at


def _open_studio(at):
    """What "Open AI Studio →" does, without its st.rerun(), after which
    AppTest's element tree still holds the main page's widgets."""
    at.session_state['page'] = PAGE_AI_STUDIO
    at.run()
    _ok(at)


def _import(at, choice):
    at.sidebar.radio(key='ai_studio_pipeline_choice').set_value(choice).run()
    at.sidebar.button(key='ai_studio_import_pipeline').click().run()
    at.run()  # the import's st.rerun() applies the prefilled sidebar
    _ok(at)


def _studio_groups(at):
    """{condition: uploaded samples whose values its columns hold}."""
    upload = at.session_state['standardized_df'].set_index('LipidMolec')
    df = at.session_state['chat_studio_df'].set_index('LipidMolec')
    common = df.index.intersection(upload.index)
    experiment = at.session_state['chat_studio_experiment']

    def origin(column):
        values = df.loc[common, column].to_numpy(float)
        return next(
            s for s in sum(UPLOADED.values(), [])
            if np.allclose(values, upload.loc[common, f'intensity[{s}]'].to_numpy(float))
        )

    return {
        condition: [
            origin(next(c for c in df.columns if c.endswith(f'[{s}]')))
            for s in experiment.samples_for(condition)
        ]
        for condition in experiment.conditions_list
    }


class TestImportAfterSidebarChange:
    """Regression: the sidebar can change after Module 1 ran (Quality Check
    only warns). The Normalized import paired Module 1's data, which still
    has every sample, with the changed sidebar setup; their samples
    disagreed, so the Studio put all 12 samples in one "Condition_1"."""

    def test_normalized_import_keeps_module1_setup(self):
        at = _to_quality_check()
        at.sidebar.multiselect(key='exclude_samples').set_value(['s2']).run()
        _ok(at)
        _open_studio(at)
        _import(at, 'Normalized data (from pipeline)')
        assert _studio_groups(at) == UPLOADED
        assert at.session_state['chat_studio_bqc_label'] == 'BQC'

    def test_qc_filtered_import_keeps_module1_setup(self):
        at = _to_quality_check()
        at.sidebar.multiselect(key='exclude_samples').set_value(['s2']).run()
        _ok(at)
        _open_studio(at)
        _import(at, 'QC-filtered data (from pipeline)')
        assert _studio_groups(at) == UPLOADED
        assert at.session_state['chat_studio_bqc_label'] == 'BQC'
