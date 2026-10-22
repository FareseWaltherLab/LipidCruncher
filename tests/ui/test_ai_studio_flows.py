"""AI Studio driven through the full app: the main pipeline's data reaches
the Studio with the experiment it belongs to.

Each test loads the Generic sample (WT s1-s4, ADGAT-DKO s5-s8, BQC s9-s12),
runs Module 1 with normalization off, moves on to Quality Check, opens the
AI Studio and imports from the pipeline, then checks which uploaded samples
each Studio condition holds.
"""

import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import PAGE_AI_STUDIO, PAGE_APP
from app.services.ai_chat import usage_limits
from app.services.ai_chat.ai_client import AIClientService
from app.ui.main_content.analysis._entry import ANALYSIS_OPTIONS
from tests.ai_turns import canned_stream

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


OWN_CSV = (
    "Lipid,WT_1,WT_2,KO_1,KO_2\n"
    "PC 34:1,1.0,1.1,5.0,5.5\n"
    "PE 36:2,2.0,2.1,6.0,6.5\n"
    "TG 52:2,3.0,3.1,7.0,7.5\n"
).encode()


def _studio_with_own_upload(fake_uploads, monkeypatch):
    """Main pipeline holding the sample data; the Studio holding the user's
    own CSV with its experiment confirmed. st.rerun() ends the run instead
    (tests call at.run() for the rerun), so a page switch leaves no widgets
    of the old page in AppTest's tree."""
    import streamlit

    from tests.ui.conftest import FakeUpload

    monkeypatch.setattr(streamlit, 'rerun', streamlit.stop)
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.button(key='load_sample').click().run()
    fake_uploads['Upload your dataset'] = FakeUpload('own.csv', OWN_CSV)
    at.session_state['page'] = PAGE_AI_STUDIO
    at.run()
    at.sidebar.checkbox(key='studio_confirm_checkbox').check().run()
    at.run()
    _ok(at)
    assert at.session_state['studio_applied_snapshot'] is not None
    return at


class TestOwnUploadAcrossPages:
    """Regression: the Studio's uploader comes back empty after a visit to
    the main pipeline (the page is unmounted), and an empty uploader
    cleared the Studio, so "Back to Main Pipeline" and back lost the
    user's file, experiment and chats."""

    def test_upload_survives_a_visit_to_the_main_pipeline(self, fake_uploads, monkeypatch):
        at = _studio_with_own_upload(fake_uploads, monkeypatch)
        experiment = at.session_state['chat_studio_experiment']
        at.button(key='ai_studio_back_main').click().run()
        at.run()
        _ok(at)
        assert at.session_state['page'] == PAGE_APP

        fake_uploads.clear()  # unmounted: the uploader comes back empty
        at.session_state['page'] = PAGE_AI_STUDIO
        at.run()
        _ok(at)
        assert at.session_state['chat_studio_file_id'] == f'own.csv_{len(OWN_CSV)}'
        assert at.session_state['chat_studio_experiment'] == experiment
        assert at.session_state['studio_applied_snapshot'] is not None
        assert any('uploaded before' in i.value for i in at.sidebar.info)

    def test_removing_the_file_on_the_studio_page_clears_it(self, fake_uploads, monkeypatch):
        at = _studio_with_own_upload(fake_uploads, monkeypatch)
        fake_uploads.clear()  # the user clicks the uploader's ×
        at.run()
        _ok(at)
        assert at.session_state['chat_studio_df'] is None
        assert at.session_state['studio_applied_snapshot'] is None


@pytest.fixture
def ai_replies(monkeypatch):
    """Data Questions turns run for real (workflow, sandbox, checks) on the
    model reply the test sets: ``ai_replies(text)``."""
    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(StreamlitAdapter, "_ai_client_id", staticmethod(lambda: None))
    # No turn id: no st.feedback, which AppTest cannot rerun.
    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn", staticmethod(lambda **kwargs: None))

    def reply(text):
        monkeypatch.setattr(
            AIClientService, "stream_tool_loop", staticmethod(canned_stream(text)),
        )
    return reply


def _ask(at, question):
    at.chat_input(key='chat_input_questions').set_value(question).run()
    at.run()  # the turn's st.rerun()
    _ok(at)
    return at.session_state['chat_questions_result_df']


class TestDataQuestionsOnPipelineData:
    """A Data Questions turn on imported QC data works on the samples Quality
    Check kept, and the AI's lc gives the main app's volcano numbers."""

    def test_condition_means_after_a_pca_exclusion(self, ai_replies):
        ai_replies(
            "Means per condition.\n\n```python\n"
            "result_df = pd.DataFrame({c: df[[f'concentration[{s}]' "
            "for s in lc.samples_for(c)]].mean(axis=1) for c in lc.conditions()})\n"
            "result_df.insert(0, 'LipidMolec', df['LipidMolec'])\n```"
        )
        at = _to_quality_check()
        at.multiselect(key='pca_samples_remove').set_value(['s2']).run()
        _open_studio(at)
        _import(at, 'QC-filtered data (from pipeline)')
        means = _ask(at, 'mean per condition').set_index('LipidMolec')

        upload = at.session_state['standardized_df'].set_index('LipidMolec')
        for condition, samples in {**UPLOADED, 'WT': ['s1', 's3', 's4']}.items():
            expected = upload.loc[means.index, [f'intensity[{s}]' for s in samples]].mean(axis=1)
            np.testing.assert_allclose(means[condition], expected)

        at.run()
        assert len(at.session_state['chat_data_questions_session'].messages) == 2
        assert at.session_state['chat_questions_result_df'] is not None

    def test_compare_species_matches_the_volcano(self, ai_replies, captured_downloads):
        ai_replies("Comparing.\n\n```python\nresult_df = lc.compare_species('WT', 'ADGAT-DKO')\n```")
        at = _to_quality_check()
        at.multiselect(key='pca_samples_remove').set_value(['s2']).run()
        at.radio(key='analysis_radio').set_value(ANALYSIS_OPTIONS[6]).run()
        _ok(at)
        volcano = pd.read_csv(
            io.BytesIO(captured_downloads['analysis_csv_volcano'])
        ).set_index('LipidMolec')
        _open_studio(at)
        _import(at, 'QC-filtered data (from pipeline)')
        compared = _ask(at, 'compare WT and ADGAT-DKO').set_index('LipidMolec')

        assert sorted(compared.index) == sorted(volcano.index)
        np.testing.assert_allclose(
            compared['p_value'], volcano.loc[compared.index, 'pValue'],
        )
        np.testing.assert_allclose(
            compared['log2FC'], volcano.loc[compared.index, 'FoldChange'],
        )
