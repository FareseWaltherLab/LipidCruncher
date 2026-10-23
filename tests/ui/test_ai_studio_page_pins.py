"""The AI Studio page as users get it, driven through the full app.

Pins what each tab shows (open, locked, and why), what the sidebar answers
to a bad file or setup, which samples each condition holds after an apply,
and what leaving the page or loading new data clears. Only main_app.py and
session state are touched, so these hold however the page is organised.

st.rerun() ends the run instead (see _studio_with_own_upload in
test_ai_studio_flows.py); a test calls at.run() for the rerun.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_AI_STUDIO, PAGE_APP, PAGE_LANDING
from app.models.chat import ChatMode, ChatSession
from tests.ui.conftest import FakeUpload

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
UPLOAD = 'Upload your dataset'

OWN_CSV = (
    "Lipid,WT_1,WT_2,KO_1,KO_2\n"
    "PC 34:1,1.0,1.1,5.0,5.5\n"
    "PE 36:2,2.0,2.1,6.0,6.5\n"
).encode()

LIPIDSEARCH_CSV = (
    "LipidMolec,ClassKey,CalcMass,BaseRt,TotalGrade,TotalSmpIDRate(%),FAKey,"
    "MeanArea[s1],MeanArea[s2]\n"
    "PC 34:1,PC,759.6,10.1,A,100,(16:0_18:1),1.0,2.0\n"
    "PE 36:2,PE,743.5,11.2,B,100,(18:1_18:1),3.0,4.0\n"
).encode()

NOT_CONFIRMED = "Configure your experiment in the sidebar"
BLOCKED = "This dataset is a known standard format"


def _ok(at):
    assert not at.exception, at.exception


@pytest.fixture
def studio(fake_uploads, monkeypatch):
    """The AI Studio page, empty; ``studio.upload(name, data)`` hands the
    sidebar uploader a file."""
    import streamlit
    monkeypatch.setattr(streamlit, 'rerun', streamlit.stop)
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_AI_STUDIO
    at.run()
    _ok(at)

    def upload(name, data):
        fake_uploads[UPLOAD] = FakeUpload(name, data)
        at.run()
        _ok(at)
        return at

    at.upload = upload
    at.fake_uploads = fake_uploads
    return at


@pytest.fixture
def pipeline(monkeypatch):
    """The main pipeline through Module 1 (Generic sample, no
    normalization), then the AI Studio page; ``pipeline.import_(choice)``
    imports from the pipeline."""
    import streamlit
    monkeypatch.setattr(streamlit, 'rerun', streamlit.stop)
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.button(key='load_sample').click().run()
    at.run()  # the load's rerun
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    at.radio(key='norm_method_selection').set_value('None (pre-normalized data)').run()
    at.session_state['page'] = PAGE_AI_STUDIO
    at.run()
    _ok(at)

    def import_(choice):
        at.sidebar.radio(key='ai_studio_pipeline_choice').set_value(choice).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        at.run()  # the import's rerun renders the prefilled sidebar
        at.run()  # the apply step's rerun
        _ok(at)
        return at

    at.import_ = import_
    return at


def _confirm(at):
    at.sidebar.checkbox(key='studio_confirm_checkbox').check().run()
    at.run()  # the apply step's rerun
    _ok(at)
    return at


def _tab(at, label):
    """The messages and chat inputs of one tab: (texts, chat input keys)."""
    tab = next(t for t in at.tabs if t.label == label)
    texts = [e.value for e in (*tab.info, *tab.warning)]
    return texts, [c.key for c in tab.chat_input]


def _shows(at, label, fragment):
    texts, _ = _tab(at, label)
    return any(fragment in t for t in texts)


def _chat_open(at, label):
    return bool(_tab(at, label)[1])


def _groups(at):
    """{condition: values of the first row, per sample, in s1..sN order}."""
    df = at.session_state['chat_studio_df']
    experiment = at.session_state['chat_studio_experiment']
    return {
        c: [df[f'intensity[{s}]'].iloc[0] for s in experiment.samples_for(c)]
        for c in experiment.conditions_list
    }


class TestTabsWithoutData:
    def test_every_data_tab_asks_for_an_upload(self, studio):
        assert any("Upload a CSV in the sidebar to begin" in i.value for i in studio.info)
        assert _shows(studio, "Data Questions", "Upload a CSV file above to ask")
        assert _shows(studio, "Transform to Generic", "Upload a CSV file above to transform")
        assert _shows(studio, "Custom Visualization", "Upload a CSV file above to create")
        assert _chat_open(studio, "LipidCruncher Questions")


class TestOwnUpload:
    def test_before_confirm_only_transform_opens(self, studio):
        at = studio.upload('own.csv', OWN_CSV)
        assert _shows(at, "Data Questions", NOT_CONFIRMED)
        assert _chat_open(at, "Transform to Generic")
        assert _shows(at, "Custom Visualization", NOT_CONFIRMED)
        assert any(
            s.value.startswith("**own.csv** — 2 rows, 5 columns") for s in at.success
        )

    def test_confirm_applies_the_guessed_conditions(self, studio):
        at = _confirm(studio.upload('own.csv', OWN_CSV))
        assert _groups(at) == {'WT': [1.0, 1.1], 'KO': [5.0, 5.5]}
        assert at.session_state['chat_studio_sample_columns_original'] == [
            'WT_1', 'WT_2', 'KO_1', 'KO_2',
        ]
        assert at.session_state['chat_studio_sample_columns_renamed'] == [
            'intensity[s1]', 'intensity[s2]', 'intensity[s3]', 'intensity[s4]',
        ]
        assert _chat_open(at, "Data Questions")
        assert _chat_open(at, "Transform to Generic")

    def test_custom_viz_names_the_missing_module1_columns(self, studio):
        at = _confirm(studio.upload('own.csv', OWN_CSV))
        assert _shows(at, "Custom Visualization", "Missing columns: `LipidMolec`, `ClassKey`.")
        assert not _chat_open(at, "Custom Visualization")

    def test_a_new_upload_starts_over(self, studio):
        at = _confirm(studio.upload('own.csv', OWN_CSV))
        at.session_state['chat_data_questions_session'] = ChatSession(
            mode=ChatMode.DATA_QUESTIONS,
        )
        other = OWN_CSV.replace(b'KO', b'HET')
        at = studio.upload('other.csv', other)
        assert at.session_state['chat_studio_file_id'] == f'other.csv_{len(other)}'
        assert at.session_state['studio_applied_snapshot'] is None
        assert at.session_state['chat_studio_experiment'] is None
        assert at.session_state['chat_data_questions_session'] is None
        assert at.sidebar.checkbox(key='studio_confirm_checkbox').value is False
        assert _shows(at, "Data Questions", NOT_CONFIRMED)


class TestBadFiles:
    def test_an_empty_table_is_refused(self, studio):
        at = studio.upload('empty.csv', b"Lipid,WT_1\n")
        assert any("The uploaded file is empty." in e.value for e in at.sidebar.error)
        assert at.session_state['chat_studio_df'] is None

    def test_an_unreadable_file_is_refused(self, studio):
        at = studio.upload('blank.csv', b"")
        assert any(e.value.startswith("Could not read CSV") for e in at.sidebar.error)
        assert at.session_state['chat_studio_df'] is None


class TestStandardFormatUpload:
    def test_lipidsearch_is_sent_to_the_main_pipeline(self, studio):
        at = studio.upload('ls.csv', LIPIDSEARCH_CSV)
        assert any("**LipidSearch** dataset detected" in w.value for w in at.sidebar.warning)
        assert any("looks like a **LipidSearch** dataset" in w.value for w in at.warning)
        for label in ("Data Questions", "Transform to Generic", "Custom Visualization"):
            assert _shows(at, label, BLOCKED), label
            assert not _chat_open(at, label), label
        assert "Sample Columns" not in [h.value for h in at.sidebar.subheader]

    def test_go_to_main_pipeline_clears_the_studio(self, studio):
        at = studio.upload('ls.csv', LIPIDSEARCH_CSV)
        at.sidebar.button(key='ai_studio_goto_main_from_advisory').click().run()
        assert at.session_state['page'] == PAGE_APP
        assert at.session_state['chat_studio_df'] is None
        assert at.session_state['chat_studio_detected_format'] is None


class TestSidebarSetup:
    def test_duplicate_sample_names_stop_the_setup(self, studio):
        csv = (
            "Lipid,Area[WT_1],Area[WT_2],Height[WT_1],Height[WT_2]\n"
            "PC 34:1,1,2,3,4\n"
        ).encode()
        at = studio.upload('fam.csv', csv)
        at.sidebar.multiselect(key='studio_sample_cols_select').set_value(
            ['Area[WT_1]', 'Area[WT_2]', 'Height[WT_1]', 'Height[WT_2]'],
        ).run()
        assert any("Duplicate sample names" in e.value for e in at.sidebar.error)
        assert "Assign Conditions" not in [h.value for h in at.sidebar.subheader]

    def test_a_sample_without_a_condition_keeps_the_tabs_locked(self, studio):
        at = studio.upload('own.csv', OWN_CSV)
        at.session_state['studio_conditions_editor'] = {
            'edited_rows': {0: {'condition': ' '}}, 'added_rows': [], 'deleted_rows': [],
        }
        at.run()
        assert any("Every sample needs a condition." in e.value for e in at.sidebar.error)
        assert "Confirm Inputs" not in [h.value for h in at.sidebar.subheader]
        assert _shows(at, "Data Questions", NOT_CONFIRMED)

    def test_bqc_needs_a_condition_with_two_samples(self, studio):
        at = studio.upload('own.csv', OWN_CSV)
        at.session_state['studio_conditions_editor'] = {
            'edited_rows': {i: {'condition': c} for i, c in enumerate('ABCD')},
            'added_rows': [], 'deleted_rows': [],
        }
        at.run()
        at.sidebar.radio(key='studio_bqc_radio').set_value('Yes').run()
        assert any("No conditions with 2+ samples" in w.value for w in at.sidebar.warning)
        at = _confirm(at)
        assert at.session_state['chat_studio_bqc_label'] is None
        assert at.session_state['chat_studio_experiment'].conditions_list == list('ABCD')


class TestPipelineImports:
    def test_raw_import_keeps_custom_viz_locked(self, pipeline):
        at = pipeline.import_('Raw uploaded data (from pipeline)')
        assert any(
            s.value.startswith("**Raw data from main pipeline (unfiltered, un-normalized")
            for s in at.success
        )
        assert _chat_open(at, "Data Questions")
        assert _chat_open(at, "Transform to Generic")
        assert _shows(at, "Custom Visualization", "You imported **raw** data")
        assert not _chat_open(at, "Custom Visualization")

    def test_normalized_import_opens_custom_viz_not_transform(self, pipeline):
        at = pipeline.import_('Normalized data (from pipeline)')
        assert any(
            s.value.startswith("**Normalized data from main pipeline**") for s in at.success
        )
        assert _chat_open(at, "Data Questions")
        assert _shows(at, "Transform to Generic", "imported as **normalized** data")
        assert not _chat_open(at, "Transform to Generic")
        assert _chat_open(at, "Custom Visualization")

    def test_before_an_import_the_sidebar_offers_one(self, pipeline):
        assert "**Or import from the main pipeline:**" in [
            m.value for m in pipeline.sidebar.markdown
        ]
        assert pipeline.sidebar.radio(key='ai_studio_pipeline_choice').options == [
            "Normalized data (from pipeline)", "Raw uploaded data (from pipeline)",
        ]
        assert pipeline.sidebar.button(key='ai_studio_import_pipeline').label == (
            "Import Pipeline Data"
        )

    def test_back_to_main_pipeline_drops_an_import(self, pipeline):
        """(AppTest cannot run again after this click: the clicked run's
        Studio widgets stay in its tree while their keys are gone.)"""
        at = pipeline.import_('Normalized data (from pipeline)')
        assert at.sidebar.button(key='ai_studio_import_pipeline').label == (
            "Re-import Pipeline Data"
        )
        at.button(key='ai_studio_back_main').click().run()
        assert at.session_state['page'] == PAGE_APP
        assert at.session_state['chat_studio_df'] is None
        assert at.session_state['chat_studio_pipeline_source'] is None
        assert at.session_state['chat_studio_upload_kept'] is False


class TestBackToHome:
    def test_back_to_home_clears_the_studio_and_its_app_chat(self, studio):
        at = _confirm(studio.upload('own.csv', OWN_CSV))
        at.session_state['chat_lc_questions_session'] = ChatSession(
            mode=ChatMode.LIPIDCRUNCHER_QUESTIONS,
        )
        at.run()
        at.button(key='ai_studio_back_home').click().run()
        assert at.session_state['page'] == PAGE_LANDING
        for key in (
            'chat_studio_df', 'chat_studio_df_original', 'chat_studio_file_id',
            'chat_studio_experiment', 'studio_applied_snapshot',
            'chat_lc_questions_session',
        ):
            assert at.session_state[key] is None, key


class TestSubsetOfDetectedColumns:
    """Regression: answering "No" to the detected intensity[...] columns
    and picking some of them left the others in the data under their old
    names, which the picked columns' new names s1..sN could repeat; the
    page then failed with "Duplicate column names found"."""

    def test_rejecting_the_detected_columns_lets_the_user_pick(self, studio):
        csv = (
            "LipidMolec,intensity[s1],intensity[s2],intensity[s3],intensity[s4]\n"
            "PC 34:1,1,2,3,4\n"
        ).encode()
        at = studio.upload('gen.csv', csv)
        at.sidebar.radio(key='studio_sample_cols_radio').set_value('No').run()
        picker = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert picker.value == [f'intensity[s{i}]' for i in range(1, 5)]
        picker.set_value(['intensity[s2]', 'intensity[s4]']).run()
        at = _confirm(at)
        # Columns the user said are not samples do not keep sample names.
        assert list(at.session_state['chat_studio_df'].columns) == [
            'LipidMolec', 'intensity[s1]', 'intensity[s2]',
        ]
        assert _groups(at) == {'Condition_1': [2, 4]}
        assert list(at.session_state['chat_studio_df_original'].columns) == [
            'LipidMolec', 'intensity[s1]', 'intensity[s2]', 'intensity[s3]', 'intensity[s4]',
        ]


@pytest.fixture
def failing_ai(monkeypatch):
    """Every AI turn fails as if the API were down."""
    from collections import Counter

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.services.ai_chat import usage_limits
    from app.services.ai_chat.ai_client import AIClientService

    def down(*args, **kwargs):
        raise RuntimeError("API down")

    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(StreamlitAdapter, "_ai_client_id", staticmethod(lambda: None))
    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn", staticmethod(lambda **kwargs: None))
    monkeypatch.setattr(AIClientService, "stream_tool_loop", staticmethod(down))


class TestNewDataDropsOldErrors:
    """Regression: a failed turn's error and Retry stayed on the Data
    Questions and Transform tabs after new data was loaded, under the new,
    empty chat; Retry sent the old question to the new data."""

    def _fail_both_tabs(self, studio):
        at = _confirm(studio.upload('own.csv', OWN_CSV))
        for key in ('chat_input_questions', 'chat_input_transform'):
            at.chat_input(key=key).set_value('old question').run()
            at.run()  # the turn's rerun
            _ok(at)
        for label in ("Data Questions", "Transform to Generic"):
            tab = next(t for t in at.tabs if t.label == label)
            assert any("API down" in e.value for e in tab.error), label
        return at

    def test_a_new_upload_drops_them(self, studio, failing_ai):
        at = self._fail_both_tabs(studio)
        at = _confirm(studio.upload('other.csv', OWN_CSV.replace(b'KO', b'HET')))
        for label in ("Data Questions", "Transform to Generic"):
            tab = next(t for t in at.tabs if t.label == label)
            assert not any("API down" in e.value for e in tab.error), label
            assert not [b for b in tab.button if b.label == "Retry"], label

    def test_removing_the_file_drops_them(self, studio, failing_ai):
        at = self._fail_both_tabs(studio)
        studio.fake_uploads.clear()
        at.run()
        _ok(at)
        for key in ('chat_dq_last_error', 'chat_transform_last_error',
                    'chat_dq_failed_input', 'chat_transform_failed_input'):
            assert key not in at.session_state, key
