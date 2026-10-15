"""Each AI Studio tab keeps its own last error and Retry (AQ1).

Regression: Data Questions, Transform and Custom Viz shared one error key
and popped it on display. The Retry button only existed on the rerun that
popped the error, so clicking it did nothing; and since st.tabs runs every
tab body, Data Questions (rendered first) showed every other tab's error,
with a Retry that would send that tab's prompt to the wrong chat."""
import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import StreamlitAdapter


def two_tabs_script():
    import pandas as pd
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    from app.ui.ai_chat import display_data_questions_chat, display_viz_chat
    StreamlitAdapter.initialize_session_state()

    df = pd.DataFrame({'LipidMolec': ['PC 34:1'], 'ClassKey': ['PC'],
                       'intensity[s1]': [1.0], 'intensity[s2]': [2.0]})
    experiment = ExperimentConfig(
        conditions_list=['A', 'B'], number_of_samples_list=[1, 1],
    )
    tab_dq, tab_viz = st.tabs(["DQ", "Viz"])
    with tab_dq:
        display_data_questions_chat(df)
    with tab_viz:
        display_viz_chat(df, experiment)


@pytest.fixture
def calls(monkeypatch):
    calls = []

    def _failing(name):
        def _stream(**kwargs):
            calls.append((name, kwargs["user_message"]))
            raise RuntimeError(f"{name} failed")
        return staticmethod(_stream)

    monkeypatch.setattr(StreamlitAdapter, "stream_ai_data_questions",
                        _failing("dq"))
    monkeypatch.setattr(StreamlitAdapter, "stream_ai_viz", _failing("viz"))
    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn",
                        staticmethod(lambda *a, **k: None))
    return calls


def _app():
    return AppTest.from_function(two_tabs_script, default_timeout=30).run()


def test_retry_resends_the_failed_question(calls):
    at = _app()
    at.chat_input(key="chat_input_questions").set_value("how many?").run()
    assert calls == [("dq", "how many?")]

    at.button(key="chat_retry_dq").click().run()
    assert calls == [("dq", "how many?"), ("dq", "how many?")]


def test_error_stays_until_a_turn_succeeds(calls):
    at = _app()
    at.chat_input(key="chat_input_questions").set_value("how many?").run()
    at.run()  # an unrelated rerun
    assert [e.value for e in at.tabs[0].error] == [
        "An error occurred: dq failed",
    ]


def test_viz_error_shows_in_the_viz_tab_only(calls):
    at = _app()
    at.chat_input(key="chat_input_viz").set_value("plot it").run()
    assert [e.value for e in at.tabs[0].error] == []
    assert [e.value for e in at.tabs[1].error] == [
        "An error occurred: viz failed",
    ]

    at.button(key="chat_retry_viz").click().run()
    assert calls == [("viz", "plot it"), ("viz", "plot it")]
