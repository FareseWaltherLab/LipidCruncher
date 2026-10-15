"""The AI Studio's usage limits hold across everything the browser can do (S5).

Regression: the per-session caps were session-state counters that every
tab's display_* zeroed whenever it started a conversation (a new upload,
Confirm, or the limit's own "Reset Chat" button), so a visitor could keep
going indefinitely; a page reload reset them too."""
from collections import Counter

import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.ai_chat import usage_limits


def dq_script():
    import pandas as pd
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.ui.ai_chat import display_data_questions_chat
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('restart_conversation'):
        # What a new upload, Confirm or "Reset Chat" did: drop the session.
        st.session_state.restart_conversation = False
        st.session_state.chat_data_questions_session = None
    df = pd.DataFrame({'LipidMolec': ['PC 34:1'], 'intensity[s1]': [1.0]})
    display_data_questions_chat(df)


@pytest.fixture
def turns(monkeypatch):
    from app.workflows.ai_chat import DataQuestionsStepResult
    from app.models.chat import ChatMessage

    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 2)
    monkeypatch.setattr(StreamlitAdapter, "_ai_client_id",
                        staticmethod(lambda: "203.0.113.7"))
    # No turn id, so no st.feedback widget (AppTest cannot rerun one).
    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn",
                        staticmethod(lambda *a, **k: None))
    calls = []

    def _stream(user_message, session, source_df):
        calls.append(user_message)
        result = DataQuestionsStepResult(
            session=session
            .add_message(ChatMessage(role="user", content=user_message))
            .add_message(ChatMessage(role="assistant", content="ok")),
            assistant_message="ok",
        )
        return iter(["ok"]), lambda: result

    monkeypatch.setattr(StreamlitAdapter, "stream_ai_data_questions",
                        staticmethod(_stream))
    return calls


def _ask(at, text):
    return at.chat_input(key="chat_input_questions").set_value(text).run()


def _limited(at):
    return (not at.chat_input
            and any("This session has used its 2 AI messages" in w.value
                    for w in at.warning))


def test_limit_survives_a_new_conversation(turns):
    at = AppTest.from_function(dq_script, default_timeout=30).run()
    _ask(at, "one")
    _ask(at, "two")
    at.run()
    assert turns == ["one", "two"]
    assert _limited(at)

    at.session_state.restart_conversation = True
    at.run()
    assert _limited(at)
    assert "Reset Chat" not in [b.label for b in at.button]

    at.button(key="dq_example_0").click().run()  # an example prompt
    assert turns == ["one", "two"]


def test_client_limit_survives_a_reload(turns, monkeypatch):
    monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 50)
    monkeypatch.setattr(usage_limits, "CLIENT_DAILY_MESSAGES", 2)
    first = AppTest.from_function(dq_script, default_timeout=30).run()
    _ask(first, "one")
    _ask(first, "two")

    reloaded = AppTest.from_function(dq_script, default_timeout=30).run()
    assert not reloaded.chat_input
    assert any("today's limit of 2 AI messages" in w.value
               for w in reloaded.warning)
    assert turns == ["one", "two"]


def viz_script():
    import pandas as pd

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    from app.ui.ai_chat import display_viz_chat
    StreamlitAdapter.initialize_session_state()

    df = pd.DataFrame({'LipidMolec': ['PC 34:1'], 'ClassKey': ['PC'],
                       'intensity[s1]': [1.0], 'intensity[s2]': [2.0]})
    display_viz_chat(df, ExperimentConfig(
        conditions_list=['A', 'B'], number_of_samples_list=[1, 1],
    ))


def test_failed_viz_code_counts_as_an_execution(monkeypatch):
    """Regression (AQ9): the Viz tab counted only successful runs, so code
    that kept failing never reached the execution limit."""
    from app.models.chat import ChatMessage
    from app.services.ai_chat.sandbox import SandboxResult
    from app.workflows.ai_chat import VizStepResult

    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(usage_limits, "SESSION_EXECUTIONS", 1)
    monkeypatch.setattr(StreamlitAdapter, "_ai_client_id",
                        staticmethod(lambda: None))
    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn",
                        staticmethod(lambda *a, **k: None))

    def _stream(user_message, session, analysis_df, experiment):
        result = VizStepResult(
            session=session
            .add_message(ChatMessage(role="user", content=user_message))
            .add_message(ChatMessage(role="assistant", content="failed",
                                     code="fig = px.bar(df, x='nope')")),
            sandbox_result=SandboxResult(success=False, error_message="boom"),
            assistant_message="failed",
        )
        return iter(["failed"]), lambda: result

    monkeypatch.setattr(StreamlitAdapter, "stream_ai_viz",
                        staticmethod(_stream))
    at = AppTest.from_function(viz_script, default_timeout=30).run()
    at.chat_input(key="chat_input_viz").set_value("plot").run()
    at.run()
    assert not at.chat_input
    assert any("run generated code 1 times" in w.value for w in at.warning)
