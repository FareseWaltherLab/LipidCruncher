"""What one turn does in each AI Studio tab: the session and the tab's result
are stored, an execution is counted, and the turn is logged with the right
flags; an empty turn says so; a failed turn is logged as an error.

Pins the tab UIs' turn handling (AQ9) so the four copies can share code."""
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.models.chat import ChatMessage
from app.services.ai_chat import usage_limits
from app.services.ai_chat.data_transformer import TransformationResult
from app.services.ai_chat.sandbox import SandboxResult
from app.workflows.ai_chat import (
    DataQuestionsStepResult,
    TransformStepResult,
    VizStepResult,
)

CODE = "result_df = df.head(1)"


def tab_script(tab):
    import pandas as pd

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    from app.ui import ai_chat
    StreamlitAdapter.initialize_session_state()

    df = pd.DataFrame({'LipidMolec': ['PC 34:1'], 'ClassKey': ['PC'],
                       'intensity[s1]': [1.0], 'intensity[s2]': [2.0]})
    if tab == "dq":
        ai_chat.display_data_questions_chat(df)
    elif tab == "transform":
        ai_chat.display_transform_chat(df)
    elif tab == "viz":
        ai_chat.display_viz_chat(df, ExperimentConfig(
            conditions_list=['A', 'B'], number_of_samples_list=[1, 1]))
    else:
        ai_chat.display_lipidcruncher_questions_chat()


INPUT_KEY = {"dq": "chat_input_questions", "transform": "chat_input_transform",
             "viz": "chat_input_viz", "lc": "chat_input_lc_questions"}
STREAM = {"dq": "stream_ai_data_questions", "transform": "stream_ai_transform",
          "viz": "stream_ai_viz", "lc": "stream_ai_lc_questions"}
SESSION_KEY = {"dq": "chat_data_questions_session",
               "transform": "chat_transform_session",
               "viz": "chat_viz_session", "lc": "chat_lc_questions_session"}


@pytest.fixture
def logged(monkeypatch):
    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(StreamlitAdapter, "_ai_client_id",
                        staticmethod(lambda: None))
    calls = []

    def _log(**kwargs):
        calls.append(kwargs)
        return None  # no turn id: no st.feedback, which AppTest can't rerun

    monkeypatch.setattr(StreamlitAdapter, "log_ai_turn", staticmethod(_log))
    monkeypatch.setattr(StreamlitAdapter, "get_cached_lc_answer",
                        staticmethod(lambda question: None))
    return calls


def _answer(session, text="answer", code=None):
    return (session
            .add_message(ChatMessage(role="user", content="q"))
            .add_message(ChatMessage(role="assistant", content=text,
                                     code=code)))


def _result(tab, session, empty=False):
    if empty:
        new = _answer(session, text="")
        return {"dq": DataQuestionsStepResult(session=new),
                "transform": TransformStepResult(session=new),
                "viz": VizStepResult(session=new)}[tab]
    new = _answer(session, code=CODE)
    if tab == "dq":
        return DataQuestionsStepResult(
            session=new, assistant_message="answer",
            result_df=pd.DataFrame({"n": [1]}), code=CODE)
    if tab == "transform":
        return TransformStepResult(
            session=new, assistant_message="answer",
            transformation=TransformationResult(
                success=True, result_df=pd.DataFrame({"LipidMolec": ["x"]}),
                code=CODE))
    return VizStepResult(
        session=new, assistant_message="answer",
        sandbox_result=SandboxResult(success=True, figure=go.Figure(
            go.Bar(x=["a"], y=[1]))))


def _stub(monkeypatch, tab, empty=False, fail=False):
    def _stream(*args, **kwargs):
        if fail:
            raise RuntimeError("api down")
        session = kwargs["session"]
        if tab == "lc":
            return (session.add_message(ChatMessage(role="user", content="q")),
                    iter(["" if empty else "answer"]))
        result = _result(tab, session, empty=empty)
        return iter(["" if empty else "answer"]), lambda: result

    monkeypatch.setattr(StreamlitAdapter, STREAM[tab], staticmethod(_stream))


def _turn(tab, text="q"):
    at = AppTest.from_function(tab_script, args=(tab,), default_timeout=30)
    at.run()
    at.chat_input(key=INPUT_KEY[tab]).set_value(text).run()
    at.run()
    return at


def _executions():
    return sum(n for (scope, _, kind), n in usage_limits._counts.items()
               if scope == "session" and kind == "executions")


@pytest.mark.parametrize("tab", ["dq", "transform", "viz"])
def test_successful_code_turn(tab, logged, monkeypatch):
    _stub(monkeypatch, tab)
    at = _turn(tab)
    assert not at.exception
    assert len(at.session_state[SESSION_KEY[tab]].messages) == 2
    assert _executions() == 1
    stored = {
        "dq": lambda: at.session_state.chat_questions_result_df,
        "transform": lambda: at.session_state.chat_transform_result,
        "viz": lambda: at.session_state.chat_viz_figures["chat_viz_fig_1"],
    }[tab]()
    assert stored is not None

    (log,) = logged
    assert log["tab"] == {"dq": "data_questions", "transform": "transform",
                          "viz": "visualization"}[tab]
    assert log["question"] == "q" and log["answer"] == "answer"
    assert log["error"] is None
    assert log["df"] is not None
    assert len(log["history"]) == 0
    assert log["produced_table"] is (tab != "viz")
    assert log["produced_figure"] is (tab == "viz")
    assert (log["figure"] is not None) is (tab == "viz")
    assert (log["experiment"] is not None) is (tab == "viz")
    assert log["code"] == CODE


@pytest.mark.parametrize("tab", ["dq", "transform", "viz"])
def test_empty_turn_says_so(tab, logged, monkeypatch):
    _stub(monkeypatch, tab, empty=True)
    at = AppTest.from_function(tab_script, args=(tab,), default_timeout=30)
    at.run()
    at.chat_input(key=INPUT_KEY[tab]).set_value("q").run()
    assert any("I couldn't produce" in i.value for i in at.info)
    assert _executions() == 0


@pytest.mark.parametrize("tab", ["dq", "transform", "viz", "lc"])
def test_failed_turn_is_logged_as_an_error(tab, logged, monkeypatch):
    _stub(monkeypatch, tab, fail=True)
    at = _turn(tab)
    assert not at.exception
    (log,) = logged
    assert log["error"] == "api down"
    assert log["answer"] == ""


def test_lc_turn(logged, monkeypatch):
    _stub(monkeypatch, "lc")
    at = _turn("lc")
    assert not at.exception
    messages = at.session_state.chat_lc_questions_session.messages
    assert [m.content for m in messages] == ["q", "answer"]
    (log,) = logged
    assert log["tab"] == "lc_questions" and log["answer"] == "answer"
    assert _executions() == 0


def raw_import_transform_script(source):
    import pandas as pd
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.ui import ai_chat
    StreamlitAdapter.initialize_session_state()

    st.session_state.chat_studio_pipeline_source = source
    df = pd.DataFrame({'LipidMolec': ['PC 34:1'], 'CalcMass': [760.5],
                       'intensity[s1]': [1.0], 'intensity[s2]': [2.0]})
    ai_chat.display_transform_chat(df)


@pytest.mark.parametrize("source", ["raw", None])
def test_transform_tab_explains_a_raw_pipeline_import(source):
    """Raw pipeline data is Generic-shaped but unfiltered: the tab says so,
    stays usable, and tells the AI not to treat it as a conversion."""
    at = AppTest.from_function(
        raw_import_transform_script, args=(source,), default_timeout=30,
    ).run()
    assert not at.exception
    diagnostic = at.session_state.chat_transform_diagnostic
    context = at.session_state.chat_transform_session.schema_context
    if source == "raw":
        assert not diagnostic.is_supported
        assert "**raw** data imported from the main pipeline" in diagnostic.message
        assert diagnostic.lipid_name_column == "LipidMolec"
        assert diagnostic.intensity_columns == ["intensity[s1]", "intensity[s2]"]
        assert diagnostic.metadata_columns == []
        assert "already in Generic shape" in context
    else:
        assert diagnostic.is_supported
        assert "already in Generic shape" not in context
