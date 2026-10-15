"""The shared LipidCruncher Questions answer cache (S7).

Example-question answers are cached process-wide and served to every user.
Regression: an answer generated after earlier turns was cached too, so one
user could steer it with those turns and have it served to everyone else
(and used as the fallback when the API is down)."""
import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import StreamlitAdapter

EXAMPLE = "What normalization methods are available?"


def lc_example_script(with_history):
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.chat import ChatMessage, ChatMode, ChatSession
    from app.ui.ai_chat._lipidcruncher_questions_ui import (
        render_lc_questions_chat,
    )
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_lc_questions_session') is None:
        session = ChatSession(mode=ChatMode.LIPIDCRUNCHER_QUESTIONS)
        if with_history:
            session = session.add_message(
                ChatMessage(role="user", content="earlier turn"),
            ).add_message(ChatMessage(role="assistant", content="ok"))
        st.session_state.chat_lc_questions_session = session
        st.session_state.chat_pending_prompt_lc = (
            "What normalization methods are available?"
        )
    render_lc_questions_chat()


@pytest.fixture
def cache_writes(monkeypatch):
    writes = []

    def _stream(user_message, session):
        return session, iter(["Answer text."])

    monkeypatch.setattr(StreamlitAdapter, "get_cached_lc_answer",
                        staticmethod(lambda question: None))
    monkeypatch.setattr(StreamlitAdapter, "set_cached_lc_answer",
                        staticmethod(lambda q, a: writes.append((q, a))))
    monkeypatch.setattr(StreamlitAdapter, "stream_ai_lc_questions",
                        staticmethod(_stream))
    return writes


def _run(with_history):
    return AppTest.from_function(
        lc_example_script, args=(with_history,), default_timeout=15,
    ).run()


def test_example_opening_the_conversation_is_cached(cache_writes):
    at = _run(with_history=False)
    assert not at.exception
    assert cache_writes == [(EXAMPLE, "Answer text.")]


def test_example_after_earlier_turns_is_not_cached(cache_writes):
    at = _run(with_history=True)
    assert not at.exception
    assert cache_writes == []
