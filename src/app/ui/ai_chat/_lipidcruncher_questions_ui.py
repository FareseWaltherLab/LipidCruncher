"""
LipidCruncher Questions mode UI for the AI chat.

Renders the chat conversation for asking questions about the
LipidCruncher application itself: features, methods, analysis
options, supported formats, terminology, etc.

Purely conversational — no code execution, no data access.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn
from ._safe_markdown import safe_markdown, safe_stream
from .chat_panel import (
    next_user_input,
    render_example_prompts,
    render_messages,
    start_turn_refused,
    usage_refused,
)

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "What normalization methods are available?",
    "How does LipidCruncher handle missing values?",
    "What statistical tests can I use for 3 conditions?",
    "What is the Generic format?",
    "What lipid nomenclature does LipidCruncher use?",
]


_ABOUT_TAB = """
**What this tab does** — answers questions about LipidCruncher itself: what a
feature does, which method sits behind it, why a default is what it is, what a
supported format requires.

**Where the answers come from** — LipidCruncher's own paper, supplementary
methods, release notes, and source code, which the AI reads while composing the
answer rather than recalling from memory. No code runs, and this tab never sees
your data — so it works with nothing uploaded.

**Ask it things like**

- *Why is Welch's t-test the default rather than Student's?*
- *What does the BQC CoV filter actually remove?*
- *Which normalization should I use if I have no internal standards?*
- *How does LipidCruncher decide which duplicate lipid row to keep?*

**It cannot** compute anything on your dataset — that's the **Ask Your Data**
tab.
"""


def render_lc_questions_chat() -> None:
    """Render the LipidCruncher questions chat interface."""
    session = st.session_state.chat_lc_questions_session

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(_ABOUT_TAB)

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "Ask me anything about LipidCruncher — features, methods, "
                "analysis options, and more."
            )

    render_messages(session)

    # Offer a one-click retry if the previous attempt failed.
    retry_input = st.session_state.get('chat_lc_retry_input')
    if retry_input and st.button("Retry last question", key="chat_retry_lc"):
        st.session_state.pop('chat_lc_retry_input', None)
        st.session_state.chat_pending_prompt_lc = retry_input
        st.rerun()

    render_example_prompts(_EXAMPLE_PROMPTS, "lc", "chat_pending_prompt_lc")
    if usage_refused(runs_code=False):
        return

    user_input = next_user_input(
        "chat_pending_prompt_lc",
        "Ask about LipidCruncher features, methods, or analysis options...",
        "chat_input_lc_questions",
    )
    if user_input:
        if start_turn_refused(runs_code=False):
            return

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # The example questions describe the app (not the user's data), so
        # their answers are cached and reused across sessions — instant on
        # repeat. Free-typed questions always go live (they may depend on
        # conversation context).
        # An answer is only cached when the example opened the conversation:
        # earlier turns can steer it, and the cache is served to every user.
        is_example = user_input in _EXAMPLE_PROMPTS
        cached_answer = (
            StreamlitAdapter.get_cached_lc_answer(user_input)
            if is_example else None
        )

        if cached_answer is not None:
            _answer_from_cache(session, user_input, cached_answer)
            st.rerun()

        error_text = _run_live_turn(session, user_input, is_example)
        if error_text:
            _record_lc_failure(session, user_input, error_text)

        st.rerun()


def _answer_from_cache(session, user_input: str, cached_answer: str) -> None:
    """Show and record a cached example answer; no API call."""
    from app.models.chat import ChatMessage

    with st.chat_message("assistant"):
        st.markdown(safe_markdown(cached_answer))
    st.session_state.chat_lc_questions_session = (
        session
        .add_message(ChatMessage(role="user", content=user_input))
        .add_message(ChatMessage(role="assistant", content=cached_answer))
    )
    st.session_state.pop('chat_lc_retry_input', None)
    record_turn(
        "lc_questions", user_input, cached_answer,
        assistant_msg=(
            st.session_state.chat_lc_questions_session.messages[-1]
        ),
        history=session.messages,
    )


def _run_live_turn(session, user_input: str, is_example: bool):
    """Stream the answer, record the turn; returns the error text, if any."""
    from app.models.chat import ChatMessage

    # Run AI (streamed token-by-token)
    error_text = None
    with st.chat_message("assistant"):
        try:
            # Reading the docs takes a few seconds before the first token;
            # the status names each lookup as it starts, then the answer
            # streams below it.
            status = st.status(":shimmer[Thinking]", type="compact")
            lookups = []

            def on_tool_call(name, tool_input):
                lookups.append(_lookup_label(name, tool_input))
                status.update(label=f":shimmer[{lookups[-1]}]")
                status.caption(lookups[-1])

            session_with_user, stream = (
                StreamlitAdapter.stream_ai_lc_questions(
                    user_message=user_input,
                    session=session,
                    on_tool_call=on_tool_call,
                )
            )
            full_text = st.write_stream(safe_stream(stream))
            status.update(
                label=(f"Looked up {len(lookups)} source"
                       f"{'' if len(lookups) == 1 else 's'}"
                       if lookups else "Answered"),
                state="complete",
            )

            if not (full_text or "").strip():
                full_text = (
                    "I wasn't able to generate a response. "
                    "Please try rephrasing your question."
                )
                st.markdown(safe_markdown(full_text))
            elif is_example and not session.messages:
                # Cache the canonical answer for instant reuse.
                StreamlitAdapter.set_cached_lc_answer(user_input, full_text)

            # Persist the accumulated answer to the session.
            assistant_msg = ChatMessage(role="assistant", content=full_text)
            st.session_state.chat_lc_questions_session = (
                session_with_user.add_message(assistant_msg)
            )
            st.session_state.pop('chat_lc_retry_input', None)
            record_turn(
                "lc_questions", user_input, full_text,
                assistant_msg=(
                    st.session_state.chat_lc_questions_session.messages[-1]
                ),
                history=session.messages,
            )

        except Exception as e:
            error_text = str(e)
    return error_text


def _lookup_label(name: str, tool_input: dict) -> str:
    """What a documentation or code tool call is doing, in plain words."""
    if name == "read_documentation":
        doc = str(tool_input.get("doc_name", "")).replace("_", " ")
        return f"Reading the documentation: {doc}"
    if name == "search_codebase":
        return f'Searching the code for "{tool_input.get("query", "")}"'
    if name == "read_file_section":
        return f"Reading {tool_input.get('file_path', 'a file')}"
    return "Looking through the project files"


def _record_lc_failure(session, user_input: str, error_text: str) -> None:
    """Keep the failed turn in the transcript and arm the Retry button."""
    from app.models.chat import ChatMessage

    user_msg = ChatMessage(role="user", content=user_input)
    error_msg = ChatMessage(
        role="assistant",
        content=f"Sorry, I encountered an error: {error_text}",
    )
    current = st.session_state.chat_lc_questions_session
    st.session_state.chat_lc_questions_session = (
        current.add_message(user_msg).add_message(error_msg)
    )
    # Remember the failed input so the Retry button can re-fire it.
    st.session_state.chat_lc_retry_input = user_input
    record_turn(
        "lc_questions", user_input, "", error=error_text,
        history=session.messages,
    )
