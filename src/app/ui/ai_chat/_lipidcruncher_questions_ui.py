"""
LipidCruncher Questions mode UI for the AI chat.

Renders the chat conversation for asking questions about the
LipidCruncher application itself: features, methods, analysis
options, supported formats, terminology, etc.

Purely conversational — no code execution, no data access.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn, render_feedback

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50

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

**It cannot** compute anything on your dataset — that's the **Data Questions**
tab.
"""


def _render_messages(session) -> None:
    """Render all chat messages in the session."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            render_feedback(msg)


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

    # Render existing messages
    _render_messages(session)

    # Offer a one-click retry if the previous attempt failed.
    retry_input = st.session_state.get('chat_lc_retry_input')
    if retry_input and st.button("Retry last question", key="chat_retry_lc"):
        st.session_state.pop('chat_lc_retry_input', None)
        st.session_state.chat_pending_prompt_lc = retry_input
        st.rerun()

    # Clickable example prompts (always visible)
    cols = st.columns(len(_EXAMPLE_PROMPTS))
    for i, prompt in enumerate(_EXAMPLE_PROMPTS):
        with cols[i]:
            if st.button(
                prompt,
                key=f"lc_example_{i}",
                width='stretch',
            ):
                st.session_state.chat_pending_prompt_lc = prompt
                st.rerun()

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    if msg_count >= MAX_MESSAGES:
        st.warning("Message limit reached. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_lc_questions"):
            st.session_state.chat_lc_questions_session = None
            st.session_state.chat_message_count = 0
            st.rerun()
        return

    # Chat input (check for pending example prompt first)
    user_input = st.session_state.pop('chat_pending_prompt_lc', None)
    if user_input is None:
        user_input = st.chat_input(
            "Ask about LipidCruncher features, methods, or analysis options...",
            key="chat_input_lc_questions",
        )

    if user_input:
        from app.models.chat import ChatMessage

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # The example questions describe the app (not the user's data), so
        # their answers are cached and reused across sessions — instant on
        # repeat, and a fallback if the API is unavailable. Free-typed
        # questions always go live (they may depend on conversation context).
        is_example = user_input in _EXAMPLE_PROMPTS
        cached_answer = (
            StreamlitAdapter.get_cached_lc_answer(user_input)
            if is_example else None
        )

        if cached_answer is not None:
            with st.chat_message("assistant"):
                st.markdown(cached_answer)
            st.session_state.chat_lc_questions_session = (
                session
                .add_message(ChatMessage(role="user", content=user_input))
                .add_message(ChatMessage(role="assistant", content=cached_answer))
            )
            st.session_state.chat_message_count = (
                st.session_state.get('chat_message_count', 0) + 1
            )
            st.session_state.pop('chat_lc_retry_input', None)
            record_turn(
                "lc_questions", user_input, cached_answer,
                assistant_msg=(
                    st.session_state.chat_lc_questions_session.messages[-1]
                ),
                history=session.messages,
            )
            st.rerun()

        # Run AI (streamed token-by-token)
        error_text = None
        with st.chat_message("assistant"):
            try:
                # The spinner covers the gap before the first token (e.g.
                # while the agent runs a codebase/doc tool call); once text
                # starts arriving, st.write_stream renders it live.
                with st.spinner("Thinking..."):
                    session_with_user, stream = (
                        StreamlitAdapter.stream_ai_lc_questions(
                            user_message=user_input,
                            session=session,
                        )
                    )
                    full_text = st.write_stream(stream)

                if not (full_text or "").strip():
                    full_text = (
                        "I wasn't able to generate a response. "
                        "Please try rephrasing your question."
                    )
                    st.markdown(full_text)
                elif is_example:
                    # Cache the canonical answer for instant reuse / fallback.
                    StreamlitAdapter.set_cached_lc_answer(user_input, full_text)

                # Persist the accumulated answer to the session.
                assistant_msg = ChatMessage(role="assistant", content=full_text)
                st.session_state.chat_lc_questions_session = (
                    session_with_user.add_message(assistant_msg)
                )
                st.session_state.chat_message_count = (
                    st.session_state.get('chat_message_count', 0) + 1
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

        if error_text:
            # If the API failed but we have a cached answer for this example
            # question, serve it instead of an error.
            fallback = (
                StreamlitAdapter.get_cached_lc_answer(user_input)
                if is_example else None
            )
            if fallback is not None:
                fallback_text = (
                    fallback
                    + "\n\n_(The AI service is unavailable right now — "
                    "showing a previously cached answer.)_"
                )
                with st.chat_message("assistant"):
                    st.markdown(fallback_text)
                st.session_state.chat_lc_questions_session = (
                    session
                    .add_message(ChatMessage(role="user", content=user_input))
                    .add_message(
                        ChatMessage(role="assistant", content=fallback_text)
                    )
                )
                st.session_state.chat_message_count = (
                    st.session_state.get('chat_message_count', 0) + 1
                )
                st.session_state.pop('chat_lc_retry_input', None)
                record_turn(
                    "lc_questions", user_input, fallback_text,
                    assistant_msg=(
                        st.session_state.chat_lc_questions_session.messages[-1]
                    ),
                    history=session.messages,
                )
            else:
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

        st.rerun()
