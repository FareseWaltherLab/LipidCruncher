"""
Data Questions mode UI for the AI chat.

Renders the chat conversation for asking ad-hoc questions about the
user's data: describing the dataset, computing summaries, counting
species, etc.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn, render_feedback
from ._safe_markdown import safe_markdown, safe_stream

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "Describe my dataset",
    "How many species per class?",
    "Are there missing values?",
    "Summary statistics",
]


_ABOUT_TAB = """
**What this tab does** — answers questions about the dataset loaded in the
sidebar. The AI writes pandas code, that code runs here on your data, and you
see what it returned.

**How to read an answer** — the reply is the AI's explanation; **View code**
shows the pandas that actually produced it, and **Result Table** holds the rows
it returned. The code is the method — read it before quoting a number.

**Ask it things like**

- *How many species per class, per condition?*
- *Which samples have the most missing values?*
- *Median intensity per sample*
- *List the internal standards present in this dataset*

**Getting better answers**

- Say what a column holds when its header is cryptic — the AI infers meaning
  from column names and three sample rows.
- Ask one thing at a time; follow-up questions keep the earlier context.
- If a result looks wrong, say what's wrong instead of rephrasing from scratch.

**It never modifies your dataset** — everything here is read-only. To produce a
new file, use **Transform to Generic**.
"""


def _render_messages(session) -> None:
    """Render all chat messages in the session."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(safe_markdown(msg.content))
            if msg.code:
                with st.expander("View code"):
                    st.code(msg.code, language="python")
            render_feedback(msg)


def render_data_questions_chat(source_df) -> None:
    """Render the data questions chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_data_questions_session

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(_ABOUT_TAB)

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "Ask questions about your uploaded dataset — summaries, "
                "counts, missing values, and more."
            )

    # Render existing messages
    _render_messages(session)

    # Show last result table if available (renders before the example
    # tiles so the full AI response — message, code, result — sits
    # together; tiles come after as next-step prompts).
    result_df = st.session_state.get('chat_questions_result_df')
    if result_df is not None:
        with st.expander("Result Table", expanded=True):
            st.dataframe(result_df, width='stretch')
            st.caption(
                f"{len(result_df)} rows, {len(result_df.columns)} columns"
            )

    # Clickable example prompts (always visible)
    cols = st.columns(len(_EXAMPLE_PROMPTS))
    for i, prompt in enumerate(_EXAMPLE_PROMPTS):
        with cols[i]:
            if st.button(
                prompt,
                key=f"dq_example_{i}",
                width='stretch',
            ):
                st.session_state.chat_pending_prompt_dq = prompt
                st.rerun()

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    exec_count = st.session_state.get('chat_execution_count', 0)
    if msg_count >= MAX_MESSAGES or exec_count >= MAX_EXECUTIONS:
        limit_type = "Message" if msg_count >= MAX_MESSAGES else "Execution"
        st.warning(f"{limit_type} limit reached for this session. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_questions"):
            st.session_state.chat_data_questions_session = None
            st.session_state.chat_questions_result_df = None
            st.session_state.chat_message_count = 0
            st.session_state.chat_execution_count = 0
            st.rerun()
        return

    # Chat input (check for pending example prompt first)
    user_input = st.session_state.pop('chat_pending_prompt_dq', None)
    if user_input is None:
        user_input = st.chat_input(
            "Ask a question about your data...",
            key="chat_input_questions",
        )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI data questions (explanation streamed; code/table render
        # canonically after the rerun below)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    gen, finalize = StreamlitAdapter.stream_ai_data_questions(
                        user_message=user_input,
                        session=session,
                        source_df=source_df,
                    )
                    streamed = st.write_stream(safe_stream(gen))
                result = finalize()

                # If the turn produced nothing at all (no text, no code, no
                # table), say so rather than leaving a blank assistant bubble.
                if (not (streamed or "").strip()
                        and result.result_df is None
                        and not result.code):
                    st.info(
                        "I couldn't produce a result for that. "
                        "Please try rephrasing your question."
                    )

                # Update session state
                st.session_state.chat_data_questions_session = result.session
                st.session_state.chat_message_count = (
                    st.session_state.get('chat_message_count', 0) + 1
                )
                if result.code:
                    st.session_state.chat_execution_count = (
                        st.session_state.get('chat_execution_count', 0) + 1
                    )

                # Store result DataFrame if produced
                if result.result_df is not None:
                    st.session_state.chat_questions_result_df = result.result_df
                else:
                    st.session_state.chat_questions_result_df = None

                record_turn(
                    "data_questions",
                    user_input,
                    result.assistant_message,
                    assistant_msg=result.session.messages[-1],
                    code=result.code or None,
                    produced_table=result.result_df is not None,
                    history=session.messages,
                    df=source_df,
                )

            except Exception as e:
                st.session_state.chat_last_error = str(e)
                st.session_state.chat_last_failed_input = user_input
                record_turn(
                    "data_questions", user_input, "", error=str(e),
                    history=session.messages, df=source_df,
                )

        st.rerun()

    # Display persistent error from last attempt, with a one-click retry.
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
        failed_input = st.session_state.get('chat_last_failed_input')
        if failed_input and st.button("Retry", key="chat_retry_dq"):
            st.session_state.pop('chat_last_failed_input', None)
            st.session_state.chat_pending_prompt_dq = failed_input
            st.rerun()
    else:
        st.session_state.pop('chat_last_failed_input', None)
