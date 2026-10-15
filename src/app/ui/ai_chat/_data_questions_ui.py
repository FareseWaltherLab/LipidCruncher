"""
Data Questions mode UI for the AI chat.

Renders the chat conversation for asking ad-hoc questions about the
user's data: describing the dataset, computing summaries, counting
species, etc.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._turn_errors import render_failure
from .chat_panel import (
    next_user_input,
    render_example_prompts,
    render_messages,
    run_code_turn,
    start_turn_refused,
    usage_refused,
)

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

    render_messages(session)

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

    render_example_prompts(_EXAMPLE_PROMPTS, "dq", "chat_pending_prompt_dq")
    if usage_refused():
        return

    user_input = next_user_input(
        "chat_pending_prompt_dq",
        "Ask a question about your data...",
        "chat_input_questions",
    )
    if user_input:
        if start_turn_refused():
            return

        def on_result(result, streamed):
            # A turn with no text, no code and no table: say so rather
            # than leaving a blank assistant bubble.
            if (not streamed.strip() and result.result_df is None
                    and not result.code):
                st.info(
                    "I couldn't produce a result for that. "
                    "Please try rephrasing your question."
                )
            st.session_state.chat_data_questions_session = result.session
            if result.code:
                StreamlitAdapter.record_ai_execution()
            st.session_state.chat_questions_result_df = result.result_df
            return {"code": result.code or None,
                    "produced_table": result.result_df is not None}

        run_code_turn(
            "dq", "data_questions", user_input, session.messages,
            lambda: StreamlitAdapter.stream_ai_data_questions(
                user_message=user_input,
                session=session,
                source_df=source_df,
            ),
            on_result,
            df=source_df,
        )

    # Display persistent error from last attempt, with a one-click retry.
    render_failure('dq', 'chat_pending_prompt_dq')
