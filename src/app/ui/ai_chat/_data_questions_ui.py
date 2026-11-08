"""
Ask Your Data mode UI for the AI chat (internally ``data_questions``).

Renders the chat conversation for asking ad-hoc questions about the
user's data — describing the dataset, computing summaries, counting
species — and for plotting it. Each turn may return a table, a figure,
or both, shown under its message.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._turn_errors import record_notice, render_failure, render_notice
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
    "Plot the top 15 most abundant lipids",
]


_ABOUT_TAB = """
**What this tab does** — answers questions about the dataset loaded in the
sidebar, and plots it. The AI writes the code, that code runs here on your
data, and you see what it returned: an answer, a table, a figure, or all three.

**How to read an answer** — the answer is written from what the code actually
returned; **View code** shows the code that produced it, **Result Table** holds
the rows it returned. The code is the method — read it before quoting a number.

**Ask it things like**

- *How many species per class, per condition?*
- *Which samples have the most missing values?*
- *List the internal standards present in this dataset*
- *Box plot of TG species per condition, log10 scale, coloured by condition*
- *Which lipids differ between WT and KO? Show them as a volcano plot.*

**Working with a figure** — the Plotly toolbar (top-right of the chart) zooms,
pans and saves a PNG. Refine a figure by asking for changes ("same plot, but
split by chain length") rather than starting over.

**Raw data** — if you imported the main pipeline's *raw* data (not cleaned,
zero-filtered or normalized), answers and figure titles say so. For results you
can report, import the normalized data.

**This is not Module 3.** Where LipidCruncher has its own method (volcano,
class bar chart, pie charts, saturation, chain length), the AI uses it; for
anything custom, no tests or corrections are applied unless you ask, and the
built-in analyses remain the reference implementation.

**Getting better answers**

- Say what a column holds when its header is cryptic — the AI infers meaning
  from column names and three sample rows.
- Ask one thing at a time; follow-up questions keep the earlier context.
- If a result looks wrong, say what's wrong instead of rephrasing from scratch.

**It never modifies your dataset** — everything here is read-only. To produce a
new file, use **Transform to Generic**.
"""


def _render_outputs(index: int) -> None:
    """The figure and table that message ``index``'s turn returned. The
    latest table opens; earlier ones stay folded so the chat stays short."""
    outputs = st.session_state.get('chat_questions_outputs') or {}
    if index not in outputs:
        return
    table, figure = outputs[index]
    if figure is not None:
        st.plotly_chart(figure, width='stretch', key=f"dq_figure_{index}")
    if table is not None:
        with st.expander("Result Table", expanded=index == max(outputs)):
            st.dataframe(table, width='stretch', key=f"dq_table_{index}")
            st.caption(f"{len(table)} rows, {len(table.columns)} columns")


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
                "counts, missing values — or ask for a plot of it."
            )

    render_messages(session, outputs=_render_outputs)

    render_example_prompts(_EXAMPLE_PROMPTS, "dq", "chat_pending_prompt_dq")
    if usage_refused():
        return

    user_input = next_user_input(
        "chat_pending_prompt_dq",
        "Ask a question or describe a plot...",
        "chat_input_questions",
    )
    if user_input:
        if start_turn_refused():
            return

        def on_result(result, streamed):
            # A turn with no text, no code and nothing returned: say so
            # rather than leaving a blank assistant bubble.
            if (not streamed.strip() and result.result_df is None
                    and result.figure is None and not result.code):
                record_notice('dq', (
                    "I couldn't produce a result for that. "
                    "Please try rephrasing your question."
                ))
            st.session_state.chat_data_questions_session = result.session
            if result.code:
                StreamlitAdapter.record_ai_execution()
            if result.result_df is not None or result.figure is not None:
                st.session_state.setdefault('chat_questions_outputs', {})[
                    len(result.session.messages) - 1
                ] = (result.result_df, result.figure)
            return {"code": result.code or None,
                    "produced_table": result.result_df is not None,
                    "produced_figure": result.figure is not None,
                    "figure": result.figure,
                    "unverified_numbers": result.unverified_numbers,
                    "sandbox": result.sandbox_result}

        run_code_turn(
            "dq", "data_questions", user_input, session.messages,
            lambda: StreamlitAdapter.stream_ai_data_questions(
                user_message=user_input,
                session=session,
                source_df=source_df,
            ),
            on_result,
            answer_streams=True,
            df=source_df,
            experiment=StreamlitAdapter.studio_experiment(),
        )

    # Display persistent error from last attempt, with a one-click retry.
    render_notice('dq')
    render_failure('dq', 'chat_pending_prompt_dq')
