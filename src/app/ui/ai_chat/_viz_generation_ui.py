"""
Visualization generation mode UI for the AI chat.

Renders the chat conversation for generating custom Plotly
visualizations from the analysis data. Figures are displayed
inline with export buttons.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.ai_chat import figure_has_data
from ._interaction_log import render_feedback
from ._safe_markdown import safe_markdown
from ._turn_errors import render_failure
from .chat_panel import (
    next_user_input,
    render_example_prompts,
    run_code_turn,
    start_turn_refused,
    usage_refused,
)

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "Species count per class",
    "Top 15 most abundant lipids",
    "CV distribution across lipids",
]


_ABOUT_TAB = """
**What this tab does** — builds a custom Plotly figure from your data. The AI
writes the plotting code, it runs here, and the figure appears in the
conversation.

**What it needs** — data that has already been through **Module 1** of the main
pipeline (cleaned, zero-filtered, normalized), i.e. `LipidMolec`, `ClassKey`, and
`intensity[sN]` / `concentration[sN]` columns. The usual route is *Start
Crunching* on the home page, then **Import from main pipeline** in the sidebar.

**Describing a plot** — name the chart type, what goes on each axis, how to group
or colour it, and which classes, species, or conditions to include. For example:
*box plot of TG species per condition, log10 scale, coloured by condition*.

**Working with a figure**

- The Plotly toolbar (top-right of the chart) zooms, pans, and saves a PNG.
- **View code** shows exactly how the figure was built — including any
  aggregation or transformation applied on the way.
- Refine by asking for changes ("same plot, but split by chain length") rather
  than starting over; the AI keeps the previous figure in context.

**This is not Module 3.** Nothing here is statistically validated: no tests,
corrections, or significance markers are applied unless you ask for them, and
even then the built-in analyses remain the reference implementation.
"""


def _render_messages(session) -> None:
    """Render all chat messages with inline figures.

    Generated code is shown in a collapsed "View code" expander next to
    every assistant message that produced code, so the user can verify
    what was run. Empty / missing figures still surface the code with a
    troubleshooting-focused label so the user knows where to look.
    """
    for i, msg in enumerate(session.messages):
        with st.chat_message(msg.role):
            st.markdown(safe_markdown(msg.content))
            render_feedback(msg)

        # Check if there's a stored figure for this message
        fig_key = f"chat_viz_fig_{i}"
        fig = st.session_state.get('chat_viz_figures', {}).get(fig_key)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')

        # Show generated code on every assistant turn. Use a different
        # label when the figure is missing/empty so it doubles as the
        # troubleshooting affordance.
        if msg.role == "assistant" and msg.code:
            label = (
                "Show generated code (for troubleshooting)"
                if not figure_has_data(fig)
                else "View code"
            )
            with st.expander(label):
                st.code(msg.code, language="python")


def render_viz_chat(analysis_df, experiment) -> None:
    """Render the custom visualization chat interface.

    Args:
        analysis_df: The analysis-ready DataFrame.
        experiment: Experiment configuration.
    """
    session = st.session_state.chat_viz_session

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(_ABOUT_TAB)

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        validation = st.session_state.get('chat_viz_validation')
        with st.chat_message("assistant"):
            if validation:
                st.markdown(validation.message + "\n\n")
            st.markdown(
                "Describe the visualization you'd like and I'll "
                "generate it."
            )

    # Render existing messages
    _render_messages(session)

    render_example_prompts(_EXAMPLE_PROMPTS, "viz", "chat_pending_prompt_viz")
    if usage_refused():
        return

    user_input = next_user_input(
        "chat_pending_prompt_viz",
        "Describe the visualization you want...",
        "chat_input_viz",
    )
    if user_input:
        if start_turn_refused():
            return

        def on_result(result, streamed):
            sr = result.sandbox_result
            if not streamed.strip() and (sr is None or sr.figure is None):
                st.info(
                    "I couldn't produce a visualization for that. "
                    "Please try rephrasing your request."
                )
            st.session_state.chat_viz_session = result.session
            # Count every run: failing code costs an execution too.
            if sr:
                StreamlitAdapter.record_ai_execution()
            if sr and sr.success and sr.figure is not None:
                # Stored per message index for replay on rerun.
                fig_key = f"chat_viz_fig_{len(result.session.messages) - 1}"
                st.session_state.setdefault('chat_viz_figures', {})[fig_key] = (
                    sr.figure
                )
            return {"code": result.session.messages[-1].code,
                    "produced_figure": bool(sr and sr.figure is not None),
                    "figure": sr.figure if sr else None,
                    "unverified_numbers": result.unverified_numbers}

        run_code_turn(
            "viz", "visualization", user_input, session.messages,
            lambda: StreamlitAdapter.stream_ai_viz(
                user_message=user_input,
                session=session,
                analysis_df=analysis_df,
                experiment=experiment,
            ),
            on_result,
            spinner="Generating visualization...",
            df=analysis_df,
            experiment=experiment,
        )

    # Display persistent error from last attempt, with a one-click retry.
    render_failure('viz', 'chat_pending_prompt_viz')
