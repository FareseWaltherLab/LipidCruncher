"""
Visualization generation mode UI for the AI chat.

Renders the chat conversation for generating custom Plotly
visualizations from the analysis data. Figures are displayed
inline with export buttons.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.ai_chat import figure_has_data
from ._interaction_log import record_turn, render_feedback
from ._safe_markdown import safe_markdown, safe_stream
from ._turn_errors import clear_failure, record_failure, render_failure

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

    # Clickable example prompts (always visible)
    cols = st.columns(len(_EXAMPLE_PROMPTS))
    for i, prompt in enumerate(_EXAMPLE_PROMPTS):
        with cols[i]:
            if st.button(
                prompt,
                key=f"viz_example_{i}",
                width='stretch',
            ):
                st.session_state.chat_pending_prompt_viz = prompt
                st.rerun()

    refusal = StreamlitAdapter.ai_usage_refusal()
    if refusal:
        st.warning(refusal)
        return

    # Chat input (check for pending example prompt first)
    user_input = st.session_state.pop('chat_pending_prompt_viz', None)
    if user_input is None:
        user_input = st.chat_input(
            "Describe the visualization you want...",
            key="chat_input_viz",
        )

    if user_input:
        refusal = StreamlitAdapter.start_ai_turn()
        if refusal:
            st.warning(refusal)
            return

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI visualization (explanation streamed; figure/code render
        # canonically after the rerun below)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Generating visualization..."):
                    gen, finalize = StreamlitAdapter.stream_ai_viz(
                        user_message=user_input,
                        session=session,
                        analysis_df=analysis_df,
                        experiment=experiment,
                    )
                    streamed = st.write_stream(safe_stream(gen))
                result = finalize()

                # If the turn produced neither text nor a figure, say so
                # rather than leaving a blank assistant bubble.
                if (not (streamed or "").strip()
                        and (result.sandbox_result is None
                             or result.sandbox_result.figure is None)):
                    st.info(
                        "I couldn't produce a visualization for that. "
                        "Please try rephrasing your request."
                    )

                # Update session state
                st.session_state.chat_viz_session = result.session
                clear_failure('viz')

                # Count every run, as the other tabs do: failing code costs
                # an execution too.
                if result.sandbox_result:
                    StreamlitAdapter.record_ai_execution()
                if result.sandbox_result and result.sandbox_result.success:
                    if result.sandbox_result.figure is not None:
                        # Store figure for replay on rerun
                        fig_index = len(result.session.messages) - 1
                        fig_key = f"chat_viz_fig_{fig_index}"
                        if 'chat_viz_figures' not in st.session_state:
                            st.session_state.chat_viz_figures = {}
                        st.session_state.chat_viz_figures[fig_key] = (
                            result.sandbox_result.figure
                        )

                _sr = result.sandbox_result
                record_turn(
                    "visualization",
                    user_input,
                    result.assistant_message,
                    assistant_msg=result.session.messages[-1],
                    code=result.session.messages[-1].code,
                    produced_figure=bool(_sr and _sr.figure is not None),
                    history=session.messages,
                    df=analysis_df,
                    experiment=experiment,
                    figure=_sr.figure if _sr else None,
                )

            except Exception as e:
                record_failure('viz', user_input, str(e))
                record_turn(
                    "visualization", user_input, "", error=str(e),
                    history=session.messages, df=analysis_df,
                    experiment=experiment,
                )

        st.rerun()

    # Display persistent error from last attempt, with a one-click retry.
    render_failure('viz', 'chat_pending_prompt_viz')
