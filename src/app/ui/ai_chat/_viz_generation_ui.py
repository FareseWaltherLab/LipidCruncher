"""
Visualization generation mode UI for the AI chat.

Renders the chat conversation for generating custom Plotly
visualizations from the analysis data. Figures are displayed
inline with export buttons.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn, render_feedback

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "Species count per class",
    "Top 15 most abundant lipids",
    "CV distribution across lipids",
]


def _figure_has_data(fig) -> bool:
    """True if the figure has at least one trace with non-empty x or y."""
    if fig is None or not getattr(fig, "data", None):
        return False
    for trace in fig.data:
        x = getattr(trace, "x", None)
        y = getattr(trace, "y", None)
        if (x is not None and len(x) > 0) or (y is not None and len(y) > 0):
            return True
    return False


def _render_messages(session) -> None:
    """Render all chat messages with inline figures.

    Generated code is shown in a collapsed "View code" expander next to
    every assistant message that produced code, so the user can verify
    what was run. Empty / missing figures still surface the code with a
    troubleshooting-focused label so the user knows where to look.
    """
    for i, msg in enumerate(session.messages):
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            render_feedback(msg)

        # Check if there's a stored figure for this message
        fig_key = f"chat_viz_fig_{i}"
        fig = st.session_state.get('chat_viz_figures', {}).get(fig_key)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        # Show generated code on every assistant turn. Use a different
        # label when the figure is missing/empty so it doubles as the
        # troubleshooting affordance.
        if msg.role == "assistant" and msg.code:
            label = (
                "Show generated code (for troubleshooting)"
                if not _figure_has_data(fig)
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

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    exec_count = st.session_state.get('chat_execution_count', 0)
    if msg_count >= MAX_MESSAGES or exec_count >= MAX_EXECUTIONS:
        limit_type = "Message" if msg_count >= MAX_MESSAGES else "Execution"
        st.warning(f"{limit_type} limit reached for this session. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_viz"):
            st.session_state.chat_viz_session = None
            st.session_state.chat_viz_figures = {}
            st.session_state.chat_message_count = 0
            st.session_state.chat_execution_count = 0
            st.rerun()
        return

    # Chat input (check for pending example prompt first)
    user_input = st.session_state.pop('chat_pending_prompt_viz', None)
    if user_input is None:
        user_input = st.chat_input(
            "Describe the visualization you want...",
            key="chat_input_viz",
        )

    if user_input:
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
                    streamed = st.write_stream(gen)
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
                st.session_state.chat_message_count = (
                    st.session_state.get('chat_message_count', 0) + 1
                )

                if result.sandbox_result and result.sandbox_result.success:
                    st.session_state.chat_execution_count = (
                        st.session_state.get('chat_execution_count', 0) + 1
                    )

                    if result.sandbox_result.figure is not None:
                        # Store figure for replay on rerun
                        fig_index = len(result.session.messages) - 1
                        fig_key = f"chat_viz_fig_{fig_index}"
                        if 'chat_viz_figures' not in st.session_state:
                            st.session_state.chat_viz_figures = {}
                        st.session_state.chat_viz_figures[fig_key] = (
                            result.sandbox_result.figure
                        )

                    if result.sandbox_result.stdout:
                        with st.expander("Output"):
                            st.text(result.sandbox_result.stdout)

                _sr = result.sandbox_result
                record_turn(
                    "visualization",
                    user_input,
                    result.assistant_message,
                    assistant_msg=result.session.messages[-1],
                    code=result.session.messages[-1].code,
                    produced_figure=bool(_sr and _sr.figure is not None),
                )

            except Exception as e:
                st.session_state.chat_last_error = str(e)
                st.session_state.chat_last_failed_input = user_input
                record_turn("visualization", user_input, "", error=str(e))

        st.rerun()

    # Display persistent error from last attempt, with a one-click retry.
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
        failed_input = st.session_state.get('chat_last_failed_input')
        if failed_input and st.button("Retry", key="chat_retry_viz"):
            st.session_state.pop('chat_last_failed_input', None)
            st.session_state.chat_pending_prompt_viz = failed_input
            st.rerun()
    else:
        st.session_state.pop('chat_last_failed_input', None)
