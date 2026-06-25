"""
Visualization generation mode UI for the AI chat.

Renders the chat conversation for generating custom Plotly
visualizations from the analysis data. Figures are displayed
inline with export buttons.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter

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
    print(
        f"[VIZ-UI] render ENTER: "
        f"pending={st.session_state.get('chat_pending_prompt_viz')!r}, "
        f"msg_count={st.session_state.get('chat_message_count', 0)}, "
        f"exec_count={st.session_state.get('chat_execution_count', 0)}, "
        f"session_msgs={len(st.session_state.chat_viz_session.messages) if st.session_state.get('chat_viz_session') else 'NO_SESSION'}, "
        f"stored_figs={list((st.session_state.get('chat_viz_figures') or {}).keys())}",
        flush=True,
    )
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
                print(f"[VIZ-UI] BUTTON CLICKED: {prompt!r}", flush=True)
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
    print(f"[VIZ-UI] after pop: user_input={user_input!r}", flush=True)
    if user_input is None:
        user_input = st.chat_input(
            "Describe the visualization you want...",
            key="chat_input_viz",
        )

    if user_input:
        print(f"[VIZ-UI] processing user_input: {user_input!r}, df shape={analysis_df.shape}", flush=True)
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
                    st.write_stream(gen)
                result = finalize()
                print(
                    f"[VIZ-UI] result: msg_len={len(result.assistant_message)}, "
                    f"sandbox_success={result.sandbox_result.success if result.sandbox_result else None}, "
                    f"has_fig={result.sandbox_result.figure is not None if result.sandbox_result else None}, "
                    f"sandbox_err={(result.sandbox_result.error_message[:200] if result.sandbox_result else '')!r}, "
                    f"session_msgs_after={len(result.session.messages)}",
                    flush=True,
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
                        print(f"[VIZ-UI] storing fig at key={fig_key}", flush=True)
                        if 'chat_viz_figures' not in st.session_state:
                            st.session_state.chat_viz_figures = {}
                        st.session_state.chat_viz_figures[fig_key] = (
                            result.sandbox_result.figure
                        )

                    if result.sandbox_result.stdout:
                        with st.expander("Output"):
                            st.text(result.sandbox_result.stdout)

            except Exception as e:
                import traceback
                print(f"[VIZ-UI] EXCEPTION: {type(e).__name__}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                st.session_state.chat_last_error = str(e)

        print("[VIZ-UI] calling st.rerun()", flush=True)
        st.rerun()

    # Display persistent error from last attempt
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
