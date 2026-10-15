"""
Main chat panel UI for AI-powered interactions.

The ``display_*`` entry points start each tab's conversation and delegate
to the tab's UI module. The helpers below are what the four tab UIs share:
message rendering, example prompts, the usage-limit gate, the chat input,
and one code-writing turn.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn, render_feedback
from ._safe_markdown import safe_markdown, safe_stream
from ._turn_errors import clear_failure, record_failure


def render_messages(session) -> None:
    """Each message, its code (if any) under "View code", and feedback."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(safe_markdown(msg.content))
            if msg.code:
                with st.expander("View code"):
                    st.code(msg.code, language="python")
            render_feedback(msg)


def render_example_prompts(prompts, key_prefix: str, pending_key: str) -> None:
    """One button per example prompt; a click sends it as the next turn."""
    cols = st.columns(len(prompts))
    for i, prompt in enumerate(prompts):
        with cols[i]:
            if st.button(
                prompt,
                key=f"{key_prefix}_example_{i}",
                width='stretch',
            ):
                st.session_state[pending_key] = prompt
                st.rerun()


def usage_refused(runs_code: bool = True) -> bool:
    """Show why no turn may start now, if so (before drawing the input)."""
    refusal = StreamlitAdapter.ai_usage_refusal(runs_code=runs_code)
    if refusal:
        st.warning(refusal)
    return bool(refusal)


def start_turn_refused(runs_code: bool = True) -> bool:
    """Count the turn about to run; show the refusal if a limit stops it."""
    refusal = StreamlitAdapter.start_ai_turn(runs_code=runs_code)
    if refusal:
        st.warning(refusal)
    return bool(refusal)


def next_user_input(pending_key: str, placeholder: str, key: str):
    """A clicked example (or Retry) prompt if there is one, else the chat
    input's value."""
    user_input = st.session_state.pop(pending_key, None)
    if user_input is None:
        user_input = st.chat_input(placeholder, key=key)
    return user_input


def run_code_turn(
    tab: str,
    log_tab: str,
    user_input: str,
    history,
    stream,
    on_result,
    spinner: str = "Thinking...",
    **log_context,
) -> None:
    """Run one turn of a code-writing tab, then rerun.

    Shows the question, streams the explanation from ``stream()``'s
    generator, then calls its ``finalize()``. ``on_result(result,
    streamed_text)`` stores the tab's result and returns the tab's own
    ``record_turn`` fields (code, produced_table, ...). A failure is kept
    for the tab's Retry. Either way the turn is logged, with
    ``log_context`` (df, experiment).
    """
    with st.chat_message("user"):
        st.markdown(user_input)

    # The explanation streams here; code, tables and figures render
    # canonically after the rerun below.
    with st.chat_message("assistant"):
        try:
            with st.spinner(spinner):
                generator, finalize = stream()
                streamed = st.write_stream(safe_stream(generator))
            result = finalize()
            clear_failure(tab)
            fields = on_result(result, streamed or "")
            record_turn(
                log_tab, user_input, result.assistant_message,
                assistant_msg=result.session.messages[-1],
                history=history, **fields, **log_context,
            )
        except Exception as e:
            record_failure(tab, user_input, str(e))
            record_turn(
                log_tab, user_input, "", error=str(e),
                history=history, **log_context,
            )

    st.rerun()


def display_transform_chat(source_df) -> None:
    """Display the Transform to Generic tab's chat.

    Creates a new transform session if one doesn't exist, then delegates
    to the transform-specific UI.

    Args:
        source_df: The user's uploaded DataFrame (pre-standardization).
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._data_transform_ui import render_transform_chat

    if st.session_state.get('chat_transform_session') is None:
        diagnostic, session = AIChatWorkflow.start_transform_chat(
            source_df,
            raw_pipeline_import=(
                st.session_state.get('chat_studio_pipeline_source') == 'raw'
            ),
        )
        st.session_state.chat_transform_diagnostic = diagnostic
        st.session_state.chat_transform_session = session

    render_transform_chat(source_df)


def display_viz_chat(analysis_df, experiment) -> None:
    """Display the Custom Visualization tab's chat.

    Creates a new viz session if one doesn't exist, then delegates to the
    viz-specific UI.

    Args:
        analysis_df: The analysis-ready DataFrame.
        experiment: Experiment configuration.
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._viz_generation_ui import render_viz_chat

    # Initialize session if needed
    if st.session_state.get('chat_viz_session') is None:
        st.session_state.chat_viz_session = (
            AIChatWorkflow.start_viz_session(analysis_df, experiment)
        )

    render_viz_chat(analysis_df, experiment)


def display_data_questions_chat(source_df) -> None:
    """Display the AI Data Questions chat panel.

    Creates a new data questions session if one doesn't exist,
    then delegates to the questions-specific UI.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._data_questions_ui import render_data_questions_chat

    # Initialize session if needed
    if st.session_state.get('chat_data_questions_session') is None:
        # Pipeline imports replace the data with the standardized /
        # normalized version, which matches Generic by structure even when
        # the user originally uploaded LipidSearch / MS-DIAL. Carry the
        # main app's format choice through so the AI doesn't tell the user
        # their LipidSearch dataset is "Generic Format".
        format_label_override = None
        if st.session_state.get('chat_studio_pipeline_source') is not None:
            format_label_override = st.session_state.get('format_type')
        st.session_state.chat_data_questions_session = (
            AIChatWorkflow.start_data_questions_session(
                source_df, format_label_override=format_label_override,
            )
        )

    render_data_questions_chat(source_df)


def display_lipidcruncher_questions_chat() -> None:
    """Display the LipidCruncher Questions chat panel.

    Answers questions about LipidCruncher features, methods,
    analysis options, and terminology. No data access needed.
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._lipidcruncher_questions_ui import render_lc_questions_chat

    # Initialize session if needed
    if st.session_state.get('chat_lc_questions_session') is None:
        st.session_state.chat_lc_questions_session = (
            AIChatWorkflow.start_lc_questions_session()
        )

    render_lc_questions_chat()

