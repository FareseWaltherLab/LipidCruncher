"""
Data transformation mode UI for the AI chat.

Renders the chat conversation for transforming arbitrary data into
the LipidCruncher Generic format. When a valid transformation is
produced, offers a download button so the user can save the result
and upload it to the main pipeline.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._interaction_log import record_turn, render_feedback

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "Transform my data to Generic format",
    "What columns need to be removed?",
    "Filter out low-quality rows, then transform",
]


_ABOUT_TAB = """
**What this tab does** — converts a dataset LipidCruncher can't read directly
into the **Generic** format the main pipeline accepts. The AI writes the
conversion code, runs it here on your file, and hands back the result.

**The Generic format** is deliberately plain: column 1 = lipid names (any
naming convention — LipidCruncher standardizes them to LIPID MAPS notation on
upload), columns 2+ = one numeric intensity column per sample, one row per
species, and **no other columns** (mass, retention time, scores and the like are
dropped).

**The round trip**

1. Describe your file, or just ask for the transformation.
2. Check **Preview transformed data**, and **View code** for what was actually
   done to your numbers.
3. Download the CSV.
4. Upload it on the main pipeline as a **Generic** dataset and run the normal
   Module 1 → 2 → 3 analysis.

**Also useful before converting** — asking to *see* something ("list the internal
standards you can find", "show the distinct values in column X") prints an
answer instead of producing a file.

**Check the preview before you download.** The AI decides which columns are
samples and which are metadata from their names; if it guessed wrong, say which
columns are which and ask it to redo the transform.
"""


def _render_messages(session) -> None:
    """Render all chat messages in the session."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.code:
                with st.expander("View code"):
                    st.code(msg.code, language="python")
            render_feedback(msg)


def render_transform_chat(source_df) -> None:
    """Render the data transformation chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_transform_session

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(_ABOUT_TAB)

    # Show introductory message with format diagnostic
    if len(session.messages) == 0:
        diagnostic = st.session_state.get('chat_transform_diagnostic')
        if diagnostic and diagnostic.is_supported:
            with st.chat_message("assistant"):
                st.markdown(diagnostic.message)
        elif diagnostic and not diagnostic.is_supported:
            with st.chat_message("assistant"):
                st.markdown(diagnostic.message)
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    "I can help you transform your data into LipidCruncher's "
                    "Generic format. Tell me about your data or ask me to "
                    "transform it."
                )

    # Render existing messages
    _render_messages(session)

    # Show the last transformation result if successful (renders before
    # the example tiles so the full AI response — message, code,
    # preview — sits together; tiles come after as next-step prompts).
    transform_result_df = st.session_state.get('chat_transform_result')
    if transform_result_df is not None:
        with st.expander("Preview transformed data", expanded=True):
            st.dataframe(transform_result_df.head(10), width='stretch')
            st.caption(
                f"{len(transform_result_df)} rows, "
                f"{len(transform_result_df.columns)} columns"
            )

            csv_data = transform_result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download transformed CSV",
                data=csv_data,
                file_name="transformed_data.csv",
                mime="text/csv",
                key="chat_download_transform",
            )
            st.caption(
                "Download this file, then upload it to LipidCruncher as a "
                "**Generic** format dataset to run it through the analysis pipeline."
            )

    # Clickable example prompts (always visible, only for unsupported formats)
    diagnostic = st.session_state.get('chat_transform_diagnostic')
    if not diagnostic or not diagnostic.is_supported:
        cols = st.columns(len(_EXAMPLE_PROMPTS))
        for i, prompt in enumerate(_EXAMPLE_PROMPTS):
            with cols[i]:
                if st.button(
                    prompt,
                    key=f"tf_example_{i}",
                    width='stretch',
                ):
                    st.session_state.chat_pending_prompt_transform = prompt
                    st.rerun()

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    exec_count = st.session_state.get('chat_execution_count', 0)
    if msg_count >= MAX_MESSAGES or exec_count >= MAX_EXECUTIONS:
        limit_type = "Message" if msg_count >= MAX_MESSAGES else "Execution"
        st.warning(f"{limit_type} limit reached for this session. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_transform"):
            st.session_state.chat_transform_session = None
            st.session_state.chat_transform_result = None
            st.session_state.chat_message_count = 0
            st.session_state.chat_execution_count = 0
            st.rerun()
        return

    # Chat input (check for pending example prompt first)
    user_input = st.session_state.pop('chat_pending_prompt_transform', None)
    if user_input is None:
        user_input = st.chat_input(
            "Ask about your data or request a transformation...",
            key="chat_input_transform",
        )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI transformation (explanation streamed; code/preview render
        # canonically after the rerun below)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    gen, finalize = StreamlitAdapter.stream_ai_transform(
                        user_message=user_input,
                        session=session,
                        source_df=source_df,
                    )
                    streamed = st.write_stream(gen)
                result = finalize()

                # If the turn produced nothing at all, say so rather than
                # leaving a blank assistant bubble.
                if (not (streamed or "").strip()
                        and result.transformation is None):
                    st.info(
                        "I couldn't produce a result for that. "
                        "Please try rephrasing your request."
                    )

                # Update session state
                st.session_state.chat_transform_session = result.session
                st.session_state.chat_message_count = (
                    st.session_state.get('chat_message_count', 0) + 1
                )
                if result.transformation and result.transformation.code:
                    st.session_state.chat_execution_count = (
                        st.session_state.get('chat_execution_count', 0) + 1
                    )

                # Store successful transformation result for download.
                # Skip info-gathering turns (no result_df) so they
                # don't clear the previous transform's preview.
                if (result.transformation
                        and result.transformation.success
                        and not result.transformation.is_info_only):
                    st.session_state.chat_transform_result = (
                        result.transformation.result_df
                    )
                elif (result.transformation
                        and not result.transformation.success):
                    # Clear stale result on failed transformation
                    st.session_state.chat_transform_result = None

                _t = result.transformation
                record_turn(
                    "transform",
                    user_input,
                    result.assistant_message,
                    assistant_msg=result.session.messages[-1],
                    code=(_t.code if _t else None) or None,
                    produced_table=bool(
                        _t and _t.success and not _t.is_info_only
                    ),
                    history=session.messages,
                    df=source_df,
                )

            except Exception as e:
                st.session_state.chat_last_error = str(e)
                st.session_state.chat_last_failed_input = user_input
                record_turn(
                    "transform", user_input, "", error=str(e),
                    history=session.messages, df=source_df,
                )

        st.rerun()

    # Display persistent error from last attempt, with a one-click retry.
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
        failed_input = st.session_state.get('chat_last_failed_input')
        if failed_input and st.button("Retry", key="chat_retry_transform"):
            st.session_state.pop('chat_last_failed_input', None)
            st.session_state.chat_pending_prompt_transform = failed_input
            st.rerun()
    else:
        st.session_state.pop('chat_last_failed_input', None)
