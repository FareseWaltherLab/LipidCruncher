"""
Data Questions mode UI for the AI chat.

Renders the chat conversation for asking ad-hoc questions about the
user's data: describing the dataset, computing summaries, counting
species, explaining LipidCruncher features, etc.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20


def _render_messages(session) -> None:
    """Render all chat messages in the session."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.code:
                with st.expander("View code"):
                    st.code(msg.code, language="python")


def render_data_questions_chat(source_df) -> None:
    """Render the data questions chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_data_questions_session

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "Ask questions about your uploaded dataset. For example:\n\n"
                "- *How many lipid species are in this dataset?*\n"
                "- *What lipid classes are present?*\n"
                "- *Are there any missing values?*\n"
                "- *Give me summary statistics for the sample columns*\n"
                "- *What does the distribution of intensities look like?*"
            )

    # Render existing messages
    _render_messages(session)

    # Show last result table if available
    result_df = st.session_state.get('chat_questions_result_df')
    if result_df is not None:
        with st.expander("Result Table", expanded=True):
            st.dataframe(result_df, use_container_width=True)
            st.caption(
                f"{len(result_df)} rows, {len(result_df.columns)} columns"
            )

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

    # Chat input
    user_input = st.chat_input(
        "Ask a question about your data...",
        key="chat_input_questions",
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI data questions
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = StreamlitAdapter.run_ai_data_questions(
                        user_message=user_input,
                        session=session,
                        source_df=source_df,
                    )

                    # Update session state
                    st.session_state.chat_data_questions_session = result.session
                    st.session_state.chat_message_count = (
                        st.session_state.get('chat_message_count', 0) + 1
                    )

                    # Display response
                    st.markdown(result.assistant_message)

                    if result.code:
                        st.session_state.chat_execution_count = (
                            st.session_state.get('chat_execution_count', 0) + 1
                        )
                        with st.expander("View code"):
                            st.code(result.code, language="python")

                    # Store result DataFrame if produced
                    if result.result_df is not None:
                        st.session_state.chat_questions_result_df = result.result_df
                    else:
                        st.session_state.chat_questions_result_df = None

                except Exception as e:
                    st.session_state.chat_last_error = str(e)

        st.rerun()

    # Display persistent error from last attempt
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
