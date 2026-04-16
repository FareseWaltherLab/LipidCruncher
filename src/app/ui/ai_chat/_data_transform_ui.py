"""
Data transformation mode UI for the AI chat.

Renders the chat conversation for transforming arbitrary data into
the LipidCruncher Generic format. When a valid transformation is
produced, offers an "Apply Transformation" button that feeds the
result into the normal pipeline.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.format_detection import DataFormat

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


def render_transform_chat(source_df) -> None:
    """Render the data transformation chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_transform_session

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        st.info(
            "Describe your data format and I'll help transform it into "
            "the LipidCruncher Generic format. For example:\n\n"
            "*'My data has columns: lipid_name, lipid_class, sample_A, sample_B, sample_C. "
            "Please transform it.'*"
        )

    # Render existing messages
    _render_messages(session)

    # Show the last transformation result if successful
    transform_result_df = st.session_state.get('chat_transform_result')
    if transform_result_df is not None:
        with st.expander("Preview transformed data", expanded=True):
            st.dataframe(transform_result_df.head(10), use_container_width=True)
            st.caption(f"{len(transform_result_df)} rows, {len(transform_result_df.columns)} columns")

        if st.button("Apply Transformation", key="chat_apply_transform", type="primary"):
            st.session_state.standardized_df = transform_result_df
            st.session_state.format_type = DataFormat.GENERIC
            # Clear the chat session
            st.session_state.chat_transform_session = None
            st.session_state.chat_transform_result = None
            st.rerun()

    # Rate limit check
    if st.session_state.get('chat_message_count', 0) >= MAX_MESSAGES:
        st.warning("Message limit reached for this session. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_transform"):
            st.session_state.chat_transform_session = None
            st.session_state.chat_transform_result = None
            st.session_state.chat_message_count = 0
            st.session_state.chat_execution_count = 0
            st.rerun()
        return

    # Chat input
    user_input = st.chat_input(
        "Describe your data or ask for changes...",
        key="chat_input_transform",
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI transformation
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = StreamlitAdapter.run_ai_transform(
                        user_message=user_input,
                        session=session,
                        source_df=source_df,
                    )

                    # Update session state
                    st.session_state.chat_transform_session = result.session
                    st.session_state.chat_message_count = (
                        st.session_state.get('chat_message_count', 0) + 1
                    )

                    # Display response
                    st.markdown(result.assistant_message)

                    if result.transformation and result.transformation.code:
                        st.session_state.chat_execution_count = (
                            st.session_state.get('chat_execution_count', 0) + 1
                        )
                        with st.expander("View code"):
                            st.code(result.transformation.code, language="python")

                    # Store successful result for "Apply" button
                    if result.transformation and result.transformation.success:
                        st.session_state.chat_transform_result = (
                            result.transformation.result_df
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        st.rerun()
