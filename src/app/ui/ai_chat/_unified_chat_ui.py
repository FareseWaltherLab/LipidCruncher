"""
Unified AI Studio chat UI.

Single chat interface that handles data questions, transformations,
and visualizations in one conversation. No tabs, no "view code".
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter

# Rate limits per session
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20


def _render_messages(session) -> None:
    """Render all chat messages with inline figures and result tables."""
    for i, msg in enumerate(session.messages):
        with st.chat_message(msg.role):
            st.markdown(msg.content)

        # Check for stored figure
        fig_key = f"chat_fig_{i}"
        fig = st.session_state.get('chat_viz_figures', {}).get(fig_key)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')

        # Check for stored result table (with download)
        table_key = f"chat_table_{i}"
        table = st.session_state.get('chat_result_tables', {}).get(table_key)
        if table is not None:
            st.dataframe(table, width='stretch')
            csv_data = table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_data,
                file_name="result_data.csv",
                mime="text/csv",
                key=f"chat_download_table_{i}",
            )


def render_unified_chat(source_df) -> None:
    """Render the unified AI Studio chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_unified_session

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown(
                "I can help you with:\n\n"
                "- **Data questions** — describe your dataset, compute summaries, "
                "explore lipid classes and species\n"
                "- **LipidCruncher questions** — explain features, normalization "
                "methods, statistical tests, and analysis options\n"
                "- **Data transformation** — convert your data into LipidCruncher "
                "Generic format\n"
                "- **Custom visualizations** — create plots from your processed data\n\n"
                "Ask me anything about your lipidomic data!"
            )

    # Render existing messages
    _render_messages(session)

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    exec_count = st.session_state.get('chat_execution_count', 0)
    if msg_count >= MAX_MESSAGES or exec_count >= MAX_EXECUTIONS:
        limit_type = "Message" if msg_count >= MAX_MESSAGES else "Execution"
        st.warning(f"{limit_type} limit reached. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_unified"):
            st.session_state.chat_unified_session = None
            st.session_state.chat_transform_result = None
            st.session_state.chat_viz_figures = {}
            st.session_state.chat_result_tables = {}
            st.session_state.chat_message_count = 0
            st.session_state.chat_execution_count = 0
            st.rerun()
        return

    # Chat input
    user_input = st.chat_input(
        "Ask about your data, request a transformation, or describe a visualization...",
        key="chat_input_unified",
    )

    if user_input:
        import logging
        logger = logging.getLogger(__name__)

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI
        error_text = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = StreamlitAdapter.run_ai_unified(
                        user_message=user_input,
                        session=session,
                        source_df=source_df,
                    )

                    # Update session state
                    st.session_state.chat_unified_session = result.session
                    st.session_state.chat_message_count = (
                        st.session_state.get('chat_message_count', 0) + 1
                    )

                    # Display response
                    if result.assistant_message:
                        st.markdown(result.assistant_message)
                    else:
                        logger.warning("AI returned empty assistant_message")
                        # Don't just show a transient warning — the rerun
                        # will wipe it. The empty message is already in the
                        # session from the workflow.

                    if result.code:
                        st.session_state.chat_execution_count = (
                            st.session_state.get('chat_execution_count', 0) + 1
                        )

                    # Store figure if produced
                    if result.figure is not None:
                        fig_index = len(result.session.messages) - 1
                        fig_key = f"chat_fig_{fig_index}"
                        if 'chat_viz_figures' not in st.session_state:
                            st.session_state.chat_viz_figures = {}
                        st.session_state.chat_viz_figures[fig_key] = result.figure

                    # Store result table if produced (always — for download)
                    if result.result_df is not None:
                        table_index = len(result.session.messages) - 1
                        table_key = f"chat_table_{table_index}"
                        if 'chat_result_tables' not in st.session_state:
                            st.session_state.chat_result_tables = {}
                        st.session_state.chat_result_tables[table_key] = result.result_df

                    # Also store transformation result separately
                    if result.is_transformation and result.result_df is not None:
                        st.session_state.chat_transform_result = result.result_df

                except Exception as e:
                    error_text = str(e)
                    logger.error("AI Studio error: %s", error_text, exc_info=True)

        # If there was an error, save both user msg + error to session
        # (done OUTSIDE the spinner/chat_message so it definitely persists)
        if error_text:
            from app.models.chat import ChatMessage
            user_msg = ChatMessage(role="user", content=user_input)
            error_msg = ChatMessage(
                role="assistant",
                content=f"Sorry, I encountered an error: {error_text}",
            )
            current = st.session_state.chat_unified_session
            st.session_state.chat_unified_session = (
                current.add_message(user_msg).add_message(error_msg)
            )

        st.rerun()
