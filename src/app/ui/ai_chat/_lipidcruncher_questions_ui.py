"""
LipidCruncher Questions mode UI for the AI chat.

Renders the chat conversation for asking questions about the
LipidCruncher application itself: features, methods, analysis
options, supported formats, terminology, etc.

Purely conversational — no code execution, no data access.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter

# Rate limits (mirrored from chat_panel to avoid circular import)
MAX_MESSAGES = 50


def _render_messages(session) -> None:
    """Render all chat messages in the session."""
    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)


def render_lc_questions_chat() -> None:
    """Render the LipidCruncher questions chat interface."""
    session = st.session_state.chat_lc_questions_session

    # Show introductory message if no messages yet
    if len(session.messages) == 0:
        st.info(
            "Ask me anything about LipidCruncher — features, methods, "
            "analysis options, and more. For example:\n\n"
            "*'What normalization methods are available?'*\n\n"
            "*'How does LipidCruncher handle missing values?'*\n\n"
            "*'What statistical tests can I use for 3 conditions?'*"
        )

    # Render existing messages
    _render_messages(session)

    # Rate limit check
    msg_count = st.session_state.get('chat_message_count', 0)
    if msg_count >= MAX_MESSAGES:
        st.warning("Message limit reached. Please start a new session.")
        if st.button("Reset Chat", key="chat_reset_lc_questions"):
            st.session_state.chat_lc_questions_session = None
            st.session_state.chat_message_count = 0
            st.rerun()
        return

    # Chat input
    user_input = st.chat_input(
        "Ask about LipidCruncher features, methods, or analysis options...",
        key="chat_input_lc_questions",
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI
        error_text = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = StreamlitAdapter.run_ai_lc_questions(
                        user_message=user_input,
                        session=session,
                    )

                    # Update session state
                    st.session_state.chat_lc_questions_session = result.session
                    st.session_state.chat_message_count = (
                        st.session_state.get('chat_message_count', 0) + 1
                    )

                    # Display response
                    st.markdown(result.assistant_message)

                except Exception as e:
                    error_text = str(e)

        # If there was an error, save both msgs to session
        if error_text:
            from app.models.chat import ChatMessage
            user_msg = ChatMessage(role="user", content=user_input)
            error_msg = ChatMessage(
                role="assistant",
                content=f"Sorry, I encountered an error: {error_text}",
            )
            current = st.session_state.chat_lc_questions_session
            st.session_state.chat_lc_questions_session = (
                current.add_message(user_msg).add_message(error_msg)
            )

        st.rerun()
