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


def _render_messages(session) -> None:
    """Render all chat messages with inline figures."""
    for i, msg in enumerate(session.messages):
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.code:
                with st.expander("View code"):
                    st.code(msg.code, language="python")

        # Check if there's a stored figure for this message
        fig_key = f"chat_viz_fig_{i}"
        fig = st.session_state.get('chat_viz_figures', {}).get(fig_key)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def render_viz_chat(analysis_df, experiment) -> None:
    """Render the custom visualization chat interface.

    Args:
        analysis_df: The analysis-ready DataFrame.
        experiment: Experiment configuration.
    """
    session = st.session_state.chat_viz_session

    # Show introductory message with validation result
    if len(session.messages) == 0:
        validation = st.session_state.get('chat_viz_validation')
        with st.chat_message("assistant"):
            if validation and not validation.is_valid:
                st.markdown(validation.message)
            else:
                if validation:
                    st.markdown(validation.message + "\n\n")
                st.markdown(
                    "Describe the visualization you'd like and I'll "
                    "generate it. For example:\n\n"
                    "- *Create a bar chart of species count per class*\n"
                    "- *Show a box plot of concentrations across conditions*\n"
                    "- *Make a volcano plot for KO vs WT*\n"
                    "- *Heatmap of the top 20 most variable lipids*\n"
                    "- *Scatter plot of PC vs PE intensities with "
                    "regression line*\n"
                    "- *Grouped bar chart comparing PC and PE across "
                    "conditions*"
                )

    # Render existing messages
    _render_messages(session)

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

    # Chat input
    user_input = st.chat_input(
        "Describe the visualization you want...",
        key="chat_input_viz",
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run AI visualization
        with st.chat_message("assistant"):
            with st.spinner("Generating visualization..."):
                try:
                    result = StreamlitAdapter.run_ai_viz(
                        user_message=user_input,
                        session=session,
                        analysis_df=analysis_df,
                        experiment=experiment,
                    )

                    # Update session state
                    st.session_state.chat_viz_session = result.session
                    st.session_state.chat_message_count = (
                        st.session_state.get('chat_message_count', 0) + 1
                    )

                    # Display response
                    st.markdown(result.assistant_message)

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

                    if (result.sandbox_result
                            and hasattr(result.sandbox_result, 'code')
                            and result.session.messages):
                        last_msg = result.session.messages[-1]
                        if last_msg.code:
                            with st.expander("View code"):
                                st.code(last_msg.code, language="python")

                except Exception as e:
                    st.session_state.chat_last_error = str(e)

        st.rerun()

    # Display persistent error from last attempt
    last_error = st.session_state.pop('chat_last_error', None)
    if last_error:
        st.error(f"An error occurred: {last_error}")
