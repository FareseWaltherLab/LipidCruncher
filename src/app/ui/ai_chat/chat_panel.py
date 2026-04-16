"""
Main chat panel UI for AI-powered interactions.

Renders chat messages, handles user input, and delegates to
mode-specific UI modules for data transformation and visualization.
"""
import streamlit as st


# Rate limits per session (from Phase 5 spec)
MAX_MESSAGES = 50
MAX_EXECUTIONS = 20


def display_transform_chat(source_df) -> None:
    """Display the AI-Assisted Import chat panel.

    Renders inside an expander in the sidebar/main area. Creates a new
    transform session if one doesn't exist, then delegates to the
    transform-specific UI.

    Args:
        source_df: The user's uploaded DataFrame (pre-standardization).
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._data_transform_ui import render_transform_chat

    # Initialize session if needed
    if st.session_state.get('chat_transform_session') is None:
        st.session_state.chat_transform_session = (
            AIChatWorkflow.start_transform_session(source_df)
        )
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0

    render_transform_chat(source_df)


def display_viz_chat(analysis_df, experiment) -> None:
    """Display the AI Custom Visualization chat panel.

    Renders inside an expander in the analysis section. Creates a new
    viz session if one doesn't exist, then delegates to the
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
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0

    render_viz_chat(analysis_df, experiment)
