"""
Main chat panel UI for AI-powered interactions.

Renders chat messages, handles user input, and delegates to
mode-specific UI modules for data transformation, visualization,
and data questions.
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
        # Run format diagnostic first
        diagnostic = AIChatWorkflow.diagnose_format(source_df)
        st.session_state.chat_transform_diagnostic = diagnostic
        st.session_state.chat_transform_session = (
            AIChatWorkflow.start_transform_session(source_df, diagnostic)
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
        # Validate Module 1 output
        st.session_state.chat_viz_validation = (
            AIChatWorkflow.validate_module1_output(analysis_df)
        )
        st.session_state.chat_viz_session = (
            AIChatWorkflow.start_viz_session(analysis_df, experiment)
        )
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0

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
        st.session_state.chat_data_questions_session = (
            AIChatWorkflow.start_data_questions_session(source_df)
        )
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0

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
        st.session_state.chat_message_count = 0

    render_lc_questions_chat()


def display_unified_chat(source_df) -> None:
    """Display the unified AI Studio chat panel.

    Single chat that handles data questions, transformations, and
    visualizations in one conversation.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    from app.workflows.ai_chat import AIChatWorkflow
    from ._unified_chat_ui import render_unified_chat

    # Initialize session if needed
    if st.session_state.get('chat_unified_session') is None:
        st.session_state.chat_unified_session = (
            AIChatWorkflow.start_unified_session(source_df)
        )
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0
        st.session_state.chat_viz_figures = {}
        st.session_state.chat_result_tables = {}

    render_unified_chat(source_df)
