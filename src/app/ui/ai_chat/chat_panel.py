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
    from dataclasses import replace

    from app.workflows.ai_chat import AIChatWorkflow
    from ._data_transform_ui import render_transform_chat

    # Initialize session if needed
    if st.session_state.get('chat_transform_session') is None:
        # Run format diagnostic first
        diagnostic = AIChatWorkflow.diagnose_format(source_df)
        # Raw pipeline imports trip the Generic-shape detection (LipidMolec +
        # intensity[*]) but are still unfiltered and un-normalized, so the
        # default "no transformation needed — upload to LipidCruncher" copy
        # is misleading. Replace it with a message that points at the main
        # pipeline for the standard filter/normalize workflow while still
        # treating the data as transformable (is_supported=False) so the
        # chat input and example prompts stay enabled.
        if st.session_state.get('chat_studio_pipeline_source') == 'raw':
            # The data is Generic-shaped (LipidMolec + intensity[*]).
            # Surface those column roles so the AI has a head start,
            # but leave metadata_columns empty — for raw pipeline data
            # the extra columns (CalcMass, BaseRt, ...) aren't drop
            # candidates the way they are for arbitrary CSVs.
            intensity_cols = [
                c for c in source_df.columns
                if str(c).startswith('intensity[')
            ]
            diagnostic = replace(
                diagnostic,
                is_supported=False,
                message=(
                    "This is **raw** data imported from the main pipeline "
                    "— column names are already standardized to "
                    "`intensity[*]`, but the data hasn't been filtered "
                    "or normalized.\n\n"
                    "For the standard workflow, run it through the main "
                    "pipeline (Cleaning → Zero Filtering → Normalization) "
                    "and re-import the **Normalized** version. Or, if you "
                    "want a custom filter/normalization, ask me here and "
                    "I'll generate the code."
                ),
                lipid_name_column=(
                    'LipidMolec' if 'LipidMolec' in source_df.columns else None
                ),
                intensity_columns=intensity_cols,
                metadata_columns=[],
            )
        st.session_state.chat_transform_diagnostic = diagnostic
        session = AIChatWorkflow.start_transform_session(source_df, diagnostic)
        # The Transform tab's system prompt is fixated on "convert to
        # Generic format" — appropriate for arbitrary CSVs, but
        # misleading for raw pipeline imports where the data is already
        # Generic-shaped and the user typically wants filtering /
        # normalization. Augment the schema_context so the AI doesn't
        # frame filter/normalize requests as "preprocessing before
        # Generic conversion".
        if st.session_state.get('chat_studio_pipeline_source') == 'raw':
            extra_note = (
                "\n\n## Important context for this dataset\n"
                "This data was imported from the LipidCruncher main "
                "pipeline. It is **already in Generic shape** "
                "(LipidMolec + intensity[*] columns) — no shape "
                "transformation is needed. When the user asks for "
                "filtering, deduplication, normalization, or other "
                "content-level operations, treat those as standalone "
                "tasks. Do NOT frame your response as 'preprocessing "
                "before Generic conversion' — the data is already "
                "Generic-shaped. The result_df should still be in "
                "Generic shape (lipid name column + numeric sample "
                "columns) so the user can download and re-upload it."
            )
            session = session.model_copy(update={
                'schema_context': session.schema_context + extra_note,
            })
        st.session_state.chat_transform_session = session
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
