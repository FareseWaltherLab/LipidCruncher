"""
LipidCruncher AI Studio — standalone page for AI-powered data
exploration, transformation, and custom visualization.

Users can upload any CSV here (independent of the main pipeline)
or import data already loaded in the pipeline. Four tabs provide
focused AI assistance:

1. LipidCruncher Questions — app features, methods, analysis options
2. Data Questions — explore and summarize the uploaded dataset
3. Transform to Generic — convert data to LipidCruncher Generic format
4. Custom Visualization — create plots from processed data
"""
import logging
from typing import Optional

import pandas as pd
import streamlit as st

from app.constants import PAGE_LANDING
from app.models.experiment import ExperimentConfig
from app.ui.landing_page import display_logo
from app.ui.ai_chat import (
    display_lipidcruncher_questions_chat,
    display_data_questions_chat,
    display_transform_chat,
    display_viz_chat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _derive_experiment(df: pd.DataFrame) -> ExperimentConfig:
    """Create a minimal single-condition ExperimentConfig from numeric columns.

    This allows the visualization chat to work without the user manually
    defining experimental conditions.
    """
    intensity_cols = [c for c in df.columns if c.startswith("intensity[")]
    if not intensity_cols:
        intensity_cols = df.select_dtypes(include="number").columns.tolist()

    n_samples = max(len(intensity_cols), 1)
    return ExperimentConfig(
        conditions_list=["All Samples"],
        number_of_samples_list=[n_samples],
    )


def _handle_file_upload() -> None:
    """Handle the file upload widget. Only processes the file on first
    upload or when the file changes (tracked by name + size)."""
    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="ai_studio_file_uploader",
    )

    if uploaded is None:
        # File was removed — clear studio data
        if st.session_state.get("chat_studio_file_id") is not None:
            _clear_data_sessions()
        return

    # Track file identity to avoid re-processing on every rerun
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("chat_studio_file_id") == file_id:
        # Same file — nothing to do
        return

    # New file — parse and store
    try:
        df = pd.read_csv(uploaded)
        if df.empty:
            st.error("The uploaded file is empty.")
            return
        # Clear old sessions first, then set new data
        _clear_data_sessions()
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_file_id = file_id
    except Exception as e:
        st.error(f"Could not read CSV: {e}")


def _clear_data_sessions() -> None:
    """Reset all data-dependent chat sessions when data changes."""
    st.session_state.chat_studio_df = None
    st.session_state.chat_studio_file_id = None
    st.session_state.chat_studio_experiment = None
    # Per-tab sessions
    st.session_state.chat_data_questions_session = None
    st.session_state.chat_transform_session = None
    st.session_state.chat_transform_diagnostic = None
    st.session_state.chat_viz_session = None
    st.session_state.chat_questions_result_df = None
    st.session_state.chat_transform_result = None
    st.session_state.chat_viz_figures = {}
    st.session_state.chat_result_tables = {}
    # Legacy unified session
    st.session_state.chat_unified_session = None
    # Counters
    st.session_state.chat_message_count = 0
    st.session_state.chat_execution_count = 0


def _offer_pipeline_import() -> Optional[pd.DataFrame]:
    """If the pipeline has data in session state, offer to import it."""
    normalized_df = st.session_state.get("normalized_df")
    raw_df = st.session_state.get("raw_df")

    if normalized_df is None and raw_df is None:
        return None

    st.markdown("**Or use data from the pipeline:**")
    options = []
    if normalized_df is not None:
        options.append("Normalized data (from pipeline)")
    if raw_df is not None:
        options.append("Raw uploaded data (from pipeline)")

    choice = st.radio(
        "Select pipeline data",
        options,
        key="ai_studio_pipeline_choice",
        label_visibility="collapsed",
    )
    if st.button("Import Pipeline Data", key="ai_studio_import_pipeline"):
        if "Normalized" in choice:
            return normalized_df
        return raw_df

    return None


# =============================================================================
# Main Page
# =============================================================================

def display_ai_studio_page() -> None:
    """Display the LipidCruncher AI Studio page with 4 tabs."""
    _, center, _ = st.columns([1, 3, 1])

    with center:
        display_logo(centered=True)
        st.subheader("AI Studio")
        st.markdown(
            "Upload any CSV and use AI to explore your data, transform it, "
            "or generate custom visualizations."
        )

        st.markdown("---")

        # --- Data Loading ---
        _handle_file_upload()
        studio_df = st.session_state.get("chat_studio_df")

        # Show file info if loaded
        if studio_df is not None:
            file_id = st.session_state.get("chat_studio_file_id", "")
            file_name = file_id.rsplit("_", 1)[0] if file_id else "data"
            st.success(
                f"**{file_name}** — {len(studio_df)} rows, "
                f"{len(studio_df.columns)} columns."
            )

        # Pipeline import (only if no file loaded)
        if studio_df is None:
            pipeline_df = _offer_pipeline_import()
            if pipeline_df is not None:
                st.session_state.chat_studio_df = pipeline_df
                st.session_state.chat_studio_file_id = "_pipeline_import"
                # Reset data-dependent sessions only
                st.session_state.chat_data_questions_session = None
                st.session_state.chat_transform_session = None
                st.session_state.chat_viz_session = None
                st.session_state.chat_transform_result = None
                st.session_state.chat_viz_figures = {}
                st.session_state.chat_result_tables = {}
                st.session_state.chat_message_count = 0
                st.session_state.chat_execution_count = 0
                st.rerun()

        # Show data preview if loaded
        if studio_df is not None:
            with st.expander("Data Preview", expanded=False):
                st.dataframe(studio_df.head(10), use_container_width=True)
                st.caption(
                    f"{len(studio_df)} rows, {len(studio_df.columns)} columns"
                )

        st.markdown("---")

        # --- 4-Tab Structure ---
        tab_lc, tab_data, tab_transform, tab_viz = st.tabs([
            "LipidCruncher Questions",
            "Data Questions",
            "Transform to Generic",
            "Custom Visualization",
        ])

        with tab_lc:
            display_lipidcruncher_questions_chat()

        with tab_data:
            if studio_df is None:
                st.info(
                    "Upload a CSV file above to ask questions about your data."
                )
            else:
                display_data_questions_chat(studio_df)

        with tab_transform:
            if studio_df is None:
                st.info(
                    "Upload a CSV file above to transform it into "
                    "LipidCruncher Generic format."
                )
            else:
                display_transform_chat(studio_df)

        with tab_viz:
            if studio_df is None:
                st.info(
                    "Upload a CSV file above to create custom visualizations. "
                    "For best results, use data that has been processed through "
                    "LipidCruncher Module 1."
                )
            else:
                experiment = st.session_state.get("chat_studio_experiment")
                if experiment is None:
                    experiment = _derive_experiment(studio_df)
                    st.session_state.chat_studio_experiment = experiment
                display_viz_chat(studio_df, experiment)

        # Navigation
        _display_navigation()


def _display_navigation() -> None:
    """Render navigation buttons."""
    st.markdown("---")
    if st.button("Back to Home", key="ai_studio_back_home"):
        # Clear all studio-specific state
        _clear_data_sessions()
        st.session_state.chat_lc_questions_session = None
        st.session_state.page = PAGE_LANDING
        st.rerun()
