"""
The AI Studio's three tabs and when each is open: LipidCruncher Questions
always; Ask Your Data once the sidebar's inputs are confirmed (on the main
pipeline's raw data too: its answers and figures then say the data are
raw); Transform once there is a file.
"""
from typing import Optional

import pandas as pd
import streamlit as st

from app.ui.ai_chat import (
    display_lipidcruncher_questions_chat,
    display_data_questions_chat,
    display_transform_chat,
)


def display_tabs(
    studio_df_original: Optional[pd.DataFrame],
    studio_df_processed: Optional[pd.DataFrame],
    blocked: bool,
) -> None:
    """Render the three tabs over the Studio's data (None before an upload).
    ``blocked``: the upload is a standard format for the main pipeline."""
    # Ask Your Data is locked until the user
    # finishes the sidebar flow and checks "Confirm Inputs". `studio_applied_
    # snapshot` is set by the apply step the moment the box is first
    # checked, and persists for the rest of the session unless the
    # data is cleared. LC Questions needs no data and Transform only the
    # uploaded file.
    is_confirmed = st.session_state.get('studio_applied_snapshot') is not None
    not_confirmed_msg = (
        "Configure your experiment in the sidebar and check "
        "**Confirm the inputs by checking this box** to unlock this tab."
    )
    blocked_tab_msg = (
        "This dataset is a known standard format. See the advisory "
        "above and use the **Go to Main Pipeline** button in the "
        "sidebar."
    )

    tab_lc, tab_data, tab_transform = st.tabs([
        "LipidCruncher Questions",
        "Ask Your Data",
        "Transform to Generic",
    ])

    with tab_lc:
        display_lipidcruncher_questions_chat()

    with tab_data:
        _data_questions_tab(
            studio_df_original, studio_df_processed, blocked, is_confirmed,
            blocked_tab_msg, not_confirmed_msg,
        )

    with tab_transform:
        _transform_tab(studio_df_original, blocked, blocked_tab_msg)


def _data_questions_tab(
    studio_df_original, studio_df_processed, blocked: bool, is_confirmed: bool,
    blocked_tab_msg: str, not_confirmed_msg: str,
) -> None:
    """Ask Your Data: locked until the upload is confirmed in the sidebar."""
    if studio_df_original is None:
        st.info(
            "Upload a CSV file above to ask questions about your data "
            "or plot it."
        )
    elif blocked:
        st.info(blocked_tab_msg)
    elif not is_confirmed:
        st.info(not_confirmed_msg)
    else:
        # The renamed frame, not the original: the experiment (and
        # so `lc` and the prompt's sample lists) uses the renamed
        # s1..sN, which after a regroup are different columns.
        display_data_questions_chat(studio_df_processed)


def _transform_tab(studio_df_original, blocked: bool, blocked_tab_msg: str) -> None:
    """Transform to Generic: works on the file as uploaded; pointless for
    normalized or QC-filtered pipeline imports."""
    pipeline_source_for_transform = st.session_state.get(
        'chat_studio_pipeline_source'
    )
    if studio_df_original is None:
        st.info(
            "Upload a CSV file above to transform it into "
            "LipidCruncher Generic format."
        )
    elif blocked:
        st.info(blocked_tab_msg)
    elif pipeline_source_for_transform in ('normalized', 'qc_filtered'):
        # Normalized and QC-filtered imports are filtered AND
        # normalized — there's genuinely nothing left to transform.
        # Raw imports, by contrast, are Generic-shaped but
        # unfiltered/un-normalized, so the user may legitimately want
        # to ask the AI to filter or normalize them in custom ways.
        # Don't lock for raw.
        source_word = (
            "QC-filtered"
            if pipeline_source_for_transform == 'qc_filtered'
            else "normalized"
        )
        st.info(
            f"Transformation isn't needed — your data was imported "
            f"as **{source_word}** data from the main pipeline, so "
            "it's already filtered and normalized. Use **Ask Your "
            "Data** to explore or plot it."
        )
    else:
        # No confirm needed: Transform works on the file as
        # uploaded, which may not have sample columns yet (e.g. one
        # row per lipid and sample).
        display_transform_chat(studio_df_original)
