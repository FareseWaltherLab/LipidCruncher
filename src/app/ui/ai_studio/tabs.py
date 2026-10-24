"""
The AI Studio's four tabs and when each is open: LipidCruncher Questions
always; Transform once there is a file; Data Questions and Custom
Visualization once the sidebar's inputs are confirmed (Custom Visualization
also needs Module 1's columns and is closed to raw pipeline data).
"""
from typing import Optional

import pandas as pd
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.ai_chat import (
    display_lipidcruncher_questions_chat,
    display_data_questions_chat,
    display_transform_chat,
    display_viz_chat,
)


def display_tabs(
    studio_df_original: Optional[pd.DataFrame],
    studio_df_processed: Optional[pd.DataFrame],
    blocked: bool,
) -> None:
    """Render the four tabs over the Studio's data (None before an upload).
    ``blocked``: the upload is a standard format for the main pipeline."""
    # Data Questions and Custom Visualization are locked until the user
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
        _data_questions_tab(
            studio_df_original, studio_df_processed, blocked, is_confirmed,
            blocked_tab_msg, not_confirmed_msg,
        )

    with tab_transform:
        _transform_tab(studio_df_original, blocked, blocked_tab_msg)

    with tab_viz:
        _viz_tab(
            studio_df_processed, blocked, is_confirmed,
            blocked_tab_msg, not_confirmed_msg,
        )


def _data_questions_tab(
    studio_df_original, studio_df_processed, blocked: bool, is_confirmed: bool,
    blocked_tab_msg: str, not_confirmed_msg: str,
) -> None:
    """Data Questions: locked until the upload is confirmed in the sidebar."""
    if studio_df_original is None:
        st.info(
            "Upload a CSV file above to ask questions about your data."
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
            "it's already filtered and normalized. Use **Data "
            "Questions** to explore it, or **Custom "
            "Visualization** to plot it."
        )
    else:
        # No confirm needed: Transform works on the file as
        # uploaded, which may not have sample columns yet (e.g. one
        # row per lipid and sample).
        display_transform_chat(studio_df_original)


def _viz_tab(
    studio_df_processed, blocked: bool, is_confirmed: bool,
    blocked_tab_msg: str, not_confirmed_msg: str,
) -> None:
    """Custom Visualization: needs confirmed, Module 1-processed data."""
    pipeline_source = st.session_state.get(
        'chat_studio_pipeline_source'
    )
    if studio_df_processed is None:
        st.info(
            "Upload a CSV file above to create custom visualizations."
        )
    elif blocked:
        st.info(blocked_tab_msg)
    elif pipeline_source == 'raw':
        # Raw pipeline imports haven't been through cleaning, zero
        # filtering, or normalization, so plotting them would be
        # misleading even when the columns happen to satisfy the
        # validation check.
        st.warning(
            "You imported **raw** data from the main pipeline. "
            "Custom Visualization requires data that has been "
            "processed through Module 1 (cleaning, zero filtering, "
            "normalization). Re-import **Normalized data** "
            "instead, or run your data through *Start Crunching* "
            "first."
        )
    elif not is_confirmed:
        st.info(not_confirmed_msg)
    else:
        # Validate Module 1 output before showing viz chat
        validation = StreamlitAdapter.studio_viz_validation(
            studio_df_processed
        )

        if not validation.is_valid:
            missing = []
            if not validation.has_lipid_molec:
                missing.append("`LipidMolec`")
            if not validation.has_class_key:
                missing.append("`ClassKey`")
            if (not validation.intensity_columns
                    and not validation.concentration_columns):
                missing.append(
                    "`intensity[sN]` / `concentration[sN]`"
                )
            st.warning(
                "Custom Visualization needs data that has already "
                "been filtered and normalized by LipidCruncher's "
                "Module 1 pipeline. Missing columns: "
                + ", ".join(missing) + ".\n\n"
                "To get there:\n"
                "1. Use the **Transform to Generic** tab to convert "
                "your file into LipidCruncher's generic format.\n"
                "2. Take that generic file to the main pipeline on "
                "the home page and run **Start Crunching** to filter "
                "and normalize it.\n"
                "3. Come back here with the filtered/normalized data "
                "to build custom visualizations."
            )
        else:
            experiment = StreamlitAdapter.studio_viz_experiment(
                studio_df_processed
            )
            display_viz_chat(studio_df_processed, experiment)
