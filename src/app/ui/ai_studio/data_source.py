"""
The AI Studio sidebar's data source: the file uploader, the import from the
main pipeline, and the advisory shown instead of the setup when an upload is
a standard format the main pipeline handles.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.ai_studio import AIStudioWorkflow


def is_blocked_standard_format() -> bool:
    """True when the uploaded file is a known standard format that should
    go through the main pipeline first. Pipeline imports are exempt: the
    user has already explicitly chosen to bring that data over."""
    return AIStudioWorkflow.is_blocked_standard_format(
        st.session_state.get('chat_studio_detected_format'),
        st.session_state.get('chat_studio_pipeline_source'),
    )


def handle_file_upload_sidebar() -> None:
    """Render the file uploader in the sidebar. Only processes the file on
    first upload or when the file changes (tracked by name + size)."""
    uploaded = st.sidebar.file_uploader(
        "Upload your dataset",
        type=["csv"],
        key="ai_studio_file_uploader",
    )

    if uploaded is None:
        # Only treat an empty uploader as "user removed their file" when the
        # file_id came from a real upload. A pipeline import sets file_id to
        # "_pipeline_import" but never populates the uploader, so without
        # this exclusion every rerun would wipe the imported data.
        file_id = st.session_state.get("chat_studio_file_id")
        if file_id is not None and not file_id.startswith("_pipeline_import"):
            # The uploader loses its file while the page is unmounted, so
            # after a visit to the main pipeline the stored upload stands in.
            if st.session_state.get("chat_studio_upload_kept"):
                st.sidebar.info(
                    "📁 Using the file you uploaded before going to the main "
                    "pipeline. Upload a file to replace it."
                )
                return
            StreamlitAdapter.clear_studio_data()
        return

    StreamlitAdapter.set_studio_upload_kept(False)
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("chat_studio_file_id") == file_id:
        return

    try:
        df, detected = AIStudioWorkflow.read_upload(uploaded.getvalue())
        if df.empty:
            st.sidebar.error("The uploaded file is empty.")
            return
        StreamlitAdapter.store_studio_upload(df, file_id, detected)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")


def offer_pipeline_import_sidebar() -> None:
    """Pipeline-import controls, rendered in the sidebar.

    Shown when the main pipeline has data available AND the currently-
    loaded Studio data is either empty or itself a previous pipeline
    import. Keeping the controls visible after a pipeline import lets
    users switch sources (raw ↔ normalized ↔ QC-filtered) or pull a fresh
    copy after changing the main pipeline (e.g. swapping the normalization
    method) — without that, Studio state stays frozen at the first import.

    Skipped when the user has uploaded their own CSV — they picked a
    different data source on purpose, so don't clutter their sidebar.

    Carries over the pipeline's experiment definition as well, so the
    Studio doesn't fall back to a single dummy condition.
    """
    frames = StreamlitAdapter.studio_pipeline_frames()
    qc_df = frames['qc_filtered']
    normalized_df = frames['normalized']
    raw_for_studio = frames['raw']
    if normalized_df is None and raw_for_studio is None and qc_df is None:
        return

    studio_has_data = st.session_state.get('chat_studio_df') is not None
    current_pipeline_source = st.session_state.get('chat_studio_pipeline_source')
    if studio_has_data and current_pipeline_source is None:
        # User uploaded their own CSV — they aren't using pipeline data,
        # so skip the import block entirely.
        return

    if studio_has_data:
        if current_pipeline_source == 'qc_filtered':
            source_label = "QC-filtered"
        elif current_pipeline_source == 'normalized':
            source_label = "Normalized"
        else:
            source_label = "Raw"
        st.sidebar.markdown(
            f"**Currently imported: {source_label} data from main pipeline.**"
        )
        st.sidebar.caption(
            "Re-import to switch sources or pull fresh data after "
            "changing the main pipeline. This resets the AI Studio chat."
        )
    else:
        st.sidebar.markdown("**Or import from the main pipeline:**")

    # Most-processed first, so the default selection is the furthest along
    # the pipeline the user has actually got.
    options = []
    if qc_df is not None:
        options.append("QC-filtered data (from pipeline)")
    if normalized_df is not None:
        options.append("Normalized data (from pipeline)")
    if raw_for_studio is not None:
        options.append("Raw uploaded data (from pipeline)")

    choice = st.sidebar.radio(
        "Select pipeline data",
        options,
        key="ai_studio_pipeline_choice",
        label_visibility="collapsed",
    )
    button_label = (
        "Re-import Pipeline Data" if studio_has_data
        else "Import Pipeline Data"
    )
    if st.sidebar.button(button_label, key="ai_studio_import_pipeline"):
        if choice.startswith("QC-filtered"):
            source_key = 'qc_filtered'
        elif choice.startswith("Normalized"):
            source_key = 'normalized'
        else:
            source_key = 'raw'
        StreamlitAdapter.import_studio_pipeline_data(source_key)
        st.rerun()


def render_standard_format_advisory_sidebar() -> None:
    """Render the redirect advisory in the sidebar when a standard format
    is detected on direct upload. Replaces the sample-columns / experiment
    flow so the only forward paths are 'Go to Main Pipeline' or removing
    the file from the uploader."""
    detected = st.session_state.get('chat_studio_detected_format', 'standard')
    st.sidebar.warning(
        f"**{detected}** dataset detected.\n\n"
        "This format is fully supported by the main LipidCruncher "
        "pipeline. Run it through the main pipeline first (cleaning, "
        "zero filtering, normalization), then come back here and use "
        "**Import from main pipeline** to bring the processed data "
        "into AI Studio."
    )
    if st.sidebar.button(
        "Go to Main Pipeline", key="ai_studio_goto_main_from_advisory"
    ):
        StreamlitAdapter.send_studio_upload_to_main_pipeline()
        st.rerun()


def render_standard_format_advisory_main(file_name: str) -> None:
    """Render the redirect advisory in the main content area when a
    standard format is detected. Replaces the data preview + tab section
    so the user sees a clear, single message instead of a row of locked
    tabs."""
    detected = st.session_state.get('chat_studio_detected_format', 'standard')
    st.warning(
        f"**{file_name}** looks like a **{detected}** dataset.\n\n"
        "AI Studio is designed for *non-standard* CSVs and for exploring "
        "data that has already been processed by Module 1. Standard "
        "formats like this one go through the main pipeline first so "
        "they're properly cleaned, zero-filtered, and normalized before "
        "any AI-driven analysis.\n\n"
        "**Next step:** click **Go to Main Pipeline** in the sidebar (or "
        "remove the file from the uploader and try a different dataset)."
    )
