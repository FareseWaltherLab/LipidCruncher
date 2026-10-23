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
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.ai_studio.data_source import (
    is_blocked_standard_format,
    render_standard_format_advisory_main,
)
from app.ui.ai_studio.sidebar import display_studio_sidebar
from app.ui.ai_studio.tabs import display_tabs
from app.ui.landing_page import display_logo


# Shown above the tabs, not in an expander: what leaves the browser session
# and what the server keeps (InteractionLoggingService's turn records).
_DATA_USE_NOTICE = (
    "**How your data is used.** The AI runs on Anthropic's API. Anthropic "
    "receives your conversation with the AI, including any results the AI's "
    "code computes from your data, and, in the data tabs, your dataset's "
    "column names, types and first 3 rows (not the whole file). Under its "
    "API terms, Anthropic does not use this data to train its models and "
    "deletes it within 30 days. To improve the AI Studio, each conversation "
    "is also saved on the LipidCruncher server: your "
    "questions, the AI's answers and code, your dataset's size and column "
    "names, a description of any figure (not its values), and your 👍/👎 "
    "feedback."
)


# Method / transparency documentation for the Studio as a whole. Kept next to
# the "How to use" copy it sits beside in the UI.
_HOW_AI_WORKS_DOC = """
**What the AI is sent**

Your dataset is not uploaded to the AI. For the three data tabs, what goes to
Anthropic's API is a **schema**: the shape, every column name with its dtype and
non-null count, and the first 3 rows as a formatting sample — plus your question
and the conversation so far. The full table stays on the machine running
LipidCruncher. (Anything the generated code *prints* becomes part of the
conversation, so it is visible to the AI on later turns.)

**How an answer is produced**

Data Questions, Transform to Generic, and Custom Visualization do not answer from
the model's memory. The AI writes **Python (pandas / Plotly)**, that code is
executed here against your data, and you are shown what it returned. Every such
answer carries a **View code** expander — that code *is* the method. Read it
before you trust the number.

Generated code runs in a separate, restricted process. It is not given the app's
settings or keys, and the operating system stops it from writing files or starting
other programs and limits it to 10 seconds and 1 GB of memory. Reading files and
using the network are blocked inside Python itself rather than by the operating
system, so treat those two as safeguards, not guarantees.

**LipidCruncher Questions** works differently — no code, no data. It answers from
LipidCruncher's own paper, supplementary methods, release notes, and source code,
which it reads with tools while composing the answer.

**Limits worth knowing**

- The AI infers what your columns mean from their names and three sample rows.
  Ambiguous names get misread — check the code, and say what a column holds when
  it isn't obvious from the header.
- The same question asked twice can produce differently written code. For
  anything headed into a figure or a manuscript, download the result (and keep
  the code) so the exact analysis is on record.
- The main pipeline's methods are validated and documented; AI Studio is an
  exploration tool. When both could answer a question, prefer the pipeline.
- Each browser session is capped at **50 messages** and **20 code
  executions**, across all four tabs, and there are daily limits per visitor
  and for the whole site (they reset at midnight UTC).
"""


def display_ai_studio_page() -> None:
    """Display the LipidCruncher AI Studio page with 4 tabs."""
    # Restore any sidebar widget keys that Streamlit garbage-collected
    # while the user was on the main app page. Must run BEFORE the
    # sidebar widgets render so they see the restored values.
    StreamlitAdapter.restore_navigation_widgets()

    # All data-loading and experiment-definition controls live in the
    # sidebar, mirroring the main app's layout.
    display_studio_sidebar()
    # `studio_df_original` keeps the user's column names — used by the
    # Transform tab. `studio_df_processed` is the renamed-to-`s1..sN` view
    # that matches `chat_studio_experiment`, used by the data preview, Data
    # Questions and Custom Viz; it equals the original until the user
    # confirms an experiment.
    studio_df_original = st.session_state.get("chat_studio_df_original")
    studio_df_processed = st.session_state.get("chat_studio_df")

    _, center, _ = st.columns([1, 3, 1])

    with center:
        display_logo(centered=True)
        st.subheader("AI Studio")
        st.markdown(
            "Upload your data in the sidebar (left), then use the AI "
            "tabs below to explore, transform, or visualize it."
        )
        st.info(_DATA_USE_NOTICE)

        with st.expander("How to use the AI Studio", expanded=False):
            st.markdown(
                """
**Quick start**

1. Upload your CSV in the sidebar (or import from the main pipeline).
2. Define your experiment in the sidebar — sample columns, each
   sample's condition, any samples to exclude, and BQC label.
3. Check **Confirm the inputs by checking this box** to unlock Data
   Questions and Custom Visualization.

**Tabs**

- **LipidCruncher Questions** — Ask anything about the app's features,
  methods, or analyses. **No upload required.**
- **Data Questions** — Explore your dataset with the AI (summaries,
  filters, distributions). Requires data + confirm.
- **Transform to Generic** — Ask the AI to convert non-LipidCruncher
  data into the Generic format the main pipeline accepts. Needs only
  the upload, no experiment setup, so you can reshape a file before its
  samples are defined. Use this when your data isn't ready for the main
  pipeline yet.
- **Custom Visualization** — Generate custom Plotly charts beyond what
  the built-in modules offer. Requires data that has been **processed
  through Module 1** of the main pipeline. Run your data through
  *Start Crunching* on the home page first, then come back here and
  use **Import from main pipeline** in the sidebar.
                """
            )

        with st.expander("📖 How the AI works — and its limits", expanded=False):
            st.markdown(_HOW_AI_WORKS_DOC)

        st.markdown("---")

        # File info + data preview when data is loaded. Preview the processed
        # df (reordered + renamed to s1..sN once an experiment is applied) so
        # sample-reordering in the sidebar is reflected here. Before apply,
        # the processed df equals the original, so the preview is unchanged.
        preview_df = (
            studio_df_processed
            if studio_df_processed is not None
            else studio_df_original
        )
        blocked = is_blocked_standard_format()
        if studio_df_original is not None:
            file_id = st.session_state.get("chat_studio_file_id", "")
            pipeline_source = st.session_state.get(
                "chat_studio_pipeline_source"
            )
            if pipeline_source == 'qc_filtered':
                file_label = (
                    "QC-filtered data from main pipeline "
                    "(normalized, high-CoV lipids and PCA-removed samples "
                    "excluded)"
                )
            elif pipeline_source == 'normalized':
                file_label = "Normalized data from main pipeline"
            elif pipeline_source == 'raw':
                file_label = (
                    "Raw data from main pipeline "
                    "(unfiltered, un-normalized — column names standardized)"
                )
            else:
                file_label = (
                    file_id.rsplit("_", 1)[0] if file_id else "data"
                )
            if blocked:
                render_standard_format_advisory_main(file_label)
            else:
                st.success(
                    f"**{file_label}** — {len(studio_df_original)} rows, "
                    f"{len(studio_df_original.columns)} columns."
                )
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(preview_df.head(10), width='stretch')
                    st.caption(
                        f"{len(preview_df)} rows, "
                        f"{len(preview_df.columns)} columns"
                    )
        else:
            st.info("Upload a CSV in the sidebar to begin.")

        st.markdown("---")

        display_tabs(studio_df_original, studio_df_processed, blocked)

        # Navigation
        _display_navigation()


def _display_navigation() -> None:
    """Render navigation buttons."""
    st.markdown("---")
    has_pipeline_data = (
        st.session_state.get("raw_df") is not None
        or st.session_state.get("normalized_df") is not None
    )

    if has_pipeline_data:
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "← Back to Main Pipeline", key="ai_studio_back_main"
            ):
                StreamlitAdapter.leave_studio_for_main_pipeline()
                st.rerun()
        with col2:
            _render_back_to_home_button()
    else:
        _render_back_to_home_button()


def _render_back_to_home_button() -> None:
    """Render the 'Back to Home' button. Splits clearing logic out so the
    button can appear in either a 1-col or 2-col layout depending on
    whether 'Back to Main Pipeline' is also shown."""
    if st.button("Back to Home", key="ai_studio_back_home"):
        StreamlitAdapter.leave_studio_for_home()
        st.rerun()
