"""
Data transformation mode UI for the AI chat.

Renders the chat conversation for transforming arbitrary data into
the LipidCruncher Generic format. When a valid transformation is
produced, offers a download button so the user can save the result
and upload it to the main pipeline.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from ._turn_errors import render_failure
from .chat_panel import (
    next_user_input,
    render_example_prompts,
    render_messages,
    run_code_turn,
    start_turn_refused,
    usage_refused,
)

# Example prompts for this tab
_EXAMPLE_PROMPTS = [
    "Transform my data to Generic format",
    "What columns need to be removed?",
    "Filter out low-quality rows, then transform",
]


_ABOUT_TAB = """
**What this tab does** — converts a dataset LipidCruncher can't read directly
into the **Generic** format the main pipeline accepts. The AI writes the
conversion code, runs it here on your file, and hands back the result.

**The Generic format** is deliberately plain: column 1 = lipid names (any
naming convention — LipidCruncher standardizes them to LIPID MAPS notation on
upload), columns 2+ = one numeric intensity column per sample, one row per
species, and **no other columns** (mass, retention time, scores and the like are
dropped).

**The round trip**

1. Describe your file, or just ask for the transformation.
2. Check **Preview transformed data**, and **View code** for what was actually
   done to your numbers.
3. Download the CSV.
4. Upload it on the main pipeline as a **Generic** dataset and run the normal
   Module 1 → 2 → 3 analysis.

**Also useful before converting** — asking to *see* something ("list the internal
standards you can find", "show the distinct values in column X") prints an
answer instead of producing a file.

**Check the preview before you download.** The AI decides which columns are
samples and which are metadata from their names; if it guessed wrong, say which
columns are which and ask it to redo the transform.
"""


def _store_result(result) -> None:
    """Keep a turn's session and, for a successful transform, its table
    for download. Info-gathering turns (no result_df) keep the previous
    preview; a failed transform clears it."""
    t = result.transformation
    st.session_state.chat_transform_session = result.session
    st.session_state.chat_transform_refused_metadata = (
        t.metadata_columns if t else ()
    )
    if t and t.success and not t.is_info_only:
        st.session_state.chat_transform_result = t.result_df
        st.session_state.chat_transform_report = t.report
    elif t and not t.success:
        st.session_state.chat_transform_result = None
        st.session_state.chat_transform_report = None


def render_transform_chat(source_df) -> None:
    """Render the data transformation chat interface.

    Args:
        source_df: The user's uploaded DataFrame.
    """
    session = st.session_state.chat_transform_session

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(_ABOUT_TAB)

    # Show introductory message with format diagnostic
    if len(session.messages) == 0:
        diagnostic = st.session_state.get('chat_transform_diagnostic')
        with st.chat_message("assistant"):
            if diagnostic:
                st.markdown(diagnostic.message)
            else:
                st.markdown(
                    "I can help you transform your data into LipidCruncher's "
                    "Generic format. Tell me about your data or ask me to "
                    "transform it."
                )

    render_messages(session)

    # Show the last transformation result if successful (renders before
    # the example tiles so the full AI response — message, code,
    # preview — sits together; tiles come after as next-step prompts).
    transform_result_df = st.session_state.get('chat_transform_result')
    if transform_result_df is not None:
        with st.expander("Preview transformed data", expanded=True):
            st.dataframe(transform_result_df.head(10), width='stretch')
            st.caption(
                f"{len(transform_result_df)} rows, "
                f"{len(transform_result_df.columns)} columns"
            )
            report = st.session_state.get('chat_transform_report')
            if report is not None:
                st.markdown(f"**Compared with your upload:** {report.summary()}.")
                if report.lipids_removed:
                    st.download_button(
                        label=f"Download the {len(report.lipids_removed):,} "
                              f"removed lipid names",
                        data="\n".join(report.lipids_removed).encode("utf-8"),
                        file_name="removed_lipids.txt",
                        mime="text/plain",
                        key="chat_download_removed_lipids",
                    )

            csv_data = transform_result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download transformed CSV",
                data=csv_data,
                file_name="transformed_data.csv",
                mime="text/csv",
                key="chat_download_transform",
            )
            st.caption(
                "Download this file, then upload it to LipidCruncher as a "
                "**Generic** format dataset to run it through the analysis pipeline."
            )

    # Example prompts (only for formats that need transforming)
    diagnostic = st.session_state.get('chat_transform_diagnostic')
    if not diagnostic or not diagnostic.is_supported:
        render_example_prompts(
            _EXAMPLE_PROMPTS, "tf", "chat_pending_prompt_transform",
        )
    if usage_refused():
        return

    # A transform refused only because sample names look like metadata
    # (e.g. 'Reference_1'): the user, not the AI, may say they are samples.
    refused = st.session_state.get('chat_transform_refused_metadata') or ()
    if refused:
        names = ", ".join(f"'{c}'" for c in refused)
        st.info(
            f"{names} {'was' if len(refused) == 1 else 'were'} refused because "
            f"the name looks like metadata (retention time, score, m/z, ...). "
            f"If {'it is a sample' if len(refused) == 1 else 'they are samples'}, "
            f"you can keep {'it' if len(refused) == 1 else 'them'}."
        )
        if st.button(f"Keep {names} as samples", key="chat_keep_metadata_columns"):
            StreamlitAdapter.record_ai_execution()
            _store_result(StreamlitAdapter.keep_ai_transform_columns(
                session, source_df,
            ))
            st.rerun()

    user_input = next_user_input(
        "chat_pending_prompt_transform",
        "Ask about your data or request a transformation...",
        "chat_input_transform",
    )
    if user_input:
        if start_turn_refused():
            return

        def on_result(result, streamed):
            t = result.transformation
            if not streamed.strip() and t is None:
                st.info(
                    "I couldn't produce a result for that. "
                    "Please try rephrasing your request."
                )
            if t and t.code:
                StreamlitAdapter.record_ai_execution()
            _store_result(result)
            return {"code": (t.code if t else None) or None,
                    "produced_table": bool(
                        t and t.success and not t.is_info_only)}

        run_code_turn(
            "transform", "transform", user_input, session.messages,
            lambda: StreamlitAdapter.stream_ai_transform(
                user_message=user_input,
                session=session,
                source_df=source_df,
            ),
            on_result,
            df=source_df,
        )

    # Display persistent error from last attempt, with a one-click retry.
    render_failure('transform', 'chat_pending_prompt_transform')
