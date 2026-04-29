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


def _detect_sample_prefix(df: pd.DataFrame) -> Optional[str]:
    """Return 'intensity' or 'concentration' depending on which sample-column
    prefix the DataFrame uses. Module 1 emits one or the other depending on
    the normalization method (raw / total intensity → intensity[*]; internal
    standard, protein, or both → concentration[*]).

    Returns None if neither prefix is present.
    """
    if any(c.startswith('intensity[') and c.endswith(']') for c in df.columns):
        return 'intensity'
    if any(c.startswith('concentration[') and c.endswith(']') for c in df.columns):
        return 'concentration'
    return None


def _sample_names_for_prefix(df: pd.DataFrame, prefix: str) -> list:
    """Return ['s1','s2',...] for the {prefix}[*] columns of df, in order."""
    head = f'{prefix}['
    return [
        c[len(head):-1]
        for c in df.columns
        if c.startswith(head) and c.endswith(']')
    ]


def _apply_user_experiment(
    df: pd.DataFrame,
    conditions_list: list,
    samples_per_condition: list,
    prefix: str,
) -> tuple:
    """Build an ExperimentConfig and a reordered DataFrame.

    The user picks samples per condition in arbitrary order (e.g. condition A
    gets s1+s3, condition B gets s2+s4). ExperimentConfig assumes conditions
    occupy contiguous sample blocks (s1..sN), so we rename the sample columns
    so condition A's samples become s1..sX and condition B's become
    s(X+1)..s(X+Y). The same prefix ('intensity' or 'concentration') is
    preserved on output. Non-sample columns are left untouched.

    Args:
        df: DataFrame with {prefix}[*] sample columns.
        conditions_list: condition names, in declared order.
        samples_per_condition: list of lists; samples_per_condition[i] is the
            list of original sample names assigned to conditions_list[i].
        prefix: 'intensity' or 'concentration'.

    Returns:
        (reordered_df, experiment).
    """
    head = f'{prefix}['
    rename_map = {}
    sample_cols_in_order = []
    new_idx = 1
    for samples in samples_per_condition:
        for old_sample in samples:
            old_col = f'{head}{old_sample}]'
            new_col = f'{head}s{new_idx}]'
            rename_map[old_col] = new_col
            sample_cols_in_order.append(old_col)
            new_idx += 1

    non_sample = [c for c in df.columns if not c.startswith(head)]
    reordered = df[non_sample + sample_cols_in_order].rename(columns=rename_map)

    experiment = ExperimentConfig(
        conditions_list=conditions_list,
        number_of_samples_list=[len(s) for s in samples_per_condition],
    )
    return reordered, experiment


def _handle_file_upload_sidebar() -> None:
    """Render the file uploader in the sidebar. Only processes the file on
    first upload or when the file changes (tracked by name + size)."""
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        key="ai_studio_file_uploader",
    )

    if uploaded is None:
        if st.session_state.get("chat_studio_file_id") is not None:
            _clear_data_sessions()
        return

    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("chat_studio_file_id") == file_id:
        return

    try:
        df = pd.read_csv(uploaded)
        if df.empty:
            st.sidebar.error("The uploaded file is empty.")
            return
        _clear_data_sessions()
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_file_id = file_id
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")


def _offer_pipeline_import_sidebar() -> None:
    """Pipeline-import controls, rendered in the sidebar. Only shows when
    no Studio data is loaded yet AND the main pipeline has data ready.
    Carries over the pipeline's experiment definition as well, so the
    Studio doesn't fall back to a single dummy condition."""
    if st.session_state.get('chat_studio_df') is not None:
        return

    normalized_df = st.session_state.get("normalized_df")
    raw_df = st.session_state.get("raw_df")
    if normalized_df is None and raw_df is None:
        return

    st.sidebar.markdown("**Or import from the main pipeline:**")
    options = []
    if normalized_df is not None:
        options.append("Normalized data (from pipeline)")
    if raw_df is not None:
        options.append("Raw uploaded data (from pipeline)")

    choice = st.sidebar.radio(
        "Select pipeline data",
        options,
        key="ai_studio_pipeline_choice",
        label_visibility="collapsed",
    )
    if st.sidebar.button("Import Pipeline Data", key="ai_studio_import_pipeline"):
        if "Normalized" in choice:
            st.session_state.chat_studio_df = normalized_df
        else:
            st.session_state.chat_studio_df = raw_df
        st.session_state.chat_studio_file_id = "_pipeline_import"
        # Carry over the pipeline's experiment definition.
        pipeline_experiment = st.session_state.get('experiment')
        if pipeline_experiment is not None:
            st.session_state.chat_studio_experiment = pipeline_experiment
        # Reset per-tab sessions since the underlying data changed.
        st.session_state.chat_data_questions_session = None
        st.session_state.chat_transform_session = None
        st.session_state.chat_viz_session = None
        st.session_state.chat_transform_result = None
        st.session_state.chat_viz_figures = {}
        st.session_state.chat_result_tables = {}
        st.session_state.chat_message_count = 0
        st.session_state.chat_execution_count = 0
        st.rerun()


def _display_define_experiment_sidebar(sample_names: list) -> tuple:
    """Render the 'Define Experiment' sidebar block, mirroring the main
    app's pattern: number of conditions + per-condition (name, count).
    Validates that the per-condition counts sum to the detected sample
    count.

    Returns:
        (condition_names, sample_counts) when valid, or (None, None).
    """
    st.sidebar.subheader("Define Experiment")
    st.sidebar.caption(
        f"Detected **{len(sample_names)}** sample(s) in the data."
    )

    n_conditions = st.sidebar.number_input(
        "Number of conditions",
        min_value=1, max_value=20,
        value=int(st.session_state.get('studio_n_conditions', 2)),
        step=1, key='studio_n_conditions',
    )

    condition_names = []
    sample_counts = []
    for i in range(int(n_conditions)):
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            name_key = f'studio_cond_name_{i}'
            name = st.text_input(
                f"Condition {i + 1}",
                value=st.session_state.get(name_key, f"Condition_{i + 1}"),
                key=name_key,
            )
        with col2:
            count_key = f'studio_n_samples_{i}'
            count = st.number_input(
                "Samples",
                min_value=1, max_value=len(sample_names),
                value=int(
                    st.session_state.get(count_key, min(3, len(sample_names)))
                ),
                step=1, key=count_key,
            )
        condition_names.append(name)
        sample_counts.append(int(count))

    if not all((n or '').strip() for n in condition_names):
        st.sidebar.error("All condition labels must be non-empty.")
        return None, None
    if len(set(condition_names)) != len(condition_names):
        st.sidebar.error("Condition names must be unique.")
        return None, None

    total = sum(sample_counts)
    if total != len(sample_names):
        st.sidebar.warning(
            f"Assigned {total} samples but dataset has {len(sample_names)}. "
            "Please adjust the sample counts so they sum to "
            f"{len(sample_names)}."
        )
        return None, None

    return condition_names, sample_counts


def _display_group_samples_sidebar(
    sample_names: list, condition_names: list, sample_counts: list,
) -> Optional[list]:
    """Render the 'Group Samples' sidebar block. Auto-assigns the first N1
    samples to condition A, next N2 to B, etc., shows the resulting table,
    asks the user to confirm. If the user picks 'No', shows multi-select
    widgets so they can manually reassign.

    Returns:
        samples_per_condition (list[list[str]]) when ready, or None if the
        user is mid-edit / hasn't completed manual assignment.
    """
    st.sidebar.subheader("Group Samples")

    # Auto-assign assuming samples are ordered by condition.
    auto_assignment = []
    idx = 0
    for count in sample_counts:
        auto_assignment.append(sample_names[idx:idx + count])
        idx += count

    group_df = pd.DataFrame({
        'sample': sample_names,
        'condition': [
            cond
            for cond, samples in zip(condition_names, auto_assignment)
            for _ in samples
        ],
    })
    st.sidebar.dataframe(group_df, use_container_width=True, hide_index=True)

    grouping_correct = st.sidebar.radio(
        "Are your samples properly grouped together?",
        ["Yes", "No"],
        key='studio_grouping_radio',
    )

    if grouping_correct == "Yes":
        return auto_assignment

    # Manual reassignment via multiselect, mirroring the main app.
    st.sidebar.write(
        "Pick the samples that belong to each condition. Each condition "
        f"needs exactly the count you set above."
    )
    manual = []
    for i, (cond, count) in enumerate(zip(condition_names, sample_counts)):
        selected = st.sidebar.multiselect(
            f"Pick {count} sample(s) for **{cond}**",
            options=sample_names,
            key=f'studio_select_{i}',
        )
        manual.append(selected)

    flat = [s for group in manual for s in group]
    duplicates = {s for s in flat if flat.count(s) > 1}
    if duplicates:
        st.sidebar.error(
            f"Sample(s) in more than one condition: "
            f"{', '.join(sorted(duplicates))}"
        )
        return None
    if len(flat) != len(sample_names):
        return None  # mid-edit; don't show errors yet
    if any(len(g) != c for g, c in zip(manual, sample_counts)):
        return None
    return manual


def _display_studio_sidebar() -> None:
    """Render the full AI Studio sidebar: upload + pipeline import +
    experiment definition + sample grouping + apply."""
    st.sidebar.subheader("Data")
    _handle_file_upload_sidebar()
    _offer_pipeline_import_sidebar()

    studio_df = st.session_state.get('chat_studio_df')
    if studio_df is None:
        return

    st.sidebar.markdown("---")

    prefix = _detect_sample_prefix(studio_df)
    if prefix is None:
        st.sidebar.info(
            "No `intensity[*]` or `concentration[*]` columns detected — "
            "Define Experiment is unavailable for this data shape. Use "
            "the Transform to Generic tab to convert it first."
        )
        return

    sample_names = _sample_names_for_prefix(studio_df, prefix)
    if not sample_names:
        return

    condition_names, sample_counts = _display_define_experiment_sidebar(
        sample_names,
    )
    if condition_names is None:
        return

    samples_per_condition = _display_group_samples_sidebar(
        sample_names, condition_names, sample_counts,
    )
    if samples_per_condition is None:
        return

    st.sidebar.markdown("---")
    if st.sidebar.button("Apply", key='studio_apply_experiment'):
        try:
            reordered_df, experiment = _apply_user_experiment(
                studio_df, condition_names, samples_per_condition, prefix,
            )
        except (ValueError, TypeError) as e:
            st.sidebar.error(str(e))
            return

        st.session_state.chat_studio_df = reordered_df
        st.session_state.chat_studio_experiment = experiment
        # Per-tab chat sessions reference the old data shape / experiment
        # context; invalidate them so they re-init.
        st.session_state.chat_data_questions_session = None
        st.session_state.chat_viz_session = None
        st.session_state.chat_viz_figures = {}
        st.sidebar.success(
            f"Applied: {experiment.n_conditions} condition(s) — "
            + ", ".join(
                f"{c} ({n})" for c, n in zip(
                    experiment.conditions_list,
                    experiment.number_of_samples_list,
                )
            )
        )
        st.rerun()


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
    st.session_state.chat_viz_validation = None
    st.session_state.chat_questions_result_df = None
    st.session_state.chat_transform_result = None
    st.session_state.chat_viz_figures = {}
    st.session_state.chat_result_tables = {}
    # Legacy unified session
    st.session_state.chat_unified_session = None
    # Counters
    st.session_state.chat_message_count = 0
    st.session_state.chat_execution_count = 0


# =============================================================================
# Main Page
# =============================================================================

def display_ai_studio_page() -> None:
    """Display the LipidCruncher AI Studio page with 4 tabs."""
    # All data-loading and experiment-definition controls live in the
    # sidebar, mirroring the main app's layout.
    _display_studio_sidebar()
    studio_df = st.session_state.get("chat_studio_df")

    _, center, _ = st.columns([1, 3, 1])

    with center:
        display_logo(centered=True)
        st.subheader("AI Studio")
        st.markdown(
            "Upload your data in the sidebar (left), then use the AI "
            "tabs below to explore, transform, or visualize it."
        )

        st.markdown("---")

        # File info + data preview when data is loaded.
        if studio_df is not None:
            file_id = st.session_state.get("chat_studio_file_id", "")
            file_name = file_id.rsplit("_", 1)[0] if file_id else "data"
            st.success(
                f"**{file_name}** — {len(studio_df)} rows, "
                f"{len(studio_df.columns)} columns."
            )
            with st.expander("Data Preview", expanded=False):
                st.dataframe(studio_df.head(10), width='stretch')
                st.caption(
                    f"{len(studio_df)} rows, {len(studio_df.columns)} columns"
                )
        else:
            st.info("Upload a CSV in the sidebar to begin.")

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
                    "Upload a CSV file above to create custom visualizations."
                )
            else:
                # Validate Module 1 output before showing viz chat
                from app.workflows.ai_chat import AIChatWorkflow
                validation = st.session_state.get('chat_viz_validation')
                if validation is None:
                    validation = AIChatWorkflow.validate_module1_output(
                        studio_df
                    )
                    st.session_state.chat_viz_validation = validation

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
                        "Custom Visualization requires data processed "
                        "through LipidCruncher's Module 1 pipeline. "
                        "Missing columns: " + ", ".join(missing) + ".\n\n"
                        "Please run your data through the main pipeline "
                        "first (**Start Crunching** on the home page), "
                        "then return here."
                    )
                else:
                    experiment = st.session_state.get(
                        "chat_studio_experiment"
                    )
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
