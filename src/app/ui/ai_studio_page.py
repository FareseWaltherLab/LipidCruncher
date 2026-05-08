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

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import PAGE_APP, PAGE_LANDING
from app.models.experiment import ExperimentConfig
from app.services.format_detection import DataFormat, FormatDetectionService
from app.ui.landing_page import display_logo
from app.ui.ai_chat import (
    display_lipidcruncher_questions_chat,
    display_data_questions_chat,
    display_transform_chat,
    display_viz_chat,
)

logger = logging.getLogger(__name__)


# Standard formats that the main pipeline already handles end-to-end
# (cleaning, zero filtering, normalization). When users upload these
# directly to AI Studio, we redirect them rather than letting them try
# to pick sample columns from raw, unfiltered data.
_AI_STUDIO_BLOCKED_FORMATS: frozenset = frozenset({
    DataFormat.LIPIDSEARCH.value,
    DataFormat.MSDIAL.value,
})


def _is_blocked_standard_format() -> bool:
    """True when the uploaded file is a known standard format that should
    go through the main pipeline first. Pipeline imports are exempt: the
    user has already explicitly chosen to bring that data over."""
    if st.session_state.get('chat_studio_pipeline_source') is not None:
        return False
    detected = st.session_state.get('chat_studio_detected_format')
    return detected in _AI_STUDIO_BLOCKED_FORMATS


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


def _extract_sample_name(col_name: str, prefix: Optional[str]) -> str:
    """Strip the {prefix}[...] wrapper from col_name if present, otherwise
    return col_name unchanged. Used to derive a human-readable sample name
    from either pipeline-shaped columns (e.g. 'intensity[A1]' → 'A1') or
    raw user-picked columns (e.g. 'S1_treated' → 'S1_treated')."""
    if prefix is not None:
        head = f'{prefix}['
        if col_name.startswith(head) and col_name.endswith(']'):
            return col_name[len(head):-1]
    return col_name


def _apply_user_experiment(
    df: pd.DataFrame,
    conditions_list: list,
    samples_per_condition: list,
    sample_to_col: dict,
    output_prefix: str,
) -> tuple:
    """Build an ExperimentConfig and a reordered DataFrame.

    The user picks samples per condition in arbitrary order (e.g. condition A
    gets s1+s3, condition B gets s2+s4). ExperimentConfig assumes conditions
    occupy contiguous sample blocks (s1..sN), so we rename the sample columns
    so condition A's samples become s1..sX and condition B's become
    s(X+1)..s(X+Y). The output uses {output_prefix}[sN] regardless of how the
    original columns were named, so override-picked raw columns are
    normalized to the same shape as auto-detected ones. Non-sample columns
    are left untouched.

    Args:
        df: DataFrame containing the sample columns.
        conditions_list: condition names, in declared order.
        samples_per_condition: list of lists; samples_per_condition[i] is the
            list of sample names assigned to conditions_list[i].
        sample_to_col: maps each sample name to its original column in df.
        output_prefix: 'intensity' or 'concentration' — the prefix used for
            renamed sample columns.

    Returns:
        (reordered_df, experiment).
    """
    rename_map = {}
    sample_cols_in_order = []
    new_idx = 1
    for samples in samples_per_condition:
        for old_sample in samples:
            old_col = sample_to_col[old_sample]
            new_col = f'{output_prefix}[s{new_idx}]'
            rename_map[old_col] = new_col
            sample_cols_in_order.append(old_col)
            new_idx += 1

    sample_col_set = set(sample_cols_in_order)
    non_sample = [c for c in df.columns if c not in sample_col_set]
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
        st.session_state.chat_studio_df_original = df
        st.session_state.chat_studio_file_id = file_id
        st.session_state.chat_studio_detected_format = (
            FormatDetectionService.detect_format(df).value
        )
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
    # For the "Raw" option we prefer `standardized_df` over `raw_df`:
    # both contain the same unfiltered, un-normalized values, but the
    # standardized version has the column rename applied (e.g. LipidSearch
    # `MeanArea[s1]` → `intensity[s1]`), which lets AI Studio's auto-
    # detection + experiment pre-populate run end-to-end. Fall back to
    # `raw_df` only if standardization hasn't happened yet.
    raw_for_studio = (
        st.session_state.get("standardized_df")
        if st.session_state.get("standardized_df") is not None
        else st.session_state.get("raw_df")
    )
    if normalized_df is None and raw_for_studio is None:
        return

    st.sidebar.markdown("**Or import from the main pipeline:**")
    options = []
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
    if st.sidebar.button("Import Pipeline Data", key="ai_studio_import_pipeline"):
        is_normalized = "Normalized" in choice
        imported = normalized_df if is_normalized else raw_for_studio
        # Read pipeline state BEFORE clearing — _clear_data_sessions() only
        # touches AI Studio state, but reading first keeps the intent clear.
        pipeline_experiment = st.session_state.get('experiment')
        pipeline_bqc_label = st.session_state.get('bqc_label')

        # Reset all studio state + sidebar widgets so the import starts
        # clean (no stale conditions, sample counts, or BQC selections
        # leaking from a previous import or upload).
        _clear_data_sessions()

        # Set the new data + provenance.
        st.session_state.chat_studio_df = imported
        st.session_state.chat_studio_df_original = imported
        st.session_state.chat_studio_file_id = "_pipeline_import"
        st.session_state.chat_studio_pipeline_source = (
            'normalized' if is_normalized else 'raw'
        )
        if pipeline_experiment is not None:
            st.session_state.chat_studio_experiment = pipeline_experiment
        if pipeline_bqc_label is not None:
            st.session_state.chat_studio_bqc_label = pipeline_bqc_label

        # Pre-populate every studio_* sidebar widget so the user doesn't
        # have to re-walk the flow — the next rerun renders the sidebar
        # with these values, the apply step runs, studio_applied_snapshot
        # is set, and the data tabs unlock immediately. Works for both
        # branches because:
        #   • Normalized imports already have intensity[*]/concentration[*]
        #     columns matching the experiment.
        #   • Raw imports use `standardized_df`, which has the same column
        #     rename applied, so auto-detection finds intensity[*] too.
        # Custom Viz is still hard-locked for raw via pipeline_source ==
        # 'raw' (the data hasn't been cleaned/zero-filtered/normalized).
        if pipeline_experiment is not None:
            st.session_state.studio_sample_cols_radio = "Yes"
            st.session_state.studio_n_conditions = (
                pipeline_experiment.n_conditions
            )
            for i, name in enumerate(pipeline_experiment.conditions_list):
                st.session_state[f'studio_cond_name_{i}'] = name
            for i, n in enumerate(
                pipeline_experiment.number_of_samples_list
            ):
                st.session_state[f'studio_n_samples_{i}'] = int(n)
            st.session_state.studio_grouping_radio = "Yes"
            st.session_state.studio_bqc_radio = (
                "Yes" if pipeline_bqc_label else "No"
            )
            if pipeline_bqc_label:
                st.session_state.studio_bqc_label_radio = pipeline_bqc_label
            st.session_state.studio_confirm_checkbox = True

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


def _display_sample_columns_sidebar(df: pd.DataFrame) -> Optional[tuple]:
    """Render the 'Sample Columns' sidebar block: lists the auto-detected
    sample columns and lets the user override the selection.

    Auto-detection looks for columns named '{prefix}[*]' where prefix is
    'intensity' or 'concentration'. The detected prefix becomes the output
    prefix used when applying the experiment (so 'concentration[*]' inputs
    stay 'concentration[s1..sN]'). When neither prefix is detected, the
    output prefix defaults to 'intensity' and the user must pick the
    sample columns manually.

    Returns:
        (output_prefix, sample_columns) where sample_columns is a list of
        column names from df. Returns None if the user has not yet picked
        any columns in the override flow.
    """
    st.sidebar.subheader("Sample Columns")

    detected_prefix = _detect_sample_prefix(df)
    if detected_prefix is not None:
        head = f'{detected_prefix}['
        auto_cols = [
            c for c in df.columns
            if c.startswith(head) and c.endswith(']')
        ]
    else:
        auto_cols = []

    output_prefix = detected_prefix or 'intensity'

    if not auto_cols:
        st.sidebar.info(
            "No `intensity[*]` or `concentration[*]` columns "
            "auto-detected. Please pick the sample columns below."
        )
        selected = st.sidebar.multiselect(
            "Pick the columns that are samples",
            options=list(df.columns),
            key='studio_sample_cols_select',
        )
        if not selected:
            return None
        return output_prefix, selected

    st.sidebar.caption(
        f"Auto-detected **{len(auto_cols)}** sample column(s):"
    )
    for c in auto_cols:
        st.sidebar.text(f"• {c}")

    correct_kwargs = {'key': 'studio_sample_cols_radio'}
    if 'studio_sample_cols_radio' not in st.session_state:
        correct_kwargs['index'] = 0  # Default to 'Yes'
    correct = st.sidebar.radio(
        "Are these the correct sample columns?",
        ['Yes', 'No'],
        **correct_kwargs,
    )

    if correct == 'Yes':
        return output_prefix, auto_cols

    selected = st.sidebar.multiselect(
        "Pick the columns that are samples",
        options=list(df.columns),
        default=auto_cols,
        key='studio_sample_cols_select',
    )
    if not selected:
        return None
    return output_prefix, selected


def _display_studio_bqc_sidebar(
    condition_names: list, sample_counts: list,
) -> Optional[str]:
    """Render the 'Specify Label of BQC Samples' sidebar block, mirroring
    the main app. Returns the chosen condition label, or None if the user
    answers 'No' (or no condition has 2+ samples)."""
    st.sidebar.subheader("Specify Label of BQC Samples")

    bqc_kwargs = {'key': 'studio_bqc_radio'}
    if 'studio_bqc_radio' not in st.session_state:
        bqc_kwargs['index'] = 1  # Default to 'No'
    bqc_ans = st.sidebar.radio(
        'Do you have Batch Quality Control (BQC) samples?',
        ['Yes', 'No'],
        **bqc_kwargs,
    )
    if bqc_ans != 'Yes':
        return None

    eligible = [
        c for c, n in zip(condition_names, sample_counts) if n > 1
    ]
    if not eligible:
        st.sidebar.warning(
            "No conditions with 2+ samples available for BQC."
        )
        return None

    label_kwargs = {'key': 'studio_bqc_label_radio'}
    if 'studio_bqc_label_radio' not in st.session_state:
        label_kwargs['index'] = 0
    return st.sidebar.radio(
        'Which label corresponds to BQC samples?',
        eligible,
        **label_kwargs,
    )


def _display_studio_confirm_inputs_sidebar(
    condition_names: list, sample_counts: list,
) -> bool:
    """Render the 'Confirm Inputs' sidebar block: total-samples line,
    a per-condition summary of the planned post-apply layout (samples are
    renamed to contiguous s1..sN per condition when applied), and a
    confirmation checkbox.

    Returns True if the box is checked.
    """
    st.sidebar.subheader("Confirm Inputs")

    total_samples = sum(sample_counts)
    st.sidebar.write(f"There are a total of {total_samples} samples.")

    idx = 1
    for cond, count in zip(condition_names, sample_counts):
        if count > 5:
            display_text = (
                f"• s{idx} to s{idx + count - 1} "
                f"(total {count}) correspond to {cond}"
            )
        else:
            samples_str = '-'.join(f's{idx + j}' for j in range(count))
            display_text = f"• {samples_str} correspond to {cond}"
        st.sidebar.text(display_text)
        idx += count

    confirm_kwargs = {'key': 'studio_confirm_checkbox'}
    if 'studio_confirm_checkbox' not in st.session_state:
        confirm_kwargs['value'] = False
    return st.sidebar.checkbox(
        "Confirm the inputs by checking this box", **confirm_kwargs,
    )


def _render_standard_format_advisory_sidebar() -> None:
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
        # Clear AI Studio state — the user has chosen to take the data
        # through the main pipeline instead. They'll re-import via the
        # 'Import from main pipeline' control after Module 1 finishes.
        _clear_data_sessions()
        st.session_state.page = PAGE_APP
        st.rerun()


def _render_standard_format_advisory_main(file_name: str) -> None:
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


def _display_studio_sidebar() -> None:
    """Render the full AI Studio sidebar, mirroring the main app:
    upload + pipeline import → Define Experiment → Group Samples →
    Specify Label of BQC Samples → Confirm Inputs."""
    _handle_file_upload_sidebar()
    _offer_pipeline_import_sidebar()

    # Read from the original (un-renamed) df so the sidebar always shows
    # the user's actual column names. The apply step below renames into
    # `chat_studio_df` for Custom Viz; reading the original here also makes
    # re-applies idempotent (we never rename already-renamed columns).
    studio_df = st.session_state.get('chat_studio_df_original')
    if studio_df is None:
        return

    st.sidebar.markdown("---")

    # Standard formats (LipidSearch, MS-DIAL) belong in the main pipeline,
    # which does grade/score filtering, zero filtering, and normalization
    # — none of which AI Studio replicates. Block the sidebar flow here
    # so the user is steered to the main pipeline instead of being asked
    # to hand-pick sample columns from raw data.
    if _is_blocked_standard_format():
        _render_standard_format_advisory_sidebar()
        return

    sample_cols_result = _display_sample_columns_sidebar(studio_df)
    if sample_cols_result is None:
        return
    output_prefix, sample_columns = sample_cols_result

    sample_names = [
        _extract_sample_name(c, output_prefix) for c in sample_columns
    ]
    if len(set(sample_names)) != len(sample_names):
        st.sidebar.error(
            "Duplicate sample names after stripping the column prefix. "
            "Please pick uniquely-named sample columns."
        )
        return
    sample_to_col = dict(zip(sample_names, sample_columns))

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

    bqc_label = _display_studio_bqc_sidebar(condition_names, sample_counts)

    confirmed = _display_studio_confirm_inputs_sidebar(
        condition_names, sample_counts,
    )
    if not confirmed:
        return

    # Streamlit reruns the script on every interaction. The checkbox stays
    # True across reruns, so we'd otherwise rebuild the experiment and wipe
    # per-tab chat sessions on every rerun. Only re-apply when the inputs
    # actually change by comparing a snapshot.
    snapshot = (
        st.session_state.get('chat_studio_file_id'),
        output_prefix,
        tuple(sample_columns),
        tuple(condition_names),
        tuple(tuple(s) for s in samples_per_condition),
        bqc_label,
    )
    if st.session_state.get('studio_applied_snapshot') == snapshot:
        return

    try:
        reordered_df, experiment = _apply_user_experiment(
            studio_df, condition_names, samples_per_condition,
            sample_to_col, output_prefix,
        )
    except (ValueError, TypeError, KeyError) as e:
        st.sidebar.error(str(e))
        return

    st.session_state.chat_studio_df = reordered_df
    st.session_state.chat_studio_experiment = experiment
    st.session_state.chat_studio_bqc_label = bqc_label
    # Only Custom Viz reads the renamed df; Data Questions and Transform
    # read the original (which doesn't change on apply), so their sessions
    # don't need to be invalidated here. They are still cleared on a fresh
    # upload via _clear_data_sessions().
    st.session_state.chat_viz_session = None
    st.session_state.chat_viz_figures = {}
    st.session_state.studio_applied_snapshot = snapshot
    st.rerun()


def _clear_data_sessions() -> None:
    """Reset all data-dependent chat sessions and sidebar widget keys when
    the underlying data changes (new upload, pipeline import, navigation).
    Widget keys must be popped (not assigned None) — Streamlit refuses
    direct assignment to a widget's key after the widget has rendered.
    """
    st.session_state.chat_studio_df = None
    st.session_state.chat_studio_df_original = None
    st.session_state.chat_studio_file_id = None
    st.session_state.chat_studio_experiment = None
    st.session_state.chat_studio_pipeline_source = None
    st.session_state.chat_studio_bqc_label = None
    st.session_state.chat_studio_detected_format = None
    st.session_state.studio_applied_snapshot = None
    # Sidebar widget keys: pop static ones explicitly + dynamic ones by
    # prefix. Pre-populated values from a pipeline import would otherwise
    # leak into a subsequent fresh upload or different import.
    static_widget_keys = (
        'studio_sample_cols_radio', 'studio_sample_cols_select',
        'studio_n_conditions', 'studio_grouping_radio',
        'studio_bqc_radio', 'studio_bqc_label_radio',
        'studio_confirm_checkbox',
    )
    for key in static_widget_keys:
        st.session_state.pop(key, None)
    dynamic_widget_prefixes = (
        'studio_cond_name_', 'studio_n_samples_', 'studio_select_',
    )
    dynamic_keys = [
        key for key in list(st.session_state.keys())
        if isinstance(key, str) and key.startswith(dynamic_widget_prefixes)
    ]
    for key in dynamic_keys:
        st.session_state.pop(key, None)
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
    # Restore any sidebar widget keys that Streamlit garbage-collected
    # while the user was on the main app page. Must run BEFORE the
    # sidebar widgets render so they see the restored values.
    StreamlitAdapter.restore_navigation_widgets()

    # All data-loading and experiment-definition controls live in the
    # sidebar, mirroring the main app's layout.
    _display_studio_sidebar()
    # `studio_df_original` keeps the user's column names — used by the
    # data preview, Data Questions, and Transform tabs.
    # `studio_df_processed` is the renamed-to-`s1..sN` view used by Custom
    # Viz, which equals the original until the user confirms an experiment.
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

        with st.expander("How to use the AI Studio", expanded=False):
            st.markdown(
                """
**Quick start**

1. Upload your CSV in the sidebar (or import from the main pipeline).
2. Define your experiment in the sidebar — sample columns, conditions,
   groupings, and BQC label.
3. Check **Confirm the inputs by checking this box** to unlock the
   data-dependent tabs.

**Tabs**

- **LipidCruncher Questions** — Ask anything about the app's features,
  methods, or analyses. **No upload required.**
- **Data Questions** — Explore your dataset with the AI (summaries,
  filters, distributions). Requires data + confirm.
- **Transform to Generic** — Ask the AI to convert non-LipidCruncher
  data into the Generic format the main pipeline accepts. Requires
  data + confirm. Use this when your data isn't ready for the main
  pipeline yet.
- **Custom Visualization** — Generate custom Plotly charts beyond what
  the built-in modules offer. Requires data that has been **processed
  through Module 1** of the main pipeline. Run your data through
  *Start Crunching* on the home page first, then come back here and
  use **Import from main pipeline** in the sidebar.
                """
            )

        st.markdown("---")

        # File info + data preview when data is loaded. Show the user's
        # original column names so the preview matches what the user sees
        # in Data Questions / Transform.
        blocked = _is_blocked_standard_format()
        if studio_df_original is not None:
            file_id = st.session_state.get("chat_studio_file_id", "")
            pipeline_source = st.session_state.get(
                "chat_studio_pipeline_source"
            )
            if pipeline_source == 'normalized':
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
                _render_standard_format_advisory_main(file_label)
            else:
                st.success(
                    f"**{file_label}** — {len(studio_df_original)} rows, "
                    f"{len(studio_df_original.columns)} columns."
                )
                with st.expander("Data Preview", expanded=False):
                    st.dataframe(studio_df_original.head(10), width='stretch')
                    st.caption(
                        f"{len(studio_df_original)} rows, "
                        f"{len(studio_df_original.columns)} columns"
                    )
        else:
            st.info("Upload a CSV in the sidebar to begin.")

        st.markdown("---")

        # The three data-dependent tabs are locked until the user finishes
        # the sidebar flow and checks "Confirm Inputs". `studio_applied_
        # snapshot` is set by the apply step the moment the box is first
        # checked, and persists for the rest of the session unless the
        # data is cleared. LC Questions is exempt — it doesn't need data.
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
            if studio_df_original is None:
                st.info(
                    "Upload a CSV file above to ask questions about your data."
                )
            elif blocked:
                st.info(blocked_tab_msg)
            elif not is_confirmed:
                st.info(not_confirmed_msg)
            else:
                display_data_questions_chat(studio_df_original)

        with tab_transform:
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
            elif pipeline_source_for_transform == 'normalized':
                # Normalized imports are filtered AND normalized — there's
                # genuinely nothing left to transform. Raw imports, by
                # contrast, are Generic-shaped but unfiltered/un-normalized,
                # so the user may legitimately want to ask the AI to filter
                # or normalize them in custom ways. Don't lock for raw.
                st.info(
                    "Transformation isn't needed — your data was imported "
                    "as **normalized** data from the main pipeline, so "
                    "it's already filtered and normalized. Use **Data "
                    "Questions** to explore it, or **Custom "
                    "Visualization** to plot it."
                )
            elif not is_confirmed:
                st.info(not_confirmed_msg)
            else:
                display_transform_chat(studio_df_original)

        with tab_viz:
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
                from app.workflows.ai_chat import AIChatWorkflow
                validation = st.session_state.get('chat_viz_validation')
                if validation is None:
                    validation = AIChatWorkflow.validate_module1_output(
                        studio_df_processed
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
                        experiment = _derive_experiment(studio_df_processed)
                        st.session_state.chat_studio_experiment = experiment
                    display_viz_chat(studio_df_processed, experiment)

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
                # Round-trip: keep both AI Studio state and pipeline state
                # so the user can hop back without losing work on either side.
                # Snapshot the studio_* widgets first — Streamlit GCs them
                # while the AI Studio page is not rendering, so without
                # this the sidebar would reset on the next return.
                StreamlitAdapter.preserve_ai_studio_widgets()
                st.session_state.page = PAGE_APP
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
        # Clear all studio-specific state
        _clear_data_sessions()
        st.session_state.chat_lc_questions_session = None
        st.session_state.page = PAGE_LANDING
        st.rerun()
