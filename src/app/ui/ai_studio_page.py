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
from app.services.format_detection import (
    DataFormat,
    FormatDetectionService,
)
from app.services.sample_inference import SampleInferenceService
from app.ui.landing_page import display_logo
from app.ui.sidebar.confirm_inputs import render_confirm_summary
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
    DataFormat.METABOLOMICS_WORKBENCH.value,
})


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
        # Metabolomics Workbench files are multi-section text documents whose
        # real table only the main pipeline's MW parser can recover.
        # detect_format() recognizes them by scanning the raw text for the
        # MS_METABOLITE_DATA markers — it cannot see them from a DataFrame
        # (a plain read_csv yields a garbage "Unnamed" frame). Detect from the
        # text first so an MW upload is flagged as MW (and thus blocked) rather
        # than mis-read as a table the user is asked to hand-pick columns from.
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        text_format = FormatDetectionService.detect_format(text)
        # LipidSearch 5.2 exports are tab-delimited with a .csv extension; sniff
        # the delimiter so they don't collapse into a single column.
        delimiter = FormatDetectionService.sniff_delimiter(text)
        uploaded.seek(0)
        df = pd.read_csv(uploaded, sep=delimiter)
        if df.empty:
            st.sidebar.error("The uploaded file is empty.")
            return
        _clear_data_sessions()
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_df_original = df
        st.session_state.chat_studio_file_id = file_id
        detected = (
            text_format
            if text_format == DataFormat.METABOLOMICS_WORKBENCH
            else FormatDetectionService.detect_format(df)
        )
        st.session_state.chat_studio_detected_format = detected.value
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")


def _offer_pipeline_import_sidebar() -> None:
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
    normalized_df = st.session_state.get("normalized_df")
    # The Quality Check module's output: normalized data with high-CoV lipids
    # dropped and any PCA-flagged samples removed. Only present once the user
    # has visited Module 2. It carries its own experiment because sample
    # removal renumbers the survivors (see SessionState.qc_experiment).
    qc_df = st.session_state.get("qc_continuation_df")
    # For the "Raw" option we prefer `standardized_df` over `raw_df`:
    # both contain the same unfiltered, un-normalized values, but the
    # standardized version has the column rename applied (e.g. LipidSearch
    # `MeanArea[s1]` → `intensity[s1]`), which lets AI Studio's auto-
    # detection + experiment pre-populate run end-to-end. Fall back to
    # `raw_df` only if standardization hasn't happened yet. Once inputs are
    # confirmed, `processing_frame` comes first: `experiment` describes it,
    # not `standardized_df`, when samples were excluded in the sidebar.
    processing = st.session_state.get("processing_frame")
    raw_for_studio = next(
        (df for df in (
            processing.df if processing is not None else None,
            st.session_state.get("standardized_df"),
            st.session_state.get("raw_df"),
        ) if df is not None),
        None,
    )
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
            source_key, imported = 'qc_filtered', qc_df
        elif choice.startswith("Normalized"):
            source_key, imported = 'normalized', normalized_df
        else:
            source_key, imported = 'raw', raw_for_studio
        # Read pipeline state BEFORE clearing — _clear_data_sessions() only
        # touches AI Studio state, but reading first keeps the intent clear.
        # QC output must pair with the QC-updated experiment: PCA sample
        # removal drops sample columns and renumbers the survivors, so the
        # pre-QC experiment would claim samples the data no longer has.
        pipeline_experiment = st.session_state.get('experiment')
        if source_key == 'qc_filtered':
            qc_experiment = st.session_state.get('qc_experiment')
            if qc_experiment is not None:
                pipeline_experiment = qc_experiment
        pipeline_bqc_label = st.session_state.get('bqc_label')
        # PCA removal can drop the whole BQC condition. Left in place, the
        # Studio's BQC radio would fall back to its first option and treat
        # an ordinary condition as BQC.
        if (
            pipeline_experiment is not None
            and pipeline_bqc_label not in pipeline_experiment.conditions_list
        ):
            pipeline_bqc_label = None

        # Reset all studio state + sidebar widgets so the import starts
        # clean (no stale conditions, sample counts, or BQC selections
        # leaking from a previous import or upload).
        _clear_data_sessions()

        # Set the new data + provenance.
        st.session_state.chat_studio_df = imported
        st.session_state.chat_studio_df_original = imported
        st.session_state.chat_studio_file_id = "_pipeline_import"
        st.session_state.chat_studio_pipeline_source = source_key
        if pipeline_experiment is not None:
            st.session_state.chat_studio_experiment = pipeline_experiment
        if pipeline_bqc_label is not None:
            st.session_state.chat_studio_bqc_label = pipeline_bqc_label

        # Pre-populate every studio_* sidebar widget so the user doesn't
        # have to re-walk the flow — the next rerun renders the sidebar
        # with these values, the apply step runs, studio_applied_snapshot
        # is set, and the data tabs unlock immediately. Works for every
        # branch because:
        #   • Normalized imports already have intensity[*]/concentration[*]
        #     columns matching the experiment.
        #   • QC-filtered imports are the normalized frame minus high-CoV
        #     lipids and PCA-removed samples, and pipeline_experiment is the
        #     QC-updated config, so the columns still match.
        #   • Raw imports use `standardized_df`, which has the same column
        #     rename applied, so auto-detection finds intensity[*] too.
        # Custom Viz is still hard-locked for raw via pipeline_source ==
        # 'raw' (the data hasn't been cleaned/zero-filtered/normalized).
        if pipeline_experiment is not None:
            st.session_state.studio_sample_cols_radio = "Yes"
            # The pipeline's columns are {prefix}[s1..sN], so the Assign
            # Conditions table's samples are the experiment's s-labels.
            st.session_state.studio_conditions = dict(zip(
                pipeline_experiment.full_samples_list,
                pipeline_experiment.extensive_conditions_list,
            ))
            st.session_state.studio_bqc_radio = (
                "Yes" if pipeline_bqc_label else "No"
            )
            if pipeline_bqc_label:
                st.session_state.studio_bqc_label_radio = pipeline_bqc_label
            st.session_state.studio_confirm_checkbox = True

        st.rerun()


def _display_assign_conditions_sidebar(sample_names: list) -> Optional[tuple]:
    """Render the 'Assign Conditions' sidebar block: one editable row per
    sample with its condition, pre-filled by guessing the condition from the
    sample name ('KO-1' → 'KO'), or 'Condition_1' for every sample when the
    names show no grouping. Samples with the same condition form a group;
    conditions keep the order they first appear in.

    The table's values live in ``studio_conditions`` (sample → condition),
    so they survive navigation, which drops the editor's own state.

    Returns:
        (condition_names, samples_per_condition), or None while a sample
        has no condition.
    """
    st.sidebar.subheader("Assign Conditions")

    stored = st.session_state.get('studio_conditions') or {}
    if list(stored) != list(sample_names):
        # Other sample columns were picked: start again from a guess. The
        # editor's edits refer to rows of the old table, so drop them too.
        st.session_state.pop('studio_conditions_editor', None)
        guesses = (
            SampleInferenceService.guess_conditions(sample_names)
            or ['Condition_1'] * len(sample_names)
        )
        stored = dict(zip(sample_names, guesses))

    st.sidebar.caption(
        "Pre-filled from the sample names. Edit any condition; samples with "
        "the same condition are grouped together."
    )
    edited = st.sidebar.data_editor(
        pd.DataFrame({
            'sample': sample_names,
            'condition': [stored[s] for s in sample_names],
        }),
        hide_index=True,
        width='stretch',
        disabled=['sample'],
        column_config={
            'sample': st.column_config.TextColumn('Sample'),
            'condition': st.column_config.TextColumn('Condition'),
        },
        key='studio_conditions_editor',
    )
    conditions = [
        str(c).strip() if pd.notna(c) else '' for c in edited['condition']
    ]
    st.session_state.studio_conditions = dict(zip(sample_names, conditions))

    if not all(conditions):
        st.sidebar.error("Every sample needs a condition.")
        return None

    condition_names = list(dict.fromkeys(conditions))
    samples_per_condition = [
        [s for s, c in zip(sample_names, conditions) if c == name]
        for name in condition_names
    ]
    return condition_names, samples_per_condition


def _pick_sample_columns(numeric: list, guess: list) -> list:
    """The 'Pick the columns that are samples' multiselect over the numeric
    columns (only those can hold sample values), starting from ``guess``
    the first time it renders."""
    key = 'studio_sample_cols_select'
    if key not in st.session_state:
        st.session_state[key] = guess
    else:
        st.session_state[key] = [c for c in st.session_state[key] if c in numeric]
    return st.sidebar.multiselect(
        "Pick the columns that are samples", options=numeric, key=key,
    )


def _guess_sample_columns_sidebar(numeric: list) -> list:
    """Guess which numeric columns are samples, for a file with no
    intensity[...] / concentration[...] columns.

    When the columns form measurement families (Area[...], Height[...]), the
    user picks the family that holds the sample values and its columns are
    the guess. Otherwise the guess is every numeric column whose name does
    not look like metadata (RT, m/z, score, ...).
    """
    families = SampleInferenceService.column_families(numeric)
    if len(families) == 1:
        return next(iter(families.values()))
    if not families:
        return [
            c for c in numeric
            if not SampleInferenceService.looks_like_metadata(c)
        ]

    own_pick = "Pick columns myself"
    labels = {
        f"{name}[…] ({len(cols)} columns)": cols
        for name, cols in families.items()
    }

    def _select_family():
        # Choosing a family replaces the picked columns with its columns.
        choice = st.session_state.studio_sample_family
        st.session_state.studio_sample_cols_select = labels.get(choice, [])

    choice = st.sidebar.radio(
        "Which measurement holds the sample values?",
        list(labels) + [own_pick],
        key='studio_sample_family',
        on_change=_select_family,
    )
    return labels.get(choice, [])


def _display_sample_columns_sidebar(df: pd.DataFrame) -> Optional[tuple]:
    """Render the 'Sample Columns' sidebar block: lists the auto-detected
    sample columns and lets the user override the selection.

    Auto-detection looks for columns named '{prefix}[*]' where prefix is
    'intensity' or 'concentration'. The detected prefix becomes the output
    prefix used when applying the experiment (so 'concentration[*]' inputs
    stay 'concentration[s1..sN]'). When neither prefix is detected, the
    output prefix defaults to 'intensity' and the user picks the sample
    columns, starting from a guess (see _guess_sample_columns_sidebar).
    Only columns holding numbers are offered.

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
    numeric = SampleInferenceService.numeric_columns(df)

    if not auto_cols:
        guess = _guess_sample_columns_sidebar(numeric)
        selected = _pick_sample_columns(numeric, guess)
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

    selected = _pick_sample_columns(
        [c for c in df.columns if c in auto_cols or c in numeric], auto_cols,
    )
    if not selected:
        return None
    return output_prefix, selected


def _display_studio_exclusion_sidebar(sample_names: list) -> list:
    """Render the optional 'Exclude Samples' block, mirroring the main app.

    Excluded samples are left out of the experiment and of the data the Data
    Questions and Custom Visualization tabs see; excluding every sample of a
    condition drops that condition. Returns the excluded sample names.
    """
    key = 'studio_exclude_samples'
    # A changed sample list (other columns picked) must not leave a selection
    # the multiselect no longer offers.
    if key in st.session_state:
        st.session_state[key] = [
            s for s in st.session_state[key] if s in sample_names
        ]
    with st.sidebar.expander(
        '🚫 Exclude Samples (optional)',
        expanded=bool(st.session_state.get(key)),
    ):
        return st.multiselect(
            'Samples to exclude',
            sample_names,
            help=(
                "Left out of Data Questions and Custom Visualization. "
                "Excluding every sample of a condition drops that condition. "
                "The remaining samples are renumbered s1, s2, …"
            ),
            key=key,
        )


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
    condition_names: list, kept: list, excluded: list,
) -> bool:
    """Render the 'Confirm Inputs' sidebar block: the main sidebar's summary
    (each condition with the samples it keeps, under the dataset's own
    sample names, and the excluded ones) and a confirmation checkbox.

    Returns True if the box is checked.
    """
    st.sidebar.subheader("Confirm Inputs")
    render_confirm_summary(condition_names, kept, excluded)

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
    upload + pipeline import → Sample Columns → Assign Conditions →
    Exclude Samples → Specify Label of BQC Samples → Confirm Inputs."""
    _handle_file_upload_sidebar()
    _offer_pipeline_import_sidebar()

    # Read from the original (un-renamed) df so the sidebar always shows
    # the user's actual column names. The apply step below renames into
    # `chat_studio_df` for the experiment-aware tabs; reading the original
    # here also makes re-applies idempotent (we never rename
    # already-renamed columns).
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
        SampleInferenceService.sample_name(c) for c in sample_columns
    ]
    if len(set(sample_names)) != len(sample_names):
        st.sidebar.error(
            "Duplicate sample names after stripping the column prefix. "
            "Please pick uniquely-named sample columns."
        )
        return
    sample_to_col = dict(zip(sample_names, sample_columns))

    assignment = _display_assign_conditions_sidebar(sample_names)
    if assignment is None:
        return
    condition_names, samples_per_condition = assignment
    sample_counts = [len(samples) for samples in samples_per_condition]

    excluded = _display_studio_exclusion_sidebar(sample_names)
    kept = [
        [s for s in samples if s not in excluded]
        for samples in samples_per_condition
    ]
    if excluded and sum(len(samples) for samples in kept) < 2:
        st.sidebar.error("At least two samples must remain after exclusion.")
        return

    bqc_label = _display_studio_bqc_sidebar(condition_names, sample_counts)
    # Same rule as the main sidebar: a BQC condition whose samples were all
    # excluded is not in the experiment the tabs get.
    if bqc_label is not None and not kept[condition_names.index(bqc_label)]:
        st.sidebar.warning(
            f"All '{bqc_label}' samples are excluded; continuing without BQC."
        )
        bqc_label = None

    confirmed = _display_studio_confirm_inputs_sidebar(
        condition_names, kept, [s for s in sample_names if s in excluded],
    )
    if not confirmed:
        # Unchecking the box must re-lock the data-dependent tabs, which gate
        # on `studio_applied_snapshot`. Clear it so the tabs turn inactive;
        # re-checking re-applies because the stored snapshot is now None.
        st.session_state.studio_applied_snapshot = None
        return

    # Streamlit reruns the script on every interaction. The checkbox stays
    # True across reruns, so we'd otherwise rebuild the experiment and wipe
    # per-tab chat sessions on every rerun. Only re-apply when the inputs
    # actually change by comparing a snapshot.
    kept_conditions = [c for c, samples in zip(condition_names, kept) if samples]
    kept_samples = [samples for samples in kept if samples]
    snapshot = (
        st.session_state.get('chat_studio_file_id'),
        output_prefix,
        tuple(sample_columns),
        tuple(kept_conditions),
        tuple(tuple(s) for s in kept_samples),
        bqc_label,
    )
    if st.session_state.get('studio_applied_snapshot') == snapshot:
        return

    try:
        reordered_df, experiment = _apply_user_experiment(
            studio_df.drop(columns=[sample_to_col[s] for s in excluded]),
            kept_conditions, kept_samples, sample_to_col, output_prefix,
        )
    except (ValueError, TypeError, KeyError) as e:
        st.sidebar.error(str(e))
        return

    st.session_state.chat_studio_df = reordered_df
    st.session_state.chat_studio_experiment = experiment
    st.session_state.chat_studio_bqc_label = bqc_label
    # Persist the sample-column names so the sandbox can pre-coerce
    # them to numeric before exposing `df` to AI code. Two lists — one
    # for the original df (used by Transform) and one for the renamed df
    # (used by Data Questions and Custom Viz). The renamed names are
    # always `{output_prefix}[s1..sN]` derived from the experiment.
    st.session_state.chat_studio_sample_columns_original = list(sample_columns)
    st.session_state.chat_studio_sample_columns_renamed = [
        f'{output_prefix}[{s}]' for s in experiment.full_samples_list
    ]
    # Data Questions and Custom Viz read the renamed df, whose columns
    # change on apply, so their sessions start over. Transform reads the
    # original (which doesn't change on apply); it is still cleared on a
    # fresh upload via _clear_data_sessions().
    st.session_state.chat_viz_session = None
    st.session_state.chat_viz_figures = {}
    st.session_state.chat_data_questions_session = None
    st.session_state.chat_questions_result_df = None
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
    st.session_state.chat_studio_sample_columns_original = None
    st.session_state.chat_studio_sample_columns_renamed = None
    st.session_state.studio_applied_snapshot = None
    # Sidebar widget keys, and the Assign Conditions table's values.
    # Pre-populated values from a pipeline import would otherwise leak into
    # a subsequent fresh upload or different import.
    static_widget_keys = (
        'studio_sample_cols_radio', 'studio_sample_family',
        'studio_sample_cols_select', 'studio_conditions',
        'studio_conditions_editor',
        'studio_bqc_radio', 'studio_bqc_label_radio',
        'studio_exclude_samples', 'studio_confirm_checkbox',
    )
    for key in static_widget_keys:
        st.session_state.pop(key, None)
    # Cross-page snapshots written by preserve_ai_studio_widgets() would
    # otherwise re-populate the sidebar on the next rerun via
    # restore_navigation_widgets(), defeating the clear.
    nav_snapshot_keys = [
        key for key in list(st.session_state.keys())
        if isinstance(key, str)
        and key.startswith('_nav_preserved_studio_')
    ]
    for key in nav_snapshot_keys:
        st.session_state.pop(key, None)
    # Per-tab sessions
    st.session_state.chat_data_questions_session = None
    st.session_state.chat_transform_session = None
    st.session_state.chat_transform_diagnostic = None
    st.session_state.chat_viz_session = None
    st.session_state.chat_viz_validation = None
    st.session_state.chat_questions_result_df = None
    st.session_state.chat_transform_result = None
    st.session_state.chat_transform_report = None
    st.session_state.chat_viz_figures = {}


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
        blocked = _is_blocked_standard_format()
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
                _render_standard_format_advisory_main(file_label)
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
                # When the current Studio data came from a pipeline
                # import, clear it on exit so the next re-entry shows a
                # fresh import-choice sidebar (the user can pull raw or
                # normalized data again). Direct uploads have no
                # pipeline source and are preserved across the round-
                # trip via the widget snapshot — Streamlit GCs studio_*
                # keys while the AI Studio page isn't rendering, and
                # restore_navigation_widgets() re-hydrates them on the
                # way back.
                if st.session_state.get(
                    'chat_studio_pipeline_source'
                ) is not None:
                    _clear_data_sessions()
                else:
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
