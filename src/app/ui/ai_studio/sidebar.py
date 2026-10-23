"""
The AI Studio sidebar, mirroring the main app's: data source, Sample Columns,
Assign Conditions, Exclude Samples, BQC, Confirm Inputs, and the apply step
that turns them into the experiment the data tabs work on.
"""
from typing import Optional

import pandas as pd
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.sample_inference import SampleInferenceService
from app.ui.ai_studio.data_source import (
    handle_file_upload_sidebar,
    is_blocked_standard_format,
    offer_pipeline_import_sidebar,
    render_standard_format_advisory_sidebar,
)
from app.ui.sidebar.confirm_inputs import render_confirm_summary
from app.workflows.ai_studio import AIStudioWorkflow


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

    detected_prefix, auto_cols = AIStudioWorkflow.detected_sample_columns(df)

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


def display_studio_sidebar() -> None:
    """Render the full AI Studio sidebar, mirroring the main app:
    upload + pipeline import → Sample Columns → Assign Conditions →
    Exclude Samples → Specify Label of BQC Samples → Confirm Inputs."""
    handle_file_upload_sidebar()
    offer_pipeline_import_sidebar()

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
    if is_blocked_standard_format():
        render_standard_format_advisory_sidebar()
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
        StreamlitAdapter.unconfirm_studio_experiment()
        return

    kept_conditions = [c for c, samples in zip(condition_names, kept) if samples]
    kept_samples = [samples for samples in kept if samples]
    snapshot = AIStudioWorkflow.apply_snapshot(
        st.session_state.get('chat_studio_file_id'),
        output_prefix, sample_columns, kept_conditions, kept_samples, bqc_label,
    )
    if st.session_state.get('studio_applied_snapshot') == snapshot:
        return

    try:
        reordered_df, experiment = AIStudioWorkflow.apply_user_experiment(
            studio_df.drop(columns=[sample_to_col[s] for s in excluded]),
            kept_conditions, kept_samples, sample_to_col, output_prefix,
        )
    except (ValueError, TypeError, KeyError) as e:
        st.sidebar.error(str(e))
        return

    StreamlitAdapter.store_studio_experiment(
        reordered_df, experiment, bqc_label, output_prefix, sample_columns,
        snapshot,
    )
    st.rerun()
