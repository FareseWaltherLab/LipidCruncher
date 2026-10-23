"""
Sample Grouping UI Components for LipidCruncher sidebar.

This module contains:
- display_group_samples: Display group samples section with dataframe and manual regrouping
- display_sample_grouping: Complete sample grouping UI orchestration
"""

import logging
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.constants import FORMAT_MSDIAL
from app.models.experiment import ExperimentConfig
from app.models.sample_frame import SampleFrame
from app.services.sample_grouping import DatasetValidationResult, SampleGroupingService
from app.ui.sample_names import (
    build_names_from_mapping,
    names_from_editor,
    sample_display_map,
)
from app.ui.sample_labels import labeler, render_sample_label_switch, showing_original_names
from app.ui.sidebar.experiment_config import detect_sample_columns, display_experiment_definition
from app.ui.sidebar.confirm_inputs import display_bqc_section, display_confirm_inputs

logger = logging.getLogger(__name__)


# =============================================================================
# UI Components
# =============================================================================

def _detected_sample_column_names() -> List[str]:
    """Original (pre-rename) names of the columns taken as sample intensities.

    By validation time the columns are already renamed to ``intensity[sN]``,
    which is useless for spotting an impostor — the mapping is where the names
    the user actually wrote still live.
    """
    mapping = st.session_state.get('column_mapping')
    if mapping is None or 'standardized_name' not in mapping:
        return []
    is_intensity = (
        mapping['standardized_name'].astype(str).str.startswith('intensity[')
    )
    return mapping.loc[is_intensity, 'original_name'].astype(str).tolist()


def _explain_sample_count_mismatch(
    validation: DatasetValidationResult, data_format: str,
) -> None:
    """Turn a bare count mismatch into something the user can act on.

    "Found 13, expected 12" reads as *I mis-declared my experiment*, but the
    usual cause is the opposite: sample columns are detected by content, so any
    numeric column the format doesn't publish as metadata gets absorbed as a
    sample. Naming the columns that were taken lets the user spot the stray one
    at a glance, and the fix differs by format — MS-DIAL has an override, the
    others need the file corrected.
    """
    expected, actual = validation.expected_samples, validation.n_intensity_cols
    if not expected or not actual or expected == actual:
        return

    detected = _detected_sample_column_names()
    if detected:
        st.sidebar.caption(
            "Read as samples: " + ", ".join(f"`{c}`" for c in detected)
        )

    if data_format == FORMAT_MSDIAL:
        st.sidebar.caption(
            "If that list is wrong, fix it under **🔧 Override Sample "
            "Detection** in the Column Name Standardization section above. "
            "Otherwise correct the experiment setup."
        )
    else:
        st.sidebar.caption(
            "Every column after the lipid name — apart from an optional "
            "lipid-class column — is read as a sample. Remove any extra "
            "columns and re-upload, or correct the experiment setup."
        )


def display_group_samples(df: pd.DataFrame, experiment: ExperimentConfig, data_format: str) -> tuple:
    """
    Display group samples section with dataframe and manual regrouping option.

    Args:
        df: Standardized DataFrame
        experiment: ExperimentConfig with conditions and samples
        data_format: The selected data format string

    Returns:
        Tuple of (group_df, updated_df)
    """
    st.sidebar.subheader('Group Samples')

    # Validate dataset
    validation = SampleGroupingService.validate_dataset(df, experiment, data_format)
    if not validation.valid:
        st.sidebar.error(validation.message)
        _explain_sample_count_mismatch(validation, data_format)
        return None, df

    # Build group_df
    workbench_conditions = st.session_state.get('workbench_conditions')
    try:
        result = SampleGroupingService.build_group_df(
            df, experiment, data_format, workbench_conditions,
        )
    except ValueError as e:
        logger.error("Error building sample groups: %s", e)
        st.sidebar.error(
            "Could not build sample groups. Please check that the number of conditions and samples "
            "matches your dataset. If the issue persists after refreshing the app, contact abdih@mskcc.org."
        )
        return None, df

    group_df = result.group_df

    # Show the original names alongside the standardized labels; both stay
    # visible here whichever the "Show samples as" switch picks, so the user
    # can read one against the other.
    names = st.session_state.get('sample_names') or {}
    display_group = group_df.rename(columns={
        'sample name': 'Standardized Label', 'condition': 'Condition',
    })
    if names:
        display_group.insert(
            1, 'Original Name', [names.get(s, '') for s in group_df['sample name']]
        )
    st.sidebar.dataframe(display_group, width='stretch')

    # Manual regrouping option
    st.sidebar.write('Are your samples properly grouped together?')
    # A run that stopped before this step (an invalid experiment setup)
    # dropped the widget keys; restore the answer so that fixing the setup
    # does not undo a manual regroup.
    if 'grouping_radio' not in st.session_state:
        preserved = st.session_state.get('_preserved_grouping_radio')
        if preserved is not None:
            st.session_state.grouping_radio = preserved
    grouping_correct = st.sidebar.radio(
        'Samples properly grouped?', ['Yes', 'No'], key='grouping_radio',
        label_visibility='collapsed',
    )
    st.session_state._preserved_grouping_radio = grouping_correct

    if grouping_correct == 'No':
        return _handle_manual_regrouping(df, group_df, experiment)
    # Undoes a previous regroup: the snapshot of the upload order comes back.
    return group_df, StreamlitAdapter.undo_regroup(df)


def _handle_manual_regrouping(df: pd.DataFrame, group_df: pd.DataFrame, experiment: ExperimentConfig) -> tuple:
    """Handle manual sample regrouping UI and apply regrouping if valid.

    Returns (group_df, updated_df).
    """
    # Always regroup from the upload-order snapshot so the operation is
    # idempotent across reruns.
    base_df, base_names = StreamlitAdapter.begin_regroup(df)

    st.session_state.grouping_complete = False
    selections = {}
    remaining_samples = group_df['sample name'].tolist()
    expected_samples = dict(zip(experiment.conditions_list, experiment.number_of_samples_list))

    # Option labels come from the pre-regroup snapshot, which is keyed like the
    # options and never changes during the regroup. NOT the live sample_names:
    # the regroup below rewrites those mid-run, which changed the option labels,
    # reset these widgets and wiped the user's selections.
    shown = sample_display_map(base_names) if showing_original_names() else {}

    for condition in experiment.conditions_list:
        st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")

        # Flipping "Show samples as" changes the option labels, which makes
        # Streamlit treat this as a new widget; seed it with the selection it
        # had so the flip does not blank the picker.
        key = f'select_{condition}'
        if key in st.session_state:
            picked = st.session_state[key] or []
        else:
            picked = st.session_state._preserved_regroup_selections.get(condition, [])
        previous = [s for s in picked if s in remaining_samples]
        selected_samples = st.sidebar.multiselect(
            f'Pick the samples that belong to condition {condition}',
            remaining_samples,
            default=previous or None,
            format_func=lambda s: shown.get(s, s),
            key=key,
        )

        selections[condition] = selected_samples

        if len(selected_samples) == expected_samples[condition]:
            remaining_samples = [s for s in remaining_samples if s not in selected_samples]
    st.session_state._preserved_regroup_selections = selections

    # Verify all conditions have correct number of samples
    all_correct = all(
        len(selections[condition]) == expected_samples[condition]
        for condition in experiment.conditions_list
    )

    if not all_correct:
        st.session_state.grouping_complete = False
        return group_df, df

    try:
        regroup_result = SampleGroupingService.regroup_samples(
            base_df, group_df, selections, experiment,
        )
        # Keep display names in sync with the renumbered/reordered samples.
        # Renumber from the stable pre-regroup snapshot (not the live, possibly
        # already-remapped map) so the result is identical on every rerun.
        regrouped = SampleFrame.from_upload(base_df, experiment, base_names).renumbered(
            regroup_result.reordered_df, experiment, regroup_result.old_to_new,
        )
        StreamlitAdapter.apply_regroup(regrouped)
        st.sidebar.write("New sample order after regrouping:")
        st.sidebar.dataframe(regroup_result.name_df, width='stretch')
        return regroup_result.group_df, regroup_result.reordered_df
    except ValueError as e:
        logger.error("Error updating groups: %s", e)
        st.sidebar.error(
            "Could not update sample groups. Please verify that every condition has the correct number "
            "of samples selected. If the issue persists after refreshing the app, contact abdih@mskcc.org."
        )
        st.session_state.grouping_complete = False
        return group_df, df


def _display_sample_name_editor(experiment: ExperimentConfig) -> None:
    """Let the user review/edit the display name for each sample.

    Names are auto-seeded from the uploaded column headers (or a LipidSearch
    5.2 Alignment Setting file's raw filenames) and stored in
    ``st.session_state.sample_names`` keyed by internal label (s1, s2, ...).
    The "Show samples as" switch decides whether they or the s-labels are
    shown in tables, plots, sample pickers and downloads.
    """
    labels = experiment.full_samples_list
    names = st.session_state.get('sample_names') or {}

    with st.sidebar.expander('✏️ Sample Names (optional)', expanded=False):
        st.caption(
            "Choose under **Show samples as** whether these names or the "
            "standardized labels (s1, s2, ...) appear in tables, plots, sample "
            "pickers and downloads. "
            "Auto-filled from your file's column headers (or, for LipidSearch "
            "5.2, the Alignment Setting file's raw filenames) — edit as needed."
        )
        editor_df = pd.DataFrame({
            'sample': labels,
            'name': [names.get(label, '') for label in labels],
        })
        edited = st.data_editor(
            editor_df,
            hide_index=True,
            width='stretch',
            column_config={
                'sample': st.column_config.TextColumn(
                    'Standardized Label', disabled=True,
                ),
                'name': st.column_config.TextColumn('Original Name'),
            },
            key='sample_names_editor',
        )
        StreamlitAdapter.set_sample_names(names_from_editor(edited))


def _display_sample_exclusion(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> Optional[SampleFrame]:
    """Optional sidebar step: drop samples before any processing.

    For a sample that should not enter the analysis at all — e.g. one whose
    internal-standard spike failed, which would otherwise stop normalization.
    Returns the frame Module 1 should run on: the sidebar's samples when
    nothing is excluded, the renumbered survivors otherwise, None when the
    selection is invalid.
    """
    # Exclusions are kept by the sample's upload label, so a regroup that
    # renumbers samples keeps excluding the same samples.
    restored, regroup_origin = StreamlitAdapter.restore_excluded_samples()
    valid_restored = [s for s in restored if s in experiment.full_samples_list]
    with st.sidebar.expander('🚫 Exclude Samples (optional)', expanded=bool(valid_restored)):
        excluded = st.multiselect(
            'Samples to exclude',
            experiment.full_samples_list,
            default=valid_restored,
            format_func=labeler('upload'),
            help=(
                "Dropped before cleaning, filtering and normalization — e.g. a "
                "sample whose internal standard failed. Excluding every sample "
                "of a condition drops that condition. The remaining samples are "
                "renumbered s1, s2, …"
            ),
            key='exclude_samples',
        )
        StreamlitAdapter.save_excluded_samples(excluded, regroup_origin)

        frame = SampleFrame(
            df=df,
            experiment=experiment,
            names=st.session_state.get('sample_names') or {},
            origin={s: regroup_origin.get(s, s) for s in experiment.full_samples_list},
        )
        if not excluded:
            return frame
        try:
            result = SampleGroupingService.exclude_samples(df, experiment, excluded)
        except ValueError as e:
            st.error(str(e))
            return None
        remaining = result.updated_experiment.full_samples_list
        st.info(
            f"{len(result.removed_samples)} sample(s) excluded; "
            f"{len(remaining)} remain, renumbered s1–s{len(remaining)}."
        )
        return frame.renumbered(
            result.updated_df, result.updated_experiment, result.old_to_new,
        )


def display_sample_grouping(df: pd.DataFrame, data_format: str) -> Tuple[Optional[ExperimentConfig], Optional[str]]:
    """
    Display complete sample grouping UI in sidebar.
    Includes: experiment definition, group samples, BQC, and confirmation.

    Args:
        df: Standardized DataFrame
        data_format: The selected data format string

    Returns:
        Tuple of (experiment, bqc_label) or (None, None) if not confirmed
    """
    # Detect sample columns
    sample_cols = detect_sample_columns(df, data_format)

    if not sample_cols:
        st.sidebar.error("No sample columns detected!")
        return None, None

    # Step 1: Define Experiment
    result = display_experiment_definition(df, data_format, sample_cols)
    if result[0] is None:
        return None, None

    n_conditions, conditions_list, number_of_samples_list = result

    # Create ExperimentConfig
    try:
        experiment = ExperimentConfig(
            n_conditions=n_conditions,
            conditions_list=conditions_list,
            number_of_samples_list=number_of_samples_list
        )
    except (ValueError, TypeError) as e:
        logger.error("Invalid experiment setup: %s", e)
        st.sidebar.error(
            "Invalid experiment setup. Please ensure all condition names are filled in and each condition "
            "has at least one sample. If the issue persists after refreshing the app, contact abdih@mskcc.org."
        )
        return None, None

    # Seed display names on first run: from the Alignment Setting file's raw
    # filenames for a LipidSearch 5.2 alignment upload (its column mapping holds
    # only per-file tokens), otherwise from the uploaded column headers.
    if st.session_state.get('sample_names') is None:
        st.session_state.sample_names = (
            st.session_state.get('lipidsearch_sample_names')
            or build_names_from_mapping(st.session_state.get('column_mapping'))
        )

    # Step 2: Group Samples (with manual regrouping option)
    group_df, updated_df = display_group_samples(df, experiment, data_format)

    if group_df is None:
        return None, None

    # Check if grouping is complete
    if not st.session_state.get('grouping_complete', False):
        st.sidebar.error("Please complete sample grouping before proceeding.")
        return None, None

    # Step 2b: Optional sample display-name editing, and which labels to show
    _display_sample_name_editor(experiment)
    render_sample_label_switch()

    # Step 2c: Optional sample exclusion. The sidebar keeps the full df and
    # experiment; Module 1 runs on the reduced frame.
    processing = _display_sample_exclusion(updated_df, experiment)

    # Step 3: BQC Section
    bqc_label = display_bqc_section(experiment)
    # A BQC condition whose samples were all excluded no longer exists in the
    # experiment Module 1 runs on.
    if (
        bqc_label is not None and processing is not None
        and bqc_label not in processing.experiment.conditions_list
    ):
        st.sidebar.warning(
            f"All '{bqc_label}' samples are excluded; continuing without BQC."
        )
        bqc_label = None

    # Step 4: Confirm Inputs
    confirmed = display_confirm_inputs(
        experiment, excluded=st.session_state.get('exclude_samples') or [],
    )

    if confirmed:
        if processing is None:
            st.sidebar.error("Fix the sample exclusion above before confirming.")
            st.session_state.confirmed = False
            return None, None
        st.session_state.experiment = processing.experiment
        st.session_state.bqc_label = bqc_label
        st.session_state.confirmed = True
        st.session_state.standardized_df = updated_df
        st.session_state.processing_frame = processing
        return processing.experiment, bqc_label
    else:
        # Checkbox unchecked - clear confirmed state
        st.session_state.confirmed = False
        return None, None
