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

logger = logging.getLogger(__name__)

from app.constants import FORMAT_MSDIAL
from app.models.experiment import ExperimentConfig
from app.services.sample_grouping import DatasetValidationResult, SampleGroupingService
from app.ui.sample_names import (
    build_names_from_mapping,
    names_from_editor,
    remap_names_after_regroup,
)
from app.ui.sidebar.experiment_config import detect_sample_columns, extract_sample_names, display_experiment_definition
from app.ui.sidebar.confirm_inputs import display_bqc_section, display_confirm_inputs


# =============================================================================
# UI Components
# =============================================================================

def _invalidate_confirmation_if_grouping_changed() -> None:
    """Detect a user change to the 'samples properly grouped?' radio
    *or* to any of the per-condition sample multiselects since the
    previous render. When anything changes, drop the Confirm Inputs
    checkbox so the user has to re-verify the summary against the new
    ordering. Uses a post-rerun comparison instead of an ``on_change``
    callback — pop'ing widget keys inside ``on_change`` causes the
    radio to need two clicks to commit a toggle in Streamlit 1.50.

    Also drops the cross-page ``_nav_preserved_confirm_checkbox``
    snapshot. Without that, a later rerun where the confirm widget
    doesn't render (e.g. mid-edit with grouping_complete=False) lets
    Streamlit GC the live key, then ``restore_navigation_widgets()``
    re-installs the stale True value from the snapshot — making the
    checkbox silently reappear as checked even after invalidation."""
    current_radio = st.session_state.get('grouping_radio')
    last_radio = st.session_state.get('_last_seen_grouping_radio')
    grouping_changed = (
        current_radio is not None and last_radio is not None
        and current_radio != last_radio
    )

    # Multiselect-change detection only makes sense in 'No' (manual
    # regrouping) mode — in 'Yes' mode the select_* widgets don't
    # render, so a diff vs. the snapshot just reflects Streamlit's
    # widget GC rather than a user intent change.
    selects_changed = False
    if current_radio == 'No':
        current_selects = {
            k: tuple(st.session_state[k] or ())
            for k in st.session_state
            if isinstance(k, str) and k.startswith('select_')
        }
        last_selects = st.session_state.get('_last_seen_selects')
        selects_changed = (
            last_selects is not None
            and current_selects != last_selects
        )

    if grouping_changed or selects_changed:
        # Explicitly assign False rather than pop. Popping + re-creating
        # the widget with value=False isn't reliable in Streamlit 1.50:
        # the widget's internal state cache (keyed by widget id) can
        # still hold the previous True, so the checkbox renders checked
        # even though session_state was cleared. Direct assignment makes
        # session_state the single source of truth — the widget reads it
        # via `key=` and renders unchecked.
        st.session_state['confirm_checkbox'] = False
        st.session_state.pop('_nav_preserved_confirm_checkbox', None)
        st.session_state.confirmed = False


def _snapshot_bqc_widgets_to_preserved() -> None:
    """Copy the live values of the BQC widgets into their ``_preserved_*``
    shadows. The save line inside ``display_bqc_section`` only runs when
    grouping is complete; while the user is mid-regroup the BQC section
    is skipped, Streamlit drops the un-rendered widget keys at the end
    of the run, and the next render of the BQC section would otherwise
    fall back to the hard-coded default of 'No'. Calling this at the
    top of ``display_sample_grouping`` (before any short-circuit return)
    keeps the shadow in sync with whatever the user last picked."""
    if 'bqc_radio' in st.session_state:
        st.session_state['_preserved_bqc_radio'] = (
            st.session_state['bqc_radio']
        )
    if 'bqc_label_radio' in st.session_state:
        st.session_state['_preserved_bqc_label_radio'] = (
            st.session_state['bqc_label_radio']
        )


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

    # Show the user-facing sample names alongside the internal s-labels.
    names = st.session_state.get('sample_names') or {}
    display_group = group_df.copy()
    if names:
        display_group.insert(
            1, 'name', [names.get(s, '') for s in group_df['sample name']]
        )
    st.sidebar.dataframe(display_group, width='stretch')

    # Manual regrouping option
    st.sidebar.write('Are your samples properly grouped together?')
    grouping_correct = st.sidebar.radio(
        '', ['Yes', 'No'], key='grouping_radio',
    )
    # Record the value the user is now seeing so the NEXT rerun can
    # detect a toggle and invalidate the Confirm Inputs checkbox.
    st.session_state['_last_seen_grouping_radio'] = grouping_correct

    if grouping_correct == 'No':
        return _handle_manual_regrouping(df, group_df, experiment)
    else:
        st.session_state.grouping_complete = True
        # Restore the original pre-regrouping DataFrame if user previously
        # reshuffled samples.  A simple reindex is insufficient because
        # regrouping renames columns (e.g. intensity[s4] → intensity[s1]),
        # so the column names are identical before and after — only the data
        # underneath differs.
        if st.session_state.get('_pre_regroup_df') is not None:
            df = st.session_state['_pre_regroup_df'].copy()
            # Commit the restored ordering to standardized_df now, not just
            # to the local df. The grouping-radio toggle popped
            # confirm_checkbox, so display_sample_grouping will return
            # (None, None) this run and skip its usual
            # `standardized_df = updated_df` write. Without this line the
            # next rerun reads df = standardized_df (still the regrouped
            # version), the user's re-check of Confirm Inputs re-commits
            # the regrouped data, and the original ordering is lost.
            st.session_state.standardized_df = df
            # Clean up regrouping state so it doesn't persist
            del st.session_state['_pre_regroup_df']
            st.session_state.pop('original_column_order', None)
            # Restore the pre-regroup display names to match the restored order.
            if '_pre_regroup_sample_names' in st.session_state:
                st.session_state.sample_names = st.session_state.pop(
                    '_pre_regroup_sample_names'
                )

    return group_df, df


def _handle_manual_regrouping(df: pd.DataFrame, group_df: pd.DataFrame, experiment: ExperimentConfig) -> tuple:
    """Handle manual sample regrouping UI and apply regrouping if valid.

    Returns (group_df, updated_df).
    """
    if st.session_state.get('original_column_order') is None:
        st.session_state.original_column_order = df.columns.tolist()
        # Save the original pre-regrouping data so that regrouping is always
        # applied to the untouched DataFrame.  Without this, every Streamlit
        # re-run re-applies the swap to already-swapped data, corrupting the
        # column assignments.
        st.session_state['_pre_regroup_df'] = df.copy()
        # Snapshot the pre-regroup display names too, so the remap below is
        # always computed from this stable base rather than compounding on
        # already-remapped names each rerun (which would churn sample_names,
        # change the multiselect option labels, and reset the widgets).
        st.session_state['_pre_regroup_sample_names'] = st.session_state.get('sample_names')

    # Always regroup from the original data to make the operation idempotent
    base_df = st.session_state.get('_pre_regroup_df', df)

    st.session_state.grouping_complete = False
    selections = {}
    remaining_samples = group_df['sample name'].tolist()
    expected_samples = dict(zip(experiment.conditions_list, experiment.number_of_samples_list))

    for condition in experiment.conditions_list:
        st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")

        # NOTE: no format_func here. The picker's options are the same s-labels
        # shown (with names) in the Group Samples table above. Deriving option
        # labels from sample_names would reset this widget whenever the regroup
        # updates sample_names mid-run, wiping the user's selections.
        selected_samples = st.sidebar.multiselect(
            f'Pick the samples that belong to condition {condition}',
            remaining_samples,
            key=f'select_{condition}',
        )

        selections[condition] = selected_samples

        if len(selected_samples) == expected_samples[condition]:
            remaining_samples = [s for s in remaining_samples if s not in selected_samples]

    # Snapshot the live select_* values so the next rerun's
    # `_invalidate_confirmation_if_grouping_changed()` can detect any
    # user-driven change. Taken before the all_correct branch so the
    # baseline updates on every render of this function, including
    # mid-edit (invalid-count) runs.
    st.session_state['_last_seen_selects'] = {
        k: tuple(st.session_state[k] or ())
        for k in st.session_state
        if isinstance(k, str) and k.startswith('select_')
    }

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
        # Remap from the stable pre-regroup snapshot (not the live, possibly
        # already-remapped map) so the result is identical on every rerun.
        st.session_state.sample_names = remap_names_after_regroup(
            st.session_state.get('_pre_regroup_sample_names'),
            regroup_result.old_to_new,
        )
        st.session_state.grouping_complete = True
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
    They appear in the Group Samples table and in CSV downloads.
    """
    labels = experiment.full_samples_list
    names = st.session_state.get('sample_names') or {}

    with st.sidebar.expander('✏️ Sample Names (optional)', expanded=False):
        st.caption(
            "These names are shown in the Group Samples table and replace s1, "
            "s2, ... in CSV downloads; plots keep the s-labels. "
            "Auto-filled from your file's column headers — edit as needed."
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
                'sample': st.column_config.TextColumn('Sample', disabled=True),
                'name': st.column_config.TextColumn('Display name'),
            },
            key='sample_names_editor',
        )
        st.session_state.sample_names = names_from_editor(edited)


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
    # Run before any short-circuit return below: shadow the BQC widget
    # values so they survive a rerun where the BQC section is skipped,
    # and clear the Confirm Inputs checkbox if the grouping radio just
    # changed.
    _snapshot_bqc_widgets_to_preserved()
    _invalidate_confirmation_if_grouping_changed()

    # Detect sample columns
    sample_cols = detect_sample_columns(df, data_format)
    sample_names = extract_sample_names(sample_cols, data_format)

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

    # Step 2b: Optional sample display-name editing
    _display_sample_name_editor(experiment)

    # Step 3: BQC Section
    bqc_label = display_bqc_section(experiment)

    # Step 4: Confirm Inputs
    confirmed = display_confirm_inputs(experiment)

    if confirmed:
        st.session_state.experiment = experiment
        st.session_state.bqc_label = bqc_label
        st.session_state.confirmed = True
        st.session_state.standardized_df = updated_df
        return experiment, bqc_label
    else:
        # Checkbox unchecked - clear confirmed state
        st.session_state.confirmed = False
        return None, None
