"""
Sample Grouping UI Components for LipidCruncher sidebar.

This module contains:
- display_group_samples: Display group samples section with dataframe and manual regrouping
- display_sample_grouping: Complete sample grouping UI orchestration
"""

import logging
from typing import Optional, Tuple

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

from app.models.experiment import ExperimentConfig
from app.services.sample_grouping import SampleGroupingService
from app.ui.sidebar.experiment_config import detect_sample_columns, extract_sample_names, display_experiment_definition
from app.ui.sidebar.confirm_inputs import display_bqc_section, display_confirm_inputs


# =============================================================================
# UI Components
# =============================================================================

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

    st.sidebar.dataframe(group_df, use_container_width=True)

    # Manual regrouping option
    st.sidebar.write('Are your samples properly grouped together?')
    grouping_correct = st.sidebar.radio('', ['Yes', 'No'], key='grouping_radio')

    if grouping_correct == 'No':
        return _handle_manual_regrouping(df, group_df, experiment)
    else:
        st.session_state.grouping_complete = True
        # Restore original column order if it was stored
        if st.session_state.get('original_column_order') is not None:
            df = df.reindex(columns=st.session_state.original_column_order)

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

    # Always regroup from the original data to make the operation idempotent
    base_df = st.session_state.get('_pre_regroup_df', df)

    st.session_state.grouping_complete = False
    selections = {}
    remaining_samples = group_df['sample name'].tolist()
    expected_samples = dict(zip(experiment.conditions_list, experiment.number_of_samples_list))

    for condition in experiment.conditions_list:
        st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")

        selected_samples = st.sidebar.multiselect(
            f'Pick the samples that belong to condition {condition}',
            remaining_samples,
            key=f'select_{condition}'
        )

        selections[condition] = selected_samples

        if len(selected_samples) == expected_samples[condition]:
            remaining_samples = [s for s in remaining_samples if s not in selected_samples]

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
        st.session_state.grouping_complete = True
        st.sidebar.write("New sample order after regrouping:")
        st.sidebar.dataframe(regroup_result.name_df, use_container_width=True)
        return regroup_result.group_df, regroup_result.reordered_df
    except ValueError as e:
        logger.error("Error updating groups: %s", e)
        st.sidebar.error(
            "Could not update sample groups. Please verify that every condition has the correct number "
            "of samples selected. If the issue persists after refreshing the app, contact abdih@mskcc.org."
        )
        st.session_state.grouping_complete = False
        return group_df, df


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

    # Step 2: Group Samples (with manual regrouping option)
    group_df, updated_df = display_group_samples(df, experiment, data_format)

    if group_df is None:
        return None, None

    # Check if grouping is complete
    if not st.session_state.get('grouping_complete', False):
        st.sidebar.error("Please complete sample grouping before proceeding.")
        return None, None

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
