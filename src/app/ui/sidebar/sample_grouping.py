"""
Sample Grouping UI Components for LipidCruncher sidebar.

This module contains:
- display_group_samples: Display group samples section with dataframe and manual regrouping
- display_sample_grouping: Complete sample grouping UI orchestration
"""

import streamlit as st
import pandas as pd

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
        st.sidebar.error(f"Error building sample groups: {e}")
        return None, df

    group_df = result.group_df

    st.sidebar.dataframe(group_df, use_container_width=True)

    # Manual regrouping option
    st.sidebar.write('Are your samples properly grouped together?')
    grouping_correct = st.sidebar.radio('', ['Yes', 'No'], key='grouping_radio')

    if grouping_correct == 'No':
        # Store original column order if not already stored
        if st.session_state.get('original_column_order') is None:
            st.session_state.original_column_order = df.columns.tolist()

        st.session_state.grouping_complete = False
        selections = {}
        remaining_samples = group_df['sample name'].tolist()

        # Keep track of expected samples per condition
        expected_samples = dict(zip(experiment.conditions_list, experiment.number_of_samples_list))

        # Process each condition
        for condition in experiment.conditions_list:
            st.sidebar.write(f"Select {expected_samples[condition]} samples for {condition}")

            selected_samples = st.sidebar.multiselect(
                f'Pick the samples that belong to condition {condition}',
                remaining_samples,
                key=f'select_{condition}'
            )

            selections[condition] = selected_samples

            # Update remaining samples only if correct number selected
            if len(selected_samples) == expected_samples[condition]:
                remaining_samples = [s for s in remaining_samples if s not in selected_samples]

        # Verify all conditions have correct number of samples
        all_correct = all(
            len(selections[condition]) == expected_samples[condition]
            for condition in experiment.conditions_list
        )

        if all_correct:
            try:
                regroup_result = SampleGroupingService.regroup_samples(
                    df, group_df, selections, experiment,
                )

                st.session_state.grouping_complete = True

                st.sidebar.write("New sample order after regrouping:")
                st.sidebar.dataframe(regroup_result.name_df, use_container_width=True)

                return regroup_result.group_df, regroup_result.reordered_df

            except ValueError as e:
                st.sidebar.error(f"Error updating groups: {str(e)}")
                st.session_state.grouping_complete = False
                return group_df, df
        else:
            st.session_state.grouping_complete = False
            return group_df, df
    else:
        st.session_state.grouping_complete = True
        # Restore original column order if it was stored
        if st.session_state.get('original_column_order') is not None:
            df = df.reindex(columns=st.session_state.original_column_order)

    return group_df, df


def display_sample_grouping(df: pd.DataFrame, data_format: str):
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
        st.sidebar.error(f"Invalid experiment setup: {e}")
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
