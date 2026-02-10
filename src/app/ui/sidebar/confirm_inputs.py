"""
Confirm Inputs UI Components for LipidCruncher sidebar.

This module contains:
- display_bqc_section: BQC (Batch Quality Control) sample specification
- display_confirm_inputs: Input confirmation section with summary
"""

import streamlit as st

from app.models.experiment import ExperimentConfig


# =============================================================================
# UI Components
# =============================================================================

def display_bqc_section(experiment: ExperimentConfig) -> str:
    """
    Display BQC sample specification section.

    Args:
        experiment: ExperimentConfig with conditions and samples

    Returns:
        The BQC label or None if no BQC samples specified
    """
    st.sidebar.subheader("Specify Label of BQC Samples")

    bqc_ans = st.sidebar.radio(
        'Do you have Batch Quality Control (BQC) samples?',
        ['Yes', 'No'],
        index=1,  # Default to 'No'
        key='bqc_radio'
    )

    bqc_label = None
    if bqc_ans == 'Yes':
        # Filter to conditions with 2+ samples
        conditions_with_two_plus = [
            condition for condition, n_samples
            in zip(experiment.conditions_list, experiment.number_of_samples_list)
            if n_samples > 1
        ]

        if conditions_with_two_plus:
            bqc_label = st.sidebar.radio(
                'Which label corresponds to BQC samples?',
                conditions_with_two_plus,
                index=0,
                key='bqc_label_radio'
            )
        else:
            st.sidebar.warning("No conditions with 2+ samples available for BQC.")

    return bqc_label


def display_confirm_inputs(experiment: ExperimentConfig) -> bool:
    """
    Display confirm inputs section with summary.

    Args:
        experiment: ExperimentConfig with conditions and samples

    Returns:
        True if user confirms, False otherwise
    """
    st.sidebar.subheader("Confirm Inputs")

    # Display total number of samples
    total_samples = sum(experiment.number_of_samples_list)
    st.sidebar.write(f"There are a total of {total_samples} samples.")

    # Display sample-condition pairings
    for i, condition in enumerate(experiment.conditions_list):
        if condition and condition.strip():
            samples = experiment.individual_samples_list[i]

            if len(samples) > 5:
                display_text = f"• {samples[0]} to {samples[-1]} (total {len(samples)}) correspond to {condition}"
            else:
                display_text = f"• {'-'.join(samples)} correspond to {condition}"

            # Use st.sidebar.text() to avoid markdown parsing of pipe characters
            st.sidebar.text(display_text)
        else:
            st.sidebar.error(f"Empty condition found at index {i}")

    # Confirmation checkbox
    return st.sidebar.checkbox("Confirm the inputs by checking this box", key='confirm_checkbox')
