"""
Confirm Inputs UI Components for LipidCruncher sidebar.

This module contains:
- display_bqc_section: BQC (Batch Quality Control) sample specification
- display_confirm_inputs: Input confirmation section with summary
"""

import streamlit as st

from app.models.experiment import ExperimentConfig
from app.ui.sample_labels import display_names_for

# A condition's samples are listed in full up to this many, then summarised.
MAX_LISTED_SAMPLES = 10

_MARKDOWN_SPECIAL = set('\\`*_{}[]()<>#+-.!|~$')


def _escape_markdown(text: str) -> str:
    """Backslash-escape Markdown syntax so a name like 'a|b' or '$5' shows as typed."""
    return ''.join('\\' + ch if ch in _MARKDOWN_SPECIAL else ch for ch in str(text))


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

    # Pass `index` only when the key is absent. The pre-fill helpers (sample
    # data / alignment) may set `bqc_radio` in session state; passing `index`
    # as well makes Streamlit warn that a keyed widget has both a default and a
    # session-state value. Once the key exists, session state preserves the
    # selection across reruns, so the default is only needed on first render.
    bqc_kwargs = {'key': 'bqc_radio'}
    if 'bqc_radio' not in st.session_state:
        bqc_kwargs['index'] = 1
    bqc_ans = st.sidebar.radio(
        'Do you have Batch Quality Control (BQC) samples?',
        ['Yes', 'No'],
        **bqc_kwargs,
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
            # Same guard as bqc_radio: the pre-fill helpers may set
            # 'bqc_label_radio' in session state, so only supply `index` when
            # the key is absent to avoid the default-plus-session-state warning.
            label_kwargs = {'key': 'bqc_label_radio'}
            if 'bqc_label_radio' not in st.session_state:
                label_kwargs['index'] = 0
            bqc_label = st.sidebar.radio(
                'Which label corresponds to BQC samples?',
                conditions_with_two_plus,
                **label_kwargs,
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

    total_samples = sum(experiment.number_of_samples_list)
    n_conditions = len(experiment.conditions_list)
    st.sidebar.markdown(
        f"{total_samples} samples in {n_conditions} "
        f"condition{'' if n_conditions == 1 else 's'}"
    )

    # One heading per condition with its samples listed underneath, under the
    # names the "Show samples as" switch picks. Every sample is listed rather
    # than a first-to-last range, which says nothing once samples have names.
    names = display_names_for('upload')
    for i, condition in enumerate(experiment.conditions_list):
        if condition and condition.strip():
            samples = [names.get(s, s) for s in experiment.individual_samples_list[i]]
            count = len(samples)
            st.sidebar.markdown(
                f"**{_escape_markdown(condition)}** · "
                f"{count} sample{'' if count == 1 else 's'}"
            )
            listed = ', '.join(_escape_markdown(s) for s in samples[:MAX_LISTED_SAMPLES])
            if count > MAX_LISTED_SAMPLES:
                listed += f", … and {count - MAX_LISTED_SAMPLES} more"
            st.sidebar.caption(listed)
        else:
            st.sidebar.error(f"Empty condition found at index {i}")

    # Confirmation checkbox
    confirm_kwargs = {'key': 'confirm_checkbox'}
    if 'confirm_checkbox' not in st.session_state:
        confirm_kwargs['value'] = False
    return st.sidebar.checkbox("Confirm the inputs by checking this box", **confirm_kwargs)
