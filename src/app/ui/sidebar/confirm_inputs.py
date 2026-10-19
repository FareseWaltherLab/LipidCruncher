"""
Confirm Inputs UI Components for LipidCruncher sidebar.

This module contains:
- display_bqc_section: BQC (Batch Quality Control) sample specification
- display_confirm_inputs: Input confirmation section with summary
- render_confirm_summary: the summary itself, shared with the AI Studio sidebar
"""

from typing import Sequence

import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.models.experiment import ExperimentConfig
from app.ui.sample_labels import display_names_for

# A condition's samples are listed in full up to this many, then summarised.
MAX_LISTED_SAMPLES = 10

_MARKDOWN_SPECIAL = set('\\`*_{}[]()<>#+-.!|~$')


def _escape_markdown(text: str) -> str:
    """Backslash-escape Markdown syntax so a name like 'a|b' or '$5' shows as typed."""
    return ''.join('\\' + ch if ch in _MARKDOWN_SPECIAL else ch for ch in str(text))


def _restore_widget_key(widget_key: str, preserved_key: str, options: list) -> None:
    """Put a dropped widget key back from its preserved value, if still an option."""
    if widget_key not in st.session_state:
        preserved = st.session_state.get(preserved_key)
        if preserved in options:
            st.session_state[widget_key] = preserved


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
    bqc_options = ['Yes', 'No']
    # On a run where this section was skipped (e.g. a manual regroup is
    # mid-edit) Streamlit drops the widget keys; restore the last answers.
    _restore_widget_key('bqc_radio', '_preserved_sidebar_bqc_radio', bqc_options)
    bqc_kwargs = {'key': 'bqc_radio'}
    if 'bqc_radio' not in st.session_state:
        bqc_kwargs['index'] = 1
    bqc_ans = st.sidebar.radio(
        'Do you have Batch Quality Control (BQC) samples?',
        bqc_options,
        **bqc_kwargs,
    )
    StreamlitAdapter.save_widget_value('bqc_radio', '_preserved_sidebar_bqc_radio')

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
            _restore_widget_key(
                'bqc_label_radio', '_preserved_sidebar_bqc_label_radio', conditions_with_two_plus
            )
            label_kwargs = {'key': 'bqc_label_radio'}
            if 'bqc_label_radio' not in st.session_state:
                label_kwargs['index'] = 0
            bqc_label = st.sidebar.radio(
                'Which label corresponds to BQC samples?',
                conditions_with_two_plus,
                **label_kwargs,
            )
            StreamlitAdapter.save_widget_value('bqc_label_radio', '_preserved_sidebar_bqc_label_radio')
        else:
            st.sidebar.warning("No conditions with 2+ samples available for BQC.")

    return bqc_label


def render_confirm_summary(
    conditions: Sequence[str],
    kept: Sequence[Sequence[str]],
    excluded: Sequence[str] = (),
) -> None:
    """Render the Confirm Inputs summary; the AI Studio sidebar uses it too.

    Args:
        conditions: Condition labels, in order.
        kept: kept[i] holds the display names of the samples of conditions[i]
            that will be analysed. A condition with none is left out, since
            excluding all its samples drops it from the analysis.
        excluded: Display names of the excluded samples.
    """
    total_samples = sum(len(samples) for samples in kept)
    n_conditions = sum(1 for samples in kept if samples)
    st.sidebar.markdown(
        f"{total_samples} samples in {n_conditions} "
        f"condition{'' if n_conditions == 1 else 's'}"
    )

    # One heading per condition with its samples listed underneath. Every
    # sample is listed rather than a first-to-last range, which says nothing
    # once samples have names.
    for i, (condition, samples) in enumerate(zip(conditions, kept)):
        if not samples:
            continue
        if condition and condition.strip():
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
    if excluded:
        st.sidebar.caption(
            "Excluded: " + ', '.join(_escape_markdown(s) for s in excluded)
        )


def display_confirm_inputs(
    experiment: ExperimentConfig, excluded: Sequence[str] = (),
) -> bool:
    """
    Display confirm inputs section with summary.

    Args:
        experiment: ExperimentConfig with conditions and samples
        excluded: Samples picked under "Exclude Samples". The summary counts
            and lists only the samples that will be analysed, and names the
            excluded ones, under the labels the sidebar uses.

    Returns:
        True if user confirms, False otherwise
    """
    st.sidebar.subheader("Confirm Inputs")

    # Samples are shown under the names the "Show samples as" switch picks.
    dropped = set(excluded)
    names = display_names_for('upload')
    render_confirm_summary(
        experiment.conditions_list,
        [
            [names.get(s, s) for s in samples if s not in dropped]
            for samples in experiment.individual_samples_list
        ],
        [names.get(s, s) for s in experiment.full_samples_list if s in dropped],
    )

    # Confirmation checkbox
    confirm_kwargs = {'key': 'confirm_checkbox'}
    if 'confirm_checkbox' not in st.session_state:
        confirm_kwargs['value'] = False
    return st.sidebar.checkbox("Confirm the inputs by checking this box", **confirm_kwargs)
