"""
Zero filtering UI component for configuring and applying zero value filtering.
"""
import logging

import pandas as pd
import streamlit as st

from app.constants import (
    ZERO_FILTER_NON_BQC_DEFAULT,
    ZERO_FILTER_BQC_DEFAULT,
)
from ..models.experiment import ExperimentConfig
from ..adapters.streamlit_adapter import StreamlitAdapter
from ..services.filter_guidance import FilterGuidanceService
from ..services.zero_filtering import ZeroFilterConfig
from ..ui.download_utils import csv_download_button
from ..ui.filter_guidance import default_detection_threshold, show_notices

logger = logging.getLogger(__name__)


def _init_threshold_state(default_detection: float, data_format: str) -> None:
    """Seed the preserved thresholds; the detection threshold resets with the format."""
    if (st.session_state.get('_preserved_zero_filter_detection_threshold') is None
            or st.session_state.get('_zero_filter_format') != data_format):
        st.session_state._preserved_zero_filter_detection_threshold = default_detection
        st.session_state._zero_filter_format = data_format
    if st.session_state.get('_preserved_non_bqc_zero_threshold') is None:
        st.session_state._preserved_non_bqc_zero_threshold = ZERO_FILTER_NON_BQC_DEFAULT
    if st.session_state.get('_preserved_bqc_zero_threshold') is None:
        st.session_state._preserved_bqc_zero_threshold = ZERO_FILTER_BQC_DEFAULT


_DIRECTION = (
    "Higher is looser: more missing values are allowed, so more lipids are kept. "
    "Lower is stricter."
)


def _threshold_sliders(has_bqc: bool, bqc_label: str) -> tuple:
    """The non-BQC and BQC sliders side by side; returns both as proportions."""
    col1, col2 = st.columns(2)

    with col1:
        non_bqc_pct = st.slider(
            "Non-BQC threshold: remove if missing in at least this % of samples, "
            "in every condition",
            min_value=50,
            max_value=100,
            value=st.session_state._preserved_non_bqc_zero_threshold,
            step=5,
            help=_DIRECTION,
            key="non_bqc_zero_threshold"
        )
        st.session_state._preserved_non_bqc_zero_threshold = non_bqc_pct
        non_bqc_threshold = non_bqc_pct / 100.0

    with col2:
        if has_bqc:
            bqc_pct = st.slider(
                f"BQC threshold: remove if missing in at least this % of {bqc_label} samples",
                min_value=25,
                max_value=100,
                value=st.session_state._preserved_bqc_zero_threshold,
                step=5,
                help=_DIRECTION,
                key="bqc_zero_threshold"
            )
            st.session_state._preserved_bqc_zero_threshold = bqc_pct
            bqc_threshold = bqc_pct / 100.0
        else:
            bqc_threshold = 0.5
            st.info("No BQC condition — only non-BQC threshold applies.")
    return non_bqc_threshold, bqc_threshold


def _detection_threshold_input() -> float:
    """The detection threshold, preserved across module switches."""
    def _save_detection_threshold():
        st.session_state._preserved_zero_filter_detection_threshold = st.session_state.zero_filter_detection_threshold

    return st.number_input(
        'Detection threshold (values ≤ this are considered zero)',
        min_value=0.0,
        value=st.session_state._preserved_zero_filter_detection_threshold,
        step=1.0,
        help="For non-LipidSearch formats, 0 means only exact zeros. Increase if your data has a noise floor.",
        key="zero_filter_detection_threshold",
        on_change=_save_detection_threshold
    )


def _show_filter_result(filter_result) -> None:
    """Say what the thresholds would remove, with the list to download."""
    removed_count = len(filter_result.removed_species)
    if removed_count > 0:
        st.warning(f"**Result:** Will remove {removed_count} species "
                   f"({filter_result.removal_percentage:.1f}% of dataset)")
        removed_df = pd.DataFrame({'LipidMolec': filter_result.removed_species})
        st.dataframe(removed_df, width='stretch', height=150)

        csv_download_button(removed_df, "removed_species.csv", key="download_removed_species")
    else:
        st.success("**Result:** No species will be removed")


_HOW_IT_WORKS = (
    "A lipid is removed when its values are at or below the detection threshold in "
    "too many samples. It is kept if at least one condition passes the non-BQC "
    "threshold and the BQC samples (if any) pass the BQC threshold. "
    "**Raising a threshold is looser** (more missing values are allowed, so more "
    "lipids are kept); **lowering it is stricter**."
)

_LOW_ABUNDANCE = (
    "**Low-abundance lipids:** loosening these settings keeps lipids measured near "
    "the noise floor. Their values are less precise, and a lipid seen in only one or "
    "two samples can't be compared reliably between conditions. If you loosen the "
    "filter, check those lipids' BQC CoV in Quality Check and treat results for them "
    "as exploratory."
)


def _show_explanation(experiment: ExperimentConfig, bqc_label, config: ZeroFilterConfig) -> None:
    """What the filter does, and what the current thresholds mean for this design."""
    st.markdown(_HOW_IT_WORKS)
    st.markdown(FilterGuidanceService.zero_filter_summary(experiment, bqc_label, config))
    st.markdown(_LOW_ABUNDANCE)


def _show_warnings(experiment: ExperimentConfig, bqc_label, config: ZeroFilterConfig,
                   default_detection: float) -> None:
    """Shown under the panel, so they're seen while it is collapsed."""
    single = FilterGuidanceService.single_detection_warning(
        experiment, bqc_label, config.non_bqc_threshold,
    )
    if single:
        st.warning(single)
    show_notices(FilterGuidanceService.zero_filter_notices(
        experiment, bqc_label, config, default_detection,
    ))


def display_zero_filtering_config(
    cleaned_df: pd.DataFrame,
    experiment: ExperimentConfig,
    bqc_label: str = None,
    data_format: str = None
) -> tuple:
    """
    Display zero filtering configuration with live preview.

    Args:
        cleaned_df: Cleaned DataFrame to filter.
        experiment: Experiment configuration with sample/condition info.
        bqc_label: Label for BQC condition (if any).
        data_format: Data format string (e.g., 'LipidSearch') for default detection threshold.

    Returns:
        tuple: (filtered_df, removed_species list, config dict)
    """
    has_bqc = bqc_label is not None and bqc_label in experiment.conditions_list
    default_detection = default_detection_threshold(data_format)

    with st.expander("⚙️ Configure Zero Filtering", expanded=False):
        if cleaned_df is None or cleaned_df.empty:
            st.error("No valid cleaned data available for zero filtering.")
            return None, [], {}

        st.markdown("Adjust thresholds for removing lipid species with too many zero/below-detection values.")
        explanation = st.container()

        _init_threshold_state(default_detection, data_format)
        non_bqc_threshold, bqc_threshold = _threshold_sliders(has_bqc, bqc_label)
        detection_threshold = _detection_threshold_input()

        # Apply filter with user-selected thresholds
        zero_config = ZeroFilterConfig(
            detection_threshold=detection_threshold,
            bqc_threshold=bqc_threshold,
            non_bqc_threshold=non_bqc_threshold
        )
        with explanation:
            _show_explanation(experiment, bqc_label if has_bqc else None, zero_config)

        try:
            filter_result = StreamlitAdapter.run_zero_filter(
                cleaned_df, experiment, zero_config, bqc_label,
            )
        except ValueError as e:
            logger.error("Zero filtering error: %s", e)
            st.error(f"Zero filtering failed: {e}")
            return cleaned_df, [], {}

        _show_filter_result(filter_result)

    _show_warnings(experiment, bqc_label if has_bqc else None, zero_config, default_detection)
    return filter_result.filtered_df, filter_result.removed_species, {
        'detection_threshold': detection_threshold,
        'bqc_threshold': bqc_threshold,
        'non_bqc_threshold': non_bqc_threshold
    }
