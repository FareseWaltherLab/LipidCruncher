"""
Zero filtering UI component for configuring and applying zero value filtering.
"""
import logging

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from app.constants import (
    FORMAT_LIPIDSEARCH,
    LIPIDSEARCH_DETECTION_THRESHOLD,
    ZERO_FILTER_NON_BQC_DEFAULT,
    ZERO_FILTER_BQC_DEFAULT,
)
from ..models.experiment import ExperimentConfig
from ..services.zero_filtering import ZeroFilteringService, ZeroFilterConfig
from ..ui.download_utils import csv_download_button


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
        data_format: Data format string (e.g., 'LipidSearch 5.0') for default detection threshold.

    Returns:
        tuple: (filtered_df, removed_species list, config dict)
    """
    has_bqc = bqc_label is not None and bqc_label in experiment.conditions_list

    # Determine default detection threshold based on format
    default_detection = LIPIDSEARCH_DETECTION_THRESHOLD if data_format == FORMAT_LIPIDSEARCH else 0.0

    with st.expander("⚙️ Configure Zero Filtering", expanded=False):
        if cleaned_df is None or cleaned_df.empty:
            st.error("No valid cleaned data available for zero filtering.")
            return None, [], {}

        st.markdown("Adjust thresholds for removing lipid species with too many zero/below-detection values.")

        # Initialize session state for persistence, reset if format changes
        if ('_preserved_zero_filter_detection_threshold' not in st.session_state
                or st.session_state.get('_zero_filter_format') != data_format):
            st.session_state._preserved_zero_filter_detection_threshold = default_detection
            st.session_state._zero_filter_format = data_format
        if st.session_state.get('_preserved_non_bqc_zero_threshold') is None:
            st.session_state._preserved_non_bqc_zero_threshold = ZERO_FILTER_NON_BQC_DEFAULT
        if st.session_state.get('_preserved_bqc_zero_threshold') is None:
            st.session_state._preserved_bqc_zero_threshold = ZERO_FILTER_BQC_DEFAULT
        if st.session_state.get('_preserved_zero_filter_detection_threshold') is None:
            st.session_state._preserved_zero_filter_detection_threshold = 0.0

        col1, col2 = st.columns(2)

        with col1:
            non_bqc_pct = st.slider(
                "Non-BQC threshold (%)",
                min_value=50,
                max_value=100,
                value=st.session_state._preserved_non_bqc_zero_threshold,
                step=5,
                help="Remove species if ALL non-BQC conditions have ≥ this % zeros",
                key="non_bqc_zero_threshold"
            )
            st.session_state._preserved_non_bqc_zero_threshold = non_bqc_pct
            non_bqc_threshold = non_bqc_pct / 100.0

        with col2:
            if has_bqc:
                bqc_pct = st.slider(
                    f"BQC threshold (%) — {bqc_label}",
                    min_value=25,
                    max_value=100,
                    value=st.session_state._preserved_bqc_zero_threshold,
                    step=5,
                    help="Remove species if BQC condition has ≥ this % zeros",
                    key="bqc_zero_threshold"
                )
                st.session_state._preserved_bqc_zero_threshold = bqc_pct
                bqc_threshold = bqc_pct / 100.0
            else:
                bqc_threshold = 0.5
                st.info("No BQC condition — only non-BQC threshold applies.")

        # Callback to save detection threshold
        def _save_detection_threshold():
            st.session_state._preserved_zero_filter_detection_threshold = st.session_state.zero_filter_detection_threshold

        # Detection threshold input (below sliders)
        detection_threshold = st.number_input(
            'Detection threshold (values ≤ this are considered zero)',
            min_value=0.0,
            value=st.session_state._preserved_zero_filter_detection_threshold,
            step=1.0,
            help="For non-LipidSearch formats, 0 means only exact zeros. Increase if your data has a noise floor.",
            key="zero_filter_detection_threshold",
            on_change=_save_detection_threshold
        )

        # Apply filter with user-selected thresholds
        zero_config = ZeroFilterConfig(
            detection_threshold=detection_threshold,
            bqc_threshold=bqc_threshold,
            non_bqc_threshold=non_bqc_threshold
        )

        try:
            filter_result = ZeroFilteringService.filter_zeros(
                df=cleaned_df,
                experiment=experiment,
                config=zero_config,
                bqc_label=bqc_label
            )

            removed_count = len(filter_result.removed_species)
            if removed_count > 0:
                st.warning(f"**Result:** Will remove {removed_count} species "
                          f"({filter_result.removal_percentage:.1f}% of dataset)")
                removed_df = pd.DataFrame({'LipidMolec': filter_result.removed_species})
                st.dataframe(removed_df, use_container_width=True, height=150)

                csv_download_button(removed_df, "removed_species.csv", key="download_removed_species")
            else:
                st.success("**Result:** No species will be removed")

            return filter_result.filtered_df, filter_result.removed_species, {
                'detection_threshold': detection_threshold,
                'bqc_threshold': bqc_threshold,
                'non_bqc_threshold': non_bqc_threshold
            }

        except ValueError as e:
            logger.error("Zero filtering error: %s", e)
            st.error(f"Zero filtering failed: {e}")
            return cleaned_df, [], {}
