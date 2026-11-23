"""
Filter settings changed from their defaults: shown under each filter's panel
(so they're visible while it is collapsed) and listed in the PDF and LSI
reports. The wording comes from ``FilterGuidanceService``.
"""
from typing import List, Optional

import streamlit as st

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, LIPIDSEARCH_DETECTION_THRESHOLD, ZERO_FILTER_BQC_DEFAULT,
)
from app.models.experiment import ExperimentConfig
from app.services.filter_guidance import FilterGuidanceService, FilterNotice
from app.services.zero_filtering import ZeroFilterConfig


def show_notices(notices: List[FilterNotice]) -> None:
    """A warning for each loosened setting, a note for each stricter one."""
    for notice in notices:
        show = st.warning if notice.loosened else st.info
        show(f"**{notice.setting}.** {notice.effect}")


def default_detection_threshold(data_format: Optional[str]) -> float:
    """The zero filter's default detection threshold for a format."""
    return LIPIDSEARCH_DETECTION_THRESHOLD if data_format == FORMAT_LIPIDSEARCH else 0.0


def filter_changes(
    experiment: ExperimentConfig, bqc_label: Optional[str], data_format: Optional[str],
) -> List[str]:
    """Every filter setting the session changed from its default, one line
    each, for the reports."""
    state = st.session_state
    has_bqc = bqc_label is not None and bqc_label in experiment.conditions_list
    notices: List[FilterNotice] = []

    if data_format == FORMAT_LIPIDSEARCH:
        grade_config = state.get('grade_config')
        notices += FilterGuidanceService.grade_notices(
            grade_config.grade_config if grade_config is not None else None,
        )
    quality_config = state.get('last_quality_config')
    if data_format == FORMAT_MSDIAL and quality_config is not None:
        notices += FilterGuidanceService.quality_notices(
            quality_config.total_score_threshold, quality_config.require_msms,
            has_score=getattr(state.get('msdial_features'), 'has_quality_score', False),
        )

    non_bqc = state.get('_preserved_non_bqc_zero_threshold')
    if non_bqc is not None:
        default_detection = default_detection_threshold(data_format)
        detection = state.get('_preserved_zero_filter_detection_threshold')
        bqc = state.get('_preserved_bqc_zero_threshold')
        notices += FilterGuidanceService.zero_filter_notices(
            experiment, bqc_label if has_bqc else None,
            ZeroFilterConfig(
                detection_threshold=default_detection if detection is None else detection,
                non_bqc_threshold=non_bqc / 100,
                bqc_threshold=(ZERO_FILTER_BQC_DEFAULT if bqc is None else bqc) / 100,
            ),
            default_detection,
        )

    cov_threshold = state.get('qc_cov_threshold')
    if has_bqc and cov_threshold is not None:
        notices += FilterGuidanceService.cov_threshold_notices(cov_threshold)
        if state.get('_preserved_bqc_filter_choice') == "Yes":
            notices += FilterGuidanceService.cov_kept_notices(
                cov_threshold, len(state.get('bqc_lipids_to_keep') or []),
            )
    return [notice.setting for notice in notices]
