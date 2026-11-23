"""What the filters mean for a design, and the notices for settings changed
from their defaults (app.services.filter_guidance)."""
import pytest

from app.constants import LIPIDSEARCH_DETECTION_THRESHOLD
from app.models.experiment import ExperimentConfig
from app.services.filter_guidance import FilterGuidanceService as Guidance
from app.services.zero_filtering import ZeroFilterConfig


def _experiment(**sizes):
    return ExperimentConfig(
        n_conditions=len(sizes),
        conditions_list=list(sizes),
        number_of_samples_list=list(sizes.values()),
    )


THREE_REPLICATES = _experiment(WT=3, KO=3, BQC=4)
DEFAULTS = ZeroFilterConfig(detection_threshold=0.0, non_bqc_threshold=0.75, bqc_threshold=0.5)


def _settings(notices):
    return [n.setting for n in notices]


class TestZeroFilterSummary:

    def test_three_replicates_at_the_defaults(self):
        assert Guidance.zero_filter_summary(THREE_REPLICATES, 'BQC', DEFAULTS) == (
            "With the current settings (non-BQC 75%, BQC 50%) and 3 samples per "
            "condition, a lipid is kept if it is detected in **at least 1 of 3 "
            "replicates in any one condition**, and in **at least 3 of 4 BQC samples**."
        )

    def test_unequal_sample_counts_are_named(self):
        summary = Guidance.zero_filter_summary(_experiment(WT=3, KO=4, HFD=3), None, DEFAULTS)
        assert summary.startswith(
            "With the current settings (non-BQC 75%) and 3 samples in WT and HFD, "
            "4 samples in KO, a lipid is kept"
        )

    def test_unequal_replicates_are_spelled_out(self):
        experiment = _experiment(WT=3, KO=4, HFD=3)
        assert Guidance.detection_rule(experiment, None, 0.75) == (
            "at least 1 of 3 replicates of WT or HFD or 2 of 4 replicates of KO"
        )

    def test_no_bqc(self):
        summary = Guidance.zero_filter_summary(_experiment(WT=4, KO=4), None, DEFAULTS)
        assert summary.endswith("**at least 2 of 4 replicates in any one condition**.")


class TestSingleDetectionWarning:
    """The default keeps a lipid detected once in 3 replicates; the user
    chose to keep that behaviour and explain it."""

    def test_warns_at_the_default_with_three_replicates(self):
        warning = Guidance.single_detection_warning(THREE_REPLICATES, 'BQC', 0.75)
        assert "just one replicate of WT or KO" in warning
        assert "lower the non-BQC threshold" in warning

    def test_only_names_conditions_where_one_is_enough(self):
        warning = Guidance.single_detection_warning(_experiment(WT=3, KO=4), None, 0.75)
        assert "just one replicate of WT is kept" in warning

    @pytest.mark.parametrize("experiment, threshold", [
        (_experiment(WT=3, KO=3), 0.5),    # needs 2 of 3
        (_experiment(WT=4, KO=4), 0.75),   # needs 2 of 4
        (_experiment(WT=1, KO=1), 0.75),   # no replicates to speak of
    ])
    def test_silent_when_more_than_one_is_needed(self, experiment, threshold):
        assert Guidance.single_detection_warning(experiment, None, threshold) is None


class TestZeroFilterNotices:

    def test_defaults_have_none(self):
        assert Guidance.zero_filter_notices(THREE_REPLICATES, 'BQC', DEFAULTS, 0.0) == []
        lipidsearch = ZeroFilterConfig(detection_threshold=LIPIDSEARCH_DETECTION_THRESHOLD)
        assert Guidance.zero_filter_notices(
            THREE_REPLICATES, 'BQC', lipidsearch, LIPIDSEARCH_DETECTION_THRESHOLD,
        ) == []

    def test_lowered_detection_threshold_warns(self):
        config = ZeroFilterConfig(detection_threshold=10000)
        notice, = Guidance.zero_filter_notices(
            THREE_REPLICATES, 'BQC', config, LIPIDSEARCH_DETECTION_THRESHOLD,
        )
        assert notice.setting == "Detection threshold: 10,000 (default 30,000)"
        assert notice.loosened and "noise" in notice.effect

    def test_raised_detection_threshold_is_a_note(self):
        notice, = Guidance.zero_filter_notices(
            THREE_REPLICATES, 'BQC', ZeroFilterConfig(detection_threshold=500), 0.0,
        )
        assert notice.setting == "Detection threshold: 500 (default 0)"
        assert not notice.loosened

    def test_raised_non_bqc_threshold_warns_with_the_rule(self):
        config = ZeroFilterConfig(non_bqc_threshold=0.9)
        notice, = Guidance.zero_filter_notices(_experiment(WT=10, KO=10), None, config, 0.0)
        assert notice.setting == "Non-BQC zero threshold: 90% (default 75%)"
        assert notice.loosened
        assert "at least 2 of 10 replicates in any one condition" in notice.effect

    def test_lowered_non_bqc_threshold_is_a_note(self):
        config = ZeroFilterConfig(non_bqc_threshold=0.5)
        notice, = Guidance.zero_filter_notices(THREE_REPLICATES, 'BQC', config, 0.0)
        assert not notice.loosened
        assert "at least 2 of 3 replicates" in notice.effect

    def test_bqc_threshold_changes(self):
        raised = ZeroFilterConfig(bqc_threshold=0.8)
        notice, = Guidance.zero_filter_notices(THREE_REPLICATES, 'BQC', raised, 0.0)
        assert notice.setting == "BQC zero threshold: 80% (default 50%)"
        assert notice.loosened and "only 1 of 4 BQC samples" in notice.effect
        lowered = ZeroFilterConfig(bqc_threshold=0.25)
        notice, = Guidance.zero_filter_notices(THREE_REPLICATES, 'BQC', lowered, 0.0)
        assert not notice.loosened and "4 of 4 BQC samples" in notice.effect

    def test_bqc_threshold_ignored_without_a_bqc(self):
        config = ZeroFilterConfig(bqc_threshold=0.8)
        assert Guidance.zero_filter_notices(_experiment(WT=3, KO=3), None, config, 0.0) == []


class TestGradeNotices:

    def test_default_has_none(self):
        assert Guidance.grade_notices(None) == []
        # Customize by Class with every class left at its default.
        assert Guidance.grade_notices({'PC': ['A', 'B'], 'LPC': ['A', 'B', 'C']}) == []

    def test_c_and_d_allowed(self):
        notice, = Guidance.grade_notices({
            'PC': ['A', 'B', 'C'], 'PE': ['A', 'B', 'C'], 'SM': ['A', 'B', 'C', 'D'],
        })
        assert notice.setting.startswith("Grades allowed: C for PC, PE; D for SM")
        assert notice.loosened
        assert "rule of thumb" in notice.effect and "spectra" in notice.effect

    def test_narrowed(self):
        notice, = Guidance.grade_notices({'PC': ['A'], 'LPC': ['A', 'B'], 'TG': []})
        assert notice.setting.startswith("Grades narrowed for LPC, PC, TG")
        assert not notice.loosened


class TestQualityNotices:

    def test_default_has_none(self):
        assert Guidance.quality_notices(60, False, has_score=True) == []

    def test_lowered_score_warns(self):
        notice, = Guidance.quality_notices(40, False, has_score=True)
        assert notice.setting == "MS-DIAL score threshold: 40 (default 60)"
        assert notice.loosened

    def test_stricter_settings_are_notes(self):
        notices = Guidance.quality_notices(80, True, has_score=True)
        assert _settings(notices) == [
            "MS-DIAL score threshold: 80 (default 60)",
            "MS/MS match required (default not required)",
        ]
        assert not any(n.loosened for n in notices)

    def test_no_score_column_only_msms_counts(self):
        assert Guidance.quality_notices(0, False, has_score=False) == []
        assert _settings(Guidance.quality_notices(0, True, has_score=False)) == [
            "MS/MS match required (default not required)",
        ]


class TestCovNotices:

    def test_default_has_none(self):
        assert Guidance.cov_threshold_notices(30) == []
        assert Guidance.cov_kept_notices(30, 0) == []

    def test_raised_threshold_is_a_rule_of_thumb_warning(self):
        notice, = Guidance.cov_threshold_notices(40)
        assert notice.setting == "BQC CoV threshold: 40% (default 30%)"
        assert notice.loosened and "rule of thumb" in notice.effect

    def test_lowered_threshold_is_a_note(self):
        notice, = Guidance.cov_threshold_notices(20)
        assert not notice.loosened

    def test_lipids_kept_despite_high_cov(self):
        notice, = Guidance.cov_kept_notices(30, 3)
        assert notice.setting == "3 lipids kept despite CoV ≥ 30%"
        assert notice.loosened
        assert Guidance.cov_kept_notices(30, 1)[0].setting == "1 lipid kept despite CoV ≥ 30%"
