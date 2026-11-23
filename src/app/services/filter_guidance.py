"""
What the post-upload filters mean for a dataset, and what changing them
from their defaults does to the reliability of the results.

Each ``*_notices`` function returns one ``FilterNotice`` per setting that
differs from LipidCruncher's default: the UI shows them under the filter's
panel, and the PDF and LSI reports list their ``setting`` lines. The
thresholds behind the defaults are rules of thumb from experience, not
standards, and the wording says so.

Pure logic — no Streamlit dependencies. All functions are static.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.constants import (
    COV_THRESHOLD_DEFAULT,
    LIPIDSEARCH_DEFAULT_GRADES,
    LIPIDSEARCH_RELAXED_GRADE_CLASSES,
    LIPIDSEARCH_RELAXED_GRADES,
    MSDIAL_DEFAULT_QUALITY_LEVEL,
    MSDIAL_QUALITY_PRESETS,
    ZERO_FILTER_BQC_DEFAULT,
    ZERO_FILTER_NON_BQC_DEFAULT,
)
from app.models.experiment import ExperimentConfig
from app.services.zero_filtering import ZeroFilteringService, ZeroFilterConfig

_MSDIAL_DEFAULT = MSDIAL_QUALITY_PRESETS[MSDIAL_DEFAULT_QUALITY_LEVEL]


@dataclass(frozen=True)
class FilterNotice:
    """A filter setting that differs from its default.

    Attributes:
        setting: What was changed, with the default, e.g.
            "Non-BQC zero threshold: 90% (default 75%)".
        effect: What the change does to the results.
        loosened: True when it keeps lipids the default would remove (shown
            as a warning); False when it is stricter (shown as a note).
    """
    setting: str
    effect: str
    loosened: bool


def _pct(proportion: float) -> int:
    return round(proportion * 100)


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,}"


def _join(items: Sequence[str], word: str = "and") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {word} {items[-1]}"


def _non_bqc_conditions(
    experiment: ExperimentConfig, bqc_label: Optional[str],
) -> List[Tuple[str, int]]:
    return [
        (condition, n)
        for condition, n in zip(experiment.conditions_list, experiment.number_of_samples_list)
        if condition != bqc_label and n > 0
    ]


def _bqc_size(experiment: ExperimentConfig, bqc_label: Optional[str]) -> int:
    if bqc_label not in experiment.conditions_list:
        return 0
    return experiment.number_of_samples_list[experiment.conditions_list.index(bqc_label)]


class FilterGuidanceService:
    """Explanations and non-default notices for the post-upload filters."""

    # ── Zero filter ─────────────────────────────────────────────────────

    @staticmethod
    def detection_rule(
        experiment: ExperimentConfig, bqc_label: Optional[str], non_bqc_threshold: float,
    ) -> str:
        """Where a lipid must be detected to pass the non-BQC threshold, e.g.
        "at least 1 of 3 replicates in any one condition"."""
        groups: Dict[Tuple[int, int], List[str]] = {}
        for condition, n in _non_bqc_conditions(experiment, bqc_label):
            k = ZeroFilteringService.detections_needed(n, non_bqc_threshold)
            groups.setdefault((k, n), []).append(condition)
        if len(groups) == 1:
            (k, n), = groups
            return f"at least {k} of {n} replicates in any one condition"
        return "at least " + _join([
            f"{k} of {n} replicates of {_join(conditions, 'or')}"
            for (k, n), conditions in groups.items()
        ], "or")

    @staticmethod
    def zero_filter_summary(
        experiment: ExperimentConfig, bqc_label: Optional[str], config: ZeroFilterConfig,
    ) -> str:
        """The rule the current thresholds apply to this design, as a sentence."""
        rule = FilterGuidanceService.detection_rule(
            experiment, bqc_label, config.non_bqc_threshold,
        )
        sentence = f"With your design, a lipid is kept if it is detected in **{rule}**"
        n_bqc = _bqc_size(experiment, bqc_label)
        if n_bqc:
            k = ZeroFilteringService.detections_needed(n_bqc, config.bqc_threshold)
            sentence += f", and in **at least {k} of {n_bqc} {bqc_label} samples**"
        return sentence + "."

    @staticmethod
    def single_detection_warning(
        experiment: ExperimentConfig, bqc_label: Optional[str], non_bqc_threshold: float,
    ) -> Optional[str]:
        """A warning when one detection in a condition with replicates is
        enough to keep a lipid (the default with 3 replicates), else None."""
        conditions = [
            condition for condition, n in _non_bqc_conditions(experiment, bqc_label)
            if n > 1 and ZeroFilteringService.detections_needed(n, non_bqc_threshold) == 1
        ]
        if not conditions:
            return None
        return (
            f"**A single detection keeps a lipid.** At the {_pct(non_bqc_threshold)}% "
            f"non-BQC threshold, a lipid detected in just one replicate of "
            f"{_join(conditions, 'or')} is kept even if every other non-BQC sample is "
            "zero. Its mean for that condition then rests on one value, and statistics on "
            "it are weak. To require more detections, lower the non-BQC threshold under "
            "Configure Zero Filtering; otherwise check such lipids before reporting them."
        )

    @staticmethod
    def zero_filter_notices(
        experiment: ExperimentConfig,
        bqc_label: Optional[str],
        config: ZeroFilterConfig,
        default_detection: float,
    ) -> List[FilterNotice]:
        """One notice per zero-filter setting that differs from its default."""
        notices = []
        threshold = config.detection_threshold
        if threshold != default_detection:
            setting = (f"Detection threshold: {_number(threshold)} "
                       f"(default {_number(default_detection)})")
            if threshold < default_detection:
                notices.append(FilterNotice(setting, (
                    f"Values below {_number(default_detection)} are often instrument noise "
                    "(the default is a rule of thumb for this format). A lower threshold keeps "
                    "more low-abundance lipids, but some of what counts as detected may be "
                    "noise. If you know your instrument's noise floor, use that."
                ), loosened=True))
            else:
                notices.append(FilterNotice(setting, (
                    f"Values at or below {_number(threshold)} count as not detected, so lipids "
                    "measured mostly below it are removed. That's right if it's your "
                    "instrument's noise floor; otherwise real low-abundance lipids are lost."
                ), loosened=False))

        non_bqc = _pct(config.non_bqc_threshold)
        if non_bqc != ZERO_FILTER_NON_BQC_DEFAULT:
            rule = FilterGuidanceService.detection_rule(
                experiment, bqc_label, config.non_bqc_threshold,
            )
            setting = f"Non-BQC zero threshold: {non_bqc}% (default {ZERO_FILTER_NON_BQC_DEFAULT}%)"
            if non_bqc > ZERO_FILTER_NON_BQC_DEFAULT:
                notices.append(FilterNotice(setting, (
                    f"At {non_bqc}%, a lipid is kept if it is detected in {rule}. "
                    "Means and statistics built mostly on zeros are unreliable."
                ), loosened=True))
            else:
                notices.append(FilterNotice(setting, (
                    f"At {non_bqc}%, a lipid must be detected in {rule}. A real but "
                    "low-abundance lipid seen less often than that in every condition is "
                    "removed. Check the list of removed lipids."
                ), loosened=False))

        n_bqc = _bqc_size(experiment, bqc_label)
        bqc = _pct(config.bqc_threshold)
        if n_bqc and bqc != ZERO_FILTER_BQC_DEFAULT:
            k = ZeroFilteringService.detections_needed(n_bqc, config.bqc_threshold)
            setting = f"BQC zero threshold: {bqc}% (default {ZERO_FILTER_BQC_DEFAULT}%)"
            if bqc > ZERO_FILTER_BQC_DEFAULT:
                notices.append(FilterNotice(setting, (
                    f"At {bqc}%, a lipid needs only {k} of {n_bqc} {bqc_label} samples. The "
                    "BQC is a pool of every sample, so a lipid that's really present should "
                    "show up in it. Keeping lipids the BQC mostly misses keeps signals that "
                    "may not be reproducible."
                ), loosened=True))
            else:
                notices.append(FilterNotice(setting, (
                    f"At {bqc}%, a lipid must be detected in {k} of {n_bqc} {bqc_label} "
                    "samples. Low-abundance lipids the BQC picks up only sometimes are removed."
                ), loosened=False))
        return notices

    # ── LipidSearch grades ──────────────────────────────────────────────

    @staticmethod
    def grade_notices(grade_config: Optional[Dict[str, List[str]]]) -> List[FilterNotice]:
        """Notices for a LipidSearch grade config (class -> grades kept);
        None is the default."""
        if not grade_config:
            return []
        allows_c, allows_d, narrowed = [], [], []
        for lipid_class, grades in sorted(grade_config.items()):
            default = (LIPIDSEARCH_RELAXED_GRADES
                       if lipid_class in LIPIDSEARCH_RELAXED_GRADE_CLASSES
                       else LIPIDSEARCH_DEFAULT_GRADES)
            if 'C' in grades and 'C' not in default:
                allows_c.append(lipid_class)
            if 'D' in grades:
                allows_d.append(lipid_class)
            if set(default) - set(grades):
                narrowed.append(lipid_class)

        default_text = (f"default {'/'.join(LIPIDSEARCH_DEFAULT_GRADES)}, plus C for "
                        f"{_join(LIPIDSEARCH_RELAXED_GRADE_CLASSES)}")
        notices = []
        loosened = [f"C for {', '.join(allows_c)}"] if allows_c else []
        loosened += [f"D for {', '.join(allows_d)}"] if allows_d else []
        if loosened:
            notices.append(FilterNotice(
                f"Grades allowed: {'; '.join(loosened)} ({default_text})",
                "Grade C and D identifications are more often wrong (wrong class, wrong "
                "chains, or an isomer). As a rule of thumb, check any C or D lipid in your "
                "results against its spectra before reporting it.",
                loosened=True,
            ))
        if narrowed:
            notices.append(FilterNotice(
                f"Grades narrowed for {', '.join(narrowed)} ({default_text})",
                "Fewer lipids are kept: correctly identified lipids with the dropped "
                "grades are removed too.",
                loosened=False,
            ))
        return notices

    # ── MS-DIAL quality ─────────────────────────────────────────────────

    @staticmethod
    def quality_notices(
        total_score_threshold: float, require_msms: bool, has_score: bool,
    ) -> List[FilterNotice]:
        """Notices for MS-DIAL's quality filter. Without a Total score column
        only the MS/MS requirement applies."""
        notices = []
        default_score = _MSDIAL_DEFAULT['total_score_threshold']
        if has_score and total_score_threshold != default_score:
            setting = (f"MS-DIAL score threshold: {_number(total_score_threshold)} "
                       f"(default {default_score})")
            if total_score_threshold < default_score:
                notices.append(FilterNotice(setting, (
                    "Identifications with a low score are tentative. A lower threshold keeps "
                    "more lipids and more misidentifications."
                ), loosened=True))
            else:
                notices.append(FilterNotice(setting, (
                    "Fewer lipids are kept: some correct identifications score below this."
                ), loosened=False))
        if require_msms != _MSDIAL_DEFAULT['require_msms']:
            notices.append(FilterNotice(
                "MS/MS match required (default not required)",
                "Only lipids with a matched MS/MS spectrum are kept. Identifications are "
                "more certain, but lipids too low in abundance to fragment well are removed.",
                loosened=False,
            ))
        return notices

    # ── BQC CoV ─────────────────────────────────────────────────────────

    @staticmethod
    def cov_threshold_notices(cov_threshold: float) -> List[FilterNotice]:
        """A notice for a BQC CoV threshold other than the default."""
        if cov_threshold == COV_THRESHOLD_DEFAULT:
            return []
        setting = f"BQC CoV threshold: {_number(cov_threshold)}% (default {COV_THRESHOLD_DEFAULT}%)"
        if cov_threshold > COV_THRESHOLD_DEFAULT:
            return [FilterNotice(setting, (
                f"{COV_THRESHOLD_DEFAULT}% is a common rule of thumb, not a fixed standard. "
                "Well above it, a lipid's technical noise can be large enough to hide or "
                "fake differences between conditions."
            ), loosened=True)]
        return [FilterNotice(setting, (
            "Fewer lipids count as reliable, and filtering on it removes more."
        ), loosened=False)]

    @staticmethod
    def cov_kept_notices(cov_threshold: float, kept_count: int) -> List[FilterNotice]:
        """A notice for lipids the CoV filter would remove but the user kept."""
        if not kept_count:
            return []
        return [FilterNotice(
            f"{kept_count} lipid{'s' if kept_count != 1 else ''} kept despite "
            f"CoV ≥ {_number(cov_threshold)}%",
            "Differences between conditions are less trustworthy for these lipids "
            "than for the rest. Say so if you report them.",
            loosened=True,
        )]
