"""
Shared UI helpers for the analysis module.

Provides common widgets for statistical options, condition/class selection,
and detailed statistics tables.
"""

from typing import List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.lsi_report import AnalysisStatistics
from app.services.report_generator import ANALYSIS_PAGES
from app.services.statistical_testing import StatisticalTestSummary
from app.workflows.analysis import AnalysisWorkflow
from app.ui.st_helpers import section_header


def _skipped_tests_message(params: dict) -> Optional[str]:
    """Say how many tests were skipped, counting what was actually tested.

    Species-level results count lipids (n_lipids_total / n_lipids_skipped,
    with reasons); saturation results count class × fatty-acid-type tests
    (n_tests run, n_skipped).
    """
    if 'n_lipids_total' in params:
        n_skipped = params.get('n_lipids_skipped', 0)
        if not n_skipped:
            return None
        skip_parts = []
        if params.get('n_skipped_insufficient_data'):
            skip_parts.append(f"{params['n_skipped_insufficient_data']} had insufficient data (<2 values per group)")
        if params.get('n_skipped_all_zero'):
            skip_parts.append(f"{params['n_skipped_all_zero']} had all-zero values in one or both groups")
        if params.get('n_skipped_test_error'):
            skip_parts.append(f"{params['n_skipped_test_error']} failed the statistical test")
        msg = f"{n_skipped} of {params['n_lipids_total']} lipids were skipped during testing"
        if skip_parts:
            msg += f" ({'; '.join(skip_parts)})"
        return msg

    n_skipped = params.get('n_skipped', 0)
    if not n_skipped:
        return None
    total = params.get('n_tests', 0) + n_skipped
    return (
        f"{n_skipped} of {total} tests (lipid class × fatty-acid type) "
        "were skipped during testing"
    )


def _publish(key: str, figure) -> None:
    """Put a figure the page draws into the PDF report.

    The one way an analysis figure reaches the report, paired with
    ``_clear_pdf_plots`` at the top of each page. ``key`` must be one the
    report has a page for (``ANALYSIS_PAGES``), so a figure cannot be stored
    where the report never looks.
    """
    if not any(
        key == page.key or (page.key.endswith('_') and key.startswith(page.key))
        for page in ANALYSIS_PAGES
    ):
        raise ValueError(f"The PDF report has no page for {key!r}")
    st.session_state.analysis_all_plots[key] = figure


def _clear_statistics(analysis: str) -> None:
    """Drop an analysis's tests from the LSI report before it re-renders
    (the ``_clear_pdf_plots`` of ``_publish_statistics``)."""
    st.session_state.analysis_statistics.pop(analysis, None)


def _publish_statistics(
    analysis: str,
    summary: Optional[StatisticalTestSummary],
    significance: Optional[str] = None,
) -> None:
    """Put the tests an analysis ran into the LSI report.

    ``significance`` defaults to the test run's alpha, on adjusted p-values
    when a correction was applied. None ``summary`` (no tests run) records
    nothing.
    """
    if summary is None:
        return
    if significance is None:
        corrected = summary.test_info.get('correction', 'uncorrected') != 'uncorrected'
        alpha = summary.parameters.get('alpha', 0.05)
        significance = f"{'adjusted ' if corrected else ''}p ≤ {alpha:g}"
    st.session_state.analysis_statistics[analysis] = AnalysisStatistics(
        summary=summary, significance=significance,
    )


def _display_correction_scope(
    summary: Optional[StatisticalTestSummary],
    df: pd.DataFrame,
    selected_classes: List[str],
    unit: str,
) -> None:
    """Warn that a correction counted only the selected classes' tests.

    The multiple-testing family is what was tested, so leaving classes out
    makes the correction less strict and adjusted p-values move with the
    class selection.
    """
    if summary is None:
        return
    if summary.test_info.get('correction', 'uncorrected') == 'uncorrected':
        return
    n_available = len(AnalysisWorkflow.get_available_classes(df))
    if len(selected_classes) >= n_available:
        return
    st.warning(
        f"The multiple-testing correction covers only the {len(summary.results)} "
        f"{unit} tested here, from the {len(selected_classes)} of {n_available} "
        "lipid classes selected. Selecting other classes changes the correction, "
        "so adjusted p-values and significance can change with the class selection."
    )


def _clear_pdf_plots(*keys: str) -> None:
    """Drop an analysis's figures from the PDF report before it re-renders.

    Each analysis stores only the figures it actually draws, so a figure the
    user can no longer see (classes deselected, plot type switched, an early
    return) does not stay in the report. A key ending in '_' is a prefix.
    """
    plots = st.session_state.analysis_all_plots
    prefixes = tuple(k for k in keys if k.endswith('_'))
    for key in list(plots):
        if key in keys or (prefixes and key.startswith(prefixes)):
            del plots[key]


def _display_statistical_options(
    key_prefix: str,
    n_classes: int,
    n_conditions: int,
    conditions: Sequence[str] = (),
    family: str = "the selected classes",
    default_mode: str = "Auto",
    default_correction: str = "fdr_bh",
) -> StatisticalTestConfig:
    """Display statistical options panel (Auto/Manual mode).

    Args:
        key_prefix: Prefix for widget keys (e.g., 'bar', 'sat', 'volcano').
        n_classes: Number of selected lipid classes.
        n_conditions: Number of selected conditions.
        conditions: The selected conditions, offered as the control when
            Level 2 is Dunnett's test.
        family: What Level 1 corrects across, for the Auto description.
        default_mode: "Auto" or "Manual", the mode a new page starts in.
        default_correction: The Level 1 correction Manual starts with.

    Returns:
        Configured StatisticalTestConfig.
    """
    section_header("⚙️ Statistical Options")

    modes = ["Manual", "Auto"]
    mode = st.radio(
        "Select Analysis Mode:",
        modes,
        index=modes.index(default_mode),
        horizontal=True,
        key=f'{key_prefix}_stats_mode',
        help="Auto picks the test and both corrections from how many tests "
             "and conditions there are (described below when selected). "
             "Manual lets you choose each one.",
    )

    auto_transform = True
    if mode == "Auto":
        st.caption(
            "Auto: Welch's t-test for 2 conditions, the Alexander-Govern test "
            f"for 3 or more. Level 1: FDR (Benjamini-Hochberg) across {family}, "
            "or uncorrected when there is only one test. Level 2, with 3 or "
            "more conditions: Tukey's HSD, run only where the overall test is "
            "significant."
        )
        auto_transform = st.checkbox(
            "Auto-transform data (log10)",
            value=True,
            key=f'{key_prefix}_auto_transform',
        )
        return StatisticalTestConfig.create_auto(auto_transform=auto_transform)

    # Manual mode
    unit = 'Lipid' if key_prefix == 'panels' else 'Class'
    col1, col2 = st.columns(2)
    with col1:
        test_type = st.selectbox(
            "Statistical Test Type",
            ["parametric", "non_parametric"],
            index=0,
            key=f'{key_prefix}_test_type',
        )
        corrections = ["uncorrected", "fdr_bh", "bonferroni"]
        correction = st.selectbox(
            f"Between-{unit} Correction (Level 1)",
            corrections,
            index=corrections.index(default_correction),
            key=f'{key_prefix}_correction',
        )
    with col2:
        posthoc = st.selectbox(
            f"Within-{unit} Correction (Level 2)",
            ["uncorrected", "tukey", "bonferroni", "dunnett"],
            index=1,
            key=f'{key_prefix}_posthoc',
            help="dunnett: compares each condition with one control condition "
                 "instead of every pair (Dunnett's test; for non-parametric "
                 "tests, Mann-Whitney U against the control, Bonferroni-"
                 "corrected). Post-hoc tests run with 3 or more conditions.",
        )
        control = None
        if posthoc == "dunnett":
            control = st.selectbox(
                "Control condition", list(conditions),
                key=f'{key_prefix}_control',
                help="Every other condition is compared with this one.",
            )

    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,
        key=f'{key_prefix}_auto_transform',
    )

    # Settings summary
    st.markdown(
        f"**Current Settings:** Test: {test_type} | "
        f"Level 1: {correction} | Level 2: {posthoc}"
        f"{f' (control: {control})' if control else ''} | "
        f"Auto-transform: {'Yes' if auto_transform else 'No'}"
    )

    return StatisticalTestConfig.create_manual(
        test_type=test_type,
        correction_method=correction,
        posthoc_correction=posthoc,
        auto_transform=auto_transform,
        control_condition=control,
    )


def _warn_if_nothing_selected(
    selected_conditions: List[str], selected_classes: List[str],
) -> bool:
    """Warn and return True when a page's selectors leave nothing to analyse."""
    if not selected_conditions:
        st.warning("Please select at least one condition.")
        return True
    if not selected_classes:
        st.warning("Please select at least one lipid class.")
        return True
    return False


def _display_condition_class_selectors(
    experiment: ExperimentConfig,
    df: pd.DataFrame,
    key_prefix: str,
    conditions: Optional[List[str]] = None,
    bqc_label: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Display condition and class multiselect widgets.

    Args:
        experiment: Experiment configuration.
        df: DataFrame with ClassKey column.
        key_prefix: Prefix for widget keys.
        conditions: Conditions to offer. Defaults to those with two or more
            samples, which statistical tests need.
        bqc_label: Offered but not selected by default: BQC samples are a
            pooled technical control, not a group to compare.

    Returns:
        Tuple of (selected_conditions, selected_classes).
    """
    section_header("🎯 Data Selection")

    valid_conditions = (
        AnalysisWorkflow.get_eligible_conditions(experiment)
        if conditions is None else conditions
    )
    all_classes = AnalysisWorkflow.get_available_classes(df)

    col1, col2 = st.columns(2)
    with col1:
        selected_conditions = st.multiselect(
            "Conditions",
            valid_conditions,
            default=[c for c in valid_conditions if c != bqc_label],
            key=f'{key_prefix}_conditions',
        )
    with col2:
        selected_classes = st.multiselect(
            "Lipid Classes",
            all_classes,
            default=all_classes,
            key=f'{key_prefix}_classes',
        )

    return selected_conditions, selected_classes


def _display_detailed_statistics(
    stat_summary: Optional[StatisticalTestSummary],
    key_prefix: str,
) -> None:
    """Display detailed statistics table in a checkbox toggle.

    Args:
        stat_summary: Statistical test summary (None to skip).
        key_prefix: Prefix for widget keys.
    """
    if stat_summary is None:
        return

    section_header("🔍 Detailed Statistics")

    show = st.checkbox(
        "Show detailed statistical analysis",
        key=f'{key_prefix}_detailed_stats',
    )
    if not show:
        return

    # Context-appropriate column label for the group key
    group_col = {'bar': 'Class', 'sat': 'Class', 'volcano': 'Lipid', 'panels': 'Lipid'}.get(
        key_prefix, 'Group'
    )

    rows = []
    for group_key, result in stat_summary.results.items():
        rows.append({
            group_col: group_key,
            'Test': result.test_name,
            'Statistic': f"{result.statistic:.4f}",
            'p-value': f"{result.p_value:.4e}",
            'Adjusted p-value': (
                f"{result.adjusted_p_value:.4e}"
                if result.adjusted_p_value is not None
                else "N/A"
            ),
            'Significant': "Yes" if result.significant else "No",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

    # Report skipped items
    skipped_msg = _skipped_tests_message(stat_summary.parameters or {})
    if skipped_msg:
        st.info(skipped_msg)

    # Post-hoc results
    if stat_summary.posthoc_results:
        st.markdown("**Post-hoc Comparisons:**")
        posthoc_rows = []
        for group_key, comparisons in stat_summary.posthoc_results.items():
            for comp in comparisons:
                posthoc_rows.append({
                    group_col: group_key,
                    'Comparison': f"{comp.group1} vs {comp.group2}",
                    'Test': comp.test_name,
                    'p-value': f"{comp.p_value:.4e}",
                    'Adjusted p-value': (
                        f"{comp.adjusted_p_value:.4e}"
                        if comp.adjusted_p_value is not None
                        else "N/A"
                    ),
                    'Significant': "Yes" if comp.significant else "No",
                })
        if posthoc_rows:
            st.dataframe(pd.DataFrame(posthoc_rows), width='stretch')
