"""
Shared UI helpers for the analysis module.

Provides common widgets for statistical options, condition/class selection,
and detailed statistics tables.
"""

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.statistical_testing import StatisticalTestSummary
from app.workflows.analysis import AnalysisWorkflow


def _display_statistical_options(
    key_prefix: str,
    n_classes: int,
    n_conditions: int,
) -> StatisticalTestConfig:
    """Display statistical options panel (Auto/Manual mode).

    Args:
        key_prefix: Prefix for widget keys (e.g., 'bar', 'sat', 'volcano').
        n_classes: Number of selected lipid classes.
        n_conditions: Number of selected conditions.

    Returns:
        Configured StatisticalTestConfig.
    """
    st.markdown("---")
    st.markdown("#### ⚙️ Statistical Options")

    mode = st.radio(
        "Select Analysis Mode:",
        ["Manual", "Auto"],
        index=1,
        horizontal=True,
        key=f'{key_prefix}_stats_mode',
    )

    auto_transform = True
    if mode == "Auto":
        auto_transform = st.checkbox(
            "Auto-transform data (log10)",
            value=True,
            key=f'{key_prefix}_auto_transform',
        )
        return StatisticalTestConfig.create_auto(auto_transform=auto_transform)

    # Manual mode
    col1, col2 = st.columns(2)
    with col1:
        test_type = st.selectbox(
            "Statistical Test Type",
            ["parametric", "non_parametric"],
            index=0,
            key=f'{key_prefix}_test_type',
        )
        correction = st.selectbox(
            "Between-Class Correction (Level 1)",
            ["uncorrected", "fdr_bh", "bonferroni"],
            index=1,
            key=f'{key_prefix}_correction',
        )
    with col2:
        posthoc = st.selectbox(
            "Within-Class Correction (Level 2)",
            ["uncorrected", "tukey", "bonferroni"],
            index=1,
            key=f'{key_prefix}_posthoc',
        )

    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,
        key=f'{key_prefix}_auto_transform',
    )

    # Settings summary
    st.markdown(
        f"**Current Settings:** Test: {test_type} | "
        f"Level 1: {correction} | Level 2: {posthoc} | "
        f"Auto-transform: {'Yes' if auto_transform else 'No'}"
    )

    return StatisticalTestConfig.create_manual(
        test_type=test_type,
        correction_method=correction,
        posthoc_correction=posthoc,
        auto_transform=auto_transform,
    )


def _display_condition_class_selectors(
    experiment: ExperimentConfig,
    df: pd.DataFrame,
    key_prefix: str,
) -> Tuple[List[str], List[str]]:
    """Display condition and class multiselect widgets.

    Args:
        experiment: Experiment configuration.
        df: DataFrame with ClassKey column.
        key_prefix: Prefix for widget keys.

    Returns:
        Tuple of (selected_conditions, selected_classes).
    """
    st.markdown("---")
    st.markdown("#### 🎯 Data Selection")

    valid_conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
    all_classes = AnalysisWorkflow.get_available_classes(df)

    col1, col2 = st.columns(2)
    with col1:
        selected_conditions = st.multiselect(
            "Conditions",
            valid_conditions,
            default=valid_conditions,
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

    st.markdown("---")
    st.markdown("#### 🔍 Detailed Statistics")

    show = st.checkbox(
        "Show detailed statistical analysis",
        key=f'{key_prefix}_detailed_stats',
    )
    if not show:
        return

    rows = []
    for group_key, result in stat_summary.results.items():
        rows.append({
            'Group': group_key,
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Post-hoc results
    if stat_summary.posthoc_results:
        st.markdown("**Post-hoc Comparisons:**")
        posthoc_rows = []
        for group_key, comparisons in stat_summary.posthoc_results.items():
            for comp in comparisons:
                posthoc_rows.append({
                    'Group': group_key,
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
            st.dataframe(pd.DataFrame(posthoc_rows), use_container_width=True)
