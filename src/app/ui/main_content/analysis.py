"""
Module 3: Visualize and Analyze UI.

Provides 7 analysis features accessed via radio selector:
1. Abundance Bar Chart (class level)
2. Abundance Pie Charts (class level)
3. Saturation Plots (class level, requires detailed FA)
4. Fatty Acid Composition Heatmaps (class level)
5. Pathway Visualization (class level, requires detailed FA)
6. Volcano Plot (species level)
7. Lipidomic Heatmap (species level)

Each feature is a private _display_* function called from the main entry point.
Shared helpers handle statistical options, condition/class selectors, and stats tables.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.statistical_testing import StatisticalTestSummary
from app.services.plotting.saturation_plot import SaturationPlotterService
from app.services.plotting.volcano_plot import VolcanoPlotterService
from app.services.plotting.lipidomic_heatmap import LipidomicHeatmapPlotterService
from app.workflows.analysis import AnalysisWorkflow
from app.ui.download_utils import (
    plotly_svg_download_button,
    matplotlib_svg_download_button,
    csv_download_button,
)


# ═══════════════════════════════════════════════════════════════════════
# Analysis Radio Options
# ═══════════════════════════════════════════════════════════════════════

ANALYSIS_OPTIONS = [
    "Class Level Breakdown - Bar Chart",
    "Class Level Breakdown - Pie Charts",
    "Class Level Breakdown - Saturation Plots (requires detailed fatty acid composition)",
    "Class Level Breakdown - Fatty Acid Composition Heatmaps",
    "Class Level Breakdown - Pathway Visualization (requires detailed fatty acid composition)",
    "Species Level Breakdown - Volcano Plot",
    "Species Level Breakdown - Lipidomic Heatmap",
]


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════


def display_analysis_module(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    bqc_label: Optional[str],
    format_type: str,
) -> None:
    """Display Module 3: Visualize and Analyze.

    Args:
        df: DataFrame with concentration[sample] columns, LipidMolec, ClassKey.
        experiment: Experiment configuration.
        bqc_label: BQC condition label, or None.
        format_type: Data format string.
    """
    st.subheader("Visualize and Analyze")

    errors = AnalysisWorkflow.validate_inputs(df, experiment)
    if errors:
        for err in errors:
            st.error(err)
        return

    analysis_type = st.radio(
        "Select Analysis",
        ANALYSIS_OPTIONS,
        key='analysis_radio',
    )

    if analysis_type == ANALYSIS_OPTIONS[0]:
        _display_bar_chart(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[1]:
        _display_pie_charts(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[2]:
        _display_saturation_plots(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[3]:
        _display_fach_heatmaps(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[4]:
        _display_pathway_viz(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[5]:
        _display_volcano_plot(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[6]:
        _display_lipidomic_heatmap(df, experiment)


# ═══════════════════════════════════════════════════════════════════════
# Shared UI Helpers
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
# Feature 1: Abundance Bar Chart
# ═══════════════════════════════════════════════════════════════════════


def _display_bar_chart(df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    """Display abundance bar chart analysis."""
    with st.expander("Class Concentration Bar Chart", expanded=True):
        st.markdown(
            "Visualize the total abundance of each lipid class across conditions."
        )

        selected_conditions, selected_classes = _display_condition_class_selectors(
            experiment, df, 'bar',
        )

        if not selected_conditions:
            st.warning("Please select at least one condition.")
            return
        if not selected_classes:
            st.warning("Please select at least one lipid class.")
            return

        stat_config = _display_statistical_options(
            'bar', len(selected_classes), len(selected_conditions),
        )

        st.markdown("---")
        st.markdown("#### 📊 Results")

        scale = st.radio(
            "Select scale:",
            ["Log10 Scale", "Linear Scale"],
            index=0,
            horizontal=True,
            key='bar_scale_radio',
        )
        scale_value = 'log10' if scale == "Log10 Scale" else 'linear'

        with st.spinner("Generating bar chart..."):
            result = AnalysisWorkflow.run_bar_chart(
                df, experiment, selected_conditions, selected_classes,
                stat_config=stat_config, scale=scale_value,
            )

        st.plotly_chart(result.figure, use_container_width=True)
        st.session_state.analysis_bar_chart_fig = result.figure
        st.session_state.analysis_all_plots['bar_chart'] = result.figure

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                result.figure,
                f"abundance_bar_chart_{scale_value}.svg",
                key="analysis_svg_bar",
            )
        with col2:
            csv_download_button(
                result.abundance_df,
                f"abundance_bar_chart_{scale_value}.csv",
                key="analysis_csv_bar",
            )

        _display_detailed_statistics(result.stat_summary, 'bar')


# ═══════════════════════════════════════════════════════════════════════
# Feature 2: Abundance Pie Charts
# ═══════════════════════════════════════════════════════════════════════


def _display_pie_charts(df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    """Display abundance pie chart analysis."""
    with st.expander("Class Concentration Pie Chart", expanded=True):
        st.markdown(
            "View the proportional distribution of lipid classes per condition."
        )

        all_classes = AnalysisWorkflow.get_available_classes(df)
        all_conditions = AnalysisWorkflow.get_all_conditions(experiment)

        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")

        selected_classes = st.multiselect(
            "Lipid Classes",
            all_classes,
            default=all_classes,
            key='pie_classes',
        )

        if not selected_classes:
            st.warning("Please select at least one lipid class to create the pie charts.")
            return

        st.markdown("---")
        st.markdown("#### 📊 Results")

        with st.spinner("Generating pie charts..."):
            results = AnalysisWorkflow.run_pie_charts(
                df, experiment, all_conditions, selected_classes,
            )

        st.session_state.analysis_pie_chart_figs = {}
        for condition, pie_result in results.items():
            st.markdown(f"###### {condition}")
            st.plotly_chart(pie_result.figure, use_container_width=True)
            st.session_state.analysis_pie_chart_figs[condition] = pie_result.figure
            st.session_state.analysis_all_plots[f'pie_{condition}'] = pie_result.figure

            col1, col2 = st.columns(2)
            with col1:
                plotly_svg_download_button(
                    pie_result.figure,
                    f"abundance_pie_chart_{condition}.svg",
                    key=f"analysis_svg_pie_{condition}",
                )
            with col2:
                csv_download_button(
                    pie_result.data_df,
                    f"abundance_pie_chart_{condition}.csv",
                    key=f"analysis_csv_pie_{condition}",
                )


# ═══════════════════════════════════════════════════════════════════════
# Feature 3: Saturation Plots
# ═══════════════════════════════════════════════════════════════════════


def _display_saturation_plots(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display SFA/MUFA/PUFA saturation analysis."""
    with st.expander("Class Level Breakdown - Saturation Plots", expanded=True):
        st.markdown(
            "Analyze the distribution of saturated (SFA), monounsaturated (MUFA), "
            "and polyunsaturated (PUFA) fatty acids per lipid class."
        )

        # Check for detailed FA composition
        _check_fa_compatibility(df)

        selected_conditions, selected_classes = _display_condition_class_selectors(
            experiment, df, 'sat',
        )

        if not selected_conditions:
            st.warning("Please select at least one condition.")
            return
        if not selected_classes:
            st.warning("Please select at least one lipid class.")
            return

        # Consolidated lipid handling
        excluded_lipids = _display_consolidated_lipids(df, selected_classes, 'sat')

        stat_config = _display_statistical_options(
            'sat', len(selected_classes), len(selected_conditions),
        )

        st.markdown("---")
        st.markdown("#### 📊 Results")

        col1, col2 = st.columns(2)
        with col1:
            plot_type = st.radio(
                "Plot type:",
                ["Concentration", "Percentage", "Both"],
                index=0,
                horizontal=True,
                key='sat_plot_type',
            )
        with col2:
            show_sig = st.checkbox(
                "Show significance asterisks",
                value=False,
                key='sat_show_significance',
            )

        # Filter excluded lipids
        working_df = df
        if excluded_lipids:
            working_df = df[~df['LipidMolec'].isin(excluded_lipids)].copy()

        # Generate plots for each plot type needed
        st.session_state.analysis_saturation_figs = {}
        _render_saturation_results(
            working_df, experiment, selected_conditions, selected_classes,
            stat_config, plot_type, show_sig,
        )


def _render_saturation_results(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    selected_conditions: List[str],
    selected_classes: List[str],
    stat_config: StatisticalTestConfig,
    plot_type: str,
    show_sig: bool,
) -> None:
    """Render saturation plots for the selected plot type(s)."""
    types_to_render = []
    if plot_type in ("Concentration", "Both"):
        types_to_render.append('concentration')
    if plot_type in ("Percentage", "Both"):
        types_to_render.append('percentage')

    stat_summary = None
    for ptype in types_to_render:
        with st.spinner(f"Generating {ptype} plots..."):
            result = AnalysisWorkflow.run_saturation(
                df, experiment, selected_conditions, selected_classes,
                stat_config=stat_config if ptype == 'concentration' else None,
                plot_type=ptype,
                show_significance=show_sig,
            )

        if ptype == 'concentration':
            stat_summary = result.stat_summary

        for lipid_class, fig in result.plots.items():
            st.markdown(f"###### {lipid_class}")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.analysis_saturation_figs[
                f"{lipid_class}_{ptype}"
            ] = fig
            st.session_state.analysis_all_plots[
                f'sat_{ptype}_{lipid_class}'
            ] = fig

            col1, col2 = st.columns(2)
            with col1:
                plotly_svg_download_button(
                    fig,
                    f"saturation_{ptype}_{lipid_class}.svg",
                    key=f"analysis_svg_sat_{ptype}_{lipid_class}",
                )
            with col2:
                # Build CSV data from the saturation data
                csv_download_button(
                    _build_saturation_csv(
                        df, experiment, selected_conditions, lipid_class
                    ),
                    f"saturation_{ptype}_{lipid_class}.csv",
                    key=f"analysis_csv_sat_{ptype}_{lipid_class}",
                )

    _display_detailed_statistics(stat_summary, 'sat')


def _build_saturation_csv(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    selected_conditions: List[str],
    lipid_class: str,
) -> pd.DataFrame:
    """Build a CSV-friendly DataFrame for saturation data."""
    sat_data = SaturationPlotterService.calculate_sfa_mufa_pufa(
        df, experiment, selected_conditions, [lipid_class],
    )
    if lipid_class in sat_data.plot_data:
        return sat_data.plot_data[lipid_class]
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# Feature 4: FACH Heatmaps
# ═══════════════════════════════════════════════════════════════════════


def _display_fach_heatmaps(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display Fatty Acid Composition Heatmap analysis."""
    with st.expander(
        "Class Level Breakdown - Fatty Acid Composition Heatmaps", expanded=True
    ):
        st.markdown(
            "Visualize the distribution of fatty acid chain lengths and "
            "double bonds within a lipid class across conditions."
        )

        _check_fa_compatibility(df)

        all_classes = AnalysisWorkflow.get_available_classes(df)
        valid_conditions = AnalysisWorkflow.get_eligible_conditions(experiment)

        if not valid_conditions:
            st.warning("No conditions with multiple samples available.")
            return

        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")

        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "Lipid Class",
                all_classes,
                key='fach_class',
            )
        with col2:
            default_conds = valid_conditions[:2] if len(valid_conditions) >= 2 else valid_conditions
            selected_conditions = st.multiselect(
                "Conditions",
                valid_conditions,
                default=default_conds,
                key='fach_conditions',
            )

        if not selected_class or not selected_conditions:
            st.info(
                "Select a lipid class and at least one condition "
                "to generate the heatmap."
            )
            return

        st.markdown("---")
        st.markdown("#### 📊 Results")

        with st.spinner("Generating FACH heatmap..."):
            result = AnalysisWorkflow.run_fach(
                df, experiment, selected_class, selected_conditions,
            )

        if result.figure is None:
            st.warning(
                "Could not generate heatmap. The selected class may not have "
                "parsable fatty acid chain information."
            )
            return

        st.plotly_chart(result.figure, use_container_width=True)
        st.session_state.analysis_fach_fig = result.figure
        st.session_state.analysis_all_plots['fach'] = result.figure

        # Build combined CSV from all conditions
        combined_rows = []
        for condition, cond_df in result.data_dict.items():
            cond_df_copy = cond_df.copy()
            cond_df_copy['Condition'] = condition
            combined_rows.append(cond_df_copy)
        combined_csv = pd.concat(combined_rows) if combined_rows else pd.DataFrame()

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                result.figure,
                f"fach_{selected_class}.svg",
                key="analysis_svg_fach",
            )
        with col2:
            csv_download_button(
                combined_csv,
                f"fach_data_{selected_class}.csv",
                key="analysis_csv_fach",
            )


# ═══════════════════════════════════════════════════════════════════════
# Feature 5: Pathway Visualization
# ═══════════════════════════════════════════════════════════════════════


def _display_pathway_viz(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display lipid pathway visualization."""
    with st.expander(
        "Class Level Breakdown - Pathway Visualization", expanded=True
    ):
        st.markdown(
            "Visualize lipid class relationships on a metabolic pathway diagram."
        )

        st.markdown(
            "**Fold Change** (determines circle size):\n\n"
            "> Fold Change = Mean(Experimental) / Mean(Control)"
        )
        st.markdown(
            "**Saturation Ratio** (determines circle color, range 0–1):\n\n"
            "> Saturation Ratio = Saturated Chains / Total Chains"
        )

        _check_fa_compatibility(df)

        valid_conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
        if len(valid_conditions) < 2:
            st.warning(
                "Pathway visualization requires at least 2 conditions "
                "with multiple samples."
            )
            return

        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")

        col1, col2 = st.columns(2)
        with col1:
            control = st.selectbox(
                "Control Condition",
                valid_conditions,
                index=0,
                key='pathway_control',
            )
        with col2:
            exp_options = [c for c in valid_conditions if c != control]
            experimental = st.selectbox(
                "Experimental Condition",
                exp_options,
                index=0,
                key='pathway_experimental',
            )

        if control == experimental:
            st.warning("Control and experimental conditions must be different.")
            return

        st.markdown("---")
        st.markdown("#### 📊 Results")

        with st.spinner("Generating pathway visualization..."):
            result = AnalysisWorkflow.run_pathway(
                df, experiment, control, experimental,
            )

        if result.figure is None:
            st.warning("Could not generate pathway visualization.")
            return

        st.pyplot(result.figure)
        st.session_state.analysis_pathway_fig = result.figure
        st.session_state.analysis_all_plots['pathway'] = result.figure

        col1, col2 = st.columns(2)
        with col1:
            matplotlib_svg_download_button(
                result.figure,
                f"pathway_visualization_{control}_vs_{experimental}.svg",
                key="analysis_svg_pathway",
            )
        with col2:
            # Build summary DataFrame
            summary_rows = []
            pathway_dict = result.pathway_dict
            if pathway_dict and 'class' in pathway_dict:
                for i, cls in enumerate(pathway_dict['class']):
                    summary_rows.append({
                        'Lipid Class': cls,
                        'Fold Change': pathway_dict['abundance ratio'][i],
                        'Saturation Ratio': pathway_dict['saturated fatty acids ratio'][i],
                    })
            summary_df = pd.DataFrame(summary_rows)
            csv_download_button(
                summary_df,
                "pathway_visualization_data.csv",
                key="analysis_csv_pathway",
            )

        st.markdown(
            f"**Data Summary:** Comparing {experimental} to {control}"
        )
        if summary_rows:
            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True
            )


# ═══════════════════════════════════════════════════════════════════════
# Feature 6: Volcano Plot
# ═══════════════════════════════════════════════════════════════════════


def _display_volcano_plot(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display volcano plot analysis."""
    with st.expander("Species Level Breakdown - Volcano Plot", expanded=True):
        st.markdown(
            "Identify differentially abundant lipid species between two conditions."
        )

        st.markdown(
            "**Fold Change** (x-axis, log2-transformed):\n\n"
            "> Log2(FC) = log2(Mean Experimental / Mean Control)"
        )
        st.markdown(
            "**Significance** (y-axis):\n\n"
            "> -log10(p-value) from statistical test"
        )

        valid_conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
        if len(valid_conditions) < 2:
            st.warning(
                "Volcano plot requires at least 2 conditions "
                "with multiple samples."
            )
            return

        # Statistical options
        stat_config = _display_volcano_statistical_options()

        # Data selection
        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")

        col1, col2 = st.columns(2)
        with col1:
            control = st.selectbox(
                "Control Condition",
                valid_conditions,
                index=0,
                key='volcano_control',
            )
        with col2:
            exp_options = [c for c in valid_conditions if c != control]
            experimental = st.selectbox(
                "Experimental Condition",
                exp_options,
                index=0,
                key='volcano_experimental',
            )

        all_classes = AnalysisWorkflow.get_available_classes(df)
        selected_classes = st.multiselect(
            "Lipid Classes",
            all_classes,
            default=all_classes,
            key='volcano_classes',
        )

        if not selected_classes:
            st.warning("Please select at least one lipid class.")
            return
        if control == experimental:
            st.warning("Control and experimental conditions must be different.")
            return

        # Significance & display settings
        st.markdown("---")
        st.markdown("#### 📈 Results")
        st.markdown("**Significance & Display Settings**")

        col1, col2 = st.columns(2)
        with col1:
            p_threshold = st.number_input(
                "P-value threshold",
                min_value=0.001,
                max_value=0.1,
                value=0.05,
                step=0.001,
                format="%.3f",
                key='volcano_p_threshold',
            )
        with col2:
            fc_threshold = st.number_input(
                "Fold change threshold",
                min_value=1.1,
                max_value=5.0,
                value=2.0,
                step=0.1,
                format="%.1f",
                key='volcano_fc_threshold',
            )

        hide_nonsig = st.checkbox(
            "Hide non-significant points",
            value=False,
            key='volcano_hide_nonsig',
        )

        # Labeling options
        _render_volcano_results(
            df, experiment, control, experimental, selected_classes,
            stat_config, p_threshold, fc_threshold, hide_nonsig,
        )


def _display_volcano_statistical_options() -> StatisticalTestConfig:
    """Display volcano-specific statistical options (no Auto mode)."""
    st.markdown("---")
    st.markdown("#### ⚙️ Statistical Options")

    col1, col2 = st.columns(2)
    with col1:
        test_type = st.selectbox(
            "Test Type",
            ["parametric", "non_parametric"],
            index=0,
            key='volcano_stats_mode',
        )
    with col2:
        correction = st.selectbox(
            "Multiple Testing Correction",
            ["uncorrected", "fdr_bh", "bonferroni"],
            index=1,
            key='volcano_correction',
        )

    auto_transform = st.checkbox(
        "Auto-transform data (log10)",
        value=True,
        key='volcano_auto_transform',
    )

    return StatisticalTestConfig.create_manual(
        test_type=test_type,
        correction_method=correction,
        posthoc_correction='uncorrected',
        auto_transform=auto_transform,
    )


def _render_volcano_results(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    control: str,
    experimental: str,
    selected_classes: List[str],
    stat_config: StatisticalTestConfig,
    p_threshold: float,
    fc_threshold: float,
    hide_nonsig: bool,
) -> None:
    """Render the volcano plot, concentration scatter, and distribution plot."""
    # Labeling options
    st.markdown("**Labeling Options**")
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.number_input(
            "Top N lipids to label:",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            key='volcano_top_n',
        )
    with col2:
        enable_adjustment = st.checkbox(
            "Enable label adjustment",
            value=False,
            key='volcano_enable_adjust',
        )

    # Additional lipids to label
    additional_labels = st.multiselect(
        "Additional lipids to label:",
        [],
        default=[],
        key='volcano_additional_labels',
    )

    # Build custom label positions
    custom_positions = _collect_label_positions(
        top_n, additional_labels, enable_adjustment,
    )

    # Log2 FC threshold
    import math
    log2_fc = math.log2(fc_threshold) if fc_threshold > 0 else 1.0

    with st.spinner("Performing statistical analysis..."):
        result = AnalysisWorkflow.run_volcano(
            df, experiment, control, experimental, selected_classes,
            stat_config,
            p_threshold=p_threshold,
            fc_threshold=log2_fc,
            hide_non_sig=hide_nonsig,
            top_n_labels=top_n,
            custom_label_positions=custom_positions if custom_positions else None,
        )

    if result.figure is None:
        st.warning("Could not generate volcano plot.")
        return

    st.session_state.analysis_volcano_fig = result.figure
    st.session_state.analysis_volcano_data = result.volcano_data
    st.session_state.analysis_all_plots['volcano'] = result.figure

    # Update additional labels multiselect options
    if result.volcano_data and result.volcano_data.volcano_df is not None:
        all_lipids = result.volcano_data.volcano_df['LipidMolec'].tolist()
        # The multiselect was already rendered; update for next rerun
        st.session_state['_volcano_available_lipids'] = all_lipids

    # Volcano plot
    st.markdown("###### Volcano Plot")
    st.plotly_chart(result.figure, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        plotly_svg_download_button(
            result.figure,
            f"volcano_plot_{experimental}_vs_{control}.svg",
            key="analysis_svg_volcano",
        )
    with col2:
        if result.volcano_data and result.volcano_data.volcano_df is not None:
            csv_download_button(
                result.volcano_data.volcano_df,
                f"volcano_statistical_results_{experimental}_vs_{control}.csv",
                key="analysis_csv_volcano",
            )

    # Show detailed results
    _display_detailed_statistics(result.stat_summary, 'volcano')

    # Concentration vs Fold Change
    if result.concentration_plot is not None:
        st.markdown("---")
        st.markdown("###### Concentration vs. Fold Change")
        st.plotly_chart(result.concentration_plot, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                result.concentration_plot,
                f"conc_vs_fc_{experimental}_vs_{control}.svg",
                key="analysis_svg_conc_fc",
            )
        with col2:
            if result.concentration_df is not None:
                csv_download_button(
                    result.concentration_df,
                    f"conc_vs_fc_{experimental}_vs_{control}.csv",
                    key="analysis_csv_conc_fc",
                )

    # Individual lipid analysis
    _display_individual_lipid_analysis(
        df, experiment, result, control, experimental,
    )

    # Excluded lipids
    _display_excluded_lipids(result, control, experimental)


def _collect_label_positions(
    top_n: int,
    additional_labels: List[str],
    enable_adjustment: bool,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Collect custom label position adjustments."""
    if not enable_adjustment:
        return None

    # Get all labeled lipids (from previous run stored in session state)
    volcano_data = st.session_state.get('analysis_volcano_data')
    if volcano_data is None or volcano_data.volcano_df is None:
        return None

    # Determine which lipids are labeled
    vdf = volcano_data.volcano_df
    if vdf.empty:
        return None

    labeled_lipids = additional_labels.copy() if additional_labels else []
    if top_n > 0 and '-log10(pValue)' in vdf.columns:
        top_lipids = vdf.nlargest(top_n, '-log10(pValue)')['LipidMolec'].tolist()
        for lip in top_lipids:
            if lip not in labeled_lipids:
                labeled_lipids.append(lip)

    if not labeled_lipids:
        return None

    st.markdown("**Label Position Adjustments:**")
    positions = {}
    for lipid in labeled_lipids:
        cols = st.columns(3)
        with cols[0]:
            st.text(lipid)
        with cols[1]:
            x_off = st.number_input(
                "X", value=0.0, step=0.1,
                label_visibility="collapsed",
                key=f"volcano_label_x_{lipid}",
            )
        with cols[2]:
            y_off = st.number_input(
                "Y", value=0.0, step=0.1,
                label_visibility="collapsed",
                key=f"volcano_label_y_{lipid}",
            )
        if x_off != 0.0 or y_off != 0.0:
            positions[lipid] = (x_off, y_off)

    return positions if positions else None


def _display_individual_lipid_analysis(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    result: 'VolcanoResult',
    control: str,
    experimental: str,
) -> None:
    """Display individual lipid distribution plots."""
    if result.volcano_data is None or result.volcano_data.volcano_df is None:
        return

    vdf = result.volcano_data.volcano_df
    if vdf.empty:
        return

    st.markdown("---")
    st.markdown("#### Individual Lipid Analysis")

    classes_in_data = sorted(vdf['ClassKey'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        detail_class = st.selectbox(
            "Lipid Class:",
            classes_in_data,
            key='volcano_detail_class',
        )
    with col2:
        class_lipids = vdf[vdf['ClassKey'] == detail_class]['LipidMolec'].tolist()
        default_lipids = class_lipids[:3] if len(class_lipids) >= 3 else class_lipids
        detail_lipids = st.multiselect(
            "Lipids:",
            class_lipids,
            default=default_lipids,
            key='volcano_detail_lipids',
        )

    if not detail_lipids:
        return

    selected_conditions = [control, experimental]
    with st.spinner("Generating distribution plot..."):
        dist_fig = VolcanoPlotterService.create_distribution_plot(
            df, detail_lipids, selected_conditions, experiment,
        )

    st.pyplot(dist_fig)

    col1, col2 = st.columns(2)
    with col1:
        matplotlib_svg_download_button(
            dist_fig,
            f"conc_dist_{experimental}_vs_{control}.svg",
            key="analysis_svg_dist",
        )
    with col2:
        # Build distribution CSV
        dist_data = _build_distribution_csv(
            df, detail_lipids, experiment, selected_conditions,
        )
        csv_download_button(
            dist_data,
            f"conc_dist_data_{experimental}_vs_{control}.csv",
            key="analysis_csv_dist",
        )


def _build_distribution_csv(
    df: pd.DataFrame,
    lipids: List[str],
    experiment: ExperimentConfig,
    conditions: List[str],
) -> pd.DataFrame:
    """Build a CSV for individual lipid distribution data."""
    rows = []
    for lipid in lipids:
        lipid_row = df[df['LipidMolec'] == lipid]
        if lipid_row.empty:
            continue
        for condition in conditions:
            if condition not in experiment.conditions_list:
                continue
            idx = experiment.conditions_list.index(condition)
            samples = experiment.individual_samples_list[idx]
            for sample in samples:
                col = f"concentration[{sample}]"
                if col in df.columns:
                    val = lipid_row[col].values[0]
                    rows.append({
                        'LipidMolec': lipid,
                        'Condition': condition,
                        'Sample': sample,
                        'Concentration': val,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _display_excluded_lipids(
    result: 'VolcanoResult',
    control: str,
    experimental: str,
) -> None:
    """Display table of excluded lipids with reasons."""
    if result.volcano_data is None:
        return

    removed_df = result.volcano_data.removed_lipids_df
    if removed_df is None or removed_df.empty:
        return

    st.markdown("---")
    st.markdown("#### Excluded Lipids")
    st.dataframe(removed_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_download_button(
            removed_df,
            f"excluded_lipids_{experimental}_vs_{control}.csv",
            key="analysis_csv_excluded",
        )
    with col2:
        # Summarize reasons
        if 'Reason' in removed_df.columns:
            reasons = removed_df['Reason'].value_counts()
            summary = " | ".join(
                f"{reason}: {count}" for reason, count in reasons.items()
            )
            st.markdown(f"**Exclusion reasons:** {summary}")


# ═══════════════════════════════════════════════════════════════════════
# Feature 7: Lipidomic Heatmap
# ═══════════════════════════════════════════════════════════════════════


def _display_lipidomic_heatmap(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display lipidomic heatmap analysis."""
    with st.expander("Species Level Breakdown - Lipidomic Heatmap", expanded=True):
        st.markdown(
            "Visualize concentration patterns across all lipid species using "
            "Z-score normalized heatmaps."
        )

        st.markdown(
            "**Z-score** (color scale):\n\n"
            "> Z = (Value - Mean) / Std Dev  (computed per lipid species)"
        )

        all_conditions = AnalysisWorkflow.get_all_conditions(experiment)
        all_classes = AnalysisWorkflow.get_available_classes(df)

        st.markdown("---")
        st.markdown("#### 🎯 Data Selection")

        col1, col2 = st.columns(2)
        with col1:
            selected_conditions = st.multiselect(
                "Conditions",
                all_conditions,
                default=all_conditions,
                key='heatmap_conditions',
            )
        with col2:
            selected_classes = st.multiselect(
                "Lipid Classes",
                all_classes,
                default=all_classes,
                key='heatmap_classes',
            )

        if not selected_conditions or not selected_classes:
            st.warning("Please select at least one condition and one lipid class.")
            return

        st.markdown("---")
        st.markdown("#### ⚙️ Heatmap Settings")

        col1, col2 = st.columns(2)
        with col1:
            heatmap_type = st.radio(
                "Heatmap Type",
                ["Clustered", "Regular"],
                index=0,
                key='heatmap_type',
            )
        with col2:
            if heatmap_type == "Clustered":
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=5,
                    key='heatmap_n_clusters',
                )
            else:
                n_clusters = 3
                st.markdown("")  # Alignment placeholder

        heatmap_type_value = 'clustered' if heatmap_type == "Clustered" else 'regular'

        st.markdown("---")
        st.markdown("#### 📈 Results")

        with st.spinner("Generating heatmap..."):
            result = AnalysisWorkflow.run_heatmap(
                df, experiment, selected_conditions, selected_classes,
                heatmap_type=heatmap_type_value,
                n_clusters=n_clusters,
            )

        if result.figure is None:
            st.warning("Could not generate heatmap.")
            return

        st.plotly_chart(result.figure, use_container_width=True)
        st.session_state.analysis_heatmap_fig = result.figure
        st.session_state.analysis_all_plots['heatmap'] = result.figure

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                result.figure,
                f"lipidomic_{heatmap_type_value}_heatmap.svg",
                key="analysis_svg_heatmap",
            )
        with col2:
            if result.z_scores_df is not None:
                csv_download_button(
                    result.z_scores_df,
                    f"{heatmap_type_value}_heatmap_data.csv",
                    key="analysis_csv_heatmap",
                )

        # Cluster composition (clustered mode only)
        if heatmap_type == "Clustered":
            _display_cluster_composition(
                result, df, experiment, selected_conditions, selected_classes,
                n_clusters,
            )


def _display_cluster_composition(
    result: 'HeatmapResult',
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    selected_conditions: List[str],
    selected_classes: List[str],
    n_clusters: int,
) -> None:
    """Display cluster composition analysis."""
    st.markdown("---")
    st.markdown("##### Cluster Composition")

    composition_view = st.radio(
        "Show composition by:",
        ["Species Count", "Total Concentration"],
        horizontal=True,
        help="Species Count: % of lipid species. Total Concentration: % of summed abundance.",
        key='heatmap_cluster_view',
    )

    mode = 'species_count' if composition_view == "Species Count" else 'total_concentration'

    # Get filtered data for concentration mode
    filtered_df, _ = LipidomicHeatmapPlotterService.filter_data(
        df, selected_conditions, selected_classes, experiment,
    )

    composition_df = LipidomicHeatmapPlotterService.get_cluster_composition(
        result.z_scores_df, n_clusters, mode=mode,
        filtered_df=filtered_df,
    )

    if composition_df is not None:
        st.dataframe(composition_df, use_container_width=True)
        st.session_state.analysis_heatmap_clusters = composition_df

        csv_download_button(
            composition_df,
            f"cluster_composition_{mode}.csv",
            key="analysis_csv_cluster",
        )


# ═══════════════════════════════════════════════════════════════════════
# Shared Utility Functions
# ═══════════════════════════════════════════════════════════════════════


def _check_fa_compatibility(df: pd.DataFrame) -> None:
    """Show a warning if lipid names lack detailed fatty acid composition."""
    if 'LipidMolec' not in df.columns:
        return
    sample = df['LipidMolec'].head(20)
    has_detailed = sample.str.contains('_').any()
    if not has_detailed:
        st.info(
            "⚠️ Your data appears to use consolidated lipid names "
            "(e.g., PC(34:1) instead of PC(16:0_18:1)). "
            "Saturation and pathway analyses work best with "
            "detailed fatty acid composition."
        )


def _display_consolidated_lipids(
    df: pd.DataFrame,
    selected_classes: List[str],
    key_prefix: str,
) -> List[str]:
    """Display consolidated lipid detection and exclusion UI.

    Returns:
        List of lipid names to exclude.
    """
    consolidated = SaturationPlotterService.identify_consolidated_lipids(
        df, selected_classes,
    )

    if not consolidated:
        return []

    st.markdown("---")
    st.markdown("#### ⚠️ Consolidated Format Lipids")

    # Build summary table
    summary_rows = []
    all_consolidated = []
    for cls, lipids in consolidated.items():
        total = len(df[df['ClassKey'] == cls])
        count = len(lipids)
        pct = (count / total * 100) if total > 0 else 0
        summary_rows.append({
            'Class': cls,
            'Total Lipids': total,
            'Consolidated': count,
            '% Consolidated': f"{pct:.1f}%",
        })
        for lip in lipids:
            all_consolidated.append(f"{lip} ({cls})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Consolidated', ascending=False)
    st.dataframe(summary_df, use_container_width=True)

    exclude_labels = st.multiselect(
        f"Select lipids to exclude ({len(all_consolidated)} detected):",
        all_consolidated,
        default=[],
        help="Exclude consolidated format lipids with multiple fatty acid chains.",
        key=f'{key_prefix}_exclude_consolidated',
    )

    if exclude_labels:
        st.success(f"{len(exclude_labels)} lipid(s) will be excluded.")

    # Extract lipid names from "lipid (class)" format
    excluded_names = []
    for label in exclude_labels:
        name = label.rsplit(' (', 1)[0]
        excluded_names.append(name)

    return excluded_names
