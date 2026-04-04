"""Feature 6: Volcano Plot analysis."""

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.plotting.volcano_plot import VolcanoPlotterService
from app.workflows.analysis import AnalysisWorkflow
from app.ui.download_utils import csv_download_button
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._shared import _display_detailed_statistics


def _display_volcano_plot(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display volcano plot analysis."""
    with st.expander("Species Level Breakdown - Volcano Plot", expanded=True):
        st.markdown(
            "Identify differentially abundant lipid species between two conditions."
        )

        st.markdown("**Fold Change** (x-axis, log2-transformed):")
        st.code("Log2(FC) = log2(Mean Experimental / Mean Control)", language=None)
        st.markdown("**Significance** (y-axis):")
        st.code("-log10(p-value) from statistical test", language=None)

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
        section_header("🎯 Data Selection")

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
        section_header("📈 Results")
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
    available_lipids = st.session_state.get('_volcano_available_lipids', [])
    additional_labels = st.multiselect(
        "Additional lipids to label:",
        available_lipids,
        default=[],
        key='volcano_additional_labels',
    )

    # Determine which p-value column the plot will use
    use_adjusted = stat_config.correction_method != 'uncorrected'
    p_col = '-log10(adjusted_pValue)' if use_adjusted else '-log10(pValue)'

    # Build custom label positions
    custom_positions = _collect_label_positions(
        top_n, additional_labels, enable_adjustment, p_col,
    )

    # Log2 FC threshold
    log2_fc = math.log2(fc_threshold) if fc_threshold > 0 else 1.0

    result = StreamlitAdapter.run_volcano(
        df, experiment, control, experimental, selected_classes,
        stat_config,
        p_threshold=p_threshold,
        fc_threshold=log2_fc,
        hide_non_sig=hide_nonsig,
        top_n_labels=top_n,
        custom_label_positions=custom_positions if custom_positions else None,
        additional_labels=tuple(additional_labels) if additional_labels else None,
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

    if result.volcano_data and result.volcano_data.volcano_df is not None:
        display_export_buttons(
            result.figure, result.volcano_data.volcano_df,
            f"volcano_plot_{experimental}_vs_{control}.svg",
            f"volcano_statistical_results_{experimental}_vs_{control}.csv",
            "analysis_svg_volcano", "analysis_csv_volcano",
        )

    # Show detailed results
    _display_detailed_statistics(result.stat_summary, 'volcano')

    # Concentration vs Fold Change
    if result.concentration_plot is not None:
        st.markdown("---")
        st.markdown("###### Concentration vs. Fold Change")
        st.plotly_chart(result.concentration_plot, use_container_width=True)

        if result.concentration_df is not None:
            display_export_buttons(
                result.concentration_plot, result.concentration_df,
                f"conc_vs_fc_{experimental}_vs_{control}.svg",
                f"conc_vs_fc_{experimental}_vs_{control}.csv",
                "analysis_svg_conc_fc", "analysis_csv_conc_fc",
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
    p_col: str = '-log10(pValue)',
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
    if top_n > 0 and p_col in vdf.columns:
        top_lipids = vdf.nlargest(top_n, p_col)['LipidMolec'].tolist()
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

    dist_data = _build_distribution_csv(
        df, detail_lipids, experiment, selected_conditions,
    )
    display_export_buttons(
        dist_fig, dist_data,
        f"conc_dist_{experimental}_vs_{control}.svg",
        f"conc_dist_data_{experimental}_vs_{control}.csv",
        "analysis_svg_dist", "analysis_csv_dist",
        is_matplotlib=True,
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

    csv_download_button(
        removed_df,
        f"excluded_lipids_{experimental}_vs_{control}.csv",
        key="analysis_csv_excluded",
    )

    # Summarize reasons
    if 'Reason' in removed_df.columns:
        reasons = removed_df['Reason'].value_counts()
        summary = " | ".join(
            f"{reason}: {count}" for reason, count in reasons.items()
        )
        st.markdown(f"**Exclusion reasons:** {summary}")
