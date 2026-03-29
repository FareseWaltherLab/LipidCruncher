"""Feature 3: Saturation Plots analysis."""

from typing import List

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.plotting.saturation_plot import SaturationPlotterService
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._shared import (
    _display_condition_class_selectors,
    _display_statistical_options,
    _display_detailed_statistics,
)
from app.ui.main_content.analysis._utils import (
    _check_fa_compatibility,
    _display_consolidated_lipids,
)


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

        section_header("📊 Results")

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
        result = StreamlitAdapter.run_saturation(
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

            display_export_buttons(
                fig,
                _build_saturation_csv(df, experiment, selected_conditions, lipid_class),
                f"saturation_{ptype}_{lipid_class}.svg",
                f"saturation_{ptype}_{lipid_class}.csv",
                f"analysis_svg_sat_{ptype}_{lipid_class}",
                f"analysis_csv_sat_{ptype}_{lipid_class}",
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
