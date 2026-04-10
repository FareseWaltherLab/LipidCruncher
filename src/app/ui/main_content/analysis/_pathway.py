"""Feature 5: Pathway Visualization analysis with editable layout."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.services.plotting.pathway_viz import PathwayVizPlotterService
from app.ui.download_utils import plotly_svg_download_button, csv_download_button

from app.ui.main_content.analysis._utils import (
    _check_fa_compatibility,
    _display_consolidated_lipids,
)
from app.ui.main_content.analysis._pathway_state import (
    init_pathway_state,
    get_active_classes,
    get_custom_nodes,
    get_added_edges,
    get_removed_edges,
    get_position_overrides,
)
from app.ui.main_content.analysis._pathway_editor import display_pathway_layout_editor


def _display_pathway_viz(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display lipid pathway visualization."""
    init_pathway_state()

    with st.expander(
        "Class Level Breakdown - Pathway Visualization", expanded=True
    ):
        st.markdown(
            "Visualize lipid class relationships on a metabolic pathway diagram."
        )

        st.markdown("**Fold Change** (determines circle size, log2-scaled):")
        st.code("Fold Change = Mean(Experimental) / Mean(Control)", language=None)
        st.markdown("**Saturation Ratio** (determines circle color, scaled 0 to max):")
        st.code("Saturation Ratio = Saturated Chains / Total Chains", language=None)
        st.markdown(
            "Classes present in the dataset are shown as filled circles. "
            "Classes on the pathway but absent from your data are shown "
            "as dashed gray outlines."
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
        st.markdown("#### Data Selection")

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

        # --- Consolidated lipid handling ---
        all_classes = list(df['ClassKey'].unique())
        excluded_lipids = _display_consolidated_lipids(df, all_classes, 'pathway')
        st.markdown(
            "Consolidated-format lipids affect **circle colors** "
            "(saturation ratio) but not **circle sizes** (fold change)."
        )

        # --- Layout editor ---
        display_pathway_layout_editor()

        # --- Compute data (cached) ---
        saturation_source = None
        if excluded_lipids:
            saturation_source = df[~df['LipidMolec'].isin(excluded_lipids)].copy()

        data = StreamlitAdapter.compute_pathway_data(
            df, experiment, control, experimental,
            saturation_source_df=saturation_source,
        )

        # --- Render figure (uses current layout, not cached) ---
        active_classes = get_active_classes()
        custom_nodes = get_custom_nodes()
        added_edges = get_added_edges()
        removed_edges = get_removed_edges()
        position_overrides = get_position_overrides()
        show_grid = st.session_state.get('pathway_show_grid', False)

        figure, pathway_dict = PathwayVizPlotterService.create_pathway_viz(
            data.fold_change_df,
            data.saturation_df,
            active_classes=active_classes,
            custom_nodes=custom_nodes if custom_nodes else None,
            added_edges=added_edges if added_edges else None,
            removed_edges=removed_edges if removed_edges else None,
            position_overrides=position_overrides if position_overrides else None,
            show_grid=show_grid,
        )

        st.markdown("---")
        st.markdown("#### Results")

        if figure is None:
            st.warning("Could not generate pathway visualization.")
            return

        st.plotly_chart(figure, use_container_width=True)
        st.session_state.analysis_pathway_fig = figure
        st.session_state.analysis_all_plots['pathway'] = figure

        col1, col2 = st.columns(2)
        with col1:
            plotly_svg_download_button(
                figure,
                f"pathway_visualization_{control}_vs_{experimental}.svg",
                key="analysis_svg_pathway",
            )
        with col2:
            summary_rows = []
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

        st.markdown(f"**Data Summary:** Comparing {experimental} to {control}")
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)