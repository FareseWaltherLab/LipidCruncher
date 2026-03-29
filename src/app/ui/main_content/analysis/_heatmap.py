"""Feature 7: Lipidomic Heatmap analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.plotting.lipidomic_heatmap import LipidomicHeatmapPlotterService
from app.workflows.analysis import AnalysisWorkflow
from app.ui.download_utils import csv_download_button
from app.ui.st_helpers import display_export_buttons, section_header


def _display_lipidomic_heatmap(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display lipidomic heatmap analysis."""
    with st.expander("Species Level Breakdown - Lipidomic Heatmap", expanded=True):
        st.markdown(
            "Visualize concentration patterns across all lipid species using "
            "Z-score normalized heatmaps."
        )

        st.markdown("**Z-score** (color scale):")
        st.code("Z = (Value - Mean) / Std Dev  (computed per lipid species)", language=None)

        all_conditions = AnalysisWorkflow.get_all_conditions(experiment)
        all_classes = AnalysisWorkflow.get_available_classes(df)

        section_header("🎯 Data Selection")

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

        section_header("⚙️ Heatmap Settings")

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

        section_header("📈 Results")

        result = StreamlitAdapter.run_heatmap(
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

        if result.z_scores_df is not None:
            display_export_buttons(
                result.figure, result.z_scores_df,
                f"lipidomic_{heatmap_type_value}_heatmap.svg",
                f"{heatmap_type_value}_heatmap_data.csv",
                "analysis_svg_heatmap", "analysis_csv_heatmap",
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
    selected_conditions: list,
    selected_classes: list,
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
