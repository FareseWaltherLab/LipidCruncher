"""Feature 7: Lipidomic Heatmap analysis."""

import math

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.services.plotting.lipidomic_heatmap import (
    GROUPED_PAGE_SIZE,
    LipidomicHeatmapPlotterService,
)
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
                ["Clustered", "Regular", "Grouped by Class", "Aggregated by Class"],
                index=0,
                key='heatmap_type',
                help=(
                    "Clustered: one row per species, ordered by hierarchical "
                    "clustering. "
                    "Regular: one row per species, in input order. "
                    "Grouped by Class: one row per species, grouped into lipid "
                    f"class blocks, {GROUPED_PAGE_SIZE} species per page. "
                    "Aggregated by Class: one row per lipid class, summing the "
                    "concentrations of its species."
                ),
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

        heatmap_type_value = {
            "Clustered": 'clustered',
            "Regular": 'regular',
            "Grouped by Class": 'class_grouped',
            "Aggregated by Class": 'class_aggregated',
        }[heatmap_type]

        species_total = LipidomicHeatmapPlotterService.count_species(
            df, selected_classes,
        )
        species_page = 0
        if heatmap_type_value == 'class_grouped':
            species_page = _select_species_page(species_total)

        section_header("📈 Results")

        result = StreamlitAdapter.run_heatmap(
            df, experiment, selected_conditions, selected_classes,
            heatmap_type=heatmap_type_value,
            n_clusters=n_clusters,
            species_page=species_page,
        )

        if not result.success:
            for message in result.validation_errors:
                st.warning(message)
            return

        if result.figure is None:
            st.warning("Could not generate heatmap.")
            return

        # The class modes size themselves so every cell is a square, which
        # container-width stretching would undo. Clustered and Regular keep
        # their original stretched layout.
        square_celled = heatmap_type_value in ('class_grouped', 'class_aggregated')
        st.plotly_chart(result.figure, use_container_width=not square_celled)

        if heatmap_type_value == 'class_grouped' and species_total > GROUPED_PAGE_SIZE:
            start, end = LipidomicHeatmapPlotterService.page_bounds(
                species_total, species_page,
            )
            st.caption(
                f"Showing species {start + 1}–{end} of {species_total}, "
                f"ordered by lipid class. The CSV download below contains "
                f"all {species_total}."
            )
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


def _select_species_page(species_total: int) -> int:
    """Show a species-range picker and return the chosen zero-based page.

    One row per species would make a tall selection unreadable, so the species
    are paged. Returns 0 without rendering anything when they all fit on one
    page. Narrowing the class selection is not an alternative here: a single
    class can hold far more species than fit.
    """
    if species_total <= GROUPED_PAGE_SIZE:
        return 0

    n_pages = math.ceil(species_total / GROUPED_PAGE_SIZE)
    pages = list(range(n_pages))

    def _label(page: int) -> str:
        start = page * GROUPED_PAGE_SIZE
        return f"{start + 1}–{min(start + GROUPED_PAGE_SIZE, species_total)}"

    return st.selectbox(
        f"Species range ({species_total} species, {n_pages} pages)",
        pages,
        format_func=_label,
        key='heatmap_species_page',
        help=(
            "One row per species, so the species are shown a page at a time "
            "in lipid class order. Use 'Aggregated by Class' to see every "
            "class at once instead."
        ),
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

    mode = 'species_count' if composition_view == "Species Count" else 'concentration'

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
