"""Feature 2: Abundance Pie Charts analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.ui.st_helpers import display_export_buttons, section_header
from app.ui.main_content.analysis._shared import _clear_pdf_plots, _publish


def _display_pie_charts(df: pd.DataFrame, experiment: ExperimentConfig) -> None:
    """Display abundance pie chart analysis."""
    _clear_pdf_plots('pie_')
    with st.expander("Class Concentration Pie Chart", expanded=False):
        st.markdown(
            "View the proportional distribution of lipid classes per "
            "condition, or of the species within one class."
        )

        all_classes = AnalysisWorkflow.get_available_classes(df)
        all_conditions = AnalysisWorkflow.get_all_conditions(experiment)

        section_header("🎯 Data Selection")

        view = st.radio(
            "Show",
            ["Lipid classes", "Species within a class"],
            horizontal=True,
            key='pie_view',
            help=(
                "Species within a class: each species of the chosen class "
                "as a percentage of that class's total."
            ),
        )

        species_class = None
        if view == "Lipid classes":
            selected_classes = st.multiselect(
                "Lipid Classes",
                all_classes,
                default=all_classes,
                key='pie_classes',
            )
            if not selected_classes:
                st.warning("Please select at least one lipid class to create the pie charts.")
                return
        else:
            species_class = st.selectbox(
                "Lipid Class", all_classes, key='pie_species_class',
            )
            selected_classes = [species_class]

        section_header("📊 Results")

        results = StreamlitAdapter.run_pie_charts(
            df, experiment, all_conditions, selected_classes, species_class,
        )

        file_stem = (
            f"{species_class}_species_pie_chart" if species_class
            else "abundance_pie_chart"
        )
        for condition, pie_result in results.items():
            st.markdown(f"###### {condition}")
            st.plotly_chart(pie_result.figure, width='stretch')
            _publish(f'pie_{condition}', pie_result.figure)

            display_export_buttons(
                pie_result.figure, pie_result.data_df,
                f"{file_stem}_{condition}.svg",
                f"{file_stem}_{condition}.csv",
                f"analysis_svg_pie_{condition}", f"analysis_csv_pie_{condition}",
            )
