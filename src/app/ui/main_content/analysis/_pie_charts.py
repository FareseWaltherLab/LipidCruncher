"""Feature 2: Abundance Pie Charts analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.ui.download_utils import plotly_svg_download_button, csv_download_button


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

        results = StreamlitAdapter.run_pie_charts(
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
