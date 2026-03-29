"""Feature 4: Fatty Acid Composition Heatmaps analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._utils import _check_fa_compatibility


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

        section_header("🎯 Data Selection")

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

        section_header("📊 Results")

        result = StreamlitAdapter.run_fach(
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

        display_export_buttons(
            result.figure, combined_csv,
            f"fach_{selected_class}.svg",
            f"fach_data_{selected_class}.csv",
            "analysis_svg_fach", "analysis_csv_fach",
        )
