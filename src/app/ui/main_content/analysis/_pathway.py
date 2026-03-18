"""Feature 5: Pathway Visualization analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.ui.download_utils import matplotlib_svg_download_button, csv_download_button

from app.ui.main_content.analysis._utils import _check_fa_compatibility


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

        result = StreamlitAdapter.run_pathway(
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
