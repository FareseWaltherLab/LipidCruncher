"""Feature 1: Abundance Bar Chart analysis."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.download_utils import plotly_svg_download_button, csv_download_button

from app.ui.main_content.analysis._shared import (
    _display_condition_class_selectors,
    _display_statistical_options,
    _display_detailed_statistics,
)


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

        result = StreamlitAdapter.run_bar_chart(
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
