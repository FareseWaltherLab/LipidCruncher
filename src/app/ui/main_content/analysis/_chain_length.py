"""Feature: Chain Length Distribution bubble charts."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._shared import (
    _display_condition_class_selectors,
)


def _display_chain_length_plots(
    df: pd.DataFrame, experiment: ExperimentConfig
) -> None:
    """Display chain length and double bond distribution bubble charts."""
    with st.expander(
        "Class Level Breakdown - Chain Length Distribution", expanded=True
    ):
        st.markdown(
            "Bubble charts showing the distribution of total carbon chain "
            "lengths and double bonds per lipid class. Bubble size reflects "
            "mean concentration across selected conditions."
        )

        selected_conditions, selected_classes = _display_condition_class_selectors(
            experiment, df, 'clen',
        )

        if not selected_conditions:
            st.warning("Please select at least one condition.")
            return
        if not selected_classes:
            st.warning("Please select at least one lipid class.")
            return

        section_header("📊 Results")

        result = StreamlitAdapter.run_chain_length(
            df, experiment, selected_conditions, selected_classes,
        )

        if not result.success:
            for err in result.validation_errors:
                st.error(err)
            return

        if result.figure is None or not result.data or not result.data.records:
            st.info(
                "No chain length data could be extracted. This analysis "
                "requires lipid names with chain information (e.g. PC 34:1)."
            )
            return

        st.plotly_chart(result.figure, use_container_width=True)

        st.session_state.analysis_chain_length_fig = result.figure
        st.session_state.analysis_all_plots['chain_length'] = result.figure

        csv_df = pd.DataFrame(result.data.records)
        display_export_buttons(
            result.figure,
            csv_df,
            "chain_length_distribution.svg",
            "chain_length_distribution.csv",
            "analysis_svg_chain_length",
            "analysis_csv_chain_length",
        )