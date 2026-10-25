"""Feature 1: Abundance Bar Chart analysis."""

from typing import Optional

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._shared import (
    _clear_pdf_plots,
    _clear_statistics,
    _display_condition_class_selectors,
    _warn_if_nothing_selected,
    _display_correction_scope,
    _display_statistical_options,
    _display_detailed_statistics,
    _publish,
    _publish_statistics,
)


def _display_bar_chart(
    df: pd.DataFrame, experiment: ExperimentConfig, bqc_label: Optional[str] = None,
) -> None:
    """Display abundance bar chart analysis. The BQC condition is left out
    of the default selection, so out of the statistics unless the user adds
    it."""
    _clear_pdf_plots('bar_chart')
    _clear_statistics('Abundance bar chart')
    with st.expander("Class Concentration Bar Chart", expanded=False):
        st.markdown(
            "Visualize the total abundance of each lipid class across conditions."
        )

        selected_conditions, selected_classes = _display_condition_class_selectors(
            experiment, df, 'bar', bqc_label=bqc_label,
        )

        if _warn_if_nothing_selected(selected_conditions, selected_classes):
            return

        stat_config = _display_statistical_options(
            'bar', len(selected_classes), len(selected_conditions),
        )

        section_header("📊 Results")

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

        st.plotly_chart(result.figure, width='stretch')
        _publish('bar_chart', result.figure)

        display_export_buttons(
            result.figure, result.abundance_df,
            f"abundance_bar_chart_{scale_value}.svg",
            f"abundance_bar_chart_{scale_value}.csv",
            "analysis_svg_bar", "analysis_csv_bar",
        )

        _publish_statistics('Abundance bar chart', result.stat_summary)
        _display_correction_scope(result.stat_summary, df, selected_classes, 'classes')
        _display_detailed_statistics(result.stat_summary, 'bar')
