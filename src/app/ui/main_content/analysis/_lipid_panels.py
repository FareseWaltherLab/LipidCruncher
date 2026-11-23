"""Individual Lipid Panels: one panel per lipid for any number of
conditions, with the bar chart's statistics per lipid."""

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.adapters.streamlit_adapter import StreamlitAdapter
from app.workflows.analysis import AnalysisWorkflow
from app.ui.color_scheme import current_palette
from app.ui.sample_labels import names_for
from app.ui.st_helpers import display_export_buttons, section_header

from app.ui.main_content.analysis._shared import (
    _clear_pdf_plots,
    _clear_statistics,
    _display_detailed_statistics,
    _display_statistical_options,
    _publish,
    _publish_statistics,
)

_ANALYSIS = 'Individual lipid panels'


def _display_lipid_panels(
    df: pd.DataFrame, experiment: ExperimentConfig, bqc_label=None,
) -> None:
    """Display one concentration panel per chosen lipid across the chosen
    conditions. BQC is offered but not selected by default."""
    _clear_pdf_plots('lipid_panels')
    _clear_statistics(_ANALYSIS)
    with st.expander("Individual Lipid Panels", expanded=False):
        st.markdown(
            "One panel per lipid: a bar (mean ± SD) or box per condition, a dot "
            "per sample, and brackets for the pairs that differ significantly. "
            "Each lipid is tested the way the bar chart tests a class: Level 1 "
            "corrects across the lipids chosen here, and with 3 or more "
            "conditions Level 2 picks out which pairs differ (Tukey's HSD, "
            "Bonferroni, uncorrected, or Dunnett's test against a control)."
        )

        section_header("🎯 Data Selection")
        conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
        selected_conditions = st.multiselect(
            "Conditions", conditions,
            default=[c for c in conditions if c != bqc_label],
            key='panels_conditions',
        )
        classes = AnalysisWorkflow.get_available_classes(df)
        col1, col2 = st.columns(2)
        with col1:
            # A new class starts from its first lipids, not an empty pick.
            lipid_class = st.selectbox(
                "Lipid Class:", classes, key='panels_class',
                on_change=lambda: st.session_state.pop('panels_lipids', None),
            )
        with col2:
            class_lipids = df.loc[df['ClassKey'] == lipid_class, 'LipidMolec'].tolist()
            lipids = st.multiselect(
                "Lipids:", class_lipids, default=class_lipids[:3], key='panels_lipids',
            )
        if not selected_conditions:
            st.warning("Please select at least one condition.")
            return
        if not lipids:
            st.warning("Please select at least one lipid.")
            return

        stat_config = _display_statistical_options(
            'panels', len(lipids), len(selected_conditions), selected_conditions,
        )

        section_header("📊 Results")
        style = st.radio(
            "Show as:", ['Bar (mean ± SD)', 'Box'], horizontal=True, key='panels_style',
        )
        result = StreamlitAdapter.run_lipid_panels(
            df, experiment, selected_conditions, lipids, stat_config,
            style='box' if style == 'Box' else 'bar', palette=current_palette(),
        )

        st.plotly_chart(result.figure, width='content')
        _publish('lipid_panels', result.figure)
        display_export_buttons(
            result.figure, result.data,
            "lipid_panels.svg", "lipid_panels_data.csv",
            "analysis_svg_panels", "analysis_csv_panels",
            sample_names=names_for('qc'),
        )

        _publish_statistics(_ANALYSIS, result.stat_summary)
        _display_detailed_statistics(result.stat_summary, 'panels')
