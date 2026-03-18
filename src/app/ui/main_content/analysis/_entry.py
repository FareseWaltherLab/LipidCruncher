"""
Entry point for Module 3: Visualize and Analyze UI.

Provides 7 analysis features accessed via radio selector:
1. Abundance Bar Chart (class level)
2. Abundance Pie Charts (class level)
3. Saturation Plots (class level, requires detailed FA)
4. Fatty Acid Composition Heatmaps (class level)
5. Pathway Visualization (class level, requires detailed FA)
6. Volcano Plot (species level)
7. Lipidomic Heatmap (species level)
"""

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.services.report_generator import generate_pdf_report, build_metadata_from_experiment
from app.workflows.analysis import AnalysisWorkflow

from app.ui.main_content.analysis._bar_chart import _display_bar_chart
from app.ui.main_content.analysis._pie_charts import _display_pie_charts
from app.ui.main_content.analysis._saturation import _display_saturation_plots
from app.ui.main_content.analysis._fach import _display_fach_heatmaps
from app.ui.main_content.analysis._pathway import _display_pathway_viz
from app.ui.main_content.analysis._volcano import _display_volcano_plot
from app.ui.main_content.analysis._heatmap import _display_lipidomic_heatmap


# ═══════════════════════════════════════════════════════════════════════
# Analysis Radio Options
# ═══════════════════════════════════════════════════════════════════════

ANALYSIS_OPTIONS = [
    "Class Level Breakdown - Bar Chart",
    "Class Level Breakdown - Pie Charts",
    "Class Level Breakdown - Saturation Plots (requires detailed fatty acid composition)",
    "Class Level Breakdown - Fatty Acid Composition Heatmaps",
    "Class Level Breakdown - Pathway Visualization (requires detailed fatty acid composition)",
    "Species Level Breakdown - Volcano Plot",
    "Species Level Breakdown - Lipidomic Heatmap",
]


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════


def display_analysis_module(
    df: pd.DataFrame,
    experiment: ExperimentConfig,
    bqc_label: Optional[str],
    format_type: str,
) -> None:
    """Display Module 3: Visualize and Analyze.

    Args:
        df: DataFrame with concentration[sample] columns, LipidMolec, ClassKey.
        experiment: Experiment configuration.
        bqc_label: BQC condition label, or None.
        format_type: Data format string.
    """
    st.subheader("Visualize and Analyze")

    errors = AnalysisWorkflow.validate_inputs(df, experiment)
    if errors:
        for err in errors:
            st.error(err)
        return

    analysis_type = st.radio(
        "Select Analysis",
        ANALYSIS_OPTIONS,
        key='analysis_radio',
    )

    if analysis_type == ANALYSIS_OPTIONS[0]:
        _display_bar_chart(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[1]:
        _display_pie_charts(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[2]:
        _display_saturation_plots(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[3]:
        _display_fach_heatmaps(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[4]:
        _display_pathway_viz(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[5]:
        _display_volcano_plot(df, experiment)
    elif analysis_type == ANALYSIS_OPTIONS[6]:
        _display_lipidomic_heatmap(df, experiment)

    # PDF Report Download
    _display_pdf_report_section(experiment, format_type)


# ═══════════════════════════════════════════════════════════════════════
# PDF Report
# ═══════════════════════════════════════════════════════════════════════


def _collect_qc_plots() -> Dict[str, Any]:
    """Collect QC plots from session state for PDF report."""
    qc_plots: Dict[str, Any] = {}

    fig1 = st.session_state.get('qc_box_plot_fig1')
    if fig1 is not None:
        qc_plots['box_plot_fig1'] = fig1

    fig2 = st.session_state.get('qc_box_plot_fig2')
    if fig2 is not None:
        qc_plots['box_plot_fig2'] = fig2

    bqc = st.session_state.get('qc_bqc_plot')
    if bqc is not None:
        qc_plots['bqc_plot'] = bqc

    rt = st.session_state.get('qc_retention_time_plot')
    if rt is not None:
        qc_plots['retention_time_plot'] = rt

    pca = st.session_state.get('qc_pca_plot')
    if pca is not None:
        qc_plots['pca_plot'] = pca

    corr = st.session_state.get('qc_correlation_plots', {})
    if corr:
        qc_plots['correlation_plots'] = corr

    return qc_plots


def _display_pdf_report_section(
    experiment: ExperimentConfig,
    format_type: str,
) -> None:
    """Display PDF report generation and download section."""
    analysis_plots = st.session_state.get('analysis_all_plots', {})
    qc_plots = _collect_qc_plots()

    if not analysis_plots and not qc_plots:
        return

    st.markdown("---")
    st.subheader("Download PDF Report")
    st.markdown(
        "Generate a PDF report containing all QC and analysis plots "
        "created during this session."
    )

    if st.button("Generate PDF Report", key="generate_pdf_report"):
        with st.spinner("Generating PDF report..."):
            metadata = build_metadata_from_experiment(experiment, format_type)
            pdf_buffer = generate_pdf_report(
                analysis_plots=analysis_plots,
                metadata=metadata,
                qc_plots=qc_plots,
            )

        if pdf_buffer is not None:
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="lipidcruncher_report.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )
        else:
            st.error(
                "Failed to generate PDF report. "
                "Please ensure kaleido is installed for plot export."
            )
