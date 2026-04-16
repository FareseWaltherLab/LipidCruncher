"""
LSI Compliance Report UI.

Renders the LSI compliance report section at the bottom of the analysis module.
Collects auto-filled fields from session state and provides text inputs for
manual fields. Offers PDF and CSV download buttons.
"""
from typing import Optional

import pandas as pd
import streamlit as st

from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.models.statistics import StatisticalTestConfig
from app.services.lsi_report import LSIReportService, _PLACEHOLDER


def display_lsi_report_section(
    experiment: ExperimentConfig,
    format_type: str,
    bqc_label: Optional[str],
) -> None:
    """Display LSI compliance report generator UI.

    Placed at the bottom of the analysis module. Collects auto-filled data
    from session state and lets the user optionally fill in manual fields.

    Args:
        experiment: Experiment configuration.
        format_type: Data format string.
        bqc_label: BQC condition label, or None.
    """
    st.markdown("---")
    with st.expander("LSI Compliance Report"):
        st.markdown(
            "The [Lipidomics Standards Initiative (LSI)]"
            "(https://lipidomicstandards.org/) recommends standardized reporting "
            "for lipidomics studies. This section generates a pre-filled LSI "
            "compliance checklist based on your analysis. Fields that LipidCruncher "
            "cannot determine are left blank for you to fill in."
        )

        # ── Collect auto-filled fields ──
        auto_fields = _collect_auto_fields(experiment, format_type, bqc_label)

        # ── Preview auto-filled fields ──
        with st.expander("Preview auto-filled fields", expanded=False):
            for item, value in auto_fields.items():
                st.markdown(f"**{item}:** {value}")

        # ── Manual fields ──
        manual_fields = _render_manual_fields()

        # ── Download buttons ──
        st.markdown("#### Download Report")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate PDF Report", key="lsi_generate_pdf"):
                try:
                    pdf_bytes = LSIReportService.generate_checklist_pdf(
                        auto_fields, manual_fields
                    )
                    st.session_state["lsi_pdf_bytes"] = pdf_bytes
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

            if st.session_state.get("lsi_pdf_bytes"):
                st.download_button(
                    label="Download LSI Report (PDF)",
                    data=st.session_state["lsi_pdf_bytes"],
                    file_name="lsi_compliance_report.pdf",
                    mime="application/pdf",
                    key="lsi_download_pdf",
                )

        with col2:
            csv_str = LSIReportService.generate_checklist_csv(
                auto_fields, manual_fields
            )
            st.download_button(
                label="Download LSI Report (CSV)",
                data=csv_str.encode("utf-8"),
                file_name="lsi_compliance_report.csv",
                mime="text/csv",
                key="lsi_download_csv",
            )


def _collect_auto_fields(
    experiment: ExperimentConfig,
    format_type: str,
    bqc_label: Optional[str],
) -> dict:
    """Gather auto-filled fields from session state and configs."""
    # Normalization config
    norm_result = st.session_state.get("normalization_result")
    norm_config: Optional[NormalizationConfig] = None
    if norm_result is not None and hasattr(norm_result, "config"):
        norm_config = norm_result.config

    # If no config from result, try to build from session state
    if norm_config is None:
        norm_method = st.session_state.get("normalization_method", "None")
        if norm_method and norm_method != "None":
            try:
                norm_config = NormalizationConfig(
                    method=norm_method.lower().replace(" ", "_"),
                    selected_classes=st.session_state.get("selected_classes", []),
                    internal_standards=st.session_state.get("class_standard_map"),
                    intsta_concentrations=st.session_state.get("standard_concentrations"),
                    protein_concentrations=None,
                )
            except Exception:
                norm_config = None

    # Statistical test config — check if any analysis stored one
    stat_config: Optional[StatisticalTestConfig] = st.session_state.get(
        "analysis_stat_config"
    )

    # Cleaned DataFrame
    cleaned_df = st.session_state.get("cleaned_df", pd.DataFrame())

    # Internal standards
    intsta_df = st.session_state.get("intsta_df")

    # Cleaning parameters
    cleaning_params = _build_cleaning_params()

    # QC summary
    qc_summary = _build_qc_summary()

    return LSIReportService.collect_auto_fields(
        format_type=format_type,
        experiment=experiment,
        normalization_config=norm_config,
        stat_config=stat_config,
        cleaned_df=cleaned_df,
        intsta_df=intsta_df,
        bqc_label=bqc_label,
        cleaning_params=cleaning_params,
        qc_summary=qc_summary,
    )


def _build_cleaning_params() -> dict:
    """Extract data cleaning parameters from session state."""
    params: dict = {}

    grade_config = st.session_state.get("grade_config")
    if grade_config is not None:
        params["Grade filter (LipidSearch)"] = str(grade_config)

    quality_config = st.session_state.get("last_quality_config")
    if quality_config is not None:
        params["Quality filter (MS-DIAL)"] = str(quality_config)

    # Zero filtering thresholds
    non_bqc = st.session_state.get("_preserved_non_bqc_zero_threshold")
    bqc = st.session_state.get("_preserved_bqc_zero_threshold")
    if non_bqc is not None:
        params["Zero filter threshold (non-BQC)"] = f"{non_bqc}%"
    if bqc is not None:
        params["Zero filter threshold (BQC)"] = f"{bqc}%"

    return params


def _build_qc_summary() -> dict:
    """Extract QC metrics from session state."""
    summary: dict = {}

    cov_threshold = st.session_state.get("qc_cov_threshold")
    if cov_threshold is not None:
        summary["BQC CoV threshold"] = f"{cov_threshold}%"

    samples_removed = st.session_state.get("qc_samples_removed", [])
    if samples_removed:
        summary["Outlier samples removed (PCA)"] = ", ".join(samples_removed)
    else:
        summary["Outlier samples removed (PCA)"] = "None"

    return summary


def _render_manual_fields() -> dict:
    """Render text inputs for manual fields and return filled values."""
    manual_fields: dict = {}

    # Initialize session state for manual fields
    if "lsi_manual_fields" not in st.session_state:
        st.session_state["lsi_manual_fields"] = {}

    saved = st.session_state["lsi_manual_fields"]

    field_defs = LSIReportService.get_manual_field_definitions()

    with st.expander("Fill in additional fields (optional)", expanded=False):
        st.markdown(
            "These fields describe your wet-lab and instrument setup. "
            "Fill in what you can — blank fields will be marked as "
            "'to be filled' in the report."
        )

        current_category = ""
        for field_def in field_defs:
            category = field_def["category"]
            item = field_def["item"]
            description = field_def["description"]

            if category != current_category:
                current_category = category
                st.markdown(f"**{category}**")

            value = st.text_input(
                item,
                value=saved.get(item, ""),
                help=description,
                key=f"lsi_manual_{item}",
            )

            if value.strip():
                manual_fields[item] = value.strip()
                saved[item] = value.strip()

    st.session_state["lsi_manual_fields"] = saved
    return manual_fields
