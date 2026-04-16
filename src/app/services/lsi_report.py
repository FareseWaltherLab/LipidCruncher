"""
LSI (Lipidomics Standards Initiative) compliance report generator.

Generates pre-filled LSI reporting checklists based on data collected during
the LipidCruncher analysis pipeline. Auto-fills fields that the app already
knows and marks remaining fields for manual researcher input.

Reference: Liebisch et al., "Lipidomics needs more standardization",
Nature Metabolism 2019.

Pure logic — no Streamlit dependencies. All methods are static.
"""
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import Color, black, HexColor
from reportlab.pdfgen import canvas

from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.models.statistics import StatisticalTestConfig

logger = logging.getLogger(__name__)

# Placeholder text for fields the researcher must fill in manually
_PLACEHOLDER = "[TO BE FILLED BY RESEARCHER]"

# Display names for normalization methods
_NORM_METHOD_DISPLAY = {
    'none': 'None',
    'internal_standard': 'Internal Standard Normalization',
    'protein': 'Protein Normalization',
    'both': 'Internal Standard + Protein Normalization',
    'total_intensity': 'Total Intensity Normalization',
}

# LSI checklist categories and their manual (blank) fields
_MANUAL_FIELDS: List[Dict[str, str]] = [
    {"category": "Sample Collection", "item": "Organism / tissue type",
     "description": "Species, tissue, cell type, or biofluid"},
    {"category": "Sample Collection", "item": "Storage conditions",
     "description": "Temperature, duration, freeze-thaw cycles"},
    {"category": "Lipid Extraction", "item": "Extraction protocol",
     "description": "e.g., Bligh & Dyer, Folch, MTBE"},
    {"category": "Lipid Extraction", "item": "Solvents and ratios",
     "description": "Solvent system and volume ratios"},
    {"category": "MS Instrument", "item": "Instrument model",
     "description": "Manufacturer and model (e.g., Thermo Q Exactive)"},
    {"category": "MS Settings", "item": "Ion mode",
     "description": "Positive, negative, or both"},
    {"category": "MS Settings", "item": "Scan type",
     "description": "e.g., DDA, DIA, MRM/SRM"},
    {"category": "MS Settings", "item": "Mass resolution",
     "description": "Resolution setting used"},
    {"category": "MS Settings", "item": "Mass range",
     "description": "m/z scan range"},
    {"category": "Chromatography", "item": "Column type",
     "description": "Column name, dimensions, particle size"},
    {"category": "Chromatography", "item": "Mobile phase",
     "description": "Mobile phase A and B composition"},
    {"category": "Chromatography", "item": "Gradient program",
     "description": "Time points and %B"},
    {"category": "Chromatography", "item": "Flow rate",
     "description": "Flow rate in µL/min or mL/min"},
    {"category": "Spike-in Details", "item": "Internal standard vendor",
     "description": "Vendor and catalog numbers"},
    {"category": "Spike-in Details", "item": "Internal standard concentrations",
     "description": "Concentrations and units of spiked standards"},
    {"category": "Data Acquisition", "item": "Acquisition software",
     "description": "e.g., MassHunter, Xcalibur, Analyst"},
]


@dataclass
class LSIReportData:
    """Structured container for all LSI report fields (auto + manual)."""
    auto_fields: Dict[str, Any] = field(default_factory=dict)
    manual_fields: Dict[str, str] = field(default_factory=dict)


class LSIReportService:
    """Generates LSI-compliant reporting checklists. All static methods."""

    @staticmethod
    def collect_auto_fields(
        format_type: str,
        experiment: ExperimentConfig,
        normalization_config: Optional[NormalizationConfig],
        stat_config: Optional[StatisticalTestConfig],
        cleaned_df: pd.DataFrame,
        intsta_df: Optional[pd.DataFrame],
        bqc_label: Optional[str],
        cleaning_params: Dict[str, Any],
        qc_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect all auto-fillable fields into a structured dict.

        Args:
            format_type: Data format string (e.g., "LipidSearch 5.0").
            experiment: Experiment configuration.
            normalization_config: Normalization settings, or None.
            stat_config: Statistical test config, or None.
            cleaned_df: The cleaned/normalized DataFrame.
            intsta_df: Internal standards DataFrame, or None.
            bqc_label: BQC condition label, or None.
            cleaning_params: Data cleaning parameters used.
            qc_summary: QC metrics summary dict.

        Returns:
            Dict with LSI category keys and their auto-filled values.
        """
        fields: Dict[str, Any] = {}

        # ── Software & Nomenclature ──
        fields["Software"] = "LipidCruncher"
        fields["Lipid nomenclature"] = (
            "LIPID MAPS shorthand notation (Liebisch et al. 2020)"
        )

        # ── Data Source ──
        fields["Data format / identification software"] = format_type or "Unknown"

        # ── Experimental Design ──
        fields["Number of conditions"] = experiment.n_conditions
        fields["Condition labels"] = ", ".join(experiment.conditions_list)
        fields["Samples per condition"] = ", ".join(
            f"{cond}: {n}"
            for cond, n in zip(
                experiment.conditions_list,
                experiment.number_of_samples_list,
            )
        )
        fields["Total samples"] = len(experiment.full_samples_list)

        if bqc_label:
            fields["BQC samples"] = f"Label: {bqc_label}"
        else:
            fields["BQC samples"] = "Not used"

        # ── Normalization ──
        if normalization_config is not None:
            method = normalization_config.method
            fields["Normalization method"] = _NORM_METHOD_DISPLAY.get(
                method, method
            )
            if normalization_config.internal_standards:
                mappings = []
                for cls, std in sorted(
                    normalization_config.internal_standards.items()
                ):
                    mappings.append(f"{cls} \u2192 {std}")
                fields["Standard-to-class mappings"] = "; ".join(mappings)
        else:
            fields["Normalization method"] = "Not configured"

        # ── Internal Standards ──
        if intsta_df is not None and not intsta_df.empty:
            if "LipidMolec" in intsta_df.columns:
                standards = intsta_df["LipidMolec"].tolist()
            else:
                standards = intsta_df.iloc[:, 0].tolist()
            fields["Internal standards detected"] = "; ".join(
                str(s) for s in standards
            )
            fields["Number of internal standards"] = len(standards)
        else:
            fields["Internal standards detected"] = "None detected"
            fields["Number of internal standards"] = 0

        # ── Data Cleaning ──
        if cleaning_params:
            for key, value in cleaning_params.items():
                fields[f"Cleaning: {key}"] = str(value)

        # ── Lipid Coverage ──
        if cleaned_df is not None and not cleaned_df.empty:
            fields["Number of lipid species (after filtering)"] = len(cleaned_df)
            if "ClassKey" in cleaned_df.columns:
                classes = sorted(cleaned_df["ClassKey"].unique().tolist())
                fields["Lipid classes detected"] = ", ".join(
                    str(c) for c in classes
                )
                fields["Number of lipid classes"] = len(classes)

        # ── QC Metrics ──
        if qc_summary:
            for key, value in qc_summary.items():
                fields[f"QC: {key}"] = str(value)

        # ── Statistical Testing ──
        if stat_config is not None:
            if stat_config.is_auto_mode():
                fields["Statistical test mode"] = "Auto"
            else:
                test_type = (
                    "Parametric" if stat_config.is_parametric()
                    else "Non-parametric"
                )
                fields["Statistical test type"] = test_type
            fields["Multiple testing correction"] = (
                stat_config.get_correction_display_name()
            )
            fields["Post-hoc method"] = (
                stat_config.get_posthoc_display_name()
            )
            fields["Significance threshold (alpha)"] = stat_config.alpha

        return fields

    @staticmethod
    def get_manual_field_definitions() -> List[Dict[str, str]]:
        """Return the list of manual field definitions.

        Returns:
            List of dicts with 'category', 'item', and 'description' keys.
        """
        return list(_MANUAL_FIELDS)

    @staticmethod
    def generate_checklist_csv(
        auto_fields: Dict[str, Any],
        manual_fields: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a CSV version of the LSI checklist.

        Args:
            auto_fields: Auto-filled fields from collect_auto_fields().
            manual_fields: User-entered manual fields (item -> value).

        Returns:
            CSV string with columns: Category, Item, Value, Source.
        """
        if manual_fields is None:
            manual_fields = {}

        lines = ["Category,Item,Value,Source"]

        # Auto-filled fields
        for item, value in auto_fields.items():
            safe_value = str(value).replace('"', '""')
            lines.append(
                f'"Auto-filled","{item}","{safe_value}","LipidCruncher"'
            )

        # Manual fields
        for field_def in _MANUAL_FIELDS:
            category = field_def["category"]
            item = field_def["item"]
            value = manual_fields.get(item, _PLACEHOLDER)
            safe_value = str(value).replace('"', '""')
            source = "Researcher" if value != _PLACEHOLDER else "Pending"
            lines.append(f'"{category}","{item}","{safe_value}","{source}"')

        return "\n".join(lines)

    @staticmethod
    def generate_checklist_pdf(
        auto_fields: Dict[str, Any],
        manual_fields: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """Generate a PDF checklist with filled + blank fields.

        Args:
            auto_fields: Auto-filled fields from collect_auto_fields().
            manual_fields: User-entered manual fields (item -> value).

        Returns:
            PDF file contents as bytes.
        """
        if manual_fields is None:
            manual_fields = {}

        pdf_buffer = io.BytesIO()
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
        page_width, page_height = letter

        # ── Cover / Title ──
        _render_lsi_title(pdf, page_width, page_height)

        # ── Auto-filled section ──
        y = page_height - 200
        y = _render_section_header(
            pdf, y, page_width, "Auto-filled from LipidCruncher"
        )

        for item, value in auto_fields.items():
            if y < 80:
                _render_footer(pdf, page_width)
                pdf.showPage()
                pdf.setPageSize(letter)
                y = page_height - 60
            y = _render_field_row(pdf, y, item, str(value), is_auto=True)

        # ── Manual section ──
        if y < 140:
            _render_footer(pdf, page_width)
            pdf.showPage()
            pdf.setPageSize(letter)
            y = page_height - 60

        y -= 15
        y = _render_section_header(
            pdf, y, page_width, "To Be Filled by Researcher"
        )

        current_category = ""
        for field_def in _MANUAL_FIELDS:
            if y < 80:
                _render_footer(pdf, page_width)
                pdf.showPage()
                pdf.setPageSize(letter)
                y = page_height - 60

            category = field_def["category"]
            if category != current_category:
                current_category = category
                y -= 5
                pdf.setFont("Helvetica-Bold", 10)
                pdf.setFillColor(HexColor("#333333"))
                pdf.drawString(55, y, category)
                y -= 18

            item = field_def["item"]
            value = manual_fields.get(item, _PLACEHOLDER)
            y = _render_field_row(pdf, y, item, value, is_auto=False)

        _render_footer(pdf, page_width)
        pdf.save()
        pdf_buffer.seek(0)
        return pdf_buffer.read()


# ── PDF Rendering Helpers ──────────────────────────────────────────────────


def _render_lsi_title(
    pdf: canvas.Canvas, page_width: float, page_height: float
) -> None:
    """Render the LSI report title block."""
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawCentredString(
        page_width / 2, page_height - 80,
        "LSI Compliance Report"
    )

    pdf.setFont("Helvetica", 11)
    pdf.drawCentredString(
        page_width / 2, page_height - 105,
        "Lipidomics Standards Initiative — Reporting Checklist"
    )

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawCentredString(
        page_width / 2, page_height - 125,
        "Reference: Liebisch et al., Nature Metabolism 2019"
    )

    # Date
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(
        page_width / 2, page_height - 145,
        f"Generated: {datetime.now().strftime('%B %d, %Y')}"
    )

    # Horizontal rule
    pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
    pdf.setLineWidth(1)
    pdf.line(50, page_height - 160, page_width - 50, page_height - 160)


def _render_section_header(
    pdf: canvas.Canvas, y: float, page_width: float, title: str
) -> float:
    """Render a section header with underline. Returns updated y."""
    pdf.setFont("Helvetica-Bold", 13)
    pdf.setFillColor(HexColor("#1a5276"))
    pdf.drawString(50, y, title)
    y -= 5
    pdf.setStrokeColor(HexColor("#1a5276"))
    pdf.setLineWidth(0.5)
    pdf.line(50, y, page_width - 50, y)
    y -= 18
    pdf.setFillColor(black)
    return y


def _render_field_row(
    pdf: canvas.Canvas, y: float, label: str, value: str, is_auto: bool
) -> float:
    """Render a single checklist row. Returns updated y."""
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(black)
    pdf.drawString(60, y, label)

    if is_auto:
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(black)
    else:
        if value == _PLACEHOLDER:
            pdf.setFont("Helvetica-Oblique", 9)
            pdf.setFillColor(HexColor("#999999"))
        else:
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(black)

    # Truncate long values for display
    max_chars = 80
    display_value = value if len(value) <= max_chars else value[:max_chars] + "..."
    pdf.drawString(250, y, display_value)

    pdf.setFillColor(black)
    y -= 16
    return y


def _render_footer(pdf: canvas.Canvas, page_width: float) -> None:
    """Render footer on the current page."""
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.setFillColor(HexColor("#666666"))
    pdf.drawCentredString(
        page_width / 2, 35,
        "Generated by LipidCruncher \u2014 The Farese and Walther Lab"
    )
    pdf.drawCentredString(
        page_width / 2, 24,
        "LSI Checklist: https://lipidomicstandards.org/reporting_checklist/"
    )
    pdf.setFillColor(black)
