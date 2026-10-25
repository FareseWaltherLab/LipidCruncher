"""
PDF report generator for LipidCruncher analysis results.

Collects all generated plots (QC + analysis) and renders them into a
multi-page PDF using ReportLab. Handles both Plotly and Matplotlib figures.
"""
import copy
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import plotly.io as pio
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)


@dataclass
class ReportMetadata:
    """Metadata for the PDF report cover page."""
    format_type: Optional[str] = None
    n_conditions: Optional[int] = None
    total_samples: Optional[int] = None
    conditions_detail: List[Tuple[str, int, List[str]]] = field(default_factory=list)


# ── Analysis pages ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisPage:
    """How one kind of analysis figure appears in the report.

    Attributes:
        key: Its ``analysis_all_plots`` key. A key ending in '_' is a family
            with one figure per condition or class (``pie_Control``).
        title: Page caption. For a family, ``{}`` is the condition or class
            and ``{kind}`` the saturation plot type.
        cover: Cover-page entry. For a family, ``{}`` is how many conditions
            or classes it has.
        orientation: 'landscape' or 'portrait'; None for the heatmap, which
            has its own page layout.
        is_matplotlib: The figure is Matplotlib rather than Plotly.
    """
    key: str
    title: str
    cover: str
    orientation: Optional[str] = 'landscape'
    is_matplotlib: bool = False


# Every analysis figure the report can hold, in page order. The analysis
# pages store their figures under these keys.
ANALYSIS_PAGES: Tuple[AnalysisPage, ...] = (
    AnalysisPage('bar_chart', "Abundance Bar Chart", "Class Concentration Bar Chart"),
    AnalysisPage('pie_', "Abundance Pie Chart for {}",
                 "Class Concentration Pie Charts ({} condition(s))", 'portrait'),
    AnalysisPage('sat_', "Saturation Plot ({kind}) for {}",
                 "Saturation Profiles ({} class(es))"),
    AnalysisPage('chain_length', "Chain Length Distribution", "Chain Length Distribution"),
    AnalysisPage('fach', "Fatty Acid Composition Heatmap", "Fatty Acid Composition Heatmap"),
    AnalysisPage('pathway', "Lipid Pathway Visualization", "Lipid Pathway Visualization"),
    AnalysisPage('volcano', "Volcano Plot", "Volcano Plot Analysis"),
    AnalysisPage('conc_vs_fc', "Concentration vs. Fold Change",
                 "Concentration vs. Fold Change"),
    AnalysisPage('lipid_distribution', "Individual Lipid Concentration Distribution",
                 "Individual Lipid Concentration Distribution", 'portrait', True),
    AnalysisPage('heatmap', "Lipidomic Heatmap", "Lipidomic Heatmap", None),
)


def _pages_of(page: AnalysisPage, plots: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The ``(key, title)`` of each figure of ``page`` in ``plots``, in order.

    Pie charts go by condition name; saturation plots by class, the
    concentration plot before the percentage plot.
    """
    if not page.key.endswith('_'):
        return [(page.key, page.title)] if page.key in plots else []
    if page.key == 'sat_':
        return [
            (f'sat_{kind}_{cls}', page.title.format(cls, kind=kind.capitalize()))
            for cls in _get_saturation_classes(plots)
            for kind in ('concentration', 'percentage')
            if f'sat_{kind}_{cls}' in plots
        ]
    return [
        (key, page.title.format(key[len(page.key):]))
        for key in sorted(k for k in plots if k.startswith(page.key))
    ]


# ── Image Conversion ────────────────────────────────────────────────────────

_PLOTLY_EXPORT_LANDSCAPE = dict(format='png', width=1000, height=700, scale=2)
_PLOTLY_EXPORT_PORTRAIT = dict(format='png', width=800, height=600, scale=2)
_PLOTLY_EXPORT_HEATMAP = dict(format='png', width=900, height=1200, scale=2)


def _plotly_to_image(fig: Any, export_params: dict) -> bytes:
    """Convert a Plotly figure to PNG bytes."""
    return pio.to_image(fig, **export_params)


def _matplotlib_to_image(fig: Any) -> bytes:
    """Convert a Matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def _figure_to_image_reader(fig: Any, is_matplotlib: bool,
                            export_params: dict) -> ImageReader:
    """Convert any figure to a ReportLab ImageReader."""
    if is_matplotlib:
        img_bytes = _matplotlib_to_image(fig)
    else:
        img_bytes = _plotly_to_image(fig, export_params)
    return ImageReader(io.BytesIO(img_bytes))


# ── Page Rendering Helpers ──────────────────────────────────────────────────

def _render_cover_page(pdf: canvas.Canvas, metadata: ReportMetadata,
                       analyses: List[str]) -> None:
    """Render the report cover page."""
    page_width, page_height = letter

    # Title
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawCentredString(page_width / 2, page_height - 100,
                          "LipidCruncher Analysis Report")

    # Subtitle
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(page_width / 2, page_height - 130,
                          "Quality Check & Lipidomic Analysis Summary")

    # Horizontal line
    pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
    pdf.setLineWidth(1)
    pdf.line(50, page_height - 150, page_width - 50, page_height - 150)

    y = page_height - 190

    # Report info
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Report Information")
    y -= 25

    pdf.setFont("Helvetica", 11)
    pdf.drawString(70, y,
                   f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    y -= 20

    if metadata.format_type:
        pdf.drawString(70, y, f"Data Format: {metadata.format_type}")
        y -= 20

    # Experimental design
    if metadata.n_conditions is not None:
        y -= 15
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Experimental Design")
        y -= 25

        pdf.setFont("Helvetica", 11)
        pdf.drawString(70, y,
                       f"Number of Conditions: {metadata.n_conditions}")
        y -= 20
        if metadata.total_samples is not None:
            pdf.drawString(70, y,
                           f"Total Samples: {metadata.total_samples}")
            y -= 25

        # Conditions table
        if metadata.conditions_detail:
            y = _render_conditions_table(pdf, y, page_width,
                                         metadata.conditions_detail)

    # Analyses list
    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Analyses Included in This Report")
    y -= 25

    pdf.setFont("Helvetica", 10)
    for analysis in analyses:
        if y < 80:
            break
        pdf.drawString(70, y, f"\u2022 {analysis}")
        y -= 16

    # Footer
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawCentredString(page_width / 2, 50,
                          "Generated by LipidCruncher - The Farese and Walther Lab")
    pdf.drawCentredString(page_width / 2, 38,
                          "https://github.com/FareseWaltherLab/LipidCruncher")


# Sample IDs column of the cover page's conditions table: at most this many
# IDs per condition, wrapped onto at most this many lines.
_COVER_SAMPLE_IDS_SHOWN = 5
_COVER_SAMPLE_IDS_MAX_LINES = 3
_COVER_LINE_HEIGHT = 12


def _render_conditions_table(pdf: canvas.Canvas, y: float,
                             page_width: float,
                             details: List[Tuple[str, int, List[str]]]) -> float:
    """Render the conditions table on the cover page. Returns updated y."""
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(70, y, "Condition")
    pdf.drawString(250, y, "Samples")
    pdf.drawString(400, y, "Sample IDs")
    y -= 5
    pdf.line(70, y, page_width - 50, y)
    y -= 15

    pdf.setFont("Helvetica", 10)
    ids_width = page_width - 50 - 400
    for condition, n_samples, samples in details:
        if y < 150:
            break
        pdf.drawString(70, y, str(condition)[:25])
        pdf.drawString(250, y, str(n_samples))
        lines = _sample_id_lines(samples, ids_width)
        for i, line in enumerate(lines):
            pdf.drawString(400, y - i * _COVER_LINE_HEIGHT, line)
        y -= 18 + max(0, len(lines) - 1) * _COVER_LINE_HEIGHT
    return y


def _sample_id_lines(samples: List[str], max_width: float) -> List[str]:
    """Wrap a condition's sample IDs to fit the Sample IDs column.

    Lists the first few IDs and counts the rest (``"s1, s2 ... (+3 more)"``),
    breaking after a comma onto up to three lines so long original sample
    names stay whole rather than being cut off mid-name. IDs that would need
    a fourth line join the count instead.
    """
    def fits(text: str) -> bool:
        return stringWidth(text, "Helvetica", 10) <= max_width

    shown = list(samples[:_COVER_SAMPLE_IDS_SHOWN])
    while True:
        more = len(samples) - len(shown)
        words = [f"{s}," for s in shown[:-1]] + shown[-1:]
        if more:
            words.append(f"... (+{more} more)")
        lines: List[str] = []
        for word in words:
            if lines and fits(f"{lines[-1]} {word}"):
                lines[-1] = f"{lines[-1]} {word}"
            else:
                lines.append(word)
        if len(lines) <= _COVER_SAMPLE_IDS_MAX_LINES or len(shown) <= 1:
            return [_clip(line, fits) for line in lines]
        shown.pop()


def _clip(text: str, fits) -> str:
    """Shorten a line that is still too wide (one very long name) with '...'."""
    if fits(text):
        return text
    while text and not fits(text + "..."):
        text = text[:-1]
    return text + "..."


def _render_plot_page(pdf: canvas.Canvas, fig: Any, title: str,
                      orientation: str, is_matplotlib: bool) -> None:
    """Render a single plot on its own page."""
    pdf.showPage()

    if orientation == 'landscape':
        pdf.setPageSize(landscape(letter))
        page_width, page_height = landscape(letter)
        export_params = _PLOTLY_EXPORT_LANDSCAPE
        img_width, img_height = 700, 500
        img_x, img_y = 50, 100
        title_y = 80
    else:
        pdf.setPageSize(letter)
        page_width, page_height = letter
        export_params = _PLOTLY_EXPORT_PORTRAIT
        img_width, img_height = 500, 400
        img_x, img_y = 50, 100
        title_y = 50

    img = _figure_to_image_reader(fig, is_matplotlib, export_params)
    pdf.drawImage(img, img_x, img_y, width=img_width, height=img_height,
                  preserveAspectRatio=True)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(img_x, title_y, title)


def _render_heatmap_page(pdf: canvas.Canvas, fig: Any, title: str) -> None:
    """Render a heatmap with optimized tall aspect ratio."""
    pdf.showPage()
    pdf.setPageSize(letter)
    page_width, page_height = letter

    fig_copy = copy.deepcopy(fig)
    # Keep a taller top margin if the figure has one: it holds the condition
    # strip and its labels, which would otherwise run into the title.
    top = max(60, fig_copy.layout.margin.t or 0)
    fig_copy.update_layout(
        margin=dict(l=180, r=100, t=top, b=60),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=10)),
    )

    img_bytes = _plotly_to_image(fig_copy, _PLOTLY_EXPORT_HEATMAP)
    img = ImageReader(io.BytesIO(img_bytes))

    available_width = page_width - 60
    available_height = page_height - 90
    img_aspect = 900 / 1200
    page_aspect = available_width / available_height

    if img_aspect > page_aspect:
        w = available_width
        h = available_width / img_aspect
    else:
        h = available_height
        w = available_height * img_aspect

    x = (page_width - w) / 2
    y = 50 + (available_height - h) / 2

    pdf.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(x, y - 20, title)


# ── Analysis List Builder ───────────────────────────────────────────────────

def _build_analyses_list(plots: Dict[str, Any],
                         qc_plots: Dict[str, Any],
                         standards_plots: Optional[List[Any]] = None) -> List[str]:
    """Build the bullet-point list of included analyses for the cover page."""
    items: List[str] = []

    # Internal standards (Module 1)
    if standards_plots:
        items.append(
            f"Internal Standards Consistency ({len(standards_plots)} class(es))"
        )

    # QC analyses
    # _render_qc_plots draws the two box plots on one page, or neither.
    if qc_plots.get('box_plot_fig1') and qc_plots.get('box_plot_fig2'):
        items.append("Distribution Box Plots")
    if qc_plots.get('bqc_plot') is not None:
        items.append("BQC Quality Assessment")
    if qc_plots.get('retention_time_plot') is not None:
        items.append("Retention Time Analysis")
    if qc_plots.get('pca_plot') is not None:
        items.append("Principal Component Analysis (PCA)")
    corr = qc_plots.get('correlation_plots', {})
    if corr:
        items.append(f"Pairwise Correlation ({len(corr)} condition(s))")

    # Analysis plots
    for page in ANALYSIS_PAGES:
        if page.key == 'sat_':
            count = len(_get_saturation_classes(plots))
        else:
            count = len(_pages_of(page, plots))
        if count:
            items.append(page.cover.format(count) if page.key.endswith('_') else page.cover)

    return items


# ── Public API ──────────────────────────────────────────────────────────────

def generate_pdf_report(
    analysis_plots: Dict[str, Any],
    metadata: ReportMetadata,
    qc_plots: Optional[Dict[str, Any]] = None,
    standards_plots: Optional[List[Any]] = None,
) -> Optional[io.BytesIO]:
    """
    Generate a PDF report containing all generated plots.

    Args:
        analysis_plots: Dict from ``analysis_all_plots`` session state,
            keyed as ``ANALYSIS_PAGES`` lists.
        metadata: Report metadata for the cover page.
        qc_plots: Optional dict of QC plots with keys:
            ``box_plot_fig1``, ``box_plot_fig2``, ``bqc_plot``,
            ``retention_time_plot``, ``pca_plot``,
            ``correlation_plots`` (Dict[condition, matplotlib fig]).
        standards_plots: Optional list of internal-standards consistency
            figures from ``standards_consistency_figs`` session state, one
            per standard class.

    Returns:
        BytesIO buffer containing the PDF, or None on error.
    """
    if qc_plots is None:
        qc_plots = {}
    if standards_plots is None:
        standards_plots = []

    analyses = _build_analyses_list(analysis_plots, qc_plots, standards_plots)

    pdf_buffer = io.BytesIO()

    try:
        pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

        # Cover page
        _render_cover_page(pdf, metadata, analyses)

        # Internal standards (Module 1)
        _render_standards_plots(pdf, standards_plots)

        # QC plots
        _render_qc_plots(pdf, qc_plots)

        # Analysis plots (ordered)
        _render_analysis_plots(pdf, analysis_plots)

        pdf.save()
        pdf_buffer.seek(0)
    except Exception as e:
        logger.error("PDF report generation failed: %s", e, exc_info=True)
        raise ValueError(f"PDF report generation failed: {e}") from e

    return pdf_buffer


def _render_standards_plots(pdf: canvas.Canvas,
                            standards_plots: List[Any]) -> None:
    """Render the internal-standards consistency charts, one per page.

    Each figure already carries the class it describes in its own title, so
    that is reused as the page caption rather than re-deriving the class list.
    """
    for idx, fig in enumerate(standards_plots, 1):
        title = _figure_title(fig) or f"Internal Standards Consistency ({idx})"
        _render_plot_page(pdf, fig, title, 'landscape', False)


def _figure_title(fig: Any) -> Optional[str]:
    """Best-effort read of a Plotly figure's own title text."""
    try:
        return fig.layout.title.text or None
    except AttributeError:
        return None


def _render_qc_plots(pdf: canvas.Canvas, qc_plots: Dict[str, Any]) -> None:
    """Render all QC plots in order."""
    # Box plots (two on one page)
    fig1 = qc_plots.get('box_plot_fig1')
    fig2 = qc_plots.get('box_plot_fig2')
    if fig1 and fig2:
        pdf.showPage()
        pdf.setPageSize(letter)
        img1 = _figure_to_image_reader(fig1, False, _PLOTLY_EXPORT_PORTRAIT)
        pdf.drawImage(img1, 50, 400, width=500, height=300,
                      preserveAspectRatio=True)
        img2 = _figure_to_image_reader(fig2, False, _PLOTLY_EXPORT_PORTRAIT)
        pdf.drawImage(img2, 50, 50, width=500, height=300,
                      preserveAspectRatio=True)

    # BQC
    bqc = qc_plots.get('bqc_plot')
    if bqc is not None:
        _render_plot_page(pdf, bqc, "BQC Quality Assessment",
                          'portrait', False)

    # Retention time
    rt = qc_plots.get('retention_time_plot')
    if rt is not None:
        _render_plot_page(pdf, rt, "Retention Time vs. Mass Plot",
                          'landscape', False)

    # PCA
    pca = qc_plots.get('pca_plot')
    if pca is not None:
        _render_plot_page(pdf, pca, "Principal Component Analysis (PCA)",
                          'landscape', False)

    # Correlation plots (matplotlib)
    for condition, corr_fig in qc_plots.get('correlation_plots', {}).items():
        _render_plot_page(pdf, corr_fig,
                          f"Correlation Plot for {condition}",
                          'portrait', True)


def _render_analysis_plots(pdf: canvas.Canvas,
                           plots: Dict[str, Any]) -> None:
    """Render all analysis plots in ``ANALYSIS_PAGES`` order."""
    for page in ANALYSIS_PAGES:
        for key, title in _pages_of(page, plots):
            if page.orientation is None:
                _render_heatmap_page(pdf, plots[key], title)
            else:
                _render_plot_page(pdf, plots[key], title,
                                  page.orientation, page.is_matplotlib)


def _get_saturation_classes(plots: Dict[str, Any]) -> List[str]:
    """Extract unique saturation class names in sorted order."""
    classes = set()
    for key in plots:
        if key.startswith('sat_'):
            # sat_concentration_ClassName or sat_percentage_ClassName
            parts = key.split('_', 2)
            if len(parts) == 3:
                classes.add(parts[2])
    return sorted(classes)


def build_metadata_from_experiment(
    experiment: Any,
    format_type: Optional[str] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> ReportMetadata:
    """
    Build ReportMetadata from an ExperimentConfig object.

    Convenience function so callers don't need to manually construct
    the conditions_detail list. ``display_names`` (optional
    ``{sample label -> text shown}``) puts the samples' shown names in the
    cover page's Sample IDs column; unlisted samples show their label.
    """
    if experiment is None:
        return ReportMetadata(format_type=format_type)

    details = []
    for condition, n_samples, samples in zip(
        experiment.conditions_list,
        experiment.number_of_samples_list,
        experiment.individual_samples_list,
    ):
        details.append((
            condition, n_samples,
            [(display_names or {}).get(s, s) for s in samples],
        ))

    return ReportMetadata(
        format_type=format_type,
        n_conditions=experiment.n_conditions,
        total_samples=len(experiment.full_samples_list),
        conditions_detail=details,
    )
