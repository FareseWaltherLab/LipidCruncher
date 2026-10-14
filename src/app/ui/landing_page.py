"""
Landing Page UI Components for LipidCruncher.

This module contains the landing page display functions including
logo display, module images, and the main landing page layout.
"""

import io
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from app.constants import PAGE_APP, PAGE_AI_STUDIO, PAGE_ISOTOPE_TRACING


try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


# =============================================================================
# Path Configuration
# =============================================================================

# Get the images directory relative to this file
# This file is at: src/app/ui/landing_page.py
# Images are at: src/images/
_UI_DIR = Path(__file__).parent
_APP_DIR = _UI_DIR.parent
_SRC_DIR = _APP_DIR.parent
IMAGES_DIR = _SRC_DIR / "images"


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data(show_spinner=False)
def _convert_pdf_to_png(pdf_path: str) -> Optional[bytes]:
    """Convert a PDF file to PNG bytes (cached).

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        PNG image bytes, or None if conversion failed.
    """
    try:
        images = convert_from_path(pdf_path, dpi=300)
        if images:
            buf = io.BytesIO()
            images[0].save(buf, format='PNG')
            return buf.getvalue()
    except (OSError, ValueError):
        return None
    return None


def load_module_image(filename: str, caption: str = None) -> bool:
    """
    Load and display a module PDF as an image.

    Args:
        filename: Name of the PDF file in the images directory
        caption: Optional caption to display below the image

    Returns:
        True if image was loaded successfully, False otherwise
    """
    if not PDF2IMAGE_AVAILABLE:
        st.warning(f"pdf2image not installed. Cannot display {filename}")
        return False

    pdf_path = IMAGES_DIR / filename
    if not pdf_path.exists():
        return False

    png_bytes = _convert_pdf_to_png(str(pdf_path))
    if png_bytes is not None:
        st.image(png_bytes, caption=caption, width='stretch')
        return True

    st.warning(f"Could not load {filename}")
    return False


def display_logo(centered: bool = False):
    """Display the LipidCruncher logo.

    Args:
        centered: If True, use full available width
    """
    try:
        logo_path = IMAGES_DIR / "new_logo.tif"
        if logo_path.exists():
            logo = Image.open(logo_path)
            if logo.mode == 'CMYK':
                logo = logo.convert('RGB')
            if centered:
                # Fill full available width of container
                st.image(logo, width='stretch')
            else:
                st.image(logo, width=720)
        else:
            st.header("LipidCruncher")
    except (OSError, ValueError):
        st.header("LipidCruncher")


# =============================================================================
# Main Landing Page - Helper Functions
# =============================================================================

def _display_hero_section() -> None:
    """Display logo, tagline, and quick highlights."""
    display_logo(centered=True)

    st.markdown("""
    *An open-source platform for processing, visualizing, and analyzing lipidomic data.*

    Built by [The Farese & Walther Lab](https://www.mskcc.org/research/ski/labs/farese-walther)
    to bridge the gap between lipidomic data generation and biological insight—no bioinformatics expertise required.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**📂 4 Formats**<br>LipidSearch, MS-DIAL, Generic, Metabolomics Workbench", unsafe_allow_html=True)
    col2.markdown("**🔬 QC + Normalization**<br>Integrated quality control with flexible normalization", unsafe_allow_html=True)
    col3.markdown("**📊 Lipid-Specific Viz**<br>Saturation profiles, pathway maps, lipidomic heatmap", unsafe_allow_html=True)
    col4.markdown("**📈 High-Quality Outputs**<br>Interactive plots, SVG export, PDF reports", unsafe_allow_html=True)

    st.markdown(
        "Alongside the main pipeline: **🧪 Stable Isotope Tracing** for ¹³C-labelled "
        "experiments, and **🤖 AI Studio** for exploring any dataset through a "
        "conversation with AI."
    )

    st.markdown("---")


def _display_modules_overview() -> None:
    """Display the three pipeline modules, then the two workflows that sit
    alongside them (isotope tracing and AI Studio)."""
    st.subheader("🚀 How It Works")
    st.markdown("The main lipidomics pipeline guides you through three intuitive modules:")

    # Module 1
    st.markdown("#### Module 1: Filter and Normalize")
    st.markdown("""
    **Get your data analysis-ready in minutes.** Import data from LipidSearch, MS-DIAL, Metabolomics Workbench, or a generic CSV format.
    Define your experiment by assigning samples to conditions, then apply automatic column standardization,
    filtering (duplicates, empty rows, zero values), and flexible normalization (internal standards, protein concentration, both, or total intensity).
    Internal standards consistency plots help verify sample preparation and instrument performance.
    """)
    load_module_image('module1.pdf')

    st.markdown("---")

    # Module 2
    st.markdown("#### Module 2: Quality Check")
    st.markdown("""
    **Trust your data before you analyze it.** Box plots assess data quality and validate normalization—replicates
    within a condition should exhibit similar medians and interquartile ranges. CoV analysis of batch quality control (BQC)
    samples evaluates measurement precision. Correlation heatmaps and PCA detect outliers and visualize sample clustering.
    """)
    load_module_image('module2.pdf')

    st.markdown("---")

    # Module 3
    st.markdown("#### Module 3: Visualize and Analyze")
    st.markdown("""
    **Turn complex lipid profiles into biological insights.** Bar & pie charts, volcano plots, saturation profiles (SFA, MUFA, PUFA),
    chain length distributions, metabolic pathway mapping, lipidomic heatmaps (Z-score or log2 fold change), and fatty acid
    composition analysis—all interactive with SVG/CSV export, plus a PDF report and a pre-filled LSI compliance checklist.
    """)
    load_module_image('module3.pdf')

    st.markdown("---")

    # The two workflows that sit alongside the pipeline rather than inside it.
    # Grouped under one subheader, at the same heading depth as each other, so
    # neither reads as a fourth module of the pipeline above.
    st.subheader("🧭 Beyond the Main Pipeline")
    st.markdown(
        "Two workflows run alongside those three modules. Neither is a step in "
        "the pipeline — each is its own page, reachable from the bottom of this one."
    )

    st.markdown("#### 🧪 Stable Isotope Tracing")
    st.markdown("""
    **A separate workflow for stable isotope (e.g. ¹³C) tracing experiments.** Upload IsoCorrectoR's three
    input files (Measurement, Molecule, Element) and run natural-isotope-abundance correction with the real
    IsoCorrectoR package—removing the naturally-occurring heavy-isotope background so the remaining signal
    reflects only the label you fed in. View and download the corrected intensities, fractions, mean
    enrichment, and residuals.
    """)

    st.markdown("#### 🤖 AI Studio")
    st.markdown("""
    **Explore any dataset through a conversation.** Four tabs: ask how LipidCruncher itself works; ask
    questions about your own data and get the answer computed on the spot; convert an unsupported CSV into
    the Generic format so the main pipeline can take it; or describe a figure and have it generated. Bring
    data in by uploading a CSV or by importing what the main pipeline is already holding—raw, normalized,
    or QC-filtered—and download transformed results to feed back into it.
    """)

    st.markdown("---")


def _display_call_to_action() -> None:
    """Display the three destinations as one stacked column.

    General Lipidomics is the primary call to action; isotope tracing and AI
    Studio follow as secondary buttons, each on its own row. Every button gets
    the same one-line caption underneath, so the three read as one list of
    starting points rather than two buttons plus a paragraph.
    """
    st.subheader("🎯 Ready to Crunch?")
    st.markdown("**Where do you want to start?**")

    _, row, _ = st.columns([1, 2, 1])
    with row:
        if st.button("🧬 General Lipidomics", width='stretch', type="primary"):
            st.session_state.page = PAGE_APP
            st.rerun()
        st.caption("Filter, normalize, QC, visualize, and analyze lipidomics data.")

    _, row, _ = st.columns([1, 2, 1])
    with row:
        if st.button("🧪 Stable Isotope Tracing", width='stretch'):
            st.session_state.page = PAGE_ISOTOPE_TRACING
            st.rerun()
        st.caption("Natural-isotope-abundance correction via IsoCorrectoR.")

    _, row, _ = st.columns([1, 2, 1])
    with row:
        if st.button("🤖 Open AI Studio", width='stretch'):
            st.session_state.page = PAGE_AI_STUDIO
            st.rerun()
        st.caption("Convert unsupported formats to Generic, and explore any dataset with AI.")

    st.markdown("---")


def _display_whats_new() -> None:
    """Display the What's New changelog section."""
    st.markdown("#### ✨ What's New in Version 1.4 (August 19, 2026)")

    with st.expander("New Features & Improvements", expanded=False):
        st.markdown("""
        ### Major Features
        - **Stable Isotope Tracing**: A separate workflow for stable isotope (e.g. ¹³C) tracing
          experiments—upload IsoCorrectoR's three input files and run natural-isotope-abundance
          correction with the real IsoCorrectoR package, then view and download the corrected
          intensities, fractions, mean enrichment, and residuals
        - **AI Studio**: A conversational workspace with four tabs—questions about LipidCruncher,
          questions about your own data, converting an unsupported CSV into the Generic format,
          and custom visualizations—working either on an uploaded file or on data imported from
          the main pipeline (raw, normalized, or QC-filtered)
        - **LipidSearch 5.2 Support**: The LipidSearch format (renamed from "LipidSearch 5.0")
          now covers both 5.0 and 5.2 exports—tab and comma delimiters are detected automatically,
          and condition-grouped dual-polarity exports are merged into single samples using the
          Alignment Setting file, which also pre-fills your experiment setup
        - **Chain Length Distribution**: New class-level bubble chart showing how carbon chain
          length is distributed within each lipid class, plotted per condition and included in
          the PDF report
        - **LSI Compliance Report**: Generate a pre-filled Lipidomics Standards Initiative
          reporting checklist from your session, downloadable as PDF or CSV
        - **Redesigned Lipidomic Heatmaps**: Two new modes—**Grouped by Class** (one row per
          species, blocked by lipid class) and **Aggregated by Class** (one row per class,
          summing its species)—with square cells, colour-coded condition strips, and paging
          so large class selections stay readable
        - **Log2 Fold Change Heatmaps**: Every heatmap mode can now be coloured by log2 fold
          change against a control condition instead of Z-score, so colour reads as the
          direction and size of change rather than a position within each row's own spread
        - **Fold-Change Sorted Species**: In Grouped by Class, species are ranked inside their
          class block from the largest increase down to the largest decrease (reversible),
          turning each block into a gradient that lines up with the colour bar
        - **Internal Standards from Your Dataset**: Choose standards already present in your
          data alongside the automatically detected ones

        ### Bug Fixes
        - Fixed sample regrouping being re-applied on every page interaction, and undo restoring
          a reshuffled dataset instead of the original
        - Fixed three plots—concentration vs. fold change, individual lipid distribution, and
          internal standards consistency—rendering on screen but never reaching the PDF report
        - Fixed saturation plot exports keeping significance asterisks after they were switched
          off on screen
        - Fixed significance asterisks not appearing on saturation plots, and bar chart
          significance indicators drifting out of line with their bars
        - Fixed the heatmap silently mislabelling samples when a selected sample had no
          concentration column
        - Fixed the LipidSearch cleaner discarding columns from extended exports
        - Fixed custom standards upload failing after the LIPID MAPS notation change, and added
          specific error messages when an upload is rejected
        - Fixed SPLASH standard class inference for chain-prefixed names
        - Fixed the MS-DIAL MS/MS filter not syncing with the selected preset, and data type
          switching not taking effect
        - Fixed Metabolomics Workbench sample data not auto-populating, and a crash when the
          detection threshold was left empty
        - Fixed pathway visualization circle sizing and volcano plot label placement
        - Fixed lipid classes being indistinguishable in plots by expanding the class colour
          palette from 10 to 20 colours

        ### UI/UX Improvements
        - **Reset Session**: One sidebar button clears your experiment setup, processing and
          cached results and returns you to Define Experiment—without refreshing the page or
          re-uploading your data
        - **Collapsed Analysis Sections**: Landing on Visualize and Analyze no longer opens a
          section on its own
        - **Sample Names in the Sidebar**: The Group Samples table shows each sample's original
          column header next to its label, so you can group without cross-referencing your file
        - **Blanket Grade Filtering**: Relax LipidSearch grades to A/B/C for every lipid class
          at once, instead of setting each class individually
        - **Corrected Format Documentation**: The data-requirements and processing panels now
          match what the code actually does for the Generic, MS-DIAL and LipidSearch formats
        - **Clearer Heatmap Documentation**: Inline notes explaining how each cell is computed,
          how class totals are aggregated, and how the rows are ordered
        - **Accurate Upload Limit**: The uploader tooltip claimed 800MB while the app was already
          accepting 1GB files. Both now agree on 1GB, and the limit is pinned to the app itself so
          the two cannot drift apart again
        """)

    with st.expander("Version 1.3 (March 30, 2026)", expanded=False):
        st.markdown("""
        ### Major Features
        - **LIPID MAPS Nomenclature**: Full compliance with the LIPID MAPS shorthand notation system
          (Liebisch et al. 2020)—all lipid names, labels, and documentation now follow the community standard
        - **Overhauled Pathway Visualization**: Expanded from 18 to 28 lipid classes with a fully data-driven,
          editable layout—drag classes to reposition, save/load custom configurations, and choose from built-in presets
        - **Interactive Pathway Plots**: Migrated pathway visualization from static Matplotlib to interactive Plotly
          with hover details, zoom, and pan
        - **Total Intensity Normalization**: New normalization method that scales each sample by its total lipid
          signal—useful when internal standards are unavailable

        ### Bug Fixes
        - Fixed statistical testing using inconsistent zero-replacement values across experimental groups,
          which could slightly shift t-statistics and p-values in saturation, bar chart, and volcano analyses.
          Zero replacement now uses a single dataset-wide detection floor derived from the smallest non-zero
          concentration in the dataset.

        ### UI/UX Improvements
        - **Auto-Populated Experiment Config**: Loading a sample dataset automatically fills in condition names
          and sample counts—no manual setup needed
        - **Clearer Error Messages**: All error messages rewritten to be specific and actionable instead of
          showing raw Python exceptions
        - **Consistent Formula Rendering**: All normalization formulas displayed uniformly using code blocks
          for better readability
        - **Updated Format Requirements**: Standardization examples and documentation aligned with LIPID MAPS
          nomenclature
        - **Streamlit 1.50 Upgrade**: Updated from Streamlit 1.22 to 1.50 for improved performance,
          stability, and modern widget behavior
        """)

    with st.expander("Version 1.2 (January 20, 2026)", expanded=False):
        st.markdown("""
        ### New Features
        - **Increased File Upload Limit**: Now supports files up to 800MB for larger lipidomics datasets
        - **MS-DIAL Format Support**: Full integration with MS-DIAL exports including quality filtering (Total score, MS/MS matched),
          dual data type selection (raw vs. pre-normalized), and automatic internal standards detection
        - **External Standards Upload**: Upload complete standards files with intensity values—standards no longer need to exist in your dataset
        - **Configurable Zero Filtering**: Adjustable thresholds for lipid filtering (non-BQC: 50-100%, BQC: 25-100%)

        ### UI/UX Improvements
        - **One-Click Sample Data Loading**: Try LipidCruncher instantly with built-in test datasets for all formats—no download required
        - **Streamlined Generic Format**: Simplified column mapping—auto-detection handles it reliably
        - **Redesigned Landing Page**: New module images, concise descriptions, and cleaner visual hierarchy
        - **Streamlined Data Processing**: Collapsible format requirements, improved column mapping validation
        - **Consolidated Statistical Documentation**: Central "About Statistical Testing" expander replaces duplicated explanations
        - **Better Statistical Defaults**: FDR for Level 1 correction, Tukey's HSD for Level 2 (renamed from "Standard")
        - **Cleaner Analysis Sections**: Consistent headers, two-column layouts, side-by-side download buttons throughout
        - **Improved PDF Reports**: Metadata cover page, fixed empty pages, better heatmap scaling for many lipids
        - **Box Plots by Condition**: Samples now colored by experimental condition with colorblind-friendly palette
        - **Internal Standards Visualization**: Separate subplots for multiple standards within the same class

        ### Bug Fixes
        - Fixed statistical testing incorrectly applying non-parametric tests with "Bonferroni All" post-hoc correction
        - Fixed sphingolipids incorrectly classified as single-chain in saturation analysis
        - Fixed fatty acid composition heatmap cells rendering as stretched rectangles with broken hover text
        - Fixed manual sample grouping error when rearranging samples
        - Fixed session state issues causing double-click required for normalization method switching
        - Fixed pathway visualization missing unit circles (LPA, LCB, CDP-DAG)
        - Fixed SPLASH standard ClassKey inference
        - Fixed volcano plot duplicate messages and duplicate checkbox in detailed stats
        - Fixed saturation plots not filtering by selected lipid classes
        - Fixed various MS-DIAL issues (group reordering, data type/quality filter selections, sample column override)
        - Fixed widget state issues causing selections to reset on page navigation
        """)

    st.markdown("---")


def _display_footer() -> None:
    """Display resources and footer sections."""
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### 📚 Resources")
        st.markdown("""
        - 📄 [Read our paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v2.article-metrics)
        - 💻 [Source code on GitHub](https://github.com/FareseWaltherLab/LipidCruncher)
        """)

    with res_col2:
        st.markdown("#### 🧪 Try Our Test Data")
        st.markdown("""
        No dataset? No problem! Sample data available for all formats:
        - **Generic/LipidSearch**: ADGAT-DKO adipose tissue study
        - **MS-DIAL**: Mouse adrenal gland fads2 KO vs WT
        - **Metabolomics Workbench**: Mouse HFD serum study

        Click **"Load Sample Data"** in the sidebar after selecting your format!
        """)

    st.markdown("---")

    foot_col1, foot_col2 = st.columns(2)
    with foot_col1:
        st.markdown("#### 💡 Pro Tip")
        st.markdown(
            "Starting a new analysis? Use **🔄 Reset session** in the sidebar to clear your "
            "setup and results while keeping the uploaded file—or refresh the page to start "
            "completely fresh."
        )
    with foot_col2:
        st.markdown("#### 📧 Support")
        st.markdown("Questions, bugs, or feature requests? Email **abdih@mskcc.org**")


# =============================================================================
# Main Landing Page
# =============================================================================

def display_landing_page():
    """Display the LipidCruncher landing page with centered, narrower layout."""
    _, center, _ = st.columns([1, 3, 1])

    with center:
        _display_hero_section()
        _display_modules_overview()
        _display_call_to_action()
        _display_whats_new()
        _display_footer()
