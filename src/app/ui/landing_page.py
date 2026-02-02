"""
Landing Page UI Components for LipidCruncher.

This module contains the landing page display functions including
logo display, module images, and the main landing page layout.
"""

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image


def _safe_rerun():
    """Rerun the app, compatible with both old and new Streamlit versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

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

    try:
        pdf_path = IMAGES_DIR / filename
        if pdf_path.exists():
            images = convert_from_path(str(pdf_path), dpi=300)
            if images:
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='PNG')
                st.image(img_byte_arr.getvalue(), caption=caption, use_column_width=True)
                return True
    except Exception as e:
        st.warning(f"Could not load {filename}: {str(e)}")
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
            if centered:
                # Fill full available width of container
                st.image(logo, use_column_width=True)
            else:
                st.image(logo, width=720)
        else:
            st.header("LipidCruncher")
    except Exception:
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

    st.markdown("---")


def _display_modules_overview() -> None:
    """Display the three module descriptions with images."""
    st.subheader("🚀 How It Works")
    st.markdown("LipidCruncher guides you through three intuitive modules:")

    # Module 1
    st.markdown("#### Module 1: Filter and Normalize")
    st.markdown("""
    **Get your data analysis-ready in minutes.** Import data from LipidSearch, MS-DIAL, Metabolomics Workbench, or a generic CSV format.
    Define your experiment by assigning samples to conditions, then apply automatic column standardization,
    filtering (duplicates, empty rows, zero values), and flexible normalization (internal standards, protein concentration, or both).
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
    metabolic pathway mapping, clustered heatmaps, and fatty acid composition analysis—all interactive with SVG/CSV export.
    """)
    load_module_image('module3.pdf')

    st.markdown("---")


def _display_call_to_action() -> None:
    """Display the Start Crunching button."""
    st.subheader("🎯 Ready to Crunch?")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        if st.button("🚀 Start Crunching", use_container_width=True, type="primary"):
            st.session_state.page = 'app'
            _safe_rerun()

    st.markdown("---")


def _display_whats_new() -> None:
    """Display the What's New changelog section."""
    st.markdown("#### ✨ What's New in Version 1.2 (January 20, 2026)")

    with st.expander("New Features & Improvements", expanded=False):
        st.markdown("""
        ### Major Features
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
        """)

    with st.expander("⚠️ Bug Fixes", expanded=False):
        st.markdown("""
        ### Statistical Testing Bug (Bonferroni All Post-Hoc)

        *Affected Feature*: Statistical testing with "Bonferroni All" post-hoc correction (Level 2) in Abundance Bar Charts and Saturation Plots

        *Issue*: When using the "Bonferroni All" option for pairwise comparisons, the software incorrectly applied non-parametric tests
        (Mann-Whitney U) even when you selected parametric analysis. This made the analysis overly conservative—combining the strictest
        correction method with the least powerful test type—potentially causing real differences to be missed.

        *Who is affected*: Users who selected Parametric tests (Welch's t-test/ANOVA) AND "Bonferroni All" for Level 2 correction AND had 3+ conditions.

        *Recommendation*: If you used these settings and found fewer significant results than expected, consider re-running your analysis.

        ---

        ### Sphingolipid Classification in Saturation Analysis

        *Problem*: Sphingolipids (Cer, SM, CerG1, CerG2, CerG3) were incorrectly classified as "single-chain" lipids and excluded from
        consolidated format detection. In reality, sphingolipids have two chains: a sphingoid base (e.g., d18:1) and a fatty acyl chain (e.g., 24:0).

        This meant consolidated sphingolipids like Cer(42:1) or SM(34:1) were silently included in saturation analysis without warning,
        potentially producing inaccurate SFA/MUFA/PUFA results.

        *Solution*: Sphingolipids are now correctly recognized as two-chain lipids. Consolidated sphingolipids are detected and flagged,
        allowing users to review and decide whether to include or exclude them.

        ---

        ### Fatty Acid Composition Heatmap Display Issues

        *Issues*: (1) Heatmap cells appeared as stretched rectangles instead of uniform squares. (2) Hover text only worked for some cells.

        *Solution*: We now construct a complete 2D grid matrix, ensuring each Carbon × Double Bond cell renders as a uniform square with
        full hover support. Empty cells display as white.

        ---

        ### Other Fixes
        - Fixed manual sample grouping error when rearranging samples
        - Fixed session state issues causing double-click required for normalization method switching
        - Fixed double-click issue on radio buttons in manage internal standards
        - Fixed number input widgets requiring double entry to update values
        - Fixed pathway visualization missing unit circles (LPA, LCB, CDP-DAG)
        - Fixed SPLASH standard ClassKey inference
        - Fixed markdown parsing of pipe characters in condition names
        - Fixed volcano plot duplicate messages
        - Fixed volcano plot duplicate checkbox in detailed stats
        - Fixed saturation plots not filtering by selected lipid classes
        - Fixed compatibility with older Streamlit versions
        - Changed concentration vs fold change plot download from PNG to CSV for better data accessibility
        - Fixed various MS-DIAL issues (group reordering, data type/quality filter selections, sample column override)
        - Fixed widget state issues causing selections to reset on page navigation (zero filtering, grade filtering, protein input, custom standards)
        """)

    st.markdown("---")


def _display_footer() -> None:
    """Display resources and footer sections."""
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### 📚 Resources")
        st.markdown("""
        - 📄 [Read our paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.28.650893v1)
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
        st.markdown("Starting a new analysis? Refresh the page first to ensure a clean session.")
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
