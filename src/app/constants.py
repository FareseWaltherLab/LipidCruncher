"""
Shared constants for the LipidCruncher application.

Centralizes constants that were previously duplicated across multiple files.
Lipid nomenclature functions live in lipid_nomenclature.py.
"""
from enum import Enum
from typing import Dict, Tuple

# Re-export lipid nomenclature for backward compatibility
from app.lipid_nomenclature import (  # noqa: F401
    HYDROXYL_PATTERN,
    LYSO_CLASSES,
    SINGLE_CHAIN_CLASSES,
    SPHINGOLIPID_CLASSES,
    format_lipid_name,
    get_chain_separator,
    normalize_hydroxyl,
    parse_lipid_name,
    remove_phantom_chains,
    sort_chains_lipid_maps,
)


# =============================================================================
# Application Routing (enum-based for type safety)
# =============================================================================

class Page(str, Enum):
    """Application page identifiers."""
    LANDING = "landing"
    APP = "app"


class Module(str, Enum):
    """Application module identifiers."""
    DATA_PROCESSING = "Data Cleaning, Filtering, & Normalization"
    QC_ANALYSIS = "Quality Check & Analysis"


# Legacy aliases — existing code can still use these; they are now enum values
PAGE_LANDING = Page.LANDING
PAGE_APP = Page.APP
MODULE_DATA_PROCESSING = Module.DATA_PROCESSING
MODULE_QC_ANALYSIS = Module.QC_ANALYSIS


# =============================================================================
# Format Display Strings
# =============================================================================

# Display names used in UI selectboxes, content dicts, and session state.
# Always use these constants instead of raw strings.
FORMAT_LIPIDSEARCH: str = 'LipidSearch 5.0'
FORMAT_MSDIAL: str = 'MS-DIAL'
FORMAT_GENERIC: str = 'Generic Format'
FORMAT_METABOLOMICS_WORKBENCH: str = 'Metabolomics Workbench'

# Ordered list for format selector dropdown (tuple for immutability)
FORMAT_OPTIONS: Tuple[str, ...] = (
    FORMAT_GENERIC,
    FORMAT_METABOLOMICS_WORKBENCH,
    FORMAT_LIPIDSEARCH,
    FORMAT_MSDIAL,
)


# =============================================================================
# Format Mappings
# =============================================================================

def get_format_display_to_enum() -> Dict:
    """Return display string → DataFormat enum mapping.

    Lazy import to avoid circular dependency between constants and services.
    """
    from app.services.format_detection import DataFormat
    return {
        FORMAT_LIPIDSEARCH: DataFormat.LIPIDSEARCH,
        FORMAT_MSDIAL: DataFormat.MSDIAL,
        FORMAT_GENERIC: DataFormat.GENERIC,
        FORMAT_METABOLOMICS_WORKBENCH: DataFormat.METABOLOMICS_WORKBENCH,
    }


def resolve_format_enum(format_type: str) -> 'DataFormat':
    """Resolve a format display string or DataFormat to a DataFormat enum.

    Accepts either a display string (e.g., 'LipidSearch 5.0') or a DataFormat
    enum value. Returns DataFormat.GENERIC as fallback.

    Args:
        format_type: Display string or DataFormat enum.

    Returns:
        DataFormat enum value.
    """
    from app.services.format_detection import DataFormat
    if isinstance(format_type, DataFormat):
        return format_type
    return get_format_display_to_enum().get(format_type, DataFormat.GENERIC)


# =============================================================================
# Internal Standard Patterns
# =============================================================================

# Regex patterns to detect internal standards in lipid names (LipidMolec column)
INTERNAL_STANDARD_LIPID_PATTERNS: Tuple[str, ...] = (
    r'\(d\d+\)',        # Deuterium labels: (d5), (d7), (d9)
    r'[+-]D\d+',        # +D7, -D7 patterns
    r'-d\d+[_\)\s]',    # Alternative deuterium: -d7_, -d9)
    r'\[d\d+\]',        # Bracketed deuterium: [d7]
    r'^Ch-D\d+',        # Cholesterol deuterated standards
    r'ISTD',            # ISTD marker
    r'SPLASH',          # SPLASH lipidomix standards
    r':\(s\)',          # :(s) notation
    r'\(IS\)',          # (IS) marker
    r'_IS$',            # Suffix _IS
)

# Regex pattern to detect internal standards in ClassKey column
INTERNAL_STANDARD_CLASS_PATTERN: str = r'ISTD|Internal'


# =============================================================================
# Default Thresholds
# =============================================================================

# LipidSearch 5.0 noise floor — values at or below this are treated as zero
LIPIDSEARCH_DETECTION_THRESHOLD: float = 30000.0

# Coefficient of variation threshold for BQC quality assessment (%)
COV_THRESHOLD_DEFAULT: int = 30

# Zero filtering defaults (displayed as percentages in UI)
ZERO_FILTER_NON_BQC_DEFAULT: int = 75   # 75% → 0.75 proportion
ZERO_FILTER_BQC_DEFAULT: int = 50       # 50% → 0.50 proportion


# =============================================================================
# LipidSearch Grade Filtering
# =============================================================================

# All possible grade values for LipidSearch 5.0
LIPIDSEARCH_GRADE_OPTIONS: Tuple[str, ...] = ('A', 'B', 'C', 'D')

# Default grades per class (classes not listed default to LIPIDSEARCH_DEFAULT_GRADES)
LIPIDSEARCH_DEFAULT_GRADES: Tuple[str, ...] = ('A', 'B')

# Classes that accept an additional grade (C) by default
LIPIDSEARCH_RELAXED_GRADE_CLASSES: Tuple[str, ...] = ('LPC', 'SM')
LIPIDSEARCH_RELAXED_GRADES: Tuple[str, ...] = ('A', 'B', 'C')


# =============================================================================
# MS-DIAL Quality Filtering
# =============================================================================

# Quality filtering presets: display label → config dict
MSDIAL_QUALITY_PRESETS: Dict[str, Dict] = {
    'Strict (Score ≥80, MS/MS required)': {'total_score_threshold': 80, 'require_msms': True},
    'Moderate (Score ≥60)': {'total_score_threshold': 60, 'require_msms': False},
    'Permissive (Score ≥40)': {'total_score_threshold': 40, 'require_msms': False},
    'No filtering': {'total_score_threshold': 0, 'require_msms': False},
}

# Default quality level selection
MSDIAL_DEFAULT_QUALITY_LEVEL: str = 'Moderate (Score ≥60)'


# =============================================================================
# Plotting Constants
# =============================================================================

# Colorblind-friendly palette for conditions (shared across all plotting services)
CONDITION_COLORS: Tuple[str, ...] = (
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
)


# =============================================================================
# Correlation Thresholds
# =============================================================================

CORRELATION_VMIN: float = 0.5
CORRELATION_THRESHOLD_BIOLOGICAL: float = 0.7
CORRELATION_THRESHOLD_TECHNICAL: float = 0.8


# =============================================================================
# PCA Constants
# =============================================================================

PCA_N_COMPONENTS: int = 2