"""
Shared constants for the LipidCruncher application.

Centralizes constants that were previously duplicated across multiple files.
"""
from typing import Dict, List


# =============================================================================
# Format Mappings
# =============================================================================

def get_format_display_to_enum() -> Dict:
    """Return display string → DataFormat enum mapping.

    Lazy import to avoid circular dependency between constants and services.
    """
    from app.services.format_detection import DataFormat
    return {
        'LipidSearch 5.0': DataFormat.LIPIDSEARCH,
        'MS-DIAL': DataFormat.MSDIAL,
        'Generic Format': DataFormat.GENERIC,
        'Metabolomics Workbench': DataFormat.METABOLOMICS_WORKBENCH,
    }


# =============================================================================
# Internal Standard Patterns
# =============================================================================

# Regex patterns to detect internal standards in lipid names (LipidMolec column)
INTERNAL_STANDARD_LIPID_PATTERNS: List[str] = [
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
]

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
