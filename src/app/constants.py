"""
Shared constants for the LipidCruncher application.

Centralizes constants that were previously duplicated across multiple files.
"""
import re
from typing import Dict, FrozenSet, List, Optional, Tuple


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


# =============================================================================
# LipidSearch Grade Filtering
# =============================================================================

# All possible grade values for LipidSearch 5.0
LIPIDSEARCH_GRADE_OPTIONS: List[str] = ['A', 'B', 'C', 'D']

# Default grades per class (classes not listed default to LIPIDSEARCH_DEFAULT_GRADES)
LIPIDSEARCH_DEFAULT_GRADES: List[str] = ['A', 'B']

# Classes that accept an additional grade (C) by default
LIPIDSEARCH_RELAXED_GRADE_CLASSES: List[str] = ['LPC', 'SM']
LIPIDSEARCH_RELAXED_GRADES: List[str] = ['A', 'B', 'C']


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
# LIPID MAPS Nomenclature (Liebisch et al. 2020)
# =============================================================================

# Sphingolipid classes — use '/' as chain separator
SPHINGOLIPID_CLASSES: FrozenSet[str] = frozenset({
    'Cer', 'SM', 'HexCer', 'GlcCer', 'GalCer',
    'CerG1', 'CerG2', 'CerG3',
    'Hex2Cer', 'Hex3Cer',
    'dhCer', 'LCB', 'SPB',
    'CerP', 'SL',
})

# Lipid classes that have only one fatty acid chain
SINGLE_CHAIN_CLASSES: FrozenSet[str] = frozenset({
    'CE', 'ChE',                                    # Cholesteryl esters
    'LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA',     # Lysophospholipids
    'MAG', 'MG',                                     # Monoacylglycerols
    'FFA', 'FA',                                     # Free fatty acids
    'LCB', 'SPB',                                    # Sphingoid bases (single chain)
})

# Lyso-species classes — phantom 0:0 chains should be removed
LYSO_CLASSES: FrozenSet[str] = frozenset({
    'LPC', 'LPE', 'LPG', 'LPI', 'LPS', 'LPA',
})

# Regex for hydroxyl notation in FA chains (matches both old ;2O and new ;O2)
HYDROXYL_PATTERN: re.Pattern = re.compile(r';[\dO()]+')


def normalize_hydroxyl(text: str) -> str:
    """Convert old hydroxyl notation to LIPID MAPS format.

    ;2O → ;O2, ;3O → ;O3, ;1O → ;O, ;O → ;O (unchanged)
    """
    def _replace(m: re.Match) -> str:
        s = m.group(0)  # e.g. ";2O" or ";O2" or ";O"
        # Already in new format: ;O or ;O2 or ;O3
        if re.fullmatch(r';O\d*', s):
            return s
        # Old format: ;2O, ;3O, etc.
        old_match = re.fullmatch(r';(\d+)O', s)
        if old_match:
            count = old_match.group(1)
            return f';O{count}' if count != '1' else ';O'
        # Fallback — return as-is
        return s
    return re.sub(r';[\dO]+', _replace, text)


def parse_lipid_name(lipid_name) -> Tuple[str, str, str]:
    """Parse a standardized LIPID MAPS name into (class, chains, modification).

    Examples:
        'PC 16:0_18:1'        → ('PC', '16:0_18:1', '')
        'Cer 18:1;O2/24:0'   → ('Cer', '18:1;O2/24:0', '')
        'LPC 18:1(d7)'        → ('LPC', '18:1', '(d7)')
        'Ch'                   → ('Ch', '', '')
    """
    if not lipid_name or not isinstance(lipid_name, str):
        return ('', '', '')

    # Extract trailing modification like (d7)
    modification = ''
    mod_match = re.search(r'\(d\d+\)$', lipid_name)
    if mod_match:
        modification = mod_match.group()
        lipid_name = lipid_name[:mod_match.start()].rstrip()

    # Split on first space
    space_idx = lipid_name.find(' ')
    if space_idx > 0:
        class_name = lipid_name[:space_idx]
        chain_info = lipid_name[space_idx + 1:]
    else:
        class_name = lipid_name
        chain_info = ''

    return (class_name, chain_info, modification)


def get_chain_separator(class_name: str) -> str:
    """Return the LIPID MAPS chain separator for a lipid class."""
    return '/' if class_name in SPHINGOLIPID_CLASSES else '_'


def sort_chains_lipid_maps(chains: List[str], class_name: str) -> List[str]:
    """Sort chains per LIPID MAPS rules.

    Sphingolipids: keep original order (sphingoid base first).
    Others: sort by (carbon_count, double_bond_count) ascending.
    """
    if class_name in SPHINGOLIPID_CLASSES:
        return chains  # Positional — don't reorder

    def _sort_key(chain: str) -> Tuple[int, int]:
        # Strip prefixes like O-, P-, d, t, m, C
        cleaned = re.sub(r'^[OPdtm]-?', '', chain)
        cleaned = re.sub(r'^C', '', cleaned)
        # Strip hydroxyl/oxidation suffixes
        cleaned = re.sub(r'[;+].*', '', cleaned)
        m = re.match(r'(\d+):(\d+)', cleaned)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (9999, 9999)

    return sorted(chains, key=_sort_key)


def remove_phantom_chains(chains: List[str]) -> List[str]:
    """Remove phantom 0:0 chains from lyso-species."""
    return [c for c in chains if c != '0:0']


def format_lipid_name(
    class_name: str,
    chains: Optional[List[str]] = None,
    modification: str = '',
) -> str:
    """Build a LIPID MAPS–compliant lipid name.

    Examples:
        format_lipid_name('PC', ['16:0', '18:1']) → 'PC 16:0_18:1'
        format_lipid_name('Cer', ['18:1;O2', '24:0']) → 'Cer 18:1;O2/24:0'
        format_lipid_name('LPC', ['18:1'], '(d7)') → 'LPC 18:1(d7)'
        format_lipid_name('Ch') → 'Ch'
    """
    if not chains or all(c == '' for c in chains):
        return f"{class_name}{modification}"

    sep = get_chain_separator(class_name)
    chain_str = sep.join(chains)
    return f"{class_name} {chain_str}{modification}"
