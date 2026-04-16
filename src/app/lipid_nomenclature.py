"""
LIPID MAPS nomenclature utilities (Liebisch et al. 2020).

Parsing, formatting, and chain-sorting functions for standardized lipid names.
Separated from constants.py to keep that file purely declarative.
"""
import re
from typing import FrozenSet, List, Optional, Tuple


# =============================================================================
# Lipid Class Sets
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


# =============================================================================
# Nomenclature Functions
# =============================================================================

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
    return HYDROXYL_PATTERN.sub(_replace, text)


def parse_lipid_name(lipid_name: str) -> Tuple[str, str, str]:
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
        parsed = parse_chain_carbon_db(chain)
        if parsed is not None:
            return parsed
        return (9999, 9999)

    return sorted(chains, key=_sort_key)


def remove_phantom_chains(chains: List[str]) -> List[str]:
    """Remove phantom 0:0 chains from lyso-species."""
    return [c for c in chains if c != '0:0']


def parse_chain_carbon_db(chain: str) -> Optional[Tuple[int, int]]:
    """Extract (carbon_count, double_bond_count) from a single chain string.

    Strips known prefixes (O-, P-, d, t, m, C), hydroxyl/oxidation suffixes,
    and then matches the carbon:db pattern.

    Examples:
        '16:0'      → (16, 0)
        'd18:1'     → (18, 1)
        'O-30:1'    → (30, 1)
        'P-16:0'    → (16, 0)
        '18:1;O2'   → (18, 1)
        '20:4+O'    → (20, 4)
        'C24:0'     → (24, 0)

    Returns:
        (carbons, double_bonds) or None if the chain cannot be parsed.
    """
    # Strip ether/sphingoid prefixes: O-, P-, d, t, m
    cleaned = re.sub(r'^[OPdtm]-?', '', chain)
    # Strip chain identifier C (e.g., C24:0)
    cleaned = re.sub(r'^C', '', cleaned)
    # Strip hydroxyl notation (e.g. ;O2, ;2O)
    cleaned = HYDROXYL_PATTERN.sub('', cleaned)
    # Strip oxidation suffixes (e.g. +O, +2O)
    cleaned = re.sub(r'\+\d*O', '', cleaned)

    m = re.match(r'(\d+):(\d+)', cleaned)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


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