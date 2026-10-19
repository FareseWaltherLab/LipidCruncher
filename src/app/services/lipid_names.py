"""
Lipid name standardization.

Turns the lipid names of every supported format into LIPID MAPS shorthand
(``PC 16:0_18:1``) and infers each name's class (``PC``). Shared by the
format parsers in ``services/standardization/`` and by the standards
service.

Pure logic — no Streamlit dependencies.
"""

import re
from typing import Tuple

import pandas as pd

from app.constants import (
    LYSO_CLASSES, normalize_hydroxyl,
    sort_chains_lipid_maps, remove_phantom_chains,
    format_lipid_name,
)


def standardize_lipid_name(lipid_name: object) -> str:
    """Standardize a lipid name to LIPID MAPS shorthand notation.

    Output format: 'Class chain1_chain2' (space-separated, no parentheses).
    Sphingolipids use '/' as chain separator.
    Hydroxyl notation: ;O2 (element before count).

    Examples:
        LPC O-17:4 -> LPC O-17:4
        Cer d18:0/C24:0 -> Cer d18:0/C24:0
        LPC 18:1(d7) -> LPC 18:1(d7)
        Cer 18:1;2O/24:0 -> Cer 18:1;O2/24:0
        PC(16:0_18:1) -> PC 16:0_18:1
        LPC(20:0_0:0) -> LPC 20:0
    """
    try:
        if not lipid_name or pd.isna(lipid_name):
            return "Unknown"

        lipid_name = str(lipid_name).strip()
        lipid_name, modification = _extract_modification(lipid_name)

        first_paren = lipid_name.find('(')
        first_colon = lipid_name.find(':')

        if first_paren > 0 and (first_colon < 0 or first_paren < first_colon):
            class_name, chain_info = _parse_parenthesized_name(
                lipid_name, first_paren
            )
        else:
            class_name, chain_info = _parse_bare_name(lipid_name)

        if not chain_info:
            return f"{class_name}{modification}" if modification else class_name

        # Normalize hydroxyl notation: ;2O → ;O2
        chain_info = normalize_hydroxyl(chain_info)

        # Split chains (input may use / or _)
        raw_chains = re.split(r'[/_]', chain_info)

        # Remove phantom 0:0 chains for lyso-species
        if class_name in LYSO_CLASSES:
            raw_chains = remove_phantom_chains(raw_chains)

        # Sort chains per LIPID MAPS rules
        raw_chains = sort_chains_lipid_maps(raw_chains, class_name)

        return format_lipid_name(class_name, raw_chains, modification)

    except (ValueError, IndexError):
        return str(lipid_name) if lipid_name else "Unknown"


def _extract_modification(lipid_name: str) -> Tuple[str, str]:
    """Extract trailing deuteration modifications like (d7), (d9)."""
    mod_match = re.search(r'\(d\d+\)$', lipid_name)
    if mod_match:
        return lipid_name[:mod_match.start()].strip(), mod_match.group()
    return lipid_name, ""


def _parse_parenthesized_name(
    lipid_name: str, first_paren: int
) -> Tuple[str, str]:
    """Parse a lipid name that has parentheses around chain info.

    Returns (class_name, chain_info) with raw separators preserved.
    """
    class_name = lipid_name[:first_paren]
    paren_depth = 0
    chain_end = first_paren
    for i in range(first_paren, len(lipid_name)):
        if lipid_name[i] == '(':
            paren_depth += 1
        elif lipid_name[i] == ')':
            paren_depth -= 1
            if paren_depth == 0:
                chain_end = i
                break
    chain_info = lipid_name[first_paren + 1:chain_end]
    trailing = lipid_name[chain_end + 1:].strip()
    if trailing:
        chain_info = f"{chain_info}{trailing}"
    return class_name, chain_info


def _parse_bare_name(lipid_name: str) -> Tuple[str, str]:
    """Parse a lipid name without parentheses (space-separated or compact).

    Returns (class_name, chain_info).
    """
    first_space = lipid_name.find(' ')
    if first_space > 0:
        class_name = lipid_name[:first_space]
        chain_info = lipid_name[first_space + 1:]
        return class_name, chain_info

    match = re.match(r'^([A-Za-z][A-Za-z0-9]*)[- ]?(.*)', lipid_name)
    if match:
        class_name, chain_info = match.groups()
        return class_name, chain_info
    return lipid_name, ""


def infer_class_key(lipid_molec) -> str:
    """Infer ClassKey from a standardized LipidMolec name.

    Handles both new format ('PC 16:0_18:1') and legacy ('PC(16:0_18:1)').

    Examples:
        PC 16:0_18:1 -> PC
        LPC 18:1(d7) -> LPC
        SPLASH LPC 18:1(d7) -> LPC
    """
    try:
        if lipid_molec is None or pd.isna(lipid_molec):
            return "Unknown"
        name = str(lipid_molec).strip()

        # SPLASH standards: extract the actual class
        # Handles both "SPLASH LPC 18:1(d7)" and "SPLASH 16:0_Cer 18:1;O2(d9)"
        if name.upper().startswith('SPLASH'):
            # Try direct class after SPLASH (e.g., "SPLASH LPC ...")
            splash_match = re.match(
                r'^SPLASH[\s(]+([A-Za-z][A-Za-z0-9-]*)', name, re.IGNORECASE
            )
            if splash_match:
                return splash_match.group(1).strip()
            # Try class after leading chain (e.g., "SPLASH 16:0_Cer ...")
            chain_class_match = re.match(
                r'^SPLASH\s+[\d:]+_([A-Za-z][A-Za-z0-9-]*)', name, re.IGNORECASE
            )
            if chain_class_match:
                return chain_class_match.group(1).strip()

        # New format: class is everything before first space
        # Legacy format: class is everything before first '('
        match = re.match(r'^([A-Za-z][A-Za-z0-9-]*)', name)
        if match:
            return match.group(1).strip()
        return "Unknown"
    except (TypeError, AttributeError):
        return "Unknown"
