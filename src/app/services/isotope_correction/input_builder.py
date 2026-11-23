"""
Build IsoCorrectoR's Molecule and Element files from the Measurement file.

The Molecule file needs each lipid's full elemental formula *as the measured
ion* plus how many atoms can carry the label (``C50H74O10P1LabC6``); the
Element file needs natural isotope abundances (constants) plus the tracer's
mass shift and purity. Users used to write both by hand, so this works them
out: lipid names come from the Measurement file's isotopologue IDs, formulas
from the lipid class and chains, and the label count is suggested from the
heaviest isotopologue measured. No Streamlit imports.
"""
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..plotting.isotopologue_distribution import IsotopologuePlotterService

# Natural abundances as IsoCorrectoR's Element file writes them:
# ``<abundance>_<mass shift>`` per isotope, joined by ``/``. C, H, N, O and P
# are the values of IsoCorrectoR's own example files; S and Na are IUPAC.
NATURAL_ABUNDANCE = {
    'C': '0.0107_1/0.9893_0',
    'H': '0.000115_1/0.999885_0',
    'N': '0.99632_0/0.00368_1',
    'O': '0.99757_0/0.00038_1/0.00205_2',
    'P': '1_0',
    'S': '0.9499_0/0.0075_1/0.0425_2/0.0001_4',
    'Na': '1_0',
}

# Tracer element -> its heavy isotope. Each is one mass unit heavier.
TRACERS = {'C': '¹³C', 'H': '²H', 'N': '¹⁵N'}
_TRACER_MASS_SHIFT = 1

# Ion form -> the atoms it adds to (or takes from) the neutral lipid.
ION_FORMS = {
    '[M-H]-': {'H': -1},
    '[M+H]+': {'H': 1},
    '[M+NH4]+': {'N': 1, 'H': 4},
    '[M+Na]+': {'Na': 1},
    '[M+HCOO]-': {'C': 1, 'H': 1, 'O': 2},
    '[M+CH3COO]-': {'C': 2, 'H': 3, 'O': 2},
}
DEFAULT_ION = '[M-H]-'

# Lipid class -> (chains, C, H, N, O, P) such that a lipid with n chain
# carbons and d double bonds in total is C(n+C) H(2n-2d+H) N O P. Each row is
# the backbone plus `chains` fatty acids minus one water per ester bond, e.g.
# PC = glycerophosphocholine C8H20NO6P + 2 FA - 2 H2O, so PC 34:1 = C42H82NO8P.
# Sphingolipids count the sphingoid base as a chain (Cer d18:1_16:0 = 34:1).
_CLASSES = {
    'PC': (2, 8, 16, 1, 8, 1),
    'PE': (2, 5, 10, 1, 8, 1),
    'PS': (2, 6, 10, 1, 10, 1),
    'PG': (2, 6, 11, 0, 10, 1),
    'BMP': (2, 6, 11, 0, 10, 1),
    'PI': (2, 9, 15, 0, 13, 1),
    'PA': (2, 3, 5, 0, 8, 1),
    'LPC': (1, 8, 18, 1, 7, 1),
    'LPE': (1, 5, 12, 1, 7, 1),
    'LPS': (1, 6, 12, 1, 9, 1),
    'LPG': (1, 6, 13, 0, 9, 1),
    'LPI': (1, 9, 17, 0, 12, 1),
    'LPA': (1, 3, 7, 0, 7, 1),
    'CL': (4, 9, 14, 0, 17, 2),
    'TG': (3, 3, 2, 0, 6, 0),
    'DG': (2, 3, 4, 0, 5, 0),
    'MG': (1, 3, 6, 0, 4, 0),
    'CE': (1, 27, 44, 0, 2, 0),
    'FA': (1, 0, 0, 0, 2, 0),
    'Cer': (2, 0, 1, 1, 3, 0),
    'SM': (2, 5, 13, 2, 6, 1),
    'HexCer': (2, 6, 11, 1, 8, 0),
    'Hex2Cer': (2, 12, 21, 1, 13, 0),
}
# LipidSearch's spellings.
_CLASSES['ChE'] = _CLASSES['CE']
_CLASSES['Hex1Cer'] = _CLASSES['HexCer']
_SPHINGOLIPIDS = {'Cer', 'SM', 'HexCer', 'Hex1Cer', 'Hex2Cer'}

_NAME = re.compile(r'^([A-Za-z][A-Za-z0-9]*)[_( ](.+?)\)?$')
# An optional ether (O-) or plasmalogen (P-) prefix, an optional sphingoid
# hydroxyl prefix (d = 2, t = 3, m = 1 OH), then carbons:double bonds.
_CHAIN = re.compile(r'^(O-|P-)?([dtm])?(\d+):(\d+)$')
_FORMULA = re.compile(r'([A-Z][a-z]?)(\d*)')

TABLE_COLUMNS = ['Molecule', 'Formula', 'Ion', 'Labelled atoms']


def _hill(atoms: Dict[str, int], ones: bool) -> str:
    """Formula text in Hill order (C, H, then alphabetical)."""
    order = [e for e in ('C', 'H') if e in atoms]
    order += sorted(e for e in atoms if e not in ('C', 'H'))
    return ''.join(
        f'{e}{atoms[e] if ones or atoms[e] != 1 else ""}'
        for e in order if atoms[e]
    )


def parse_formula(formula: str) -> Optional[Dict[str, int]]:
    """``C50H75O10P`` -> {'C': 50, 'H': 75, 'O': 10, 'P': 1}; None if the
    text isn't a formula."""
    text = str(formula).replace(' ', '')
    if not text or ''.join(m.group(0) for m in _FORMULA.finditer(text)) != text:
        return None
    atoms: Dict[str, int] = {}
    for element, count in _FORMULA.findall(text):
        atoms[element] = atoms.get(element, 0) + int(count or 1)
    return atoms


def lipid_formula(name: str) -> Optional[str]:
    """The neutral formula of a lipid named like ``BMP_22:6_22:6``,
    ``PC(34:1)`` or ``Cer_d18:1_16:0``; None if the class or the name's
    form isn't known, so the user types it in."""
    match = _NAME.match(str(name).strip())
    if not match or match.group(1) not in _CLASSES:
        return None
    lipid_class = match.group(1)
    chains, c0, h0, n0, o0, p0 = _CLASSES[lipid_class]
    tokens = re.split(r'[_/]', match.group(2))
    if len(tokens) not in (1, chains):  # a sum composition, or every chain
        return None
    carbons = bonds = o_shift = h_shift = 0
    for token in tokens:
        chain = _CHAIN.match(token)
        if not chain:
            return None
        ether, hydroxyls, c, d = chain.groups()
        if hydroxyls and lipid_class not in _SPHINGOLIPIDS:
            return None
        carbons += int(c)
        bonds += int(d)
        if ether:  # the ester's C=O becomes CH2 (O-) or CH= (P-)
            o_shift -= 1
            h_shift += 2 if ether == 'O-' else 0
        o_shift += {'t': 1, 'm': -1}.get(hydroxyls, 0)
    atoms = {
        'C': carbons + c0, 'H': 2 * carbons - 2 * bonds + h0 + h_shift,
        'N': n0, 'O': o0 + o_shift, 'P': p0,
    }
    return _hill(atoms, ones=False)


def default_tracer(measurement_df: pd.DataFrame) -> str:
    """The tracer element the IDs name (``_C3`` in ultra-high resolution),
    else carbon."""
    for isotopologue_id in measurement_df.iloc[:, 0].dropna():
        element = re.sub(r'\d', '', IsotopologuePlotterService.split_id(isotopologue_id)[1])
        if element in TRACERS:
            return element
    return 'C'


def molecule_table(measurement_df: pd.DataFrame) -> pd.DataFrame:
    """One row per lipid in the Measurement file, for the user to check:
    its neutral formula (blank when the class isn't known), the ion form,
    and the labelled atoms suggested from its heaviest isotopologue
    (``_6`` -> 6; blank when no ID carries a number)."""
    heaviest: Dict[str, Optional[int]] = {}
    for isotopologue_id in measurement_df.iloc[:, 0].dropna().astype(str):
        molecule, label = IsotopologuePlotterService.split_id(isotopologue_id.strip())
        digits = re.sub(r'\D', '', label)
        shift = int(digits) if digits else None
        current = heaviest.get(molecule)
        heaviest[molecule] = shift if current is None else max(current, shift or 0)
    return pd.DataFrame(
        [(m, lipid_formula(m) or '', DEFAULT_ION, n) for m, n in heaviest.items()],
        columns=TABLE_COLUMNS,
    ).astype({'Labelled atoms': 'Int64'})


def build_input_files(
    table: pd.DataFrame, tracer: str, purity: float,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    """The Molecule and Element files from a checked ``molecule_table``.

    Args:
        table: ``molecule_table``'s columns, as the user edited them.
        tracer: The labelled element, a key of ``TRACERS``.
        purity: The tracer's isotopic purity (0-1).

    Returns:
        (molecule file, element file, problems). The files are None while
        any row has a problem.
    """
    problems: List[str] = []
    rows = []
    elements = {tracer}
    for _, row in table.iterrows():
        molecule = str(row['Molecule'])
        formula = row['Formula'] if pd.notna(row['Formula']) else ''
        atoms = parse_formula(formula)
        if not str(formula).strip():
            problems.append(f"{molecule}: enter its formula (its lipid class isn't one LipidCruncher knows).")
            continue
        if atoms is None:
            problems.append(f"{molecule}: '{formula}' is not a formula (write it like C42H82NO8P).")
            continue
        unknown = sorted(set(atoms) - set(NATURAL_ABUNDANCE))
        if unknown:
            problems.append(f"{molecule}: no natural abundances for {', '.join(unknown)}; upload your own Element file.")
            continue
        ion = row['Ion']
        if ion not in ION_FORMS:
            problems.append(f'{molecule}: choose the ion form.')
            continue
        for element, change in ION_FORMS[ion].items():
            atoms[element] = atoms.get(element, 0) + change
        labelled = row['Labelled atoms']
        if pd.isna(labelled) or int(labelled) < 1:
            problems.append(f'{molecule}: enter how many {tracer} atoms can carry the label.')
            continue
        if int(labelled) > atoms.get(tracer, 0):
            problems.append(
                f'{molecule}: {int(labelled)} labelled {tracer} atoms, but the '
                f'{ion} ion has only {atoms.get(tracer, 0)}.'
            )
            continue
        elements |= {e for e, n in atoms.items() if n}
        rows.append((molecule, f'{_hill(atoms, ones=True)}Lab{tracer}{int(labelled)}', ''))
    if problems:
        return None, None, problems

    molecule_file = pd.DataFrame(
        rows, columns=['Molecule', 'MS ion or MS/MS product ion', 'MS/MS neutral loss'],
    )
    element_file = pd.DataFrame(
        [
            (e, NATURAL_ABUNDANCE[e],
             _TRACER_MASS_SHIFT if e == tracer else None,
             purity if e == tracer else None)
            for e in sorted(elements, key=lambda e: (e != 'C', e != 'H', e))
        ],
        columns=['Element', 'Isotope abundance_Mass shift',
                 'Tracer isotope mass shift', 'Tracer purity'],
    ).astype({'Tracer isotope mass shift': 'Int64'})
    return molecule_file, element_file, []
