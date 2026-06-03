"""
Structural validation of the three IsoCorrectoR input files, in pure Python.

Runs *before* R is invoked so users get actionable errors (missing columns,
molecule-name mismatch, non-numeric intensities) instead of an R stack trace.
This does NOT re-implement IsoCorrectoR's chemistry — it only checks the
structural contract the package relies on. No Streamlit imports.

Input-file contract (IsoCorrectoR uses the first column positionally):
  - Measurement: col 0 = isotopologue IDs ("<Molecule>_<labeling>"),
                 remaining columns = per-sample numeric intensities.
  - Molecule:    col 0 = molecule names (must be a prefix of the measurement IDs).
  - Element:     col 0 = element symbols.
"""
from typing import List, Optional, Tuple

import pandas as pd


def _nonempty_first_column(df: pd.DataFrame) -> List[str]:
    """Return stripped, non-empty values of the first column."""
    if df.shape[1] == 0:
        return []
    values = df.iloc[:, 0].dropna().astype(str).str.strip()
    return [v for v in values if v]


def _matching_molecule(measurement_id: str, molecule_names: List[str]) -> Optional[str]:
    """Return the molecule name that prefixes this ID (longest match wins).

    Longest-match so that, given both "PC" and "PC_34:1", the ID "PC_34:1_C0"
    resolves to "PC_34:1" rather than "PC".
    """
    matches = [
        name for name in molecule_names
        if measurement_id == name or measurement_id.startswith(name + "_")
    ]
    return max(matches, key=len) if matches else None


def _resolution_style(
    measurement_ids: List[str], molecule_names: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Infer the labeling style from the isotopologue-ID suffixes.

    IsoCorrectoR encodes resolution mode in the ID suffix after the molecule
    name: ultra-high-res uses element-specific labels ("_C0", "_N1"), standard
    resolution uses a plain count ("_0", "_1"). The suffix's first character
    therefore determines the required mode.

    Returns:
        (style, example) where style is "uhr", "standard", or None when the
        IDs are ambiguous/mixed (let IsoCorrectoR decide), and example is a
        representative ID for the message.
    """
    uhr_example = standard_example = None
    saw_uhr = saw_standard = False
    for measurement_id in measurement_ids:
        name = _matching_molecule(measurement_id, molecule_names)
        if name is None:
            continue
        suffix = measurement_id[len(name):].lstrip("_")
        if not suffix:
            continue
        if suffix[0].isalpha():
            saw_uhr = True
            uhr_example = uhr_example or measurement_id
        elif suffix[0].isdigit():
            saw_standard = True
            standard_example = standard_example or measurement_id

    if saw_uhr and not saw_standard:
        return "uhr", uhr_example
    if saw_standard and not saw_uhr:
        return "standard", standard_example
    return None, None  # mixed or undetermined → don't second-guess IsoCorrectoR


def validate_inputs(
    measurement_df: pd.DataFrame,
    molecule_df: pd.DataFrame,
    element_df: pd.DataFrame,
    ultra_high_res: Optional[bool] = None,
) -> List[str]:
    """Validate the three input DataFrames structurally.

    Args:
        measurement_df: Measurement file (IDs in col 0, samples in the rest).
        molecule_df: Molecule file (molecule names in col 0).
        element_df: Element file (element symbols in col 0).
        ultra_high_res: The UHR setting that will be passed to IsoCorrectoR.
            When provided, the isotopologue-ID labeling style is checked against
            it so a mode/ID mismatch is reported up front (instead of a cryptic
            R abort). Pass None to skip this check.

    Returns:
        A list of human-readable error messages. Empty list == valid.
    """
    errors: List[str] = []

    # --- Emptiness ---------------------------------------------------------
    if measurement_df is None or measurement_df.empty:
        errors.append("Measurement file is empty.")
    if molecule_df is None or molecule_df.empty:
        errors.append("Molecule file is empty.")
    if element_df is None or element_df.empty:
        errors.append("Element file is empty.")
    if errors:
        return errors

    # --- Shapes ------------------------------------------------------------
    if measurement_df.shape[1] < 2:
        errors.append(
            "Measurement file must have an ID column plus at least one sample "
            f"column; found {measurement_df.shape[1]} column(s)."
        )
    if element_df.shape[1] < 1:
        errors.append("Element file must have at least an element column.")

    measurement_ids = _nonempty_first_column(measurement_df)
    molecule_names = _nonempty_first_column(molecule_df)
    element_symbols = _nonempty_first_column(element_df)

    if not measurement_ids:
        errors.append("Measurement file has no isotopologue IDs in its first column.")
    if not molecule_names:
        errors.append("Molecule file has no molecule names in its first column.")
    if not element_symbols:
        errors.append("Element file has no element symbols in its first column.")

    # Can't run the cross-file / numeric checks without IDs and molecules.
    if errors:
        return errors

    # --- Numeric intensities ----------------------------------------------
    sample_columns = list(measurement_df.columns[1:])
    for col in sample_columns:
        coerced = pd.to_numeric(measurement_df[col], errors="coerce")
        bad_mask = coerced.isna() & measurement_df[col].notna()
        if bad_mask.any():
            errors.append(
                f"Sample column '{col}' in the Measurement file contains "
                "non-numeric intensity values."
            )

    # --- Molecule-name consistency ----------------------------------------
    # Every measurement ID must be claimed by a molecule name as a prefix.
    # Prefix-matching (not splitting on '_') because molecule names themselves
    # contain underscores, e.g. "BMP_18:1_18:1".
    def claimed_by_molecule(measurement_id: str) -> bool:
        return any(
            measurement_id == name or measurement_id.startswith(name + "_")
            for name in molecule_names
        )

    unmatched = [mid for mid in measurement_ids if not claimed_by_molecule(mid)]
    if unmatched:
        preview = ", ".join(unmatched[:5])
        more = f" (and {len(unmatched) - 5} more)" if len(unmatched) > 5 else ""
        errors.append(
            f"{len(unmatched)} measurement row(s) do not match any molecule in "
            f"the Molecule file: {preview}{more}. Molecule names must match the "
            "prefix of each isotopologue ID."
        )

    # --- Resolution-mode consistency --------------------------------------
    # IsoCorrectoR aborts cryptically if the UHR setting doesn't match the
    # isotopologue-ID labeling style. Catch it here with an actionable message.
    if ultra_high_res is not None and not unmatched:
        style, example = _resolution_style(measurement_ids, molecule_names)
        if style == "uhr" and not ultra_high_res:
            errors.append(
                f"Measurement IDs use ultra-high-resolution labeling (e.g. '{example}'). "
                "Enable 'Ultra-high-resolution mode' in Correction settings, or relabel "
                "isotopologues with a plain count ('_0', '_1', …) for standard resolution."
            )
        elif style == "standard" and ultra_high_res:
            errors.append(
                f"Measurement IDs use standard-resolution labeling (e.g. '{example}'). "
                "Disable 'Ultra-high-resolution mode' in Correction settings, or relabel "
                "isotopologues with element-specific labels ('_C0', '_C1', …) for UHR mode."
            )

    return errors
