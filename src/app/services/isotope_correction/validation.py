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
from typing import List

import pandas as pd


def _nonempty_first_column(df: pd.DataFrame) -> List[str]:
    """Return stripped, non-empty values of the first column."""
    if df.shape[1] == 0:
        return []
    values = df.iloc[:, 0].dropna().astype(str).str.strip()
    return [v for v in values if v]


def validate_inputs(
    measurement_df: pd.DataFrame,
    molecule_df: pd.DataFrame,
    element_df: pd.DataFrame,
) -> List[str]:
    """Validate the three input DataFrames structurally.

    Args:
        measurement_df: Measurement file (IDs in col 0, samples in the rest).
        molecule_df: Molecule file (molecule names in col 0).
        element_df: Element file (element symbols in col 0).

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

    return errors
