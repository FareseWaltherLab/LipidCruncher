"""Normalization method documentation."""

NORMALIZATION_METHODS_DOCS = """
### Normalization Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **None** | Raw values | Data already normalized externally |
| **Internal Standards** | `(Intensity_lipid / Intensity_standard) × Conc_standard` | Correct for extraction/ionization variability |
| **Protein-based** | `Intensity_lipid / Protein_conc` | Normalize to starting material (e.g., BCA assay) |
| **Both** | `(Intensity_lipid / Intensity_standard) × (Conc_standard / Protein_conc)` | Combined correction |

After normalization, `intensity[...]` columns become `concentration[...]` columns.
"""
