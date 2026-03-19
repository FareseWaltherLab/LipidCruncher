"""Data processing pipeline documentation for each format."""

PROCESSING_DOCS = {
    'LipidSearch 5.0': """
### Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | Extract LipidMolec, ClassKey, CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey, MeanArea columns |
| 2. Data Type Conversion | Convert MeanArea to numeric (non-numeric → 0) |
| 3. Lipid Name Standardization | Standardize to LIPID MAPS shorthand (`Class chains`) |
| 4. Grade Filtering | Filter by quality grade (**configurable below**) |
| 5. Best Peak Selection | Keep entry with highest TotalSmpIDRate(%) per lipid |
| 6. Missing FA Keys | Remove rows without FAKey (except Ch class, deuterated standards) |
| 7. Duplicate Removal | Remove duplicates by LipidMolec |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### Grade Filtering (Configurable)

LipidSearch assigns quality grades to each identification:

| Grade | Confidence | Default Action |
|-------|------------|----------------|
| A | Highest | Keep |
| B | Good | Keep |
| C | Lower | Keep for LPC/SM only |
| D | Lowest | Remove |

**Configure in "Configure Grade Filtering" section below.**
""",

    'MS-DIAL': """
### Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Header Detection | Auto-detect data start row (skip metadata rows) |
| 2. Column Mapping | `Metabolite name` → LipidMolec, `Average Rt(min)` → BaseRt, `Average Mz` → CalcMass |
| 3. ClassKey Inference | Extract class from lipid name (e.g., `Cer 18:1;O2/24:0` → `Cer`) |
| 4. Lipid Name Standardization | Standardize to LIPID MAPS shorthand, normalize hydroxyl (`;2O` → `;O2`) |
| 5. Quality Filtering | Filter by Total Score and/or MS/MS validation (**configurable below**) |
| 6. Data Type Selection | Choose raw or pre-normalized (if both available) |
| 7. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 8. Smart Deduplication | Keep entry with highest Total Score per lipid |
| 9. Internal Standards | Auto-detect: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH` patterns |
| 10. Duplicate Removal | Remove remaining duplicates by LipidMolec |
| 11. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### Quality Filtering (Configurable)

MS-DIAL provides quality metrics for filtering:

| Preset | Total Score | MS/MS Required | Use Case |
|--------|-------------|----------------|----------|
| Strict | ≥80 | Yes | Publication-ready |
| Moderate | ≥60 | No | Exploratory analysis |
| Permissive | ≥40 | No | Discovery |

**Configure in "Configure Quality Filtering" section below.**
""",

    'Metabolomics Workbench': """
### Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Section Extraction | Extract data between `MS_METABOLITE_DATA_START` and `MS_METABOLITE_DATA_END` |
| 2. Header Processing | Row 1 → sample names, Row 2 → conditions |
| 3. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 4. Lipid Name Standardization | Standardize to LIPID MAPS shorthand (`Class chains`) |
| 5. ClassKey Extraction | Extract class from lipid name |
| 6. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 7. Conditions Storage | Store conditions for experiment setup suggestions |
| 8. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
""",

    'Generic Format': """
### Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Column Standardization | First column → LipidMolec, remaining → `intensity[s1]`, `intensity[s2]`, ... |
| 2. Lipid Name Standardization | Standardize to LIPID MAPS shorthand (`Class chains`), normalize hydroxyl |
| 3. ClassKey Extraction | Extract class from lipid name (e.g., `PC 16:0_18:1` → `PC`) |
| 4. Data Type Conversion | Convert intensity to numeric (non-numeric → 0) |
| 5. Invalid Lipid Removal | Remove empty names, single special characters |
| 6. Duplicate Removal | Remove duplicates by LipidMolec |
| 7. Zero Filtering | Remove species failing zero threshold (**configurable below**) |
"""
}

ZERO_FILTERING_DOCS = """
#### Zero Filtering (Configurable)

Removes lipid species with too many zero/below-detection values:

| Condition Type | Default Threshold | Action |
|----------------|-------------------|--------|
| BQC (if present) | ≥50% zeros | Remove species |
| All non-BQC conditions | ≥75% zeros each | Remove species |

*Thresholds are adjustable in "Configure Zero Filtering" section below.*
"""


def get_processing_docs(data_format: str) -> str:
    """Get processing documentation for a specific format."""
    return PROCESSING_DOCS.get(data_format, PROCESSING_DOCS['Generic Format'])
