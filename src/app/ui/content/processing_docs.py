"""Data processing pipeline documentation for each format."""

from app.constants import (
    FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, FORMAT_METABOLOMICS_WORKBENCH, FORMAT_GENERIC,
)

PROCESSING_DOCS = {
    FORMAT_LIPIDSEARCH: """
Covers both **LipidSearch 5.0** and **5.2** exports. The delimiter (tab or
comma) is detected automatically — 5.2 exports are tab-delimited despite the
`.csv` extension.

### Reading Intensities

Two export layouts are recognized, and they reach the same place:

| Layout | Intensity columns | What happens |
|--------|-------------------|--------------|
| Flat (5.0, and sample-grouped 5.2) | `MeanArea[s1]`, `MeanArea[s2]`, … | Renamed directly to `intensity[s1]`, `intensity[s2]`, … |
| Condition-grouped **dual-polarity** (5.2) | `OriginalArea[s{condition}-{file}]` | Requires the **Alignment Setting file** (see below) |

#### Dual-polarity exports and the Alignment Setting file

In the condition-grouped 5.2 layout each biological sample was run twice — once
in positive and once in negative mode — and appears as two separate per-file
columns. Nothing in the data file itself says which two files belong to the
same sample, so the **Alignment Setting file is required** and is requested at
upload.

It is used to pair each sample's positive and negative runs; the paired columns
are then summed into a single `intensity[s1..sN]` per sample. Because any given
lipid is detected in only one polarity, that sum is equivalent to taking
whichever polarity saw it. Samples are renumbered flat in alignment order, and
the conditions and sample counts read from the alignment pre-populate the
experiment setup in the sidebar.

---

### Data Cleaning Pipeline

| Step | Action |
|------|--------|
| 1. Missing FA Keys | Remove rows without FAKey (except Ch class and `Ch-D*` deuterated standards) |
| 2. Data Type Conversion | Convert intensity columns to numeric (non-numeric → 0) |
| 3. Grade Filtering | Filter by quality grade (**configurable below**) |
| 4. Lipid Name Standardization | Standardize to LIPID MAPS shorthand (`Class chains`) |
| 5. Best Peak Selection | One entry per lipid: best grade first, then highest TotalSmpIDRate(%) |
| 6. Column Projection | Keep LipidMolec, ClassKey, CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey and the intensity columns; drop extended-export extras (`OrgMeanArea[*]`, `MeanHeight[*]`, `MeanConc[*]`, `LipidMolecGroup`, …) |
| 7. Zero Filtering | Remove species failing zero threshold (**configurable below**) |

---

#### Grade Filtering (Configurable)

LipidSearch assigns quality grades to each identification:

| Grade | Confidence | Default Action |
|-------|------------|----------------|
| A | Highest | Keep |
| B | Good | Keep |
| C | Lower | Keep for LPC/SM only |
| D | Lowest | Remove |

Grades also break ties in step 5: when a lipid has several eligible entries, the
better grade wins, and TotalSmpIDRate(%) only decides between entries of the
same grade.

**Configure in "Configure Grade Filtering" section below.**
""",

    FORMAT_MSDIAL: """
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

    FORMAT_METABOLOMICS_WORKBENCH: """
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

    FORMAT_GENERIC: """
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
    return PROCESSING_DOCS.get(data_format, PROCESSING_DOCS[FORMAT_GENERIC])
