"""
Format Requirements UI Components for LipidCruncher.

This module contains the format-specific documentation displayed
to users when they select a data format.
"""

import streamlit as st


# =============================================================================
# Format Requirement Strings
# =============================================================================

METABOLOMICS_WORKBENCH_REQUIREMENTS = """
### 🔬 Metabolomics Workbench Format

**Required Structure:**

| Component | Description |
|-----------|-------------|
| `MS_METABOLITE_DATA_START` | Section start marker |
| Row 1 | Sample names |
| Row 2 | Condition labels (one per sample) |
| Row 3+ | Lipid data (name in first column) |
| `MS_METABOLITE_DATA_END` | Section end marker |

**Example:**
```
MS_METABOLITE_DATA_START
Samples,Sample1,Sample2,Sample3,Sample4
Factors,WT,WT,KO,KO
LPC 16:0,234.5,256.7,189.3,201.4
PE 18:0_20:4,456.7,478.2,390.1,405.6
MS_METABOLITE_DATA_END
```

**✨ Auto-processing:** Lipid names standardized, intensity columns created, conditions extracted.
"""

LIPIDSEARCH_REQUIREMENTS = """
### 🔬 LipidSearch 5.0 Format

**Required Columns:**

| Column | Description |
|--------|-------------|
| `LipidMolec` | Lipid molecule identifier |
| `ClassKey` | Lipid class (e.g., PC, PE, TG) |
| `CalcMass` | Calculated mass |
| `BaseRt` | Retention time |
| `TotalGrade` | Quality grade (A/B/C/D) |
| `TotalSmpIDRate(%)` | Sample identification rate |
| `FAKey` | Fatty acid key |
| `MeanArea[s1]`, `MeanArea[s2]`, ... | Intensity per sample |

**Example column structure:**
```
LipidMolec | ClassKey | CalcMass | BaseRt | TotalGrade | ... | MeanArea[s1] | MeanArea[s2] | ...
```

**💡 Tip:** Export directly from LipidSearch — column names should match automatically.
"""

MSDIAL_REQUIREMENTS = """
### 🔬 MS-DIAL Format

**How to Export:** File → Export → Alignment Result → CSV

**Required Columns:**

| Column | Description |
|--------|-------------|
| `Metabolite name` | Lipid identifiers (exact name required) |
| Sample columns | Intensity values — **must be LAST columns** and **uniquely named** |

**Optional Columns** (enable extra features):

| Column | Feature Enabled |
|--------|-----------------|
| `Total score` | Quality filtering (0-100) |
| `MS/MS matched` | MS/MS validation filter |
| `Average Rt(min)` | Retention time plots |
| `Average Mz` | Retention time plots |

---

**📁 File Structure Options:**

**Option A — Raw data only:**
```
[metadata cols...] [sample1] [sample2] ... [sampleN]
```

**Option B — Raw + Pre-normalized:**
```
[metadata cols...] [raw1]...[rawN] [Lipid IS] [norm1]...[normN]
```
The `Lipid IS` column separates raw from normalized data. You'll choose which to use after upload.

---

**⚠️ Important:**
- All sample column names must be **unique** — duplicate names will cause parsing errors

---

**✨ Auto-features:**
- ClassKey inferred from lipid names
- Hydroxyl notation normalized to LIPID MAPS format (`;O2`, `;O3`)
- Internal standards detected: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH`
- Column mappings reviewable after upload
"""

GENERIC_REQUIREMENTS = """
### 🔬 Generic Format

**Simple structure — just two things:**

| Position | Content |
|----------|---------|
| Column 1 | Lipid names |
| Columns 2+ | Sample intensities (one column per sample) |

**⚠️ Important:** No extra columns allowed! Remove any metadata columns before upload.

---

**Lipid Name Standardization Examples:**

| Your Format | → Standardized |
|-------------|----------------|
| `LPC O-18:1` | `LPC O-18:1` |
| `LPC 18:1(d7)` | `LPC 18:1(d7)` |
| `Cer d18:1;2O/24:0` | `Cer d18:1;O2/24:0` |
| `PC 16:0/18:1` | `PC 16:0_18:1` |
| `LPC(20:0_0:0)` | `LPC 20:0` |

---

**✨ Auto-features:**
- Lipid names standardized to LIPID MAPS shorthand: `Class chains` (space-separated, no parentheses)
- Sphingolipid chains separated by `/`, others by `_`
- Hydroxyl notation normalized (`;2O` → `;O2`)
- ClassKey extracted from lipid names
- Intensity columns renamed to `intensity[s1]`, `intensity[s2]`, ...
"""


# =============================================================================
# Display Functions
# =============================================================================

def display_format_requirements(data_format: str):
    """
    Display format-specific requirements in a collapsible section.

    Args:
        data_format: The selected data format string
    """
    requirements_map = {
        'Metabolomics Workbench': METABOLOMICS_WORKBENCH_REQUIREMENTS,
        'LipidSearch 5.0': LIPIDSEARCH_REQUIREMENTS,
        'MS-DIAL': MSDIAL_REQUIREMENTS,
        'Generic Format': GENERIC_REQUIREMENTS,
    }

    requirements_text = requirements_map.get(data_format, GENERIC_REQUIREMENTS)

    with st.expander("📋 Data Format Requirements", expanded=False):
        st.markdown(requirements_text)
