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
### Metabolomics Workbench Format

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
LPC(16:0),234.5,256.7,189.3,201.4
PE(18:0_20:4),456.7,478.2,390.1,405.6
MS_METABOLITE_DATA_END
```
"""

LIPIDSEARCH_REQUIREMENTS = """
### LipidSearch 5.0 Format

**Required Columns:**

| Column | Description |
|--------|-------------|
| `LipidMolec` | Lipid molecule identifier |
| `ClassKey` | Lipid class (e.g., PC, PE, TG) |
| `CalcMass` | Calculated mass |
| `BaseRt` | Retention time |
| `TotalGrade` | Quality grade (A/B/C/D) |
| `FAKey` | Fatty acid key |
| `MeanArea[s1]`, `MeanArea[s2]`, ... | Intensity per sample |

**Tip:** Export directly from LipidSearch — column names should match automatically.
"""

MSDIAL_REQUIREMENTS = """
### MS-DIAL Format

**How to Export:** File → Export → Alignment Result → CSV

**Required Columns:**

| Column | Description |
|--------|-------------|
| `Metabolite name` | Lipid identifiers |
| Sample columns | Intensity values — must be LAST columns |

**Optional Columns** (enable extra features):

| Column | Feature Enabled |
|--------|-----------------|
| `Total score` | Quality filtering (0-100) |
| `MS/MS matched` | MS/MS validation filter |
| `Average Rt(min)` | Retention time plots |

**Important:** All sample column names must be unique.
"""

GENERIC_REQUIREMENTS = """
### Generic Format

**Required Columns:**

| Column | Description |
|--------|-------------|
| Column 1 | Lipid names (will become `LipidMolec`) |
| Remaining columns | Sample intensities |

**Optional:** A `ClassKey` column for lipid class assignments.

**Internal Standards:** Detected automatically by patterns: `(d5)`, `(d7)`, `(d9)`, `ISTD`, `SPLASH`
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

    with st.expander("Data Format Requirements", expanded=False):
        st.markdown(requirements_text)
