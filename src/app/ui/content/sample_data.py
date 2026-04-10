"""Sample dataset information for each supported format."""

from app.constants import (
    FORMAT_GENERIC, FORMAT_LIPIDSEARCH, FORMAT_MSDIAL, FORMAT_METABOLOMICS_WORKBENCH,
)

SAMPLE_DATA_INFO = {
    FORMAT_GENERIC: {
        'file': 'generic_test_dataset.csv',
        'description': """ADGAT-DKO case study (normalized): inguinal white adipose tissue, WT vs ADGAT-DKO.

**Sample order:**
1. WT (s1–s4, n=4)
2. ADGAT-DKO (s5–s8, n=4)
3. BQC (s9–s12, n=4)""",
        'experiment': {
            'n_conditions': 3,
            'conditions': ['WT', 'ADGAT-DKO', 'BQC'],
            'samples_per_condition': [4, 4, 4],
            'bqc_label': 'BQC',
        },
    },
    FORMAT_LIPIDSEARCH: {
        'file': 'lipidsearch5_test_dataset.csv',
        'description': """ADGAT-DKO case study (raw): inguinal white adipose tissue, WT vs ADGAT-DKO. Includes quality grades and retention times.

**Sample order:**
1. WT (s1–s4, n=4)
2. ADGAT-DKO (s5–s8, n=4)
3. BQC (s9–s12, n=4)""",
        'experiment': {
            'n_conditions': 3,
            'conditions': ['WT', 'ADGAT-DKO', 'BQC'],
            'samples_per_condition': [4, 4, 4],
            'bqc_label': 'BQC',
        },
    },
    FORMAT_MSDIAL: {
        'file': 'msdial_test_dataset.csv',
        'description': """Mouse adrenal gland lipidomics: fads2 knockout vs wild-type.

**Sample order:**
1. Blank (n=1)
2. fads2 KO (n=3)
3. Wild-type (n=3)""",
        'experiment': {
            'n_conditions': 3,
            'conditions': ['Blank', 'fads2 KO', 'Wild-type'],
            'samples_per_condition': [1, 3, 3],
            'bqc_label': None,
        },
    },
    FORMAT_METABOLOMICS_WORKBENCH: {
        'file': 'mw_test_dataset.csv',
        'description': """Mouse serum HFD study: 2×2 factorial (Normal/HFD × Water/DCA).

**Sample order:**
1. Normal+Water (S1A–S11A, n=11)
2. Normal+DCA (S1B–S11B, n=11)
3. HFD+Water (S1C–S11C, n=11)
4. HFD+DCA (S1D–S11D, n=11)
5. Blank (n=2)
6. TQC (n=12)""",
        'bqc_label': 'Diet:QC | BileAcid:QC',
    },
}


def get_sample_data_info(data_format: str) -> dict:
    """Get sample dataset info including file path and description."""
    return SAMPLE_DATA_INFO.get(data_format)
