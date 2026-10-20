"""Help text for internal standards and protein concentration CSV uploads."""

STANDARDS_EXTRACT_HELP = """
**CSV format:** Single column with lipid names (must exist in your dataset).

Example:
```
LipidMolec
PC 15:0_18:1+D7:(s)
PE 15:0_18:1+D7:(s)
SM 18:1+D9:(s)
```
"""

STANDARDS_COMPLETE_HELP = """
**CSV format:** 1st column = lipid names, remaining columns = intensity values, one per sample in the order shown in the sidebar (s1, s2, …).

Example:
```
LipidMolec,s1,s2,s3,s4
PC 15:0_18:1+D7:(s),1000,1200,1100,1050
PE 15:0_18:1+D7:(s),800,850,820,810
```
"""

PROTEIN_CSV_HELP = """**CSV format:** a column named `Concentration` with one value per sample. Add a `Sample` column (each sample's label, e.g. s1, or its name) to match rows to samples; without one, rows are taken in sample order (s1, s2, ...)."""
