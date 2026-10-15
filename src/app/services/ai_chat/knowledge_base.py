"""
LipidCruncher knowledge base for AI system prompts.

Provides a condensed reference about the application's capabilities,
supported formats, terminology, and analysis features. Included in
AI system prompts so the assistant can answer user questions accurately.
"""


def get_app_knowledge() -> str:
    """Return a condensed knowledge base about LipidCruncher.

    This is appended to AI system prompts so the assistant understands
    the application context without needing access to the full codebase.
    """
    return r"""\
## About LipidCruncher

LipidCruncher is an open-source lipidomics data analysis platform built by the Farese & Walther \
Lab at Memorial Sloan Kettering Cancer Center. It helps researchers process, visualize, and \
analyze lipidomic data without bioinformatics expertise.

## Supported Input Formats

1. **LipidSearch 5.0 and 5.2** — exports from Thermo LipidSearch software (one format option \
for both). Key columns: LipidMolec, ClassKey, BaseRt, CalcMass, Grade, MeanArea[s1], \
MeanArea[s2], etc. Tab- or comma-delimited files are both read. A 5.2 export grouped by \
condition with separate positive and negative runs (OriginalArea[s{condition}-{file}] \
columns) also needs the **Alignment Setting file** from LipidSearch, which pairs each sample's \
two runs before they are merged.
2. **MS-DIAL** — exports from MS-DIAL software. Key columns: Metabolite name, Ontology, \
Total score, plus sample columns.
3. **Generic** — the simplest format. Column 1 = lipid names, columns 2+ = sample intensities. \
No extra columns. Column names can be anything.
4. **Metabolomics Workbench** — tab-delimited files with START/END markers from the \
Metabolomics Workbench repository.

## What LipidCruncher Does Automatically

After uploading data in any supported format, LipidCruncher automatically:
- **Standardizes lipid names** to LIPID MAPS shorthand notation (e.g., PC(34:1) → PC 34:1, \
Cer(d18:1_24:0) → Cer 18:1;O2/24:0)
- **Extracts ClassKey** (lipid class) from lipid names
- **Renames sample columns** to intensity[s1], intensity[s2], etc.
- **Detects internal standards** via regex patterns (see below)
- **Handles format-specific cleaning** (grade filtering for LipidSearch, score filtering for MS-DIAL)

## Internal Standard Detection — exact rules

LipidCruncher flags a row as an internal standard when **either**:

1. The lipid name (`LipidMolec`) matches *any* of these regex patterns:
   - `\(d\d+\)`        — explicit deuterium count in parens, e.g. `(d5)`, `(d7)`, `(d9)`. \
NOT to be confused with the sphingoid backbone notation `d18:1` in lipids like `Cer(d18:1_24:0)` — \
that is `d` followed by carbon count + colon, NOT `d` followed by digits + closing paren.
   - `[+-]D\d+`        — `+D7`, `-D7`, etc.
   - `-d\d+[_\)\s]`    — `-d7_`, `-d9)`, etc.
   - `\[d\d+\]`        — `[d7]` etc.
   - `^Ch-D\d+`        — deuterated cholesterol standards
   - `ISTD`            — explicit ISTD marker
   - `SPLASH`          — SPLASH lipidomix standards
   - `:\(s\)`          — `:(s)` notation (LipidSearch standards suffix)
   - `\(IS\)`          — `(IS)` marker
   - `_IS$`            — `_IS` suffix
2. OR the `ClassKey` matches `ISTD|Internal`.

When asked about internal standards, generate sandbox code that applies these *exact* patterns to \
`LipidMolec` (and `ClassKey` if present) — do not invent looser heuristics. Examples like \
`Cer(d18:1_24:0)`, `BisMePA(16:1_18:1)`, `SM(d34:1)`, `LPC(17:0)` are NOT internal standards: \
the sphingoid `d18:1` / `d34:1` notation is a backbone descriptor, and odd-chain fatty acids are \
not standards either. Examples that ARE standards: `LPC(18:1)+D7:(s)`, `PC(33:1)+D7:(s)`, \
`SM(d36:2)+D9:(s)`, `Cer(d18:1_15:0)(d7)`, anything with `ISTD`, `SPLASH`, or `_IS` in the name.

## Analysis Pipeline (3 Modules)

### Module 1: Data Cleaning, Filtering, & Normalization
- Format detection and column standardization
- Format-specific data cleaning (deduplication, grade/score filtering)
- Zero filtering (removes lipids with too many missing values)
- Internal standards management
- Normalization (5 options): none, internal standard, protein, both, total intensity

### Module 2: Quality Check
- **Box plots** — concentration distributions per sample
- **BQC assessment** — coefficient of variation across batch quality control samples
- **Retention time plots** — RT consistency (LipidSearch/MS-DIAL only)
- **Pairwise correlation** — Pearson correlation heatmaps between replicates
- **PCA** — principal component analysis colored by condition, with confidence ellipses

### Module 3: Visualization & Analysis
- **Bar charts** — mean ± std per condition per lipid class, with optional statistical testing
- **Pie charts** — lipid class composition per condition
- **Saturation profiles** — SFA/MUFA/PUFA distribution
- **Chain length analysis** — carbon chain length and double bond bubble charts
- **FACH** — fatty acid composition heatmap
- **Pathway visualization** — interactive metabolic pathway network (28 lipid classes)
- **Volcano plots** — fold-change vs. p-value
- **Lipidomic heatmaps** — heatmap with hierarchical clustering (Ward linkage, Euclidean distance)

### Stable Isotope Tracing (separate page)
Opened from the home page, independent of Modules 1–3. Runs natural-isotope-abundance \
correction with the IsoCorrectoR R package on three uploaded files — Measurement \
(isotopologue intensities per sample), Molecule (formulas) and Element (isotope abundances, \
tracer purity) — and returns corrected intensities, corrected fractions, mean enrichment and \
residuals. Needs R with IsoCorrectoR on the server; a sample dataset is built in.

## Statistical Testing

- **Parametric**: Welch's t-test (2 conditions), Alexander-Govern ANOVA (3+ conditions)
- **Non-parametric**: Mann-Whitney U (2 conditions), Kruskal-Wallis (3+ conditions)
- **Auto mode**: selects parametric with log10 transform
- **Multiple testing correction**: FDR (Benjamini-Hochberg) or Bonferroni
- **Post-hoc**: Tukey's HSD or Bonferroni

## Lipid Nomenclature (LIPID MAPS)

All lipid names are standardized to LIPID MAPS shorthand notation:
- Class and chains separated by space: PC 34:1 (not PC(34:1))
- General chains separated by underscore: PC 16:0_18:1
- Sphingolipid chains separated by slash: Cer 18:1;O2/24:0
- Hydroxyl notation: ;O2 (not ;2O)
- Lyso species: phantom chains removed (LPC 20:0, not LPC 20:0_0:0)
- Chains sorted by carbon count, then double bonds

## Common Lipid Classes

FA (fatty acid), CE (cholesteryl ester), MG (monoacylglycerol), DG (diacylglycerol), \
TG (triacylglycerol), PC (phosphatidylcholine), PE (phosphatidylethanolamine), \
PS (phosphatidylserine), PI (phosphatidylinositol), PG (phosphatidylglycerol), \
PA (phosphatidic acid), SM (sphingomyelin), Cer (ceramide), HexCer (hexosylceramide), \
LPC (lysophosphatidylcholine), LPE (lysophosphatidylethanolamine), CL (cardiolipin)

## Key Terminology

- **BQC**: Batch Quality Control samples — used to assess measurement precision
- **CoV**: Coefficient of Variation — measure of data variability
- **Internal standards**: Known-concentration lipids spiked into samples for normalization
- **Zero filtering**: Removing lipids with excessive missing/zero values across samples
- **Intensity**: Raw signal/area values from mass spectrometry
"""
