"""
Few-shot examples library for AI chat system prompts.

Each function returns a string of structured input/output examples
for a specific tab. Examples are kept here (not inline in prompts)
for maintainability. Each tab gets ~8-12 examples (~1500 tokens).

All functions return plain strings — no Streamlit, no side effects.
"""


def get_transform_examples() -> str:
    """Few-shot examples for the Transform to Generic tab."""
    return """\
## Few-Shot Examples

### Example 1: Data with extra metadata columns
User: "My data has columns: lipid_name, lipid_class, mass, rt, sample_1, sample_2, sample_3. Transform it."
```python
def transform(df):
    # Keep lipid names (col 1) + sample intensities only.
    # Drop metadata (mass, rt, lipid_class).
    result = pd.DataFrame()
    result['lipid_name'] = df['lipid_name']
    for col in ['sample_1', 'sample_2', 'sample_3']:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result

result_df = transform(df)
```

### Example 2: Data with non-standard layout
User: "Columns: Name (like 'PC(34:1)'), Score, Area_S1, Area_S2. Transform please."
```python
def transform(df):
    result = pd.DataFrame()
    result['Name'] = df['Name']
    sample_cols = [c for c in df.columns if c.startswith('Area_')]
    for col in sample_cols:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result

result_df = transform(df)
```

### Example 3: Pivot from long format
User: "My data is in long format with columns: lipid, sample_id, value"
```python
def transform(df):
    pivoted = df.pivot_table(
        index='lipid', columns='sample_id', values='value', aggfunc='first'
    ).reset_index()
    return pivoted

result_df = transform(df)
```

### Example 4: Filter by score before transforming
User: "Remove rows where TotalScore < 20, then transform. Lipid names in 'Compound', samples are Intensity_1 through Intensity_6."
```python
def transform(df):
    # Filter by score first
    filtered = df[df['TotalScore'] >= 20].copy()
    # Build Generic format
    result = pd.DataFrame()
    result['Compound'] = filtered['Compound']
    intensity_cols = [c for c in filtered.columns if c.startswith('Intensity_')]
    for col in intensity_cols:
        result[col] = pd.to_numeric(filtered[col], errors='coerce')
    return result

result_df = transform(df)
```

### Example 5: Column 1 is not lipid names
User: "My first column is a numeric ID. Lipid names are in column 'Lipid_Species'. Samples are 'KO_1', 'KO_2', 'WT_1', 'WT_2'."
```python
def transform(df):
    result = pd.DataFrame()
    result['Lipid_Species'] = df['Lipid_Species']
    for col in ['KO_1', 'KO_2', 'WT_1', 'WT_2']:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result

result_df = transform(df)
```

### Example 6: Auto-detect sample columns by pattern
User: "Transform my data. Lipid names are in the first column. Sample columns all start with 'Sample_'."
```python
def transform(df):
    result = pd.DataFrame()
    result[df.columns[0]] = df.iloc[:, 0]
    sample_cols = [c for c in df.columns if c.startswith('Sample_')]
    for col in sample_cols:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result

result_df = transform(df)
```

### Example 7: Remove duplicates and transform
User: "Some lipid names appear multiple times. Keep the row with the highest mean intensity, then transform."
```python
def transform(df):
    lipid_col = df.columns[0]
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    # Compute mean intensity per row
    df['_mean'] = df[numeric_cols].mean(axis=1)
    # Keep row with highest mean per lipid
    df = df.sort_values('_mean', ascending=False).drop_duplicates(subset=lipid_col, keep='first')
    # Build Generic format
    result = pd.DataFrame()
    result[lipid_col] = df[lipid_col]
    for col in numeric_cols:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result.reset_index(drop=True)

result_df = transform(df)
```

### Example 8: Data with condition prefixes in column names
User: "Columns: Lipid, Control_rep1, Control_rep2, Control_rep3, KO_rep1, KO_rep2, KO_rep3, Mass, RT. Transform it."
```python
def transform(df):
    result = pd.DataFrame()
    result['Lipid'] = df['Lipid']
    sample_cols = [c for c in df.columns if '_rep' in c]
    for col in sample_cols:
        result[col] = pd.to_numeric(df[col], errors='coerce')
    return result

result_df = transform(df)
```
"""


def get_data_questions_examples() -> str:
    """Few-shot examples for the Data Questions tab."""
    return """\
## Few-Shot Examples

### Example 1: Simple schema question
User: "How many columns does my data have?"
(No code needed — answer from schema: "Your dataset has 15 columns: ...")

### Example 2: Species count per class
User: "How many lipid species are in each class?"
```python
lipid_col = df.columns[0]
classes = df[lipid_col].str.extract(r'^([A-Za-z]+)')[0]
result_df = classes.value_counts().reset_index()
result_df.columns = ['Lipid Class', 'Species Count']
```

### Example 3: Missing value analysis
User: "Are there any missing values?"
```python
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("No missing values found in your dataset.")
else:
    result_df = missing.reset_index()
    result_df.columns = ['Column', 'Missing Count']
```

### Example 4: Summary statistics
User: "Give me summary statistics for the sample columns"
```python
numeric_cols = df.select_dtypes(include='number').columns.tolist()
result_df = df[numeric_cols].describe().T.reset_index()
result_df.columns = ['Column'] + list(df[numeric_cols].describe().index)
```

### Example 5: Top N most abundant lipids
User: "What are the top 10 most abundant lipids across all samples?"
```python
lipid_col = df.columns[0]
numeric_cols = df.select_dtypes(include='number').columns.tolist()
df['_total'] = df[numeric_cols].sum(axis=1)
top10 = df.nlargest(10, '_total')[[lipid_col, '_total']].copy()
top10.columns = ['Lipid', 'Total Intensity']
result_df = top10.reset_index(drop=True)
```

### Example 6: Sample intensity totals
User: "What is the total intensity per sample?"
```python
numeric_cols = df.select_dtypes(include='number').columns.tolist()
totals = df[numeric_cols].sum()
result_df = totals.reset_index()
result_df.columns = ['Sample', 'Total Intensity']
```

### Example 7: Zero/missing value percentage per sample
User: "What percentage of values are zero or missing in each sample?"
```python
numeric_cols = df.select_dtypes(include='number').columns.tolist()
rows = []
for col in numeric_cols:
    n_total = len(df[col])
    n_zero = (df[col] == 0).sum()
    n_missing = df[col].isnull().sum()
    pct = round((n_zero + n_missing) / n_total * 100, 1)
    rows.append({'Sample': col, 'Zero': n_zero, 'Missing': n_missing, 'Pct': pct})
result_df = pd.DataFrame(rows)
```

### Example 8: Unique lipid classes
User: "What lipid classes are in my data?"
(No code needed if ClassKey is in schema — answer directly: "Your dataset contains X classes: PC, PE, SM, ...")
If no ClassKey column:
```python
lipid_col = df.columns[0]
classes = df[lipid_col].str.extract(r'^([A-Za-z]+)')[0].dropna().unique()
print(f"Found {len(classes)} lipid classes: {', '.join(sorted(classes))}")
```
"""


def get_viz_examples() -> str:
    """Few-shot examples for the Custom Visualization tab.

    NOTE: LipidCruncher Module 3 already provides bar charts (class-level
    mean ± std), pie charts (class composition), saturation profiles,
    chain length analysis, FACH, pathway diagrams, volcano plots, and
    clustered heatmaps. Examples here focus on custom analyses that
    the built-in pipeline does NOT offer.
    """
    return """\
## Few-Shot Examples

### Example 1: Species count per class
User: "Plot the total number of species in each lipid class"
```python
species_counts = df.groupby('ClassKey').size().reset_index(name='Count')
species_counts = species_counts.sort_values('Count', ascending=False)
fig = px.bar(species_counts, x='ClassKey', y='Count',
             title='Number of Lipid Species per Class',
             color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Species Count')
# Pin axis order — DataFrame sort is not enough for categorical bars.
fig.update_xaxes(categoryorder='array',
                 categoryarray=species_counts['ClassKey'].tolist())
```

### Example 2: Top N most abundant individual lipids
User: "Bar chart of the top 15 most abundant lipids by mean concentration"
```python
conc_cols = [c for c in df.columns if c.startswith('concentration[') or c.startswith('intensity[')]
df_plot = df.copy()
df_plot['_mean'] = df_plot[conc_cols].mean(axis=1)
top15 = df_plot.nlargest(15, '_mean')
fig = px.bar(top15, x='LipidMolec', y='_mean',
             title='Top 15 Most Abundant Lipids',
             color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Lipid', yaxis_title='Mean Concentration',
                  xaxis_tickangle=-45)
# Pin axis order — DataFrame sort is not enough for categorical bars.
fig.update_xaxes(categoryorder='array',
                 categoryarray=top15['LipidMolec'].tolist())
```

### Example 3: Scatter plot comparing two samples
User: "Scatter plot of intensity[s1] vs intensity[s2] with a trend line"
```python
fig = px.scatter(df, x='intensity[s1]', y='intensity[s2]',
                 hover_name='LipidMolec', trendline='ols',
                 title='Sample s1 vs s2 Intensities',
                 color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Intensity (s1)', yaxis_title='Intensity (s2)')
```

### Example 4: Distribution of intensities per class (violin)
User: "Violin plot showing the distribution of intensities for each lipid class"
```python
conc_cols = [c for c in df.columns if c.startswith('concentration[') or c.startswith('intensity[')]
df_plot = df.copy()
df_plot['_mean'] = df_plot[conc_cols].mean(axis=1)
fig = px.violin(df_plot, x='ClassKey', y='_mean', box=True, points='outliers',
                title='Intensity Distribution per Lipid Class',
                color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Mean Intensity')
```

### Example 5: Stacked bar of species composition across conditions
User: "Stacked bar chart showing how many species each class contributes, split by condition"
```python
# Count species per class (one bar, stacked by class)
species_counts = df['ClassKey'].value_counts().reset_index()
species_counts.columns = ['ClassKey', 'Count']
species_counts = species_counts.sort_values('Count', ascending=False)
fig = px.bar(species_counts, x='ClassKey', y='Count',
             title='Species Count per Class',
             color='ClassKey')
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Number of Species',
                  showlegend=False)
# Pin axis order — DataFrame sort is not enough for categorical bars.
fig.update_xaxes(categoryorder='array',
                 categoryarray=species_counts['ClassKey'].tolist())
```

### Example 6: Ratio plot between two conditions
User: "Plot the log2 ratio of KO/WT mean intensity for each lipid class"
```python
# Adapt column names to your experiment
ko_cols = [c for c in df.columns if c.startswith('intensity[s1') or c.startswith('intensity[s2')]
wt_cols = [c for c in df.columns if c.startswith('intensity[s3') or c.startswith('intensity[s4')]
df_plot = df.copy()
df_plot['KO_mean'] = df_plot[ko_cols].mean(axis=1)
df_plot['WT_mean'] = df_plot[wt_cols].mean(axis=1)
class_ratio = df_plot.groupby('ClassKey')[['KO_mean', 'WT_mean']].mean()
class_ratio['log2_ratio'] = np.log2(class_ratio['KO_mean'] / class_ratio['WT_mean'])
class_ratio = class_ratio.reset_index().sort_values('log2_ratio')
fig = px.bar(class_ratio, x='ClassKey', y='log2_ratio',
             title='Log2 Fold Change (KO/WT) per Lipid Class',
             color='log2_ratio', color_continuous_scale='RdBu_r',
             color_continuous_midpoint=0)
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Log2(KO/WT)')
# Pin axis order — DataFrame sort is not enough for categorical bars.
fig.update_xaxes(categoryorder='array',
                 categoryarray=class_ratio['ClassKey'].tolist())
```

### Example 7: Coefficient of variation across samples
User: "Plot the CV for each lipid across all samples"
```python
conc_cols = [c for c in df.columns if c.startswith('concentration[') or c.startswith('intensity[')]
df_plot = df.copy()
df_plot['CV'] = df_plot[conc_cols].std(axis=1) / df_plot[conc_cols].mean(axis=1) * 100
df_plot = df_plot.sort_values('CV', ascending=False)
fig = px.histogram(df_plot, x='CV', nbins=50,
                   title='Distribution of Coefficient of Variation Across Lipids',
                   color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='CV (%)', yaxis_title='Number of Lipids')
```

### Example 8: Class-level intensity comparison (dot plot)
User: "Dot plot comparing mean intensity of each class between two conditions"
```python
ko_cols = [c for c in df.columns if c.startswith('intensity[s1') or c.startswith('intensity[s2')]
wt_cols = [c for c in df.columns if c.startswith('intensity[s3') or c.startswith('intensity[s4')]
df_plot = df.copy()
df_plot['KO'] = df_plot[ko_cols].mean(axis=1)
df_plot['WT'] = df_plot[wt_cols].mean(axis=1)
class_means = df_plot.groupby('ClassKey')[['KO', 'WT']].mean().reset_index()
melted = class_means.melt(id_vars='ClassKey', var_name='Condition', value_name='Mean Intensity')
fig = px.scatter(melted, x='ClassKey', y='Mean Intensity', color='Condition',
                 title='Class-Level Intensity: KO vs WT',
                 color_discrete_sequence=['#1f77b4', '#ff7f0e'])
fig.update_traces(marker_size=12)
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Mean Intensity')
```
"""


def get_lc_questions_examples() -> str:
    """Few-shot examples for the LipidCruncher Questions tab."""
    return """\
## Few-Shot Examples

### Example 1: Normalization methods
User: "What normalization methods does LipidCruncher support?"
Answer: "LipidCruncher offers 5 normalization options: (1) None — use raw intensities, \
(2) Internal standard — normalize each lipid class by a user-assigned internal standard \
(you map each class to a standard; classes without their own standard can use one from \
a structurally similar class, e.g., PS can use a PE standard), \
(3) Protein — divide by protein concentration (BCA assay), (4) Both — internal standard \
then protein, (5) Total intensity — for each sample, divide every lipid intensity by that \
sample's total, then multiply by the median total across all samples (keeps values in a \
biologically meaningful range while equalizing total signal)."

### Example 2: Statistical testing
User: "What statistical test should I use for comparing 3 conditions?"
Answer: "For 3+ conditions, LipidCruncher uses Alexander-Govern ANOVA (parametric) or \
Kruskal-Wallis (non-parametric). Post-hoc pairwise comparisons use Tukey's HSD \
(with parametric) or Bonferroni-corrected Mann-Whitney U (with non-parametric). \
Multiple testing correction across all lipids is available via FDR (Benjamini-Hochberg) \
or Bonferroni. Auto mode selects parametric testing with log10 transformation."

### Example 3: Data format
User: "What is the Generic format?"
Answer: "The Generic format is the simplest input for LipidCruncher. Column 1 = lipid names \
(any notation), columns 2+ = numeric sample intensities. No metadata columns."

### Example 4: Pipeline overview
User: "What does the LipidCruncher analysis pipeline look like?"
Answer: "LipidCruncher has 3 modules: Module 1 (format detection, column standardization, \
data cleaning, zero filtering, internal standards management, normalization), \
Module 2 (quality checks — box plots, BQC coefficient of variation, retention time \
consistency, pairwise Pearson correlation, PCA with confidence ellipses), and \
Module 3 (analysis & visualization — bar charts with statistical testing, pie charts, \
saturation profiles, chain length, FACH, pathway diagrams, volcano plots, heatmaps)."

### Example 5: Internal standards
User: "How does internal standard normalization work?"
Answer: "LipidCruncher auto-detects internal standards in your data via regex patterns \
in lipid names — it looks for markers like (d7), (d9), (IS), _IS, ISTD, SPLASH, and \
similar patterns, as well as ClassKey values containing 'ISTD' or 'Internal'. \
You then map each lipid class to a standard — this mapping is flexible, so if you \
don't have a standard for a particular class (e.g., PS), you can assign a structurally \
similar one (e.g., a PE standard). Each species is normalized by: \
(intensity / IS_intensity) * IS_concentration. Every class must have a mapping — \
LipidCruncher will raise an error if any class is unmapped."

### Example 6: Supported formats
User: "What data formats can I upload?"
Answer: "LipidCruncher supports 4 formats: (1) LipidSearch 5.0 — identified by MeanArea \
columns plus metadata like LipidMolec, ClassKey, CalcMass, BaseRt, TotalGrade, \
(2) MS-DIAL — identified by Metabolite name column plus signature columns like \
Alignment ID, Average Rt(min), Total score, Ontology, \
(3) Generic — simplest format: lipid names in column 1, numeric intensities in columns 2+, \
(4) Metabolomics Workbench — tab-delimited with MS_METABOLITE_DATA_START/END markers. \
Format is auto-detected on upload."

### Example 7: Quality checks
User: "What quality checks does LipidCruncher perform?"
Answer: "Module 2 includes 5 quality checks: (1) Box plots showing concentration \
distributions per sample, (2) BQC assessment — calculates coefficient of variation \
for each lipid across BQC replicate samples to assess measurement precision, \
(3) Retention time plots for RT consistency (available for LipidSearch and MS-DIAL only), \
(4) Pairwise Pearson correlation heatmaps between replicates, and \
(5) PCA — 2-component principal component analysis with z-score standardization, \
colored by condition, with confidence ellipses."

### Example 8: Lipid nomenclature
User: "How does LipidCruncher standardize lipid names?"
Answer: "All names are converted to LIPID MAPS shorthand notation: class and chains \
separated by space (PC 34:1), general chains by underscore (PC 16:0_18:1), sphingolipid \
chains by slash (Cer 18:1;O2/24:0). Lyso species have phantom chains removed. Chains \
are sorted by carbon count, then double bonds."
"""
