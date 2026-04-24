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
    """Few-shot examples for the Custom Visualization tab."""
    return """\
## Few-Shot Examples

### Example 1: Bar chart of species count per class
User: "Show me a bar chart of the number of species per class"
```python
species_counts = df.groupby('ClassKey').size().reset_index(name='Count')
species_counts = species_counts.sort_values('Count', ascending=False)
fig = px.bar(species_counts, x='ClassKey', y='Count',
             title='Number of Lipid Species per Class',
             color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Lipid Class', yaxis_title='Species Count')
```

### Example 2: Box plot across conditions
User: "Box plot of concentrations for PC class across conditions"
```python
conc_cols = [c for c in df.columns if c.startswith('concentration[')]
pc_data = df[df['ClassKey'] == 'PC']
melted = pc_data.melt(id_vars=['LipidMolec'], value_vars=conc_cols,
                       var_name='Sample', value_name='Concentration')
fig = px.box(melted, x='Sample', y='Concentration',
             title='PC Concentration Distribution per Sample',
             color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Sample', yaxis_title='Concentration')
```

### Example 3: Grouped bar chart comparing classes across conditions
User: "Grouped bar chart of mean intensity for each class, grouped by condition"
```python
# Assumes conditions mapped to sample columns
condition_map = {}
for cond, samples in zip({conditions}, {sample_lists}):
    for s in samples:
        condition_map[f'intensity[{{s}}]'] = cond

intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
rows = []
for _, row in df.iterrows():
    for col in intensity_cols:
        rows.append({{'ClassKey': row['ClassKey'], 'Condition': condition_map.get(col, col),
                     'Intensity': row[col]}})
plot_df = pd.DataFrame(rows)
means = plot_df.groupby(['ClassKey', 'Condition'])['Intensity'].mean().reset_index()
fig = px.bar(means, x='ClassKey', y='Intensity', color='Condition', barmode='group',
             title='Mean Intensity per Class by Condition',
             color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
```

### Example 4: Pie chart of class composition
User: "Pie chart of the lipid class composition"
```python
class_counts = df['ClassKey'].value_counts().reset_index()
class_counts.columns = ['ClassKey', 'Count']
fig = px.pie(class_counts, values='Count', names='ClassKey',
             title='Lipid Class Composition')
```

### Example 5: Scatter plot with regression line
User: "Scatter plot of intensity[s1] vs intensity[s2] with a trend line"
```python
fig = px.scatter(df, x='intensity[s1]', y='intensity[s2]',
                 hover_name='LipidMolec', trendline='ols',
                 title='Sample s1 vs s2 Intensities',
                 color_discrete_sequence=['#1f77b4'])
fig.update_layout(xaxis_title='Intensity (s1)', yaxis_title='Intensity (s2)')
```

### Example 6: Volcano plot
User: "Volcano plot comparing KO vs WT"
```python
from scipy import stats
ko_cols = [c for c in df.columns if 'KO' in c or c in ['intensity[s1]', 'intensity[s2]']]
wt_cols = [c for c in df.columns if 'WT' in c or c in ['intensity[s3]', 'intensity[s4]']]
# Calculate fold change and p-value per lipid
results = []
for _, row in df.iterrows():
    ko_vals = [row[c] for c in ko_cols if pd.notna(row[c])]
    wt_vals = [row[c] for c in wt_cols if pd.notna(row[c])]
    if len(ko_vals) >= 2 and len(wt_vals) >= 2:
        fc = np.log2(np.mean(ko_vals) / np.mean(wt_vals)) if np.mean(wt_vals) > 0 else 0
        _, pval = stats.ttest_ind(ko_vals, wt_vals)
        results.append({{'Lipid': row['LipidMolec'], 'log2FC': fc, '-log10p': -np.log10(pval)}})
plot_df = pd.DataFrame(results)
fig = px.scatter(plot_df, x='log2FC', y='-log10p', hover_name='Lipid',
                 title='Volcano Plot: KO vs WT')
fig.add_hline(y=-np.log10(0.05), line_dash='dash', line_color='red')
fig.add_vline(x=-1, line_dash='dash', line_color='gray')
fig.add_vline(x=1, line_dash='dash', line_color='gray')
```

### Example 7: Heatmap of top variable lipids
User: "Heatmap of the top 20 most variable lipids"
```python
intensity_cols = [c for c in df.columns if c.startswith('intensity[') or c.startswith('concentration[')]
df['_cv'] = df[intensity_cols].std(axis=1) / df[intensity_cols].mean(axis=1)
top20 = df.nlargest(20, '_cv')
heat_data = top20.set_index('LipidMolec')[intensity_cols]
fig = px.imshow(heat_data, aspect='auto',
                title='Top 20 Most Variable Lipids',
                color_continuous_scale='RdBu_r')
fig.update_layout(xaxis_title='Sample', yaxis_title='Lipid')
```

### Example 8: Violin plot per condition
User: "Violin plot of total intensity per condition"
```python
intensity_cols = [c for c in df.columns if c.startswith('intensity[')]
totals = df[intensity_cols].sum()
# Build condition mapping from experiment context
rows = []
for col in intensity_cols:
    rows.append({{'Sample': col, 'Total Intensity': totals[col]}})
plot_df = pd.DataFrame(rows)
fig = px.violin(plot_df, y='Total Intensity', box=True, points='all',
                title='Total Intensity Distribution per Sample',
                color_discrete_sequence=['#1f77b4'])
```
"""


def get_lc_questions_examples() -> str:
    """Few-shot examples for the LipidCruncher Questions tab."""
    return """\
## Few-Shot Examples

### Example 1: Normalization methods
User: "What normalization methods does LipidCruncher support?"
Answer: "LipidCruncher offers 5 normalization options: (1) None — use raw intensities, \
(2) Internal standard — normalize each class by its internal standard, \
(3) Protein — divide by protein concentration, (4) Both — internal standard then protein, \
(5) Total intensity — normalize by total signal per sample."

### Example 2: Statistical testing
User: "What statistical test should I use for comparing 3 conditions?"
Answer: "For 3+ conditions, LipidCruncher uses Alexander-Govern ANOVA (parametric) or \
Kruskal-Wallis (non-parametric), with post-hoc pairwise comparisons via Tukey's HSD \
or Bonferroni. Auto mode selects parametric with log10 transformation."

### Example 3: Data format
User: "What is the Generic format?"
Answer: "The Generic format is the simplest input for LipidCruncher. Column 1 = lipid names \
(any notation), columns 2+ = numeric sample intensities. No metadata columns."

### Example 4: Pipeline overview
User: "What does the LipidCruncher analysis pipeline look like?"
Answer: "LipidCruncher has 3 modules: Module 1 (data cleaning, filtering, normalization), \
Module 2 (quality checks — box plots, BQC, retention time, correlation, PCA), and \
Module 3 (visualizations — bar charts, pie charts, saturation, chain length, FACH, \
pathway, volcano plots, heatmaps)."

### Example 5: Internal standards
User: "How does internal standard normalization work?"
Answer: "LipidCruncher detects internal standards via regex patterns in lipid names \
(e.g., names containing 'IS' or 'd7'). Each lipid species is normalized by dividing \
its intensity by the internal standard intensity from the same lipid class. If a class \
has no IS, the species are left unnormalized."

### Example 6: Supported formats
User: "What data formats can I upload?"
Answer: "LipidCruncher supports 4 formats: (1) LipidSearch 5.0 — with MeanArea columns, \
(2) MS-DIAL — with Metabolite name and Alignment ID, (3) Generic — lipid names + intensities, \
(4) Metabolomics Workbench — tab-delimited with START/END markers. Format is auto-detected."

### Example 7: Quality checks
User: "What quality checks does LipidCruncher perform?"
Answer: "Module 2 includes 5 quality checks: box plots (concentration distributions), \
BQC assessment (coefficient of variation for batch QC samples), retention time consistency \
(LipidSearch/MS-DIAL only), pairwise correlation heatmaps, and PCA with outlier removal."

### Example 8: Lipid nomenclature
User: "How does LipidCruncher standardize lipid names?"
Answer: "All names are converted to LIPID MAPS shorthand notation: class and chains \
separated by space (PC 34:1), general chains by underscore (PC 16:0_18:1), sphingolipid \
chains by slash (Cer 18:1;O2/24:0). Lyso species have phantom chains removed. Chains \
are sorted by carbon count, then double bonds."
"""
