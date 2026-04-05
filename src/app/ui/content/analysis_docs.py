"""Static documentation for analysis features (statistical testing, saturation profiles)."""

STATISTICAL_TESTING_DOCS = """
## 🧪 Statistical Testing in LipidCruncher

LipidCruncher uses rigorous statistical methods to help you identify meaningful differences in your lipidomic data. This guide explains everything you need to know—no statistics PhD required!

---

### 🎯 The Big Picture

When comparing lipid levels across conditions, we're asking: *"Is this difference real, or just random noise?"*

Statistical tests give us a **p-value**—the probability of seeing this difference (or more extreme) by chance alone. Lower p-values = stronger evidence of a real difference.

**The catch?** When you test many things at once, some will look significant just by luck. Test 100 lipids, and ~5 will have p < 0.05 purely by chance. That's where multiple testing corrections come in.

---

### 🔬 Available Tests

| Test Type | 2 Conditions | 3+ Conditions | When to Use |
|-----------|--------------|---------------|-------------|
| **Parametric** | Welch's t-test | Welch's ANOVA | Default choice. Works great with log-transformed lipidomic data |
| **Non-parametric** | Mann-Whitney U | Kruskal-Wallis | More conservative. Use when you want maximum rigor or have unusual distributions |

**Why Welch's?** Unlike classic t-tests and ANOVA, Welch's versions don't assume equal variances across groups—important for biological data where variability often differs between conditions.

**Why log transformation?** Lipidomic data is typically right-skewed (a few very high values). Log10 transformation makes the data more symmetric and suitable for parametric tests. This is standard practice in the field.

---

### 🎚️ Two-Level Correction Framework

LipidCruncher uses a **two-level system** to control false discoveries at different scales:

#### Level 1: Between-Class Correction
*"Across all the lipid classes I'm testing, how do I control false discoveries?"*

| Method | What It Does | Best For |
|--------|--------------|----------|
| **Uncorrected** | Raw p-values, no adjustment | Single pre-specified hypothesis |
| **FDR** (Benjamini-Hochberg) | Controls the *proportion* of false discoveries | Exploratory analysis (recommended default) |
| **Bonferroni** | Strictest control, adjusts for every test | Confirmatory studies where false positives are costly |

#### Level 2: Within-Class Post-hoc Correction
*"With 3+ conditions, which specific pairs are different?"*

After a significant omnibus test (ANOVA/Kruskal-Wallis), post-hoc tests identify *which* groups differ:

| Method | What It Does | Best For |
|--------|--------------|----------|
| **Uncorrected** | Raw pairwise p-values | Maximum discovery (higher false positive risk) |
| **Tukey's HSD** | Controls family-wise error rate efficiently | Recommended for parametric tests |
| **Bonferroni** | Most conservative pairwise correction | Maximum rigor |

**Note:** Tukey's HSD is only available for parametric tests. When you select non-parametric analysis, the "Tukey's HSD" option automatically switches to Bonferroni-corrected Mann-Whitney U pairwise tests.

---

### 🤖 Auto Mode: Smart Defaults

Don't want to think about statistics? Auto mode applies field-standard choices:

- **Test type:** Parametric (Welch's) with log10 transformation
- **Level 1:** Uncorrected for single class → FDR for multiple classes
- **Level 2:** Uncorrected for ≤2 conditions → Tukey's HSD for 3+ conditions

These defaults balance discovery power with false positive control for typical lipidomics experiments.

---

### 📊 The Testing Process

**2 Conditions:** Direct pairwise test → Level 1 correction (if testing multiple classes)

**3+ Conditions:**

1. **Omnibus test** for each lipid class: "Are there ANY differences among groups?"
2. **Level 1 correction** adjusts all omnibus p-values across classes
3. **Post-hoc tests** run only for classes with significant omnibus results
4. **Level 2 correction** adjusts pairwise p-values within each class

**How the levels work together:**
- Level 1 asks: "Which lipid classes show any difference?" (corrects across classes)
- Level 2 asks: "Within those classes, which specific condition pairs differ?" (corrects within each class)

---

### ⚠️ Critical Best Practices

1. **Pre-specify your hypotheses.** Decide which lipid classes to test *before* looking at results. Adding "interesting-looking" classes after the fact inflates false positives.

2. **More tests = less power.** Each additional class or comparison dilutes your statistical power. Focus on what matters biologically.

3. **Sample size matters.** With n=3 per group (common in lipidomics), non-parametric tests have very limited resolution—you may see identical p-values for many comparisons. Parametric tests with log transformation typically perform better.

4. **Significance ≠ importance.** A tiny p-value doesn't mean a biologically meaningful effect. Always consider effect sizes (fold changes) alongside p-values.

**Rule of thumb:** FDR + Tukey's HSD works well for most lipidomics analyses.
"""

SATURATION_PROFILE_DOCS = """
## How SFA, MUFA, and PUFA Values Are Computed

For each lipid molecule in your dataset, the algorithm:

### 1. Fatty Acid Chain Parsing
Identifies individual fatty acid chains from the lipid name:
- ✓ `PC 16:0_18:1` → Successfully parses to chains: [16:0, 18:1]
- ✓ `PE 18:0_20:4` → Successfully parses to chains: [18:0, 20:4]
- ✗ `PC 34:1` → Cannot parse individual chains (consolidated format)

### 2. Classification by Saturation
Counts double bonds in each chain:
- **SFA** (0 double bonds): 16:0, 18:0, 24:0
- **MUFA** (1 double bond): 16:1, 18:1, 24:1
- **PUFA** (2+ double bonds): 18:2, 20:4, 22:6

### 3. Weighted Contribution Calculation
Multiplies lipid concentration by fatty acid ratio.

**Example:** `PC 16:0_18:1` at 100 µM contributes:
- SFA: 100 × 0.5 = 50 µM (1 of 2 chains is saturated)
- MUFA: 100 × 0.5 = 50 µM (1 of 2 chains is monounsaturated)
- PUFA: 100 × 0 = 0 µM (no polyunsaturated chains)

**Why consolidated format fails:** `PC 36:2` at 100 µM could be:
- `PC 18:0_18:2`: 50 µM SFA + 50 µM PUFA
- `PC 18:1_18:1`: 100 µM MUFA
- `PC 16:0_20:2`: 50 µM SFA + 50 µM PUFA
- **Same total, completely different saturation profiles** — algorithm cannot determine which

### 4. Class-Level Summation
All contributions within a lipid class are summed:
- Total SFA = sum of all SFA contributions from all species
- Total MUFA = sum of all MUFA contributions from all species
- Total PUFA = sum of all PUFA contributions from all species

### Two Visualizations

**Concentration Profile:** Absolute SFA, MUFA, PUFA values with error bars (standard deviation)

**Percentage Distribution:** Relative proportions (always sums to 100%)

### Handling Consolidated Format Data

LipidCruncher automatically detects lipids in consolidated format within your selected classes.

**What happens if you keep them:** The algorithm classifies based on total double bonds only.
For example, `PC 34:1` is treated as 100% MUFA, when it might actually be 50% SFA + 50% MUFA.

**What happens if you exclude them:** The remaining lipids are classified accurately,
but you lose the abundance contribution from excluded species.

You'll be prompted to review detected lipids and decide based on your analysis goals.
"""