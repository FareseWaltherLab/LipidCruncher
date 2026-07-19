# Canonical analysis dataset — frozen ground truth for the "computation" tier

`canonical_analysis_dataset.csv` is one small, frozen, analysis-ready dataset with a
**deliberately planted biological effect**. It is the reusable base for the whole
*deterministic-computation* category of eval questions — "how many species?", "top N
most abundant", "top N most significant changes", "fold change per lipid", "per-class
totals", and so on. Pay the cost of designing the dataset **once**; each new question is
then a few lines against `tests/evals/reference_analysis.py`.

## Design

- **Format:** analysis-ready (post-processing) — `LipidMolec`, `ClassKey`, and
  `concentration[s1..s6]`. This is the stage the statistics engine consumes.
- **24 lipid species** across **6 classes**: PC (6), TG (5), PE (5), SM (3), Cer (3), DG (2).
- **6 samples, 2 conditions × 3 replicates:** `s1,s2,s3` = **Control**, `s4,s5,s6` = **KO**.

### Planted effect (the biological positive control)

Six species were given a strong, low-variance change; the other 18 are flat (KO ≈ control).
The direction is known by construction — no computation needed to assert it:

| Lipid | Direction in KO | Expected log2FC sign |
|---|---|---|
| `PC 16:0_18:1` | **up** (~4×) | + |
| `TG 16:0_18:1_18:1` | **up** (~6×) | + |
| `PE 18:0_20:4` | **up** (~4×) | + |
| `PC 18:0_18:1` | **down** (~4×) | − |
| `SM d18:1_16:0` | **down** (~5×) | − |
| `Cer d18:1_24:0` | **down** (~5×) | − |

## Ground-truth answers

**Statistical values are produced by the app's own `StatisticalTestingService`** (Welch's
t-test on log10 concentrations, Benjamini-Hochberg correction, log2 fold change) — the
documented method. They are reproduced by `reference_analysis.species_stats()` and pinned
in `tests/unit/test_canonical_analysis.py`. Descriptive values below are independent pandas
and hand-checkable.

### Significant changes (adj. p ≤ 0.05), most significant first

Exactly the six planted species are significant; all 18 others are not.

| Lipid | log2FC | adj. p |
|---|---|---|
| `PE 18:0_20:4` | +1.99 | 2.6e-05 |
| `SM d18:1_16:0` | −2.30 | 6.7e-05 |
| `Cer d18:1_24:0` | −2.29 | 7.4e-05 |
| `PC 16:0_18:1` | +1.97 | 1.3e-04 |
| `PC 18:0_18:1` | −1.97 | 1.3e-04 |
| `TG 16:0_18:1_18:1` | +2.54 | 2.3e-04 |

(Exact p-values are asserted with tolerance in the test — they track the app's method and
will change intentionally if that method changes.)

### Descriptive (hand-checkable)

| Question | Answer |
|---|---|
| Number of species | **24** |
| Number of classes | **6** (PC, TG, PE, SM, Cer, DG) |
| Grand total intensity | **22 306** |
| Per-class total (desc.) | PC 6350, PE 4379, SM 3445, Cer 3415, TG 3063, DG 1654 |
| Top lipids by abundance | `PC 16:0_18:1` = `PC 18:0_18:1` = **1545** (tie), then `Cer d18:1_16:0` 1337, `SM d18:1_18:0` 1277, `PC 16:0_16:0` 1228 |

> Note the top-abundance tie (both PC species total 1545) — an intentional wrinkle so that
> "the single most abundant lipid" is genuinely ambiguous, testing whether generated code
> handles ties sensibly rather than silently picking one.

## What this dataset does and doesn't need

- **No new biology sign-off** for the statistical questions: LipidCruncher already documents
  the test, so the app's method *is* the ground truth. Your biologist need only confirm, once,
  "yes, that's the test we use." The hard, per-question expert sign-off from
  `CHAIN_FIXTURES.md` (choosing a counting convention) does **not** recur here.
- **Adding a new question is cheap:** call the matching `reference_analysis` function, record
  the expected value, add an assertion. No new dataset.
- **It is a regression net, not a proof of universal correctness.** It covers the common
  computation questions and the planted effect; novel questions outside it rely on the
  sandbox, the methodology-clarification guardrail, and human review.

### Sign-off

| Field | Value |
|---|---|
| Reviewed by | _(name)_ |
| Date | _(YYYY-MM-DD)_ |
| Confirmed: significance test matches documented method (Welch's t-test / log10 / BH) | _(yes/no)_ |
| Confirmed: planted directions are biologically sensible | _(yes/no)_ |
