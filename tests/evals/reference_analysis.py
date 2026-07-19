"""Reference computations for the canonical analysis dataset.

This is the **oracle** the eval harness (and any AI-generated analysis code)
is checked against for the "deterministic computation" tier of questions —
"how many species?", "top N by abundance", "most significant changes",
"per-class totals", "fold changes", etc.

Two sources of ground truth, deliberately kept separate:

- **Descriptive / aggregation** answers use plain pandas here. They are trivial
  and hand-verifiable against the values recorded in CANONICAL_ANALYSIS.md.

- **Statistical** answers delegate to the app's own ``StatisticalTestingService``
  — Welch's t-test on log10-transformed concentrations with Benjamini-Hochberg
  correction and log2 fold change. That service *is* LipidCruncher's documented
  method, so it defines "correct" by construction. It is independent of any
  AI-generated code path, so using it as the oracle for the AI eval is not
  circular. The dataset also carries a **planted effect** (see the module doc),
  which lets tests assert the biologically-expected outcome without recomputing
  any p-value — and would catch the stats service itself regressing.
"""
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.models.statistics import StatisticalTestConfig
from app.services.statistical_testing import StatisticalTestingService

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "canonical_analysis_dataset.csv"
)

CONTROL_SAMPLES = ["s1", "s2", "s3"]
KO_SAMPLES = ["s4", "s5", "s6"]


def load_canonical() -> pd.DataFrame:
    """Load the frozen analysis-ready canonical dataset."""
    return pd.read_csv(DATASET_PATH)


def _conc_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("concentration[")]


# ── Descriptive / aggregation (independent pandas) ─────────────────────────

def n_species(df: pd.DataFrame) -> int:
    return len(df)


def n_classes(df: pd.DataFrame) -> int:
    return int(df["ClassKey"].nunique())


def grand_total(df: pd.DataFrame) -> float:
    return float(df[_conc_cols(df)].values.sum())


def total_per_lipid(df: pd.DataFrame) -> pd.Series:
    """Total concentration per lipid across all samples."""
    return df.set_index("LipidMolec")[_conc_cols(df)].sum(axis=1)


def top_n_by_abundance(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """The n most abundant lipids by summed concentration (stable order)."""
    return total_per_lipid(df).sort_values(ascending=False, kind="stable").head(n)


def per_class_totals(df: pd.DataFrame) -> pd.Series:
    """Total concentration per lipid class across all samples (descending)."""
    return (
        df.groupby("ClassKey")[_conc_cols(df)]
        .sum()
        .sum(axis=1)
        .sort_values(ascending=False, kind="stable")
    )


# ── Statistical (app's documented method — the oracle) ─────────────────────

def species_stats(
    df: pd.DataFrame,
    control: Optional[List[str]] = None,
    ko: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-lipid statistics via the app's own species-level test engine.

    Returns a frame indexed by lipid with columns:
    ``log2fc`` (log2 fold change, KO / control), ``p_value``, ``adj_p``
    (Benjamini-Hochberg), and ``significant`` (adj_p <= alpha).
    """
    control = control or CONTROL_SAMPLES
    ko = ko or KO_SAMPLES
    cfg = StatisticalTestConfig.create_auto()
    summary = StatisticalTestingService.run_species_level_tests(df, control, ko, cfg)
    rows = [
        (lipid, r.effect_size, r.p_value, r.adjusted_p_value, r.significant)
        for lipid, r in summary.results.items()
    ]
    return pd.DataFrame(
        rows, columns=["lipid", "log2fc", "p_value", "adj_p", "significant"]
    ).set_index("lipid")


def significant_lipids(df: pd.DataFrame) -> pd.DataFrame:
    """Lipids passing the significance threshold, most significant first."""
    stats = species_stats(df)
    return stats[stats.significant].sort_values("adj_p", kind="stable")


def top_n_by_significance(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n lipids with the smallest adjusted p-value (stable order)."""
    return species_stats(df).sort_values("adj_p", kind="stable").head(n)
