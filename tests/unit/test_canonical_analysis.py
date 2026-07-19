"""
Verification of the canonical analysis dataset and its reference computations.

Two jobs, mirroring test_chain_fixtures.py:

1. **Guard the dataset + reference module** against silent drift — the descriptive
   values are pinned to the hand-checkable numbers in CANONICAL_ANALYSIS.md.

2. **Assert the planted biological effect independently of exact statistics** — the
   six planted lipids (and only those) must come out significant with the expected
   fold-change direction. This holds regardless of the precise p-values, and would
   also catch the app's own statistics engine regressing.

The reference computations live in ``tests/evals/reference_analysis.py`` and are the
oracle the eval harness / AI-generated analysis code get compared against.
"""
import sys
from pathlib import Path

import pytest

# Make the tests/ dir importable so `evals` (the eval-harness package) resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import reference_analysis as ref  # noqa: E402


PLANTED_UP = {"PC 16:0_18:1", "TG 16:0_18:1_18:1", "PE 18:0_20:4"}
PLANTED_DOWN = {"PC 18:0_18:1", "SM d18:1_16:0", "Cer d18:1_24:0"}
PLANTED = PLANTED_UP | PLANTED_DOWN


@pytest.fixture(scope="module")
def df():
    return ref.load_canonical()


# ── Descriptive (hand-checkable ground truth) ──────────────────────────────

def test_species_and_class_counts(df):
    assert ref.n_species(df) == 24
    assert ref.n_classes(df) == 6


def test_grand_total(df):
    assert ref.grand_total(df) == pytest.approx(22306)


def test_per_class_totals(df):
    totals = ref.per_class_totals(df)
    assert list(totals.index) == ["PC", "PE", "SM", "Cer", "TG", "DG"]
    assert totals.to_dict() == pytest.approx(
        {"PC": 6350, "PE": 4379, "SM": 3445, "Cer": 3415, "TG": 3063, "DG": 1654}
    )


def test_top_abundance_and_the_intentional_tie(df):
    top = ref.top_n_by_abundance(df, n=5)
    # The two most abundant tie at 1545 — generated code must not pretend there's
    # a unique winner.
    assert top.iloc[0] == pytest.approx(1545)
    assert top.iloc[1] == pytest.approx(1545)
    assert set(top.index[:2]) == {"PC 16:0_18:1", "PC 18:0_18:1"}
    assert top.loc["Cer d18:1_16:0"] == pytest.approx(1337)


# ── Statistical: planted effect (independent of exact p-values) ────────────

def test_exactly_the_planted_lipids_are_significant(df):
    sig = set(ref.significant_lipids(df).index)
    assert sig == PLANTED


def test_fold_change_directions_match_the_plant(df):
    stats = ref.species_stats(df)
    for lipid in PLANTED_UP:
        assert stats.loc[lipid, "log2fc"] > 0, f"{lipid} should be up in KO"
    for lipid in PLANTED_DOWN:
        assert stats.loc[lipid, "log2fc"] < 0, f"{lipid} should be down in KO"


def test_unchanged_lipids_are_not_significant(df):
    stats = ref.species_stats(df)
    unchanged = [l for l in stats.index if l not in PLANTED]
    assert not stats.loc[unchanged, "significant"].any()


def test_top_significance_ranking_is_the_planted_set(df):
    # The 6 smallest adjusted p-values are exactly the planted lipids.
    top6 = ref.top_n_by_significance(df, n=6)
    assert set(top6.index) == PLANTED
    assert top6["significant"].all()


# ── Statistical: pin exact values (regression against the documented method) ─

# Tracks the app's own Welch/log10/BH pipeline; changes intentionally if that
# method changes. Values from CANONICAL_ANALYSIS.md.
_EXPECTED_STATS = {
    "PE 18:0_20:4":      (+1.987, 2.56e-05),
    "SM d18:1_16:0":     (-2.298, 6.66e-05),
    "Cer d18:1_24:0":    (-2.294, 7.42e-05),
    "PC 16:0_18:1":      (+1.965, 1.30e-04),
    "PC 18:0_18:1":      (-1.965, 1.30e-04),
    "TG 16:0_18:1_18:1": (+2.543, 2.31e-04),
}


@pytest.mark.parametrize("lipid,log2fc,adj_p", [
    (k, v[0], v[1]) for k, v in _EXPECTED_STATS.items()
])
def test_documented_stats_values_are_stable(df, lipid, log2fc, adj_p):
    stats = ref.species_stats(df)
    assert stats.loc[lipid, "log2fc"] == pytest.approx(log2fc, abs=1e-3)
    assert stats.loc[lipid, "adj_p"] == pytest.approx(adj_p, rel=1e-2)
