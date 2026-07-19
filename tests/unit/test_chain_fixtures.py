"""
Verification of the fatty-acid chain fixtures in tests/fixtures/.

These tests serve two purposes:

1. **Guard the fixtures.** They pin the hand-computed ground truth documented in
   tests/fixtures/CHAIN_FIXTURES.md, so if a fixture CSV is ever edited the
   documented values and the data can't silently drift apart.

2. **Provide an independent reference implementation.** ``_fa_pools`` below computes
   the C16/C18 pools using the app's own nomenclature primitives but is written
   deliberately and reviewed, separate from any AI-generated code path. It is the
   oracle that AI-generated chain-aggregation code can be compared against.

The expected numbers here are arithmetic, verifiable by hand on 3 rows — they are
NOT a judgement about which counting convention is biologically correct. Both
conventions are computed; choosing between them is an expert decision recorded in
CHAIN_FIXTURES.md.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from app.lipid_nomenclature import parse_chain_carbon_db, parse_lipid_name
from app.services.data_standardization import DataStandardizationService

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def _split_chains(lipid_name: str) -> list[str]:
    """Return the individual chain tokens of a lipid name, or [] if unsplittable.

    A consolidated / sum-composition name (e.g. 'PC 34:1') has a single token and
    cannot be decomposed into chains — this returns a single-element list, and
    callers must check ``is_consolidated`` to distinguish that case.
    """
    standardized = DataStandardizationService.standardize_lipid_name(lipid_name)
    _, chain_info, _ = parse_lipid_name(standardized)
    if not chain_info:
        return []
    return re.split(r"[/_]", chain_info)


def _is_consolidated(lipid_name: str) -> bool:
    """True if the name is sum-composition only (individual chains unrecoverable)."""
    standardized = DataStandardizationService.standardize_lipid_name(lipid_name)
    _, chain_info, _ = parse_lipid_name(standardized)
    return bool(chain_info) and "_" not in chain_info and "/" not in chain_info


def _fa_pools(
    df: pd.DataFrame,
    sample: str,
    carbons: tuple[int, ...] = (16, 18),
    convention: str = "presence",
) -> dict[int, float]:
    """Sum intensity per fatty-acid carbon number for one sample.

    Args:
        df: Generic-format frame ('Lipids' + 'sample[sN]' columns).
        sample: Column name to aggregate, e.g. 'sample[s1]'.
        carbons: Carbon numbers to pool.
        convention: 'presence' (species contributes full intensity to every pool it
            touches) or 'split' (intensity divided equally across its chains).

    Returns:
        {carbon_number: pooled_intensity}
    """
    if convention not in ("presence", "split"):
        raise ValueError(f"unknown convention: {convention}")

    pools = {c: 0.0 for c in carbons}
    for _, row in df.iterrows():
        name = row["Lipids"]
        intensity = float(row[sample])
        chains = _split_chains(name)
        parsed = [parse_chain_carbon_db(c) for c in chains]
        chain_carbons = [p[0] for p in parsed if p is not None]
        if not chain_carbons:
            continue

        if convention == "presence":
            for c in carbons:
                if c in chain_carbons:
                    pools[c] += intensity
        else:  # split
            per_chain = intensity / len(chain_carbons)
            for cc in chain_carbons:
                if cc in pools:
                    pools[cc] += per_chain
    return pools


def _ratio(pools: dict[int, float]) -> float:
    return pools[18] / pools[16]


@pytest.fixture(scope="module")
def molecular() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "chain_molecular.csv")


@pytest.fixture(scope="module")
def consolidated() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "chain_consolidated.csv")


@pytest.fixture(scope="module")
def edge_cases() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "chain_edge_cases.csv")


# ---------------------------------------------------------------------------
# Fixture 1 — chain_molecular.csv
# ---------------------------------------------------------------------------

# (sample, convention, C16 pool, C18 pool, ratio) — mirrors CHAIN_FIXTURES.md
_MOLECULAR_EXPECTED = [
    ("sample[s1]", "presence", 300, 500, 5 / 3),
    ("sample[s1]", "split", 200, 400, 2.0),
    ("sample[s2]", "presence", 600, 1000, 5 / 3),
    ("sample[s2]", "split", 400, 800, 2.0),
    ("sample[s3]", "presence", 500, 300, 0.6),
    ("sample[s3]", "split", 400, 200, 0.5),
    ("sample[s4]", "presence", 250, 150, 0.6),
    ("sample[s4]", "split", 200, 100, 0.5),
]


@pytest.mark.parametrize("sample,convention,c16,c18,ratio", _MOLECULAR_EXPECTED)
def test_molecular_pools_match_documented_values(
    molecular, sample, convention, c16, c18, ratio
):
    """The documented hand-computed ground truth is reproducible."""
    pools = _fa_pools(molecular, sample, convention=convention)
    assert pools[16] == pytest.approx(c16)
    assert pools[18] == pytest.approx(c18)
    assert _ratio(pools) == pytest.approx(ratio)


@pytest.mark.parametrize("convention", ["presence", "split"])
def test_elovl6_ko_ratio_is_lower_than_control(molecular, convention):
    """Biological positive control: ELOVL6 KO must show a lower C18/C16 than control.

    ELOVL6 elongates C16 -> C18, so knocking it out reduces the C18 pool relative to
    C16. This is the assertion a domain expert can make without computing anything,
    and it must hold under either counting convention.
    """
    control = [_ratio(_fa_pools(molecular, s, convention=convention))
               for s in ("sample[s1]", "sample[s2]")]
    ko = [_ratio(_fa_pools(molecular, s, convention=convention))
          for s in ("sample[s3]", "sample[s4]")]
    assert max(ko) < min(control)


@pytest.mark.parametrize(
    "a,b,convention",
    [
        ("sample[s1]", "sample[s2]", "presence"),  # s2 = 2 x s1
        ("sample[s1]", "sample[s2]", "split"),
        ("sample[s3]", "sample[s4]", "presence"),  # s4 = 0.5 x s3
        ("sample[s3]", "sample[s4]", "split"),
    ],
)
def test_ratio_is_scale_invariant(molecular, a, b, convention):
    """Scaling every intensity in a sample must not change the ratio."""
    ra = _ratio(_fa_pools(molecular, a, convention=convention))
    rb = _ratio(_fa_pools(molecular, b, convention=convention))
    assert ra == pytest.approx(rb)


@pytest.mark.parametrize("sample", ["sample[s1]", "sample[s2]", "sample[s3]", "sample[s4]"])
def test_chain_split_conserves_mass(molecular, sample):
    """Under chain-split, the pools must sum to the sample's total intensity.

    Every chain in this fixture is C16 or C18, so nothing falls outside the pools.
    """
    pools = _fa_pools(molecular, sample, convention="split")
    assert pools[16] + pools[18] == pytest.approx(molecular[sample].sum())


def test_conventions_give_different_answers(molecular):
    """The fixture discriminates the two conventions (otherwise it couldn't detect
    which one an implementation used)."""
    presence = _ratio(_fa_pools(molecular, "sample[s1]", convention="presence"))
    split = _ratio(_fa_pools(molecular, "sample[s1]", convention="split"))
    assert presence != pytest.approx(split)


# ---------------------------------------------------------------------------
# Fixture 2 — chain_consolidated.csv (the must-refuse case)
# ---------------------------------------------------------------------------

def test_consolidated_names_are_detected_as_unsplittable(consolidated):
    """Every row is sum-composition, so individual chains cannot be recovered.

    This is what makes a C18/C16 ratio on this fixture meaningless — the correct
    behaviour for the AI is to refuse, not to return a number.
    """
    for name in consolidated["Lipids"]:
        assert _is_consolidated(name), f"{name} should be consolidated/sum-composition"


def test_consolidated_fixture_mirrors_molecular_intensities(molecular, consolidated):
    """The two fixtures are a clean A/B: identical numbers, different resolvability."""
    sample_cols = [c for c in molecular.columns if c.startswith("sample[")]
    assert sample_cols == [c for c in consolidated.columns if c.startswith("sample[")]
    for col in sample_cols:
        assert list(molecular[col]) == list(consolidated[col])


def test_consolidated_chains_do_not_resolve_to_c16_or_c18(consolidated):
    """A consolidated name parses to its summed carbon count, never to C16/C18.

    'PC 34:1' yields carbon 34 — it must not be mistaken for a C16 or C18 chain.
    """
    for name in consolidated["Lipids"]:
        chains = _split_chains(name)
        carbons = [p[0] for p in (parse_chain_carbon_db(c) for c in chains) if p]
        assert 16 not in carbons and 18 not in carbons


# ---------------------------------------------------------------------------
# Fixture 3 — chain_edge_cases.csv (sphingoid & ether rules)
# ---------------------------------------------------------------------------

def test_edge_case_chain_inventory(edge_cases):
    """Each edge-case row decomposes into the chains documented in CHAIN_FIXTURES.md."""
    expected = {
        "PC(16:0_18:1)": [(16, 0), (18, 1)],       # plain acyl reference
        "Cer(d18:1/16:0)": [(18, 1), (16, 0)],     # d18:1 sphingoid base + 16:0 acyl
        "PC(O-16:0_18:1)": [(16, 0), (18, 1)],     # O-16:0 ether chain + 18:1 acyl
    }
    for name, chains in expected.items():
        assert name in set(edge_cases["Lipids"])
        parsed = [parse_chain_carbon_db(c) for c in _split_chains(name)]
        assert parsed == chains


def test_edge_case_pools_under_count_everything_rule(edge_cases):
    """Rule (a) in CHAIN_FIXTURES.md: sphingoid base counts as C18, ether as C16.

    The reference implementation counts every parsed chain, so it implements rule (a).
    Rules (b)-(d) require excluding sphingoid/ether chains — a biology decision that
    is deliberately left open, so only (a) is asserted here.
    """
    pools = _fa_pools(edge_cases, "sample[s1]", convention="presence")
    assert pools[16] == pytest.approx(300)
    assert pools[18] == pytest.approx(300)
