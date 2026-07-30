"""The golden set: questions with checkable expectations.

Expectations are deliberately *not* exact answers. LipidCruncher is the
source of truth, so if the AI routes a question to `lc.*` the numbers are
right by construction — which makes "did it route correctly, and is the
answer usable?" the thing worth asserting. See section 2 of the eval plan.

Every case records **why it exists**. Cases mined from real failures keep
their history, since a fixed bug is the cheapest regression test we have.
"""
from dataclasses import dataclass, field
from typing import Tuple


# Datasets a case can run against (see tests/fixtures/).
CANONICAL = "canonical"          # 24 species, WT vs KO, 6 planted effects
CHAIN_MOLECULAR = "chain_molecular"        # answerable chain question
CHAIN_CONSOLIDATED = "chain_consolidated"  # must refuse: sum composition


@dataclass(frozen=True)
class EvalCase:
    """One graded question.

    Attributes:
        id: Stable identifier, used in test output.
        question: Verbatim user prompt.
        dataset: Fixture to load.
        tab: 'data_questions' or 'visualization'.
        expect_lc: Substrings that must appear in the generated code —
            the pipeline calls this question should route to.
        forbid: Substrings that must NOT appear — hand-rolled analysis the
            pipeline already implements.
        expect: Usability requirements: 'table', 'figure', 'method_stated',
            'refusal'.
        note: Why this case is in the set.
    """
    id: str
    question: str
    dataset: str
    tab: str = "data_questions"
    expect_lc: Tuple[str, ...] = ()
    forbid: Tuple[str, ...] = ()
    expect: Tuple[str, ...] = ()
    note: str = ""


# Analysis the pipeline already implements; writing it by hand means the
# answer can silently disagree with what the app reports for the same data.
HAND_ROLLED_STATS = (
    "ttest_ind", "ttest_rel", "mannwhitneyu",
    "multipletests", "fdr_", "bonferroni",
)


CASES = (
    # ---- Mined from real preview failures (2026-07-27/28) --------------
    EvalCase(
        id="top10_significant",
        question=(
            "top 10 lipids with the most statistical significance when "
            "comparing WT/KO lipids"
        ),
        dataset=CANONICAL,
        expect_lc=("lc.compare_species",),
        forbid=HAND_ROLLED_STATS,
        expect=("table", "method_stated"),
        note=(
            "Broke three separate times: hand-rolled Welch+FDR and hit the "
            "scipy block; then printed lc.conditions() and ended the turn; "
            "then rendered as one mangled line with no methodology."
        ),
    ),
    EvalCase(
        id="saturation_tg",
        question="Show me the saturation profile for TG, comparing WT and KO.",
        dataset=CANONICAL,
        tab="visualization",
        expect_lc=("lc.saturation",),
        expect=("figure",),
        note=(
            "AttributeError: saturation returned bare figures while every "
            "other helper returned .figure. API shape, not model error."
        ),
    ),

    # ---- Routing: the pipeline owns these analyses ---------------------
    EvalCase(
        id="volcano_plot",
        question="Make a volcano plot comparing WT and KO.",
        dataset=CANONICAL,
        tab="visualization",
        expect_lc=("lc.volcano",),
        forbid=HAND_ROLLED_STATS,
        expect=("figure",),
        note="Should reuse the app's volcano, not scatter it by hand.",
    ),
    EvalCase(
        id="chain_length_distribution",
        question=(
            "Compare the fatty acid chain length distribution between "
            "WT and KO."
        ),
        dataset=CANONICAL,
        tab="visualization",
        expect_lc=("lc.chain_length",),
        # Parsing chain composition by hand re-invents the counting
        # convention the biologist is signing off on (see plan 3.D).
        forbid=("re.findall", "re.search", "split(':')", 'split(":")'),
        expect=("figure",),
        note="Chain-counting convention must come from the pipeline.",
    ),
    EvalCase(
        id="class_abundance",
        question="Which lipid classes are most abundant?",
        dataset=CANONICAL,
        note=(
            "Deliberately has NO routing expectation. The first run failed "
            "this for using df.groupby('ClassKey').sum() instead of "
            "lc.class_summary — but a group-and-sum is not a contested "
            "method and cannot diverge from the app in any way that matters. "
            "Requiring lc here would train the eval to flag harmless "
            "behaviour, which costs more than it catches. Routing is "
            "enforced where divergence is real: statistics and chain rules."
        ),
    ),

    # ---- Usability: plain questions must still answer cleanly ----------
    EvalCase(
        id="describe_dataset",
        question="Describe my dataset.",
        dataset=CANONICAL,
        note="No lc call required — but it must produce a real answer.",
    ),

    # ---- Honesty: the correct answer is a refusal ----------------------
    EvalCase(
        id="chain_pools_consolidated",
        question=(
            "What is the total abundance of C18 fatty acids versus C16 "
            "fatty acids?"
        ),
        dataset=CHAIN_CONSOLIDATED,
        expect=("refusal",),
        note=(
            "Sum-composition data cannot resolve individual chains, so the "
            "correct answer is to say so — not a number."
        ),
    ),
)


def by_id(case_id: str) -> EvalCase:
    for case in CASES:
        if case.id == case_id:
            return case
    raise KeyError(f"No eval case named {case_id!r}")
