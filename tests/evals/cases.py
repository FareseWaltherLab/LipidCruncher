"""The golden set: questions with checkable expectations.

LipidCruncher is the source of truth, so if the AI routes a question to
`lc.*` the numbers are right by construction — which makes "did it route
correctly, and is the answer usable?" the first thing worth asserting.
Where the main pipeline answers the question, a case also carries that
answer (`pipeline_answer` / `pipeline_number`, from reference_analysis.py)
and the turn must reproduce it: a non-default argument or a later filter
changes the numbers even when the routing is right.

Every case records **why it exists**. Cases mined from real failures keep
their history, since a fixed bug is the cheapest regression test we have.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd

from tests.evals import reference_analysis as ref


# Datasets a case can run against (see tests/fixtures/).
CANONICAL = "canonical"          # 24 species, WT vs KO, 6 planted effects
CHAIN_MOLECULAR = "chain_molecular"        # answerable chain question
CHAIN_CONSOLIDATED = "chain_consolidated"  # must refuse: sum composition
STUDIO = "studio"                # ai_studio_test_dataset.csv: raw, with metadata


@dataclass(frozen=True)
class EvalCase:
    """One graded question.

    Attributes:
        id: Stable identifier, used in test output.
        question: Verbatim user prompt.
        dataset: Fixture to load.
        tab: 'data_questions', 'visualization' or 'transform'.
        expect_lc: Substrings that must appear in the generated code —
            the pipeline calls this question should route to.
        forbid: Substrings that must NOT appear — hand-rolled analysis the
            pipeline already implements.
        expect: Usability requirements: 'table', 'figure', 'method_stated',
            'refusal'.
        note: Why this case is in the set.
        pipeline_answer: What the main pipeline answers, as a Series of
            values indexed by lipid (or class). The turn's table must list
            exactly these keys, with a column holding these values.
        pipeline_number: The main pipeline's single-number answer; the
            turn must state it.
        expect_table: For transforms, the Generic table the request
            should give (names, then samples) — computed from the upload.
    """
    id: str
    question: str
    dataset: str
    tab: str = "data_questions"
    expect_lc: Tuple[str, ...] = ()
    forbid: Tuple[str, ...] = ()
    expect: Tuple[str, ...] = ()
    note: str = ""
    pipeline_answer: Optional[Callable[[pd.DataFrame], pd.Series]] = None
    pipeline_number: Optional[Callable[[pd.DataFrame], float]] = None
    expect_table: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None


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
        pipeline_answer=lambda df: ref.top_n_by_significance(df, 10)["p_value"],
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

    # ---- Parity: the answer must be the main pipeline's answer ----------
    # A question the pipeline can answer has one right answer: the one the
    # app gives (reference_analysis.py, which calls the app's own services).
    # Routing to lc makes that likely; these check the numbers that reach
    # the user, which a non-default argument or a later filter can change.
    EvalCase(
        id="significant_changes",
        question="Which lipids change significantly between WT and KO?",
        dataset=CANONICAL,
        expect_lc=("lc.compare_species", "lc.volcano"),
        forbid=HAND_ROLLED_STATS,
        expect=("table", "method_stated"),
        pipeline_answer=lambda df: ref.significant_lipids(df)["p_value"],
        note="Exactly the six planted lipids, with the Volcano's p-values.",
    ),
    EvalCase(
        id="class_totals",
        question=(
            "What is the total abundance of each lipid class, summed over "
            "all samples?"
        ),
        dataset=CANONICAL,
        expect=("table",),
        pipeline_answer=ref.per_class_totals,
        note="Descriptive; hand-checkable in CANONICAL_ANALYSIS.md.",
    ),
    EvalCase(
        id="top5_abundant",
        question=(
            "What are the 5 most abundant lipids, by total abundance over "
            "all samples?"
        ),
        dataset=CANONICAL,
        expect=("table",),
        pipeline_answer=lambda df: ref.top_n_by_abundance(df, 5),
        note="The top two tie at 1545; both must be listed.",
    ),
    EvalCase(
        id="species_count",
        question="How many lipid species are in my dataset?",
        dataset=CANONICAL,
        pipeline_number=ref.n_species,
        note="The simplest parity check: 24.",
    ),

    # ---- Transform to Generic: mined from real failures (2026-10-20) ---
    EvalCase(
        id="transform_filter_rt_quality",
        question=(
            "filter the dataset, keep species with retention time above 5 "
            "minutes and quality score above 70, then transform to generic "
            "format"
        ),
        dataset=STUDIO,
        tab="transform",
        expect_table=lambda df: _studio_generic(
            df[(df["RetentionTime"] > 5) & (df["QualityScore"] > 70)]
        ),
        note=(
            "The original incident: index-aligned columns gave 351 of 757 "
            "lipids another lipid's values. Must keep 757, values untouched."
        ),
    ),
    EvalCase(
        id="transform_filter_at_least",
        question=(
            "keep only species with a retention time of at least 5 minutes "
            "and a quality score of at least 70, then transform to generic "
            "format"
        ),
        dataset=STUDIO,
        tab="transform",
        expect_table=lambda df: _studio_generic(
            df[(df["RetentionTime"] >= 5) & (df["QualityScore"] >= 70)]
        ),
        note=(
            "'At least' means >=: 761 lipids, four more than '>' (five "
            "have a quality score of exactly 70). No check can catch the "
            "wrong operator; only the result can."
        ),
    ),
    EvalCase(
        id="transform_plain",
        question="Transform my data to the generic format.",
        dataset=STUDIO,
        tab="transform",
        expect_table=lambda df: _studio_generic(df),
        note=(
            "RetentionTime, PrecursorMZ and QualityScore are numeric, so "
            "keeping one passed the schema check and it was analysed as a "
            "sample. All 1,394 lipids, 15 samples, values untouched."
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


def _studio_generic(rows: pd.DataFrame) -> pd.DataFrame:
    """The studio fixture's rows as Generic: names, then the 15 samples
    (everything after the metadata columns)."""
    return rows[["feature"] + list(rows.columns[6:])].reset_index(drop=True)


def by_id(case_id: str) -> EvalCase:
    for case in CASES:
        if case.id == case_id:
            return case
    raise KeyError(f"No eval case named {case_id!r}")
