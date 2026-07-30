"""Deterministic scorers for the AI Studio eval.

Pure functions over what a turn produced — no API calls, no Streamlit — so
they are unit-testable on their own and cheap to run.

Two scorers, in the order that matters:

* **routing** — did the question go to LipidCruncher's own method? If it
  did, the numbers are right by construction, which is why this is the
  first and highest-value check rather than a numeric comparison.
* **usability** — did the turn actually produce a complete, readable,
  self-explanatory answer? Every failure observed in live testing was of
  this kind, and none would have been caught by checking numbers.

Each returns a list of human-readable failure strings; empty means pass.
"""
from typing import Any, List, Optional

# Phrases that indicate the answer explains how it was computed. Any one
# is enough — we are checking that a method was stated, not policing wording.
_METHOD_MARKERS = (
    "welch", "t-test", "ttest", "benjamini", "hochberg", "fdr",
    "log10", "log2", "false discovery",
)

# A refusal has to decline *and* pin the blame on the data. Checking only
# for "cannot" would pass a methodology question ("I cannot proceed until
# you choose a convention"), which is the opposite behaviour: it implies
# the question is answerable once the user picks.
_DECLINE_MARKERS = (
    "cannot", "can't", "unable", "not possible", "isn't possible",
    "does not allow", "doesn't allow", "no way to", "not recoverable",
    "not resolvable",
)
_DATA_LIMIT_MARKERS = (
    "sum composition", "sum-composition", "individual chain",
    "individual fatty acid", "molecular species", "not resolvable",
    "not recoverable", "cannot be determined", "cannot be resolved",
    "does not identify", "doesn't identify", "only the total",
    "total carbon",
)


def score_routing(code: str, case) -> List[str]:
    """Check the turn used the pipeline instead of re-deriving it."""
    failures = []
    code = code or ""

    if case.expect_lc:
        # Alternatives: any one of the listed helpers is acceptable.
        if not any(call in code for call in case.expect_lc):
            expected = " or ".join(case.expect_lc)
            failures.append(
                f"did not call {expected} (code was {len(code)} chars)"
            )

    for pattern in case.forbid:
        if pattern in code:
            failures.append(
                f"hand-rolled analysis: found {pattern!r}, which the "
                f"pipeline already implements"
            )
    return failures


def score_usability(
    code: str,
    answer: str,
    case,
    *,
    result_df: Any = None,
    figure: Any = None,
    error: Optional[str] = None,
) -> List[str]:
    """Check the turn produced an answer a researcher can actually use."""
    failures = []
    code = code or ""
    answer = answer or ""

    if error:
        failures.append(f"turn errored: {error[:200]}")

    produced_anything = bool(
        answer.strip() or result_df is not None or figure is not None
    )
    if not produced_anything:
        failures.append("turn produced no answer at all")

    if _is_introspection_only(code):
        failures.append(
            "turn ended on an inspection step (the model never sees its "
            "own output, so this wastes the turn)"
        )

    if "table" in case.expect and result_df is None:
        failures.append(
            "expected a returned table (result_df); printed tables render "
            "as one unreadable line"
        )

    if "figure" in case.expect and figure is None:
        failures.append("expected a figure, none was produced")

    if "method_stated" in case.expect and not _states_method(answer):
        failures.append(
            "reported statistics without saying how they were computed"
        )

    if "refusal" in case.expect and not _is_refusal(answer):
        failures.append(
            "should have explained that the data cannot answer this, "
            "but gave an answer anyway"
        )

    return failures


def _is_introspection_only(code: str) -> bool:
    """True if the code only inspects and never computes an answer.

    The failure this catches: the model calls ``lc.conditions()``, prints
    it, and ends the turn — leaving the user with a fragment.
    """
    if not code.strip():
        return False
    meaningful = [
        line.strip() for line in code.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not meaningful:
        return False
    introspection = ("lc.conditions(", "lc.samples_for(",
                     "lc.available_classes(", ".columns", ".dtypes")
    touches_introspection = any(
        any(probe in line for probe in introspection) for line in meaningful
    )
    if not touches_introspection:
        return False
    # It is only a wasted turn if nothing else of substance happened.
    produces = ("result_df", "fig", "lc.compare", "lc.volcano", "lc.bar_chart",
                "lc.heatmap", "lc.saturation", "lc.chain_length",
                "lc.pathway", "lc.pie_charts", "lc.class_summary",
                "lc.fatty_acid_composition")
    return not any(
        any(token in line for token in produces) for line in meaningful
    )


def _states_method(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _METHOD_MARKERS)


def _is_refusal(answer: str) -> bool:
    """True only if the answer declines *and* names the data as the reason."""
    lowered = answer.lower()
    declines = any(m in lowered for m in _DECLINE_MARKERS)
    blames_data = any(m in lowered for m in _DATA_LIMIT_MARKERS)
    return declines and blames_data


def score_case(
    case,
    code: str,
    answer: str,
    *,
    result_df: Any = None,
    figure: Any = None,
    error: Optional[str] = None,
) -> List[str]:
    """Run every scorer for one case. Empty list means the case passed."""
    return (
        score_routing(code, case)
        + score_usability(
            code, answer, case,
            result_df=result_df, figure=figure, error=error,
        )
    )
