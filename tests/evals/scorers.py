"""Deterministic scorers for the AI Studio eval.

Pure functions over what a turn produced — no API calls, no Streamlit — so
they are unit-testable on their own and cheap to run.

Scorers, in the order that matters:

* **routing** — did the question go to LipidCruncher's own method? If it
  did, the numbers are right by construction, which is why this is the
  first and highest-value check rather than a numeric comparison.
* **usability** — did the turn actually produce a complete, readable,
  self-explanatory answer? Every failure observed in live testing was of
  this kind, and none would have been caught by checking numbers.
* **pipeline parity** — for a question the main pipeline answers, is the
  answer the pipeline's? Routing makes that likely, but a non-default
  argument or a later filter changes the numbers that reach the user.
* **transform** — did a Transform to Generic request give exactly the
  rows it asked for, with every value still on its own lipid?

Each returns a list of human-readable failure strings; empty means pass.
"""
import math
import re
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from app.services.ai_chat.output_checks import check_direction_colors

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

    for call in case.expect_lc_all:
        if call not in code:
            failures.append(f"did not call {call}")

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

    if figure is not None:
        failures += check_direction_colors(figure)

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


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def score_pipeline_parity(
    case, df: pd.DataFrame, answer: str, *, result_df: Any = None,
) -> List[str]:
    """Check the turn gave the main pipeline's answer, not a different one."""
    failures = []

    if case.pipeline_number is not None:
        expected = float(case.pipeline_number(df))
        stated = [float(n.replace(",", "")) for n in _NUMBER.findall(answer or "")]
        if result_df is not None:
            stated += [
                float(v) for v in np.ravel(result_df.select_dtypes("number"))
                if pd.notna(v)
            ]
        if not any(math.isclose(n, expected, rel_tol=0.01) for n in stated):
            failures.append(
                f"the pipeline's answer is {expected:g}; the turn never states it"
            )

    if case.pipeline_answer is None:
        return failures
    expected = case.pipeline_answer(df)
    want = set(map(str, expected.index))
    if result_df is None:
        return failures + ["no table to compare with the pipeline's answer"]

    table = (
        result_df if isinstance(result_df.index, pd.RangeIndex)
        else result_df.reset_index()
    )
    key = max(
        (c for c in table.columns if table[c].dtype == object),
        key=lambda c: table[c].astype(str).isin(want).sum(),
        default=None,
    )
    if key is None or not table[key].astype(str).isin(want).any():
        example = next(iter(want))
        return failures + [f"no column lists the pipeline's keys (e.g. {example!r})"]

    got = set(table[key].astype(str))
    # A table of every lipid with a yes/no column marking exactly the
    # pipeline's answer gives that answer too.
    for flag in table.select_dtypes(bool).columns:
        marked = set(table.loc[table[flag], key].astype(str))
        if got > want and marked == want:
            table = table[table[flag]]
            got = marked
            break
    if got != want:
        missing, extra = sorted(want - got), sorted(got - want)
        return failures + [
            f"listed different rows from the pipeline: missing {missing[:5]}, "
            f"extra {extra[:5]}"
        ]

    rows = table.assign(_key=table[key].astype(str)).drop_duplicates("_key")
    rows = rows.set_index("_key").loc[[str(k) for k in expected.index]]
    target = expected.to_numpy(dtype=float)
    for col in rows.columns:
        # Numbers formatted as text ("1.069e-06") count as numbers.
        values = pd.to_numeric(rows[col], errors="coerce")
        if values.notna().all() and np.allclose(values, target, rtol=0.01):
            return failures
    first = expected.index[0]
    return failures + [
        f"no column matches the pipeline's values (e.g. {first!r}: "
        f"pipeline {expected.iloc[0]:.3g})"
    ]


def score_transform(case, df: pd.DataFrame, transformation: Any) -> List[str]:
    """Check a transform kept exactly the expected lipids, with values
    unchanged and each on its own lipid (compared by position, so renamed
    sample columns are fine)."""
    if case.expect_table is None:
        return []
    if transformation is None or not transformation.success:
        return ["the transform did not produce an accepted table"]
    expected = case.expect_table(df)
    result = transformation.result_df
    names = result.iloc[:, 0].astype(str)
    want = expected.iloc[:, 0].astype(str)

    if names.duplicated().any():
        return [f"{names.duplicated().sum()} lipid names appear more than once"]
    if set(names) != set(want):
        missing, extra = sorted(set(want) - set(names)), sorted(set(names) - set(want))
        return [
            f"kept {len(names)} lipids, expected {len(want)}: "
            f"missing {missing[:5]}, extra {extra[:5]}"
        ]
    samples = list(transformation.report.sample_columns)
    if len(samples) != expected.shape[1] - 1:
        return [
            f"{len(samples)} sample columns, expected {expected.shape[1] - 1} "
            f"({samples[:5]})"
        ]
    got = result[samples].set_index(names.values).loc[want.values]
    differ = ~np.isclose(
        got.to_numpy(dtype=float), expected.iloc[:, 1:].to_numpy(dtype=float),
        rtol=1e-9, equal_nan=True,
    )
    if differ.any():
        return [f"{int(differ.sum())} values differ from the upload"]
    return []


def score_case(
    case,
    code: str,
    answer: str,
    *,
    result_df: Any = None,
    figure: Any = None,
    error: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    transformation: Any = None,
) -> List[str]:
    """Run every scorer for one case. Empty list means the case passed."""
    failures = (
        score_routing(code, case)
        + score_usability(
            code, answer, case,
            result_df=result_df, figure=figure, error=error,
        )
    )
    if df is not None:
        failures += score_pipeline_parity(case, df, answer, result_df=result_df)
        failures += score_transform(case, df, transformation)
    return failures
