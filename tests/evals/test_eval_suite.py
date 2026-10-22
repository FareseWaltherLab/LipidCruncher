"""The eval suite: runs golden-set questions against the real AI.

Excluded from the normal test run — it costs API calls and is non-
deterministic. Run it deliberately, when prompts, the model, or the `lc`
catalog change:

    pytest tests/evals/test_eval_suite.py -m eval -v

    # one case
    pytest tests/evals/test_eval_suite.py -m eval -k top10_significant

Scoring lives in ``scorers.py`` and is unit-tested separately, so a failure
here means the *AI's answer* regressed, not the grader.
"""
import os
import re
import warnings

import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.workflows.ai_chat import AIChatWorkflow

from tests.ai_turns import finish
from tests.evals import cases as C
from tests.evals import reference_analysis as ref
from tests.evals.scorers import score_case

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="eval suite needs ANTHROPIC_API_KEY",
    ),
]

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def load_dataset(name: str):
    """Return (df, experiment) for a case's dataset."""
    if name == C.CANONICAL:
        return ref.load_canonical(), ExperimentConfig(
            conditions_list=["WT", "KO"], number_of_samples_list=[3, 3],
        )
    if name in (C.CHAIN_MOLECULAR, C.CHAIN_CONSOLIDATED):
        df = pd.read_csv(os.path.join(FIXTURES, f"{name}.csv"))
        # A single condition over all four samples. An experiment must
        # exist, otherwise `lc` would refuse for lack of one and the
        # refusal case would pass for entirely the wrong reason.
        return df, ExperimentConfig(
            conditions_list=["All"], number_of_samples_list=[4],
        )
    if name == C.STUDIO:
        # Raw, pre-Generic data: the Transform tab needs no experiment.
        return pd.read_csv(os.path.join(FIXTURES, "ai_studio_test_dataset.csv")), None
    raise KeyError(f"Unknown dataset {name!r}")


def run_case(case: C.EvalCase) -> dict:
    """Execute one case against the real AI and normalise what came back."""
    df, experiment = load_dataset(case.dataset)
    outcome = {"code": "", "answer": "", "result_df": None,
               "figure": None, "error": None, "transformation": None, "df": df,
               "unverified_numbers": ()}

    try:
        if case.tab == "transform":
            _, session = AIChatWorkflow.start_transform_chat(df)
            result = finish(AIChatWorkflow.stream_transform_message(
                user_message=case.question,
                session=session,
                source_df=df,
                # As confirmed in the sidebar: the 15 intensity columns.
                sample_columns=list(df.columns[6:]),
            ))
            outcome["answer"] = result.assistant_message
            t = result.transformation
            outcome["transformation"] = t
            if t is not None:
                outcome["code"] = t.code
                outcome["result_df"] = t.result_df
                if not t.success:
                    outcome["error"] = "; ".join(t.validation_errors)
        elif case.tab == "visualization":
            session = AIChatWorkflow.start_viz_session(df, experiment)
            result = finish(AIChatWorkflow.stream_viz_message(
                user_message=case.question,
                session=session,
                analysis_df=df,
                experiment=experiment,
            ))
            outcome["answer"] = result.assistant_message
            outcome["unverified_numbers"] = result.unverified_numbers
            last = result.session.messages[-1]
            outcome["code"] = last.code or ""
            if result.sandbox_result is not None:
                outcome["figure"] = result.sandbox_result.figure
                if not result.sandbox_result.success:
                    outcome["error"] = result.sandbox_result.error_message
        else:
            session = AIChatWorkflow.start_data_questions_session(df)
            result = finish(AIChatWorkflow.stream_data_questions_message(
                user_message=case.question,
                session=session,
                source_df=df,
                experiment=experiment,
            ))
            outcome["answer"] = result.assistant_message
            outcome["code"] = result.code or ""
            outcome["result_df"] = result.result_df
            outcome["unverified_numbers"] = result.unverified_numbers
    except Exception as e:  # a crash is a graded failure, not a test error
        outcome["error"] = f"{type(e).__name__}: {e}"

    return outcome


@pytest.mark.parametrize("case", C.CASES, ids=lambda c: c.id)
def test_eval_case(case):
    outcome = run_case(case)
    if outcome["unverified_numbers"]:
        # Measured, not graded (plan B2): shows in the warnings summary,
        # with each number's sentence so it can be judged.
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s|\n", outcome["answer"])
            if any(n in s for n in outcome["unverified_numbers"])
        ]
        warnings.warn(
            f"{case.id}: numbers the answer states that the code never "
            f"produced: {', '.join(outcome['unverified_numbers'])} — "
            + " | ".join(sentences)[:600]
        )
    failures = score_case(
        case,
        code=outcome["code"],
        answer=outcome["answer"],
        result_df=outcome["result_df"],
        figure=outcome["figure"],
        error=outcome["error"],
        df=outcome["df"],
        transformation=outcome["transformation"],
    )
    if failures:
        pytest.fail(
            f"\n{case.id} — {case.note}\n"
            f"\nQuestion: {case.question}\n"
            + "\nFailures:\n" + "\n".join(f"  - {f}" for f in failures)
            + f"\n\nCode:\n{outcome['code'][:1500]}"
            + f"\n\nAnswer:\n{outcome['answer'][:800]}\n"
        )
