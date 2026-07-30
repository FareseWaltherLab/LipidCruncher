"""Unit tests for the eval scorers.

These run in the normal suite (no API key, no network). A scorer that
silently passes everything is worse than no eval at all, so each one is
checked against both a good turn and the specific bad turn it exists to
catch — using the real failures observed in preview testing.
"""
import pandas as pd
import pytest

from tests.evals import cases as C
from tests.evals.scorers import score_routing, score_usability, score_case


TOP10 = C.by_id("top10_significant")
SATURATION = C.by_id("saturation_tg")
CHAIN = C.by_id("chain_length_distribution")
REFUSAL = C.by_id("chain_pools_consolidated")
DESCRIBE = C.by_id("describe_dataset")


class TestRouting:
    def test_passes_when_the_pipeline_is_used(self):
        code = "res = lc.compare_species('WT', 'KO')\nresult_df = res.head(10)"
        assert score_routing(code, TOP10) == []

    def test_catches_the_original_hand_rolled_statistics(self):
        """The real first failure: its own Welch loop and BH correction."""
        code = (
            "from scipy import stats\n"
            "for _, row in df.iterrows():\n"
            "    _, p = stats.ttest_ind(wt, ko, equal_var=False)\n"
            "adj = multipletests(pvals, method='fdr_bh')\n"
        )
        failures = score_routing(code, TOP10)
        assert any("did not call lc.compare_species" in f for f in failures)
        assert any("hand-rolled" in f for f in failures)

    def test_accepts_any_listed_alternative(self):
        """expect_lc is a set of acceptable routes, not a single required one.

        Uses a synthetic case: no real case currently lists alternatives,
        but the scorer supports it and the behaviour should stay pinned.
        """
        case = C.EvalCase(
            id="synthetic", question="q", dataset=C.CANONICAL,
            expect_lc=("lc.class_summary", "lc.bar_chart"), note="test only",
        )
        assert score_routing("x = lc.bar_chart()", case) == []
        assert score_routing("x = lc.class_summary()", case) == []
        assert score_routing("x = df.groupby('ClassKey').sum()", case) != []

    def test_catches_hand_parsed_chain_composition(self):
        """Parsing chain names re-invents the counting convention."""
        code = "chains = re.findall(r'(\\d+):(\\d+)', name)"
        assert score_routing(code, CHAIN) != []

    def test_no_expectation_means_no_routing_failure(self):
        assert score_routing("result_df = df.describe()", DESCRIBE) == []


class TestUsability:
    def test_passes_a_complete_answer(self):
        failures = score_usability(
            "result_df = lc.compare_species('WT','KO').head(10)",
            "Comparing WT vs KO with Welch's t-test on log10 concentrations, "
            "Benjamini-Hochberg corrected.",
            TOP10,
            result_df=pd.DataFrame({"a": [1]}),
        )
        assert failures == []

    def test_catches_the_stalled_introspection_turn(self):
        """The real second failure: printed lc.conditions() and stopped."""
        failures = score_usability(
            "conditions = lc.conditions()\nprint('Defined:', conditions)",
            "I need to check what conditions are defined.",
            TOP10,
        )
        assert any("inspection step" in f for f in failures)

    def test_introspection_is_fine_alongside_real_work(self):
        code = (
            "conds = lc.conditions()\n"
            "result_df = lc.compare_species(conds[0], conds[1])"
        )
        failures = score_usability(
            code, "Welch's t-test, BH corrected.", TOP10,
            result_df=pd.DataFrame({"a": [1]}),
        )
        assert not any("inspection step" in f for f in failures)

    def test_catches_a_printed_table(self):
        """The real third failure: printed instead of returned."""
        failures = score_usability(
            "print(lc.compare_species('WT','KO').head(10).to_string())",
            "Welch's t-test, BH corrected.\n```\n...table...\n```",
            TOP10,
            result_df=None,
        )
        assert any("returned table" in f for f in failures)

    def test_catches_missing_methodology(self):
        failures = score_usability(
            "result_df = lc.compare_species('WT','KO')",
            "Here are the top 10 lipids.",
            TOP10,
            result_df=pd.DataFrame({"a": [1]}),
        )
        assert any("without saying how" in f for f in failures)

    def test_catches_a_missing_figure(self):
        failures = score_usability(
            "d = lc.saturation(classes=['TG'])", "Here you go.",
            SATURATION, figure=None,
        )
        assert any("expected a figure" in f for f in failures)

    def test_catches_an_errored_turn(self):
        """The real fourth failure surfaced as an AttributeError."""
        failures = score_usability(
            "fig = lc.saturation(classes=['TG'])['TG'].figure", "",
            SATURATION,
            error="AttributeError: 'Figure' object has no attribute 'figure'",
        )
        assert any("errored" in f for f in failures)

    def test_catches_an_empty_turn(self):
        assert any(
            "no answer at all" in f
            for f in score_usability("", "", DESCRIBE)
        )


class TestHonesty:
    def test_accepts_an_explained_refusal(self):
        answer = (
            "This dataset uses sum composition, so individual fatty acid "
            "chains cannot be resolved — I can't give a C18 vs C16 total."
        )
        assert score_usability("", answer, REFUSAL) == []

    def test_rejects_a_confident_number(self):
        failures = score_usability(
            "result_df = totals", "C18 totals 4,182 and C16 totals 3,004.",
            REFUSAL, result_df=pd.DataFrame({"a": [1]}),
        )
        assert any("cannot answer this" in f for f in failures)


class TestScoreCase:
    def test_combines_both_scorers(self):
        failures = score_case(
            TOP10,
            code="from scipy import stats\np = stats.ttest_ind(a, b)",
            answer="Here are the results.",
        )
        assert any("lc.compare_species" in f for f in failures)   # routing
        assert any("without saying how" in f for f in failures)   # usability

    def test_clean_turn_scores_zero(self):
        assert score_case(
            TOP10,
            code="result_df = lc.compare_species('WT','KO').head(10)",
            answer="Welch's t-test on log10 values, BH-FDR corrected.",
            result_df=pd.DataFrame({"a": [1]}),
        ) == []


class TestCaseSet:
    def test_ids_are_unique(self):
        ids = [c.id for c in C.CASES]
        assert len(ids) == len(set(ids))

    def test_every_case_explains_why_it_exists(self):
        for case in C.CASES:
            assert case.note, f"{case.id} has no note"

    def test_by_id_rejects_unknown(self):
        with pytest.raises(KeyError):
            C.by_id("nope")


class TestRefusalIsNotJustAnyDecline:
    """A methodology question must not count as a refusal.

    The first eval run produced exactly this: on sum-composition data the
    model offered presence-vs-chain-split options instead of saying the
    chains are unrecoverable. Offering options implies the question is
    answerable once the user chooses — it isn't.
    """

    def test_methodology_question_is_not_a_refusal(self):
        answer = (
            "Before I compute this I need to clarify the methodology. "
            "I cannot proceed until you choose: presence-based, where "
            "PC 16:0_18:1 counts in both pools, or chain-split."
        )
        failures = score_usability("", answer, REFUSAL)
        assert any("cannot answer this" in f for f in failures)

    def test_naming_the_data_limitation_is_a_refusal(self):
        answer = (
            "These lipids are annotated at sum-composition level, e.g. "
            "PC(32:0), which gives only the total carbons across chains. "
            "The individual fatty acids cannot be resolved from this data, "
            "so a C18-vs-C16 total isn't recoverable. You'd need "
            "molecular species identifications."
        )
        assert score_usability("", answer, REFUSAL) == []
