"""Unit tests for the eval scorers.

These run in the normal suite (no API key, no network). A scorer that
silently passes everything is worse than no eval at all, so each one is
checked against both a good turn and the specific bad turn it exists to
catch — using the real failures observed in preview testing.
"""
from pathlib import Path

import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.services.ai_chat.data_transformer import DataTransformerService
from app.workflows.pipeline_api import PipelineAPI
from tests.evals import cases as C
from tests.evals import reference_analysis as ref
from tests.evals.scorers import (
    score_case,
    score_pipeline_parity,
    score_routing,
    score_transform,
    score_usability,
)


TOP10 = C.by_id("top10_significant")
SATURATION = C.by_id("saturation_tg")
CHAIN = C.by_id("chain_length_distribution")
REFUSAL = C.by_id("chain_pools_consolidated")
DESCRIBE = C.by_id("describe_dataset")
SIGNIFICANT = C.by_id("significant_changes")
TOP5 = C.by_id("top5_abundant")
COUNT = C.by_id("species_count")
FILTER = C.by_id("transform_filter_rt_quality")
AT_LEAST = C.by_id("transform_filter_at_least")
PLAIN = C.by_id("transform_plain")


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


class TestPipelineParity:
    """A question the main pipeline answers has one right answer. Each bad
    turn below routes to lc, so the routing scorer passes it."""

    @pytest.fixture(scope="class")
    def canonical(self):
        return ref.load_canonical()

    @pytest.fixture(scope="class")
    def lc(self, canonical):
        return PipelineAPI(canonical, ExperimentConfig(
            conditions_list=["WT", "KO"], number_of_samples_list=[3, 3],
        ))

    def test_the_pipelines_own_table_passes(self, canonical, lc):
        table = lc.compare_species("WT", "KO").head(10)
        assert score_pipeline_parity(TOP10, canonical, "", result_df=table) == []
        significant = lc.compare_species("WT", "KO").query("significant")
        assert score_pipeline_parity(
            SIGNIFICANT, canonical, "", result_df=significant,
        ) == []

    def test_lipid_names_as_the_index_pass(self, canonical, lc):
        table = lc.compare_species("WT", "KO").head(10).set_index("LipidMolec")
        assert score_pipeline_parity(TOP10, canonical, "", result_df=table) == []

    def test_a_non_default_correction_is_caught(self, canonical, lc):
        """BH-adjusted p-values are not what the app reports by default;
        the same six lipids are significant, so only the values differ."""
        table = lc.compare_species("WT", "KO", correction="fdr_bh")
        table = table.query("significant")[["LipidMolec", "adj_p_value"]]
        failures = score_pipeline_parity(SIGNIFICANT, canonical, "", result_df=table)
        assert any("no column matches the pipeline's values" in f for f in failures)

    def test_a_different_set_of_lipids_is_caught(self, canonical, lc):
        table = lc.compare_species("WT", "KO").head(5)
        failures = score_pipeline_parity(SIGNIFICANT, canonical, "", result_df=table)
        assert any("listed different rows" in f for f in failures)

    def test_ties_must_both_be_listed(self, canonical):
        top = ref.total_per_lipid(canonical).sort_values(ascending=False)
        good = top.head(5).rename("total").reset_index()
        assert score_pipeline_parity(TOP5, canonical, "", result_df=good) == []
        one_of_the_tie = top.iloc[1:6].rename("total").reset_index()
        assert score_pipeline_parity(TOP5, canonical, "", result_df=one_of_the_tie)

    def test_a_number_must_be_stated(self, canonical):
        assert score_pipeline_parity(
            COUNT, canonical, "Your dataset has 24 lipid species.",
        ) == []
        failures = score_pipeline_parity(
            COUNT, canonical, "Your dataset has 23 lipid species.",
        )
        assert failures == ["the pipeline's answer is 24; the turn never states it"]

    def test_p_values_formatted_as_text_pass(self, canonical, lc):
        """First real run (2026-10-21): the model's own code, which was
        right, failed because its p-values were strings."""
        top10 = lc.compare_species('WT', 'KO').head(10)[
            ['LipidMolec', 'ClassKey', 'log2FC', 'p_value', 'significant']].copy()
        top10['log2FC'] = top10['log2FC'].round(3)
        top10['p_value'] = top10['p_value'].apply(lambda x: f"{x:.3e}")
        top10 = top10.reset_index(drop=True)
        top10.index += 1
        assert score_pipeline_parity(TOP10, canonical, "", result_df=top10) == []

    def test_every_lipid_with_a_significant_flag_passes(self, canonical, lc):
        """Same run: all 24 lipids, the six marked significant."""
        result = lc.compare_species('WT', 'KO', alpha=0.05)
        table = result[['LipidMolec', 'ClassKey', 'log2FC', 'p_value',
                        'adj_p_value', 'significant']].copy()
        table['p_value'] = table['p_value'].apply(lambda x: f"{x:.3e}")
        assert score_pipeline_parity(SIGNIFICANT, canonical, "", result_df=table) == []

    def test_a_flag_marking_other_lipids_is_still_caught(self, canonical, lc):
        table = lc.compare_species('WT', 'KO', alpha=0.6)
        failures = score_pipeline_parity(SIGNIFICANT, canonical, "", result_df=table)
        assert any("listed different rows" in f for f in failures)

    def test_no_table_is_a_failure(self, canonical):
        assert score_pipeline_parity(TOP10, canonical, "PE 18:0_20:4 ...") == [
            "no table to compare with the pipeline's answer"
        ]


_SAMPLES = ['KO-1', 'KO-2', 'KO-3', 'KOWT-1', 'KOWT-2', 'KOWT-3',
            'mut1-1', 'mut1-2', 'mut1-3', 'mut2-1', 'mut2-2', 'mut2-3',
            'WT-1', 'WT-2', 'WT-3']


class TestTransform:
    """Graded by running real transform code on the 2026-10-20 fixture."""

    @pytest.fixture(scope="class")
    def studio(self):
        path = Path(__file__).parents[1] / "fixtures" / "ai_studio_test_dataset.csv"
        return pd.read_csv(path)

    def _run(self, code, studio):
        return DataTransformerService.execute_transformation(
            f"_SAMPLES = {_SAMPLES!r}\n{code}", studio,
        )

    def test_the_right_filter_passes(self, studio):
        t = self._run(
            "kept = df[(df['RetentionTime'] > 5) & (df['QualityScore'] > 70)]\n"
            "result_df = kept[['feature'] + _SAMPLES].reset_index(drop=True)\n",
            studio,
        )
        assert score_transform(FILTER, studio, t) == []

    def test_renamed_columns_and_a_classkey_column_pass(self, studio):
        t = self._run(
            "result_df = df[['feature', 'LipidClass'] + _SAMPLES].rename("
            "columns={'LipidClass': 'ClassKey', 'KO-1': 'S1'})\n",
            studio,
        )
        assert score_transform(PLAIN, studio, t) == []

    def test_greater_than_for_at_least_is_caught(self, studio):
        t = self._run(
            "kept = df[(df['RetentionTime'] > 5) & (df['QualityScore'] > 70)]\n"
            "result_df = kept[['feature'] + _SAMPLES].reset_index(drop=True)\n",
            studio,
        )
        (failure,) = score_transform(AT_LEAST, studio, t)
        assert failure.startswith("kept 757 lipids, expected 761")

    def test_values_moved_and_changed_are_caught(self, studio):
        """Passes every refusal in the app (A3 only warns)."""
        t = self._run(
            "kept = df[(df['RetentionTime'] > 5) & (df['QualityScore'] > 70)]\n"
            "result_df = pd.DataFrame({'feature': kept['feature'].values})\n"
            "for c in _SAMPLES:\n"
            "    result_df[c] = kept[c] / 2\n",
            studio,
        )
        assert t.success
        (failure,) = score_transform(FILTER, studio, t)
        assert failure.endswith("values differ from the upload")

    def test_a_refused_transform_fails(self, studio):
        t = self._run("result_df = df[['feature', 'RetentionTime'] + _SAMPLES]\n", studio)
        assert score_transform(PLAIN, studio, t) == [
            "the transform did not produce an accepted table"
        ]


def test_every_case_is_scored_against_the_pipeline_where_it_can_be():
    """Cases with a pipeline answer must run on a dataset that has one."""
    for case in C.CASES:
        if case.pipeline_answer or case.pipeline_number:
            assert case.dataset == C.CANONICAL, case.id
        if case.expect_table:
            assert case.tab == "transform" and case.dataset == C.STUDIO, case.id


def test_a_miscolored_direction_chart_fails_usability():
    from app.workflows.pipeline_api import PipelineAPI
    from tests.unit.test_ai_output_checks import MISCOLORED_BARS
    import plotly.graph_objects as go
    lc = PipelineAPI(ref.load_canonical(), ExperimentConfig(
        conditions_list=["WT", "KO"], number_of_samples_list=[3, 3]))
    scope = {"lc": lc, "go": go}
    exec(MISCOLORED_BARS, scope)
    case = C.by_id("direction_colored_bars")
    failures = score_usability(MISCOLORED_BARS, "", case, figure=scope["fig"])
    assert any("colors do not follow the direction" in f for f in failures)
