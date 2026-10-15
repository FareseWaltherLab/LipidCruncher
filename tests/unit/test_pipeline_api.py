"""Unit tests for the ``lc`` pipeline namespace exposed to sandbox code.

The load-bearing test is ``TestAgreesWithReferenceOracle``: whatever the AI
gets from ``lc.compare_species`` must equal what the eval's independent
reference implementation computes. That is what makes "defer to the
pipeline" mean the same numbers the main app reports.
"""
import pytest

from app.models.experiment import ExperimentConfig
from app.workflows.pipeline_api import PipelineAPI

from tests.evals import reference_analysis as ref


PLANTED_UP = {"PC 16:0_18:1", "TG 16:0_18:1_18:1", "PE 18:0_20:4"}
PLANTED_DOWN = {"PC 18:0_18:1", "SM d18:1_16:0", "Cer d18:1_24:0"}
PLANTED = PLANTED_UP | PLANTED_DOWN


@pytest.fixture
def df():
    return ref.load_canonical()


@pytest.fixture
def experiment():
    # Canonical dataset: s1-s3 control, s4-s6 KO.
    return ExperimentConfig(
        conditions_list=["WT", "KO"],
        number_of_samples_list=[3, 3],
    )


@pytest.fixture
def lc(df, experiment):
    return PipelineAPI(df, experiment)


class TestIntrospection:
    def test_conditions(self, lc):
        assert lc.conditions() == ["WT", "KO"]

    def test_samples_for(self, lc):
        assert lc.samples_for("WT") == ["s1", "s2", "s3"]
        assert lc.samples_for("KO") == ["s4", "s5", "s6"]

    def test_unknown_condition_lists_valid_names(self, lc):
        with pytest.raises(ValueError, match="Unknown condition"):
            lc.samples_for("nope")

    def test_stats_without_experiment_explain_the_fix(self, df):
        bare = PipelineAPI(df, experiment=None)
        with pytest.raises(ValueError, match="No experiment is defined"):
            bare.compare_species("WT", "KO")


class TestCompareSpecies:
    def test_returns_expected_columns(self, lc):
        out = lc.compare_species("WT", "KO")
        assert list(out.columns) == [
            "LipidMolec", "ClassKey", "p_value",
            "adj_p_value", "log2FC", "significant",
        ]

    def test_sorted_by_adjusted_p_ascending(self, lc):
        out = lc.compare_species("WT", "KO")
        assert out["adj_p_value"].is_monotonic_increasing

    def test_finds_exactly_the_planted_lipids(self, lc):
        out = lc.compare_species("WT", "KO")
        significant = set(out.loc[out["significant"], "LipidMolec"])
        assert significant == PLANTED

    def test_direction_of_planted_effects(self, lc):
        out = lc.compare_species("WT", "KO").set_index("LipidMolec")
        for lipid in PLANTED_UP:
            assert out.loc[lipid, "log2FC"] > 0
        for lipid in PLANTED_DOWN:
            assert out.loc[lipid, "log2FC"] < 0

    def test_class_key_is_populated(self, lc):
        out = lc.compare_species("WT", "KO")
        assert out["ClassKey"].notna().all()


class TestAgreesWithReferenceOracle:
    """lc.* must equal the eval's independent reference implementation."""

    def test_same_lipids_tested(self, lc, df):
        out = lc.compare_species("WT", "KO")
        assert set(out["LipidMolec"]) == set(ref.species_stats(df).index)

    def test_same_pvalues_and_fold_changes(self, lc, df):
        out = lc.compare_species("WT", "KO").set_index("LipidMolec")
        expected = ref.species_stats(df)
        for lipid in expected.index:
            assert out.loc[lipid, "p_value"] == pytest.approx(
                expected.loc[lipid, "p_value"]
            )
            assert out.loc[lipid, "adj_p_value"] == pytest.approx(
                expected.loc[lipid, "adj_p"]
            )
            assert out.loc[lipid, "log2FC"] == pytest.approx(
                expected.loc[lipid, "log2fc"]
            )

    def test_same_top_10_by_significance(self, lc, df):
        out = lc.compare_species("WT", "KO")
        assert (
            out["LipidMolec"].head(10).tolist()
            == ref.top_n_by_significance(df, 10).index.tolist()
        )


class TestIntensityColumnData:
    """Datasets normalized to intensity[*] must work identically."""

    def test_intensity_columns_are_accepted(self, df, experiment):
        renamed = df.rename(columns={
            c: c.replace("concentration[", "intensity[")
            for c in df.columns if c.startswith("concentration[")
        })
        out = PipelineAPI(renamed, experiment).compare_species("WT", "KO")
        significant = set(out.loc[out["significant"], "LipidMolec"])
        assert significant == PLANTED


class TestClassSummary:
    def test_counts_match_the_data(self, lc, df):
        out = lc.class_summary()
        expected = df.groupby("ClassKey").size().to_dict()
        got = dict(zip(out["ClassKey"], out["n_species"]))
        assert got == expected

    def test_sorted_by_species_count_desc(self, lc):
        out = lc.class_summary()
        assert out["n_species"].is_monotonic_decreasing

    def test_has_mean_abundance(self, lc):
        assert "mean_abundance" in lc.class_summary().columns


class TestCompareClasses:
    def test_returns_expected_columns(self, lc):
        out = lc.compare_classes()
        assert list(out.columns) == [
            "ClassKey", "p_value", "adj_p_value", "significant",
        ]

    def test_covers_the_datasets_classes(self, lc, df):
        out = lc.compare_classes()
        assert set(out["ClassKey"]) <= set(df["ClassKey"].unique())
        assert not out.empty


class TestSandboxIntegration:
    """`lc` must be reachable from AI-generated code, and the widened
    import allowlist must admit exactly the intended libraries."""

    def _run(self, code, df, experiment=None):
        from app.services.ai_chat.sandbox import SandboxService
        return SandboxService.execute(
            code, df, experiment=experiment,
            lc=PipelineAPI(df, experiment),
        )

    def test_lc_is_available_to_generated_code(self, df, experiment):
        res = self._run(
            "result_df = lc.compare_species('WT', 'KO')", df, experiment
        )
        assert res.success, res.error_message
        assert set(res.result_df.loc[res.result_df["significant"],
                                     "LipidMolec"]) == PLANTED

    def test_lc_without_experiment_reports_clearly(self, df):
        res = self._run("result_df = lc.compare_species('WT', 'KO')", df)
        assert not res.success
        assert "No experiment is defined" in res.error_message

    def test_class_summary_from_generated_code(self, df, experiment):
        res = self._run("result_df = lc.class_summary()", df, experiment)
        assert res.success, res.error_message
        assert "n_species" in res.result_df.columns

    @pytest.mark.parametrize("module", ["scipy", "statsmodels", "sklearn"])
    def test_fallback_libraries_are_importable(self, module, df):
        res = self._run(f"import {module}\nprint('ok')", df)
        assert res.success, res.error_message

    @pytest.mark.parametrize("module", ["os", "sys", "subprocess", "socket",
                                        "requests", "pathlib", "shutil"])
    def test_dangerous_modules_still_blocked(self, module, df):
        res = self._run(f"import {module}", df)
        assert not res.success
        assert "not allowed" in res.error_message

    def test_scipy_stats_actually_usable(self, df):
        res = self._run(
            "from scipy import stats\n"
            "print(round(float(stats.ttest_ind([1,2,3],[4,5,6]).pvalue), 5))",
            df,
        )
        assert res.success, res.error_message
        assert res.stdout.strip() == "0.02131"


class TestPromptsAdvertisePipelineFirst:
    def test_all_code_generating_prompts_carry_the_rule(self):
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        from app.services.ai_chat.viz_generator import _viz_system_prompt
        for prompt in (data_questions_system_prompt(), _viz_system_prompt()):
            assert "Pipeline first" in prompt
            assert "lc.compare_species" in prompt


class TestExperimentContextReachesTheModel:
    """The model must be told the condition names, not have to ask.

    A turn ends when its code runs and the model never sees that output, so
    a turn spent printing `lc.conditions()` produces no answer. Regression
    test for exactly that failure in the Data Questions tab.
    """

    def test_context_lists_conditions_and_samples(self, experiment):
        from app.services.ai_chat.prompt_templates import (
            build_experiment_context,
        )
        ctx = build_experiment_context(experiment)
        assert "WT" in ctx and "KO" in ctx
        assert "s1, s2, s3" in ctx
        assert "do not call lc.conditions()" in ctx

    def test_no_experiment_yields_empty_context(self):
        from app.services.ai_chat.prompt_templates import (
            build_experiment_context,
        )
        assert build_experiment_context(None) == ""

    def test_data_questions_turn_includes_the_experiment_block(
        self, df, experiment, monkeypatch
    ):
        """The block must actually reach the API call, not just exist."""
        from app.workflows.ai_chat import AIChatWorkflow
        from app.services.ai_chat import ai_client

        captured = {}

        def fake_tool_loop(messages, **kwargs):
            captured["messages"] = messages
            return "ok"

        monkeypatch.setattr(
            ai_client.AIClientService, "execute_tool_loop", fake_tool_loop
        )
        session = AIChatWorkflow.start_data_questions_session(df)
        AIChatWorkflow.handle_data_questions_message(
            user_message="top 10 most significant lipids WT vs KO",
            session=session,
            source_df=df,
            experiment=experiment,
        )
        sent = captured["messages"][-1]["content"]
        assert "<experiment>" in sent
        assert "WT" in sent and "KO" in sent

    def test_prompt_forbids_ending_a_turn_on_introspection(self):
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        prompt = data_questions_system_prompt()
        assert "never see its output" in prompt.lower()
        assert "Never end a turn on an inspection step" in prompt


class TestResultPresentation:
    """A correct answer the user cannot read is not a usable answer."""

    def test_printed_output_is_fenced_so_tables_keep_their_shape(
        self, df, experiment, monkeypatch
    ):
        """Printed DataFrames render as markdown, which collapses the
        column padding into one unreadable line unless fenced."""
        from app.workflows.ai_chat import AIChatWorkflow
        from app.services.ai_chat import ai_client

        response = (
            "Comparing WT vs KO.\n\n"
            "```python\n"
            "res = lc.compare_species('WT', 'KO')\n"
            "print(res.head(2).to_string())\n"
            "```"
        )
        monkeypatch.setattr(
            ai_client.AIClientService, "execute_tool_loop",
            lambda *a, **k: response,
        )
        session = AIChatWorkflow.start_data_questions_session(df)
        result = AIChatWorkflow.handle_data_questions_message(
            user_message="top lipids", session=session, source_df=df,
            experiment=experiment,
        )
        assert "```" in result.assistant_message
        # the table body must sit inside the fence
        body = result.assistant_message.split("```")[1]
        assert "LipidMolec" in body

    def test_prompt_requires_stating_the_method(self):
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        prompt = data_questions_system_prompt()
        assert "Always say what method produced the numbers" in prompt
        assert "Benjamini-Hochberg" in prompt

    def test_prompt_prefers_returned_tables_over_printed_ones(self):
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        assert "Return tables, don't print them" in (
            data_questions_system_prompt()
        )


class TestAnalysisPlotsUseTheAppsOwnCode:
    """Every plot helper must delegate to AnalysisWorkflow, so a figure the
    AI produces is the same figure the app's analysis pages produce."""

    def test_volcano(self, lc):
        res = lc.volcano("WT", "KO")
        assert res.figure is not None and len(res.figure.data) > 0
        assert res.data is not None

    def test_bar_chart(self, lc):
        res = lc.bar_chart()
        assert res.figure is not None
        assert not res.data.empty

    def test_bar_chart_log_scale(self, lc):
        assert lc.bar_chart(scale="log10").figure is not None

    def test_heatmap(self, lc):
        assert lc.heatmap().figure is not None

    def test_clustered_heatmap(self, lc):
        assert lc.heatmap(heatmap_type="clustered", n_clusters=2).figure

    def test_chain_length(self, lc):
        assert lc.chain_length().figure is not None

    def test_pathway(self, lc):
        assert lc.pathway("WT", "KO").figure is not None

    def test_pie_charts_keyed_by_condition(self, lc):
        out = lc.pie_charts()
        assert set(out) == {"WT", "KO"}
        assert all(r.figure is not None for r in out.values())

    def test_saturation_keyed_by_class(self, lc, df):
        out = lc.saturation()
        assert set(out) <= set(df["ClassKey"].unique())
        assert out

    def test_every_helper_exposes_figure_uniformly(self, lc):
        """One shape for all of them. An exception here cost a real turn:
        the model wrote `lc.saturation(...)['TG'].figure` — correct by
        analogy with every other helper — and hit an AttributeError."""
        singles = [lc.volcano("WT", "KO"), lc.bar_chart(), lc.heatmap(),
                   lc.chain_length(), lc.pathway("WT", "KO"),
                   lc.fatty_acid_composition("PC")]
        for res in singles:
            assert res.figure is not None

        for mapping in (lc.pie_charts(), lc.saturation()):
            for key, res in mapping.items():
                assert hasattr(res, "figure"), f"{key} is not an LCResult"
                assert res.figure is not None

    def test_saturation_percentage_mode(self, lc):
        out = lc.saturation(classes=["PC"], plot_type="percentage")
        assert out["PC"].figure is not None

    def test_fatty_acid_composition(self, lc):
        assert lc.fatty_acid_composition("PC").figure is not None

    def test_available_classes(self, lc, df):
        assert set(lc.available_classes()) == set(df["ClassKey"].unique())

    def test_class_and_condition_subsetting(self, lc):
        res = lc.bar_chart(conditions=["WT"], classes=["PC"])
        assert res.figure is not None

    def test_plots_need_an_experiment(self, df):
        bare = PipelineAPI(df, experiment=None)
        with pytest.raises(ValueError, match="No experiment is defined"):
            bare.volcano("WT", "KO")


class TestPlotsReachTheSandbox:
    def test_volcano_figure_crosses_the_process_boundary(
        self, df, experiment, lc
    ):
        from app.services.ai_chat.sandbox import SandboxService
        res = SandboxService.execute(
            "fig = lc.volcano('WT', 'KO').figure",
            df, experiment=experiment, lc=lc,
        )
        assert res.success, res.error_message
        assert res.figure is not None

    def test_prompt_lists_the_plot_catalog(self):
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        prompt = data_questions_system_prompt()
        for fn in ("lc.volcano", "lc.bar_chart", "lc.heatmap",
                   "lc.chain_length", "lc.pathway", "lc.saturation",
                   "lc.pie_charts", "lc.fatty_acid_composition"):
            assert fn in prompt, f"{fn} missing from prompt catalog"


class TestEveryPathSuppliesThePipeline:
    """Regression tests for two wiring bugs found during the layer move."""

    def test_missing_lc_explains_itself(self, df):
        """Binding None surfaced as \"'NoneType' object has no attribute
        'volcano'\", which told nobody anything."""
        from app.services.ai_chat.sandbox import SandboxService
        res = SandboxService.execute("fig = lc.volcano('a', 'b').figure", df)
        assert not res.success
        assert "lc.volcano is unavailable" in res.error_message
        assert "pipeline was not supplied" in res.error_message


class TestMainAppDefaultsAndDisclosure:
    """Regression (AQ5): `lc` ran species-level tests with Benjamini-Hochberg
    correction while the app's Volcano page defaults to none, so the AI's
    "significant" lipids disagreed with the app's volcano for the same data.
    `lc` now uses each page's defaults and records the choices it made."""

    def test_species_default_is_uncorrected_like_the_volcano_page(self, lc):
        out = lc.compare_species("WT", "KO")
        assert (out["adj_p_value"] == out["p_value"]).all()

    def test_volcano_default_is_uncorrected(self, lc):
        res = lc.volcano("WT", "KO")
        assert res.raw.stat_summary.test_info["correction"] == "uncorrected"

    def test_correction_can_be_requested(self, lc):
        out = lc.compare_species("WT", "KO", correction="fdr_bh")
        assert (out["adj_p_value"] >= out["p_value"]).all()
        assert (out["adj_p_value"] > out["p_value"]).any()

    def test_class_level_uses_auto_mode_like_the_bar_chart_page(
        self, lc, monkeypatch,
    ):
        from app.services.statistical_testing import StatisticalTestingService

        seen = []
        real = StatisticalTestingService.run_class_level_tests

        def spy(*args, **kwargs):
            seen.append(kwargs["config"])
            return real(*args, **kwargs)

        monkeypatch.setattr(
            StatisticalTestingService, "run_class_level_tests", spy,
        )
        lc.compare_classes(alpha=0.01)
        assert seen[0].mode == "auto"
        assert seen[0].alpha == 0.01

    def test_each_analysis_records_its_choices(self, lc):
        lc.compare_species("WT", "KO")
        lc.compare_species("WT", "KO")  # recorded once
        lc.compare_species("WT", "KO", correction="fdr_bh")
        lc.compare_classes()
        default, changed, auto = lc.methods_used
        assert "no multiple-testing correction" in default
        assert "main app's Volcano defaults" in default
        assert "Benjamini-Hochberg" in changed
        assert "Changed on request" in changed
        assert "Auto mode" in auto

    def test_descriptive_calls_record_nothing(self, lc):
        lc.class_summary()
        lc.pie_charts()
        assert lc.methods_used == []


class TestDuplicateLipidsAreRefused:
    """Regression (AQ3): results were keyed by lipid name, so a second row
    for the same LipidMolec silently replaced the first (a 4x effect came
    back as p=NaN, log2FC=0) and class totals and counts double-counted it.
    Raw imports have such rows (every LipidSearch candidate)."""

    @pytest.fixture
    def lc_dupes(self, df, experiment):
        import pandas as pd
        return PipelineAPI(
            pd.concat([df, df.iloc[[0]]], ignore_index=True), experiment,
        )

    @pytest.mark.parametrize("call", [
        lambda lc: lc.compare_species("WT", "KO"),
        lambda lc: lc.compare_classes(),
        lambda lc: lc.class_summary(),
        lambda lc: lc.volcano("WT", "KO"),
        lambda lc: lc.bar_chart(),
        lambda lc: lc.pie_charts(),
    ])
    def test_analyses_refuse_duplicate_lipids(self, lc_dupes, df, call):
        with pytest.raises(ValueError, match="more than one row") as err:
            call(lc_dupes)
        assert df.iloc[0]["LipidMolec"] in str(err.value)

    def test_listing_classes_still_works(self, lc_dupes):
        assert lc_dupes.available_classes()
