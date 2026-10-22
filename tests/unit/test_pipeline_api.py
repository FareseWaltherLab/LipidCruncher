"""Unit tests for the ``lc`` pipeline namespace exposed to sandbox code.

The load-bearing test is ``TestAgreesWithReferenceOracle``: whatever the AI
gets from ``lc.compare_species`` must equal what the eval's independent
reference implementation computes. That is what makes "defer to the
pipeline" mean the same numbers the main app reports.
"""
import numpy as np
import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.services.ai_chat.sandbox import ANALYSIS_MODULES
from app.workflows.pipeline_api import PipelineAPI

from tests.ai_turns import canned_stream, finish
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
            code, df, lc=PipelineAPI(df, experiment),
        )

    def test_lc_is_available_to_generated_code(self, df, experiment):
        res = self._run(
            "result_df = lc.compare_species('WT', 'KO')", df, experiment
        )
        assert res.success, res.error_message
        assert set(res.result_df.loc[res.result_df["significant"],
                                     "LipidMolec"]) == PLANTED

    def test_colored_bars_from_generated_code(self, df, experiment):
        res = self._run(
            "top = lc.compare_species('WT', 'KO').head(10)\n"
            "top['Direction'] = np.where(top['log2FC'] > 0, 'Up', 'Down')\n"
            "top = top.sort_values('log2FC')\n"
            "fig = lc.colored_bars(top, x='log2FC', y='LipidMolec', "
            "color_by='Direction', orientation='h').figure",
            df, experiment,
        )
        assert res.success, res.error_message
        assert res.figure is not None

    def test_lc_without_experiment_reports_clearly(self, df):
        res = self._run("result_df = lc.compare_species('WT', 'KO')", df)
        assert not res.success
        assert "No experiment is defined" in res.error_message

    def test_class_summary_from_generated_code(self, df, experiment):
        res = self._run("result_df = lc.class_summary()", df, experiment)
        assert res.success, res.error_message
        assert "n_species" in res.result_df.columns

    @pytest.mark.parametrize("module", ANALYSIS_MODULES)
    def test_every_advertised_analysis_module_imports(self, module, df):
        res = self._run(f"import {module}\nprint('ok')", df)
        assert res.success, res.error_message

    def test_usual_import_forms_work(self, df):
        res = self._run(
            "from scipy.stats import ttest_ind\n"
            "from sklearn.decomposition import PCA\n"
            "from statsmodels.stats.multitest import multipletests\n"
            "import statsmodels.formula.api as smf\n"
            "from scipy.cluster import hierarchy\n"
            "d = pd.DataFrame({'y': [1.0, 2.1, 2.9, 4.2], 'x': [1, 2, 3, 4]})\n"
            "print(round(smf.ols('y ~ x', data=d).fit().params['x'], 2))",
            df,
        )
        assert res.success, res.error_message
        assert res.stdout.strip() == "1.04"

    @pytest.mark.parametrize("code", [
        "import scipy", "import statsmodels", "import sklearn",
        "from scipy import io", "import scipy.io", "import sklearn.datasets",
        "from sklearn import datasets", "import statsmodels.api",
        "from statsmodels import api", "from scipy import stats, io",
        "from scipy import *",
    ])
    def test_whole_packages_and_unlisted_submodules_are_refused(self, code, df):
        """S10: the allowlist matched only the top-level package, so every
        submodule (scipy.io, sklearn.datasets, statsmodels.api.load) came
        with it."""
        res = self._run(code, df)
        assert not res.success
        assert "not allowed" in res.error_message

    @pytest.mark.parametrize("name, fromlist", [
        ("scipy", None), ("scipy", ("io",)), ("sklearn.datasets", None),
        ("statsmodels", ("api",)),
    ])
    def test_runtime_import_hook_applies_the_same_rule(self, name, fromlist):
        from app.services.ai_chat.sandbox import _restricted_import
        with pytest.raises(ImportError, match="not allowed"):
            _restricted_import(name, None, None, fromlist, 0)

    def test_runtime_import_hook_allows_listed_forms(self):
        from app.services.ai_chat.sandbox import _restricted_import
        assert _restricted_import("scipy", None, None, ("stats",), 0).stats
        assert _restricted_import("scipy.stats", None, None, None, 0)

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

    def test_viz_prompt_colors_bars_with_lc_colored_bars(self):
        """2026-10-21: 3 of 6 "colored by direction" charts listed the
        colors, then re-sorted the bars, so lipids got another lipid's
        direction color. The prompt and its worked example taught that
        list; they now use lc.colored_bars, which reads each bar's color
        from its own row, and say to build any other list last."""
        from app.services.ai_chat.viz_generator import _viz_system_prompt
        prompt = _viz_system_prompt()
        assert "use `lc.colored_bars`" in prompt
        assert "lc.colored_bars(top15, x='LipidMolec'" in prompt  # the example
        assert "bar_colors = [" not in prompt
        assert "build it last, from the DataFrame in its final order" in prompt

    def test_lc_settles_chain_length_before_any_question_is_listed(self):
        """First eval run (2026-10-21): asked to compare the chain length
        distribution, the model asked presence-vs-chain-split. The rule not
        to ask came after the list of questions to ask; lc.chain_length
        counts no individual chains at all, which the prompt did not say."""
        from app.services.ai_chat.prompt_templates import (
            data_questions_system_prompt,
        )
        from app.services.ai_chat.viz_generator import _viz_system_prompt
        for prompt in (data_questions_system_prompt(), _viz_system_prompt()):
            exception = prompt.index("no methodology question when `lc`")
            assert exception < prompt.index("Common ambiguities to watch for")
            assert "**total** carbons and double bonds" in prompt


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

        stream = canned_stream("ok")
        monkeypatch.setattr(
            ai_client.AIClientService, "stream_tool_loop", stream
        )
        session = AIChatWorkflow.start_data_questions_session(df)
        finish(AIChatWorkflow.stream_data_questions_message(
            user_message="top 10 most significant lipids WT vs KO",
            session=session,
            source_df=df,
            experiment=experiment,
        ))
        sent = stream.calls[-1]["messages"][-1]["content"]
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
            ai_client.AIClientService, "stream_tool_loop",
            canned_stream(response),
        )
        session = AIChatWorkflow.start_data_questions_session(df)
        result = finish(AIChatWorkflow.stream_data_questions_message(
            user_message="top lipids", session=session, source_df=df,
            experiment=experiment,
        ))
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
            df, lc=lc,
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


class TestArgumentsAreHonoured:
    """AQ11: ignoring these arguments broke no test."""

    def test_compare_classes_uses_alpha(self, lc):
        first = lc.compare_classes().iloc[0]
        p = first["adj_p_value"]
        assert 0 < p < 0.5
        strict = lc.compare_classes(alpha=p / 2).set_index("ClassKey")
        loose = lc.compare_classes(alpha=min(2 * p, 0.99)).set_index("ClassKey")
        assert not strict.loc[first["ClassKey"], "significant"]
        assert loose.loc[first["ClassKey"], "significant"]

    def test_volcano_keeps_only_the_requested_classes(self, lc):
        data = lc.volcano("WT", "KO", classes=["PC"]).data.volcano_df
        assert len(data) > 0
        assert set(data["ClassKey"]) == {"PC"}

    def test_volcano_p_value_does_not_depend_on_the_classes(self):
        """M1 / AQ5: the zero floor is the whole dataset's, as on the main
        Volcano page, so narrowing the classes leaves a lipid's p-value as is."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 34:1', 'PC 36:2', 'SM 34:1'],
            'ClassKey': ['PC', 'PC', 'SM'],
            # PC 34:1 has a zero; SM holds the dataset's smallest value.
            **{f'concentration[s{i}]': [v, 1000.0 + i, 0.5 + i]
               for i, v in enumerate([0.0, 900.0, 950.0, 400.0, 420.0, 380.0], 1)},
        })
        lc = PipelineAPI(df, ExperimentConfig(
            conditions_list=["WT", "KO"], number_of_samples_list=[3, 3],
        ))

        def p_value(classes):
            res = lc.volcano("WT", "KO", classes=classes).raw
            return res.stat_summary.results['PC 34:1'].p_value

        assert p_value(['PC']) == pytest.approx(p_value(['PC', 'SM']))


class TestColoredBars:
    """lc.colored_bars: each bar's color comes from its own row."""

    def test_colors_follow_each_row_after_a_sort(self, lc):
        from app.services.ai_chat.output_checks import check_direction_colors
        top = lc.compare_species("WT", "KO").head(10)
        top["Direction"] = np.where(top["log2FC"] > 0, "Up in KO", "Down in KO")
        top = top.sort_values("log2FC")
        colors = {"Up in KO": "#d62728", "Down in KO": "#1f77b4"}
        fig = lc.colored_bars(top, x="log2FC", y="LipidMolec",
                              color_by="Direction", colors=colors,
                              orientation="h").figure
        bars = fig.data[0]
        assert list(bars.marker.color) == [colors[d] for d in top["Direction"]]
        assert list(fig.layout.yaxis.categoryarray) == list(top["LipidMolec"])
        assert [t.name for t in fig.data[1:]] == list(dict.fromkeys(top["Direction"]))
        assert check_direction_colors(fig) == []

    def test_default_colors_and_vertical_bars(self, lc):
        table = pd.DataFrame({"lipid": ["a", "b", "c"], "v": [3.0, 2.0, 1.0],
                              "cls": ["PC", "PE", "PC"]})
        result = lc.colored_bars(table, x="lipid", y="v", color_by="cls")
        colors = list(result.figure.data[0].marker.color)
        assert colors[0] == colors[2] != colors[1]
        assert list(result.figure.layout.xaxis.categoryarray) == ["a", "b", "c"]
        assert result.data.equals(table)
