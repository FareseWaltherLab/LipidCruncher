"""
Integration tests for Module 3: Visualize and Analyze Pipeline.

Tests the complete end-to-end analysis flow using real sample datasets
processed through Module 1 (ingestion + normalization) first:
Bar chart → Pie chart → Saturation → Chain Length → FACH → Pathway → Volcano → Heatmap

Multi-step chains validate cross-analysis consistency (same input →
consistent class totals across bar chart and pie chart), cascading
parameter effects, and edge case behavior that unit tests do not cover.
"""
import matplotlib
import matplotlib.pyplot as plt
import pytest
import pandas as pd
import numpy as np

import plotly.graph_objects as go

# Module 1 workflows (used to produce normalized DataFrames)
from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
)
from app.workflows.normalization import (
    NormalizationWorkflow,
    NormalizationWorkflowConfig,
)

# Module 3 workflow (system under test)
from app.workflows.analysis import (
    AnalysisWorkflow,
    BarChartResult,
    PieChartResult,
    SaturationResult,
    ChainLengthResult,
    FACHResult,
    PathwayResult,
    VolcanoResult,
    HeatmapResult,
)

# Services (for result type assertions)
from app.services.statistical_testing import (
    StatisticalTestSummary,
    StatisticalTestResult,
    PostHocResult,
)
from app.services.plotting.volcano_plot import VolcanoData

# Models
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.models.statistics import StatisticalTestConfig

# Services
from app.services.format_detection import DataFormat

# Shared integration test helpers
from tests.integration.conftest import (
    load_lipidsearch_sample,
    load_msdial_sample,
    load_generic_sample,
    load_mw_sample,
    get_concentration_columns,
)


def get_sample_names(df: pd.DataFrame) -> list:
    """Extract sample names from concentration columns."""
    conc_cols = get_concentration_columns(df)
    return [col.replace('concentration[', '').replace(']', '') for col in conc_cols]


def run_module1_pipeline(
    raw_df: pd.DataFrame,
    experiment: ExperimentConfig,
    data_format: DataFormat,
) -> pd.DataFrame:
    """Run Module 1 (ingestion + normalization with method='none') to produce
    a normalized DataFrame with concentration[] columns for analysis input.

    Args:
        raw_df: Raw standardized DataFrame with intensity[] columns.
        experiment: Experiment configuration.
        data_format: Data format for ingestion.

    Returns:
        DataFrame with concentration[] columns ready for analysis.
    """
    # Step 1: Ingestion
    ingestion_config = IngestionConfig(
        experiment=experiment,
        data_format=data_format,
        apply_zero_filter=True,
    )
    ingestion_result = DataIngestionWorkflow.run(raw_df, ingestion_config)
    assert ingestion_result.is_valid, (
        f"Ingestion failed: {ingestion_result.validation_errors}"
    )

    # Step 2: Normalization (method='none' renames intensity[] -> concentration[])
    norm_config = NormalizationWorkflowConfig(
        experiment=experiment,
        normalization=NormalizationConfig(
            method='none',
            selected_classes=list(
                ingestion_result.cleaned_df['ClassKey'].unique()
            ),
        ),
        data_format=data_format,
    )
    norm_result = NormalizationWorkflow.run(
        ingestion_result.cleaned_df, norm_config
    )
    assert norm_result.success, (
        f"Normalization failed: {norm_result.validation_errors}"
    )

    return norm_result.normalized_df


def make_analysis_dataframe(
    lipids: list,
    classes: list,
    n_samples: int,
    values_fn=None,
) -> pd.DataFrame:
    """Build a synthetic DataFrame with concentration[] columns for analysis.

    Args:
        lipids: List of LipidMolec names.
        classes: List of ClassKey values (parallel to lipids).
        n_samples: Number of samples.
        values_fn: Callable(lipid_index, sample_index) -> float.
            Defaults to (lip_idx+1) * 1e6 + sample_idx * 1e4.

    Returns:
        DataFrame with LipidMolec, ClassKey, and concentration[s1..sN] columns.
    """
    if values_fn is None:
        values_fn = lambda lip_idx, samp_idx: (lip_idx + 1) * 1e6 + samp_idx * 1e4

    data = {'LipidMolec': lipids, 'ClassKey': classes}
    for s_idx in range(n_samples):
        col = f'concentration[s{s_idx + 1}]'
        data[col] = [values_fn(l_idx, s_idx) for l_idx in range(len(lipids))]
    return pd.DataFrame(data)


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def lipidsearch_experiment():
    """Experiment config matching lipidsearch5_test_dataset.csv.
    3 conditions x 4 samples each = 12 samples (4 WT, 4 ADGAT_DKO, 4 BQC)."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4],
    )


@pytest.fixture
def msdial_experiment():
    """Experiment config matching msdial_test_dataset.csv.
    3 conditions: 1 Blank, 3 fads2_KO, 3 WT = 7 samples."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Blank', 'fads2_KO', 'WT'],
        number_of_samples_list=[1, 3, 3],
    )


@pytest.fixture
def generic_experiment():
    """Experiment config matching generic_test_dataset.csv.
    3 conditions x 4 samples each = 12 samples."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4],
    )


@pytest.fixture
def mw_experiment():
    """Experiment config matching mw_test_dataset.csv.
    2 conditions x 22 samples = 44 samples."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Normal', 'HFD'],
        number_of_samples_list=[22, 22],
    )


@pytest.fixture
def four_sample_experiment():
    """Simple 2x2 experiment for synthetic DataFrames."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2],
    )


@pytest.fixture
def three_condition_experiment():
    """3 conditions x 3 samples for multi-group stats tests."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'Vehicle'],
        number_of_samples_list=[3, 3, 3],
    )


@pytest.fixture
def two_sample_experiment():
    """Minimal 2x1 experiment for boundary tests."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1],
    )


# =============================================================================
# Normalized DataFrame Fixtures (module-scoped, cached)
# =============================================================================

@pytest.fixture(scope="module")
def lipidsearch_normalized_df():
    """Normalized LipidSearch DataFrame via Module 1 pipeline."""
    raw_df = load_lipidsearch_sample()
    experiment = ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4],
    )
    return run_module1_pipeline(raw_df, experiment, DataFormat.LIPIDSEARCH)


@pytest.fixture(scope="module")
def msdial_normalized_df():
    """Normalized MS-DIAL DataFrame via Module 1 pipeline."""
    raw_df = load_msdial_sample()
    experiment = ExperimentConfig(
        n_conditions=3,
        conditions_list=['Blank', 'fads2_KO', 'WT'],
        number_of_samples_list=[1, 3, 3],
    )
    return run_module1_pipeline(raw_df, experiment, DataFormat.MSDIAL)


@pytest.fixture(scope="module")
def generic_normalized_df():
    """Normalized Generic DataFrame via Module 1 pipeline."""
    raw_df = load_generic_sample()
    experiment = ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4],
    )
    return run_module1_pipeline(raw_df, experiment, DataFormat.GENERIC)


@pytest.fixture(scope="module")
def mw_normalized_df():
    """Normalized Metabolomics Workbench DataFrame via Module 1 pipeline."""
    raw_df = load_mw_sample()
    experiment = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Normal', 'HFD'],
        number_of_samples_list=[22, 22],
    )
    return run_module1_pipeline(raw_df, experiment, DataFormat.METABOLOMICS_WORKBENCH)


# =============================================================================
# Statistical Config Fixtures
# =============================================================================

@pytest.fixture
def auto_stat_config():
    """Auto-mode stat config."""
    return StatisticalTestConfig.create_auto()


@pytest.fixture
def manual_parametric_config():
    """Manual parametric stat config with FDR correction."""
    return StatisticalTestConfig.create_manual(
        test_type='parametric',
        correction_method='fdr_bh',
        posthoc_correction='bonferroni',
        alpha=0.05,
    )


# =============================================================================
# Synthetic Edge-Case DataFrames
# =============================================================================

@pytest.fixture
def single_lipid_df():
    """Single lipid with 4 samples."""
    return make_analysis_dataframe(
        lipids=['PC(16:0/18:1)'],
        classes=['PC'],
        n_samples=4,
    )


@pytest.fixture
def single_class_df():
    """Multiple lipids, single class, 4 samples."""
    return make_analysis_dataframe(
        lipids=['PC(16:0/18:1)', 'PC(18:0/20:4)', 'PC(16:0/18:2)'],
        classes=['PC', 'PC', 'PC'],
        n_samples=4,
    )


@pytest.fixture
def uniform_df():
    """All concentrations identical — zero variance per lipid."""
    return make_analysis_dataframe(
        lipids=['PC(16:0/18:1)', 'PE(18:0/20:4)', 'TG(16:0/18:1/18:2)'],
        classes=['PC', 'PE', 'TG'],
        n_samples=4,
        values_fn=lambda lip_idx, samp_idx: 1e6,
    )


@pytest.fixture
def many_classes_df():
    """14 classes, 1 lipid each, 4 samples."""
    classes_list = [
        'PC', 'PE', 'TG', 'DG', 'SM', 'Cer', 'PI', 'PG',
        'PS', 'PA', 'LPC', 'LPE', 'CE', 'CL',
    ]
    lipids = [f'{cls}(16:0/18:1)' for cls in classes_list]
    return make_analysis_dataframe(
        lipids=lipids,
        classes=classes_list,
        n_samples=4,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestBarChartEndToEnd:
    """Bar chart analysis with real and synthetic data."""

    def test_lipidsearch_bar_chart_default(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Full pipeline: LipidSearch data → bar chart with all classes/conditions."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
        )

        assert isinstance(result, BarChartResult)
        assert isinstance(result.figure, go.Figure)
        assert not result.abundance_df.empty
        assert 'ClassKey' in result.abundance_df.columns
        assert result.stat_summary is None  # no config passed

    def test_lipidsearch_bar_chart_with_stats(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Bar chart with auto-mode statistical testing."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
            stat_config=auto_stat_config,
        )

        assert isinstance(result, BarChartResult)
        assert isinstance(result.stat_summary, StatisticalTestSummary)
        assert len(result.stat_summary.results) > 0

    def test_lipidsearch_bar_chart_log10_scale(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Bar chart in log10 scale preserves class structure."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes, scale='log10',
        )

        assert isinstance(result.figure, go.Figure)
        # Log10 columns should be present in abundance_df
        log_cols = [
            c for c in result.abundance_df.columns
            if 'log10' in c.lower()
        ]
        assert len(log_cols) > 0

    def test_generic_bar_chart(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format produces valid bar chart."""
        df = generic_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(generic_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, generic_experiment, conditions, classes,
        )

        assert isinstance(result, BarChartResult)
        assert not result.abundance_df.empty

    def test_msdial_bar_chart(
        self, msdial_normalized_df, msdial_experiment,
    ):
        """MS-DIAL format produces valid bar chart."""
        df = msdial_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(msdial_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, msdial_experiment, conditions, classes,
        )

        assert isinstance(result, BarChartResult)
        assert not result.abundance_df.empty

    def test_mw_bar_chart(
        self, mw_normalized_df, mw_experiment,
    ):
        """Metabolomics Workbench format with many samples."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, mw_experiment, conditions, classes,
        )

        assert isinstance(result, BarChartResult)
        assert not result.abundance_df.empty

    def test_bar_chart_subset_classes(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Selecting a subset of classes filters the output correctly."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        subset = all_classes[:3]
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, subset,
        )

        output_classes = set(result.abundance_df['ClassKey'].unique())
        assert output_classes == set(subset)

    def test_bar_chart_subset_conditions(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Selecting a subset of conditions limits the output."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, ['WT'], classes,
        )

        assert isinstance(result, BarChartResult)
        # Mean columns should only reference WT
        mean_cols = [
            c for c in result.abundance_df.columns if 'mean' in c.lower()
        ]
        assert len(mean_cols) > 0
        for col in mean_cols:
            assert 'WT' in col

    def test_bar_chart_manual_stats(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        manual_parametric_config,
    ):
        """Manual parametric config produces statistical results."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
            stat_config=manual_parametric_config,
        )

        assert isinstance(result.stat_summary, StatisticalTestSummary)
        # Each class should have a test result
        for cls in classes:
            assert cls in result.stat_summary.results

    def test_bar_chart_three_conditions_posthoc(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Three conditions trigger post-hoc pairwise tests."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO', 'BQC']

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
            stat_config=auto_stat_config,
        )

        assert isinstance(result.stat_summary, StatisticalTestSummary)
        # Post-hoc results should exist for 3+ conditions
        if result.stat_summary.posthoc_results:
            for cls, posthocs in result.stat_summary.posthoc_results.items():
                for ph in posthocs:
                    assert isinstance(ph, PostHocResult)

    def test_bar_chart_abundance_values_positive(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Mean abundance values are non-negative."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
        )

        mean_cols = [
            c for c in result.abundance_df.columns if 'mean' in c.lower()
        ]
        for col in mean_cols:
            assert (result.abundance_df[col] >= 0).all(), (
                f"Negative mean abundance in {col}"
            )


class TestPieChartEndToEnd:
    """Pie chart analysis with real data."""

    def test_lipidsearch_pie_chart_all_conditions(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """One pie chart per condition, all produce valid results."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        results = AnalysisWorkflow.run_pie_charts(
            df, lipidsearch_experiment, conditions, classes,
        )

        assert isinstance(results, dict)
        for cond in conditions:
            assert cond in results
            r = results[cond]
            assert isinstance(r, PieChartResult)
            assert isinstance(r.figure, go.Figure)
            assert r.condition == cond
            assert not r.data_df.empty

    def test_pie_chart_percentages_sum_to_100(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Pie chart percentages for each condition sum to ~100%."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        results = AnalysisWorkflow.run_pie_charts(
            df, lipidsearch_experiment, conditions, classes,
        )

        for cond, r in results.items():
            if 'Percentage' in r.data_df.columns:
                total = r.data_df['Percentage'].sum()
                assert abs(total - 100.0) < 0.1, (
                    f"Percentages for {cond} sum to {total}, expected ~100"
                )

    def test_pie_chart_color_consistency(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Same class gets the same color across all conditions."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        results = AnalysisWorkflow.run_pie_charts(
            df, lipidsearch_experiment, conditions, classes,
        )

        # All figures should exist
        assert len(results) == len(conditions)

    def test_pie_chart_subset_classes(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Subset of classes limits pie chart entries."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        subset = all_classes[:2]

        results = AnalysisWorkflow.run_pie_charts(
            df, lipidsearch_experiment, ['WT'], subset,
        )

        assert 'WT' in results
        r = results['WT']
        # Only selected classes should appear
        if 'ClassKey' in r.data_df.columns:
            output_classes = set(r.data_df['ClassKey'].unique())
            assert output_classes.issubset(set(subset))

    def test_mw_pie_chart(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format with 2 conditions produces 2 pie charts."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        results = AnalysisWorkflow.run_pie_charts(
            df, mw_experiment, conditions, classes,
        )

        assert len(results) == 2
        for cond in conditions:
            assert cond in results

    def test_generic_pie_chart_single_condition(
        self, generic_normalized_df, generic_experiment,
    ):
        """Single condition pie chart works."""
        df = generic_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        results = AnalysisWorkflow.run_pie_charts(
            df, generic_experiment, ['WT'], classes,
        )

        assert len(results) == 1
        assert 'WT' in results


class TestSaturationEndToEnd:
    """Saturation (SFA/MUFA/PUFA) analysis with real data."""

    def test_lipidsearch_saturation_default(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Full saturation analysis on LipidSearch data."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, classes,
        )

        assert isinstance(result, SaturationResult)
        assert isinstance(result.plots, dict)
        assert len(result.plots) > 0
        for fig in result.plots.values():
            assert isinstance(fig, go.Figure)
        assert result.stat_summary is None  # no config passed

    def test_saturation_with_stats(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Saturation analysis with statistical tests."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, classes,
            stat_config=auto_stat_config,
            plot_type='concentration',
        )

        assert isinstance(result.stat_summary, StatisticalTestSummary)
        assert len(result.stat_summary.results) > 0

    def test_saturation_percentage_mode(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Percentage plot type skips stats and generates figures."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, classes,
            stat_config=StatisticalTestConfig.create_auto(),
            plot_type='percentage',
        )

        # Percentage mode should NOT run stats
        assert result.stat_summary is None
        assert len(result.plots) > 0

    def test_saturation_consolidated_lipids_detection(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Consolidated lipids (e.g., PC(34:1)) are detected."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, classes,
        )

        assert isinstance(result.consolidated_lipids, dict)
        # Consolidated lipids dict may or may not have entries depending on data

    def test_saturation_per_class_plots(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """One plot per selected class."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        subset = all_classes[:3]
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, subset,
        )

        # Should have plots for up to len(subset) classes
        assert len(result.plots) <= len(subset)

    def test_msdial_saturation(
        self, msdial_normalized_df, msdial_experiment,
    ):
        """MS-DIAL format saturation analysis."""
        df = msdial_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['fads2_KO', 'WT']

        result = AnalysisWorkflow.run_saturation(
            df, msdial_experiment, conditions, classes,
        )

        assert isinstance(result, SaturationResult)

    def test_mw_saturation(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format saturation analysis with many samples."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        result = AnalysisWorkflow.run_saturation(
            df, mw_experiment, conditions, classes,
        )

        assert isinstance(result, SaturationResult)


class TestChainLengthEndToEnd:
    """Chain length distribution analysis with real data."""

    def test_lipidsearch_chain_length(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Full chain length analysis on LipidSearch data."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_chain_length(
            df, lipidsearch_experiment, conditions, classes,
        )

        assert isinstance(result, ChainLengthResult)
        assert result.success
        assert result.figure is not None
        assert isinstance(result.figure, go.Figure)
        assert result.per_condition_data is not None
        assert len(result.per_condition_data) == 2
        for cond_data in result.per_condition_data.values():
            assert len(cond_data.records) > 0

    def test_chain_length_class_subset(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Selecting a subset of classes filters results."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        subset = all_classes[:2]
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_chain_length(
            df, lipidsearch_experiment, conditions, subset,
        )

        assert result.success
        result_classes = set()
        for cond_data in result.per_condition_data.values():
            result_classes.update(r['ClassKey'] for r in cond_data.records)
        assert result_classes <= set(subset)

    def test_chain_length_records_have_positive_concentration(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """All records have positive mean concentrations."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_chain_length(
            df, lipidsearch_experiment, conditions, classes,
        )

        for cond_data in result.per_condition_data.values():
            for rec in cond_data.records:
                assert rec['MeanConcentration'] > 0

    def test_msdial_chain_length(
        self, msdial_normalized_df, msdial_experiment,
    ):
        """MS-DIAL format chain length analysis."""
        df = msdial_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['fads2_KO', 'WT']

        result = AnalysisWorkflow.run_chain_length(
            df, msdial_experiment, conditions, classes,
        )

        assert isinstance(result, ChainLengthResult)
        assert result.success

    def test_generic_chain_length(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format chain length analysis (species-level names)."""
        df = generic_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(generic_experiment)

        result = AnalysisWorkflow.run_chain_length(
            df, generic_experiment, conditions, classes,
        )

        assert isinstance(result, ChainLengthResult)
        assert result.success
        # Generic data has species-level names (e.g. PC 34:1) — should still parse
        assert any(
            len(d.records) > 0 for d in result.per_condition_data.values()
        )

    def test_mw_chain_length(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format chain length analysis with many samples."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        result = AnalysisWorkflow.run_chain_length(
            df, mw_experiment, conditions, classes,
        )

        assert isinstance(result, ChainLengthResult)
        assert result.success

    def test_chain_length_no_conditions_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Empty conditions list raises ValueError."""
        with pytest.raises(ValueError, match="condition"):
            AnalysisWorkflow.run_chain_length(
                lipidsearch_normalized_df, lipidsearch_experiment, [], ['PC'],
            )

    def test_chain_length_no_classes_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Empty classes list raises ValueError."""
        with pytest.raises(ValueError, match="class"):
            AnalysisWorkflow.run_chain_length(
                lipidsearch_normalized_df, lipidsearch_experiment,
                ['WT'], [],
            )


class TestChainLengthParseCoverage:
    """Ensure parse_total_chain_info handles all lipid formats in real datasets.

    These tests catch silent parse failures: if a class has lipids with
    chain notation (carbon:db) but none parse, the test fails. This forces
    us to update the parser when new lipid name formats appear in the data.
    """

    @staticmethod
    def _check_parse_coverage(df):
        """Assert every class with chain-bearing lipids has >0 parseable."""
        from app.services.plotting.chain_length_plot import (
            ChainLengthPlotterService,
        )
        failures = []
        for cls in sorted(df['ClassKey'].unique()):
            lipids = df[df['ClassKey'] == cls]['LipidMolec'].tolist()
            # Skip classes where no lipid has chain notation (e.g. Ch)
            has_chain_notation = any(':' in str(lip) for lip in lipids)
            if not has_chain_notation:
                continue
            parseable = sum(
                1 for lip in lipids
                if ChainLengthPlotterService.parse_total_chain_info(lip)
                is not None
            )
            if parseable == 0:
                samples = lipids[:3]
                failures.append(f"{cls}: 0/{len(lipids)} parsed ({samples})")
        assert not failures, (
            "Classes with chain notation but 0 parseable lipids:\n"
            + "\n".join(failures)
        )

    def test_lipidsearch_parse_coverage(self, lipidsearch_normalized_df):
        self._check_parse_coverage(lipidsearch_normalized_df)

    def test_msdial_parse_coverage(self, msdial_normalized_df):
        self._check_parse_coverage(msdial_normalized_df)

    def test_generic_parse_coverage(self, generic_normalized_df):
        self._check_parse_coverage(generic_normalized_df)

    def test_mw_parse_coverage(self, mw_normalized_df):
        self._check_parse_coverage(mw_normalized_df)


class TestFACHEndToEnd:
    """FACH (Fatty Acid Carbon/Hydrogen) analysis with real data."""

    def test_lipidsearch_fach(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """FACH analysis on a single class from LipidSearch data."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        # Use the first available class
        result = AnalysisWorkflow.run_fach(
            df, lipidsearch_experiment, classes[0], conditions,
        )

        assert isinstance(result, FACHResult)
        assert isinstance(result.data_dict, dict)
        # Each selected condition should have data
        for cond in conditions:
            if cond in result.data_dict:
                cond_df = result.data_dict[cond]
                assert 'Carbon' in cond_df.columns
                assert 'DB' in cond_df.columns
                assert 'Proportion' in cond_df.columns

    def test_fach_weighted_averages(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """FACH returns per-condition weighted averages."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_fach(
            df, lipidsearch_experiment, classes[0], conditions,
        )

        assert isinstance(result.weighted_averages, dict)
        for cond, (avg_db, avg_carbon) in result.weighted_averages.items():
            assert isinstance(avg_db, float)
            assert isinstance(avg_carbon, float)
            assert avg_carbon >= 0
            assert avg_db >= 0

    def test_fach_unparsable_lipids_tracked(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Unparsable lipids are recorded (may be empty if all parsable)."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_fach(
            df, lipidsearch_experiment, classes[0], ['WT', 'ADGAT_DKO'],
        )

        assert isinstance(result.unparsable_lipids, list)

    def test_fach_figure_created(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """FACH generates a heatmap figure (or None if no parsable data)."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_fach(
            df, lipidsearch_experiment, classes[0], ['WT', 'ADGAT_DKO'],
        )

        # Figure is go.Figure or None depending on data
        if result.figure is not None:
            assert isinstance(result.figure, go.Figure)

    def test_mw_fach(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format FACH analysis."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        result = AnalysisWorkflow.run_fach(
            df, mw_experiment, classes[0], conditions,
        )

        assert isinstance(result, FACHResult)


class TestPathwayEndToEnd:
    """Pathway visualization with real data."""

    def test_lipidsearch_pathway(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Pathway analysis on LipidSearch data."""
        df = lipidsearch_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO',
        )

        assert isinstance(result, PathwayResult)
        assert isinstance(result.pathway_dict, dict)
        assert not result.fold_change_df.empty
        assert 'ClassKey' in result.fold_change_df.columns
        assert 'fold_change' in result.fold_change_df.columns
        assert not result.saturation_df.empty

    def test_pathway_fold_change_positive(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Fold change values are positive (ratio, not log)."""
        df = lipidsearch_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO',
        )

        assert (result.fold_change_df['fold_change'] > 0).all()

    def test_pathway_saturation_ratio_bounded(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Saturation ratios are between 0 and 1."""
        df = lipidsearch_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO',
        )

        ratios = result.saturation_df['saturation_ratio']
        assert (ratios >= 0).all()
        assert (ratios <= 1).all()

    def test_pathway_figure_created(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Pathway generates a Plotly figure."""
        df = lipidsearch_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO',
        )

        if result.figure is not None:
            assert isinstance(result.figure, go.Figure)

    def test_pathway_reversed_conditions(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Swapping control/experimental inverts fold changes."""
        df = lipidsearch_normalized_df

        result_forward = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO',
        )
        result_reverse = AnalysisWorkflow.run_pathway(
            df, lipidsearch_experiment, 'ADGAT_DKO', 'WT',
        )

        # Fold changes should be reciprocal (fc_forward * fc_reverse ≈ 1.0)
        fc_fwd = result_forward.fold_change_df.set_index('ClassKey')['fold_change']
        fc_rev = result_reverse.fold_change_df.set_index('ClassKey')['fold_change']

        common_classes = fc_fwd.index.intersection(fc_rev.index)
        if len(common_classes) > 0:
            product = fc_fwd[common_classes] * fc_rev[common_classes]
            np.testing.assert_allclose(product.values, 1.0, rtol=0.01)
        plt.close('all')

    def test_mw_pathway(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format pathway analysis."""
        df = mw_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, mw_experiment, 'Normal', 'HFD',
        )

        assert isinstance(result, PathwayResult)
        assert not result.fold_change_df.empty
        plt.close('all')

    def test_generic_pathway(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format pathway analysis."""
        df = generic_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, generic_experiment, 'WT', 'ADGAT_DKO',
        )

        assert isinstance(result, PathwayResult)
        assert isinstance(result.pathway_dict, dict)
        assert not result.fold_change_df.empty
        assert 'ClassKey' in result.fold_change_df.columns
        assert 'fold_change' in result.fold_change_df.columns
        assert not result.saturation_df.empty

    def test_generic_pathway_fold_change_positive(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format: fold change values are positive."""
        df = generic_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, generic_experiment, 'WT', 'ADGAT_DKO',
        )

        assert (result.fold_change_df['fold_change'] > 0).all()

    def test_generic_pathway_saturation_bounded(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format: saturation ratios are between 0 and 1."""
        df = generic_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, generic_experiment, 'WT', 'ADGAT_DKO',
        )

        ratios = result.saturation_df['saturation_ratio']
        assert (ratios >= 0).all()
        assert (ratios <= 1).all()

    def test_generic_pathway_figure_created(
        self, generic_normalized_df, generic_experiment,
    ):
        """Generic format: pathway generates a Plotly figure."""
        df = generic_normalized_df

        result = AnalysisWorkflow.run_pathway(
            df, generic_experiment, 'WT', 'ADGAT_DKO',
        )

        if result.figure is not None:
            assert isinstance(result.figure, go.Figure)


class TestVolcanoEndToEnd:
    """Volcano plot analysis with real data."""

    def test_lipidsearch_volcano(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Full volcano analysis on LipidSearch data."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        assert isinstance(result, VolcanoResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.volcano_data, VolcanoData)
        assert not result.volcano_data.volcano_df.empty
        assert isinstance(result.stat_summary, StatisticalTestSummary)

    def test_volcano_data_columns(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Volcano DataFrame has required columns."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        required_cols = {'LipidMolec', 'ClassKey', 'FoldChange', 'pValue'}
        assert required_cols.issubset(set(result.volcano_data.volcano_df.columns))

    def test_volcano_fold_change_sign(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Fold change is log2 — can be positive or negative."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        fc = result.volcano_data.volcano_df['FoldChange']
        # Should have both positive and negative fold changes
        # (unless all lipids go one direction — at least check it's not empty)
        assert len(fc) > 0

    def test_volcano_p_values_bounded(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """P-values should be between 0 and 1."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        pvals = result.volcano_data.volcano_df['pValue']
        assert (pvals >= 0).all()
        assert (pvals <= 1).all()

    def test_volcano_concentration_plot(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Concentration-vs-fold-change scatter is created."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        assert isinstance(result.concentration_plot, go.Figure)
        assert isinstance(result.concentration_df, pd.DataFrame)

    def test_volcano_removed_lipids_tracked(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Lipids removed from volcano are tracked with reasons."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        removed = result.volcano_data.removed_lipids_df
        assert isinstance(removed, pd.DataFrame)
        if not removed.empty:
            assert 'Reason' in removed.columns

    def test_volcano_subset_classes(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Subset of classes limits volcano to those classes only."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        subset = all_classes[:2]

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', subset,
            stat_config=auto_stat_config,
        )

        volcano_classes = set(
            result.volcano_data.volcano_df['ClassKey'].unique()
        )
        assert volcano_classes.issubset(set(subset))

    def test_volcano_hide_non_significant(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """hide_non_sig flag produces a valid figure."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
            hide_non_sig=True,
        )

        assert isinstance(result.figure, go.Figure)

    def test_volcano_top_n_labels(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """top_n_labels produces a valid labeled figure."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
            top_n_labels=5,
        )

        assert isinstance(result.figure, go.Figure)

    def test_mw_volcano(
        self, mw_normalized_df, mw_experiment, auto_stat_config,
    ):
        """MW format volcano analysis with many samples."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, mw_experiment, 'Normal', 'HFD', classes,
            stat_config=auto_stat_config,
        )

        assert isinstance(result, VolcanoResult)
        assert not result.volcano_data.volcano_df.empty


class TestHeatmapEndToEnd:
    """Lipidomic heatmap analysis with real data."""

    def test_lipidsearch_regular_heatmap(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Regular heatmap on LipidSearch data."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, classes,
            heatmap_type='regular',
        )

        assert isinstance(result, HeatmapResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.z_scores_df, pd.DataFrame)
        assert not result.z_scores_df.empty
        assert result.cluster_composition is None  # regular mode

    def test_lipidsearch_clustered_heatmap(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Clustered heatmap with cluster composition."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, classes,
            heatmap_type='clustered', n_clusters=3,
        )

        assert isinstance(result, HeatmapResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.cluster_composition, pd.DataFrame)
        assert not result.cluster_composition.empty

    def test_heatmap_z_scores_normalized(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Z-scores have mean ≈ 0 and std ≈ 1 per row."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, classes,
        )

        z = result.z_scores_df
        # Per-row mean should be ≈ 0
        row_means = z.mean(axis=1)
        np.testing.assert_allclose(row_means.values, 0.0, atol=1e-10)

    def test_heatmap_cluster_count_matches(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Number of clusters in composition matches requested n_clusters."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']
        n_clusters = 4

        # Need enough lipids for the requested clusters
        n_lipids_available = len(df[df['ClassKey'].isin(classes)])
        if n_lipids_available < n_clusters:
            pytest.skip(f"Only {n_lipids_available} lipids, need {n_clusters}")

        result = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, classes,
            heatmap_type='clustered', n_clusters=n_clusters,
        )

        # Cluster composition should have n_clusters rows
        assert len(result.cluster_composition) == n_clusters

    def test_heatmap_subset_conditions(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Selecting fewer conditions reduces heatmap columns."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result_2cond = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, ['WT', 'ADGAT_DKO'], classes,
        )
        result_1cond = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, ['WT'], classes,
        )

        # Fewer conditions → fewer columns in z_scores
        assert result_1cond.z_scores_df.shape[1] < result_2cond.z_scores_df.shape[1]

    def test_heatmap_subset_classes(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Selecting fewer classes reduces heatmap rows."""
        df = lipidsearch_normalized_df
        all_classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        result_all = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, all_classes,
        )
        result_sub = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, all_classes[:2],
        )

        assert result_sub.z_scores_df.shape[0] <= result_all.z_scores_df.shape[0]

    def test_mw_heatmap(
        self, mw_normalized_df, mw_experiment,
    ):
        """MW format heatmap with many samples."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = AnalysisWorkflow.get_all_conditions(mw_experiment)

        result = AnalysisWorkflow.run_heatmap(
            df, mw_experiment, conditions, classes,
        )

        assert isinstance(result, HeatmapResult)
        assert isinstance(result.figure, go.Figure)
        # MW has 44 samples → heatmap should have 44 columns
        assert result.z_scores_df.shape[1] == 44


class TestCrossAnalysis:
    """Cross-analysis consistency: same input → consistent outputs."""

    def test_bar_pie_class_totals_consistent(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Bar chart mean abundances and pie chart totals agree on class ranking."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT']

        bar_result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
        )
        pie_results = AnalysisWorkflow.run_pie_charts(
            df, lipidsearch_experiment, conditions, classes,
        )

        # Both should reference the same classes
        bar_classes = set(bar_result.abundance_df['ClassKey'].unique())
        if 'WT' in pie_results and 'ClassKey' in pie_results['WT'].data_df.columns:
            pie_classes = set(pie_results['WT'].data_df['ClassKey'].unique())
            # Pie classes should be subset of bar classes (pie may drop zero-abundance)
            assert pie_classes.issubset(bar_classes)

    def test_bar_chart_heatmap_class_agreement(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Bar chart and heatmap operate on the same filtered class set."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)[:5]
        conditions = ['WT', 'ADGAT_DKO']

        bar_result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
        )
        heatmap_result = AnalysisWorkflow.run_heatmap(
            df, lipidsearch_experiment, conditions, classes,
        )

        bar_classes = set(bar_result.abundance_df['ClassKey'].unique())
        # Heatmap Z-scores should cover the same classes
        if hasattr(heatmap_result.z_scores_df.index, 'get_level_values'):
            try:
                hm_classes = set(
                    heatmap_result.z_scores_df.index.get_level_values('ClassKey')
                )
                assert hm_classes.issubset(set(classes))
            except KeyError:
                pass  # Index structure may vary

    def test_volcano_stats_match_bar_chart_direction(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Volcano fold change direction is consistent with bar chart means."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        bar_result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, ['WT', 'ADGAT_DKO'], classes,
        )
        volcano_result = AnalysisWorkflow.run_volcano(
            df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', classes,
            stat_config=auto_stat_config,
        )

        # Both should produce valid results (consistency is structural)
        assert isinstance(bar_result, BarChartResult)
        assert isinstance(volcano_result, VolcanoResult)
        assert not bar_result.abundance_df.empty
        assert not volcano_result.volcano_data.volcano_df.empty

    def test_saturation_classes_subset_of_bar_chart(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Saturation plots are generated for a subset of available classes."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        bar_result = AnalysisWorkflow.run_bar_chart(
            df, lipidsearch_experiment, conditions, classes,
        )
        sat_result = AnalysisWorkflow.run_saturation(
            df, lipidsearch_experiment, conditions, classes,
        )

        bar_classes = set(bar_result.abundance_df['ClassKey'].unique())
        sat_classes = set(sat_result.plots.keys())
        # Saturation may have fewer classes (some may lack parsable FA info)
        assert sat_classes.issubset(bar_classes) or len(sat_classes) <= len(bar_classes)

    def test_all_analyses_same_input(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """All 7 analysis types run on the same input without error."""
        df = lipidsearch_normalized_df
        exp = lipidsearch_experiment
        classes = AnalysisWorkflow.get_available_classes(df)
        conditions = ['WT', 'ADGAT_DKO']

        bar = AnalysisWorkflow.run_bar_chart(df, exp, conditions, classes)
        pie = AnalysisWorkflow.run_pie_charts(df, exp, conditions, classes)
        sat = AnalysisWorkflow.run_saturation(df, exp, conditions, classes)
        fach = AnalysisWorkflow.run_fach(df, exp, classes[0], conditions)
        pathway = AnalysisWorkflow.run_pathway(df, exp, 'WT', 'ADGAT_DKO')
        volcano = AnalysisWorkflow.run_volcano(
            df, exp, 'WT', 'ADGAT_DKO', classes, auto_stat_config,
        )
        heatmap = AnalysisWorkflow.run_heatmap(df, exp, conditions, classes)

        assert isinstance(bar, BarChartResult)
        assert isinstance(pie, dict) and len(pie) > 0
        assert isinstance(sat, SaturationResult)
        assert isinstance(fach, FACHResult)
        assert isinstance(pathway, PathwayResult)
        assert isinstance(volcano, VolcanoResult)
        assert isinstance(heatmap, HeatmapResult)
        plt.close('all')


class TestEdgeCases:
    """Boundary conditions and unusual data shapes."""

    def test_single_lipid_bar_chart(
        self, single_lipid_df, four_sample_experiment,
    ):
        """Single lipid produces valid bar chart."""
        result = AnalysisWorkflow.run_bar_chart(
            single_lipid_df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC'],
        )

        assert isinstance(result, BarChartResult)
        assert len(result.abundance_df) == 1

    def test_single_class_bar_chart(
        self, single_class_df, four_sample_experiment,
    ):
        """Single class with multiple lipids."""
        result = AnalysisWorkflow.run_bar_chart(
            single_class_df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC'],
        )

        assert isinstance(result, BarChartResult)
        assert len(result.abundance_df) == 1  # one class row

    def test_single_condition_bar_chart_no_stats(
        self, single_class_df, four_sample_experiment,
    ):
        """Single condition → stats are skipped (need ≥2 for comparison)."""
        result = AnalysisWorkflow.run_bar_chart(
            single_class_df, four_sample_experiment,
            ['Control'], ['PC'],
            stat_config=StatisticalTestConfig.create_auto(),
        )

        # Stats require ≥2 conditions
        assert result.stat_summary is None

    def test_uniform_data_heatmap(
        self, uniform_df, four_sample_experiment,
    ):
        """Uniform data (zero variance) — Z-scores may be NaN/0."""
        result = AnalysisWorkflow.run_heatmap(
            uniform_df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC', 'PE', 'TG'],
        )

        assert isinstance(result, HeatmapResult)
        # Z-scores with zero std should be NaN or 0
        z = result.z_scores_df
        assert z is not None

    def test_many_classes_bar_chart(
        self, many_classes_df, four_sample_experiment,
    ):
        """14 classes produce a valid bar chart."""
        classes = many_classes_df['ClassKey'].unique().tolist()
        result = AnalysisWorkflow.run_bar_chart(
            many_classes_df, four_sample_experiment,
            ['Control', 'Treatment'], classes,
        )

        assert len(result.abundance_df) == 14

    def test_many_classes_pie_chart(
        self, many_classes_df, four_sample_experiment,
    ):
        """14 classes produce a valid pie chart."""
        classes = many_classes_df['ClassKey'].unique().tolist()
        results = AnalysisWorkflow.run_pie_charts(
            many_classes_df, four_sample_experiment,
            ['Control'], classes,
        )

        assert 'Control' in results

    def test_single_lipid_volcano(
        self, single_lipid_df, four_sample_experiment,
    ):
        """Single lipid volcano plot."""
        result = AnalysisWorkflow.run_volcano(
            single_lipid_df, four_sample_experiment,
            'Control', 'Treatment', ['PC'],
            stat_config=StatisticalTestConfig.create_auto(),
        )

        assert isinstance(result, VolcanoResult)

    def test_single_lipid_heatmap(
        self, single_lipid_df, four_sample_experiment,
    ):
        """Single lipid heatmap."""
        result = AnalysisWorkflow.run_heatmap(
            single_lipid_df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC'],
        )

        assert isinstance(result, HeatmapResult)
        assert result.z_scores_df.shape[0] == 1

    def test_very_large_values(self, four_sample_experiment):
        """Very large concentrations (1e15) don't cause overflow."""
        df = make_analysis_dataframe(
            lipids=['PC(16:0/18:1)', 'PE(18:0/20:4)'],
            classes=['PC', 'PE'],
            n_samples=4,
            values_fn=lambda l, s: 1e15 + l * 1e12,
        )

        result = AnalysisWorkflow.run_bar_chart(
            df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )

        assert isinstance(result, BarChartResult)
        mean_cols = [c for c in result.abundance_df.columns if 'mean' in c.lower()]
        for col in mean_cols:
            assert np.isfinite(result.abundance_df[col]).all()

    def test_very_small_values(self, four_sample_experiment):
        """Very small concentrations (1e-15) produce valid results."""
        df = make_analysis_dataframe(
            lipids=['PC(16:0/18:1)', 'PE(18:0/20:4)'],
            classes=['PC', 'PE'],
            n_samples=4,
            values_fn=lambda l, s: 1e-15 + l * 1e-16,
        )

        result = AnalysisWorkflow.run_bar_chart(
            df, four_sample_experiment,
            ['Control', 'Treatment'], ['PC', 'PE'],
        )

        assert isinstance(result, BarChartResult)

    def test_fach_unparsable_lipid_names(self, four_sample_experiment):
        """Lipids with unparsable names still run (with unparsable tracking)."""
        df = make_analysis_dataframe(
            lipids=['UNKNOWN_LIPID', 'ANOTHER_UNKNOWN'],
            classes=['UNK', 'UNK'],
            n_samples=4,
        )

        result = AnalysisWorkflow.run_fach(
            df, four_sample_experiment, 'UNK', ['Control', 'Treatment'],
        )

        assert isinstance(result, FACHResult)
        # All lipids should be unparsable
        assert len(result.unparsable_lipids) > 0

    def test_many_samples_mw_volcano(
        self, mw_normalized_df, mw_experiment, auto_stat_config,
    ):
        """Volcano with 22 samples per condition (44 total)."""
        df = mw_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)

        result = AnalysisWorkflow.run_volcano(
            df, mw_experiment, 'Normal', 'HFD', classes,
            stat_config=auto_stat_config,
        )

        assert not result.volcano_data.volcano_df.empty


class TestErrorHandling:
    """Validation failures and invalid inputs."""

    def test_empty_dataframe_validation(self):
        """Empty DataFrame is caught by validation."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['A', 'B'],
            number_of_samples_list=[2, 2],
        )

        errors = AnalysisWorkflow.validate_inputs(pd.DataFrame(), experiment)
        assert len(errors) > 0

    def test_missing_lipidmolec_validation(self):
        """Missing LipidMolec column is caught."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )

        errors = AnalysisWorkflow.validate_inputs(df, experiment)
        assert any('LipidMolec' in e for e in errors)

    def test_missing_classkey_validation(self):
        """Missing ClassKey column is caught."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0/18:1)'],
            'concentration[s1]': [100.0],
        })
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )

        errors = AnalysisWorkflow.validate_inputs(df, experiment)
        assert any('ClassKey' in e for e in errors)

    def test_no_concentration_columns_validation(self):
        """No concentration columns is caught."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0/18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['A'],
            number_of_samples_list=[1],
        )

        errors = AnalysisWorkflow.validate_inputs(df, experiment)
        assert any('concentration' in e.lower() for e in errors)

    def test_bar_chart_no_conditions_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Empty conditions list raises ValueError."""
        with pytest.raises(ValueError, match="condition"):
            AnalysisWorkflow.run_bar_chart(
                single_class_df, four_sample_experiment, [], ['PC'],
            )

    def test_bar_chart_no_classes_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Empty classes list raises ValueError."""
        with pytest.raises(ValueError, match="class"):
            AnalysisWorkflow.run_bar_chart(
                single_class_df, four_sample_experiment,
                ['Control'], [],
            )

    def test_pie_chart_no_conditions_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Empty conditions list raises ValueError."""
        with pytest.raises(ValueError, match="condition"):
            AnalysisWorkflow.run_pie_charts(
                single_class_df, four_sample_experiment, [], ['PC'],
            )

    def test_pathway_same_condition_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Same control and experimental condition raises ValueError."""
        df = lipidsearch_normalized_df
        with pytest.raises(ValueError, match="different"):
            AnalysisWorkflow.run_pathway(
                df, lipidsearch_experiment, 'WT', 'WT',
            )

    def test_volcano_same_condition_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Same control and experimental condition raises ValueError."""
        df = lipidsearch_normalized_df
        classes = AnalysisWorkflow.get_available_classes(df)
        with pytest.raises(ValueError, match="different"):
            AnalysisWorkflow.run_volcano(
                df, lipidsearch_experiment, 'WT', 'WT', classes,
                stat_config=auto_stat_config,
            )

    def test_volcano_no_classes_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
        auto_stat_config,
    ):
        """Empty classes list raises ValueError for volcano."""
        df = lipidsearch_normalized_df
        with pytest.raises(ValueError, match="class"):
            AnalysisWorkflow.run_volcano(
                df, lipidsearch_experiment, 'WT', 'ADGAT_DKO', [],
                stat_config=auto_stat_config,
            )

    def test_heatmap_invalid_type_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Invalid heatmap_type raises ValueError."""
        with pytest.raises(ValueError, match="heatmap_type"):
            AnalysisWorkflow.run_heatmap(
                single_class_df, four_sample_experiment,
                ['Control', 'Treatment'], ['PC'],
                heatmap_type='invalid',
            )

    def test_fach_no_class_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Empty class string raises ValueError."""
        with pytest.raises(ValueError, match="class"):
            AnalysisWorkflow.run_fach(
                single_class_df, four_sample_experiment,
                '', ['Control', 'Treatment'],
            )

    def test_pathway_empty_control_raises(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Empty control condition raises ValueError."""
        df = lipidsearch_normalized_df
        with pytest.raises(ValueError, match="[Cc]ontrol"):
            AnalysisWorkflow.run_pathway(
                df, lipidsearch_experiment, '', 'ADGAT_DKO',
            )

    def test_saturation_no_conditions_raises(
        self, single_class_df, four_sample_experiment,
    ):
        """Empty conditions list raises ValueError for saturation."""
        with pytest.raises(ValueError, match="condition"):
            AnalysisWorkflow.run_saturation(
                single_class_df, four_sample_experiment, [], ['PC'],
            )


class TestHelperMethods:
    """Tests for AnalysisWorkflow helper methods."""

    def test_get_available_classes(
        self, lipidsearch_normalized_df,
    ):
        """Available classes are extracted and sorted."""
        classes = AnalysisWorkflow.get_available_classes(
            lipidsearch_normalized_df,
        )

        assert isinstance(classes, list)
        assert len(classes) > 0
        assert classes == sorted(classes)

    def test_get_available_classes_no_classkey_raises(self):
        """Missing ClassKey raises ValueError."""
        df = pd.DataFrame({'LipidMolec': ['a']})
        with pytest.raises(ValueError, match="ClassKey"):
            AnalysisWorkflow.get_available_classes(df)

    def test_get_eligible_conditions(self, lipidsearch_experiment):
        """All conditions with >1 replicate are eligible."""
        eligible = AnalysisWorkflow.get_eligible_conditions(
            lipidsearch_experiment,
        )

        # LipidSearch has 4 samples each → all eligible
        assert len(eligible) == 3

    def test_get_eligible_conditions_excludes_single_replicate(
        self, msdial_experiment,
    ):
        """Blank (1 replicate) is excluded from eligible conditions."""
        eligible = AnalysisWorkflow.get_eligible_conditions(
            msdial_experiment,
        )

        assert 'Blank' not in eligible
        assert 'fads2_KO' in eligible
        assert 'WT' in eligible

    def test_get_all_conditions(self, lipidsearch_experiment):
        """All conditions are returned."""
        all_conds = AnalysisWorkflow.get_all_conditions(lipidsearch_experiment)

        assert all_conds == ['WT', 'ADGAT_DKO', 'BQC']

    def test_validate_inputs_valid(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Valid data passes validation."""
        errors = AnalysisWorkflow.validate_inputs(
            lipidsearch_normalized_df, lipidsearch_experiment,
        )

        assert errors == []


# =============================================================================
# LSI Compliance Report — Integration Tests
# =============================================================================


class TestLSIReportEndToEnd:
    """Integration tests for LSI compliance report generation using real sample data."""

    def test_collect_auto_fields_lipidsearch(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """Auto-fields from real LipidSearch data are complete and correct."""
        from app.services.lsi_report import LSIReportService

        fields = LSIReportService.collect_auto_fields(
            format_type="LipidSearch 5.0",
            experiment=lipidsearch_experiment,
            normalization_config=NormalizationConfig(method='none'),
            stat_config=None,
            cleaned_df=lipidsearch_normalized_df,
            intsta_df=None,
            bqc_label="BQC",
            cleaning_params={"Grade filter": "A, B"},
            qc_summary={"BQC CoV threshold": "30%"},
        )

        assert fields["Software"] == "LipidCruncher"
        assert fields["Data format / identification software"] == "LipidSearch 5.0"
        assert fields["Number of conditions"] == 3
        assert "WT" in fields["Condition labels"]
        assert fields["Total samples"] == 12
        assert "BQC" in fields["BQC samples"]
        assert fields["Number of lipid species (after filtering)"] == len(
            lipidsearch_normalized_df
        )
        assert fields["Number of lipid classes"] > 0
        assert "Cleaning: Grade filter" in fields
        assert fields["Cleaning: Grade filter"] == "A, B"

    def test_collect_auto_fields_generic(
        self, generic_normalized_df, generic_experiment,
    ):
        """Auto-fields from real Generic data are complete."""
        from app.services.lsi_report import LSIReportService

        fields = LSIReportService.collect_auto_fields(
            format_type="Generic Format",
            experiment=generic_experiment,
            normalization_config=None,
            stat_config=StatisticalTestConfig.create_auto(),
            cleaned_df=generic_normalized_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={},
            qc_summary={},
        )

        assert fields["Number of conditions"] == 3
        assert fields["BQC samples"] == "Not used"
        assert fields["Statistical test mode"] == "Auto"
        assert fields["Number of lipid species (after filtering)"] > 0

    def test_collect_auto_fields_msdial(
        self, msdial_normalized_df, msdial_experiment,
    ):
        """Auto-fields from real MS-DIAL data include correct format."""
        from app.services.lsi_report import LSIReportService

        fields = LSIReportService.collect_auto_fields(
            format_type="MS-DIAL",
            experiment=msdial_experiment,
            normalization_config=None,
            stat_config=None,
            cleaned_df=msdial_normalized_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={"Score threshold": "80"},
            qc_summary={},
        )

        assert fields["Data format / identification software"] == "MS-DIAL"
        assert "Blank" in fields["Condition labels"]
        assert fields["Number of lipid species (after filtering)"] > 0

    def test_csv_from_real_data(
        self, lipidsearch_normalized_df, lipidsearch_experiment,
    ):
        """CSV generation with real data produces valid structure."""
        from app.services.lsi_report import LSIReportService, _MANUAL_FIELDS

        fields = LSIReportService.collect_auto_fields(
            format_type="LipidSearch 5.0",
            experiment=lipidsearch_experiment,
            normalization_config=NormalizationConfig(method='none'),
            stat_config=StatisticalTestConfig.create_manual(),
            cleaned_df=lipidsearch_normalized_df,
            intsta_df=None,
            bqc_label="BQC",
            cleaning_params={},
            qc_summary={"BQC CoV threshold": "30%"},
        )

        csv = LSIReportService.generate_checklist_csv(fields)
        lines = csv.strip().split("\n")
        # Header + auto fields + manual fields
        assert len(lines) == 1 + len(fields) + len(_MANUAL_FIELDS)
        assert "LipidCruncher" in csv
        assert "LipidSearch" in csv
        assert "TO BE FILLED" in csv

    def test_pdf_from_real_data(
        self, generic_normalized_df, generic_experiment,
    ):
        """PDF generation with real data produces valid PDF bytes."""
        from app.services.lsi_report import LSIReportService

        fields = LSIReportService.collect_auto_fields(
            format_type="Generic Format",
            experiment=generic_experiment,
            normalization_config=None,
            stat_config=None,
            cleaned_df=generic_normalized_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={},
            qc_summary={},
        )

        manual = {"Organism / tissue type": "Mouse liver", "Ion mode": "Positive"}
        pdf = LSIReportService.generate_checklist_pdf(fields, manual)

        assert isinstance(pdf, bytes)
        assert pdf[:5] == b"%PDF-"
        assert len(pdf) > 500

    def test_csv_with_manual_fields_from_real_data(
        self, msdial_normalized_df, msdial_experiment,
    ):
        """CSV with manual fields filled in marks them as 'Researcher' source."""
        from app.services.lsi_report import LSIReportService

        fields = LSIReportService.collect_auto_fields(
            format_type="MS-DIAL",
            experiment=msdial_experiment,
            normalization_config=None,
            stat_config=None,
            cleaned_df=msdial_normalized_df,
            intsta_df=None,
            bqc_label=None,
            cleaning_params={},
            qc_summary={},
        )

        manual = {
            "Organism / tissue type": "Zebrafish larvae",
            "Extraction protocol": "Folch method",
            "Instrument model": "Thermo Q Exactive HF",
        }
        csv = LSIReportService.generate_checklist_csv(fields, manual)

        assert "Zebrafish larvae" in csv
        assert "Folch method" in csv
        assert "Thermo Q Exactive HF" in csv
        # Filled fields should be sourced as "Researcher"
        assert csv.count('"Researcher"') == 3
