"""
Unit tests for AnalysisWorkflow.

Tests the analysis orchestration layer:
validation → bar chart → pie chart → saturation → FACH →
pathway → volcano → heatmap

Comprehensive test coverage matching QualityCheckWorkflow depth.
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt

from app.workflows.analysis import (
    AnalysisWorkflow,
    AnalysisConfig,
    BarChartResult,
    PieChartResult,
    SaturationResult,
    FACHResult,
    PathwayResult,
    VolcanoResult,
    HeatmapResult,
)
from app.models.experiment import ExperimentConfig
from app.models.statistics import StatisticalTestConfig
from app.services.format_detection import DataFormat
from app.services.statistical_testing import StatisticalTestSummary


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================


@pytest.fixture
def exp_2x3():
    """2 conditions x 3 samples each = 6 total."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )


@pytest.fixture
def exp_3x3():
    """3 conditions x 3 samples each = 9 total."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Control', 'Treatment', 'Vehicle'],
        number_of_samples_list=[3, 3, 3],
    )


@pytest.fixture
def exp_2x2():
    """2 conditions x 2 samples each = 4 total."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2],
    )


@pytest.fixture
def exp_mixed():
    """3 conditions with mixed replicate counts."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Single', 'Pair', 'Triple'],
        number_of_samples_list=[1, 2, 3],
    )


@pytest.fixture
def exp_single_rep():
    """2 conditions x 1 sample each."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['A', 'B'],
        number_of_samples_list=[1, 1],
    )


# =============================================================================
# DataFrame Fixtures
# =============================================================================


def _make_conc_df(n_lipids=5, n_samples=6, classes=None, lipids=None,
                  seed=42):
    """Build a concentration DataFrame with realistic lipid names."""
    rng = np.random.RandomState(seed)
    default_classes = ['PC', 'PE', 'TG', 'SM', 'PI']
    if classes is None:
        classes = default_classes[:n_lipids]
    if lipids is None:
        lipids = [
            f'{cls} {16 + i}:{i}_{18}:{i + 1}' for i, cls in enumerate(classes)
        ]
    data = {
        'LipidMolec': lipids,
        'ClassKey': classes,
    }
    for s in range(1, n_samples + 1):
        data[f'concentration[s{s}]'] = rng.uniform(100, 5000, n_lipids)
    return pd.DataFrame(data)


@pytest.fixture
def basic_df(exp_2x3):
    """6-sample DataFrame with 5 lipids across 5 classes."""
    return _make_conc_df(n_lipids=5, n_samples=6)


@pytest.fixture
def multi_species_df(exp_2x3):
    """6-sample DataFrame with multiple species per class."""
    classes = ['PC', 'PC', 'PC', 'PE', 'PE', 'TG', 'TG', 'SM']
    lipids = [
        'PC 16:0_18:1', 'PC 16:0_18:2', 'PC 18:0_20:4',
        'PE 16:0_18:1', 'PE 18:0_18:2',
        'TG 16:0_18:1_18:2', 'TG 18:0_18:1_18:1',
        'SM 18:1;O2/16:0',
    ]
    return _make_conc_df(n_lipids=8, n_samples=6, classes=classes, lipids=lipids)


@pytest.fixture
def df_9_samples(exp_3x3):
    """9-sample DataFrame for 3-condition experiments."""
    classes = ['PC', 'PC', 'PE', 'PE', 'TG']
    lipids = [
        'PC 16:0_18:1', 'PC 18:0_20:4',
        'PE 16:0_18:1', 'PE 18:0_18:2',
        'TG 16:0_18:1_18:2',
    ]
    return _make_conc_df(n_lipids=5, n_samples=9, classes=classes, lipids=lipids)


@pytest.fixture
def df_4_samples(exp_2x2):
    """4-sample DataFrame for 2x2 experiments."""
    classes = ['PC', 'PE', 'TG']
    lipids = ['PC 16:0_18:1', 'PE 18:0_20:4', 'TG 16:0_18:1_18:2']
    return _make_conc_df(n_lipids=3, n_samples=4, classes=classes, lipids=lipids)


@pytest.fixture
def manual_stat_config():
    """Default manual-mode statistical test config."""
    return StatisticalTestConfig.create_manual()


@pytest.fixture
def auto_stat_config():
    """Auto-mode statistical test config."""
    return StatisticalTestConfig.create_auto()


# =============================================================================
# AnalysisConfig Tests
# =============================================================================


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_default_values(self):
        config = AnalysisConfig()
        assert config.format_type == DataFormat.GENERIC
        assert config.bqc_label is None

    def test_custom_values(self):
        config = AnalysisConfig(
            format_type=DataFormat.LIPIDSEARCH,
            bqc_label='BQC',
        )
        assert config.format_type == DataFormat.LIPIDSEARCH
        assert config.bqc_label == 'BQC'


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestResultDataclasses:
    """Tests for all result dataclasses."""

    def test_bar_chart_result_defaults(self):
        fig = go.Figure()
        df = pd.DataFrame()
        result = BarChartResult(figure=fig, abundance_df=df)
        assert result.figure is fig
        assert result.abundance_df is df
        assert result.stat_summary is None

    def test_bar_chart_result_with_stats(self):
        summary = StatisticalTestSummary()
        result = BarChartResult(
            figure=go.Figure(),
            abundance_df=pd.DataFrame(),
            stat_summary=summary,
        )
        assert result.stat_summary is summary

    def test_pie_chart_result_defaults(self):
        result = PieChartResult(
            figure=go.Figure(),
            data_df=pd.DataFrame(),
        )
        assert result.condition == ""

    def test_pie_chart_result_with_condition(self):
        result = PieChartResult(
            figure=go.Figure(),
            data_df=pd.DataFrame(),
            condition='Control',
        )
        assert result.condition == 'Control'

    def test_saturation_result_defaults(self):
        result = SaturationResult()
        assert result.plots == {}
        assert result.stat_summary is None
        assert result.consolidated_lipids == {}

    def test_fach_result_defaults(self):
        result = FACHResult()
        assert result.figure is None
        assert result.data_dict == {}
        assert result.unparsable_lipids == []
        assert result.weighted_averages == {}

    def test_pathway_result_defaults(self):
        result = PathwayResult()
        assert result.figure is None
        assert result.pathway_dict == {}
        assert isinstance(result.fold_change_df, pd.DataFrame)
        assert isinstance(result.saturation_df, pd.DataFrame)

    def test_volcano_result_defaults(self):
        result = VolcanoResult()
        assert result.figure is None
        assert result.volcano_data is None
        assert result.concentration_plot is None
        assert result.concentration_df is None
        assert result.stat_summary is None

    def test_heatmap_result_defaults(self):
        result = HeatmapResult()
        assert result.figure is None
        assert result.z_scores_df is None
        assert result.cluster_composition is None


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidateInputs:
    """Tests for AnalysisWorkflow.validate_inputs."""

    def test_valid_inputs(self, basic_df, exp_2x3):
        errors = AnalysisWorkflow.validate_inputs(basic_df, exp_2x3)
        assert errors == []

    def test_none_df(self, exp_2x3):
        errors = AnalysisWorkflow.validate_inputs(None, exp_2x3)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_empty_df(self, exp_2x3):
        errors = AnalysisWorkflow.validate_inputs(pd.DataFrame(), exp_2x3)
        assert len(errors) == 1
        assert 'empty' in errors[0].lower()

    def test_missing_lipidmolec(self, exp_2x3):
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'concentration[s1]': [100.0],
        })
        errors = AnalysisWorkflow.validate_inputs(df, exp_2x3)
        assert any('LipidMolec' in e for e in errors)

    def test_missing_classkey(self, exp_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0'],
            'concentration[s1]': [100.0],
        })
        errors = AnalysisWorkflow.validate_inputs(df, exp_2x3)
        assert any('ClassKey' in e for e in errors)

    def test_no_concentration_columns(self, exp_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
        })
        errors = AnalysisWorkflow.validate_inputs(df, exp_2x3)
        assert any('concentration' in e.lower() for e in errors)

    def test_mismatched_sample_labels(self, exp_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0'],
            'ClassKey': ['PC'],
            'concentration[x1]': [100.0],
        })
        errors = AnalysisWorkflow.validate_inputs(df, exp_2x3)
        assert any('match' in e.lower() for e in errors)

    def test_multiple_errors_collected(self, exp_2x3):
        df = pd.DataFrame({
            'other': [1],
            'concentration[s1]': [100.0],
        })
        errors = AnalysisWorkflow.validate_inputs(df, exp_2x3)
        assert len(errors) >= 2


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestGetAvailableClasses:
    """Tests for AnalysisWorkflow.get_available_classes."""

    def test_basic(self, basic_df):
        classes = AnalysisWorkflow.get_available_classes(basic_df)
        assert isinstance(classes, list)
        assert len(classes) == 5
        assert classes == sorted(classes)

    def test_deduplication(self, multi_species_df):
        classes = AnalysisWorkflow.get_available_classes(multi_species_df)
        assert len(classes) == len(set(classes))

    def test_missing_classkey(self):
        df = pd.DataFrame({'LipidMolec': ['PC 16:0']})
        with pytest.raises(ValueError, match='ClassKey'):
            AnalysisWorkflow.get_available_classes(df)

    def test_nan_values_excluded(self):
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0', 'PE 18:0', 'TG 16:0'],
            'ClassKey': ['PC', np.nan, 'TG'],
        })
        classes = AnalysisWorkflow.get_available_classes(df)
        assert 'PC' in classes
        assert 'TG' in classes
        assert len(classes) == 2


class TestGetEligibleConditions:
    """Tests for AnalysisWorkflow.get_eligible_conditions."""

    def test_all_multi_replicate(self, exp_2x3):
        eligible = AnalysisWorkflow.get_eligible_conditions(exp_2x3)
        assert eligible == ['Control', 'Treatment']

    def test_mixed_replicates(self, exp_mixed):
        eligible = AnalysisWorkflow.get_eligible_conditions(exp_mixed)
        assert 'Single' not in eligible
        assert 'Pair' in eligible
        assert 'Triple' in eligible

    def test_single_replicate_excluded(self, exp_single_rep):
        eligible = AnalysisWorkflow.get_eligible_conditions(exp_single_rep)
        assert eligible == []


class TestGetAllConditions:
    """Tests for AnalysisWorkflow.get_all_conditions."""

    def test_returns_all(self, exp_3x3):
        conds = AnalysisWorkflow.get_all_conditions(exp_3x3)
        assert conds == ['Control', 'Treatment', 'Vehicle']

    def test_preserves_order(self, exp_mixed):
        conds = AnalysisWorkflow.get_all_conditions(exp_mixed)
        assert conds == ['Single', 'Pair', 'Triple']


class TestGetSamplesForCondition:
    """Tests for AnalysisWorkflow._get_samples_for_condition."""

    def test_returns_correct_samples(self, exp_2x3):
        samples = AnalysisWorkflow._get_samples_for_condition(
            exp_2x3, 'Control',
        )
        assert samples == ['s1', 's2', 's3']

    def test_second_condition(self, exp_2x3):
        samples = AnalysisWorkflow._get_samples_for_condition(
            exp_2x3, 'Treatment',
        )
        assert samples == ['s4', 's5', 's6']

    def test_invalid_condition(self, exp_2x3):
        with pytest.raises(ValueError, match='not found'):
            AnalysisWorkflow._get_samples_for_condition(
                exp_2x3, 'NonExistent',
            )


# =============================================================================
# Bar Chart Tests
# =============================================================================


class TestRunBarChart:
    """Tests for AnalysisWorkflow.run_bar_chart."""

    def test_basic_without_stats(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE', 'TG'],
        )
        assert isinstance(result, BarChartResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.abundance_df, pd.DataFrame)
        assert result.stat_summary is None

    def test_with_stats(self, multi_species_df, exp_2x3, manual_stat_config):
        result = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is not None
        assert isinstance(result.stat_summary, StatisticalTestSummary)

    def test_log10_scale(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            scale='log10',
        )
        assert isinstance(result.figure, go.Figure)

    def test_single_condition_no_stats(self, basic_df, exp_2x3, manual_stat_config):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3,
            selected_conditions=['Control'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is None

    def test_empty_conditions_raises(self, basic_df, exp_2x3):
        with pytest.raises(ValueError, match='condition'):
            AnalysisWorkflow.run_bar_chart(
                basic_df, exp_2x3, [], ['PC'],
            )

    def test_empty_classes_raises(self, basic_df, exp_2x3):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_bar_chart(
                basic_df, exp_2x3, ['Control'], [],
            )

    def test_abundance_df_has_expected_columns(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
        )
        assert 'ClassKey' in result.abundance_df.columns

    def test_three_conditions(self, df_9_samples, exp_3x3, manual_stat_config):
        result = AnalysisWorkflow.run_bar_chart(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is not None


class TestRunBarChartEdgeCases:
    """Edge cases for bar chart analysis."""

    def test_single_class(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
        )
        assert isinstance(result.figure, go.Figure)

    def test_auto_stat_config(self, multi_species_df, exp_2x3, auto_stat_config):
        result = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=auto_stat_config,
        )
        assert result.stat_summary is not None


# =============================================================================
# Pie Chart Tests
# =============================================================================


class TestRunPieCharts:
    """Tests for AnalysisWorkflow.run_pie_charts."""

    def test_basic(self, basic_df, exp_2x3):
        results = AnalysisWorkflow.run_pie_charts(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE', 'TG'],
        )
        assert isinstance(results, dict)
        assert 'Control' in results
        assert 'Treatment' in results
        assert len(results) == 2

    def test_result_structure(self, basic_df, exp_2x3):
        results = AnalysisWorkflow.run_pie_charts(
            basic_df, exp_2x3,
            selected_conditions=['Control'],
            selected_classes=['PC', 'PE'],
        )
        result = results['Control']
        assert isinstance(result, PieChartResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.data_df, pd.DataFrame)
        assert result.condition == 'Control'

    def test_single_condition(self, basic_df, exp_2x3):
        results = AnalysisWorkflow.run_pie_charts(
            basic_df, exp_2x3,
            selected_conditions=['Treatment'],
            selected_classes=['PC', 'PE'],
        )
        assert len(results) == 1
        assert 'Treatment' in results

    def test_empty_conditions_raises(self, basic_df, exp_2x3):
        with pytest.raises(ValueError, match='condition'):
            AnalysisWorkflow.run_pie_charts(
                basic_df, exp_2x3, [], ['PC'],
            )

    def test_empty_classes_raises(self, basic_df, exp_2x3):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_pie_charts(
                basic_df, exp_2x3, ['Control'], [],
            )

    def test_skips_unknown_condition(self, basic_df, exp_2x3):
        results = AnalysisWorkflow.run_pie_charts(
            basic_df, exp_2x3,
            selected_conditions=['Control', 'Unknown'],
            selected_classes=['PC', 'PE'],
        )
        assert 'Control' in results
        assert 'Unknown' not in results

    def test_three_conditions(self, df_9_samples, exp_3x3):
        results = AnalysisWorkflow.run_pie_charts(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
        )
        assert len(results) == 3


# =============================================================================
# Saturation Tests
# =============================================================================


class TestRunSaturation:
    """Tests for AnalysisWorkflow.run_saturation."""

    def test_basic_concentration(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
        )
        assert isinstance(result, SaturationResult)
        assert isinstance(result.plots, dict)
        assert result.stat_summary is None

    def test_with_stats(self, multi_species_df, exp_2x3, manual_stat_config):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
            plot_type='concentration',
        )
        assert result.stat_summary is not None

    def test_percentage_mode_no_stats(self, multi_species_df, exp_2x3,
                                      manual_stat_config):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
            plot_type='percentage',
        )
        assert result.stat_summary is None

    def test_consolidated_lipids_returned(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
        )
        assert isinstance(result.consolidated_lipids, dict)

    def test_single_condition_no_stats(self, multi_species_df, exp_2x3,
                                       manual_stat_config):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control'],
            selected_classes=['PC'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is None

    def test_empty_conditions_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='condition'):
            AnalysisWorkflow.run_saturation(
                multi_species_df, exp_2x3, [], ['PC'],
            )

    def test_empty_classes_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_saturation(
                multi_species_df, exp_2x3, ['Control'], [],
            )

    def test_plots_per_class(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
        )
        for cls, fig in result.plots.items():
            assert isinstance(fig, go.Figure)

    def test_show_significance_flag(self, multi_species_df, exp_2x3,
                                     manual_stat_config):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
            stat_config=manual_stat_config,
            show_significance=True,
        )
        assert isinstance(result, SaturationResult)


# =============================================================================
# FACH Tests
# =============================================================================


class TestRunFACH:
    """Tests for AnalysisWorkflow.run_fach."""

    def test_basic(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3,
            selected_class='PC',
            selected_conditions=['Control', 'Treatment'],
        )
        assert isinstance(result, FACHResult)
        assert isinstance(result.data_dict, dict)
        assert isinstance(result.unparsable_lipids, list)
        assert isinstance(result.weighted_averages, dict)

    def test_figure_created(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3,
            selected_class='PC',
            selected_conditions=['Control', 'Treatment'],
        )
        # Figure may be None if not enough parsable data
        if result.figure is not None:
            assert isinstance(result.figure, go.Figure)

    def test_empty_class_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_fach(
                multi_species_df, exp_2x3, '', ['Control'],
            )

    def test_empty_conditions_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='condition'):
            AnalysisWorkflow.run_fach(
                multi_species_df, exp_2x3, 'PC', [],
            )

    def test_single_condition(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3,
            selected_class='PC',
            selected_conditions=['Control'],
        )
        assert isinstance(result, FACHResult)

    def test_weighted_averages_per_condition(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3,
            selected_class='PC',
            selected_conditions=['Control', 'Treatment'],
        )
        for cond, (avg_c, avg_db) in result.weighted_averages.items():
            assert isinstance(avg_c, float)
            assert isinstance(avg_db, float)


# =============================================================================
# Pathway Tests
# =============================================================================


class TestRunPathway:
    """Tests for AnalysisWorkflow.run_pathway."""

    def test_basic(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
        )
        assert isinstance(result, PathwayResult)
        assert isinstance(result.pathway_dict, dict)
        assert isinstance(result.fold_change_df, pd.DataFrame)
        assert isinstance(result.saturation_df, pd.DataFrame)
        plt.close('all')

    def test_figure_type(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
        )
        if result.figure is not None:
            assert isinstance(result.figure, go.Figure)

    def test_empty_control_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='Control'):
            AnalysisWorkflow.run_pathway(
                multi_species_df, exp_2x3, '', 'Treatment',
            )

    def test_empty_experimental_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='Experimental'):
            AnalysisWorkflow.run_pathway(
                multi_species_df, exp_2x3, 'Control', '',
            )

    def test_same_conditions_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='different'):
            AnalysisWorkflow.run_pathway(
                multi_species_df, exp_2x3, 'Control', 'Control',
            )

    def test_fold_change_df_not_empty(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
        )
        assert not result.fold_change_df.empty
        plt.close('all')

    def test_pathway_dict_structure(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
        )
        if result.pathway_dict:
            assert 'class' in result.pathway_dict
            assert 'abundance ratio' in result.pathway_dict
            assert 'saturated fatty acids ratio' in result.pathway_dict
        plt.close('all')


# =============================================================================
# Volcano Tests
# =============================================================================


class TestRunVolcano:
    """Tests for AnalysisWorkflow.run_volcano."""

    def test_basic(self, multi_species_df, exp_2x3, manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert isinstance(result, VolcanoResult)
        assert isinstance(result.figure, go.Figure)
        assert result.volcano_data is not None
        assert result.stat_summary is not None

    def test_concentration_plot_generated(self, multi_species_df, exp_2x3,
                                          manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC'],
            stat_config=manual_stat_config,
        )
        assert isinstance(result.concentration_plot, go.Figure)
        assert isinstance(result.concentration_df, pd.DataFrame)

    def test_empty_control_raises(self, multi_species_df, exp_2x3,
                                   manual_stat_config):
        with pytest.raises(ValueError, match='Control'):
            AnalysisWorkflow.run_volcano(
                multi_species_df, exp_2x3, '', 'Treatment',
                ['PC'], manual_stat_config,
            )

    def test_empty_experimental_raises(self, multi_species_df, exp_2x3,
                                        manual_stat_config):
        with pytest.raises(ValueError, match='Experimental'):
            AnalysisWorkflow.run_volcano(
                multi_species_df, exp_2x3, 'Control', '',
                ['PC'], manual_stat_config,
            )

    def test_same_conditions_raises(self, multi_species_df, exp_2x3,
                                     manual_stat_config):
        with pytest.raises(ValueError, match='different'):
            AnalysisWorkflow.run_volcano(
                multi_species_df, exp_2x3, 'Control', 'Control',
                ['PC'], manual_stat_config,
            )

    def test_empty_classes_raises(self, multi_species_df, exp_2x3,
                                   manual_stat_config):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_volcano(
                multi_species_df, exp_2x3, 'Control', 'Treatment',
                [], manual_stat_config,
            )

    def test_no_data_for_class_raises(self, multi_species_df, exp_2x3,
                                       manual_stat_config):
        with pytest.raises(ValueError, match='No data'):
            AnalysisWorkflow.run_volcano(
                multi_species_df, exp_2x3, 'Control', 'Treatment',
                ['NonExistentClass'], manual_stat_config,
            )

    def test_custom_thresholds(self, multi_species_df, exp_2x3,
                                manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
            p_threshold=0.01,
            fc_threshold=2.0,
        )
        assert isinstance(result, VolcanoResult)

    def test_hide_non_significant(self, multi_species_df, exp_2x3,
                                   manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC'],
            stat_config=manual_stat_config,
            hide_non_sig=True,
        )
        assert isinstance(result.figure, go.Figure)

    def test_top_n_labels(self, multi_species_df, exp_2x3,
                           manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC'],
            stat_config=manual_stat_config,
            top_n_labels=5,
        )
        assert isinstance(result.figure, go.Figure)

    def test_auto_stat_config(self, multi_species_df, exp_2x3,
                               auto_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC', 'PE'],
            stat_config=auto_stat_config,
        )
        assert result.stat_summary is not None


class TestRunVolcanoEdgeCases:
    """Edge cases for volcano analysis."""

    def test_single_class(self, multi_species_df, exp_2x3,
                           manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['SM'],
            stat_config=manual_stat_config,
        )
        assert isinstance(result, VolcanoResult)

    def test_volcano_data_has_expected_fields(self, multi_species_df, exp_2x3,
                                               manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3,
            control='Control',
            experimental='Treatment',
            selected_classes=['PC'],
            stat_config=manual_stat_config,
        )
        assert hasattr(result.volcano_data, 'volcano_df')
        assert hasattr(result.volcano_data, 'removed_lipids_df')


# =============================================================================
# Heatmap Tests
# =============================================================================


class TestRunHeatmap:
    """Tests for AnalysisWorkflow.run_heatmap."""

    def test_regular_heatmap(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            heatmap_type='regular',
        )
        assert isinstance(result, HeatmapResult)
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.z_scores_df, pd.DataFrame)
        assert result.cluster_composition is None

    def test_clustered_heatmap(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            heatmap_type='clustered',
            n_clusters=2,
        )
        assert isinstance(result, HeatmapResult)
        assert isinstance(result.figure, go.Figure)
        assert result.cluster_composition is not None
        assert isinstance(result.cluster_composition, pd.DataFrame)

    def test_empty_conditions_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='condition'):
            AnalysisWorkflow.run_heatmap(
                multi_species_df, exp_2x3, [], ['PC'],
            )

    def test_empty_classes_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='class'):
            AnalysisWorkflow.run_heatmap(
                multi_species_df, exp_2x3, ['Control'], [],
            )

    def test_invalid_heatmap_type_raises(self, multi_species_df, exp_2x3):
        with pytest.raises(ValueError, match='heatmap_type'):
            AnalysisWorkflow.run_heatmap(
                multi_species_df, exp_2x3,
                ['Control'], ['PC'],
                heatmap_type='invalid',
            )

    def test_z_scores_shape(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
            heatmap_type='regular',
        )
        assert not result.z_scores_df.empty

    def test_custom_n_clusters(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE', 'TG'],
            heatmap_type='clustered',
            n_clusters=3,
        )
        assert result.cluster_composition is not None


class TestRunHeatmapEdgeCases:
    """Edge cases for heatmap analysis."""

    def test_single_condition(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control'],
            selected_classes=['PC', 'PE'],
            heatmap_type='regular',
        )
        assert isinstance(result.figure, go.Figure)

    def test_single_class(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
            heatmap_type='regular',
        )
        assert isinstance(result.figure, go.Figure)


# =============================================================================
# Cross-Analysis Tests
# =============================================================================


class TestCrossAnalysis:
    """Tests for consistency across analysis types."""

    def test_bar_and_pie_use_same_data(self, multi_species_df, exp_2x3):
        bar_result = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
        )
        pie_results = AnalysisWorkflow.run_pie_charts(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
        )
        assert isinstance(bar_result, BarChartResult)
        assert len(pie_results) == 2

    def test_all_analyses_accept_same_df(self, multi_species_df, exp_2x3,
                                          manual_stat_config):
        """All analysis types work with the same input DataFrame."""
        bar = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(bar, BarChartResult)

        pies = AnalysisWorkflow.run_pie_charts(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(pies, dict)

        sat = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(sat, SaturationResult)

        fach = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3, 'PC', ['Control', 'Treatment'],
        )
        assert isinstance(fach, FACHResult)

        pathway = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
        )
        assert isinstance(pathway, PathwayResult)
        plt.close('all')

        volcano = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
            ['PC', 'PE'], manual_stat_config,
        )
        assert isinstance(volcano, VolcanoResult)

        heatmap = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(heatmap, HeatmapResult)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_lipid_bar_chart(self, exp_2x3):
        df = _make_conc_df(n_lipids=1, n_samples=6,
                           classes=['PC'], lipids=['PC 16:0_18:1'])
        result = AnalysisWorkflow.run_bar_chart(
            df, exp_2x3, ['Control', 'Treatment'], ['PC'],
        )
        assert isinstance(result, BarChartResult)

    def test_single_lipid_pie_chart(self, exp_2x3):
        df = _make_conc_df(n_lipids=1, n_samples=6,
                           classes=['PC'], lipids=['PC 16:0_18:1'])
        result = AnalysisWorkflow.run_pie_charts(
            df, exp_2x3, ['Control'], ['PC'],
        )
        assert len(result) == 1

    def test_many_classes_bar_chart(self, exp_2x3):
        classes = [f'Class{i}' for i in range(15)]
        lipids = [f'Class{i} {16 + i}:0' for i in range(15)]
        df = _make_conc_df(n_lipids=15, n_samples=6,
                           classes=classes, lipids=lipids)
        result = AnalysisWorkflow.run_bar_chart(
            df, exp_2x3, ['Control', 'Treatment'], classes,
        )
        assert isinstance(result, BarChartResult)

    def test_zero_values_in_data(self, exp_2x3):
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': [0.0, 0.0],
            'concentration[s2]': [100.0, 200.0],
            'concentration[s3]': [0.0, 0.0],
            'concentration[s4]': [150.0, 250.0],
            'concentration[s5]': [0.0, 0.0],
            'concentration[s6]': [0.0, 0.0],
        })
        result = AnalysisWorkflow.run_bar_chart(
            df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(result, BarChartResult)

    def test_string_numeric_values(self, exp_2x3):
        """DataFrame with string-encoded numeric values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'ClassKey': ['PC', 'PE'],
            'concentration[s1]': ['1000.0', '2000.0'],
            'concentration[s2]': ['1100.0', '2100.0'],
            'concentration[s3]': ['1200.0', '2200.0'],
            'concentration[s4]': ['1300.0', '2300.0'],
            'concentration[s5]': ['1400.0', '2400.0'],
            'concentration[s6]': ['1500.0', '2500.0'],
        })
        # Convert strings to float
        for col in df.columns:
            if col.startswith('concentration['):
                df[col] = pd.to_numeric(df[col])
        result = AnalysisWorkflow.run_bar_chart(
            df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert isinstance(result, BarChartResult)

    def test_validate_inputs_returns_list(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.validate_inputs(basic_df, exp_2x3)
        assert isinstance(result, list)

    def test_pathway_with_no_matching_classes(self, exp_2x3):
        """Pathway with classes not in the pathway diagram."""
        classes = ['CustomClass1', 'CustomClass2']
        lipids = ['CustomClass1 16:0_18:1', 'CustomClass2 18:0_20:4']
        df = _make_conc_df(n_lipids=2, n_samples=6,
                           classes=classes, lipids=lipids)
        result = AnalysisWorkflow.run_pathway(
            df, exp_2x3, 'Control', 'Treatment',
        )
        assert isinstance(result, PathwayResult)
        plt.close('all')


# =============================================================================
# Type Coercion Tests
# =============================================================================


class TestTypeCoercion:
    """Tests that results contain correctly typed values."""

    def test_bar_chart_abundance_df_is_dataframe(self, basic_df, exp_2x3):
        result = AnalysisWorkflow.run_bar_chart(
            basic_df, exp_2x3, ['Control', 'Treatment'], ['PC'],
        )
        assert isinstance(result.abundance_df, pd.DataFrame)

    def test_pie_chart_data_df_is_dataframe(self, basic_df, exp_2x3):
        results = AnalysisWorkflow.run_pie_charts(
            basic_df, exp_2x3, ['Control'], ['PC', 'PE'],
        )
        result = results['Control']
        assert isinstance(result.data_df, pd.DataFrame)

    def test_saturation_plots_are_figures(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC'],
        )
        for fig in result.plots.values():
            assert isinstance(fig, go.Figure)

    def test_fach_data_dict_values_are_dataframes(self, multi_species_df,
                                                    exp_2x3):
        result = AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3, 'PC', ['Control', 'Treatment'],
        )
        for cond, cond_df in result.data_dict.items():
            assert isinstance(cond, str)
            assert isinstance(cond_df, pd.DataFrame)

    def test_volcano_result_types(self, multi_species_df, exp_2x3,
                                   manual_stat_config):
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
            ['PC'], manual_stat_config,
        )
        assert isinstance(result.figure, go.Figure)
        assert isinstance(result.concentration_plot, go.Figure)
        assert isinstance(result.concentration_df, pd.DataFrame)
        assert isinstance(result.stat_summary, StatisticalTestSummary)

    def test_heatmap_z_scores_are_float(self, multi_species_df, exp_2x3):
        result = AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        assert result.z_scores_df.dtypes.apply(
            lambda dt: np.issubdtype(dt, np.floating)
        ).all()

    def test_pathway_fold_change_is_dataframe(self, multi_species_df,
                                               exp_2x3):
        result = AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
        )
        assert isinstance(result.fold_change_df, pd.DataFrame)
        assert isinstance(result.saturation_df, pd.DataFrame)
        plt.close('all')


# =============================================================================
# Immutability Tests
# =============================================================================


class TestImmutability:
    """Tests that workflow methods do not mutate input data."""

    def test_bar_chart_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        pd.testing.assert_frame_equal(multi_species_df, original)

    def test_pie_chart_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_pie_charts(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        pd.testing.assert_frame_equal(multi_species_df, original)

    def test_saturation_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        pd.testing.assert_frame_equal(multi_species_df, original)

    def test_fach_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_fach(
            multi_species_df, exp_2x3, 'PC', ['Control', 'Treatment'],
        )
        pd.testing.assert_frame_equal(multi_species_df, original)

    def test_pathway_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_pathway(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
        )
        pd.testing.assert_frame_equal(multi_species_df, original)
        plt.close('all')

    def test_volcano_preserves_input(self, multi_species_df, exp_2x3,
                                      manual_stat_config):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
            ['PC', 'PE'], manual_stat_config,
        )
        pd.testing.assert_frame_equal(multi_species_df, original)

    def test_heatmap_preserves_input(self, multi_species_df, exp_2x3):
        original = multi_species_df.copy()
        AnalysisWorkflow.run_heatmap(
            multi_species_df, exp_2x3, ['Control', 'Treatment'], ['PC', 'PE'],
        )
        pd.testing.assert_frame_equal(multi_species_df, original)


# =============================================================================
# Three-Condition Tests
# =============================================================================


class TestThreeConditions:
    """Tests with 3+ conditions to exercise multi-group paths."""

    def test_bar_chart_three_conditions(self, df_9_samples, exp_3x3,
                                         manual_stat_config):
        result = AnalysisWorkflow.run_bar_chart(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is not None

    def test_pie_charts_three_conditions(self, df_9_samples, exp_3x3):
        results = AnalysisWorkflow.run_pie_charts(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
        )
        assert len(results) == 3

    def test_saturation_three_conditions(self, df_9_samples, exp_3x3,
                                          manual_stat_config):
        result = AnalysisWorkflow.run_saturation(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
            stat_config=manual_stat_config,
        )
        assert result.stat_summary is not None

    def test_fach_three_conditions(self, df_9_samples, exp_3x3):
        result = AnalysisWorkflow.run_fach(
            df_9_samples, exp_3x3,
            selected_class='PC',
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
        )
        assert isinstance(result, FACHResult)

    def test_heatmap_three_conditions(self, df_9_samples, exp_3x3):
        result = AnalysisWorkflow.run_heatmap(
            df_9_samples, exp_3x3,
            selected_conditions=['Control', 'Treatment', 'Vehicle'],
            selected_classes=['PC', 'PE'],
        )
        assert isinstance(result, HeatmapResult)


# =============================================================================
# Statistical Config Variation Tests
# =============================================================================


class TestStatisticalConfigVariations:
    """Tests with different statistical configurations."""

    def test_bar_chart_non_parametric(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
            correction_method='bonferroni',
        )
        result = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=config,
        )
        assert result.stat_summary is not None

    def test_volcano_non_parametric(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
            correction_method='bonferroni',
        )
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
            ['PC', 'PE'], config,
        )
        assert result.stat_summary is not None

    def test_bar_chart_uncorrected(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_manual(
            correction_method='uncorrected',
            posthoc_correction='uncorrected',
        )
        result = AnalysisWorkflow.run_bar_chart(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC', 'PE'],
            stat_config=config,
        )
        assert result.stat_summary is not None

    def test_volcano_uncorrected(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_manual(
            correction_method='uncorrected',
        )
        result = AnalysisWorkflow.run_volcano(
            multi_species_df, exp_2x3, 'Control', 'Treatment',
            ['PC'], config,
        )
        assert result.stat_summary is not None

    def test_saturation_non_parametric(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_manual(
            test_type='non_parametric',
        )
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
            stat_config=config,
        )
        assert result.stat_summary is not None

    def test_saturation_auto_config(self, multi_species_df, exp_2x3):
        config = StatisticalTestConfig.create_auto()
        result = AnalysisWorkflow.run_saturation(
            multi_species_df, exp_2x3,
            selected_conditions=['Control', 'Treatment'],
            selected_classes=['PC'],
            stat_config=config,
        )
        assert result.stat_summary is not None
