"""
End-to-end integration tests: raw CSV → Module 1 → Module 2 → Module 3.

Tests the complete pipeline across all three modules to verify state
passing, column renaming, and cross-module consistency. These are the
only tests that exercise the full data flow in a single test function.
"""

import pytest
import pandas as pd
import numpy as np

import plotly.graph_objects as go

from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig
from app.models.statistics import StatisticalTestConfig
from app.services.format_detection import DataFormat
from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
)
from app.workflows.normalization import (
    NormalizationWorkflow,
    NormalizationWorkflowConfig,
)
from app.workflows.quality_check import (
    QualityCheckWorkflow,
    QualityCheckConfig,
)
from app.workflows.analysis import (
    AnalysisWorkflow,
    BarChartResult,
    PieChartResult,
    VolcanoResult,
    HeatmapResult,
)

from tests.integration.conftest import (
    load_lipidsearch_sample,
    load_generic_sample,
)


# =============================================================================
# Pipeline helpers
# =============================================================================


def run_module1(raw_df, experiment, data_format):
    """Module 1: Ingestion + normalization (method='none')."""
    ingestion_config = IngestionConfig(
        experiment=experiment,
        data_format=data_format,
        apply_zero_filter=True,
    )
    ingestion_result = DataIngestionWorkflow.run(raw_df, ingestion_config)
    assert ingestion_result.is_valid, (
        f"Ingestion failed: {ingestion_result.validation_errors}"
    )

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


def run_module2_qc(normalized_df, experiment, bqc_label=None, data_format=DataFormat.GENERIC):
    """Module 2: QC pipeline — box plots, BQC, correlation, PCA."""
    # Box plots
    box_result = QualityCheckWorkflow.run_box_plots(normalized_df, experiment)
    assert box_result is not None
    assert len(box_result.available_samples) > 0

    # BQC (if applicable)
    qc_df = normalized_df
    config = QualityCheckConfig(
        bqc_label=bqc_label,
        format_type=data_format,
        cov_threshold=30.0,
    )
    if bqc_label and bqc_label in experiment.conditions_list:
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            normalized_df, experiment, config,
        )
        assert bqc_result is not None
        # Apply BQC filter
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            normalized_df,
            bqc_result.high_cov_lipids,
            lipids_to_keep=[],
        )
        qc_df = filter_result.filtered_df

    # Retention time
    rt_result = QualityCheckWorkflow.check_retention_time_availability(qc_df, config)

    # PCA
    pca_result = QualityCheckWorkflow.run_pca(qc_df, experiment)
    assert pca_result is not None
    assert 'PC1' in pca_result.pc_df.columns
    assert 'PC2' in pca_result.pc_df.columns

    return qc_df


def run_module3_analyses(qc_df, experiment):
    """Module 3: Run bar chart, pie chart, volcano, and heatmap."""
    conditions = AnalysisWorkflow.get_eligible_conditions(experiment)
    all_classes = AnalysisWorkflow.get_available_classes(qc_df)
    stat_config = StatisticalTestConfig.create_auto()

    results = {}

    # Bar chart
    bar_result = AnalysisWorkflow.run_bar_chart(
        qc_df, experiment, conditions, all_classes, stat_config=stat_config,
    )
    assert isinstance(bar_result, BarChartResult)
    assert isinstance(bar_result.figure, go.Figure)
    assert not bar_result.abundance_df.empty
    results['bar'] = bar_result

    # Pie charts
    all_conditions = AnalysisWorkflow.get_all_conditions(experiment)
    pie_results = AnalysisWorkflow.run_pie_charts(
        qc_df, experiment, all_conditions, all_classes,
    )
    assert len(pie_results) > 0
    for cond, pie_result in pie_results.items():
        assert isinstance(pie_result, PieChartResult)
        assert isinstance(pie_result.figure, go.Figure)
    results['pie'] = pie_results

    # Volcano (requires exactly 2 conditions with >1 sample)
    if len(conditions) >= 2:
        volcano_result = AnalysisWorkflow.run_volcano(
            qc_df, experiment,
            control=conditions[0],
            experimental=conditions[1],
            selected_classes=all_classes,
            stat_config=StatisticalTestConfig.create_manual(
                test_type='parametric',
                correction_method='fdr_bh',
            ),
        )
        assert isinstance(volcano_result, VolcanoResult)
        if volcano_result.figure is not None:
            assert isinstance(volcano_result.figure, go.Figure)
        results['volcano'] = volcano_result

    # Heatmap
    heatmap_result = AnalysisWorkflow.run_heatmap(
        qc_df, experiment, all_conditions, all_classes,
        heatmap_type='clustered', n_clusters=3,
    )
    assert isinstance(heatmap_result, HeatmapResult)
    if heatmap_result.figure is not None:
        assert isinstance(heatmap_result.figure, go.Figure)
    results['heatmap'] = heatmap_result

    return results


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEndLipidSearch:
    """Full pipeline with LipidSearch sample data: Module 1 → 2 → 3."""

    def test_full_pipeline(self):
        """Raw CSV → ingestion → normalization → QC → analysis."""
        # Module 1
        raw_df = load_lipidsearch_sample()
        experiment = ExperimentConfig(
            n_conditions=3,
            conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
            number_of_samples_list=[4, 4, 4],
        )
        normalized_df = run_module1(raw_df, experiment, DataFormat.LIPIDSEARCH)

        # Verify Module 1 output
        assert not normalized_df.empty
        assert 'LipidMolec' in normalized_df.columns
        assert 'ClassKey' in normalized_df.columns
        conc_cols = [c for c in normalized_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) == 12  # 4+4+4

        # Module 2
        qc_df = run_module2_qc(
            normalized_df, experiment, bqc_label='BQC',
            data_format=DataFormat.LIPIDSEARCH,
        )

        # Verify Module 2 output
        assert not qc_df.empty
        assert len(qc_df) <= len(normalized_df)  # BQC filtering may remove lipids
        assert 'LipidMolec' in qc_df.columns

        # Module 3
        results = run_module3_analyses(qc_df, experiment)

        # Verify cross-module consistency
        bar_classes = set(results['bar'].abundance_df['ClassKey'].tolist())
        data_classes = set(qc_df['ClassKey'].unique().tolist())
        assert bar_classes.issubset(data_classes)


class TestEndToEndGeneric:
    """Full pipeline with Generic sample data: Module 1 → 2 → 3."""

    def test_full_pipeline(self):
        """Raw CSV → ingestion → normalization → QC → analysis."""
        raw_df = load_generic_sample()
        experiment = ExperimentConfig(
            n_conditions=3,
            conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
            number_of_samples_list=[4, 4, 4],
        )
        normalized_df = run_module1(raw_df, experiment, DataFormat.GENERIC)

        assert not normalized_df.empty
        conc_cols = [c for c in normalized_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) == 12

        qc_df = run_module2_qc(
            normalized_df, experiment, bqc_label='BQC',
            data_format=DataFormat.GENERIC,
        )
        assert not qc_df.empty

        results = run_module3_analyses(qc_df, experiment)
        assert 'bar' in results
        assert 'pie' in results


class TestEndToEndColumnConsistency:
    """Verify column naming stays consistent across module boundaries."""

    def test_intensity_to_concentration_rename(self):
        """Module 1 renames intensity[] → concentration[], Module 2 and 3 use concentration[]."""
        raw_df = load_generic_sample()
        experiment = ExperimentConfig(
            n_conditions=3,
            conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
            number_of_samples_list=[4, 4, 4],
        )

        # Module 1 input has intensity[] columns
        assert any(c.startswith('intensity[') for c in raw_df.columns)

        # Module 1 output has concentration[] columns
        normalized_df = run_module1(raw_df, experiment, DataFormat.GENERIC)
        assert not any(c.startswith('intensity[') for c in normalized_df.columns)
        assert any(c.startswith('concentration[') for c in normalized_df.columns)

        # Module 2 preserves concentration[] columns
        qc_df = run_module2_qc(normalized_df, experiment, bqc_label='BQC')
        assert any(c.startswith('concentration[') for c in qc_df.columns)

        # Module 3 reads concentration[] columns
        errors = AnalysisWorkflow.validate_inputs(qc_df, experiment)
        assert errors == []

    def test_lipidmolec_and_classkey_preserved(self):
        """LipidMolec and ClassKey columns survive all three modules."""
        raw_df = load_generic_sample()
        experiment = ExperimentConfig(
            n_conditions=3,
            conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
            number_of_samples_list=[4, 4, 4],
        )
        normalized_df = run_module1(raw_df, experiment, DataFormat.GENERIC)
        qc_df = run_module2_qc(normalized_df, experiment, bqc_label='BQC')

        assert 'LipidMolec' in qc_df.columns
        assert 'ClassKey' in qc_df.columns
        # All LipidMolec values should be non-empty strings
        assert qc_df['LipidMolec'].notna().all()
        assert (qc_df['LipidMolec'].str.len() > 0).all()
