"""
Integration tests for Module 2: Quality Check Pipeline.

Tests the complete end-to-end QC flow using real sample datasets
processed through Module 1 (ingestion + normalization) first:
Box plots → BQC assessment → BQC filter → Retention time →
Pairwise correlation → PCA → Sample removal

Multi-step chains validate cascading effects (BQC filter → correlation
on filtered data, PCA → sample removal → re-run PCA) that unit tests
do not cover.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Module 1 workflows (used to produce normalized DataFrames)
from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
)
from app.workflows.normalization import (
    NormalizationWorkflow,
    NormalizationWorkflowConfig,
)

# Module 2 workflow (system under test)
from app.workflows.quality_check import (
    QualityCheckWorkflow,
    QualityCheckConfig,
)

# Result dataclasses (for type assertions)
from app.services.quality_check import (
    BoxPlotResult,
    BQCPrepareResult,
    BQCFilterResult,
    RetentionTimeDataResult,
    CorrelationResult,
    PCAResult,
    SampleRemovalResult,
)

# Models
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig

# Services
from app.services.format_detection import DataFormat

# New architecture standardization service
from app.services.data_standardization import DataStandardizationService


# =============================================================================
# Constants
# =============================================================================

SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / 'sample_datasets'


# =============================================================================
# Helper Functions
# =============================================================================

def load_lipidsearch_sample() -> pd.DataFrame:
    """Load and preprocess LipidSearch 5.0 sample dataset."""
    path = SAMPLE_DATA_DIR / 'lipidsearch5_test_dataset.csv'
    df = pd.read_csv(path)
    result = DataStandardizationService.validate_and_standardize(df, DataFormat.LIPIDSEARCH)
    if not result.success:
        raise ValueError("Failed to standardize LipidSearch dataset")
    return result.standardized_df


def load_msdial_sample() -> pd.DataFrame:
    """Load and preprocess MS-DIAL sample dataset."""
    path = SAMPLE_DATA_DIR / 'msdial_test_dataset.csv'
    df = pd.read_csv(path)
    result = DataStandardizationService.validate_and_standardize(df, DataFormat.MSDIAL)
    if not result.success:
        raise ValueError("Failed to standardize MS-DIAL dataset")
    return result.standardized_df


def load_generic_sample() -> pd.DataFrame:
    """Load and preprocess Generic sample dataset."""
    path = SAMPLE_DATA_DIR / 'generic_test_dataset.csv'
    df = pd.read_csv(path)
    result = DataStandardizationService.validate_and_standardize(df, DataFormat.GENERIC)
    if not result.success:
        raise ValueError("Failed to standardize Generic dataset")
    return result.standardized_df


def load_mw_sample() -> pd.DataFrame:
    """Load and preprocess Metabolomics Workbench sample dataset."""
    path = SAMPLE_DATA_DIR / 'mw_test_dataset.csv'
    with open(path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    result = DataStandardizationService.validate_and_standardize(
        text_content, DataFormat.METABOLOMICS_WORKBENCH
    )
    if not result.success:
        raise ValueError("Failed to standardize Metabolomics Workbench dataset")
    return result.standardized_df


def get_concentration_columns(df: pd.DataFrame) -> list:
    """Get list of concentration column names from DataFrame."""
    return [col for col in df.columns if col.startswith('concentration[')]


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
    a normalized DataFrame with concentration[] columns for QC input.

    Args:
        raw_df: Raw standardized DataFrame with intensity[] columns.
        experiment: Experiment configuration.
        data_format: Data format for ingestion.

    Returns:
        DataFrame with concentration[] columns ready for QC.
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


def make_qc_dataframe(
    lipids: list,
    classes: list,
    n_samples: int,
    values_fn=None,
) -> pd.DataFrame:
    """Build a synthetic QC-ready DataFrame with concentration[] columns.

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
    3 conditions x 4 samples each = 12 samples (4 WT, 4 ADGAT_DKO, 4 BQC)."""
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
    return run_module1_pipeline(raw_df, experiment, DataFormat.GENERIC)


# =============================================================================
# QC Config Fixtures
# =============================================================================

@pytest.fixture
def lipidsearch_qc_config():
    """QC config for LipidSearch format with BQC."""
    return QualityCheckConfig(
        bqc_label='BQC',
        format_type=DataFormat.LIPIDSEARCH,
        cov_threshold=30.0,
    )


@pytest.fixture
def generic_qc_config():
    """QC config for Generic format with BQC."""
    return QualityCheckConfig(
        bqc_label='BQC',
        format_type=DataFormat.GENERIC,
        cov_threshold=30.0,
    )


@pytest.fixture
def msdial_qc_config():
    """QC config for MS-DIAL format (no BQC in test dataset)."""
    return QualityCheckConfig(
        bqc_label=None,
        format_type=DataFormat.MSDIAL,
        cov_threshold=30.0,
    )


@pytest.fixture
def no_bqc_config():
    """QC config with no BQC label."""
    return QualityCheckConfig(
        bqc_label=None,
        format_type=DataFormat.GENERIC,
        cov_threshold=30.0,
    )


# =============================================================================
# Edge Case DataFrame Fixtures (synthetic, QC-ready)
# =============================================================================

@pytest.fixture
def single_lipid_df():
    """Synthetic DataFrame with one lipid species, 4 samples."""
    return make_qc_dataframe(['PC(16:0_18:1)'], ['PC'], n_samples=4)


@pytest.fixture
def two_sample_df():
    """Synthetic DataFrame with 3 lipids, 2 samples (minimum for PCA)."""
    return make_qc_dataframe(
        ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        ['PC', 'PE', 'TG'],
        n_samples=2,
    )


@pytest.fixture
def uniform_df():
    """Synthetic DataFrame where all concentration values are identical (CoV=0)."""
    return make_qc_dataframe(
        ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        ['PC', 'PE', 'TG'],
        n_samples=4,
        values_fn=lambda lip_idx, samp_idx: 1e6,
    )


@pytest.fixture
def high_cov_all_df():
    """Synthetic DataFrame where all lipids have very high CoV (>30%) among
    BQC samples (s3, s4). One BQC sample is very low, the other very high."""
    def values_fn(lip_idx, samp_idx):
        base = (lip_idx + 1) * 1e6
        if samp_idx == 2:  # s3 (BQC) — very low
            return base * 0.2
        elif samp_idx == 3:  # s4 (BQC) — very high
            return base * 1.8
        else:  # s1, s2 (Control) — normal
            return base * (1.0 + samp_idx * 0.01)
    return make_qc_dataframe(
        ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        ['PC', 'PE', 'TG'],
        n_samples=4,
        values_fn=values_fn,
    )


# =============================================================================
# TEST CLASS: TestLipidSearchEndToEnd
# =============================================================================

class TestLipidSearchEndToEnd:
    """Full QC pipeline for LipidSearch format using real 12-sample dataset
    with retention time, BQC, and all QC sections."""

    def test_validation_passes(self, lipidsearch_normalized_df, lipidsearch_experiment):
        """Normalized LipidSearch data passes QC validation."""
        errors = QualityCheckWorkflow.validate_inputs(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        assert errors == [], f"Validation should pass: {errors}"

    def test_box_plots_all_samples(self, lipidsearch_normalized_df, lipidsearch_experiment):
        """Box plots return data for all 12 samples."""
        result = QualityCheckWorkflow.run_box_plots(
            lipidsearch_normalized_df, lipidsearch_experiment
        )

        assert isinstance(result, BoxPlotResult)
        assert len(result.available_samples) == 12
        assert result.mean_area_df.shape[1] == 12
        assert len(result.missing_values_percent) == 12
        assert all(p >= 0 for p in result.missing_values_percent)

    def test_bqc_assessment_identifies_bqc(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """BQC assessment identifies BQC condition and computes CoV data."""
        result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )

        assert result is not None
        assert isinstance(result, BQCPrepareResult)
        assert len(result.bqc_samples) == 4
        # BQC is the 3rd condition (index 2)
        assert result.bqc_sample_index == 2
        assert 'cov' in result.prepared_df.columns
        assert 'mean' in result.prepared_df.columns
        assert 0 <= result.reliable_data_percent <= 100

    def test_bqc_filter_removes_high_cov(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """BQC filter removes high-CoV lipids when no lipids are kept."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )

        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids found in LipidSearch dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        assert isinstance(filter_result, BQCFilterResult)
        assert filter_result.lipids_after < filter_result.lipids_before
        assert filter_result.removed_count == len(bqc_result.high_cov_lipids)
        # Filtered DataFrame should not contain any removed lipids
        remaining_lipids = set(filter_result.filtered_df['LipidMolec'].tolist())
        for removed in filter_result.removed_lipids:
            assert removed not in remaining_lipids

    def test_retention_time_available(
        self, lipidsearch_normalized_df, lipidsearch_qc_config
    ):
        """Retention time data is available for LipidSearch format."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_normalized_df, lipidsearch_qc_config
        )

        assert isinstance(result, RetentionTimeDataResult)
        assert result.available is True
        assert len(result.lipid_classes) > 0

    def test_correlation_all_conditions(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Correlation computed for all 3 conditions (all have >1 replicate)."""
        results = QualityCheckWorkflow.run_all_correlations(
            lipidsearch_normalized_df, lipidsearch_experiment, bqc_label='BQC'
        )

        assert len(results) == 3
        for condition, corr_result in results.items():
            assert isinstance(corr_result, CorrelationResult)
            assert corr_result.condition == condition
            # Each condition has 4 samples -> 4x4 correlation matrix
            assert corr_result.correlation_df.shape == (4, 4)
            # Diagonal should be 1.0
            diag = np.diag(corr_result.correlation_df.values)
            np.testing.assert_allclose(diag, 1.0)
            # bqc_label is set so all conditions use technical threshold
            assert corr_result.threshold == 0.8
            assert corr_result.sample_type == 'technical replicates'

    def test_pca_all_samples(self, lipidsearch_normalized_df, lipidsearch_experiment):
        """PCA includes all 12 samples with correct condition mapping."""
        result = QualityCheckWorkflow.run_pca(
            lipidsearch_normalized_df, lipidsearch_experiment
        )

        assert isinstance(result, PCAResult)
        assert result.pc_df.shape == (12, 2)
        assert len(result.available_samples) == 12
        assert len(result.conditions) == 12
        assert set(result.conditions) == {'WT', 'ADGAT_DKO', 'BQC'}
        # PC labels should contain percentage
        for label in result.pc_labels:
            assert 'PC' in label
            assert '%' in label

    def test_full_interactive_pipeline(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Full interactive chain: box plots -> BQC -> filter -> correlation -> PCA."""
        df = lipidsearch_normalized_df
        experiment = lipidsearch_experiment
        original_lipid_count = len(df)

        # Step 1: Box plots
        box_result = QualityCheckWorkflow.run_box_plots(df, experiment)
        assert len(box_result.available_samples) == 12

        # Step 2: BQC assessment
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            df, experiment, lipidsearch_qc_config
        )
        assert bqc_result is not None

        # Step 3: BQC filter
        if bqc_result.high_cov_lipids:
            filter_result = QualityCheckWorkflow.apply_bqc_filter(
                df, bqc_result.high_cov_lipids
            )
            filtered_df = filter_result.filtered_df
            assert len(filtered_df) < original_lipid_count
        else:
            filtered_df = df

        # Step 4: Correlation on filtered data
        corr_results = QualityCheckWorkflow.run_all_correlations(
            filtered_df, experiment, bqc_label='BQC'
        )
        assert len(corr_results) == 3
        for corr in corr_results.values():
            assert corr.correlation_df.shape == (4, 4)

        # Step 5: PCA on filtered data (same samples, fewer lipids)
        pca_result = QualityCheckWorkflow.run_pca(filtered_df, experiment)
        assert pca_result.pc_df.shape == (12, 2)


# =============================================================================
# TEST CLASS: TestMSDIALEndToEnd
# =============================================================================

class TestMSDIALEndToEnd:
    """Full QC pipeline for MS-DIAL format using real 7-sample dataset.
    Blank condition has single replicate (excluded from correlation).
    RT data available."""

    def test_validation_passes(self, msdial_normalized_df, msdial_experiment):
        """Normalized MS-DIAL data passes QC validation."""
        errors = QualityCheckWorkflow.validate_inputs(
            msdial_normalized_df, msdial_experiment
        )
        assert errors == [], f"Validation should pass: {errors}"

    def test_box_plots_seven_samples(self, msdial_normalized_df, msdial_experiment):
        """Box plots return data for all 7 samples."""
        result = QualityCheckWorkflow.run_box_plots(
            msdial_normalized_df, msdial_experiment
        )

        assert len(result.available_samples) == 7
        assert result.mean_area_df.shape[1] == 7
        assert len(result.missing_values_percent) == 7

    def test_bqc_assessment_none_without_bqc(
        self, msdial_normalized_df, msdial_experiment, msdial_qc_config
    ):
        """BQC assessment returns None when no BQC label is configured."""
        result = QualityCheckWorkflow.run_bqc_assessment(
            msdial_normalized_df, msdial_experiment, msdial_qc_config
        )
        assert result is None

    def test_retention_time_available(self, msdial_normalized_df, msdial_qc_config):
        """Retention time data is available for MS-DIAL format."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            msdial_normalized_df, msdial_qc_config
        )

        assert result.available is True
        assert len(result.lipid_classes) > 0

    def test_correlation_excludes_blank(
        self, msdial_normalized_df, msdial_experiment
    ):
        """Blank condition (1 replicate) excluded from correlation."""
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            msdial_experiment
        )

        assert 'Blank' not in eligible
        assert len(eligible) == 2
        assert 'fads2_KO' in eligible
        assert 'WT' in eligible

        results = QualityCheckWorkflow.run_all_correlations(
            msdial_normalized_df, msdial_experiment
        )
        assert len(results) == 2
        # fads2_KO and WT each have 3 samples -> 3x3 matrices
        for corr_result in results.values():
            assert corr_result.correlation_df.shape == (3, 3)
            # No bqc_label -> biological thresholds
            assert corr_result.threshold == 0.7
            assert corr_result.sample_type == 'biological replicates'

    def test_pca_all_samples(self, msdial_normalized_df, msdial_experiment):
        """PCA includes all 7 samples with correct condition mapping."""
        result = QualityCheckWorkflow.run_pca(
            msdial_normalized_df, msdial_experiment
        )

        assert result.pc_df.shape == (7, 2)
        assert len(result.available_samples) == 7
        assert set(result.conditions) == {'Blank', 'fads2_KO', 'WT'}


# =============================================================================
# TEST CLASS: TestGenericEndToEnd
# =============================================================================

class TestGenericEndToEnd:
    """Full QC pipeline for Generic format. No retention time data.
    Tests biological vs technical replicate correlation thresholds."""

    def test_validation_passes(self, generic_normalized_df, generic_experiment):
        """Normalized Generic data passes QC validation."""
        errors = QualityCheckWorkflow.validate_inputs(
            generic_normalized_df, generic_experiment
        )
        assert errors == [], f"Validation should pass: {errors}"

    def test_retention_time_not_available(
        self, generic_normalized_df, generic_qc_config
    ):
        """Retention time data is not available for Generic format."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            generic_normalized_df, generic_qc_config
        )
        assert result.available is False
        assert result.lipid_classes == []

    def test_bqc_assessment_with_bqc(
        self, generic_normalized_df, generic_experiment, generic_qc_config
    ):
        """BQC assessment works for Generic format with BQC condition."""
        result = QualityCheckWorkflow.run_bqc_assessment(
            generic_normalized_df, generic_experiment, generic_qc_config
        )

        assert result is not None
        assert isinstance(result, BQCPrepareResult)
        # BQC is the 3rd condition (index 2)
        assert result.bqc_sample_index == 2
        assert isinstance(result.high_cov_lipids, list)

    def test_correlation_threshold_with_vs_without_bqc_label(
        self, generic_normalized_df, generic_experiment
    ):
        """With bqc_label set: all conditions use technical threshold (0.8).
        Without bqc_label: all conditions use biological threshold (0.7)."""
        # With bqc_label
        corr_with_bqc = QualityCheckWorkflow.run_correlation(
            generic_normalized_df, generic_experiment, 'WT', bqc_label='BQC'
        )
        assert corr_with_bqc.threshold == 0.8
        assert corr_with_bqc.sample_type == 'technical replicates'

        # Without bqc_label
        corr_without_bqc = QualityCheckWorkflow.run_correlation(
            generic_normalized_df, generic_experiment, 'WT', bqc_label=None
        )
        assert corr_without_bqc.threshold == 0.7
        assert corr_without_bqc.sample_type == 'biological replicates'

    def test_pca_twelve_samples(self, generic_normalized_df, generic_experiment):
        """PCA includes all 12 samples."""
        result = QualityCheckWorkflow.run_pca(
            generic_normalized_df, generic_experiment
        )

        assert result.pc_df.shape == (12, 2)
        assert set(result.conditions) == {'WT', 'ADGAT_DKO', 'BQC'}


# =============================================================================
# TEST CLASS: TestMWEndToEnd
# =============================================================================

class TestMWEndToEnd:
    """Full QC pipeline for Metabolomics Workbench format.
    44 samples, large correlation matrices, no BQC, no RT."""

    def test_validation_passes(self, mw_normalized_df, mw_experiment):
        """Normalized MW data passes QC validation."""
        errors = QualityCheckWorkflow.validate_inputs(
            mw_normalized_df, mw_experiment
        )
        assert errors == [], f"Validation should pass: {errors}"

    def test_box_plots_44_samples(self, mw_normalized_df, mw_experiment):
        """Box plots handle 44 samples."""
        result = QualityCheckWorkflow.run_box_plots(
            mw_normalized_df, mw_experiment
        )

        assert len(result.available_samples) == 44
        assert result.mean_area_df.shape[1] == 44

    def test_correlation_large_matrices(self, mw_normalized_df, mw_experiment):
        """Correlation produces 22x22 matrices for each condition."""
        results = QualityCheckWorkflow.run_all_correlations(
            mw_normalized_df, mw_experiment
        )

        assert len(results) == 2
        for condition in ['Normal', 'HFD']:
            assert condition in results
            corr = results[condition]
            assert corr.correlation_df.shape == (22, 22)
            # Diagonal should be 1.0
            diag = np.diag(corr.correlation_df.values)
            np.testing.assert_allclose(diag, 1.0)

    def test_pca_44_samples(self, mw_normalized_df, mw_experiment):
        """PCA handles 44 samples with correct condition mapping."""
        result = QualityCheckWorkflow.run_pca(
            mw_normalized_df, mw_experiment
        )

        assert result.pc_df.shape == (44, 2)
        assert len(result.available_samples) == 44
        assert set(result.conditions) == {'Normal', 'HFD'}


# =============================================================================
# TEST CLASS: TestBQCCascadingEffects
# =============================================================================

class TestBQCCascadingEffects:
    """Tests cascading effects of BQC filtering on downstream QC steps.
    Uses LipidSearch data (has BQC condition)."""

    def test_bqc_filter_reduces_lipid_count(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """BQC filter reduces lipid count when high-CoV lipids exist."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )
        assert filter_result.lipids_after < filter_result.lipids_before
        assert filter_result.removed_count > 0

    def test_filtered_df_used_in_correlation(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Correlation on BQC-filtered data uses fewer lipids."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        # Correlation on original vs filtered
        corr_original = QualityCheckWorkflow.run_correlation(
            lipidsearch_normalized_df, lipidsearch_experiment, 'WT'
        )
        corr_filtered = QualityCheckWorkflow.run_correlation(
            filter_result.filtered_df, lipidsearch_experiment, 'WT'
        )

        # Both should be 4x4 (same samples), but values may differ
        assert corr_original.correlation_df.shape == (4, 4)
        assert corr_filtered.correlation_df.shape == (4, 4)

    def test_filtered_df_used_in_pca(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """PCA on BQC-filtered data: same samples, potentially different variance explained."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        pca_before = QualityCheckWorkflow.run_pca(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        pca_after = QualityCheckWorkflow.run_pca(
            filter_result.filtered_df, lipidsearch_experiment
        )

        # Same number of samples (12) but variance explained likely differs
        assert pca_before.pc_df.shape == (12, 2)
        assert pca_after.pc_df.shape == (12, 2)

    def test_filtered_df_used_in_box_plots(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Box plots on BQC-filtered data have fewer rows."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        box_before = QualityCheckWorkflow.run_box_plots(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        box_after = QualityCheckWorkflow.run_box_plots(
            filter_result.filtered_df, lipidsearch_experiment
        )

        assert box_after.mean_area_df.shape[0] < box_before.mean_area_df.shape[0]

    def test_keeping_some_high_cov_lipids(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Keeping some high-CoV lipids retains them in filtered DataFrame."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        keep_one = [bqc_result.high_cov_lipids[0]]
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids,
            lipids_to_keep=keep_one,
        )

        assert filter_result.kept_despite_high_cov == keep_one
        expected_removed = len(bqc_result.high_cov_lipids) - 1
        assert filter_result.removed_count == expected_removed

    def test_keeping_all_high_cov_lipids(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Keeping all high-CoV lipids results in no removal."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids,
            lipids_to_keep=bqc_result.high_cov_lipids,
        )

        assert filter_result.lipids_after == filter_result.lipids_before
        assert filter_result.removed_count == 0

    def test_bqc_filter_then_correlation_then_pca(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Full downstream chain: BQC filter -> correlation -> PCA all succeed."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        # Filter
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )
        filtered_df = filter_result.filtered_df

        # Correlation on filtered
        corr_results = QualityCheckWorkflow.run_all_correlations(
            filtered_df, lipidsearch_experiment, bqc_label='BQC'
        )
        assert len(corr_results) == 3
        for corr in corr_results.values():
            assert corr.correlation_df.shape == (4, 4)

        # PCA on filtered
        pca_result = QualityCheckWorkflow.run_pca(
            filtered_df, lipidsearch_experiment
        )
        assert pca_result.pc_df.shape == (12, 2)

    def test_filter_preserves_non_concentration_columns(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """BQC filter preserves LipidMolec, ClassKey, and concentration columns."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        assert 'LipidMolec' in filter_result.filtered_df.columns
        assert 'ClassKey' in filter_result.filtered_df.columns
        conc_cols_before = get_concentration_columns(lipidsearch_normalized_df)
        conc_cols_after = get_concentration_columns(filter_result.filtered_df)
        assert len(conc_cols_after) == len(conc_cols_before)

    def test_filter_result_computed_properties(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """BQC filter result computed properties are correct."""
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        assert filter_result.removed_count == (
            filter_result.lipids_before - filter_result.lipids_after
        )
        expected_pct = (
            filter_result.removed_count / filter_result.lipids_before * 100
        )
        assert abs(filter_result.removed_percentage - expected_pct) < 0.01


# =============================================================================
# TEST CLASS: TestPCASampleRemoval
# =============================================================================

class TestPCASampleRemoval:
    """Tests PCA-driven sample removal and cascading effects.
    Uses LipidSearch normalized data (12 samples)."""

    def test_remove_single_sample(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Removing one sample updates counts correctly."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        assert isinstance(result, SampleRemovalResult)
        assert result.samples_before == 12
        assert result.samples_after == 11
        assert result.removed_samples == ['s1']

    def test_remove_multiple_samples(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Removing 2 samples from different conditions."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1', 's5'],
        )

        assert result.samples_after == 10
        assert set(result.removed_samples) == {'s1', 's5'}
        # No condition fully removed (WT had 4, ADGAT_DKO had 4)
        assert result.updated_experiment.n_conditions == 3

    def test_updated_experiment_config(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """After removing s1 (WT), WT condition has 3 samples instead of 4."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        # WT was first condition with 4 samples, now has 3
        assert result.updated_experiment.number_of_samples_list[0] == 3
        # Other conditions unchanged
        assert result.updated_experiment.number_of_samples_list[1] == 4
        assert result.updated_experiment.number_of_samples_list[2] == 4
        assert result.updated_experiment.conditions_list == ['WT', 'ADGAT_DKO', 'BQC']

    def test_column_renaming_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Concentration columns renumbered sequentially after removal."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        conc_cols = get_concentration_columns(result.updated_df)
        assert len(conc_cols) == 11
        # Columns should be sequentially named s1..s11
        expected_cols = [f'concentration[s{i}]' for i in range(1, 12)]
        assert conc_cols == expected_cols

    def test_pca_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """PCA re-runs correctly on updated data after sample removal."""
        removal = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        pca_result = QualityCheckWorkflow.run_pca(
            removal.updated_df, removal.updated_experiment
        )

        assert pca_result.pc_df.shape == (11, 2)
        assert len(pca_result.available_samples) == 11

    def test_correlation_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Correlation re-runs correctly with updated experiment after removal."""
        removal = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        results = QualityCheckWorkflow.run_all_correlations(
            removal.updated_df, removal.updated_experiment
        )

        assert len(results) == 3
        # WT now has 3 samples -> 3x3 matrix
        assert results['WT'].correlation_df.shape == (3, 3)
        # Others still have 4 -> 4x4
        assert results['ADGAT_DKO'].correlation_df.shape == (4, 4)
        assert results['BQC'].correlation_df.shape == (4, 4)

    def test_remove_then_remove_again(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Two successive removals work correctly."""
        # First removal
        result1 = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )
        assert result1.samples_after == 11

        # Second removal (s1 in the updated data was originally s2)
        result2 = QualityCheckWorkflow.remove_samples(
            result1.updated_df, result1.updated_experiment,
            samples_to_remove=['s1'],
        )
        assert result2.samples_after == 10
        # WT condition lost 2 samples: now has 2
        assert result2.updated_experiment.number_of_samples_list[0] == 2

    def test_remove_entire_condition(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Removing all samples from BQC condition drops that condition."""
        # BQC samples are s9, s10, s11, s12
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s9', 's10', 's11', 's12'],
        )

        assert result.samples_after == 8
        assert result.updated_experiment.n_conditions == 2
        assert 'BQC' not in result.updated_experiment.conditions_list
        assert result.updated_experiment.conditions_list == ['WT', 'ADGAT_DKO']

    def test_remove_preserves_lipid_data(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Sample removal does not change the number of lipid rows."""
        original_lipid_count = len(lipidsearch_normalized_df)

        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        assert len(result.updated_df) == original_lipid_count
        assert (
            result.updated_df['LipidMolec'].tolist()
            == lipidsearch_normalized_df['LipidMolec'].tolist()
        )

    def test_cannot_remove_all_but_one(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Removing too many samples raises ValueError (need >=2)."""
        samples_to_remove = [f's{i}' for i in range(1, 12)]  # Remove 11 of 12

        with pytest.raises(ValueError):
            QualityCheckWorkflow.remove_samples(
                lipidsearch_normalized_df, lipidsearch_experiment,
                samples_to_remove=samples_to_remove,
            )


# =============================================================================
# TEST CLASS: TestFormatSpecificBehavior
# =============================================================================

class TestFormatSpecificBehavior:
    """Tests format-specific behavior: retention time availability
    and column preservation across formats."""

    def test_lipidsearch_has_retention_time(
        self, lipidsearch_normalized_df, lipidsearch_qc_config
    ):
        """LipidSearch format has retention time data."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            lipidsearch_normalized_df, lipidsearch_qc_config
        )
        assert result.available is True

    def test_msdial_has_retention_time(
        self, msdial_normalized_df, msdial_qc_config
    ):
        """MS-DIAL format has retention time data."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            msdial_normalized_df, msdial_qc_config
        )
        assert result.available is True

    def test_generic_no_retention_time(
        self, generic_normalized_df, generic_qc_config
    ):
        """Generic format has no retention time data."""
        result = QualityCheckWorkflow.check_retention_time_availability(
            generic_normalized_df, generic_qc_config
        )
        assert result.available is False

    def test_mw_no_retention_time(self, mw_normalized_df):
        """Metabolomics Workbench format has no retention time data."""
        config = QualityCheckConfig(format_type=DataFormat.GENERIC)
        result = QualityCheckWorkflow.check_retention_time_availability(
            mw_normalized_df, config
        )
        assert result.available is False

    def test_lipidsearch_preserves_calcmass_basert(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """CalcMass and BaseRt columns survive BQC filtering for LipidSearch."""
        assert 'CalcMass' in lipidsearch_normalized_df.columns
        assert 'BaseRt' in lipidsearch_normalized_df.columns

        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if not bqc_result.high_cov_lipids:
            pytest.skip("No high-CoV lipids in dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, bqc_result.high_cov_lipids
        )

        assert 'CalcMass' in filter_result.filtered_df.columns
        assert 'BaseRt' in filter_result.filtered_df.columns


# =============================================================================
# TEST CLASS: TestDataIntegrity
# =============================================================================

class TestDataIntegrity:
    """Tests data integrity through the QC pipeline.
    Verifies column preservation, sample count consistency, and non-negative values."""

    def test_concentration_columns_preserved_through_pipeline(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """Concentration column count is preserved through QC steps (no sample removal)."""
        original_conc_count = len(get_concentration_columns(lipidsearch_normalized_df))

        # Box plots don't modify DataFrame but we verify the result references correct samples
        box_result = QualityCheckWorkflow.run_box_plots(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        assert box_result.mean_area_df.shape[1] == original_conc_count

        # BQC filter (if applicable) preserves concentration columns
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if bqc_result and bqc_result.high_cov_lipids:
            filter_result = QualityCheckWorkflow.apply_bqc_filter(
                lipidsearch_normalized_df, bqc_result.high_cov_lipids
            )
            conc_after_filter = len(get_concentration_columns(filter_result.filtered_df))
            assert conc_after_filter == original_conc_count

    def test_sample_count_consistent_across_results(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Available sample count is consistent between box plots and PCA."""
        box_result = QualityCheckWorkflow.run_box_plots(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        pca_result = QualityCheckWorkflow.run_pca(
            lipidsearch_normalized_df, lipidsearch_experiment
        )

        assert box_result.available_samples == pca_result.available_samples

    def test_non_negative_concentration_values(self, lipidsearch_normalized_df):
        """All concentration values are non-negative."""
        conc_cols = get_concentration_columns(lipidsearch_normalized_df)
        for col in conc_cols:
            assert (lipidsearch_normalized_df[col] >= 0).all(), (
                f"Negative values found in {col}"
            )

    def test_lipidmolec_preserved_after_all_steps(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """LipidMolec column is present after BQC filter and sample removal."""
        # After BQC filter
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if bqc_result and bqc_result.high_cov_lipids:
            filter_result = QualityCheckWorkflow.apply_bqc_filter(
                lipidsearch_normalized_df, bqc_result.high_cov_lipids
            )
            assert 'LipidMolec' in filter_result.filtered_df.columns

        # After sample removal
        removal_result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )
        assert 'LipidMolec' in removal_result.updated_df.columns

    def test_classkey_preserved_after_all_steps(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """ClassKey column is present after BQC filter and sample removal."""
        # After BQC filter
        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        if bqc_result and bqc_result.high_cov_lipids:
            filter_result = QualityCheckWorkflow.apply_bqc_filter(
                lipidsearch_normalized_df, bqc_result.high_cov_lipids
            )
            assert 'ClassKey' in filter_result.filtered_df.columns

        # After sample removal
        removal_result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )
        assert 'ClassKey' in removal_result.updated_df.columns


# =============================================================================
# TEST CLASS: TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests edge cases using synthetic QC-ready DataFrames."""

    def test_single_lipid_box_plots(self, single_lipid_df, four_sample_experiment):
        """Box plots work with a single lipid species."""
        result = QualityCheckWorkflow.run_box_plots(
            single_lipid_df, four_sample_experiment
        )
        assert result.mean_area_df.shape[0] == 1
        assert len(result.available_samples) == 4

    def test_single_lipid_pca_raises(self, single_lipid_df, four_sample_experiment):
        """PCA with a single lipid (1 feature) raises ValueError because
        n_components=2 exceeds min(n_samples, n_features)=1."""
        with pytest.raises(ValueError):
            QualityCheckWorkflow.run_pca(
                single_lipid_df, four_sample_experiment
            )

    def test_two_samples_minimum_pca(self, two_sample_df, two_sample_experiment):
        """PCA works at minimum boundary (2 samples)."""
        result = QualityCheckWorkflow.run_pca(
            two_sample_df, two_sample_experiment
        )
        assert result.pc_df.shape == (2, 2)

    def test_uniform_data_bqc(self, uniform_df):
        """Uniform data (all identical values) has CoV=0, so no high-CoV lipids."""
        # Need a BQC experiment for the uniform_df (4 samples -> 2 BQC + 2 other)
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'BQC'],
            number_of_samples_list=[2, 2],
        )
        config = QualityCheckConfig(bqc_label='BQC', cov_threshold=30.0)

        result = QualityCheckWorkflow.run_bqc_assessment(
            uniform_df, experiment, config
        )

        assert result is not None
        assert result.high_cov_lipids == []
        assert result.reliable_data_percent == 100.0

    def test_high_cov_all_lipids_filter(self, high_cov_all_df):
        """When all lipids have high CoV, filtering removes them all."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'BQC'],
            number_of_samples_list=[2, 2],
        )
        config = QualityCheckConfig(bqc_label='BQC', cov_threshold=30.0)

        bqc_result = QualityCheckWorkflow.run_bqc_assessment(
            high_cov_all_df, experiment, config
        )

        assert bqc_result is not None
        assert len(bqc_result.high_cov_lipids) == 3  # All 3 lipids

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            high_cov_all_df, bqc_result.high_cov_lipids
        )
        assert filter_result.lipids_after == 0
        assert len(filter_result.filtered_df) == 0

    def test_single_replicate_correlation_raises(self):
        """Correlation on a condition with 1 replicate raises ValueError."""
        df = make_qc_dataframe(
            ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            ['PC', 'PE'],
            n_samples=3,
        )
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Solo', 'Pair'],
            number_of_samples_list=[1, 2],
        )

        with pytest.raises(ValueError, match="at least 2"):
            QualityCheckWorkflow.run_correlation(
                df, experiment, 'Solo'
            )


# =============================================================================
# TEST CLASS: TestErrorHandling
# =============================================================================

class TestErrorHandling:
    """Tests error handling for invalid inputs to the QC pipeline."""

    def test_missing_concentration_columns(self, four_sample_experiment):
        """Validation fails when DataFrame has intensity[] but no concentration[] columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1e6, 2e6],
            'intensity[s2]': [1.1e6, 2.1e6],
            'intensity[s3]': [1.2e6, 2.2e6],
            'intensity[s4]': [1.3e6, 2.3e6],
        })

        errors = QualityCheckWorkflow.validate_inputs(df, four_sample_experiment)
        assert len(errors) > 0

    def test_invalid_bqc_label(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """BQC assessment returns None when label not in experiment."""
        config = QualityCheckConfig(bqc_label='NonExistent')

        result = QualityCheckWorkflow.run_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, config
        )
        assert result is None

    def test_single_replicate_correlation_error(
        self, msdial_normalized_df, msdial_experiment
    ):
        """Correlation on Blank condition (1 replicate) raises ValueError."""
        with pytest.raises(ValueError):
            QualityCheckWorkflow.run_correlation(
                msdial_normalized_df, msdial_experiment, 'Blank'
            )

    def test_empty_dataframe_validation(self, four_sample_experiment):
        """Validation fails on empty DataFrame."""
        df = pd.DataFrame()

        errors = QualityCheckWorkflow.validate_inputs(df, four_sample_experiment)
        assert len(errors) > 0


# =============================================================================
# TEST CLASS: TestNonInteractivePipeline
# =============================================================================

class TestNonInteractivePipeline:
    """Tests the run_non_interactive method for batch execution."""

    def test_non_interactive_returns_all_fields(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """run_non_interactive returns dataclass with all expected fields."""
        result = QualityCheckWorkflow.run_non_interactive(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )

        assert hasattr(result, 'box_plot')
        assert hasattr(result, 'bqc')
        assert hasattr(result, 'retention_time')
        assert hasattr(result, 'correlations')
        assert hasattr(result, 'pca')
        assert hasattr(result, 'validation_errors')
        assert result.validation_errors == []

        # All results should be populated for LipidSearch with BQC
        assert result.box_plot is not None
        assert result.bqc is not None
        assert result.retention_time is not None
        assert len(result.correlations) > 0
        assert result.pca is not None

    def test_non_interactive_consistent_with_individual_calls(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
    ):
        """run_non_interactive results match calling each method individually."""
        ni_result = QualityCheckWorkflow.run_non_interactive(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )

        # Compare box plots
        box_individual = QualityCheckWorkflow.run_box_plots(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        assert (
            ni_result.box_plot.available_samples
            == box_individual.available_samples
        )

        # Compare PCA
        pca_individual = QualityCheckWorkflow.run_pca(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        assert ni_result.pca.pc_df.shape == pca_individual.pc_df.shape

        # Compare correlation count
        corr_individual = QualityCheckWorkflow.run_all_correlations(
            lipidsearch_normalized_df, lipidsearch_experiment, bqc_label='BQC'
        )
        assert len(ni_result.correlations) == len(corr_individual)

    def test_non_interactive_lipidsearch_vs_generic(
        self,
        lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config,
        generic_normalized_df, generic_experiment, generic_qc_config,
    ):
        """Non-interactive: LipidSearch has RT, Generic does not. Both have BQC."""
        ls_result = QualityCheckWorkflow.run_non_interactive(
            lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_qc_config
        )
        gen_result = QualityCheckWorkflow.run_non_interactive(
            generic_normalized_df, generic_experiment, generic_qc_config
        )

        # RT availability differs
        assert ls_result.retention_time.available is True
        assert gen_result.retention_time.available is False

        # Both have BQC results
        assert ls_result.bqc is not None
        assert gen_result.bqc is not None
