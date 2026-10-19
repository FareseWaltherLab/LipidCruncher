"""
Integration tests for Module 2: Quality Check Pipeline.

Tests the complete end-to-end QC flow using real sample datasets
processed through Module 1 (ingestion + normalization) first:
Box plots → BQC assessment → BQC filter → Retention time →
Pairwise correlation → PCA → Sample removal

Each step runs the way the app runs it (the services behind the adapter's
QC methods, see ``tests/integration/conftest.py``). Multi-step chains
validate cascading effects (BQC filter → correlation on filtered data,
PCA → sample removal → re-run PCA) that unit tests do not cover.
"""
import pytest
import pandas as pd
import numpy as np

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
    BQCFilterResult,
    RetentionTimeDataResult,
    SampleRemovalResult,
)

# Models
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig

# Services
from app.services.format_detection import DataFormat

# Shared integration test helpers
from tests.integration.conftest import (
    load_lipidsearch_sample,
    load_msdial_sample,
    load_generic_sample,
    load_mw_sample,
    get_concentration_columns,
    qc_box_plot_data,
    qc_bqc_assessment,
    qc_correlation,
    qc_pca,
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
        def values_fn(lip_idx, samp_idx):
            return (lip_idx + 1) * 1e6 + samp_idx * 1e4

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
    """QC config for LipidSearch format."""
    return QualityCheckConfig(format_type=DataFormat.LIPIDSEARCH)


@pytest.fixture
def generic_qc_config():
    """QC config for Generic format."""
    return QualityCheckConfig(format_type=DataFormat.GENERIC)


@pytest.fixture
def msdial_qc_config():
    """QC config for MS-DIAL format."""
    return QualityCheckConfig(format_type=DataFormat.MSDIAL)


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
        samples, mean_area_df, missing = qc_box_plot_data(
            lipidsearch_normalized_df, lipidsearch_experiment
        )

        assert len(samples) == 12
        assert mean_area_df.shape[1] == 12
        assert len(missing) == 12
        assert all(p >= 0 for p in missing)

    def test_bqc_assessment_computes_cov(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """BQC assessment computes CoV and mean for every lipid."""
        prepared_df, reliable_pct, _ = qc_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, 'BQC'
        )

        assert 'cov' in prepared_df.columns
        assert 'mean' in prepared_df.columns
        assert len(prepared_df) == len(lipidsearch_normalized_df)
        assert 0 <= reliable_pct <= 100

    def test_bqc_filter_removes_high_cov(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """BQC filter removes high-CoV lipids when no lipids are kept."""
        _, _, high_cov = qc_bqc_assessment(
            lipidsearch_normalized_df, lipidsearch_experiment, 'BQC'
        )

        if not high_cov:
            pytest.skip("No high-CoV lipids found in LipidSearch dataset")

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, high_cov
        )

        assert isinstance(filter_result, BQCFilterResult)
        assert filter_result.lipids_after < filter_result.lipids_before
        assert filter_result.removed_count == len(high_cov)
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
        """Correlation computed for all 3 conditions (all have >1 replicate);
        only the BQC condition uses the technical-replicate threshold."""
        eligible = QualityCheckWorkflow.get_eligible_correlation_conditions(
            lipidsearch_experiment
        )
        assert eligible == ['WT', 'ADGAT_DKO', 'BQC']

        for condition in eligible:
            sample_type = (
                'technical replicates' if condition == 'BQC'
                else 'biological replicates'
            )
            correlation_df, threshold = qc_correlation(
                lipidsearch_normalized_df, lipidsearch_experiment,
                condition, sample_type,
            )
            # Each condition has 4 samples -> 4x4 correlation matrix
            assert correlation_df.shape == (4, 4)
            np.testing.assert_allclose(np.diag(correlation_df.values), 1.0)
            assert threshold == (0.8 if condition == 'BQC' else 0.7)

    def test_pca_all_samples(self, lipidsearch_normalized_df, lipidsearch_experiment):
        """PCA includes all 12 samples with correct condition mapping."""
        pca_df = qc_pca(lipidsearch_normalized_df, lipidsearch_experiment)

        assert pca_df[['PC1', 'PC2']].shape == (12, 2)
        assert pca_df['Sample'].tolist() == lipidsearch_experiment.full_samples_list
        assert pca_df['Condition'].tolist() == (
            lipidsearch_experiment.extensive_conditions_list
        )

    def test_full_interactive_pipeline(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Full interactive chain: box plots -> BQC -> filter -> correlation -> PCA."""
        df = lipidsearch_normalized_df
        experiment = lipidsearch_experiment
        original_lipid_count = len(df)

        # Step 1: Box plots
        samples, _, _ = qc_box_plot_data(df, experiment)
        assert len(samples) == 12

        # Step 2: BQC assessment
        _, _, high_cov = qc_bqc_assessment(df, experiment, 'BQC')

        # Step 3: BQC filter
        if high_cov:
            filter_result = QualityCheckWorkflow.apply_bqc_filter(df, high_cov)
            filtered_df = filter_result.filtered_df
            assert len(filtered_df) < original_lipid_count
        else:
            filtered_df = df

        # Step 4: Correlation on filtered data
        for condition in experiment.conditions_list:
            correlation_df, _ = qc_correlation(filtered_df, experiment, condition)
            assert correlation_df.shape == (4, 4)

        # Step 5: PCA on filtered data (same samples, fewer lipids)
        assert len(qc_pca(filtered_df, experiment)) == 12


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
        samples, mean_area_df, missing = qc_box_plot_data(
            msdial_normalized_df, msdial_experiment
        )

        assert len(samples) == 7
        assert mean_area_df.shape[1] == 7
        assert len(missing) == 7

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

        assert eligible == ['fads2_KO', 'WT']
        # fads2_KO and WT each have 3 samples -> 3x3 matrices
        for condition in eligible:
            correlation_df, threshold = qc_correlation(
                msdial_normalized_df, msdial_experiment, condition
            )
            assert correlation_df.shape == (3, 3)
            assert threshold == 0.7

    def test_pca_all_samples(self, msdial_normalized_df, msdial_experiment):
        """PCA includes all 7 samples with correct condition mapping."""
        pca_df = qc_pca(msdial_normalized_df, msdial_experiment)

        assert len(pca_df) == 7
        assert set(pca_df['Condition']) == {'Blank', 'fads2_KO', 'WT'}


# =============================================================================
# TEST CLASS: TestGenericEndToEnd
# =============================================================================

class TestGenericEndToEnd:
    """Full QC pipeline for Generic format. No retention time data."""

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

    def test_bqc_assessment_with_bqc(self, generic_normalized_df, generic_experiment):
        """BQC assessment works for Generic format with BQC condition."""
        prepared_df, _, high_cov = qc_bqc_assessment(
            generic_normalized_df, generic_experiment, 'BQC'
        )

        assert len(prepared_df) == len(generic_normalized_df)
        assert isinstance(high_cov, list)

    def test_pca_twelve_samples(self, generic_normalized_df, generic_experiment):
        """PCA includes all 12 samples."""
        pca_df = qc_pca(generic_normalized_df, generic_experiment)

        assert len(pca_df) == 12
        assert set(pca_df['Condition']) == {'WT', 'ADGAT_DKO', 'BQC'}


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
        samples, mean_area_df, _ = qc_box_plot_data(mw_normalized_df, mw_experiment)

        assert len(samples) == 44
        assert mean_area_df.shape[1] == 44

    def test_correlation_large_matrices(self, mw_normalized_df, mw_experiment):
        """Correlation produces 22x22 matrices for each condition."""
        assert QualityCheckWorkflow.get_eligible_correlation_conditions(
            mw_experiment
        ) == ['Normal', 'HFD']
        for condition in ['Normal', 'HFD']:
            correlation_df, _ = qc_correlation(mw_normalized_df, mw_experiment, condition)
            assert correlation_df.shape == (22, 22)
            np.testing.assert_allclose(np.diag(correlation_df.values), 1.0)

    def test_pca_44_samples(self, mw_normalized_df, mw_experiment):
        """PCA handles 44 samples with correct condition mapping."""
        pca_df = qc_pca(mw_normalized_df, mw_experiment)

        assert len(pca_df) == 44
        assert set(pca_df['Condition']) == {'Normal', 'HFD'}


# =============================================================================
# TEST CLASS: TestBQCCascadingEffects
# =============================================================================

@pytest.fixture
def lipidsearch_high_cov(lipidsearch_normalized_df, lipidsearch_experiment):
    """The lipids the BQC filter removes from the LipidSearch sample data."""
    _, _, high_cov = qc_bqc_assessment(
        lipidsearch_normalized_df, lipidsearch_experiment, 'BQC'
    )
    if not high_cov:
        pytest.skip("No high-CoV lipids in dataset")
    return high_cov


class TestBQCCascadingEffects:
    """Tests cascading effects of BQC filtering on downstream QC steps.
    Uses LipidSearch data (has BQC condition)."""

    def test_bqc_filter_reduces_lipid_count(
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """BQC filter reduces lipid count when high-CoV lipids exist."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )
        assert filter_result.lipids_after < filter_result.lipids_before
        assert filter_result.removed_count > 0

    def test_filtered_df_used_in_correlation(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_high_cov
    ):
        """Correlation on BQC-filtered data keeps the same samples."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )

        corr_original, _ = qc_correlation(
            lipidsearch_normalized_df, lipidsearch_experiment, 'WT'
        )
        corr_filtered, _ = qc_correlation(
            filter_result.filtered_df, lipidsearch_experiment, 'WT'
        )

        # Both should be 4x4 (same samples), but values may differ
        assert corr_original.shape == (4, 4)
        assert corr_filtered.shape == (4, 4)

    def test_filtered_df_used_in_pca(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_high_cov
    ):
        """PCA on BQC-filtered data: same samples."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )

        assert len(qc_pca(lipidsearch_normalized_df, lipidsearch_experiment)) == 12
        assert len(qc_pca(filter_result.filtered_df, lipidsearch_experiment)) == 12

    def test_filtered_df_used_in_box_plots(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_high_cov
    ):
        """Box plots on BQC-filtered data have fewer rows."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )

        _, before, _ = qc_box_plot_data(lipidsearch_normalized_df, lipidsearch_experiment)
        _, after, _ = qc_box_plot_data(filter_result.filtered_df, lipidsearch_experiment)

        assert after.shape[0] < before.shape[0]

    def test_keeping_some_high_cov_lipids(
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """Keeping some high-CoV lipids retains them in filtered DataFrame."""
        keep_one = [lipidsearch_high_cov[0]]
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov,
            lipids_to_keep=keep_one,
        )

        assert filter_result.kept_despite_high_cov == keep_one
        assert filter_result.removed_count == len(lipidsearch_high_cov) - 1
        assert keep_one[0] in set(filter_result.filtered_df['LipidMolec'])

    def test_keeping_all_high_cov_lipids(
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """Keeping all high-CoV lipids results in no removal."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov,
            lipids_to_keep=lipidsearch_high_cov,
        )

        assert filter_result.lipids_after == filter_result.lipids_before
        assert filter_result.removed_count == 0

    def test_filter_preserves_non_concentration_columns(
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """BQC filter preserves LipidMolec, ClassKey, and concentration columns."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )

        assert 'LipidMolec' in filter_result.filtered_df.columns
        assert 'ClassKey' in filter_result.filtered_df.columns
        conc_cols_before = get_concentration_columns(lipidsearch_normalized_df)
        conc_cols_after = get_concentration_columns(filter_result.filtered_df)
        assert len(conc_cols_after) == len(conc_cols_before)

    def test_filter_result_computed_properties(
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """BQC filter result computed properties are correct."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
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

        assert result.updated_experiment.number_of_samples_list == [3, 4, 4]
        assert result.updated_experiment.conditions_list == ['WT', 'ADGAT_DKO', 'BQC']

    def test_column_renaming_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Concentration columns renumbered sequentially after removal, each
        survivor keeping its own data."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        conc_cols = get_concentration_columns(result.updated_df)
        assert conc_cols == [f'concentration[s{i}]' for i in range(1, 12)]
        assert result.old_to_new['concentration[s2]'] == 'concentration[s1]'
        pd.testing.assert_series_equal(
            result.updated_df['concentration[s1]'],
            lipidsearch_normalized_df['concentration[s2]'],
            check_names=False,
        )

    def test_pca_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """PCA re-runs correctly on updated data after sample removal."""
        removal = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

        pca_df = qc_pca(removal.updated_df, removal.updated_experiment)

        assert len(pca_df) == 11
        assert pca_df['Condition'].tolist() == ['WT'] * 3 + ['ADGAT_DKO'] * 4 + ['BQC'] * 4

    def test_correlation_after_removal(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Correlation re-runs correctly with updated experiment after removal."""
        removal = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )
        experiment = removal.updated_experiment

        # WT now has 3 samples -> 3x3 matrix; others still have 4
        assert qc_correlation(removal.updated_df, experiment, 'WT')[0].shape == (3, 3)
        assert qc_correlation(removal.updated_df, experiment, 'ADGAT_DKO')[0].shape == (4, 4)
        assert qc_correlation(removal.updated_df, experiment, 'BQC')[0].shape == (4, 4)

    def test_remove_then_remove_again(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Two successive removals work correctly."""
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
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s9', 's10', 's11', 's12'],
        )

        assert result.samples_after == 8
        assert result.updated_experiment.conditions_list == ['WT', 'ADGAT_DKO']

    def test_remove_preserves_lipid_data(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Sample removal does not change the number of lipid rows."""
        result = QualityCheckWorkflow.remove_samples(
            lipidsearch_normalized_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )

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
        self, lipidsearch_normalized_df, lipidsearch_high_cov
    ):
        """CalcMass and BaseRt columns survive BQC filtering for LipidSearch."""
        assert 'CalcMass' in lipidsearch_normalized_df.columns
        assert 'BaseRt' in lipidsearch_normalized_df.columns

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
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
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_high_cov
    ):
        """Concentration column count is preserved through QC steps (no sample removal)."""
        original_conc_count = len(get_concentration_columns(lipidsearch_normalized_df))

        _, mean_area_df, _ = qc_box_plot_data(
            lipidsearch_normalized_df, lipidsearch_experiment
        )
        assert mean_area_df.shape[1] == original_conc_count

        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )
        assert len(get_concentration_columns(filter_result.filtered_df)) == original_conc_count

    def test_sample_count_consistent_across_results(
        self, lipidsearch_normalized_df, lipidsearch_experiment
    ):
        """Box plots and PCA see the same samples."""
        samples, _, _ = qc_box_plot_data(lipidsearch_normalized_df, lipidsearch_experiment)
        pca_df = qc_pca(lipidsearch_normalized_df, lipidsearch_experiment)

        assert samples == pca_df['Sample'].tolist()

    def test_non_negative_concentration_values(self, lipidsearch_normalized_df):
        """All concentration values are non-negative."""
        conc_cols = get_concentration_columns(lipidsearch_normalized_df)
        for col in conc_cols:
            assert (lipidsearch_normalized_df[col] >= 0).all(), (
                f"Negative values found in {col}"
            )

    def test_identifier_columns_preserved_after_all_steps(
        self, lipidsearch_normalized_df, lipidsearch_experiment, lipidsearch_high_cov
    ):
        """LipidMolec and ClassKey are present after BQC filter and sample removal."""
        filter_result = QualityCheckWorkflow.apply_bqc_filter(
            lipidsearch_normalized_df, lipidsearch_high_cov
        )
        removal_result = QualityCheckWorkflow.remove_samples(
            filter_result.filtered_df, lipidsearch_experiment,
            samples_to_remove=['s1'],
        )
        for column in ('LipidMolec', 'ClassKey'):
            assert column in filter_result.filtered_df.columns
            assert column in removal_result.updated_df.columns


# =============================================================================
# TEST CLASS: TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests edge cases using synthetic QC-ready DataFrames."""

    def test_single_lipid_box_plots(self, single_lipid_df, four_sample_experiment):
        """Box plots work with a single lipid species."""
        samples, mean_area_df, _ = qc_box_plot_data(
            single_lipid_df, four_sample_experiment
        )
        assert mean_area_df.shape[0] == 1
        assert len(samples) == 4

    def test_single_lipid_pca_raises(self, single_lipid_df, four_sample_experiment):
        """PCA with a single lipid (1 feature) raises ValueError because
        n_components=2 exceeds min(n_samples, n_features)=1."""
        with pytest.raises(ValueError):
            qc_pca(single_lipid_df, four_sample_experiment)

    def test_two_samples_minimum_pca(self, two_sample_df, two_sample_experiment):
        """PCA works at minimum boundary (2 samples)."""
        assert len(qc_pca(two_sample_df, two_sample_experiment)) == 2

    def test_uniform_data_bqc(self, uniform_df):
        """Uniform data (all identical values) has CoV=0, so no high-CoV lipids."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'BQC'],
            number_of_samples_list=[2, 2],
        )

        _, reliable_pct, high_cov = qc_bqc_assessment(uniform_df, experiment, 'BQC')

        assert high_cov == []
        assert reliable_pct == 100.0

    def test_high_cov_all_lipids_filter(self, high_cov_all_df):
        """When all lipids have high CoV, filtering removes them all."""
        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'BQC'],
            number_of_samples_list=[2, 2],
        )

        _, _, high_cov = qc_bqc_assessment(high_cov_all_df, experiment, 'BQC')
        assert len(high_cov) == 3  # All 3 lipids

        filter_result = QualityCheckWorkflow.apply_bqc_filter(high_cov_all_df, high_cov)
        assert filter_result.lipids_after == 0
        assert len(filter_result.filtered_df) == 0


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

    def test_empty_dataframe_validation(self, four_sample_experiment):
        """Validation fails on empty DataFrame."""
        errors = QualityCheckWorkflow.validate_inputs(pd.DataFrame(), four_sample_experiment)
        assert len(errors) > 0
