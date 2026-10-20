"""
Shared fixtures and helpers for integration tests.

Centralizes sample dataset loading to eliminate duplication across
test_module1_pipeline.py, test_module2_pipeline.py, and test_module3_pipeline.py.
"""

import pandas as pd
from pathlib import Path

from app.services.format_detection import DataFormat
from app.services.data_standardization import DataStandardizationService


SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / 'sample_datasets'


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


def get_intensity_columns(df: pd.DataFrame) -> list:
    """Get list of intensity column names from DataFrame."""
    return [col for col in df.columns if col.startswith('intensity[')]


def get_concentration_columns(df: pd.DataFrame) -> list:
    """Get list of concentration column names from DataFrame."""
    return [col for col in df.columns if col.startswith('concentration[')]


# =============================================================================
# Quality Check steps as the app runs them (StreamlitAdapter.run_box_plots /
# run_bqc_scatter / run_correlation / run_pca, without the caching)
# =============================================================================

def qc_box_plot_data(df: pd.DataFrame, experiment):
    """Returns (samples, mean_area_df, missing_values_percent)."""
    from app.services.plotting.box_plot import BoxPlotService
    from app.services.validation import get_matching_concentration_columns

    samples = get_matching_concentration_columns(df, experiment)
    mean_area_df = BoxPlotService.create_mean_area_df(df, samples)
    return samples, mean_area_df, BoxPlotService.calculate_missing_values_percentage(mean_area_df)


def qc_bqc_assessment(df: pd.DataFrame, experiment, bqc_label: str, cov_threshold: float = 30):
    """Returns (prepared_df, reliable_data_percent, high_cov_lipids): the CoV
    plot data and the lipids the BQC filter removes."""
    from app.services.plotting.bqc_plotter import BQCPlotterService
    from app.workflows.quality_check import QualityCheckWorkflow

    _, prepared_df, reliable_pct = BQCPlotterService.generate_and_display_cov_plot(
        df, experiment, experiment.conditions_list.index(bqc_label),
        cov_threshold=cov_threshold,
    )
    return prepared_df, reliable_pct, QualityCheckWorkflow.high_cov_lipids(
        prepared_df, cov_threshold,
    )


def qc_correlation(df: pd.DataFrame, experiment, condition: str,
                   sample_type: str = 'biological replicates'):
    """Returns (correlation_df, threshold) for one condition."""
    from app.services.plotting.correlation import CorrelationPlotterService

    mean_area_df = CorrelationPlotterService.prepare_data_for_correlation(
        df, experiment.individual_samples_list,
        experiment.conditions_list.index(condition),
    )
    correlation_df, _, threshold = CorrelationPlotterService.compute_correlation(
        mean_area_df, sample_type,
    )
    return correlation_df, threshold


def qc_pca(df: pd.DataFrame, experiment) -> pd.DataFrame:
    """Returns the PCA table (PC1, PC2, Sample, Condition)."""
    from app.services.plotting.pca import PCAPlotterService

    _, pca_df = PCAPlotterService.plot_pca(
        df, experiment.full_samples_list, experiment.extensive_conditions_list,
    )
    return pca_df


def zero_filter_as_app(df: pd.DataFrame, experiment, data_format: DataFormat,
                       bqc_label=None, config=None):
    """Zero filtering as the Data Processing page applies it
    (``ui/zero_filtering.py``), by default with its default thresholds.

    Returns the ZeroFilteringResult.
    """
    from app.constants import (
        LIPIDSEARCH_DETECTION_THRESHOLD,
        ZERO_FILTER_BQC_DEFAULT,
        ZERO_FILTER_NON_BQC_DEFAULT,
    )
    from app.services.zero_filtering import ZeroFilterConfig, ZeroFilteringService

    if config is None:
        config = ZeroFilterConfig(
            detection_threshold=(
                LIPIDSEARCH_DETECTION_THRESHOLD
                if data_format == DataFormat.LIPIDSEARCH else 0.0
            ),
            bqc_threshold=ZERO_FILTER_BQC_DEFAULT / 100,
            non_bqc_threshold=ZERO_FILTER_NON_BQC_DEFAULT / 100,
        )
    return ZeroFilteringService.filter_zeros(
        df=df, experiment=experiment, config=config, bqc_label=bqc_label,
    )
