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
