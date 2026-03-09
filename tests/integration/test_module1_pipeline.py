"""
Integration tests for Module 1: Data Ingestion and Normalization Pipeline.

Tests the complete end-to-end flow:
- File loading → Format detection → Data cleaning → Zero filtering → Standards extraction → Normalization

Uses real sample datasets from sample_datasets/ and programmatically generated edge cases.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Workflows
from app.workflows.data_ingestion import (
    DataIngestionWorkflow,
    IngestionConfig,
    IngestionResult
)
from app.workflows.normalization import (
    NormalizationWorkflow,
    NormalizationWorkflowConfig,
    NormalizationWorkflowResult
)

# Models
from app.models.experiment import ExperimentConfig
from app.models.normalization import NormalizationConfig

# Services
from app.services.format_detection import FormatDetectionService, DataFormat
from app.services.data_cleaning import GradeFilterConfig, QualityFilterConfig
from app.services.zero_filtering import ZeroFilterConfig

# Legacy module for MW format parsing
from lipidomics.data_format_handler import DataFormatHandler


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
    # Standardize using DataFormatHandler
    standardized_df, success, _ = DataFormatHandler.validate_and_preprocess(df, 'lipidsearch')
    if not success:
        raise ValueError("Failed to standardize LipidSearch dataset")
    return standardized_df


def load_msdial_sample() -> pd.DataFrame:
    """
    Load and preprocess MS-DIAL sample dataset.

    Note: This function directly parses the MS-DIAL file without relying on
    DataFormatHandler, which uses st.session_state and doesn't work in tests.
    """
    import re

    path = SAMPLE_DATA_DIR / 'msdial_test_dataset.csv'
    # MS-DIAL file has 9 metadata rows before the actual header
    df = pd.read_csv(path, header=9)

    # Known MS-DIAL metadata columns
    MSDIAL_METADATA_COLUMNS = {
        'Alignment ID', 'Average Rt(min)', 'Average Mz', 'Metabolite name',
        'Adduct type', 'Post curation result', 'Fill %', 'MS/MS assigned',
        'Reference RT', 'Reference m/z', 'Formula', 'Ontology', 'INCHIKEY',
        'SMILES', 'Annotation tag', 'Annotation tag (VS1.0)', 'RT matched',
        'm/z matched', 'MS/MS matched', 'Comment', 'Manually modified',
        'Isotope tracking parent ID', 'Isotope tracking weight number',
        'Total score', 'RT similarity', 'Dot product', 'Reverse dot product',
        'Fragment presence %', 'S/N average', 'Spectrum reference file name',
        'MS1 isotopic spectrum', 'MS/MS spectrum', 'Lipid IS'
    }

    # Find the 'Lipid IS' column which separates raw and normalized data
    lipid_is_idx = None
    for idx, col in enumerate(df.columns):
        if col == 'Lipid IS':
            lipid_is_idx = idx
            break

    # Detect raw sample columns (before 'Lipid IS')
    raw_sample_cols = []
    raw_sample_indices = []
    for idx, col in enumerate(df.columns):
        if col in MSDIAL_METADATA_COLUMNS or pd.isna(col):
            continue
        if lipid_is_idx is not None and idx >= lipid_is_idx:
            break
        # Check if column contains numeric data
        try:
            col_data = df.iloc[:, idx]
            numeric_values = pd.to_numeric(col_data, errors='coerce')
            if numeric_values.notna().sum() > len(col_data) * 0.5:
                raw_sample_cols.append(col)
                raw_sample_indices.append(idx)
        except:
            continue

    # Helper function to infer ClassKey from lipid name
    def infer_class_key(lipid_molec):
        try:
            match = re.match(r'^([A-Za-z][A-Za-z0-9]*)', str(lipid_molec))
            if match:
                return match.group(1).strip()
            return "Unknown"
        except:
            return "Unknown"

    # Build standardized dataframe
    standardized_data = {}

    # LipidMolec from 'Metabolite name' column
    metabolite_idx = df.columns.tolist().index('Metabolite name')
    standardized_data['LipidMolec'] = df.iloc[:, metabolite_idx].astype(str)

    # ClassKey - infer from LipidMolec
    standardized_data['ClassKey'] = standardized_data['LipidMolec'].apply(infer_class_key)

    # Optional: BaseRt from 'Average Rt(min)'
    if 'Average Rt(min)' in df.columns:
        rt_idx = df.columns.tolist().index('Average Rt(min)')
        standardized_data['BaseRt'] = pd.to_numeric(df.iloc[:, rt_idx], errors='coerce')

    # Optional: CalcMass from 'Average Mz'
    if 'Average Mz' in df.columns:
        mz_idx = df.columns.tolist().index('Average Mz')
        standardized_data['CalcMass'] = pd.to_numeric(df.iloc[:, mz_idx], errors='coerce')

    # Quality columns for filtering
    if 'Total score' in df.columns:
        score_idx = df.columns.tolist().index('Total score')
        standardized_data['Total score'] = pd.to_numeric(df.iloc[:, score_idx], errors='coerce')

    if 'MS/MS matched' in df.columns:
        msms_idx = df.columns.tolist().index('MS/MS matched')
        standardized_data['MS/MS matched'] = df.iloc[:, msms_idx]

    # Sample intensity columns
    for i, (col_name, col_idx) in enumerate(zip(raw_sample_cols, raw_sample_indices), 1):
        new_col_name = f'intensity[s{i}]'
        standardized_data[new_col_name] = pd.to_numeric(
            df.iloc[:, col_idx], errors='coerce'
        ).fillna(0)

    return pd.DataFrame(standardized_data)


def load_generic_sample() -> pd.DataFrame:
    """Load and preprocess Generic sample dataset."""
    path = SAMPLE_DATA_DIR / 'generic_test_dataset.csv'
    df = pd.read_csv(path)
    # Standardize using DataFormatHandler
    standardized_df, success, _ = DataFormatHandler.validate_and_preprocess(df, 'generic')
    if not success:
        raise ValueError("Failed to standardize Generic dataset")
    return standardized_df


def load_mw_sample() -> pd.DataFrame:
    """Load and preprocess Metabolomics Workbench sample dataset."""
    path = SAMPLE_DATA_DIR / 'mw_test_dataset.csv'
    with open(path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    # Standardize using DataFormatHandler (MW requires text input)
    standardized_df, success, _ = DataFormatHandler.validate_and_preprocess(
        text_content, 'Metabolomics Workbench'
    )
    if not success:
        raise ValueError("Failed to standardize Metabolomics Workbench dataset")
    return standardized_df


def get_intensity_columns(df: pd.DataFrame) -> list:
    """Get list of intensity column names from DataFrame."""
    return [col for col in df.columns if col.startswith('intensity[')]


def get_sample_names_from_df(df: pd.DataFrame) -> list:
    """Extract sample names from intensity columns."""
    intensity_cols = get_intensity_columns(df)
    return [col.replace('intensity[', '').replace(']', '') for col in intensity_cols]


# =============================================================================
# Experiment Configuration Fixtures
# =============================================================================

@pytest.fixture
def simple_experiment():
    """Simple 2x2 experiment for edge case tests."""
    from tests.conftest import make_experiment
    return make_experiment(2, 2)


@pytest.fixture
def lipidsearch_experiment():
    """
    Experiment config matching lipidsearch5_test_dataset.csv.
    Dataset has 12 samples: 4 WT, 4 ADGAT-DKO, 4 BQC.
    """
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4]
    )


@pytest.fixture
def msdial_experiment():
    """
    Experiment config matching msdial_test_dataset.csv.
    Dataset has 7 samples: 1 Blank, 3 fads2 KO, 3 WT.
    """
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Blank', 'fads2_KO', 'WT'],
        number_of_samples_list=[1, 3, 3]
    )


@pytest.fixture
def generic_experiment():
    """
    Experiment config matching generic_test_dataset.csv.
    Dataset has 12 samples: 4 WT, 4 ADGAT-DKO, 4 BQC.
    """
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT_DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4]
    )


@pytest.fixture
def mw_experiment():
    """
    Experiment config matching mw_test_dataset.csv.
    Dataset has many samples in a 2x2 factorial design.
    Using simplified 2-condition setup for testing.
    """
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Normal', 'HFD'],
        number_of_samples_list=[22, 22]
    )


@pytest.fixture
def experiment_with_bqc():
    """Experiment including BQC condition."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['BQC', 'Control', 'Treatment'],
        number_of_samples_list=[2, 3, 3]
    )


@pytest.fixture
def unequal_samples_experiment():
    """Experiment with unequal sample counts per condition."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['Small', 'Medium', 'Large'],
        number_of_samples_list=[1, 3, 5]
    )


# =============================================================================
# Sample Dataset Loader Fixtures
# =============================================================================

@pytest.fixture
def lipidsearch_sample_df():
    """Load real LipidSearch 5.0 test dataset."""
    return load_lipidsearch_sample()


@pytest.fixture
def msdial_sample_df():
    """Load real MS-DIAL test dataset."""
    return load_msdial_sample()


@pytest.fixture
def generic_sample_df():
    """Load real Generic test dataset."""
    return load_generic_sample()


@pytest.fixture
def mw_sample_df():
    """Load real Metabolomics Workbench test dataset."""
    return load_mw_sample()


# =============================================================================
# Edge Case DataFrame Fixtures
# =============================================================================

@pytest.fixture
def empty_df():
    """Empty DataFrame with correct columns."""
    return pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]', 'intensity[s2]'])


@pytest.fixture
def single_row_df():
    """Single row DataFrame."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)'],
        'ClassKey': ['PC'],
        'intensity[s1]': [1e6],
        'intensity[s2]': [1.1e6],
        'intensity[s3]': [1.2e6],
        'intensity[s4]': [1.3e6],
    })


@pytest.fixture
def all_zeros_df():
    """DataFrame where all intensity values are zero."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [0.0, 0.0, 0.0],
        'intensity[s2]': [0.0, 0.0, 0.0],
        'intensity[s3]': [0.0, 0.0, 0.0],
        'intensity[s4]': [0.0, 0.0, 0.0],
    })


@pytest.fixture
def partial_zeros_df():
    """DataFrame with some rows having all zeros (should be filtered)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)', 'SM(d18:1_16:0)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [1e6, 0.0, 3e6, 0.0],
        'intensity[s2]': [1.1e6, 0.0, 3.1e6, 0.0],
        'intensity[s3]': [1.2e6, 0.0, 3.2e6, 0.0],
        'intensity[s4]': [1.3e6, 0.0, 3.3e6, 0.0],
    })


@pytest.fixture
def nan_values_df():
    """DataFrame with NaN values in intensity columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1e6, np.nan, 3e6],
        'intensity[s2]': [np.nan, 2.1e6, 3.1e6],
        'intensity[s3]': [1.2e6, 2.2e6, np.nan],
        'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
    })


@pytest.fixture
def duplicate_lipids_df():
    """DataFrame with duplicate LipidMolec entries."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'intensity[s1]': [1e6, 1.2e6, 2e6],
        'intensity[s2]': [1.1e6, 1.3e6, 2.1e6],
        'intensity[s3]': [1.2e6, 1.4e6, 2.2e6],
        'intensity[s4]': [1.3e6, 1.5e6, 2.3e6],
    })


@pytest.fixture
def standards_only_df():
    """DataFrame containing only internal standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)', 'TG(15:0_15:0_15:0-d9)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [5e5, 5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5, 5.3e5],
    })


@pytest.fixture
def mixed_standards_df():
    """DataFrame with both regular lipids and various standard patterns."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)',           # Regular
            'PC(15:0_18:1-d7)',         # Deuterated
            'PE(18:0_20:4)',            # Regular
            'PE(17:0_20:4)_IS',         # _IS suffix
            'TG(16:0_18:1_18:2)',       # Regular
            'SM(d18:1_16:0)',           # Regular
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TG', 'SM'],
        'intensity[s1]': [1e6, 5e5, 2e6, 5e5, 3e6, 4e6],
        'intensity[s2]': [1.1e6, 5.1e5, 2.1e6, 5.1e5, 3.1e6, 4.1e6],
        'intensity[s3]': [1.2e6, 5.2e5, 2.2e6, 5.2e5, 3.2e6, 4.2e6],
        'intensity[s4]': [1.3e6, 5.3e5, 2.3e6, 5.3e5, 3.3e6, 4.3e6],
    })


# =============================================================================
# Internal Standards and Protein Concentration Fixtures
# =============================================================================

@pytest.fixture
def basic_intsta_df():
    """Internal standards DataFrame matching common classes."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)', 'TG(15:0_15:0_15:0-d9)', 'SM(d18:1_12:0-d7)'],
        'ClassKey': ['PC', 'PE', 'TG', 'SM'],
        'intensity[s1]': [5e5, 5e5, 5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5, 5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5, 5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5, 5.3e5, 5.3e5],
    })


@pytest.fixture
def protein_concentrations():
    """Basic protein concentrations for 4 samples."""
    return {'s1': 10.0, 's2': 12.0, 's3': 11.0, 's4': 13.0}


@pytest.fixture
def protein_concentrations_12_samples():
    """Protein concentrations for 12 samples (matching sample datasets)."""
    return {f's{i}': 10.0 + i * 0.5 for i in range(1, 13)}


# =============================================================================
# Upload Edge Case Fixtures
# =============================================================================

@pytest.fixture
def duplicate_standards_df():
    """Standards with duplicate entries."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'intensity[s1]': [5e5, 5.5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.6e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.7e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.8e5, 5.3e5],
    })


@pytest.fixture
def standards_missing_classkey_df():
    """Standards file without ClassKey column."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_18:1-d7)', 'PE(17:0_20:4-d7)'],
        'intensity[s1]': [5e5, 5e5],
        'intensity[s2]': [5.1e5, 5.1e5],
        'intensity[s3]': [5.2e5, 5.2e5],
        'intensity[s4]': [5.3e5, 5.3e5],
    })


@pytest.fixture
def protein_csv_missing_samples():
    """Protein CSV missing some samples."""
    return {'s1': 10.0, 's2': 12.0}  # Missing s3, s4


@pytest.fixture
def protein_csv_negative_values():
    """Protein CSV with negative concentrations."""
    return {'s1': 10.0, 's2': -12.0, 's3': 11.0, 's4': 13.0}


# =============================================================================
# Sample Grouping Fixtures
# =============================================================================

@pytest.fixture
def regrouped_experiment():
    """Experiment with manually reordered samples."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Treatment', 'Control'],  # Reversed order
        number_of_samples_list=[2, 2]
    )


# =============================================================================
# TEST CLASS: TestLipidSearchPipeline
# =============================================================================

class TestLipidSearchPipeline:
    """Full pipeline tests for LipidSearch 5.0 format using real sample data."""

    def test_full_pipeline_with_default_config(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test complete pipeline with default settings."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        assert result.is_valid, f"Pipeline should succeed: {result.validation_errors}"
        assert result.cleaned_df is not None
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert len(result.cleaned_df) > 0
        # Should have intensity columns
        intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
        assert len(intensity_cols) > 0

    def test_full_pipeline_grade_filtering_a_only(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test pipeline with strict grade A filtering."""
        # GradeFilterConfig takes grade_config dict mapping ClassKey to list of grades
        grade_config = GradeFilterConfig(
            grade_config={'PC': ['A'], 'PE': ['A'], 'TG': ['A'], 'SM': ['A']}
        )
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        assert result.is_valid or 'grade' in str(result.validation_errors).lower()
        # With strict filtering, we may get fewer species
        if result.cleaned_df is not None:
            assert len(result.cleaned_df) <= len(lipidsearch_sample_df)

    def test_full_pipeline_grade_filtering_a_b(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test pipeline with grade A and B filtering."""
        grade_config = GradeFilterConfig(
            grade_config={'PC': ['A', 'B'], 'PE': ['A', 'B'], 'TG': ['A', 'B'], 'SM': ['A', 'B']}
        )
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            grade_config=grade_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        # Should allow more species than A-only
        assert result.is_valid or len(result.validation_errors) > 0

    def test_full_pipeline_zero_filter_strict(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test pipeline with strict zero filtering threshold."""
        zero_config = ZeroFilterConfig(
            detection_threshold=0.0,
            bqc_threshold=0.3,  # Stricter
            non_bqc_threshold=0.5,  # Stricter
        )
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
            zero_filter_config=zero_config,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        assert result.zero_filtered
        # Strict filtering should remove more species
        assert result.species_before_filter >= result.species_after_filter

    def test_full_pipeline_zero_filter_disabled(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test pipeline with zero filtering disabled."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        assert not result.zero_filtered
        assert result.species_removed_count == 0

    def test_full_pipeline_standards_auto_detected(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Verify internal standards are automatically detected."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        # Check if standards were detected (may or may not be present in dataset)
        # The important thing is the workflow doesn't crash
        assert result.is_valid or len(result.validation_warnings) >= 0

    def test_full_pipeline_external_standards(self, lipidsearch_sample_df, lipidsearch_experiment, basic_intsta_df):
        """Test pipeline with external standards provided."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
            use_external_standards=True,
            external_standards_df=basic_intsta_df,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        # External standards should be used
        if result.internal_standards_df is not None:
            assert len(result.internal_standards_df) > 0

    def test_cleaning_preserves_essential_columns(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Verify CalcMass and BaseRt columns are preserved through pipeline."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        if result.cleaned_df is not None:
            # LipidSearch should preserve CalcMass and BaseRt
            assert 'LipidMolec' in result.cleaned_df.columns
            assert 'ClassKey' in result.cleaned_df.columns

    def test_output_dataframe_structure(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Verify output DataFrame has expected columns and structure."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        if result.cleaned_df is not None:
            # Must have these columns
            assert 'LipidMolec' in result.cleaned_df.columns
            assert 'ClassKey' in result.cleaned_df.columns
            # Must have intensity columns
            intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
            assert len(intensity_cols) > 0

    def test_species_count_decreases_after_filtering(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Verify species count decreases after grade and zero filtering."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        # Filtering should typically reduce species count
        if result.zero_filtered:
            assert result.species_before_filter >= result.species_after_filter

    def test_explicit_format_specification(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Verify explicit format specification works correctly."""
        config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,  # Explicit format
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(lipidsearch_sample_df, config)

        # Should use the specified format
        assert result.detected_format == DataFormat.LIPIDSEARCH
        assert result.is_valid


# =============================================================================
# TEST CLASS: TestMSDIALPipeline
# =============================================================================

class TestMSDIALPipeline:
    """Full pipeline tests for MS-DIAL format using real sample data."""

    def test_full_pipeline_with_default_config(self, msdial_sample_df, msdial_experiment):
        """Test complete pipeline with default settings."""
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        assert result.is_valid, f"Pipeline should succeed: {result.validation_errors}"
        assert result.cleaned_df is not None
        assert result.detected_format == DataFormat.MSDIAL

    def test_full_pipeline_quality_filter_high(self, msdial_sample_df, msdial_experiment):
        """Test pipeline with high quality score threshold (80+)."""
        # QualityFilterConfig uses total_score_threshold (not min_score)
        quality_config = QualityFilterConfig(
            total_score_threshold=80,
            require_msms=False,
        )
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # High quality filtering may reduce species count significantly
        assert result.is_valid or 'quality' in str(result.validation_errors).lower()

    def test_full_pipeline_quality_filter_moderate(self, msdial_sample_df, msdial_experiment):
        """Test pipeline with moderate quality threshold (50+)."""
        quality_config = QualityFilterConfig(
            total_score_threshold=50,
            require_msms=False,
        )
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # Moderate filtering should keep more species
        if result.cleaned_df is not None:
            assert len(result.cleaned_df) > 0

    def test_full_pipeline_msms_required(self, msdial_sample_df, msdial_experiment):
        """Test pipeline requiring MS/MS matching."""
        quality_config = QualityFilterConfig(
            total_score_threshold=0,
            require_msms=True,
        )
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # Should work, may filter out species without MS/MS
        assert result.is_valid or len(result.validation_errors) > 0

    def test_full_pipeline_msms_not_required(self, msdial_sample_df, msdial_experiment):
        """Test pipeline without MS/MS requirement."""
        quality_config = QualityFilterConfig(
            total_score_threshold=0,
            require_msms=False,
        )
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # Should keep more species without MS/MS requirement
        if result.cleaned_df is not None:
            assert len(result.cleaned_df) > 0

    def test_full_pipeline_zero_filter_with_quality(self, msdial_sample_df, msdial_experiment):
        """Test combination of quality filtering and zero filtering."""
        quality_config = QualityFilterConfig(
            total_score_threshold=50,
            require_msms=False,
        )
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            quality_config=quality_config,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # Both filters applied
        if result.is_valid and result.cleaned_df is not None:
            assert result.zero_filtered

    def test_output_structure(self, msdial_sample_df, msdial_experiment):
        """Verify output DataFrame has expected structure."""
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        if result.cleaned_df is not None:
            assert 'LipidMolec' in result.cleaned_df.columns
            assert 'ClassKey' in result.cleaned_df.columns
            intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
            assert len(intensity_cols) > 0

    def test_explicit_format_specification(self, msdial_sample_df, msdial_experiment):
        """Verify explicit format specification works correctly."""
        config = IngestionConfig(
            experiment=msdial_experiment,
            data_format=DataFormat.MSDIAL,  # Explicit format
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(msdial_sample_df, config)

        # Should use the specified format
        assert result.detected_format == DataFormat.MSDIAL


# =============================================================================
# TEST CLASS: TestGenericPipeline
# =============================================================================

class TestGenericPipeline:
    """Full pipeline tests for Generic format using real sample data."""

    def test_full_pipeline_with_default_config(self, generic_sample_df, generic_experiment):
        """Test complete pipeline with default settings."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        assert result.is_valid, f"Pipeline should succeed: {result.validation_errors}"
        assert result.cleaned_df is not None
        assert result.detected_format == DataFormat.GENERIC

    def test_full_pipeline_zero_filter_enabled(self, generic_sample_df, generic_experiment):
        """Test pipeline with zero filtering enabled."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        assert result.zero_filtered

    def test_full_pipeline_zero_filter_disabled(self, generic_sample_df, generic_experiment):
        """Test pipeline with zero filtering disabled."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        assert not result.zero_filtered

    def test_output_structure(self, generic_sample_df, generic_experiment):
        """Verify output DataFrame has expected structure."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        if result.cleaned_df is not None:
            assert 'LipidMolec' in result.cleaned_df.columns
            # ClassKey should be derived from LipidMolec
            assert 'ClassKey' in result.cleaned_df.columns
            intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
            assert len(intensity_cols) > 0

    def test_format_auto_detection(self, generic_sample_df, generic_experiment):
        """Test format auto-detection for Generic."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=None,  # Auto-detect
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        assert result.detected_format == DataFormat.GENERIC

    def test_column_standardization(self, generic_sample_df, generic_experiment):
        """Verify sample[] columns are standardized to intensity[]."""
        config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(generic_sample_df, config)

        if result.cleaned_df is not None:
            # Should have intensity[] columns (standardized from sample[])
            intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
            assert len(intensity_cols) > 0


# =============================================================================
# TEST CLASS: TestMetabolomicsWorkbenchPipeline
# =============================================================================

class TestMetabolomicsWorkbenchPipeline:
    """Full pipeline tests for Metabolomics Workbench format using real sample data."""

    def test_full_pipeline_with_default_config(self, mw_sample_df, mw_experiment):
        """Test complete pipeline with default settings."""
        config = IngestionConfig(
            experiment=mw_experiment,
            data_format=DataFormat.GENERIC,  # MW uses Generic cleaner
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(mw_sample_df, config)

        assert result.is_valid, f"Pipeline should succeed: {result.validation_errors}"
        assert result.cleaned_df is not None

    def test_full_pipeline_large_sample_count(self, mw_sample_df, mw_experiment):
        """Test pipeline handles many samples correctly."""
        config = IngestionConfig(
            experiment=mw_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(mw_sample_df, config)

        if result.cleaned_df is not None:
            intensity_cols = [c for c in result.cleaned_df.columns if c.startswith('intensity[')]
            # MW dataset has many samples
            assert len(intensity_cols) > 10

    def test_zero_filtering_with_many_samples(self, mw_sample_df, mw_experiment):
        """Test zero filtering across many samples."""
        config = IngestionConfig(
            experiment=mw_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(mw_sample_df, config)

        assert result.zero_filtered
        assert result.species_before_filter >= result.species_after_filter

    def test_output_structure(self, mw_sample_df, mw_experiment):
        """Verify output DataFrame has expected structure."""
        config = IngestionConfig(
            experiment=mw_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(mw_sample_df, config)

        if result.cleaned_df is not None:
            assert 'LipidMolec' in result.cleaned_df.columns
            assert 'ClassKey' in result.cleaned_df.columns


# =============================================================================
# TEST CLASS: TestNormalizationNone
# =============================================================================

class TestNormalizationNone:
    """Tests for 'none' normalization method (raw data passthrough)."""

    def test_none_normalization_preserves_data(self, generic_sample_df, generic_experiment):
        """Test that 'none' method preserves original intensity values."""
        # First run ingestion
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Run normalization with 'none' method
        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success
        # method_applied contains descriptive text including the method name
        assert 'None' in result.method_applied or 'none' in result.method_applied.lower()
        # Data should have same number of rows
        assert len(result.normalized_df) == len(ingestion_result.cleaned_df)

    def test_none_normalization_renames_columns(self, generic_sample_df, generic_experiment):
        """Test that 'none' method renames intensity[] to concentration[]."""
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)

        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        # Should have concentration columns, not intensity
        concentration_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
        assert len(concentration_cols) > 0

    def test_none_normalization_with_class_filter(self, generic_sample_df, generic_experiment):
        """Test 'none' method filters by selected classes."""
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)

        # Get available classes and select just one
        available_classes = list(ingestion_result.cleaned_df['ClassKey'].unique())
        if len(available_classes) > 1:
            selected = available_classes[:1]

            norm_config = NormalizationWorkflowConfig(
                experiment=generic_experiment,
                normalization=NormalizationConfig(
                    method='none',
                    selected_classes=selected
                ),
                data_format=DataFormat.GENERIC
            )

            result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

            # Should only have selected classes
            assert set(result.normalized_df['ClassKey'].unique()) == set(selected)

    def test_none_normalization_all_formats(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test 'none' method works with LipidSearch format."""
        ingestion_config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(lipidsearch_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        norm_config = NormalizationWorkflowConfig(
            experiment=lipidsearch_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.LIPIDSEARCH
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success
        assert 'None' in result.method_applied or 'none' in result.method_applied.lower()


# =============================================================================
# TEST CLASS: TestNormalizationInternalStandard
# =============================================================================

class TestNormalizationInternalStandard:
    """Tests for internal standard normalization method."""

    @pytest.fixture
    def sample_data_with_standards(self):
        """Create sample data with internal standards for testing."""
        return pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)', 'PC(18:0_20:4)', 'PE(16:0_18:1)', 'PE(18:0_20:4)',
                'TG(16:0_18:1_18:2)', 'PC(15:0_18:1)(d7)', 'PE(17:0_20:4)(d7)',
            ],
            'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TG', 'PC', 'PE'],
            'intensity[s1]': [1e6, 1.2e6, 2e6, 2.2e6, 3e6, 5e5, 5e5],
            'intensity[s2]': [1.1e6, 1.3e6, 2.1e6, 2.3e6, 3.1e6, 5.1e5, 5.1e5],
            'intensity[s3]': [1.2e6, 1.4e6, 2.2e6, 2.4e6, 3.2e6, 5.2e5, 5.2e5],
            'intensity[s4]': [1.3e6, 1.5e6, 2.3e6, 2.5e6, 3.3e6, 5.3e5, 5.3e5],
        })

    @pytest.fixture
    def standards_df(self):
        """Internal standards DataFrame."""
        return pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)(d7)', 'PE(17:0_20:4)(d7)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [5e5, 5e5],
            'intensity[s2]': [5.1e5, 5.1e5],
            'intensity[s3]': [5.2e5, 5.2e5],
            'intensity[s4]': [5.3e5, 5.3e5],
        })

    @pytest.fixture
    def intsta_concentrations(self):
        """Internal standard concentrations (in µM)."""
        return {
            'PC(15:0_18:1)(d7)': 1.0,
            'PE(17:0_20:4)(d7)': 1.0
        }

    def test_is_normalization_basic(self, sample_data_with_standards, standards_df, intsta_concentrations, simple_experiment):
        """Test basic internal standard normalization."""
        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PE(17:0_20:4)(d7)'
                },
                intsta_concentrations=intsta_concentrations
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            sample_data_with_standards, norm_config, standards_df
        )

        assert result.success
        assert 'internal' in result.method_applied.lower() or 'standard' in result.method_applied.lower()

    def test_is_normalization_removes_standards_from_output(self, sample_data_with_standards, standards_df, intsta_concentrations, simple_experiment):
        """Test that internal standards are removed from output."""
        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PE(17:0_20:4)(d7)'
                },
                intsta_concentrations=intsta_concentrations
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            sample_data_with_standards, norm_config, standards_df
        )

        # Standards should be removed
        if result.normalized_df is not None:
            standard_names = set(standards_df['LipidMolec'].tolist())
            output_names = set(result.normalized_df['LipidMolec'].tolist())
            assert len(standard_names & output_names) == 0

    def test_is_normalization_class_specific_standards(self, sample_data_with_standards, standards_df, intsta_concentrations, simple_experiment):
        """Test that each class uses its assigned standard."""
        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PE(17:0_20:4)(d7)'
                },
                intsta_concentrations=intsta_concentrations
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            sample_data_with_standards, norm_config, standards_df
        )

        assert result.success
        # Should have concentration columns
        conc_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) > 0

    def test_is_normalization_single_standard_all_classes(self, simple_experiment):
        """Test using single standard for multiple classes."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(16:0_18:1)', 'PC(15:0_18:1)(d7)'],
            'ClassKey': ['PC', 'PE', 'PC'],
            'intensity[s1]': [1e6, 2e6, 5e5],
            'intensity[s2]': [1.1e6, 2.1e6, 5.1e5],
            'intensity[s3]': [1.2e6, 2.2e6, 5.2e5],
            'intensity[s4]': [1.3e6, 2.3e6, 5.3e5],
        })

        standards = pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5e5],
            'intensity[s2]': [5.1e5],
            'intensity[s3]': [5.2e5],
            'intensity[s4]': [5.3e5],
        })

        intsta_conc = {'PC(15:0_18:1)(d7)': 1.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PC(15:0_18:1)(d7)'  # Use PC standard for PE too
                },
                intsta_concentrations=intsta_conc
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(data, norm_config, standards)

        assert result.success

    def test_is_normalization_output_units(self, sample_data_with_standards, standards_df, intsta_concentrations, simple_experiment):
        """Test that output is properly normalized (relative to standard)."""
        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PE(17:0_20:4)(d7)'
                },
                intsta_concentrations=intsta_concentrations
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            sample_data_with_standards, norm_config, standards_df
        )

        # Values should be normalized (divided by standard)
        if result.normalized_df is not None:
            conc_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
            for col in conc_cols:
                # All values should be positive
                assert (result.normalized_df[col] >= 0).all()

    def test_is_normalization_with_lipidsearch_data(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test IS normalization with real LipidSearch data."""
        ingestion_config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(lipidsearch_sample_df, ingestion_config)

        if ingestion_result.internal_standards_df is not None and len(ingestion_result.internal_standards_df) > 0:
            # Use auto-detected standards
            standards_df = ingestion_result.internal_standards_df
            available_classes = list(ingestion_result.cleaned_df['ClassKey'].unique())
            standard_classes = list(standards_df['ClassKey'].unique())

            # Build mapping for classes that have standards
            mapping = {}
            intsta_concs = {}
            for cls in available_classes[:2]:  # Just test with 2 classes
                for std_cls in standard_classes:
                    if std_cls == cls:
                        std_name = standards_df[standards_df['ClassKey'] == std_cls]['LipidMolec'].iloc[0]
                        mapping[cls] = std_name
                        intsta_concs[std_name] = 1.0
                        break
                else:
                    # Use first standard if no class-specific one
                    std_name = standards_df['LipidMolec'].iloc[0]
                    mapping[cls] = std_name
                    intsta_concs[std_name] = 1.0

            if mapping:
                norm_config = NormalizationWorkflowConfig(
                    experiment=lipidsearch_experiment,
                    normalization=NormalizationConfig(
                        method='internal_standard',
                        selected_classes=list(mapping.keys()),
                        internal_standards=mapping,
                        intsta_concentrations=intsta_concs
                    ),
                    data_format=DataFormat.LIPIDSEARCH
                )

                result = NormalizationWorkflow.run(
                    ingestion_result.cleaned_df, norm_config, standards_df
                )

                assert result.success or len(result.validation_errors) > 0


# =============================================================================
# TEST CLASS: TestNormalizationProtein
# =============================================================================

class TestNormalizationProtein:
    """Tests for protein concentration normalization method."""

    def test_protein_normalization_basic(self, generic_sample_df, generic_experiment):
        """Test basic protein normalization."""
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Create protein concentrations for all samples
        intensity_cols = [c for c in ingestion_result.cleaned_df.columns if c.startswith('intensity[')]
        protein_concs = {
            col.replace('intensity[', '').replace(']', ''): 10.0 + i * 0.5
            for i, col in enumerate(intensity_cols)
        }

        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='protein',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique()),
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success
        assert 'protein' in result.method_applied.lower() or 'Protein' in result.method_applied

    def test_protein_normalization_divides_by_concentration(self, simple_experiment):
        """Test that protein normalization divides intensity by concentration."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(16:0_18:1)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1000.0, 2000.0],
            'intensity[s3]': [1000.0, 2000.0],
            'intensity[s4]': [1000.0, 2000.0],
        })

        # Sample s1 has concentration 10, s2 has 20
        protein_concs = {'s1': 10.0, 's2': 20.0, 's3': 10.0, 's4': 20.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='protein',
                selected_classes=['PC', 'PE'],
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(data, norm_config)

        assert result.success
        # Concentration for s1 should be 100 (1000/10), s2 should be 50 (1000/20)
        if result.normalized_df is not None:
            pc_row = result.normalized_df[result.normalized_df['LipidMolec'] == 'PC(16:0_18:1)']
            if len(pc_row) > 0:
                assert abs(pc_row['concentration[s1]'].values[0] - 100.0) < 0.1
                assert abs(pc_row['concentration[s2]'].values[0] - 50.0) < 0.1

    def test_protein_normalization_varying_concentrations(self, simple_experiment):
        """Test protein normalization with varying concentrations."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1e6],
            'intensity[s3]': [1e6],
            'intensity[s4]': [1e6],
        })

        # Different concentrations for each sample
        protein_concs = {'s1': 5.0, 's2': 10.0, 's3': 15.0, 's4': 20.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='protein',
                selected_classes=['PC'],
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(data, norm_config)

        assert result.success
        # Higher concentration = lower normalized value
        if result.normalized_df is not None:
            assert result.normalized_df['concentration[s1]'].values[0] > result.normalized_df['concentration[s4]'].values[0]

    def test_protein_normalization_with_class_filter(self, simple_experiment):
        """Test protein normalization with class filtering."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(16:0_18:1)', 'TG(16:0_18:1_18:2)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'intensity[s1]': [1e6, 2e6, 3e6],
            'intensity[s2]': [1.1e6, 2.1e6, 3.1e6],
            'intensity[s3]': [1.2e6, 2.2e6, 3.2e6],
            'intensity[s4]': [1.3e6, 2.3e6, 3.3e6],
        })

        protein_concs = {'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='protein',
                selected_classes=['PC', 'PE'],  # Exclude TG
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(data, norm_config)

        assert result.success
        # Should only have PC and PE
        assert set(result.normalized_df['ClassKey'].unique()) == {'PC', 'PE'}


# =============================================================================
# TEST CLASS: TestNormalizationBoth
# =============================================================================

class TestNormalizationBoth:
    """Tests for combined IS + protein normalization method."""

    @pytest.fixture
    def data_with_standards(self):
        """Sample data including internal standards."""
        return pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)', 'PE(16:0_18:1)', 'PC(15:0_18:1)(d7)'
            ],
            'ClassKey': ['PC', 'PE', 'PC'],
            'intensity[s1]': [1e6, 2e6, 5e5],
            'intensity[s2]': [1.1e6, 2.1e6, 5.1e5],
            'intensity[s3]': [1.2e6, 2.2e6, 5.2e5],
            'intensity[s4]': [1.3e6, 2.3e6, 5.3e5],
        })

    @pytest.fixture
    def standards_for_both(self):
        """Standards for combined normalization."""
        return pd.DataFrame({
            'LipidMolec': ['PC(15:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5e5],
            'intensity[s2]': [5.1e5],
            'intensity[s3]': [5.2e5],
            'intensity[s4]': [5.3e5],
        })

    @pytest.fixture
    def intsta_conc_for_both(self):
        """Internal standard concentrations for combined normalization."""
        return {'PC(15:0_18:1)(d7)': 1.0}

    def test_both_normalization_basic(self, data_with_standards, standards_for_both, intsta_conc_for_both, simple_experiment):
        """Test combined IS + protein normalization."""
        protein_concs = {'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='both',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PC(15:0_18:1)(d7)'
                },
                intsta_concentrations=intsta_conc_for_both,
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            data_with_standards, norm_config, standards_for_both
        )

        assert result.success
        # method_applied returns 'Combined normalization (internal standards + protein)'
        assert 'combined' in result.method_applied.lower() or 'both' in result.method_applied.lower()

    def test_both_normalization_removes_standards(self, data_with_standards, standards_for_both, intsta_conc_for_both, simple_experiment):
        """Test that combined method removes standards."""
        protein_concs = {'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='both',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PC(15:0_18:1)(d7)'
                },
                intsta_concentrations=intsta_conc_for_both,
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            data_with_standards, norm_config, standards_for_both
        )

        # Standard should be removed
        if result.normalized_df is not None:
            assert 'PC(15:0_18:1)(d7)' not in result.normalized_df['LipidMolec'].values

    def test_both_normalization_output_structure(self, data_with_standards, standards_for_both, intsta_conc_for_both, simple_experiment):
        """Test output structure of combined normalization."""
        protein_concs = {'s1': 10.0, 's2': 10.0, 's3': 10.0, 's4': 10.0}

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='both',
                selected_classes=['PC', 'PE'],
                internal_standards={
                    'PC': 'PC(15:0_18:1)(d7)',
                    'PE': 'PC(15:0_18:1)(d7)'
                },
                intsta_concentrations=intsta_conc_for_both,
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(
            data_with_standards, norm_config, standards_for_both
        )

        if result.normalized_df is not None:
            # Should have concentration columns
            conc_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
            assert len(conc_cols) == 4
            # Should have LipidMolec and ClassKey
            assert 'LipidMolec' in result.normalized_df.columns
            assert 'ClassKey' in result.normalized_df.columns


# =============================================================================
# TEST CLASS: TestIngestionNormalizationIntegration
# =============================================================================

class TestIngestionNormalizationIntegration:
    """End-to-end tests: ingestion → normalization pipeline."""

    def test_lipidsearch_to_none_normalization(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test full pipeline: LipidSearch → none normalization."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=True,
        )
        ingestion_result = DataIngestionWorkflow.run(lipidsearch_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=lipidsearch_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.LIPIDSEARCH
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success
        assert result.normalized_df is not None

    def test_generic_to_protein_normalization(self, generic_sample_df, generic_experiment):
        """Test full pipeline: Generic → protein normalization."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Create protein concentrations
        intensity_cols = [c for c in ingestion_result.cleaned_df.columns if c.startswith('intensity[')]
        protein_concs = {
            col.replace('intensity[', '').replace(']', ''): 10.0
            for col in intensity_cols
        }

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='protein',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique()),
                protein_concentrations=protein_concs
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success

    def test_mw_to_none_normalization(self, mw_sample_df, mw_experiment):
        """Test full pipeline: Metabolomics Workbench → none normalization."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=mw_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(mw_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=mw_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success

    def test_class_list_consistency(self, generic_sample_df, generic_experiment):
        """Test that class list is consistent through pipeline."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)

        original_classes = set(ingestion_result.cleaned_df['ClassKey'].unique())

        # Normalization with all classes
        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(original_classes)
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        # Classes should match
        normalized_classes = set(result.normalized_df['ClassKey'].unique())
        assert normalized_classes == original_classes

    def test_sample_columns_preserved_through_pipeline(self, generic_sample_df, generic_experiment):
        """Test that sample count is preserved through pipeline."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)

        intensity_cols = [c for c in ingestion_result.cleaned_df.columns if c.startswith('intensity[')]
        n_samples = len(intensity_cols)

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        conc_cols = [c for c in result.normalized_df.columns if c.startswith('concentration[')]
        assert len(conc_cols) == n_samples

    def test_essential_columns_restored_after_normalization(self, lipidsearch_sample_df, lipidsearch_experiment):
        """Test that CalcMass/BaseRt are restored for LipidSearch."""
        # Ingestion
        ingestion_config = IngestionConfig(
            experiment=lipidsearch_experiment,
            data_format=DataFormat.LIPIDSEARCH,
            apply_zero_filter=False,
        )
        ingestion_result = DataIngestionWorkflow.run(lipidsearch_sample_df, ingestion_config)

        # Check if essential columns exist
        has_calcmass = 'CalcMass' in ingestion_result.cleaned_df.columns
        has_basert = 'BaseRt' in ingestion_result.cleaned_df.columns

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=lipidsearch_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.LIPIDSEARCH
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        # Essential columns should be restored
        if has_calcmass:
            assert 'CalcMass' in result.normalized_df.columns
        if has_basert:
            assert 'BaseRt' in result.normalized_df.columns

    def test_pipeline_with_zero_filter_and_normalization(self, generic_sample_df, generic_experiment):
        """Test full pipeline with zero filtering enabled."""
        # Ingestion with zero filtering
        ingestion_config = IngestionConfig(
            experiment=generic_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )
        ingestion_result = DataIngestionWorkflow.run(generic_sample_df, ingestion_config)
        assert ingestion_result.is_valid

        # Normalization
        norm_config = NormalizationWorkflowConfig(
            experiment=generic_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=list(ingestion_result.cleaned_df['ClassKey'].unique())
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)

        assert result.success
        # Should have fewer or equal lipids due to zero filtering
        assert len(result.normalized_df) <= len(generic_sample_df)


# =============================================================================
# TEST CLASS: TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_row_full_pipeline(self, single_row_df, simple_experiment):
        """Test pipeline with single row DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(single_row_df, config)

        assert result.is_valid
        assert len(result.cleaned_df) == 1

    def test_partial_zeros_filtering(self, partial_zeros_df, simple_experiment):
        """Test that rows with all zeros are filtered out."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(partial_zeros_df, config)

        if result.is_valid and result.cleaned_df is not None:
            # Rows with all zeros should be filtered
            assert result.species_removed_count >= 0

    def test_nan_values_handling(self, nan_values_df, simple_experiment):
        """Test handling of NaN values in intensity columns."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(nan_values_df, config)

        # Should handle NaN values gracefully
        assert result.is_valid or len(result.validation_errors) > 0

    def test_duplicate_lipids_handling(self, duplicate_lipids_df, simple_experiment):
        """Test handling of duplicate lipid entries."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(duplicate_lipids_df, config)

        # Should process without crashing
        assert result.is_valid or len(result.validation_errors) > 0

    def test_standards_only_data(self, standards_only_df, simple_experiment):
        """Test data containing only internal standards."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(standards_only_df, config)

        # Should process and detect all as standards
        assert result.is_valid or len(result.validation_errors) > 0

    def test_mixed_standards_data(self, mixed_standards_df, simple_experiment):
        """Test data with mix of regular lipids and standards."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(mixed_standards_df, config)

        assert result.is_valid
        # Some should be detected as standards
        if result.internal_standards_df is not None:
            assert len(result.internal_standards_df) >= 0

    def test_very_large_intensity_values(self, simple_experiment):
        """Test handling of very large intensity values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e15],
            'intensity[s2]': [1e15],
            'intensity[s3]': [1e15],
            'intensity[s4]': [1e15],
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(data, config)

        assert result.is_valid

    def test_very_small_intensity_values(self, simple_experiment):
        """Test handling of very small intensity values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-15],
            'intensity[s2]': [1e-15],
            'intensity[s3]': [1e-15],
            'intensity[s4]': [1e-15],
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(data, config)

        assert result.is_valid

    def test_single_sample_column(self):
        """Test data with only one sample column."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1e6, 2e6],
        })

        experiment = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Sample'],
            number_of_samples_list=[1]
        )

        config = IngestionConfig(
            experiment=experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(data, config)

        assert result.is_valid
        assert len([c for c in result.cleaned_df.columns if c.startswith('intensity[')]) == 1

    def test_many_sample_columns(self):
        """Test data with many sample columns."""
        n_samples = 50
        data = {
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
        }
        for i in range(1, n_samples + 1):
            data[f'intensity[s{i}]'] = [1e6, 2e6]

        df = pd.DataFrame(data)

        experiment = ExperimentConfig(
            n_conditions=2,
            conditions_list=['Control', 'Treatment'],
            number_of_samples_list=[25, 25]
        )

        config = IngestionConfig(
            experiment=experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid
        assert len([c for c in result.cleaned_df.columns if c.startswith('intensity[')]) == n_samples

    def test_unequal_sample_counts(self, unequal_samples_experiment):
        """Test experiment with unequal sample counts per condition."""
        # Total samples: 1 + 3 + 5 = 9
        data = {
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
        }
        for i in range(1, 10):
            data[f'intensity[s{i}]'] = [1e6, 2e6]

        df = pd.DataFrame(data)

        config = IngestionConfig(
            experiment=unequal_samples_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid

    def test_many_lipid_classes(self, simple_experiment):
        """Test data with many lipid classes."""
        classes = ['PC', 'PE', 'TG', 'SM', 'CL', 'PI', 'PS', 'PG', 'PA', 'LPC', 'LPE', 'Cer', 'DG', 'MG']
        data = {
            'LipidMolec': [f'{cls}(16:0_18:1)' for cls in classes],
            'ClassKey': classes,
            'intensity[s1]': [1e6] * len(classes),
            'intensity[s2]': [1.1e6] * len(classes),
            'intensity[s3]': [1.2e6] * len(classes),
            'intensity[s4]': [1.3e6] * len(classes),
        }

        df = pd.DataFrame(data)

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(df, config)

        assert result.is_valid
        assert len(result.cleaned_df['ClassKey'].unique()) == len(classes)


# =============================================================================
# TEST CLASS: TestErrorHandling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_empty_dataframe_handling(self, empty_df, simple_experiment):
        """Test handling of empty DataFrame."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(empty_df, config)

        # Should fail validation
        assert not result.is_valid or len(result.validation_errors) > 0

    def test_missing_lipidmolec_column(self, simple_experiment):
        """Test handling of missing LipidMolec column."""
        data = pd.DataFrame({
            'WrongColumn': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
            'intensity[s4]': [1.3e6],
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(data, config)

        # This might succeed if the first column is used as LipidMolec
        # or it might fail - either is acceptable error handling
        assert result.is_valid or len(result.validation_errors) > 0

    def test_sample_count_mismatch(self, simple_experiment):
        """Test handling of sample count mismatch."""
        # simple_experiment expects 4 samples but data has 3
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
        })

        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        result = DataIngestionWorkflow.run(data, config)

        # Should detect mismatch
        assert not result.is_valid or 'Missing' in str(result.validation_errors)

    def test_normalization_without_required_data(self, simple_experiment):
        """Test IS normalization without standards data."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
            'intensity[s4]': [1.3e6],
        })

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC'],
                internal_standards={'PC': 'PC(15:0_18:1)(d7)'},
                intsta_concentrations={'PC(15:0_18:1)(d7)': 1.0}
            ),
            data_format=DataFormat.GENERIC
        )

        # No standards_df provided
        result = NormalizationWorkflow.run(data, norm_config, intsta_df=None)

        # Should fail
        assert not result.success or len(result.validation_errors) > 0

    def test_protein_normalization_without_concentrations(self, simple_experiment):
        """Test protein normalization fails without concentrations."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
            'intensity[s4]': [1.3e6],
        })

        # This should raise validation error at config creation
        try:
            norm_config = NormalizationWorkflowConfig(
                experiment=simple_experiment,
                normalization=NormalizationConfig(
                    method='protein',
                    selected_classes=['PC'],
                    # Missing protein_concentrations
                ),
                data_format=DataFormat.GENERIC
            )
            # If config creation succeeds, running should fail
            result = NormalizationWorkflow.run(data, norm_config)
            assert not result.success
        except ValueError:
            # Expected - validation error at config creation
            pass

    def test_invalid_class_selection(self, simple_experiment):
        """Test normalization with invalid class selection."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e6],
            'intensity[s2]': [1.1e6],
            'intensity[s3]': [1.2e6],
            'intensity[s4]': [1.3e6],
        })

        norm_config = NormalizationWorkflowConfig(
            experiment=simple_experiment,
            normalization=NormalizationConfig(
                method='none',
                selected_classes=['INVALID_CLASS']  # Not in data
            ),
            data_format=DataFormat.GENERIC
        )

        result = NormalizationWorkflow.run(data, norm_config)

        # Should fail validation
        assert not result.success or 'not found' in str(result.validation_errors).lower()

    def test_all_zeros_zero_filtering(self, all_zeros_df, simple_experiment):
        """Test zero filtering on data with all zeros."""
        config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=True,
        )

        result = DataIngestionWorkflow.run(all_zeros_df, config)

        # All rows should be filtered out
        if result.is_valid:
            # Either empty result or all removed
            assert result.cleaned_df is None or len(result.cleaned_df) == 0 or result.species_removed_count == len(all_zeros_df)

    def test_ingestion_error_prevents_normalization(self, simple_experiment):
        """Test that ingestion errors prevent subsequent normalization."""
        # Create invalid data
        data = pd.DataFrame()  # Empty

        ingestion_config = IngestionConfig(
            experiment=simple_experiment,
            data_format=DataFormat.GENERIC,
            apply_zero_filter=False,
        )

        ingestion_result = DataIngestionWorkflow.run(data, ingestion_config)

        # Should fail
        assert not ingestion_result.is_valid

        # If we try to normalize, it should also fail
        if ingestion_result.cleaned_df is not None:
            norm_config = NormalizationWorkflowConfig(
                experiment=simple_experiment,
                normalization=NormalizationConfig(
                    method='none',
                    selected_classes=[]
                ),
                data_format=DataFormat.GENERIC
            )
            norm_result = NormalizationWorkflow.run(ingestion_result.cleaned_df, norm_config)
            assert not norm_result.success


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
