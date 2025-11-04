"""
Integration tests for data format handling across all three formats.

Tests:
1. Loading and validation of all three formats (LipidSearch, Generic, Metabolomics Workbench)
2. Format standardization and column mapping
3. Cross-format consistency
4. Edge cases in format detection

Based on INTEGRATION_TESTING_STRATEGY.md - Phase 1
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import services
from src.lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from src.lipidcruncher.core.services.zero_filtering_service import ZeroFilteringService
from src.lipidcruncher.core.services.normalization_service import NormalizationService
from src.lipidcruncher.core.services.standards_service import StandardsService
from src.lipidcruncher.core.models.experiment import ExperimentConfig
from src.lipidcruncher.core.models.normalization import NormalizationConfig
from src.lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_data(df: pd.DataFrame, format_type: str = 'lipidsearch') -> pd.DataFrame:
    """
    Preprocess data using FormatPreprocessingService.
    
    Args:
        df: Raw DataFrame to preprocess
        format_type: One of 'lipidsearch', 'generic', or 'metabolomics_workbench'
        
    Returns:
        Preprocessed DataFrame
        
    Raises:
        ValueError: If preprocessing fails
    """
    service = FormatPreprocessingService()
    preprocessed_df, success, message = service.validate_and_preprocess(df, format_type)
    
    if not success:
        raise ValueError(f"Preprocessing failed: {message}")
    
    return preprocessed_df

# ============================================================================
# FIXTURES - REAL SAMPLE DATA
# ============================================================================

@pytest.fixture
def lipidsearch_sample_path():
    """Path to real LipidSearch sample data."""
    return Path(__file__).parent / "fixtures" / "lipidsearch_sample.csv"

@pytest.fixture
def generic_sample_path():
    """Path to real Generic format sample data."""
    return Path(__file__).parent / "fixtures" / "generic_sample.csv"

@pytest.fixture
def metabolomics_sample_path():
    """Path to real Metabolomics Workbench sample data."""
    return Path(__file__).parent / "fixtures" / "metabolomic_sample.csv"


@pytest.fixture
def experiment_config_12_samples():
    """Experiment config for 12 samples in 3 conditions."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT-DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4]
    )


# ============================================================================
# TEST CLASS 1: FORMAT LOADING AND VALIDATION
# ============================================================================

class TestFormatLoading:
    """Test loading and validation of all three data formats."""
    
    def test_load_lipidsearch_format(self, lipidsearch_sample_path):
        """
        Test loading LipidSearch format data.
        
        Given: Valid LipidSearch CSV with all required columns
        When: File loaded and columns checked
        Then: 
            - Returns valid DataFrame
            - All required columns present (LipidMolec, ClassKey, FAKey, TotalGrade, MeanArea[])
            - MeanArea columns detected correctly
        """
        # Load the file
        df = pd.read_csv(lipidsearch_sample_path)
        
        # Verify it loaded successfully
        assert not df.empty, "LipidSearch file should not be empty"
        assert len(df) > 0, "Should have data rows"
        
        # Verify required columns present
        required_columns = ['LipidMolec', 'ClassKey', 'FAKey', 'TotalGrade']
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing"
        
        # Verify MeanArea columns present (raw format before preprocessing)
        meanarea_cols = [col for col in df.columns if col.startswith('MeanArea[')]
        assert len(meanarea_cols) == 12, "Should have 12 MeanArea columns"
        
        # Verify column names follow pattern MeanArea[s1], MeanArea[s2], etc.
        for i, col in enumerate(meanarea_cols, 1):
            assert col == f'MeanArea[s{i}]', f"Column should be MeanArea[s{i}] but got {col}"
    
    def test_load_generic_format(self, generic_sample_path):
        """
        Test loading Generic format data.
        
        Given: Valid Generic CSV (LipidMolec + intensity columns)
        When: File loaded
        Then:
            - Returns valid DataFrame
            - First column is LipidMolec
            - Other columns follow intensity[s1], intensity[s2] pattern
            - No grade or FAKey columns
        """
        # Load the file
        df = pd.read_csv(generic_sample_path)
        
        # Verify it loaded successfully
        assert not df.empty, "Generic file should not be empty"
        assert len(df) > 0, "Should have data rows"
        
        # Verify first column is LipidMolec
        assert df.columns[0] == 'LipidMolec', "First column should be LipidMolec"
        
        # Verify intensity columns present
        intensity_cols = [col for col in df.columns if col.startswith('intensity[')]
        assert len(intensity_cols) == 12, "Should have 12 intensity columns"
        
        # Verify no LipidSearch-specific columns
        assert 'TotalGrade' not in df.columns, "Generic format should not have TotalGrade"
        assert 'FAKey' not in df.columns, "Generic format should not have FAKey"
        assert 'ClassKey' not in df.columns, "Generic format should not have ClassKey initially"
    
    def test_load_metabolomics_workbench_format(self, metabolomics_sample_path):
        """
        Test loading Metabolomics Workbench format data.
        
        Given: Valid MW format with MS_METABOLITE_DATA markers
        When: File loaded and parsed correctly
        Then:
            - Returns valid DataFrame starting from MS_METABOLITE_DATA_START
            - Sample names in first data row
            - Conditions in second data row
            - Lipid data starts from third row
            - 12 samples present
        """
        # Read the file and find the data section
        with open(metabolomics_sample_path, 'r') as f:
            lines = f.readlines()
        
        # Find MS_METABOLITE_DATA_START
        start_idx = None
        for i, line in enumerate(lines):
            if 'MS_METABOLITE_DATA_START' in line:
                start_idx = i + 1  # Data starts on next line
                break
        
        assert start_idx is not None, "Should find MS_METABOLITE_DATA_START marker"
        
        # Parse from that point
        data_lines = []
        for line in lines[start_idx:]:
            if 'MS_METABOLITE_DATA_END' in line:
                break
            data_lines.append(line)
        
        # First line should be sample names
        assert 'Samples' in data_lines[0], "First line should contain 'Samples'"
        
        # Second line should be conditions/factors
        assert 'Factors' in data_lines[1], "Second line should contain 'Factors'"
        assert 'Condition:WT' in data_lines[1], "Should have WT condition"
        assert 'Condition:ADGAT-DKO' in data_lines[1], "Should have ADGAT-DKO condition"
        assert 'Condition:BQC' in data_lines[1], "Should have BQC condition"
        
        # Third line onwards should be lipid data
        assert len(data_lines) > 2, "Should have lipid data rows"
        assert 'AcCa' in data_lines[2], "First lipid should be AcCa"


# ============================================================================
# TEST CLASS 2: FORMAT STANDARDIZATION
# ============================================================================

class TestFormatStandardization:
    """Test that all formats are standardized to common structure."""
    
    def test_lipidsearch_standardization(
        self, 
        lipidsearch_sample_path,
        experiment_config_12_samples
    ):
        """
        Test LipidSearch data standardization.
        
        Given: Raw LipidSearch data with MeanArea[] columns
        When: DataCleaningService processes it
        Then:
            - MeanArea[] columns renamed to intensity[]
            - Lipid names standardized (space → parentheses format)
            - Duplicates removed (best quality kept)
            - ClassKey extracted/preserved
        """
        # Load data
        df = pd.read_csv(lipidsearch_sample_path)
        df = preprocess_data(df, 'lipidsearch')
        
        # Process with DataCleaningService
        service = DataCleaningService()
        cleaned_df = service.clean_lipidsearch_data(
            df,
            experiment_config_12_samples
        )
        
        # Verify standardization
        assert not cleaned_df.empty, "Should have cleaned data"
        
        # Check intensity columns created
        intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
        assert len(intensity_cols) == 12, "Should have 12 intensity columns"
        
        # Check no MeanArea columns remain
        meanarea_cols = [col for col in cleaned_df.columns if col.startswith('MeanArea[')]
        assert len(meanarea_cols) == 0, "Should not have MeanArea columns after cleaning"
        
        # Check lipid name standardization (should have parentheses format)
        sample_lipids = cleaned_df['LipidMolec'].head(5).tolist()
        for lipid in sample_lipids:
            assert '(' in lipid and ')' in lipid, f"Lipid {lipid} should be in parentheses format"
        
        # Check duplicates removed
        assert len(cleaned_df) == cleaned_df['LipidMolec'].nunique(), \
            "Should not have duplicate lipid names"
    
    def test_generic_format_standardization(
        self,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test Generic format data standardization.
        
        Given: Raw Generic data with intensity[] columns
        When: DataCleaningService processes it
        Then:
            - ClassKey extracted from lipid names
            - Lipid names in parentheses format
            - Intensity columns preserved
            - No grade filtering applied
        """
        # Load data
        df = pd.read_csv(generic_sample_path)
        
        # Preprocess: Add ClassKey and standardize format
        df = preprocess_data(df, 'generic')
        
        # Process with DataCleaningService
        service = DataCleaningService()
        cleaned_df = service.clean_generic_data(
            df,
            experiment_config_12_samples
        )
        
        # Verify standardization
        assert not cleaned_df.empty, "Should have cleaned data"
        
        # Check ClassKey was extracted
        assert 'ClassKey' in cleaned_df.columns, "Should have ClassKey column"
        assert cleaned_df['ClassKey'].notna().all(), "All rows should have ClassKey"
        
        # Check intensity columns preserved
        intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
        assert len(intensity_cols) == 12, "Should have 12 intensity columns"
        
        # Check lipid name format
        sample_lipids = cleaned_df['LipidMolec'].head(5).tolist()
        for lipid in sample_lipids:
            assert '(' in lipid and ')' in lipid, \
                f"Lipid {lipid} should be in parentheses format"
    
    def test_column_mapping_consistency(
        self,
        lipidsearch_sample_path,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that both formats map to same column structure.
        
        Given: LipidSearch with MeanArea[] and Generic with intensity[]
        When: Both processed by DataCleaningService
        Then:
            - Both result in intensity[s1], intensity[s2], ... intensity[s12]
            - Column order is consistent
            - Data types match
        """
        # Load and process LipidSearch
        ls_df = pd.read_csv(lipidsearch_sample_path)
        ls_df = preprocess_data(ls_df, 'lipidsearch')
        service = DataCleaningService()
        ls_cleaned = service.clean_lipidsearch_data(
            ls_df,
            experiment_config_12_samples
        )
        
        # Load and process Generic
        gen_df = pd.read_csv(generic_sample_path)
        gen_cleaned = service.clean_generic_data(
            gen_df,
            experiment_config_12_samples
        )
        
        # Get intensity columns from both
        ls_intensity_cols = [col for col in ls_cleaned.columns if col.startswith('intensity[')]
        gen_intensity_cols = [col for col in gen_cleaned.columns if col.startswith('intensity[')]
        
        # Verify same columns
        assert ls_intensity_cols == gen_intensity_cols, \
            "LipidSearch and Generic should have same intensity column names"
        
        # Verify same number of columns
        assert len(ls_intensity_cols) == 12, "Should have 12 intensity columns"
        
        # Verify data types match
        for col in ls_intensity_cols:
            assert ls_cleaned[col].dtype == gen_cleaned[col].dtype, \
                f"Data type mismatch for column {col}"


# ============================================================================
# TEST CLASS 3: CROSS-FORMAT CONSISTENCY
# ============================================================================

class TestCrossFormatConsistency:
    """Test that same lipids are consistently processed across formats."""
    
    def test_same_lipids_detected_across_formats(
        self,
        lipidsearch_sample_path,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that same lipids are detected in all formats.
        
        Given: Same lipid dataset in LipidSearch and Generic formats
        When: Both processed
        Then:
            - Same lipid species detected
            - Same ClassKey extraction
            - Lipid names match after standardization
        """
        service = DataCleaningService()
        
        # Process LipidSearch
        ls_df = pd.read_csv(lipidsearch_sample_path)
        ls_df = preprocess_data(ls_df, 'lipidsearch')
        ls_cleaned = service.clean_lipidsearch_data(
            ls_df,
            experiment_config_12_samples
        )
        
        # Process Generic
        gen_df = pd.read_csv(generic_sample_path)
        gen_df = preprocess_data(gen_df, 'generic')
        gen_cleaned = service.clean_generic_data(
            gen_df,
            experiment_config_12_samples
        )
        
        # Get lipid sets
        ls_lipids = set(ls_cleaned['LipidMolec'])
        gen_lipids = set(gen_cleaned['LipidMolec'])
        
        # Should have significant overlap (may not be exact due to grade filtering)
        overlap = ls_lipids & gen_lipids
        assert len(overlap) > 0.8 * len(gen_lipids), \
            f"Should have >80% overlap. Overlap: {len(overlap)}, Generic: {len(gen_lipids)}"
        
        # Get ClassKey sets
        ls_classes = set(ls_cleaned['ClassKey'])
        gen_classes = set(gen_cleaned['ClassKey'])
        
        # Classes should match exactly
        assert ls_classes == gen_classes, \
            f"ClassKey sets should match. LS: {ls_classes}, Gen: {gen_classes}"
    
    def test_intensity_values_consistent_across_formats(
        self,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that intensity values are preserved correctly.
        
        Given: Generic format with known intensity values
        When: Processed by DataCleaningService
        Then:
            - Intensity values preserved exactly
            - No unexpected transformations
            - Sample order maintained
        """
        # Load data
        df = pd.read_csv(generic_sample_path)
        
        # Store original first lipid's first sample value
        original_value = df.iloc[0, 1]  # First lipid, first sample
        
        # Process
        service = DataCleaningService()
        cleaned_df = service.clean_generic_data(
            df,
            experiment_config_12_samples
        )
        
        # Get first lipid name
        first_lipid = cleaned_df.iloc[0]['LipidMolec']
        cleaned_value = cleaned_df.iloc[0]['intensity[s1]']
        
        # Should match original value
        assert cleaned_value == pytest.approx(original_value, rel=1e-6), \
            f"Intensity value should be preserved. Original: {original_value}, Cleaned: {cleaned_value}"


# ============================================================================
# TEST CLASS 4: EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestFormatEdgeCases:
    """Test edge cases in format detection and handling."""
    
    def test_empty_dataset_detected(self):
        """
        Test that empty datasets are properly detected.
        
        Given: DataFrame with only headers, no data rows
        When: Validation attempted
        Then:
            - Clear error message: "Dataset is empty"
            - No cryptic errors
        """
        # Create empty DataFrame with headers
        df = pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'intensity[s1]'])
        
        service = DataCleaningService()
        experiment_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="Dataset is empty"):
            service.clean_generic_data(df, experiment_config)
    
    def test_missing_required_columns_lipidsearch(self):
        """
        Test error handling for missing required columns in LipidSearch format.
        
        Given: LipidSearch data missing required columns (e.g., FAKey)
        When: Cleaning attempted
        Then:
            - Clear error listing missing columns
            - Specific column names mentioned
        """
        # Create DataFrame missing FAKey
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'TotalGrade': ['A'],
            'MeanArea[s1]': [1000.0]
        })
        
        service = DataCleaningService()
        experiment_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        # Should raise KeyError when trying to access missing FAKey column
        # (cleaning service expects preprocessing to have been done first)
        with pytest.raises(KeyError, match="FAKey"):
            service.clean_lipidsearch_data(df, experiment_config)
    
    def test_single_lipid_dataset(self):
        """
        Test handling of dataset with only one lipid.
        
        Given: Dataset with single lipid
        When: Processed
        Then:
            - Successfully processed
            - No errors
            - Single row in output
        """
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0]
        })
        
        # Preprocess: Add ClassKey
        df = preprocess_data(df, 'generic')
        
        service = DataCleaningService()
        experiment_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[2]
        )
        
        cleaned_df = service.clean_generic_data(df, experiment_config)
        
        assert len(cleaned_df) == 1, "Should have one lipid"
        assert 'ClassKey' in cleaned_df.columns, "Should have ClassKey"
        assert cleaned_df.iloc[0]['ClassKey'] == 'PC', "Should extract PC class"
    
    def test_nan_values_in_intensities(self):
        """
        Test handling of NaN values in intensity columns.
        
        Given: Dataset with some NaN intensity values
        When: Processed
        Then:
            - NaN values converted to 0.0 (cleaning service behavior)
            - No errors raised
            - Other data integrity maintained
        """
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'intensity[s1]': [1000.0, np.nan],
            'intensity[s2]': [1100.0, 1200.0]
        })
        
        # Preprocess: Add ClassKey
        df = preprocess_data(df, 'generic')
        
        service = DataCleaningService()
        experiment_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[2]
        )
        
        cleaned_df = service.clean_generic_data(df, experiment_config)
        
        assert len(cleaned_df) == 2, "Should have both lipids"
        # NaN values are intentionally converted to 0.0 by cleaning service (line 421)
        assert cleaned_df.iloc[1]['intensity[s1]'] == 0.0, "NaN should be converted to 0.0"
        assert cleaned_df.iloc[1]['intensity[s2]'] == 1200.0, "Other values should be intact"


# ============================================================================
# TEST CLASS 5: DATA INTEGRITY
# ============================================================================

class TestDataIntegrity:
    """Test that data integrity is maintained during format conversion."""
    
    def test_no_data_loss_during_standardization(
        self,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that no data is lost during standardization.
        
        Given: Raw dataset with N lipids
        When: Cleaned and standardized
        Then:
            - All lipids present in output (or documented why removed)
            - All samples present
            - No silent filtering
        """
        # Load original data
        df_original = pd.read_csv(generic_sample_path)
        original_lipid_count = len(df_original)
        
        # Process
        service = DataCleaningService()
        cleaned_df = service.clean_generic_data(
            df_original,
            experiment_config_12_samples
        )
        
        # Check counts
        cleaned_lipid_count = len(cleaned_df)
        
        # Should have same or fewer lipids (due to valid filtering only)
        assert cleaned_lipid_count <= original_lipid_count, \
            "Cleaned data should not have more lipids than original"
        
        # Should retain most lipids (>95% for Generic format with no quality filtering)
        retention_rate = cleaned_lipid_count / original_lipid_count
        assert retention_rate > 0.95, \
            f"Should retain >95% of lipids. Retained: {retention_rate:.2%}"
    
    def test_sample_order_preserved(
        self,
        generic_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that sample column order is preserved.
        
        Given: Dataset with samples s1, s2, ..., s12
        When: Processed
        Then:
            - Sample order maintained: intensity[s1], intensity[s2], ..., intensity[s12]
            - No shuffling or reordering
        """
        # Load and process
        df = pd.read_csv(generic_sample_path)
        service = DataCleaningService()
        cleaned_df = service.clean_generic_data(
            df,
            experiment_config_12_samples
        )
        
        # Get intensity columns
        intensity_cols = [col for col in cleaned_df.columns if col.startswith('intensity[')]
        
        # Check order
        expected_order = [f'intensity[s{i}]' for i in range(1, 13)]
        assert intensity_cols == expected_order, \
            f"Sample order not preserved. Expected: {expected_order}, Got: {intensity_cols}"
    
    def test_lipid_identity_preserved(
        self,
        lipidsearch_sample_path,
        experiment_config_12_samples
    ):
        """
        Test that lipid chemical identity is preserved through standardization.
        
        Given: LipidSearch format with lipid "PC 16:0_18:1"
        When: Standardized to "PC(16:0_18:1)"
        Then:
            - Same fatty acid composition
            - Same lipid class
            - No information loss
        """
        # Load data
        df = pd.read_csv(lipidsearch_sample_path)
        df = preprocess_data(df, 'lipidsearch')

        # Find a specific lipid in original
        target_raw = 'PC 16:0_18:1'
        original_rows = df[df['LipidMolec'] == target_raw]
        
        if len(original_rows) > 0:
            # Process data
            service = DataCleaningService()
            cleaned_df = service.clean_lipidsearch_data(
                df,
                experiment_config_12_samples
            )
            
            # Find standardized version
            target_clean = 'PC(16:0_18:1)'
            cleaned_row = cleaned_df[cleaned_df['LipidMolec'] == target_clean]
            
            assert len(cleaned_row) > 0, f"Should find standardized lipid {target_clean}"
            assert cleaned_row.iloc[0]['ClassKey'] == 'PC', "ClassKey should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])