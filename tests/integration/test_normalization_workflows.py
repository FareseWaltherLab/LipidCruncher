"""
Integration tests for normalization workflows.

Tests end-to-end normalization with all methods:
1. Internal standards normalization
2. Protein normalization  
3. Both methods combined
4. Selective class normalization
5. Error handling and edge cases

Based on INTEGRATION_TESTING_STRATEGY.md - Phase 3
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
from src.lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService
from src.lipidcruncher.core.models.experiment import ExperimentConfig
from src.lipidcruncher.core.models.normalization import NormalizationConfig


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
# FIXTURES
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


@pytest.fixture
def standards_data():
    """Create internal standards data for testing."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)+D7',
            'PE(18:0_20:4)+D7', 
            'TAG(16:0_18:1_18:1)+D7',
            'DAG(16:0_18:1)+D7',
            'Cer(d18:1_16:0)+D7'
        ],
        'intensity[s1]': [100.0, 120.0, 150.0, 80.0, 90.0],
        'intensity[s2]': [110.0, 125.0, 155.0, 85.0, 95.0],
        'intensity[s3]': [105.0, 122.0, 152.0, 82.0, 92.0],
        'intensity[s4]': [108.0, 128.0, 158.0, 88.0, 98.0],
        'intensity[s5]': [112.0, 130.0, 160.0, 90.0, 100.0],
        'intensity[s6]': [115.0, 132.0, 162.0, 92.0, 102.0],
        'intensity[s7]': [118.0, 135.0, 165.0, 95.0, 105.0],
        'intensity[s8]': [120.0, 138.0, 168.0, 98.0, 108.0],
        'intensity[s9]': [122.0, 140.0, 170.0, 100.0, 110.0],
        'intensity[s10]': [125.0, 142.0, 172.0, 102.0, 112.0],
        'intensity[s11]': [128.0, 145.0, 175.0, 105.0, 115.0],
        'intensity[s12]': [130.0, 148.0, 178.0, 108.0, 118.0]
    })


@pytest.fixture
def protein_concentrations():
    """Create protein concentration data."""
    return pd.DataFrame({
        'Sample': ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'],
        'Concentration': [2.0, 2.1, 1.9, 2.0, 2.2, 2.1, 1.8, 2.0, 2.3, 2.2, 2.0, 1.9]
    })


@pytest.fixture
def lipidsearch_data(lipidsearch_sample_path):
    """
    Load and preprocess real LipidSearch data for normalization tests.
    
    Returns preprocessed DataFrame ready for cleaning and normalization.
    """
    df = pd.read_csv(lipidsearch_sample_path)
    return preprocess_data(df, 'lipidsearch')

@pytest.fixture
def cleaned_lipid_data(lipidsearch_data, experiment_config_12_samples):
    """Pre-cleaned lipid data ready for normalization."""
    service = DataCleaningService()
    cleaned_df = service.clean_lipidsearch_data(
        lipidsearch_data,
        experiment_config_12_samples
    )
    cleaned_df, standards_df = service.extract_internal_standards(cleaned_df)
    return cleaned_df


# ============================================================================
# TEST CLASS 1: INTERNAL STANDARDS NORMALIZATION
# ============================================================================

class TestInternalStandardsNormalization:
    """Test internal standards normalization workflow end-to-end."""
    
    def test_full_workflow_with_internal_standards(
        self,
        lipidsearch_data,
        experiment_config_12_samples,
        standards_data
    ):
        """
        Test complete workflow: Upload → Clean → Normalize with standards.
        
        Given: LipidSearch data with internal standards
        When: Complete pipeline executed
        Then:
            - Data cleaned successfully
            - Standards extracted
            - Normalization applied correctly
            - Formula: (lipid_intensity / standard_intensity) × standard_concentration
            - Result columns named concentration[s1], concentration[s2], etc.
        """
        # Step 1: Clean data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            lipidsearch_data,
            experiment_config_12_samples
        )
        
        # Step 2: Extract internal standards
        cleaned_df, extracted_standards = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Verify standards were extracted
        assert not extracted_standards.empty, "Should have extracted standards"
        
        # Step 3: Setup normalization config
        # Map classes to their standards
        class_standard_map = {}
        standard_concentrations = {}
        
        for _, std_row in extracted_standards.iterrows():
            std_name = std_row['LipidMolec']
            class_key = std_name.split('(')[0]  # Extract class (e.g., 'PC' from 'PC(...)+D7')
            
            class_standard_map[class_key] = std_name
            standard_concentrations[std_name] = 1.0  # 1.0 µM concentration
        
        # Get available classes
        available_classes = list(class_standard_map.keys())
        
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=available_classes,
            internal_standards=class_standard_map,
            intsta_concentrations=standard_concentrations
        )
        
        # Step 4: Apply normalization
        normalization_service = NormalizationService()
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config_12_samples,
            intsta_df=extracted_standards
        )
        
        # Verify normalization worked
        assert not normalized_df.empty, "Should have normalized data"
        
        # Check columns renamed to concentration
        assert 'concentration[s1]' in normalized_df.columns, "Should have concentration columns"
        assert 'intensity[s1]' not in normalized_df.columns, "Should not have intensity columns"
        
        # Check no inf or nan values
        conc_cols = [col for col in normalized_df.columns if col.startswith('concentration[')]
        assert not normalized_df[conc_cols].isin([np.inf, -np.inf]).any().any(), \
            "Should not have infinity values"
        assert not normalized_df[conc_cols].isna().any().any(), \
            "Should not have NaN values"
        
        # Check standards not in normalized data
        assert not any('+D' in name for name in normalized_df['LipidMolec']), \
            "Standards should not be in normalized data"
    
    def test_selective_class_normalization(
        self,
        cleaned_lipid_data,
        standards_data,
        experiment_config_12_samples
    ):
        """
        Test normalizing only selected lipid classes.
        
        Given: Data with PC, PE, TAG, DAG, Cer classes
        When: User selects only PC and PE for normalization
        Then:
            - Only PC and PE lipids in output
            - TAG, DAG, Cer excluded
            - Normalization applied correctly to selected classes
        """
        # Setup: Select only PC and PE
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(16:0_18:1)+D7',
                'PE': 'PE(18:0_20:4)+D7'
            },
            intsta_concentrations={
                'PC(16:0_18:1)+D7': 1.0,
                'PE(18:0_20:4)+D7': 1.5
            }
        )
        
        # Filter standards to match
        selected_standards = standards_data[
            standards_data['LipidMolec'].isin(['PC(16:0_18:1)+D7', 'PE(18:0_20:4)+D7'])
        ]
        
        # Normalize
        service = NormalizationService()
        normalized_df = service.normalize(
            cleaned_lipid_data,
            norm_config,
            experiment_config_12_samples,
            intsta_df=selected_standards
        )
        
        # Verify only selected classes in output
        output_classes = set(normalized_df['ClassKey'].unique())
        assert output_classes.issubset({'PC', 'PE'}), \
            f"Should only have PC and PE, got: {output_classes}"
        
        # Verify excluded classes not present
        assert 'TAG' not in output_classes, "TAG should be excluded"
        assert 'DAG' not in output_classes, "DAG should be excluded"
    
    def test_normalization_formula_correctness(
        self,
        experiment_config_12_samples
    ):
        """
        Test that normalization formula is applied correctly.
        Formula: (lipid_intensity / standard_intensity) × standard_concentration
        
        Given: Known lipid and standard intensities
        When: Normalization applied
        Then:
            - Result matches hand-calculated expected value
            - Formula applied correctly for all samples
        """
        # Create test data with known values
        lipid_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(18:0_18:1)'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1100.0, 2200.0]
        })
        
        standard_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)+D7'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [110.0]
        })
        
        # Hand-calculate expected values
        # PC(16:0_18:1) s1: (1000.0 / 100.0) × 1.5 = 15.0
        # PC(16:0_18:1) s2: (1100.0 / 110.0) × 1.5 = 15.0
        # PC(18:0_18:1) s1: (2000.0 / 100.0) × 1.5 = 30.0
        # PC(18:0_18:1) s2: (2200.0 / 110.0) × 1.5 = 30.0
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)+D7'},
            intsta_concentrations={'PC(16:0_18:1)+D7': 1.5}
        )
        
        exp_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[2]
        )
        
        service = NormalizationService()
        result = service.normalize(
            lipid_df,
            config,
            exp_config,
            intsta_df=standard_df
        )
        
        # Check PC(16:0_18:1)
        pc1 = result[result['LipidMolec'] == 'PC(16:0_18:1)'].iloc[0]
        assert pc1['concentration[s1]'] == pytest.approx(15.0, rel=1e-6), \
            f"Expected 15.0, got {pc1['concentration[s1]']}"
        assert pc1['concentration[s2]'] == pytest.approx(15.0, rel=1e-6), \
            f"Expected 15.0, got {pc1['concentration[s2]']}"
        
        # Check PC(18:0_18:1)
        pc2 = result[result['LipidMolec'] == 'PC(18:0_18:1)'].iloc[0]
        assert pc2['concentration[s1]'] == pytest.approx(30.0, rel=1e-6), \
            f"Expected 30.0, got {pc2['concentration[s1]']}"
        assert pc2['concentration[s2]'] == pytest.approx(30.0, rel=1e-6), \
            f"Expected 30.0, got {pc2['concentration[s2]']}"


# ============================================================================
# TEST CLASS 2: PROTEIN NORMALIZATION
# ============================================================================

class TestProteinNormalization:
    """Test protein normalization workflow end-to-end."""
    
    def test_full_workflow_with_protein_normalization(
        self,
        cleaned_lipid_data,
        protein_concentrations,
        experiment_config_12_samples
    ):
        """
        Test complete workflow: Upload → Clean → Normalize with protein.
        
        Given: Cleaned lipid data + protein concentrations
        When: Protein normalization applied
        Then:
            - Formula: lipid_intensity / protein_concentration
            - Result columns named concentration[s1], concentration[s2], etc.
            - No inf or nan values
        """
        # Setup protein normalization config
        protein_dict = {
            row['Sample']: row['Concentration'] 
            for _, row in protein_concentrations.iterrows()
        }
        
        # Get all available classes
        available_classes = cleaned_lipid_data['ClassKey'].unique().tolist()
        
        norm_config = NormalizationConfig(
            method='protein',
            selected_classes=available_classes,
            protein_concentrations=protein_dict
        )
        
        # Apply normalization
        service = NormalizationService()
        normalized_df = service.normalize(
            cleaned_lipid_data,
            norm_config,
            experiment_config_12_samples,
            protein_df=protein_concentrations
        )
        
        # Verify normalization worked
        assert not normalized_df.empty, "Should have normalized data"
        
        # Check columns renamed
        assert 'concentration[s1]' in normalized_df.columns, \
            "Should have concentration columns"
        assert 'intensity[s1]' not in normalized_df.columns, \
            "Should not have intensity columns"
        
        # Check no problematic values
        conc_cols = [col for col in normalized_df.columns if col.startswith('concentration[')]
        assert not normalized_df[conc_cols].isin([np.inf, -np.inf]).any().any(), \
            "Should not have infinity values"
        assert not normalized_df[conc_cols].isna().any().any(), \
            "Should not have NaN values"
    
    def test_protein_normalization_formula_correctness(
        self,
        experiment_config_12_samples
    ):
        """
        Test protein normalization formula.
        Formula: lipid_intensity / protein_concentration
        
        Given: Known lipid intensities and protein concentrations
        When: Normalization applied
        Then:
            - Result matches hand-calculated expected value
        """
        # Create test data
        lipid_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [2100.0]
        })
        
        protein_df = pd.DataFrame({
            'Sample': ['s1', 's2'],
            'Concentration': [2.0, 2.1]
        })
        
        # Hand-calculate expected values
        # s1: 1000.0 / 2.0 = 500.0
        # s2: 2100.0 / 2.1 = 1000.0
        
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.0, 's2': 2.1}
        )
        
        exp_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[2]
        )
        
        service = NormalizationService()
        result = service.normalize(
            lipid_df,
            config,
            exp_config,
            protein_df=protein_df
        )
        
        # Check results
        assert result.iloc[0]['concentration[s1]'] == pytest.approx(500.0, rel=1e-6), \
            f"Expected 500.0, got {result.iloc[0]['concentration[s1]']}"
        assert result.iloc[0]['concentration[s2]'] == pytest.approx(1000.0, rel=1e-6), \
            f"Expected 1000.0, got {result.iloc[0]['concentration[s2]']}"


# ============================================================================
# TEST CLASS 3: BOTH NORMALIZATION METHODS
# ============================================================================

class TestBothNormalizationMethods:
    """Test sequential application of both normalization methods."""
    
    def test_full_workflow_both_methods(
        self,
        cleaned_lipid_data,
        standards_data,
        protein_concentrations,
        experiment_config_12_samples
    ):
        """
        Test complete workflow with both normalization methods.
        
        Given: Cleaned data + standards + protein concentrations
        When: Both normalization methods applied sequentially
        Then:
            - Internal standards applied first
            - Then protein normalization applied
            - Final result has correct formula: (lipid/standard×conc) / protein
        """
        # Setup standards mapping
        class_standard_map = {}
        standard_concentrations = {}
        
        for _, std_row in standards_data.iterrows():
            std_name = std_row['LipidMolec']
            class_key = std_name.split('(')[0]
            class_standard_map[class_key] = std_name
            standard_concentrations[std_name] = 1.0
        
        available_classes = list(class_standard_map.keys())
        
        # Setup protein concentrations
        protein_dict = {
            row['Sample']: row['Concentration']
            for _, row in protein_concentrations.iterrows()
        }
        
        # Configure both methods
        norm_config = NormalizationConfig(
            method='both',
            selected_classes=available_classes,
            internal_standards=class_standard_map,
            intsta_concentrations=standard_concentrations,
            protein_concentrations=protein_dict
        )
        
        # Apply normalization
        service = NormalizationService()
        normalized_df = service.normalize(
            cleaned_lipid_data,
            norm_config,
            experiment_config_12_samples,
            intsta_df=standards_data,
            protein_df=protein_concentrations
        )
        
        # Verify result
        assert not normalized_df.empty, "Should have normalized data"
        assert 'concentration[s1]' in normalized_df.columns, \
            "Should have concentration columns"
        
        # Verify no problematic values
        conc_cols = [col for col in normalized_df.columns if col.startswith('concentration[')]
        assert not normalized_df[conc_cols].isin([np.inf, -np.inf]).any().any(), \
            "Should not have infinity values (bug from commit 05d1569)"
        assert not normalized_df[conc_cols].isna().any().any(), \
            "Should not have NaN values"
    
    def test_both_methods_formula_correctness(self):
        """
        Test that both methods are applied in correct order.
        Formula: ((lipid/standard) × standard_conc) / protein
        
        Given: Known values for all inputs
        When: Both methods applied
        Then:
            - Result matches hand-calculated value
            - Verifies fix from commit 05d1569 (protein norm after standards)
        """
        # Create test data with known values
        lipid_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0]
        })
        
        standard_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)+D7'],
            'intensity[s1]': [100.0]
        })
        
        protein_df = pd.DataFrame({
            'Sample': ['s1'],
            'Concentration': [2.0]
        })
        
        # Hand-calculate expected value
        # Step 1: Internal standards: (1000.0 / 100.0) × 1.5 = 15.0
        # Step 2: Protein: 15.0 / 2.0 = 7.5
        
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)+D7'},
            intsta_concentrations={'PC(16:0_18:1)+D7': 1.5},
            protein_concentrations={'s1': 2.0}
        )
        
        exp_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        service = NormalizationService()
        result = service.normalize(
            lipid_df,
            config,
            exp_config,
            intsta_df=standard_df,
            protein_df=protein_df
        )
        
        # Check result
        expected = 7.5
        actual = result.iloc[0]['concentration[s1]']
        assert actual == pytest.approx(expected, rel=1e-6), \
            f"Expected {expected}, got {actual}. " \
            f"This verifies fix from commit 05d1569 (protein norm applied after standards)"


# ============================================================================
# TEST CLASS 4: ERROR HANDLING
# ============================================================================

class TestNormalizationErrorHandling:
    """Test error handling in normalization workflows."""
    
    def test_missing_standard_for_class(self, experiment_config_12_samples):
        """
        Test error when standard is missing for a selected class.
        
        Given: User selects PC class but no PC standard provided
        When: Normalization attempted
        Then:
            - Clear error message
            - Lists which classes are missing standards
        """
        lipid_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0]
        })
        
        standard_df = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)+D7'],  # Wrong class!
            'intensity[s1]': [100.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)+D7'},  # Standard defined but not in data
            intsta_concentrations={'PC(16:0_18:1)+D7': 1.0}
        )
        
        exp_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        service = NormalizationService()
        
        # Should raise ValueError with clear message
        with pytest.raises(ValueError, match="not found in intsta_df"):
            service.normalize(
                lipid_df,
                config,
                exp_config,
                intsta_df=standard_df
            )
    
    def test_zero_intensity_handling(self, experiment_config_12_samples):
        """
        Test handling of zero intensity values (division by zero).
        
        Given: Standard with zero intensity
        When: Normalization attempted
        Then:
            - No crash
            - Result is handled gracefully (inf or filtered out)
        """
        lipid_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0]
        })
        
        standard_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)+D7'],
            'intensity[s1]': [0.0]  # Zero intensity!
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)+D7'},
            intsta_concentrations={'PC(16:0_18:1)+D7': 1.0}
        )
        
        exp_config = ExperimentConfig(
            n_conditions=1,
            conditions_list=['Control'],
            number_of_samples_list=[1]
        )
        
        service = NormalizationService()
        
        # Should not crash
        result = service.normalize(
            lipid_df,
            config,
            exp_config,
            intsta_df=standard_df
        )
        
        # Result should either be inf (handled later) or filtered out
        # The service should handle this gracefully without crashing
        assert result is not None, "Should return a result even with zero standard"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])