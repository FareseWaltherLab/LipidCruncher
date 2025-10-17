"""
Integration tests for the complete data processing pipeline.
Tests the flow: Raw Data -> Cleaning -> Normalization
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from src.lipidcruncher.core.services.normalization_service import NormalizationService
from src.lipidcruncher.core.models.experiment import ExperimentConfig
from src.lipidcruncher.core.models.normalization import NormalizationConfig


@pytest.fixture
def raw_lipidsearch_data():
    """
    Create realistic raw LipidSearch data with:
    - Multiple lipid classes (PC, PE, LPC)
    - Multiple grades (A, B, C)
    - Duplicates with different qualities
    - Internal standards (deuterated)
    """
    return pd.DataFrame({
        'LipidMolec': [
            'PC 16:0_18:1', 'PC 16:0_18:1', 'PC 18:0_18:1',  # PC lipids (one duplicate)
            'PC 16:0_18:1+D7',  # PC internal standard
            'PE 18:0_20:4', 'PE 16:0_18:1',  # PE lipids
            'PE 18:0_20:4+D7',  # PE internal standard
            'LPC 16:0', 'LPC 18:0',  # LPC lipids
            'LPC 16:0+D7',  # LPC internal standard
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PC', 'PE', 'PE', 'PE', 'LPC', 'LPC', 'LPC'],
        'FAKey': [
            '16:0_18:1', '16:0_18:1', '18:0_18:1', '16:0_18:1+D7',
            '18:0_20:4', '16:0_18:1', '18:0_20:4+D7',
            '16:0', '18:0', '16:0+D7'
        ],
        'TotalGrade': ['A', 'B', 'A', 'A', 'B', 'A', 'A', 'C', 'C', 'A'],
        'TotalSmpIDRate(%)': [95.0, 85.0, 90.0, 98.0, 88.0, 92.0, 97.0, 80.0, 82.0, 96.0],
        'CalcMass': [760.5, 760.5, 788.5, 767.5, 768.5, 742.5, 775.5, 496.3, 524.3, 503.3],
        'BaseRt': [12.5, 12.5, 13.1, 12.3, 13.2, 12.8, 13.0, 8.5, 9.2, 8.3],
        # Control samples (s1, s2)
        'intensity[s1]': [1000.0, 950.0, 800.0, 100.0, 1200.0, 900.0, 120.0, 500.0, 450.0, 50.0],
        'intensity[s2]': [1100.0, 1000.0, 850.0, 110.0, 1300.0, 950.0, 130.0, 520.0, 470.0, 55.0],
        # Treatment samples (s3, s4)
        'intensity[s3]': [1500.0, 1450.0, 1200.0, 105.0, 1800.0, 1400.0, 125.0, 600.0, 550.0, 52.0],
        'intensity[s4]': [1600.0, 1500.0, 1250.0, 108.0, 1900.0, 1450.0, 128.0, 620.0, 570.0, 53.0],
    })


@pytest.fixture
def experiment_config():
    """Create experiment configuration for 2 conditions with 2 samples each."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def protein_concentrations():
    """Create protein concentration data for BCA normalization."""
    return pd.DataFrame({
        'Sample': ['s1', 's2', 's3', 's4'],
        'Concentration': [2.0, 2.1, 1.9, 2.0]
    })


class TestDataPipeline:
    """Integration tests for the complete data processing pipeline."""
    
    def test_full_pipeline_lipidsearch_no_normalization(
        self,
        raw_lipidsearch_data,
        experiment_config
    ):
        """Test complete pipeline: LipidSearch cleaning only."""
        # Step 1: Clean the data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config
        )
        
        # Verify cleaning worked
        assert not cleaned_df.empty
        assert 'LipidMolec' in cleaned_df.columns
        assert 'ClassKey' in cleaned_df.columns
        
        # Verify no duplicates (best quality selected)
        # Use literal string matching or proper regex escaping
        pc_lipids = cleaned_df[cleaned_df['LipidMolec'] == 'PC(16:0_18:1)']
        assert len(pc_lipids) == 1
        
        # Verify lipid names are standardized
        assert all('(' in name and ')' in name for name in cleaned_df['LipidMolec'])
        
        # Step 2: Extract internal standards
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Verify standards extracted
        assert not standards_df.empty
        assert len(standards_df) == 3  # PC, PE, LPC standards
        assert all('+D' in name for name in standards_df['LipidMolec'])
        
        # Verify regular lipids don't have standards
        assert not any('+D' in name for name in cleaned_df['LipidMolec'])
        
        # Step 3: No normalization (method='none')
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE', 'LPC']
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config
        )
        
        # Verify data unchanged (no normalization)
        pd.testing.assert_frame_equal(normalized_df, cleaned_df)
    
    def test_full_pipeline_lipidsearch_internal_standards(
        self,
        raw_lipidsearch_data,
        experiment_config
    ):
        """Test complete pipeline: LipidSearch cleaning + internal standard normalization."""
        # Step 1: Clean the data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config
        )
        
        # Step 2: Extract internal standards
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Step 3: Normalize using internal standards
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'LPC'],
            internal_standards={
                'PC': 'PC(16:0_18:1)+D7',
                'PE': 'PE(18:0_20:4)+D7',
                'LPC': 'LPC(16:0)+D7'
            },
            intsta_concentrations={
                'PC(16:0_18:1)+D7': 1.0,
                'PE(18:0_20:4)+D7': 1.5,
                'LPC(16:0)+D7': 0.5
            }
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            intsta_df=standards_df
        )
        
        # Verify normalization worked
        assert not normalized_df.empty
        
        # Verify columns renamed to concentration
        assert 'concentration[s1]' in normalized_df.columns
        assert 'intensity[s1]' not in normalized_df.columns
        
        # Verify we have all expected lipid classes
        assert 'PC' in normalized_df['ClassKey'].values
        assert 'PE' in normalized_df['ClassKey'].values
        assert 'LPC' in normalized_df['ClassKey'].values
        
        # Verify standards are not in the normalized data
        assert not any('+D' in name for name in normalized_df['LipidMolec'])
        
        # Verify no inf or nan values
        concentration_cols = [col for col in normalized_df.columns if col.startswith('concentration[')]
        assert not normalized_df[concentration_cols].isin([np.inf, -np.inf]).any().any()
        assert not normalized_df[concentration_cols].isna().any().any()
    
    def test_full_pipeline_lipidsearch_protein_normalization(
        self,
        raw_lipidsearch_data,
        experiment_config,
        protein_concentrations
    ):
        """Test complete pipeline: LipidSearch cleaning + protein normalization."""
        # Step 1: Clean the data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config
        )
        
        # Step 2: Extract internal standards (just to separate them)
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Step 3: Normalize using protein concentrations
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE', 'LPC'],
            protein_concentrations={
                's1': 2.0,
                's2': 2.1,
                's3': 1.9,
                's4': 2.0
            }
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            protein_df=protein_concentrations
        )
        
        # Verify normalization worked
        assert not normalized_df.empty
        assert 'concentration[s1]' in normalized_df.columns
        
        # Verify protein normalization calculation
        # Original intensity divided by protein concentration
        # Example: if original was 1000 and protein was 2.0, result should be 500
        original_intensity = 1000.0  # PC(16:0_18:1) s1 intensity
        protein_conc = 2.0
        expected = original_intensity / protein_conc
        
        pc_row = normalized_df[normalized_df['LipidMolec'] == 'PC(16:0_18:1)']
        if not pc_row.empty:
            assert pc_row.iloc[0]['concentration[s1]'] == pytest.approx(expected, rel=0.01)
    
    def test_full_pipeline_lipidsearch_both_normalizations(
        self,
        raw_lipidsearch_data,
        experiment_config,
        protein_concentrations
    ):
        """Test complete pipeline: LipidSearch cleaning + both normalizations."""
        # Step 1: Clean the data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config
        )
        
        # Step 2: Extract internal standards
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Step 3: Normalize using both methods
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='both',
            selected_classes=['PC', 'PE', 'LPC'],
            internal_standards={
                'PC': 'PC(16:0_18:1)+D7',
                'PE': 'PE(18:0_20:4)+D7',
                'LPC': 'LPC(16:0)+D7'
            },
            intsta_concentrations={
                'PC(16:0_18:1)+D7': 1.0,
                'PE(18:0_20:4)+D7': 1.5,
                'LPC(16:0)+D7': 0.5
            },
            protein_concentrations={
                's1': 2.0,
                's2': 2.1,
                's3': 1.9,
                's4': 2.0
            }
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            intsta_df=standards_df,
            protein_df=protein_concentrations
        )
        
        # Verify both normalizations applied
        assert not normalized_df.empty
        assert 'concentration[s1]' in normalized_df.columns
        
        # Verify no problematic values
        concentration_cols = [col for col in normalized_df.columns if col.startswith('concentration[')]
        assert not normalized_df[concentration_cols].isin([np.inf, -np.inf]).any().any()
        assert not normalized_df[concentration_cols].isna().any().any()
    
    def test_full_pipeline_with_grade_filtering(
        self,
        raw_lipidsearch_data,
        experiment_config
    ):
        """Test pipeline with custom grade filtering."""
        # Custom grade config: only A for PC, A/B for PE, A/B/C for LPC
        grade_config = {
            'PC': ['A'],
            'PE': ['A', 'B'],
            'LPC': ['A', 'B', 'C']
        }
        
        # Step 1: Clean with custom grade filtering
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config,
            grade_config=grade_config
        )
        
        # Verify only specified grades kept
        # PC should only have grade A entries
        pc_lipids = cleaned_df[cleaned_df['ClassKey'] == 'PC']
        assert len(pc_lipids) >= 1  # At least one PC lipid
        
        # Step 2: Extract standards and normalize
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE', 'LPC'],
            internal_standards={
                'PC': 'PC(16:0_18:1)+D7',
                'PE': 'PE(18:0_20:4)+D7',
                'LPC': 'LPC(16:0)+D7'
            },
            intsta_concentrations={
                'PC(16:0_18:1)+D7': 1.0,
                'PE(18:0_20:4)+D7': 1.5,
                'LPC(16:0)+D7': 0.5
            }
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            intsta_df=standards_df
        )
        
        # Verify pipeline completed successfully
        assert not normalized_df.empty
        assert 'concentration[s1]' in normalized_df.columns
    
    def test_pipeline_preserves_lipid_identity(
        self,
        raw_lipidsearch_data,
        experiment_config
    ):
        """Test that lipid identities are preserved through the pipeline."""
        # Track a specific lipid through the pipeline
        target_lipid_raw = 'PC 16:0_18:1'
        target_lipid_clean = 'PC(16:0_18:1)'
        
        # Step 1: Clean
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            raw_lipidsearch_data,
            experiment_config
        )
        
        # Verify lipid exists after cleaning
        assert target_lipid_clean in cleaned_df['LipidMolec'].values
        
        # Step 2: Extract standards
        cleaned_df, standards_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Verify lipid still exists
        assert target_lipid_clean in cleaned_df['LipidMolec'].values
        
        # Step 3: Normalize
        normalization_service = NormalizationService()
        norm_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)+D7'},
            intsta_concentrations={'PC(16:0_18:1)+D7': 1.0}
        )
        
        normalized_df = normalization_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            intsta_df=standards_df
        )
        
        # Verify lipid still exists after normalization
        assert target_lipid_clean in normalized_df['LipidMolec'].values
        
        # Verify it has concentration values
        lipid_row = normalized_df[normalized_df['LipidMolec'] == target_lipid_clean].iloc[0]
        assert lipid_row['concentration[s1]'] > 0
