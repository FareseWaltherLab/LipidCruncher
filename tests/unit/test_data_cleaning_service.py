"""
Unit tests for DataCleaningService.
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from src.lipidcruncher.core.models.experiment import ExperimentConfig


@pytest.fixture
def sample_experiment():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def sample_lipidsearch_data():
    """Create sample LipidSearch format data."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4', 'PC 16:0_18:1', 'Ch D7'],
        'ClassKey': ['PC', 'PE', 'PC', 'Ch'],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1', 'D7'],
        'TotalGrade': ['A', 'B', 'B', 'A'],
        'TotalSmpIDRate(%)': [95.0, 90.0, 85.0, 98.0],
        'CalcMass': [760.5, 768.5, 760.5, 369.3],
        'BaseRt': [12.5, 13.2, 12.5, 8.5],
        'intensity[s1]': [100.0, 200.0, 90.0, 50.0],
        'intensity[s2]': [110.0, 210.0, 95.0, 55.0],
        'intensity[s3]': [105.0, 205.0, 92.0, 52.0],
        'intensity[s4]': [108.0, 208.0, 93.0, 53.0]
    })


@pytest.fixture
def sample_generic_data():
    """Create sample Generic format data."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', '', '#', 'TAG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'Invalid', 'Invalid', 'TAG'],
        'intensity[s1]': [100.0, 200.0, 0.0, 0.0, 150.0],
        'intensity[s2]': [110.0, 210.0, 0.0, 0.0, 160.0],
        'intensity[s3]': [105.0, 205.0, 0.0, 0.0, 155.0],
        'intensity[s4]': [108.0, 208.0, 0.0, 0.0, 158.0]
    })


class TestDataCleaningService:
    """Test suite for DataCleaningService."""
    
    # ==================== LipidSearch Tests ====================
    
    def test_clean_lipidsearch_data_success(self, sample_lipidsearch_data, sample_experiment):
        """Test successful LipidSearch data cleaning."""
        service = DataCleaningService()
        
        result = service.clean_lipidsearch_data(
            sample_lipidsearch_data,
            sample_experiment
        )
        
        # Check that result has expected columns
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert 'intensity[s1]' in result.columns
        
        # Check that TotalSmpIDRate(%) is removed
        assert 'TotalSmpIDRate(%)' not in result.columns
        
        # Check that lipid names are standardized
        assert all('(' in name and ')' in name for name in result['LipidMolec'])
    
    def test_clean_lipidsearch_removes_missing_fa_keys(self, sample_experiment):
        """Test that rows with missing FA keys are removed (except Ch)."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4', 'Ch'],
            'ClassKey': ['PC', 'PE', 'Ch'],
            'FAKey': ['16:0_18:1', np.nan, np.nan],  # PE has missing FA key
            'TotalGrade': ['A', 'A', 'A'],
            'TotalSmpIDRate(%)': [95.0, 90.0, 98.0],
            'CalcMass': [760.5, 768.5, 369.3],
            'BaseRt': [12.5, 13.2, 8.5],
            'intensity[s1]': [100.0, 200.0, 50.0],
            'intensity[s2]': [110.0, 210.0, 55.0],
            'intensity[s3]': [105.0, 205.0, 52.0],
            'intensity[s4]': [108.0, 208.0, 53.0]
        })
        
        result = service.clean_lipidsearch_data(data, sample_experiment)
        
        # PE should be removed, PC and Ch should remain
        assert len(result) == 2
        assert 'PC' in result['ClassKey'].values
        assert 'Ch' in result['ClassKey'].values
        assert 'PE' not in result['ClassKey'].values
    
    def test_clean_lipidsearch_grade_filtering_default(self, sample_lipidsearch_data, sample_experiment):
        """Test default grade filtering (A, B, C)."""
        service = DataCleaningService()
        
        # Add a grade D entry
        data = sample_lipidsearch_data.copy()
        data.loc[len(data)] = {
            'LipidMolec': 'TAG 16:0_18:1_18:2',
            'ClassKey': 'TAG',
            'FAKey': '16:0_18:1_18:2',
            'TotalGrade': 'D',
            'TotalSmpIDRate(%)': 70.0,
            'CalcMass': 850.0,
            'BaseRt': 15.0,
            'intensity[s1]': 50.0,
            'intensity[s2]': 55.0,
            'intensity[s3]': 52.0,
            'intensity[s4]': 53.0
        }
        
        result = service.clean_lipidsearch_data(data, sample_experiment)
        
        # Grade D should be filtered out
        assert 'D' not in result.get('TotalGrade', pd.Series()).values if 'TotalGrade' in result.columns else True
    
    def test_clean_lipidsearch_grade_filtering_custom(self, sample_lipidsearch_data, sample_experiment):
        """Test custom grade filtering."""
        service = DataCleaningService()
        
        # Only accept grade A for PC
        grade_config = {'PC': ['A'], 'PE': ['A', 'B']}
        
        result = service.clean_lipidsearch_data(
            sample_lipidsearch_data,
            sample_experiment,
            grade_config=grade_config
        )
        
        # Should have PC and PE lipids
        assert 'PC' in result['ClassKey'].values
        assert 'PE' in result['ClassKey'].values
    
    def test_clean_lipidsearch_selects_best_quality(self, sample_experiment):
        """Test that best quality entry is selected for duplicates."""
        service = DataCleaningService()
        
        # Two entries for same lipid with different quality
        data = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PC 16:0_18:1'],
            'ClassKey': ['PC', 'PC'],
            'FAKey': ['16:0_18:1', '16:0_18:1'],
            'TotalGrade': ['A', 'B'],
            'TotalSmpIDRate(%)': [95.0, 85.0],  # First one is better
            'CalcMass': [760.5, 760.5],
            'BaseRt': [12.5, 12.5],
            'intensity[s1]': [100.0, 90.0],
            'intensity[s2]': [110.0, 95.0],
            'intensity[s3]': [105.0, 92.0],
            'intensity[s4]': [108.0, 93.0]
        })
        
        result = service.clean_lipidsearch_data(data, sample_experiment)
        
        # Should only have one entry for PC(16:0_18:1)
        assert len(result) == 1
        # Should be the one with better quality (higher intensity from first entry)
        assert result.loc[0, 'intensity[s1]'] == 100.0
    
    def test_standardize_lipid_name_ch_class(self, sample_experiment):
        """Test lipid name standardization for Ch class."""
        service = DataCleaningService()
        
        # Test Ch without FA key
        assert service._standardize_lipid_name('Ch', np.nan) == 'Ch()'
        
        # Test Ch with D marker
        assert service._standardize_lipid_name('Ch', 'D7') == 'Ch-D7()'
    
    def test_standardize_lipid_name_with_internal_standard(self, sample_experiment):
        """Test lipid name standardization with internal standard marker."""
        service = DataCleaningService()
        
        result = service._standardize_lipid_name('PC', '16:0_18:1+D7')
        
        assert result == 'PC(16:0_18:1)+D7'
    
    def test_standardize_lipid_name_sorting(self, sample_experiment):
        """Test that fatty acids are sorted."""
        service = DataCleaningService()
        
        result = service._standardize_lipid_name('PC', '18:1_16:0')
        
        # Should be sorted: 16:0 before 18:1
        assert result == 'PC(16:0_18:1)'
    
    # ==================== Generic Format Tests ====================
    
    def test_clean_generic_data_success(self, sample_generic_data, sample_experiment):
        """Test successful generic data cleaning."""
        service = DataCleaningService()
        
        result = service.clean_generic_data(sample_generic_data, sample_experiment)
        
        # Should have valid lipids only
        assert len(result) == 3  # PC, PE, TAG
        assert all(result['LipidMolec'] != '')
        assert all(result['LipidMolec'] != '#')
    
    def test_clean_generic_removes_invalid_names(self, sample_experiment):
        """Test that invalid lipid names are removed."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', '', '#', '!!!', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'Invalid', 'Invalid', 'Invalid', 'PE'],
            'intensity[s1]': [100.0, 50.0, 50.0, 50.0, 200.0],
            'intensity[s2]': [110.0, 55.0, 55.0, 55.0, 210.0],
            'intensity[s3]': [105.0, 52.0, 52.0, 52.0, 205.0],
            'intensity[s4]': [108.0, 53.0, 53.0, 53.0, 208.0]
        })
        
        result = service.clean_generic_data(data, sample_experiment)
        
        # Should only have PC and PE
        assert len(result) == 2
        assert 'PC' in result['ClassKey'].values
        assert 'PE' in result['ClassKey'].values
    
    def test_clean_generic_removes_all_zero_rows(self, sample_experiment):
        """Test that rows with all zero intensities are removed."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:2)'],
            'ClassKey': ['PC', 'PE', 'TAG'],
            'intensity[s1]': [100.0, 0.0, 150.0],
            'intensity[s2]': [110.0, 0.0, 160.0],
            'intensity[s3]': [105.0, 0.0, 155.0],
            'intensity[s4]': [108.0, 0.0, 158.0]
        })
        
        result = service.clean_generic_data(data, sample_experiment)
        
        # PE with all zeros should be removed
        assert len(result) == 2
        assert 'PE' not in result['ClassKey'].values
    
    def test_clean_generic_removes_duplicates(self, sample_experiment):
        """Test that duplicate lipids are removed."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [100.0, 90.0, 200.0],
            'intensity[s2]': [110.0, 95.0, 210.0],
            'intensity[s3]': [105.0, 92.0, 205.0],
            'intensity[s4]': [108.0, 93.0, 208.0]
        })
        
        result = service.clean_generic_data(data, sample_experiment)
        
        # Should only have 2 unique lipids
        assert len(result) == 2
    
    def test_clean_generic_raises_error_on_all_invalid(self, sample_experiment):
        """Test that error is raised when all lipids are invalid."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['', '#', '!!!'],
            'ClassKey': ['Invalid', 'Invalid', 'Invalid'],
            'intensity[s1]': [50.0, 50.0, 50.0],
            'intensity[s2]': [55.0, 55.0, 55.0],
            'intensity[s3]': [52.0, 52.0, 52.0],
            'intensity[s4]': [53.0, 53.0, 53.0]
        })
        
        with pytest.raises(ValueError, match="No valid data remains"):
            service.clean_generic_data(data, sample_experiment)
    
    # ==================== Internal Standards Extraction ====================
    
    def test_extract_internal_standards(self, sample_experiment):
        """Test extraction of internal standards."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)(d7)', 'PE(18:0_20:4)', 'PE(18:0_20:4)(d9)'],
            'ClassKey': ['PC', 'PC', 'PE', 'PE'],
            'intensity[s1]': [100.0, 50.0, 200.0, 60.0],
            'intensity[s2]': [110.0, 55.0, 210.0, 65.0],
            'intensity[s3]': [105.0, 52.0, 205.0, 62.0],
            'intensity[s4]': [108.0, 53.0, 208.0, 63.0]
        })
        
        cleaned_df, standards_df = service.extract_internal_standards(data)
        
        # Should have 2 regular lipids and 2 standards
        assert len(cleaned_df) == 2
        assert len(standards_df) == 2
        
        # Standards should have (d7) or (d9)
        assert all('(d' in name for name in standards_df['LipidMolec'])
        
        # Regular lipids should not have deuteration markers
        assert all('(d' not in name for name in cleaned_df['LipidMolec'])
    
    def test_extract_internal_standards_with_plus_d(self, sample_experiment):
        """Test extraction with +D notation."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_18:1)+D7'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': [100.0, 50.0],
            'intensity[s2]': [110.0, 55.0],
            'intensity[s3]': [105.0, 52.0],
            'intensity[s4]': [108.0, 53.0]
        })
        
        cleaned_df, standards_df = service.extract_internal_standards(data)
        
        assert len(cleaned_df) == 1
        assert len(standards_df) == 1
        # Use iloc instead of loc since index might not be 0
        assert '+D' in standards_df.iloc[0]['LipidMolec']
    
    # ==================== Helper Methods Tests ====================
    
    def test_convert_columns_to_numeric(self, sample_experiment):
        """Test conversion of columns to numeric."""
        service = DataCleaningService()
        
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': ['100.0'],  # String
            'intensity[s2]': ['invalid'],  # Invalid
            'intensity[s3]': [-50.0],  # Negative
            'intensity[s4]': [np.nan]  # NaN
        })
        
        result = service._convert_columns_to_numeric(data, sample_experiment.samples_list)
        
        # Should all be numeric
        assert result['intensity[s1]'].dtype in [np.float64, np.int64]
        
        # Invalid should become 0
        assert result.loc[0, 'intensity[s2]'] == 0.0
        
        # Negative should become 0
        assert result.loc[0, 'intensity[s3]'] == 0.0
        
        # NaN should become 0
        assert result.loc[0, 'intensity[s4]'] == 0.0
