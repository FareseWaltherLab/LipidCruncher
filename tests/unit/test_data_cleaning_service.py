"""
Unit tests for DataCleaningService.

This file contains:
- Basic functionality tests (original tests)
- Edge case tests (added after production bugs in commits 770cff0-5cc334f)
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from src.lipidcruncher.core.models.experiment import ExperimentConfig


# ==================== Fixtures ====================

@pytest.fixture
def sample_experiment():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def sample_experiment_six_samples():
    """Create experiment configuration with 6 samples (for edge case tests)."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3]
    )


@pytest.fixture
def service():
    """DataCleaningService instance."""
    return DataCleaningService()


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


# ==================== Basic Functionality Tests ====================

class TestDataCleaningService:
    """Test suite for basic DataCleaningService functionality."""
    
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
        
        with pytest.raises(ValueError, match="No valid lipid species found"):
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
        
        result = service._convert_columns_to_numeric(data, sample_experiment.full_samples_list)
        
        # Should all be numeric
        assert result['intensity[s1]'].dtype in [np.float64, np.int64]
        
        # Invalid should become 0
        assert result.loc[0, 'intensity[s2]'] == 0.0
        
        # Negative should become 0
        assert result.loc[0, 'intensity[s3]'] == 0.0
        
        # NaN should become 0
        assert result.loc[0, 'intensity[s4]'] == 0.0


# ==================== Edge Case Tests ====================

class TestDataCleaningEdgeCases:
    """
    Edge cases and error handling for DataCleaningService.
    
    These tests were added after production bugs revealed gaps in coverage
    (commits 770cff0 through 5cc334f). They focus on:
    - Empty datasets and NaN handling
    - Invalid data types
    - Complete data removal scenarios
    - Error message quality
    """
    
    # ==================== Empty Dataset Handling ====================
    
    def test_empty_lipidsearch_dataframe(self, service, sample_experiment):
        """Test that empty LipidSearch DataFrame raises clear error."""
        empty_df = pd.DataFrame(columns=['LipidMolec', 'ClassKey', 'FAKey', 'TotalGrade'])
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(empty_df, sample_experiment)
        
        assert "Dataset is empty" in str(exc_info.value)
        assert "valid LipidSearch data file" in str(exc_info.value)
    
    def test_empty_generic_dataframe(self, service, sample_experiment):
        """Test that empty Generic DataFrame raises clear error."""
        empty_df = pd.DataFrame(columns=['LipidMolec'])
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_generic_data(empty_df, sample_experiment)
        
        assert "Dataset is empty" in str(exc_info.value)
        assert "valid data file" in str(exc_info.value)
    
    def test_dataframe_with_only_nan_rows(self, service, sample_experiment_six_samples):
        """Test DataFrame with rows containing only NaN values."""
        # This is a common edge case from Excel exports
        df = pd.DataFrame({
            'LipidMolec': [np.nan, np.nan],
            'ClassKey': [np.nan, np.nan],
            'FAKey': [np.nan, np.nan],
            'TotalGrade': [np.nan, np.nan],
            'intensity[s1]': [np.nan, np.nan],
            'intensity[s2]': [np.nan, np.nan],
            'intensity[s3]': [np.nan, np.nan],
            'intensity[s4]': [np.nan, np.nan],
            'intensity[s5]': [np.nan, np.nan],
            'intensity[s6]': [np.nan, np.nan]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        assert "Dataset is empty" in str(exc_info.value)
    
    # ==================== No Valid Species ====================
    
    def test_all_missing_fa_keys(self, service, sample_experiment_six_samples):
        """Test when all rows have missing FA keys (except Ch class)."""
        df = pd.DataFrame({
            'LipidMolec': ['Lipid1', 'Lipid2', 'Lipid3'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'FAKey': [np.nan, np.nan, np.nan],  # All missing FA keys
            'TotalGrade': ['A', 'B', 'C'],
            'CalcMass': [800.0, 750.0, 900.0],
            'BaseRt': [5.0, 6.0, 7.0],
            'TotalSmpIDRate(%)': [90.0, 85.0, 80.0],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1100, 2100, 3100],
            'intensity[s3]': [1200, 2200, 3200],
            'intensity[s4]': [1300, 2300, 3300],
            'intensity[s5]': [1400, 2400, 3400],
            'intensity[s6]': [1500, 2500, 3500]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        assert "No valid lipid species found after filtering" in str(exc_info.value)
        assert "fatty acid (FA) keys" in str(exc_info.value)
    
    def test_all_invalid_grades(self, service, sample_experiment_six_samples):
        """Test when grade filter removes all species."""
        df = pd.DataFrame({
            'LipidMolec': ['Lipid1', 'Lipid2', 'Lipid3'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
            'TotalGrade': ['D', 'D', 'D'],  # All bad grades
            'CalcMass': [800.0, 750.0, 900.0],
            'BaseRt': [5.0, 6.0, 7.0],
            'TotalSmpIDRate(%)': [90.0, 85.0, 80.0],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1100, 2100, 3100],
            'intensity[s3]': [1200, 2200, 3200],
            'intensity[s4]': [1300, 2300, 3300],
            'intensity[s5]': [1400, 2400, 3400],
            'intensity[s6]': [1500, 2500, 3500]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        assert "No lipid species remain after applying grade filters" in str(exc_info.value)
        assert "adjust your grade filter settings" in str(exc_info.value)
    
    def test_all_invalid_lipid_names_generic(self, service, sample_experiment_six_samples):
        """Test when all lipid names are invalid in generic format."""
        df = pd.DataFrame({
            'LipidMolec': ['', '#', '###', np.nan, '   '],  # All invalid
            'intensity[s1]': [1000, 2000, 3000, 4000, 5000],
            'intensity[s2]': [1100, 2100, 3100, 4100, 5100],
            'intensity[s3]': [1200, 2200, 3200, 4200, 5200],
            'intensity[s4]': [1300, 2300, 3300, 4300, 5300],
            'intensity[s5]': [1400, 2400, 3400, 4400, 5400],
            'intensity[s6]': [1500, 2500, 3500, 4500, 5500]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_generic_data(df, sample_experiment_six_samples)
        
        assert "No valid lipid species found" in str(exc_info.value)
        assert "invalid or missing lipid names" in str(exc_info.value)
        assert "LipidMolec" in str(exc_info.value)
    
    def test_all_zero_intensities_generic(self, service, sample_experiment_six_samples):
        """Test when all intensity values are zero."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
            'intensity[s1]': [0, 0, 0],
            'intensity[s2]': [0, 0, 0],
            'intensity[s3]': [0, 0, 0],
            'intensity[s4]': [0, 0, 0],
            'intensity[s5]': [0, 0, 0],
            'intensity[s6]': [0, 0, 0]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_generic_data(df, sample_experiment_six_samples)
        
        assert "No lipid species with non-zero intensity values found" in str(exc_info.value)
        assert "only zeros or null values" in str(exc_info.value)
    
    # ==================== Non-String LipidMolec ====================
    
    def test_numeric_lipid_molec(self, service, sample_experiment_six_samples):
        """Test that numeric LipidMolec values are handled correctly."""
        df = pd.DataFrame({
            'LipidMolec': [123.45, 678.90, np.nan],  # Numeric values
            'ClassKey': ['PC', 'PE', 'Ch'],
            'FAKey': ['16:0_18:1', '18:0_20:4', np.nan],
            'TotalGrade': ['A', 'B', 'A'],
            'CalcMass': [800.0, 750.0, 900.0],
            'BaseRt': [5.0, 6.0, 7.0],
            'TotalSmpIDRate(%)': [90.0, 85.0, 80.0],
            'intensity[s1]': [1000, 2000, 3000],
            'intensity[s2]': [1100, 2100, 3100],
            'intensity[s3]': [1200, 2200, 3200],
            'intensity[s4]': [1300, 2300, 3300],
            'intensity[s5]': [1400, 2400, 3400],
            'intensity[s6]': [1500, 2500, 3500]
        })
        
        # Should not raise AttributeError about .str accessor
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        # Should successfully clean the data
        assert not result.empty
        assert 'LipidMolec' in result.columns
    
    def test_mixed_type_lipid_molec(self, service, sample_experiment_six_samples):
        """Test that mixed-type LipidMolec values (strings and floats) are handled."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 123.45, 'PE(18:0_20:4)', np.nan],
            'ClassKey': ['PC', 'PE', 'PE', 'Ch'],
            'FAKey': ['16:0_18:1', '18:0_20:4', '18:0_20:4', np.nan],
            'TotalGrade': ['A', 'B', 'A', 'A'],
            'CalcMass': [800.0, 750.0, 750.0, 900.0],
            'BaseRt': [5.0, 6.0, 6.5, 7.0],
            'TotalSmpIDRate(%)': [90.0, 85.0, 88.0, 80.0],
            'intensity[s1]': [1000, 2000, 2100, 3000],
            'intensity[s2]': [1100, 2100, 2200, 3100],
            'intensity[s3]': [1200, 2200, 2300, 3200],
            'intensity[s4]': [1300, 2300, 2400, 3300],
            'intensity[s5]': [1400, 2400, 2500, 3400],
            'intensity[s6]': [1500, 2500, 2600, 3500]
        })
        
        # Should not raise AttributeError
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        # Should successfully clean the data
        assert not result.empty
        assert 'LipidMolec' in result.columns
    
    # ==================== ClassKey Sorting with NaN ====================
    
    def test_classkey_with_nan_values(self, service, sample_experiment_six_samples):
        """Test that sorting ClassKey with NaN values doesn't cause TypeError."""
        df = pd.DataFrame({
            'LipidMolec': ['Lipid1', 'Lipid2', 'Lipid3'],
            'ClassKey': ['PC', np.nan, 'PE'],  # NaN in ClassKey
            'FAKey': ['16:0_18:1', '18:0_20:4', '18:0_22:6'],
            'TotalGrade': ['A', 'B', 'A'],
            'CalcMass': [800.0, 750.0, 770.0],
            'BaseRt': [5.0, 6.0, 6.5],
            'TotalSmpIDRate(%)': [90.0, 85.0, 88.0],
            'intensity[s1]': [1000, 2000, 2100],
            'intensity[s2]': [1100, 2100, 2200],
            'intensity[s3]': [1200, 2200, 2300],
            'intensity[s4]': [1300, 2300, 2400],
            'intensity[s5]': [1400, 2400, 2500],
            'intensity[s6]': [1500, 2500, 2600]
        })
        
        # Should not raise TypeError when sorting
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        # The main goal: no TypeError should occur during processing
        # NaN ClassKey rows are NOT filtered out by the service  
        assert not result.empty
        assert len(result) == 3  # All 3 rows present (including NaN ClassKey)
    
    # ==================== Custom Grade Config ====================
    
    def test_empty_grade_config(self, service, sample_experiment):
        """Test when grade config has no selected grades for any class."""
        df = pd.DataFrame({
            'LipidMolec': ['Lipid1', 'Lipid2'],
            'ClassKey': ['PC', 'PE'],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            'TotalGrade': ['A', 'B'],
            'CalcMass': [800.0, 750.0],
            'BaseRt': [5.0, 6.0],
            'TotalSmpIDRate(%)': [90.0, 85.0],
            'intensity[s1]': [1000, 2000],
            'intensity[s2]': [1100, 2100],
            'intensity[s3]': [1200, 2200],
            'intensity[s4]': [1300, 2300]
        })
        
        # Empty grade config - no grades selected for any class
        grade_config = {
            'PC': [],
            'PE': []
        }
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(df, sample_experiment, grade_config)
        
        assert "No lipid species remain after applying grade filters" in str(exc_info.value)
    
    def test_grade_config_for_nonexistent_class(self, service, sample_experiment):
        """Test grade config for class that doesn't exist in data."""
        df = pd.DataFrame({
            'LipidMolec': ['Lipid1'],
            'ClassKey': ['PC'],
            'FAKey': ['16:0_18:1'],
            'TotalGrade': ['A'],
            'CalcMass': [800.0],
            'BaseRt': [5.0],
            'TotalSmpIDRate(%)': [90.0],
            'intensity[s1]': [1000],
            'intensity[s2]': [1100],
            'intensity[s3]': [1200],
            'intensity[s4]': [1300]
        })
        
        # Config for PE class which doesn't exist
        grade_config = {
            'PC': ['A', 'B'],
            'PE': ['A']  # PE doesn't exist in data
        }
        
        # Should not fail, just ignore PE config
        result = service.clean_lipidsearch_data(df, sample_experiment, grade_config)
        assert len(result) == 1
    
    # ==================== Error Message Quality ====================
    
    def test_error_messages_are_descriptive(self, service, sample_experiment):
        """Verify all error messages contain helpful guidance."""
        # Test 1: Empty dataset
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(empty_df, sample_experiment)
        error_msg = str(exc_info.value)
        assert len(error_msg) > 20  # Should be descriptive
        assert "Please" in error_msg or "please" in error_msg  # Should give guidance
        
        # Test 2: No valid species
        df_no_fa = pd.DataFrame({
            'LipidMolec': ['Lipid1'],
            'ClassKey': ['PC'],
            'FAKey': [np.nan],
            'TotalGrade': ['A'],
            'CalcMass': [800.0],
            'BaseRt': [5.0],
            'TotalSmpIDRate(%)': [90.0],
            'intensity[s1]': [1000],
            'intensity[s2]': [1100],
            'intensity[s3]': [1200],
            'intensity[s4]': [1300]
        })
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(df_no_fa, sample_experiment)
        error_msg = str(exc_info.value)
        assert len(error_msg) > 20
        assert "check your data" in error_msg.lower()
    
    def test_error_messages_avoid_technical_jargon(self, service, sample_experiment):
        """Error messages should be understandable by non-programmers."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(empty_df, sample_experiment)
        
        error_msg = str(exc_info.value)
        # Should not contain Python-specific errors
        assert "AttributeError" not in error_msg
        assert "TypeError" not in error_msg
        assert "NoneType" not in error_msg
        # Should contain user-friendly terms
        assert any(word in error_msg.lower() for word in ['dataset', 'file', 'data', 'upload'])
    
    # ==================== Successful Edge Cases ====================
    
    def test_single_lipid_species(self, service, sample_experiment_six_samples):
        """Test dataset with only one lipid species."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'FAKey': ['16:0_18:1'],
            'TotalGrade': ['A'],
            'CalcMass': [800.0],
            'BaseRt': [5.0],
            'TotalSmpIDRate(%)': [90.0],
            'intensity[s1]': [1000],
            'intensity[s2]': [1100],
            'intensity[s3]': [1200],
            'intensity[s4]': [1300],
            'intensity[s5]': [1400],
            'intensity[s6]': [1500]
        })
        
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        assert len(result) == 1
        assert result['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'
    
    def test_ch_class_without_fa_key(self, service, sample_experiment_six_samples):
        """Test that Ch class lipids without FA keys are properly handled."""
        df = pd.DataFrame({
            'LipidMolec': ['Ch-D7'],
            'ClassKey': ['Ch'],
            'FAKey': [np.nan],  # Ch class can have missing FA key
            'TotalGrade': ['A'],
            'CalcMass': [400.0],
            'BaseRt': [3.0],
            'TotalSmpIDRate(%)': [95.0],
            'intensity[s1]': [5000],
            'intensity[s2]': [5100],
            'intensity[s3]': [5200],
            'intensity[s4]': [5300],
            'intensity[s5]': [5400],
            'intensity[s6]': [5500]
        })
        
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        assert len(result) == 1
        assert 'Ch' in result['LipidMolec'].iloc[0]
    
    def test_very_large_dataset(self, service, sample_experiment_six_samples):
        """Test that large datasets are handled efficiently."""
        # Create dataset with 1000 lipid species
        n_lipids = 1000
        df = pd.DataFrame({
            'LipidMolec': [f'PC(16:0_18:{i%10})' for i in range(n_lipids)],
            'ClassKey': ['PC'] * n_lipids,
            'FAKey': [f'16:0_18:{i%10}' for i in range(n_lipids)],
            'TotalGrade': ['A', 'B', 'C'] * (n_lipids // 3) + ['A'] * (n_lipids % 3),
            'CalcMass': [800.0 + i for i in range(n_lipids)],
            'BaseRt': [5.0 + i * 0.01 for i in range(n_lipids)],
            'TotalSmpIDRate(%)': [90.0] * n_lipids,
            'intensity[s1]': [1000 + i for i in range(n_lipids)],
            'intensity[s2]': [1100 + i for i in range(n_lipids)],
            'intensity[s3]': [1200 + i for i in range(n_lipids)],
            'intensity[s4]': [1300 + i for i in range(n_lipids)],
            'intensity[s5]': [1400 + i for i in range(n_lipids)],
            'intensity[s6]': [1500 + i for i in range(n_lipids)]
        })
        
        result = service.clean_lipidsearch_data(df, sample_experiment_six_samples)
        
        # Should complete without errors
        assert not result.empty
        assert len(result) <= n_lipids  # May have duplicates removed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])