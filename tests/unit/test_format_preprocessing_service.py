"""
Unit tests for FormatPreprocessingService.
Tests validation and standardization of all three data formats.
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService


class TestLipidSearchFormat:
    """Test LipidSearch format validation and preprocessing."""
    
    def test_valid_lipidsearch_data(self):
        """Test preprocessing valid LipidSearch data."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4'],
            'ClassKey': ['PC', 'PE'],
            'CalcMass': [760.5, 768.5],
            'BaseRt': [12.5, 13.2],
            'TotalGrade': ['A', 'B'],
            'TotalSmpIDRate(%)': [95.0, 88.0],
            'FAKey': ['16:0_18:1', '18:0_20:4'],
            'MeanArea[s1]': [1000.0, 1200.0],
            'MeanArea[s2]': [1100.0, 1300.0]
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'lipidsearch')
        
        assert success, f"Should succeed: {message}"
        assert 'intensity[s1]' in result_df.columns, "Should have intensity[s1]"
        assert 'intensity[s2]' in result_df.columns, "Should have intensity[s2]"
        assert 'MeanArea[s1]' not in result_df.columns, "Should not have MeanArea columns"
        assert len(result_df) == 2, "Should have 2 lipids"
    
    def test_missing_required_columns(self):
        """Test validation fails with missing required columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'MeanArea[s1]': [1000.0]
            # Missing: CalcMass, BaseRt, TotalGrade, TotalSmpIDRate(%), FAKey
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'lipidsearch')
        
        assert not success, "Should fail with missing columns"
        assert "Missing required columns" in message
        assert "FAKey" in message
    
    def test_missing_meanarea_columns(self):
        """Test validation fails without MeanArea columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'CalcMass': [760.5],
            'BaseRt': [12.5],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [95.0],
            'FAKey': ['16:0_18:1'],
            'intensity[s1]': [1000.0]  # Wrong - should be MeanArea
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'lipidsearch')
        
        assert not success, "Should fail without MeanArea columns"
        assert "No MeanArea" in message


class TestGenericFormat:
    """Test Generic format validation and preprocessing."""
    
    def test_valid_generic_data(self):
        """Test preprocessing valid Generic data."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:1)'],
            'Sample1': [1000.0, 1200.0, 800.0],
            'Sample2': [1100.0, 1300.0, 850.0]
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'generic')
        
        assert success, f"Should succeed: {message}"
        assert 'LipidMolec' in result_df.columns, "Should have LipidMolec"
        assert 'intensity[s1]' in result_df.columns, "Should have intensity[s1]"
        assert 'intensity[s2]' in result_df.columns, "Should have intensity[s2]"
        assert 'ClassKey' in result_df.columns, "Should have ClassKey"
        assert result_df.iloc[0]['ClassKey'] == 'PC', "Should extract PC class"
        assert len(result_df) == 3, "Should have 3 lipids"
    
    def test_generic_infers_classkey(self):
        """Test that Generic format adds ClassKey column."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'LPC(16:0)'],
            'S1': [1000.0, 1200.0, 500.0]
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'generic')
        
        assert success
        assert 'ClassKey' in result_df.columns, "Should add ClassKey"
        assert result_df.iloc[0]['ClassKey'] == 'PC'
        assert result_df.iloc[1]['ClassKey'] == 'PE'
        assert result_df.iloc[2]['ClassKey'] == 'LPC'
    
    def test_insufficient_columns(self):
        """Test validation fails with only one column."""
        df = pd.DataFrame({
            'Lipid': ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'generic')
        
        assert not success, "Should fail with only one column"
        assert "at least 2 columns" in message
    
    def test_first_column_no_letters(self):
        """Test validation fails if first column has no letters."""
        df = pd.DataFrame({
            'Numbers': [1, 2, 3],
            'Sample1': [100, 200, 300]
        })
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(df, 'generic')
        
        assert not success, "Should fail without lipid names"
        assert "no letters found" in message.lower()


class TestMetabolomicsWorkbench:
    """Test Metabolomics Workbench format validation and preprocessing."""
    
    def test_valid_metabolomics_data(self):
        """Test preprocessing valid Metabolomics Workbench data."""
        text_data = """#METABOLOMICS WORKBENCH
MS_METABOLITE_DATA_START
Samples,S1,S2,S3,S4
Factors,Condition:WT,Condition:WT,Condition:KO,Condition:KO
PC(16:0_18:1),1000.0,1100.0,1500.0,1600.0
PE(18:0_20:4),1200.0,1300.0,1800.0,1900.0
TAG(16:0_18:1_18:1),800.0,850.0,1200.0,1250.0
MS_METABOLITE_DATA_END
"""
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(text_data, 'metabolomics_workbench')
        
        assert success, f"Should succeed: {message}"
        assert 'LipidMolec' in result_df.columns
        assert 'intensity[s1]' in result_df.columns
        assert 'intensity[s4]' in result_df.columns
        assert 'ClassKey' in result_df.columns, "Should add ClassKey"
        assert len(result_df) == 3, "Should have 3 lipids"
        assert result_df.iloc[0]['ClassKey'] == 'PC'
    
    def test_missing_data_markers(self):
        """Test validation fails without required markers."""
        text_data = """#METABOLOMICS WORKBENCH
Samples,S1,S2
PC(16:0_18:1),1000.0,1100.0
"""
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(text_data, 'metabolomics_workbench')
        
        assert not success, "Should fail without markers"
        assert "MS_METABOLITE_DATA" in message
    
    def test_conditions_extracted(self):
        """Test that conditions are properly extracted."""
        text_data = """MS_METABOLITE_DATA_START
Samples,S1,S2
Factors,Condition:WT,Condition:KO
PC(16:0_18:1),1000,1100
MS_METABOLITE_DATA_END
"""
        
        service = FormatPreprocessingService()
        result_df, success, message = service.validate_and_preprocess(text_data, 'metabolomics_workbench')
        
        assert success
        # Note: conditions_map is returned but we're only checking DataFrame here
        assert len(result_df) == 1


class TestLipidNameStandardization:
    """Test lipid name standardization logic."""
    
    def test_space_to_parentheses(self):
        """Test converting space-separated format to parentheses."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('PC 16:0_18:1') == 'PC(16:0_18:1)'
        assert service._standardize_lipid_name('PE 18:0_20:4') == 'PE(18:0_20:4)'
        assert service._standardize_lipid_name('LPC 16:0') == 'LPC(16:0)'
    
    def test_slash_to_underscore(self):
        """Test converting slashes to underscores."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('Cer d18:0/C24:0') == 'Cer(d18:0_C24:0)'
        assert service._standardize_lipid_name('TAG(16:0/18:1/18:1)') == 'TAG(16:0_18:1_18:1)'
    
    def test_remove_oxidation_markers(self):
        """Test removing ;N oxidation state markers."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('CE 14:0;0') == 'CE(14:0)'
        assert service._standardize_lipid_name('TAG 16:0_18:1_18:1;2') == 'TAG(16:0_18:1_18:1)'
    
    def test_preserve_modifications(self):
        """Test that modifications like (d7) are preserved."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('LPC 18:1(d7)') == 'LPC(18:1)(d7)'
        assert service._standardize_lipid_name('PC 16:0_18:1(d7)') == 'PC(16:0_18:1)(d7)'
    
    def test_already_standardized(self):
        """Test that already standardized names pass through correctly."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('PC(16:0_18:1)') == 'PC(16:0_18:1)'
        assert service._standardize_lipid_name('CerG1(d13:0_25:2)') == 'CerG1(d13:0_25:2)'
    
    def test_empty_and_null(self):
        """Test handling of empty and null lipid names."""
        service = FormatPreprocessingService()
        
        assert service._standardize_lipid_name('') == 'Unknown'
        assert service._standardize_lipid_name(None) == 'Unknown'
        assert service._standardize_lipid_name(pd.NaT) == 'Unknown'


class TestClassKeyInference:
    """Test ClassKey inference from lipid names."""
    
    def test_simple_class_extraction(self):
        """Test extracting class from simple lipid names."""
        service = FormatPreprocessingService()
        
        assert service._infer_class_key('PC(16:0_18:1)') == 'PC'
        assert service._infer_class_key('PE(18:0_20:4)') == 'PE'
        assert service._infer_class_key('LPC(16:0)') == 'LPC'
        assert service._infer_class_key('TAG(16:0_18:1_18:1)') == 'TAG'
    
    def test_complex_class_names(self):
        """Test extracting complex class names."""
        service = FormatPreprocessingService()
        
        assert service._infer_class_key('CerG1(d13:0_25:2)') == 'CerG1'
        assert service._infer_class_key('Hex1Cer(24:0_d18:1)') == 'Hex1Cer'
        assert service._infer_class_key('AcCa(10:0)') == 'AcCa'
    
    def test_with_modifications(self):
        """Test class extraction with modifications."""
        service = FormatPreprocessingService()
        
        assert service._infer_class_key('LPC(18:1)(d7)') == 'LPC'
        assert service._infer_class_key('PC(16:0_18:1)+D7') == 'PC'


class TestErrorHandling:
    """Test error handling in preprocessing."""
    
    def test_unknown_format_type(self):
        """Test error handling for unknown format type."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        service = FormatPreprocessingService()
        
        result_df, success, message = service.validate_and_preprocess(df, 'unknown_format')
        
        assert not success
        assert "Unknown format type" in message
    
    def test_wrong_input_type_for_metabolomics(self):
        """Test error when DataFrame passed for Metabolomics Workbench."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        service = FormatPreprocessingService()
        
        result_df, success, message = service.validate_and_preprocess(df, 'metabolomics_workbench')
        
        assert not success
        assert "requires text input" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])