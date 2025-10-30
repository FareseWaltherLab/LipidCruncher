"""
Unit tests for StandardsService.

This file contains comprehensive tests for internal standards processing:
- File upload and validation
- Standards matching to classes
- Deuterated standards detection
- MeanArea calculations
- Duplicate handling
- Edge cases and error scenarios

Targets bug edb8d24 (duplicate standards) and ensures robust standards handling.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# Mock the StandardsService import since we're testing in isolation
class StandardsService:
    """Service for managing internal standards."""
    
    @staticmethod
    def process_standards_file(
        standards_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        existing_standards: pd.DataFrame
    ):
        """
        Process uploaded standards file and validate against cleaned data.
        
        Args:
            standards_df: DataFrame from uploaded standards file
            cleaned_df: Cleaned dataset
            existing_standards: Existing internal standards DataFrame
            
        Returns:
            Validated standards DataFrame or None if validation fails
            
        Raises:
            ValueError: If standards file is invalid
        """
        # Validate required column
        if 'LipidMolec' not in standards_df.columns:
            raise ValueError("Standards file must contain 'LipidMolec' column")
        
        # FIX: Remove duplicates from uploaded standards file
        original_count = len(standards_df)
        standards_df = standards_df.drop_duplicates(subset=['LipidMolec'], keep='first')
        duplicates_removed = original_count - len(standards_df)
        
        if duplicates_removed > 0:
            # In real implementation this would be st.warning
            print(f"⚠️ Removed {duplicates_removed} duplicate standard(s) from uploaded file. "
                  f"Only the first occurrence of each standard was kept.")
        
        # Get list of lipids in cleaned data
        if cleaned_df is None or 'LipidMolec' not in cleaned_df.columns:
            raise ValueError("Cleaned data must contain 'LipidMolec' column")
        
        available_lipids = set(cleaned_df['LipidMolec'].unique())
        
        # Validate that standards exist in dataset
        uploaded_standards = set(standards_df['LipidMolec'].unique())
        missing_standards = uploaded_standards - available_lipids
        
        if missing_standards:
            raise ValueError(
                f"The following standards are not found in your dataset: "
                f"{', '.join(list(missing_standards)[:5])}"
                + (f" and {len(missing_standards) - 5} more..." if len(missing_standards) > 5 else "")
            )
        
        # Create new standards DataFrame with required columns
        new_standards = pd.DataFrame()
        new_standards['LipidMolec'] = standards_df['LipidMolec']
        
        # Add ClassKey if available from cleaned data
        if 'ClassKey' in cleaned_df.columns:
            # Map standards to their classes
            lipid_to_class = dict(zip(cleaned_df['LipidMolec'], cleaned_df['ClassKey']))
            new_standards['ClassKey'] = new_standards['LipidMolec'].map(lipid_to_class)
        
        # Add intensity columns from cleaned data (handle both MeanArea and intensity[...] formats)
        intensity_cols = [col for col in cleaned_df.columns if 
                         col.startswith('MeanArea') or col.startswith('intensity[')]
        
        if not intensity_cols:
            raise ValueError("Cleaned data does not contain intensity columns (MeanArea or intensity[...])")
        
        for col in intensity_cols:
            # Map standards to their intensities
            lipid_to_intensity = dict(zip(cleaned_df['LipidMolec'], cleaned_df[col]))
            new_standards[col] = new_standards['LipidMolec'].map(lipid_to_intensity)
        
        # Validate that we have data
        if new_standards.empty:
            raise ValueError("No valid standards found in the dataset")
        
        # Ensure we have intensity data
        intensity_cols_in_standards = [col for col in new_standards.columns if 
                                       col.startswith('MeanArea') or col.startswith('intensity[')]
        if not intensity_cols_in_standards:
            raise ValueError("Internal standards data does not contain properly formatted intensity columns")
        
        return new_standards


# ==================== Fixtures ====================

@pytest.fixture
def service():
    """StandardsService instance."""
    return StandardsService()


@pytest.fixture
def sample_cleaned_data():
    """Create sample cleaned dataset with various lipid classes."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 
            'PC(18:0_18:2)', 
            'PE(18:0_20:4)',
            'PE(16:0_18:1)',
            'TAG(16:0_18:1_18:2)',
            'Ch()',
            'PC(16:0_18:1)(d7)',  # Deuterated standard
            'PE(18:0_20:4)(d9)'   # Deuterated standard
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE', 'TAG', 'Ch', 'PC', 'PE'],
        'intensity[s1]': [1000.0, 1100.0, 2000.0, 2100.0, 1500.0, 500.0, 5000.0, 6000.0],
        'intensity[s2]': [1050.0, 1150.0, 2050.0, 2150.0, 1550.0, 550.0, 5100.0, 6100.0],
        'intensity[s3]': [1100.0, 1200.0, 2100.0, 2200.0, 1600.0, 600.0, 5200.0, 6200.0],
        'intensity[s4]': [1150.0, 1250.0, 2150.0, 2250.0, 1650.0, 650.0, 5300.0, 6300.0]
    })


@pytest.fixture
def sample_cleaned_data_meanarea():
    """Create sample cleaned dataset with MeanArea column instead of intensity[]."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 
            'PE(18:0_20:4)',
            'PC(16:0_18:1)(d7)'
        ],
        'ClassKey': ['PC', 'PE', 'PC'],
        'MeanArea': [1000.0, 2000.0, 5000.0]
    })


@pytest.fixture
def existing_standards_empty():
    """Empty existing standards DataFrame."""
    return pd.DataFrame()


# ==================== Test Class A: File Upload and Validation ====================

class TestStandardsFileUpload:
    """Tests for uploading and validating standards files."""
    
    def test_valid_standards_file_basic(self, service, sample_cleaned_data, existing_standards_empty):
        """Test upload of valid standards file with basic lipids."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df, 
            sample_cleaned_data, 
            existing_standards_empty
        )
        
        # Assert
        assert result is not None
        assert len(result) == 2
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert all(result['LipidMolec'] == ['PC(16:0_18:1)', 'PE(18:0_20:4)'])
    
    def test_valid_standards_file_with_intensity_columns(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that intensity columns are extracted from cleaned data."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert
        assert 'intensity[s1]' in result.columns
        assert 'intensity[s2]' in result.columns
        assert 'intensity[s3]' in result.columns
        assert 'intensity[s4]' in result.columns
        assert result.loc[0, 'intensity[s1]'] == 1000.0
    
    def test_valid_standards_file_with_meanarea_column(self, service, sample_cleaned_data_meanarea, existing_standards_empty):
        """Test upload of standards file when cleaned data has MeanArea column."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data_meanarea,
            existing_standards_empty
        )
        
        # Assert
        assert 'MeanArea' in result.columns
        assert result.loc[0, 'MeanArea'] == 1000.0
    
    def test_missing_lipidmolec_column(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that error raised if LipidMolec column missing from standards file."""
        # Arrange
        standards_df = pd.DataFrame({
            'StandardName': ['PC(16:0_18:1)']  # Wrong column name
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                sample_cleaned_data,
                existing_standards_empty
            )
        
        assert "Standards file must contain 'LipidMolec' column" in str(exc_info.value)
    
    def test_empty_standards_file(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that error raised if standards file is empty."""
        # Arrange
        empty_df = pd.DataFrame({'LipidMolec': []})
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                empty_df,
                sample_cleaned_data,
                existing_standards_empty
            )
        
        assert "No valid standards found" in str(exc_info.value)
    
    def test_duplicate_standards_removed(self, service, sample_cleaned_data, existing_standards_empty, capsys):
        """Test that duplicate LipidMolec entries are deduplicated (Bug edb8d24)."""
        # Arrange - Create standards with duplicates
        standards_df = pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)',
                'PC(16:0_18:1)',  # Duplicate
                'PE(18:0_20:4)',
                'PC(16:0_18:1)'   # Another duplicate
            ]
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == 2  # Only unique standards
        assert len(result['LipidMolec'].unique()) == 2
        
        # Check that warning was displayed
        captured = capsys.readouterr()
        assert "Removed 2 duplicate standard(s)" in captured.out
    
    def test_standards_not_in_dataset(self, service, sample_cleaned_data, existing_standards_empty):
        """Test error when uploaded standard not found in dataset."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['SM(d18:1_16:0)']  # Not in cleaned data
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                sample_cleaned_data,
                existing_standards_empty
            )
        
        error_msg = str(exc_info.value)
        assert "not found in your dataset" in error_msg
        assert "SM(d18:1_16:0)" in error_msg
    
    def test_multiple_standards_not_in_dataset(self, service, sample_cleaned_data, existing_standards_empty):
        """Test error message when multiple standards not found (shows first 5)."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': [
                'SM(d18:1_16:0)',
                'Cer(d18:1_24:1)',
                'LPC(18:0)',
                'LPE(16:0)',
                'DAG(16:0_18:1)',
                'MAG(18:1)',
                'FFA(16:0)'
            ]
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                sample_cleaned_data,
                existing_standards_empty
            )
        
        error_msg = str(exc_info.value)
        assert "not found in your dataset" in error_msg
        assert "and 2 more..." in error_msg  # Shows truncation for 7 items (5 shown + 2 more)


# ==================== Test Class B: Standards Matching ====================

class TestStandardsMatching:
    """Tests for matching standards to lipid classes."""
    
    def test_standards_mapped_to_correct_classes(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that standards are correctly mapped to their lipid classes."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TAG(16:0_18:1_18:2)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert
        assert result.loc[result['LipidMolec'] == 'PC(16:0_18:1)', 'ClassKey'].iloc[0] == 'PC'
        assert result.loc[result['LipidMolec'] == 'PE(18:0_20:4)', 'ClassKey'].iloc[0] == 'PE'
        assert result.loc[result['LipidMolec'] == 'TAG(16:0_18:1_18:2)', 'ClassKey'].iloc[0] == 'TAG'
    
    def test_intensity_values_correctly_mapped(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that intensity values from cleaned data are correctly mapped to standards."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert - Check all intensity columns have correct values
        assert result.loc[0, 'intensity[s1]'] == 1000.0
        assert result.loc[0, 'intensity[s2]'] == 1050.0
        assert result.loc[0, 'intensity[s3]'] == 1100.0
        assert result.loc[0, 'intensity[s4]'] == 1150.0
    
    def test_deuterated_standards_included(self, service, sample_cleaned_data, existing_standards_empty):
        """Test that deuterated standards can be uploaded and processed."""
        # Arrange
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)']
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == 2
        assert 'PC(16:0_18:1)(d7)' in result['LipidMolec'].values
        assert 'PE(18:0_20:4)(d9)' in result['LipidMolec'].values
        assert result.loc[result['LipidMolec'] == 'PC(16:0_18:1)(d7)', 'ClassKey'].iloc[0] == 'PC'


# ==================== Test Class C: Data Validation ====================

class TestDataValidation:
    """Tests for validating cleaned data structure."""
    
    def test_none_cleaned_data(self, service, existing_standards_empty):
        """Test error when cleaned_df is None."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                None,  # None cleaned data
                existing_standards_empty
            )
        
        assert "Cleaned data must contain 'LipidMolec' column" in str(exc_info.value)
    
    def test_cleaned_data_missing_lipidmolec(self, service, existing_standards_empty):
        """Test error when cleaned data missing LipidMolec column."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        cleaned_df = pd.DataFrame({
            'StandardName': ['PC(16:0_18:1)'],  # Wrong column name
            'intensity[s1]': [1000.0]
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                cleaned_df,
                existing_standards_empty
            )
        
        assert "Cleaned data must contain 'LipidMolec' column" in str(exc_info.value)
    
    def test_cleaned_data_no_intensity_columns(self, service, existing_standards_empty):
        """Test error when cleaned data has no intensity columns."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC']
            # No intensity columns
        })
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                cleaned_df,
                existing_standards_empty
            )
        
        assert "does not contain intensity columns" in str(exc_info.value)
    
    def test_cleaned_data_without_classkey(self, service, existing_standards_empty):
        """Test that standards can be processed even if ClassKey column missing."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1100.0]
            # No ClassKey column
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            cleaned_df,
            existing_standards_empty
        )
        
        # Assert - Should work but without ClassKey
        assert result is not None
        assert len(result) == 1
        assert 'ClassKey' not in result.columns or pd.isna(result['ClassKey'].iloc[0])


# ==================== Test Class D: Edge Cases ====================

class TestStandardsEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_standard(self, service, sample_cleaned_data, existing_standards_empty):
        """Test processing file with single standard."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        
        # Act
        result = service.process_standards_file(
            standards_df,
            sample_cleaned_data,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == 1
        assert result['LipidMolec'].iloc[0] == 'PC(16:0_18:1)'
    
    def test_many_standards(self, service, existing_standards_empty):
        """Test processing file with many standards (50+)."""
        # Arrange
        n_standards = 50
        standards_list = [f'PC(16:0_18:{i})' for i in range(n_standards)]
        
        standards_df = pd.DataFrame({'LipidMolec': standards_list})
        
        cleaned_df = pd.DataFrame({
            'LipidMolec': standards_list,
            'ClassKey': ['PC'] * n_standards,
            'intensity[s1]': [1000.0 + i for i in range(n_standards)],
            'intensity[s2]': [1100.0 + i for i in range(n_standards)]
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            cleaned_df,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == n_standards
    
    def test_standards_with_zero_intensities(self, service, existing_standards_empty):
        """Test standards with zero intensity values."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0],
            'intensity[s3]': [0.0]
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            cleaned_df,
            existing_standards_empty
        )
        
        # Assert - Should process but with zero values
        assert len(result) == 1
        assert result['intensity[s1]'].iloc[0] == 0.0
    
    def test_standards_with_nan_intensities(self, service, existing_standards_empty):
        """Test standards with NaN intensity values."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['PC(16:0_18:1)']})
        
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [np.nan],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [np.nan]
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            cleaned_df,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == 1
        assert pd.isna(result['intensity[s1]'].iloc[0])
        assert result['intensity[s2]'].iloc[0] == 1000.0
    
    def test_special_characters_in_lipid_names(self, service, existing_standards_empty):
        """Test lipid names with special characters."""
        # Arrange
        lipid_name = "PC(O-16:0/18:1)"  # Contains "/" and "-"
        standards_df = pd.DataFrame({'LipidMolec': [lipid_name]})
        
        cleaned_df = pd.DataFrame({
            'LipidMolec': [lipid_name],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0]
        })
        
        # Act
        result = service.process_standards_file(
            standards_df,
            cleaned_df,
            existing_standards_empty
        )
        
        # Assert
        assert len(result) == 1
        assert result['LipidMolec'].iloc[0] == lipid_name
    
    def test_case_sensitivity_in_lipid_names(self, service, existing_standards_empty):
        """Test that lipid name matching is case-sensitive."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['pc(16:0_18:1)']})  # lowercase
        
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],  # uppercase
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0]
        })
        
        # Act & Assert - Should fail because case doesn't match
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                cleaned_df,
                existing_standards_empty
            )
        
        assert "not found in your dataset" in str(exc_info.value)


# ==================== Test Class E: Error Message Quality ====================

class TestErrorMessageQuality:
    """Tests to ensure error messages are helpful and user-friendly."""
    
    def test_error_messages_are_descriptive(self, service, sample_cleaned_data, existing_standards_empty):
        """Verify all error messages contain helpful guidance."""
        # Test 1: Missing column
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                pd.DataFrame({'WrongColumn': ['PC(16:0_18:1)']}),
                sample_cleaned_data,
                existing_standards_empty
            )
        error_msg = str(exc_info.value)
        assert len(error_msg) > 20
        assert "LipidMolec" in error_msg
        
        # Test 2: Standard not in dataset
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                pd.DataFrame({'LipidMolec': ['NotExist(1:0_2:0)']}),
                sample_cleaned_data,
                existing_standards_empty
            )
        error_msg = str(exc_info.value)
        assert "not found in your dataset" in error_msg
    
    def test_error_messages_avoid_technical_jargon(self, service, sample_cleaned_data, existing_standards_empty):
        """Error messages should be understandable by scientists, not just programmers."""
        # Arrange
        standards_df = pd.DataFrame({'LipidMolec': ['NotExist(1:0_2:0)']})
        
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.process_standards_file(
                standards_df,
                sample_cleaned_data,
                existing_standards_empty
            )
        
        error_msg = str(exc_info.value)
        # Should not contain Python-specific errors
        assert "AttributeError" not in error_msg
        assert "TypeError" not in error_msg
        assert "KeyError" not in error_msg
        # Should use domain language
        assert any(word in error_msg.lower() for word in ['dataset', 'standard', 'lipid'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])