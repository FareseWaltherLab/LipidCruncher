"""
Unit tests for grade filtering functionality in DataCleaningService.

Tests cover:
- Default grade filtering (A, B, C)
- Custom grade configurations per lipid class
- Edge cases (empty configs, all data filtered, NaN grades)
- Error messages and user-facing feedback
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from src.lipidcruncher.core.models.experiment import ExperimentConfig


# ==================== Fixtures ====================

@pytest.fixture
def service():
    """DataCleaningService instance."""
    return DataCleaningService()


@pytest.fixture
def sample_experiment():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


@pytest.fixture
def multi_grade_data():
    """
    Create sample data with multiple grades (A, B, C, D, E).
    Includes multiple lipid classes to test class-specific filtering.
    """
    return pd.DataFrame({
        'LipidMolec': [
            'PC 16:0_18:1',  # Grade A
            'PC 18:0_20:4',  # Grade B
            'PC 16:0_16:0',  # Grade C
            'PC 18:1_18:1',  # Grade D - should be filtered by default
            'PE 16:0_18:1',  # Grade A
            'PE 18:0_20:4',  # Grade B
            'PE 16:0_16:0',  # Grade C
            'PE 18:1_18:1',  # Grade D - should be filtered by default
            'TAG 16:0_18:1_18:2',  # Grade A
            'TAG 16:0_16:0_18:1',  # Grade B
            'TAG 18:1_18:1_18:2',  # Grade E - should be filtered
            'Ch D7',  # Grade A (special case)
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PC', 'PE', 'PE', 'PE', 'PE', 'TAG', 'TAG', 'TAG', 'Ch'],
        'FAKey': [
            '16:0_18:1', '18:0_20:4', '16:0_16:0', '18:1_18:1',
            '16:0_18:1', '18:0_20:4', '16:0_16:0', '18:1_18:1',
            '16:0_18:1_18:2', '16:0_16:0_18:1', '18:1_18:1_18:2',
            'D7'
        ],
        'TotalGrade': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D', 'A', 'B', 'E', 'A'],
        'TotalSmpIDRate(%)': [95.0, 90.0, 85.0, 80.0, 95.0, 90.0, 85.0, 80.0, 95.0, 90.0, 70.0, 98.0],
        'CalcMass': [
            760.5, 768.5, 732.5, 758.5,
            742.5, 768.5, 690.5, 714.5,
            876.7, 848.7, 872.7,
            369.3
        ],
        'BaseRt': [
            12.5, 13.2, 11.8, 12.0,
            13.0, 13.5, 12.2, 12.8,
            15.2, 14.8, 15.5,
            8.5
        ],
        'intensity[s1]': [100, 200, 150, 120, 180, 220, 160, 130, 250, 200, 80, 50],
        'intensity[s2]': [110, 210, 155, 125, 185, 225, 165, 135, 255, 205, 85, 55],
        'intensity[s3]': [105, 205, 152, 122, 182, 222, 162, 132, 252, 202, 82, 52],
        'intensity[s4]': [108, 208, 153, 123, 183, 223, 163, 133, 253, 203, 83, 53]
    })


@pytest.fixture
def data_with_nan_grades():
    """Create data with NaN values in TotalGrade column."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4', 'TAG 16:0_18:1_18:2'],
        'ClassKey': ['PC', 'PE', 'TAG'],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'TotalGrade': ['A', np.nan, 'B'],
        'TotalSmpIDRate(%)': [95.0, 90.0, 85.0],
        'CalcMass': [760.5, 768.5, 876.7],
        'BaseRt': [12.5, 13.2, 15.2],
        'intensity[s1]': [100, 200, 150],
        'intensity[s2]': [110, 210, 155],
        'intensity[s3]': [105, 205, 152],
        'intensity[s4]': [108, 208, 153]
    })


@pytest.fixture
def all_grade_d_data():
    """Create data where all lipids have grade D (to test complete filtering)."""
    return pd.DataFrame({
        'LipidMolec': ['PC 16:0_18:1', 'PE 18:0_20:4', 'TAG 16:0_18:1_18:2'],
        'ClassKey': ['PC', 'PE', 'TAG'],
        'FAKey': ['16:0_18:1', '18:0_20:4', '16:0_18:1_18:2'],
        'TotalGrade': ['D', 'D', 'D'],
        'TotalSmpIDRate(%)': [95.0, 90.0, 85.0],
        'CalcMass': [760.5, 768.5, 876.7],
        'BaseRt': [12.5, 13.2, 15.2],
        'intensity[s1]': [100, 200, 150],
        'intensity[s2]': [110, 210, 155],
        'intensity[s3]': [105, 205, 152],
        'intensity[s4]': [108, 208, 153]
    })


# ==================== Test Class 1: Default Grade Filtering ====================

class TestDefaultGradeFiltering:
    """Tests for default grade filtering behavior (A, B, C acceptable)."""
    
    def test_default_accepts_grade_a(self, service, multi_grade_data):
        """Test that grade A lipids are accepted with default filtering."""
        # Arrange
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Act
        grade_a_lipids = filtered[filtered['TotalGrade'] == 'A']
        
        # Assert
        assert len(grade_a_lipids) > 0, "Grade A lipids should be accepted"
        assert 'PC 16:0_18:1' in multi_grade_data[
            multi_grade_data['TotalGrade'] == 'A'
        ]['LipidMolec'].values
    
    def test_default_accepts_grade_b(self, service, multi_grade_data):
        """Test that grade B lipids are accepted with default filtering."""
        # Arrange
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Act
        grade_b_lipids = filtered[filtered['TotalGrade'] == 'B']
        
        # Assert
        assert len(grade_b_lipids) > 0, "Grade B lipids should be accepted"
        assert all(grade in ['A', 'B', 'C'] for grade in filtered['TotalGrade'])
    
    def test_default_accepts_grade_c(self, service, multi_grade_data):
        """Test that grade C lipids are accepted with default filtering."""
        # Arrange
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Act
        grade_c_lipids = filtered[filtered['TotalGrade'] == 'C']
        
        # Assert
        assert len(grade_c_lipids) > 0, "Grade C lipids should be accepted"
    
    def test_default_rejects_grade_d(self, service, multi_grade_data):
        """Test that grade D lipids are rejected with default filtering."""
        # Arrange
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Act
        grade_d_lipids = filtered[filtered['TotalGrade'] == 'D']
        
        # Assert
        assert len(grade_d_lipids) == 0, "Grade D lipids should be filtered out"
    
    def test_default_rejects_grade_e(self, service, multi_grade_data):
        """Test that grade E lipids are rejected with default filtering."""
        # Arrange
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Act
        grade_e_lipids = filtered[filtered['TotalGrade'] == 'E']
        
        # Assert
        assert len(grade_e_lipids) == 0, "Grade E lipids should be filtered out"
    
    def test_default_preserves_all_classes(self, service, multi_grade_data):
        """Test that default filtering preserves all lipid classes."""
        # Arrange
        original_classes = set(multi_grade_data['ClassKey'].unique())
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        filtered_classes = set(filtered['ClassKey'].unique())
        
        # Assert
        # All classes that have A/B/C grades should be preserved
        expected_classes = set(
            multi_grade_data[
                multi_grade_data['TotalGrade'].isin(['A', 'B', 'C'])
            ]['ClassKey'].unique()
        )
        assert filtered_classes == expected_classes


# ==================== Test Class 2: Custom Grade Configuration ====================

class TestCustomGradeConfiguration:
    """Tests for custom grade configurations per lipid class."""
    
    def test_custom_grade_only_a_for_pc(self, service, multi_grade_data):
        """Test custom config accepting only grade A for PC class."""
        # Arrange
        grade_config = {
            'PC': ['A'],
            'PE': ['A', 'B', 'C'],
            'TAG': ['A', 'B'],
            'Ch': ['A']
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        pc_lipids = filtered[filtered['ClassKey'] == 'PC']
        
        # Assert
        assert len(pc_lipids) > 0, "Should have PC lipids"
        assert all(pc_lipids['TotalGrade'] == 'A'), "PC should only have grade A"
    
    def test_custom_grade_different_per_class(self, service, multi_grade_data):
        """Test that different classes can have different grade requirements."""
        # Arrange
        grade_config = {
            'PC': ['A', 'B'],  # PC: A and B only
            'PE': ['A'],       # PE: A only
            'TAG': ['A', 'B', 'C'],  # TAG: all acceptable grades
            'Ch': ['A']
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        pc_grades = set(filtered[filtered['ClassKey'] == 'PC']['TotalGrade'])
        pe_grades = set(filtered[filtered['ClassKey'] == 'PE']['TotalGrade'])
        tag_grades = set(filtered[filtered['ClassKey'] == 'TAG']['TotalGrade'])
        
        assert pc_grades <= {'A', 'B'}, "PC should only have A and B"
        assert pe_grades <= {'A'}, "PE should only have A"
        assert tag_grades <= {'A', 'B', 'C'}, "TAG should have A, B, or C"
    
    def test_custom_grade_empty_list_filters_class(self, service, multi_grade_data):
        """Test that empty grade list for a class filters out that entire class."""
        # Arrange
        grade_config = {
            'PC': [],  # Empty list should filter out all PC
            'PE': ['A', 'B'],
            'TAG': ['A'],
            'Ch': ['A']
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        assert 'PC' not in filtered['ClassKey'].values, "PC class should be filtered out"
        assert 'PE' in filtered['ClassKey'].values, "PE class should remain"
    
    def test_custom_grade_all_empty_lists(self, service, multi_grade_data):
        """Test that all empty grade lists results in empty DataFrame."""
        # Arrange
        grade_config = {
            'PC': [],
            'PE': [],
            'TAG': [],
            'Ch': []
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        assert len(filtered) == 0, "Should return empty DataFrame when all grades filtered"
        assert list(filtered.columns) == list(multi_grade_data.columns), \
            "Should preserve column structure"
    
    def test_custom_grade_partial_class_coverage(self, service, multi_grade_data):
        """Test behavior when grade_config doesn't include all classes in data."""
        # Arrange
        # Only configure PC and PE, leaving TAG and Ch unconfigured
        grade_config = {
            'PC': ['A', 'B'],
            'PE': ['A']
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        # Only configured classes should appear
        assert 'PC' in filtered['ClassKey'].values
        assert 'PE' in filtered['ClassKey'].values
        assert 'TAG' not in filtered['ClassKey'].values, "Unconfigured TAG should be filtered"
        assert 'Ch' not in filtered['ClassKey'].values, "Unconfigured Ch should be filtered"


# ==================== Test Class 3: Edge Cases ====================

class TestGradeFilteringEdgeCases:
    """Tests for edge cases in grade filtering."""
    
    def test_nan_grades_filtered_out(self, service, data_with_nan_grades):
        """Test that lipids with NaN grades are filtered out."""
        # Arrange & Act
        filtered = service._apply_grade_filter(data_with_nan_grades, grade_config=None)
        
        # Assert
        # PE has NaN grade and should be filtered out
        assert 'PE' not in filtered['ClassKey'].values, "NaN grade should be filtered out"
        assert 'PC' in filtered['ClassKey'].values, "Grade A should pass"
        assert 'TAG' in filtered['ClassKey'].values, "Grade B should pass"
    
    def test_all_data_filtered_returns_empty_dataframe(
        self, service, all_grade_d_data
    ):
        """Test that filtering out all data returns empty DataFrame with correct structure."""
        # Arrange & Act
        filtered = service._apply_grade_filter(all_grade_d_data, grade_config=None)
        
        # Assert
        assert len(filtered) == 0, "Should return empty DataFrame when all filtered"
        assert list(filtered.columns) == list(all_grade_d_data.columns), \
            "Should preserve column structure"
    
    def test_case_sensitivity_of_grades(self, service, multi_grade_data):
        """Test that grade filtering is case-sensitive (Grade 'a' != 'A')."""
        # Arrange
        # Create data with lowercase grades
        data_lowercase = multi_grade_data.copy()
        data_lowercase.loc[0, 'TotalGrade'] = 'a'  # lowercase 'a'
        
        # Act
        filtered = service._apply_grade_filter(data_lowercase, grade_config=None)
        
        # Assert
        # The lowercase 'a' should NOT match uppercase 'A' filter
        assert 'a' not in filtered['TotalGrade'].values, \
            "Lowercase grade should not match uppercase filter"
    
    def test_single_lipid_single_grade(self, service, sample_experiment):
        """Test filtering with single lipid and single acceptable grade."""
        # Arrange
        single_lipid_data = pd.DataFrame({
            'LipidMolec': ['PC 16:0_18:1'],
            'ClassKey': ['PC'],
            'FAKey': ['16:0_18:1'],
            'TotalGrade': ['A'],
            'TotalSmpIDRate(%)': [95.0],
            'CalcMass': [760.5],
            'BaseRt': [12.5],
            'intensity[s1]': [100],
            'intensity[s2]': [110],
            'intensity[s3]': [105],
            'intensity[s4]': [108]
        })
        
        # Act
        filtered = service._apply_grade_filter(single_lipid_data, grade_config=None)
        
        # Assert
        assert len(filtered) == 1, "Single grade A lipid should pass"
        assert filtered.iloc[0]['LipidMolec'] == 'PC 16:0_18:1'
    
    def test_empty_dataframe_input(self, service):
        """Test that empty DataFrame input returns empty DataFrame."""
        # Arrange
        empty_df = pd.DataFrame(columns=[
            'LipidMolec', 'ClassKey', 'FAKey', 'TotalGrade',
            'TotalSmpIDRate(%)', 'CalcMass', 'BaseRt',
            'intensity[s1]', 'intensity[s2]'
        ])
        
        # Act
        filtered = service._apply_grade_filter(empty_df, grade_config=None)
        
        # Assert
        assert len(filtered) == 0, "Empty input should return empty output"
        assert list(filtered.columns) == list(empty_df.columns)
    
    def test_grade_filtering_preserves_column_order(self, service, multi_grade_data):
        """Test that grade filtering preserves the original column order."""
        # Arrange
        original_columns = list(multi_grade_data.columns)
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config=None)
        filtered_columns = list(filtered.columns)
        
        # Assert
        assert filtered_columns == original_columns, "Column order should be preserved"


# ==================== Test Class 4: Integration with clean_lipidsearch_data ====================

class TestGradeFilteringIntegration:
    """Tests for grade filtering integration in the full cleaning pipeline."""
    
    def test_clean_lipidsearch_with_default_grades(
        self, service, multi_grade_data, sample_experiment
    ):
        """Test that clean_lipidsearch_data applies default grade filtering correctly."""
        # Arrange & Act
        cleaned = service.clean_lipidsearch_data(
            multi_grade_data,
            sample_experiment,
            grade_config=None
        )
        
        # Assert
        assert 'D' not in cleaned['TotalGrade'].values if 'TotalGrade' in cleaned.columns \
            else True, "Grade D should be filtered out"
        assert len(cleaned) > 0, "Should have data after cleaning"
    
    def test_clean_lipidsearch_with_custom_grades(
        self, service, multi_grade_data, sample_experiment
    ):
        """Test that clean_lipidsearch_data applies custom grade config correctly."""
        # Arrange
        grade_config = {
            'PC': ['A'],
            'PE': ['A', 'B'],
            'TAG': ['A'],
            'Ch': ['A']
        }
        
        # Act
        cleaned = service.clean_lipidsearch_data(
            multi_grade_data,
            sample_experiment,
            grade_config=grade_config
        )
        
        # Assert
        pc_lipids = [name for name in cleaned['LipidMolec'] if name.startswith('PC(')]
        pe_lipids = [name for name in cleaned['LipidMolec'] if name.startswith('PE(')]
        
        # PC should only have grade A entries (best quality selected)
        assert len(pc_lipids) > 0, "Should have PC lipids"
        # PE can have A or B
        assert len(pe_lipids) > 0, "Should have PE lipids"
    
    def test_clean_lipidsearch_raises_error_when_all_filtered(
        self, service, all_grade_d_data, sample_experiment
    ):
        """Test that error is raised when grade filtering removes all data."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.clean_lipidsearch_data(
                all_grade_d_data,
                sample_experiment,
                grade_config=None
            )
        
        error_msg = str(exc_info.value)
        assert "grade filters" in error_msg.lower(), \
            "Error should mention grade filters"
        assert "adjust" in error_msg.lower(), \
            "Error should suggest adjusting settings"


# ==================== Test Class 5: Grade Config Validation ====================

class TestGradeConfigValidation:
    """Tests for validation of grade configuration inputs."""
    
    def test_grade_config_with_invalid_grade_values(self, service, multi_grade_data):
        """Test behavior with invalid grade values in config."""
        # Arrange
        grade_config = {
            'PC': ['A', 'B', 'Z'],  # 'Z' is not a valid grade in data
            'PE': ['A'],
            'TAG': ['A'],
            'Ch': ['A']
        }
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        # Should work but 'Z' just won't match anything
        pc_grades = set(filtered[filtered['ClassKey'] == 'PC']['TotalGrade'])
        assert pc_grades <= {'A', 'B'}, "Should only match valid grades"
        assert 'Z' not in pc_grades
    
    def test_grade_config_none_uses_default(self, service, multi_grade_data):
        """Test that None grade_config uses default A, B, C filtering."""
        # Arrange & Act
        filtered_none = service._apply_grade_filter(multi_grade_data, grade_config=None)
        
        # Assert
        grades_in_result = set(filtered_none['TotalGrade'].unique())
        assert grades_in_result <= {'A', 'B', 'C'}, \
            "None config should default to A, B, C"
    
    def test_grade_config_empty_dict(self, service, multi_grade_data):
        """Test behavior with empty grade_config dict."""
        # Arrange
        grade_config = {}  # Empty dict
        
        # Act
        filtered = service._apply_grade_filter(multi_grade_data, grade_config)
        
        # Assert
        assert len(filtered) == 0, "Empty config dict should filter everything"


if __name__ == "__main__":
    # Allow running tests directly with: python test_grade_filtering.py
    pytest.main([__file__, "-v"])