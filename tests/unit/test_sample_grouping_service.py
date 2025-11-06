"""
Unit Tests for SampleGroupingService

Tests all functionality of the sample grouping service including:
- Building group DataFrames
- Validating selections
- Reordering DataFrames
- Creating name mappings
- Extracting sample names
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add the service to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from sample_grouping_service import SampleGroupingService


@pytest.fixture
def service():
    """Fixture providing a SampleGroupingService instance."""
    return SampleGroupingService()


@pytest.fixture
def simple_experiment():
    """Fixture providing simple experiment configuration."""
    return {
        'sample_list': ['s1', 's2', 's3', 's4', 's5', 's6'],
        'condition_list': ['WT', 'WT', 'WT', 'KO', 'KO', 'KO'],
        'conditions': ['WT', 'KO'],
        'n_samples': [3, 3]
    }


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample DataFrame with intensity columns."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [100, 200],
        'intensity[s2]': [110, 210],
        'intensity[s3]': [120, 220],
        'intensity[s4]': [130, 230],
        'intensity[s5]': [140, 240],
        'intensity[s6]': [150, 250]
    })


# =============================================================================
# Test: build_group_dataframe
# =============================================================================

class TestBuildGroupDataframe:
    """Tests for build_group_dataframe method."""
    
    def test_creates_correct_structure(self, service, simple_experiment):
        """Test that DataFrame has correct structure and data."""
        result = service.build_group_dataframe(
            simple_experiment['sample_list'],
            simple_experiment['condition_list']
        )
        
        assert len(result) == 6
        assert list(result.columns) == ['sample_name', 'condition']
        assert result['sample_name'].tolist() == ['s1', 's2', 's3', 's4', 's5', 's6']
        assert result['condition'].tolist() == ['WT', 'WT', 'WT', 'KO', 'KO', 'KO']
    
    def test_handles_single_sample(self, service):
        """Test with single sample per condition."""
        result = service.build_group_dataframe(['s1'], ['Control'])
        
        assert len(result) == 1
        assert result.iloc[0]['sample_name'] == 's1'
        assert result.iloc[0]['condition'] == 'Control'
    
    def test_handles_many_conditions(self, service):
        """Test with many conditions."""
        samples = [f's{i}' for i in range(1, 11)]
        conditions = ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C5']
        
        result = service.build_group_dataframe(samples, conditions)
        
        assert len(result) == 10
        assert result['condition'].tolist() == conditions
    
    def test_raises_on_mismatched_lengths(self, service):
        """Test that error is raised when lists have different lengths."""
        with pytest.raises(ValueError, match="Sample list length"):
            service.build_group_dataframe(['s1', 's2'], ['WT'])


# =============================================================================
# Test: validate_selections
# =============================================================================

class TestValidateSelections:
    """Tests for validate_selections method."""
    
    def test_valid_selections(self, service, simple_experiment):
        """Test with valid selections."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6']
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is True
        assert message == ""
    
    def test_wrong_sample_count(self, service, simple_experiment):
        """Test with wrong number of samples for a condition."""
        selections = {
            'WT': ['s1', 's2'],  # Only 2 instead of 3
            'KO': ['s4', 's5', 's6']
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is False
        assert "WT" in message
        assert "needs 3" in message
        assert "got 2" in message
    
    def test_duplicate_samples(self, service, simple_experiment):
        """Test detection of duplicate samples across conditions."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s3', 's4', 's5']  # s3 duplicated
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is False
        assert "Duplicate" in message
        assert "s3" in message
    
    def test_missing_condition(self, service, simple_experiment):
        """Test when a condition is missing from selections."""
        selections = {
            'WT': ['s1', 's2', 's3']
            # KO is missing
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is False
        assert "Missing conditions" in message
        assert "KO" in message
    
    def test_extra_condition(self, service, simple_experiment):
        """Test when selections include an unexpected condition."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6'],
            'Extra': ['s7']  # Not in experiment
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is False
        assert "Extra conditions" in message
        assert "Extra" in message
    
    def test_multiple_duplicates(self, service, simple_experiment):
        """Test detection of multiple duplicate samples."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s1', 's2', 's3']  # All duplicated
        }
        
        valid, message = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        
        assert valid is False
        assert "Duplicate" in message


# =============================================================================
# Test: reorder_dataframe
# =============================================================================

class TestReorderDataframe:
    """Tests for reorder_dataframe method."""
    
    def test_reorders_correctly(self, service, sample_dataframe, simple_experiment):
        """Test that DataFrame columns are reordered correctly."""
        selections = {
            'WT': ['s1', 's3', 's5'],  # Reordered
            'KO': ['s2', 's4', 's6']
        }
        
        result = service.reorder_dataframe(
            sample_dataframe,
            selections,
            simple_experiment['conditions']
        )
        
        # Check column order
        intensity_cols = [col for col in result.columns if col.startswith('intensity[')]
        assert intensity_cols == [f'intensity[s{i}]' for i in range(1, 7)]
        
        # Check values reordered correctly (new s1 = old s1, new s2 = old s3, etc.)
        assert result['intensity[s1]'].tolist() == sample_dataframe['intensity[s1]'].tolist()
        assert result['intensity[s2]'].tolist() == sample_dataframe['intensity[s3]'].tolist()
        assert result['intensity[s3]'].tolist() == sample_dataframe['intensity[s5]'].tolist()
        assert result['intensity[s4]'].tolist() == sample_dataframe['intensity[s2]'].tolist()
    
    def test_preserves_static_columns(self, service, sample_dataframe, simple_experiment):
        """Test that non-intensity columns are preserved."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6']
        }
        
        result = service.reorder_dataframe(
            sample_dataframe,
            selections,
            simple_experiment['conditions']
        )
        
        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert result['LipidMolec'].tolist() == sample_dataframe['LipidMolec'].tolist()
        assert result['ClassKey'].tolist() == sample_dataframe['ClassKey'].tolist()
    
    def test_handles_no_reordering(self, service, sample_dataframe, simple_experiment):
        """Test when samples are already in correct order."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6']
        }
        
        result = service.reorder_dataframe(
            sample_dataframe,
            selections,
            simple_experiment['conditions']
        )
        
        # Values should match original
        for i in range(1, 7):
            col = f'intensity[s{i}]'
            assert result[col].tolist() == sample_dataframe[col].tolist()
    
    def test_raises_on_missing_sample(self, service, sample_dataframe, simple_experiment):
        """Test error when referenced sample doesn't exist."""
        selections = {
            'WT': ['s1', 's2', 's99'],  # s99 doesn't exist
            'KO': ['s4', 's5', 's6']
        }
        
        with pytest.raises(ValueError, match="not found in DataFrame"):
            service.reorder_dataframe(
                sample_dataframe,
                selections,
                simple_experiment['conditions']
            )
    
    def test_raises_on_missing_condition(self, service, sample_dataframe):
        """Test error when condition not in selections."""
        selections = {
            'WT': ['s1', 's2', 's3']
            # Missing KO
        }
        
        with pytest.raises(ValueError, match="not found in selections"):
            service.reorder_dataframe(
                sample_dataframe,
                selections,
                ['WT', 'KO']
            )


# =============================================================================
# Test: create_name_mapping
# =============================================================================

class TestCreateNameMapping:
    """Tests for create_name_mapping method."""
    
    def test_creates_correct_mapping(self, service, simple_experiment):
        """Test that mapping DataFrame is created correctly."""
        selections = {
            'WT': ['s1', 's3', 's5'],
            'KO': ['s2', 's4', 's6']
        }
        
        result = service.create_name_mapping(
            selections,
            simple_experiment['conditions']
        )
        
        assert len(result) == 6
        assert list(result.columns) == ['old_name', 'new_name', 'condition']
        assert result['old_name'].tolist() == ['s1', 's3', 's5', 's2', 's4', 's6']
        assert result['new_name'].tolist() == ['s1', 's2', 's3', 's4', 's5', 's6']
        assert result['condition'].tolist() == ['WT', 'WT', 'WT', 'KO', 'KO', 'KO']
    
    def test_handles_no_reordering(self, service, simple_experiment):
        """Test mapping when no reordering occurs."""
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6']
        }
        
        result = service.create_name_mapping(
            selections,
            simple_experiment['conditions']
        )
        
        # Old and new names should match
        assert result['old_name'].tolist() == result['new_name'].tolist()
    
    def test_handles_complete_reordering(self, service, simple_experiment):
        """Test mapping with complete reordering."""
        selections = {
            'WT': ['s6', 's5', 's4'],
            'KO': ['s3', 's2', 's1']
        }
        
        result = service.create_name_mapping(
            selections,
            simple_experiment['conditions']
        )
        
        assert result['old_name'].tolist() == ['s6', 's5', 's4', 's3', 's2', 's1']
        assert result['new_name'].tolist() == ['s1', 's2', 's3', 's4', 's5', 's6']


# =============================================================================
# Test: extract_sample_names_from_dataframe
# =============================================================================

class TestExtractSampleNames:
    """Tests for extract_sample_names_from_dataframe method."""
    
    def test_extracts_sample_names(self, service, sample_dataframe):
        """Test extraction of sample names from intensity columns."""
        result = service.extract_sample_names_from_dataframe(sample_dataframe)
        
        assert result == ['s1', 's2', 's3', 's4', 's5', 's6']
    
    def test_handles_no_intensity_columns(self, service):
        """Test with DataFrame that has no intensity columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC']
        })
        
        result = service.extract_sample_names_from_dataframe(df)
        
        assert result == []
    
    def test_handles_mixed_columns(self, service):
        """Test with DataFrame having intensity and non-intensity columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100],
            'MeanArea': [500],
            'intensity[s2]': [200],
            'TotalGrade': ['A']
        })
        
        result = service.extract_sample_names_from_dataframe(df)
        
        assert result == ['s1', 's2']
    
    def test_preserves_sample_order(self, service):
        """Test that sample order matches column order."""
        df = pd.DataFrame({
            'intensity[s5]': [100],
            'intensity[s2]': [200],
            'intensity[s8]': [300],
            'intensity[s1]': [400]
        })
        
        result = service.extract_sample_names_from_dataframe(df)
        
        # Order should match DataFrame column order
        assert result == ['s5', 's2', 's8', 's1']


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests simulating real usage scenarios."""
    
    def test_complete_reordering_workflow(self, service, sample_dataframe, simple_experiment):
        """Test complete workflow: build -> validate -> reorder -> map."""
        # Step 1: Build initial group DataFrame
        group_df = service.build_group_dataframe(
            simple_experiment['sample_list'],
            simple_experiment['condition_list']
        )
        assert len(group_df) == 6
        
        # Step 2: User makes manual selections
        selections = {
            'WT': ['s3', 's1', 's5'],
            'KO': ['s6', 's2', 's4']
        }
        
        # Step 3: Validate selections
        valid, msg = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        assert valid is True
        
        # Step 4: Reorder DataFrame
        reordered_df = service.reorder_dataframe(
            sample_dataframe,
            selections,
            simple_experiment['conditions']
        )
        
        # Step 5: Create name mapping for display
        mapping = service.create_name_mapping(
            selections,
            simple_experiment['conditions']
        )
        
        # Verify final result
        assert len(reordered_df) == len(sample_dataframe)
        assert 'intensity[s1]' in reordered_df.columns
        assert len(mapping) == 6
        assert mapping.iloc[0]['old_name'] == 's3'
        assert mapping.iloc[0]['new_name'] == 's1'
    
    def test_no_manual_grouping_workflow(self, service, sample_dataframe, simple_experiment):
        """Test workflow when user doesn't manually regroup (samples already correct)."""
        # Selections match default order
        selections = {
            'WT': ['s1', 's2', 's3'],
            'KO': ['s4', 's5', 's6']
        }
        
        valid, msg = service.validate_selections(
            selections,
            simple_experiment['conditions'],
            simple_experiment['n_samples']
        )
        assert valid is True
        
        reordered_df = service.reorder_dataframe(
            sample_dataframe,
            selections,
            simple_experiment['conditions']
        )
        
        # DataFrame should be essentially unchanged (same values, same order)
        for i in range(1, 7):
            col = f'intensity[s{i}]'
            assert reordered_df[col].tolist() == sample_dataframe[col].tolist()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])