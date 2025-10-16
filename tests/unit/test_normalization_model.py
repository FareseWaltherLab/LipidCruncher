"""
Unit tests for NormalizationConfig model.
"""
import pytest
from pydantic import ValidationError
from src.lipidcruncher.core.models.normalization import NormalizationConfig


class TestNormalizationConfig:
    """Test suite for NormalizationConfig model."""
    
    def test_valid_internal_standard_config(self):
        """Test creating valid internal standard normalization config."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)', 'PE': 'PE(18:0_20:4)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0, 'PE(18:0_20:4)(d7)': 1.5}
        )
        
        assert config.method == 'internal_standard'
        assert config.selected_classes == ['PC', 'PE']
        assert config.requires_internal_standards() == True
        assert config.requires_protein_data() == False
    
    def test_valid_protein_config(self):
        """Test creating valid protein normalization config."""
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.5, 's2': 3.0, 's3': 2.8}
        )
        
        assert config.method == 'protein'
        assert config.requires_internal_standards() == False
        assert config.requires_protein_data() == True
    
    def test_valid_both_config(self):
        """Test creating valid combined normalization config."""
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0},
            protein_concentrations={'s1': 2.5, 's2': 3.0}
        )
        
        assert config.method == 'both'
        assert config.requires_internal_standards() == True
        assert config.requires_protein_data() == True
    
    def test_valid_none_config(self):
        """Test creating valid no-normalization config."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC', 'PE', 'TAG']
        )
        
        assert config.method == 'none'
        assert config.requires_internal_standards() == False
        assert config.requires_protein_data() == False
    
    def test_invalid_method_raises_error(self):
        """Test that invalid normalization method raises error."""
        with pytest.raises(ValidationError, match="Method must be one of"):
            NormalizationConfig(
                method='invalid_method',
                selected_classes=['PC']
            )
    
    def test_negative_intsta_concentration_raises_error(self):
        """Test that negative internal standard concentrations are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC'],
                internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
                intsta_concentrations={'PC(16:0_18:1)(d7)': -1.0}
            )
    
    def test_zero_intsta_concentration_raises_error(self):
        """Test that zero internal standard concentrations are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='internal_standard',
                selected_classes=['PC'],
                internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
                intsta_concentrations={'PC(16:0_18:1)(d7)': 0.0}
            )
    
    def test_negative_protein_concentration_raises_error(self):
        """Test that negative protein concentrations are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='protein',
                selected_classes=['PC'],
                protein_concentrations={'s1': -2.5}
            )
    
    def test_zero_protein_concentration_raises_error(self):
        """Test that zero protein concentrations are rejected."""
        with pytest.raises(ValidationError, match="must be positive"):
            NormalizationConfig(
                method='protein',
                selected_classes=['PC'],
                protein_concentrations={'s1': 0.0}
            )
    
    def test_preserve_column_prefix_default(self):
        """Test that preserve_column_prefix defaults to False."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC']
        )
        
        assert config.preserve_column_prefix == False
    
    def test_preserve_column_prefix_true(self):
        """Test setting preserve_column_prefix to True."""
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.5},
            preserve_column_prefix=True
        )
        
        assert config.preserve_column_prefix == True
    
    def test_validate_complete_internal_standard_success(self):
        """Test validate_complete passes for valid internal standard config."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        # Should not raise any exception
        config.validate_complete()
    
    def test_validate_complete_missing_internal_standards(self):
        """Test validate_complete fails when internal_standards missing."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC']
        )
        
        with pytest.raises(ValueError, match="internal_standards required"):
            config.validate_complete()
    
    def test_validate_complete_missing_concentrations(self):
        """Test validate_complete fails when concentrations missing."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'}
        )
        
        with pytest.raises(ValueError, match="intsta_concentrations required"):
            config.validate_complete()
    
    def test_validate_complete_missing_concentration_for_standard(self):
        """Test validate_complete fails when a standard lacks concentration."""
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d7)'
            },
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}  # Missing PE standard
        )
        
        with pytest.raises(ValueError, match="Missing concentrations for internal standards"):
            config.validate_complete()
    
    def test_validate_complete_missing_protein_concentrations(self):
        """Test validate_complete fails when protein_concentrations missing."""
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC']
        )
        
        with pytest.raises(ValueError, match="protein_concentrations required"):
            config.validate_complete()
    
    def test_validate_complete_both_method_missing_internal_standards(self):
        """Test validate_complete fails for 'both' method missing internal standards."""
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.5}
        )
        
        with pytest.raises(ValueError, match="internal_standards required"):
            config.validate_complete()
    
    def test_validate_complete_both_method_missing_protein(self):
        """Test validate_complete fails for 'both' method missing protein data."""
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 1.0}
        )
        
        with pytest.raises(ValueError, match="protein_concentrations required"):
            config.validate_complete()
    
    def test_validate_complete_none_method_always_passes(self):
        """Test validate_complete always passes for 'none' method."""
        config = NormalizationConfig(
            method='none',
            selected_classes=['PC']
        )
        
        # Should not raise any exception
        config.validate_complete()
