"""
Unit tests for InternalStandardsVisualizationService.

Tests cover:
- Input validation and edge cases
- Core plotting functionality
- CSV export functionality
- Integration scenarios

Author: Hamed Abdi / Claude
Date: November 2025
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = SCRIPT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from lipidcruncher.core.services.internal_standards_visualization_service import (
    InternalStandardsVisualizationService
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_intsta_df():
    """Create valid internal standards DataFrame with multiple classes."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)(d7)',
            'PC(18:1_18:1)(d7)',
            'PE(18:0_20:4)(d9)',
            'PE(16:0_18:1)(d7)'
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [5000.0, 4800.0, 6000.0, 5800.0],
        'intensity[s2]': [5100.0, 4900.0, 6100.0, 5900.0],
        'intensity[s3]': [5200.0, 5000.0, 6200.0, 6000.0],
        'intensity[s4]': [5300.0, 5100.0, 6300.0, 6100.0]
    })


@pytest.fixture
def sample_names():
    """Sample names matching the fixture."""
    return ['s1', 's2', 's3', 's4']


@pytest.fixture
def single_class_df():
    """DataFrame with only one class (PC)."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)(d7)', 'PC(18:1_18:1)(d7)'],
        'ClassKey': ['PC', 'PC'],
        'intensity[s1]': [5000.0, 4800.0],
        'intensity[s2]': [5100.0, 4900.0]
    })


@pytest.fixture
def single_standard_df():
    """DataFrame with only one standard per class."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0)(d7)', 'PE(18:0)(d9)'],
        'ClassKey': ['PC', 'PE'],
        'intensity[s1]': [5000.0, 6000.0],
        'intensity[s2]': [5100.0, 6100.0]
    })


# ============================================================================
# TEST CLASS A: INPUT VALIDATION & EDGE CASES
# ============================================================================

class TestInputValidation:
    """Tests for input validation and edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame returns empty list."""
        empty_df = pd.DataFrame()
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            empty_df, ['s1', 's2']
        )
        
        assert figs == []
        assert isinstance(figs, list)
    
    def test_none_dataframe(self):
        """Test with None DataFrame returns empty list."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            None, ['s1', 's2']
        )
        
        assert figs == []
        assert isinstance(figs, list)
    
    def test_missing_lipidmolec_column(self):
        """Test with missing LipidMolec column returns empty list."""
        df = pd.DataFrame({
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1']
        )
        
        assert figs == []
    
    def test_missing_classkey_column(self):
        """Test with missing ClassKey column returns empty list."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'intensity[s1]': [5000.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1']
        )
        
        assert figs == []
    
    def test_no_intensity_columns(self):
        """Test with no intensity columns returns empty list."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC']
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1']
        )
        
        assert figs == []
    
    def test_empty_samples_list(self):
        """Test with empty samples list returns empty list."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, []
        )
        
        assert figs == []
    
    def test_nonexistent_samples(self):
        """Test with samples that don't exist in data returns empty list."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s999', 's888']
        )
        
        assert figs == []
    
    def test_partial_sample_match(self, valid_intsta_df):
        """Test with some samples existing, some not - should succeed with available."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, ['s1', 's2', 's999']  # s999 doesn't exist
        )
        
        # Should succeed with available samples only
        assert len(figs) == 2  # PC and PE classes
        assert all(fig is not None for fig in figs)


# ============================================================================
# TEST CLASS B: CORE FUNCTIONALITY
# ============================================================================

class TestCoreFunctionality:
    """Tests for core plotting functionality."""
    
    def test_single_class_single_standard(self):
        """Test with one class and one standard generates one figure."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [5100.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1', 's2']
        )
        
        assert len(figs) == 1
        assert 'PC' in figs[0].layout.title.text
    
    def test_single_class_multiple_standards(self, single_class_df):
        """Test with one class, multiple standards (should stack)."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            single_class_df, ['s1', 's2']
        )
        
        assert len(figs) == 1
        assert figs[0].layout.barmode == 'stack'
        assert 'PC' in figs[0].layout.title.text
    
    def test_multiple_classes(self, valid_intsta_df, sample_names):
        """Test with multiple classes generates multiple figures."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, sample_names
        )
        
        assert len(figs) == 2  # PC and PE
        
        # Check titles contain class names
        titles = [fig.layout.title.text for fig in figs]
        assert any('PC' in title for title in titles)
        assert any('PE' in title for title in titles)
    
    def test_sample_order_preserved(self, valid_intsta_df):
        """Test that sample order is preserved in plots."""
        samples_custom_order = ['s3', 's1', 's4', 's2']
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, samples_custom_order
        )
        
        assert len(figs) == 2
        # Check first figure's x-axis order matches input (returns tuple)
        assert list(figs[0].layout.xaxis.categoryarray) == samples_custom_order
    
    def test_plot_has_correct_attributes(self, valid_intsta_df, sample_names):
        """Test that plots have correct layout attributes."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, sample_names
        )
        
        assert len(figs) > 0
        fig = figs[0]
        
        # Check layout attributes
        assert fig.layout.xaxis.title.text == 'Samples'
        assert fig.layout.yaxis.title.text == 'Raw Intensity'
        assert fig.layout.legend.title.text == 'Internal Standard'
        assert fig.layout.height == 500
        assert fig.layout.barmode == 'stack'
    
    def test_text_color_is_black(self, valid_intsta_df, sample_names):
        """Test that all text is black (not white which would be invisible)."""
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, sample_names
        )
        
        assert len(figs) > 0
        fig = figs[0]
        
        # Check various text colors are black
        assert fig.layout.font.color == 'black'
        assert fig.layout.xaxis.tickfont.color == 'black'
        assert fig.layout.yaxis.tickfont.color == 'black'
        assert fig.layout.title.font.color == 'black'


# ============================================================================
# TEST CLASS C: EDGE CASES WITH DATA
# ============================================================================

class TestDataEdgeCases:
    """Tests for edge cases in data values."""
    
    def test_nan_values_in_intensities(self):
        """Test with NaN values in intensity columns."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [np.nan],
            'intensity[s3]': [5200.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1', 's2', 's3']
        )
        
        # Should handle NaN gracefully and still generate plot
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_zero_intensities(self):
        """Test with zero intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [0.0],
            'intensity[s2]': [0.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1', 's2']
        )
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_very_large_intensities(self):
        """Test with very large intensity values."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e10],
            'intensity[s2]': [1.5e10]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1', 's2']
        )
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_single_sample(self):
        """Test with only one sample."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1']
        )
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_special_characters_in_lipid_names(self):
        """Test with special characters in lipid names."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(O-18:1/20:4)(d7)', 'LPC(P-16:0)(d7)'],
            'ClassKey': ['PC', 'LPC'],
            'intensity[s1]': [5000.0, 3000.0],
            'intensity[s2]': [5100.0, 3100.0]
        })
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s1', 's2']
        )
        
        assert len(figs) == 2  # PC and LPC
        assert all(fig is not None for fig in figs)


# ============================================================================
# TEST CLASS D: CSV EXPORT FUNCTIONALITY
# ============================================================================

class TestCSVExport:
    """Tests for CSV export functionality."""
    
    def test_prepare_csv_data_basic(self, valid_intsta_df, sample_names):
        """Test basic CSV data preparation."""
        csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            valid_intsta_df,
            'PC',
            sample_names
        )
        
        # Check structure
        assert not csv_df.empty
        assert 'LipidMolec' in csv_df.columns
        assert 'ClassKey' in csv_df.columns
        assert 'Sample' in csv_df.columns
        assert 'Intensity' in csv_df.columns
        
        # Check filtering worked
        assert all(csv_df['ClassKey'] == 'PC')
        
        # Check we have right number of rows (2 PC standards × 4 samples = 8 rows)
        assert len(csv_df) == 8
    
    def test_csv_sample_filtering(self, valid_intsta_df):
        """Test that CSV only includes requested samples."""
        csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            valid_intsta_df,
            'PC',
            ['s1', 's3']  # Only s1 and s3
        )
        
        unique_samples = csv_df['Sample'].unique()
        assert set(unique_samples) == {'s1', 's3'}
        
        # 2 PC standards × 2 samples = 4 rows
        assert len(csv_df) == 4
    
    def test_csv_class_filtering(self, valid_intsta_df, sample_names):
        """Test that CSV only includes requested class."""
        csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            valid_intsta_df,
            'PE',
            sample_names
        )
        
        # Check only PE class
        assert all(csv_df['ClassKey'] == 'PE')
        assert 'PC' not in csv_df['ClassKey'].values
        
        # 2 PE standards × 4 samples = 8 rows
        assert len(csv_df) == 8
    
    def test_csv_column_order(self, valid_intsta_df, sample_names):
        """Test that CSV columns are in correct order."""
        csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            valid_intsta_df,
            'PC',
            sample_names
        )
        
        expected_columns = ['LipidMolec', 'ClassKey', 'Sample', 'Intensity']
        assert list(csv_df.columns) == expected_columns
    
    def test_csv_with_single_standard(self, single_standard_df):
        """Test CSV export with single standard per class."""
        csv_df = InternalStandardsVisualizationService.prepare_csv_data(
            single_standard_df,
            'PC',
            ['s1', 's2']
        )
        
        # 1 PC standard × 2 samples = 2 rows
        assert len(csv_df) == 2
        assert len(csv_df['LipidMolec'].unique()) == 1


# ============================================================================
# TEST CLASS E: INTEGRATION SCENARIOS
# ============================================================================

class TestIntegrationScenarios:
    """Tests for realistic integration scenarios."""
    
    def test_typical_workflow(self, valid_intsta_df, sample_names):
        """Test typical workflow: create plots + export CSV."""
        # Step 1: Create plots
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, sample_names
        )
        
        assert len(figs) == 2  # PC and PE
        
        # Step 2: Export CSV for each class
        for fig in figs:
            # Extract class name from title
            title = fig.layout.title.text
            class_name = title.split(' for ')[-1]
            
            # Prepare CSV
            csv_df = InternalStandardsVisualizationService.prepare_csv_data(
                valid_intsta_df,
                class_name,
                sample_names
            )
            
            assert not csv_df.empty
            assert all(csv_df['ClassKey'] == class_name)
    
    def test_subset_of_samples(self, valid_intsta_df):
        """Test with subset of available samples (common use case)."""
        # User selects only 2 out of 4 samples
        selected_samples = ['s1', 's3']
        
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            valid_intsta_df, selected_samples
        )
        
        assert len(figs) == 2
        
        # Verify plots only show selected samples (categoryarray returns tuple)
        for fig in figs:
            assert list(fig.layout.xaxis.categoryarray) == selected_samples
    
    def test_empty_class_filtered_out(self):
        """Test that empty classes (after filtering) don't create plots."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(d7)', 'PE(18:0)(d9)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [5000.0, 6000.0],
            'intensity[s2]': [5100.0, 6100.0]
        })
        
        # Request samples that don't exist
        figs = InternalStandardsVisualizationService.create_consistency_plots(
            df, ['s999']
        )
        
        # Should return empty list (no valid samples)
        assert figs == []


# ============================================================================
# SUMMARY
# ============================================================================

"""
Test Summary:
=============

Test Class A: Input Validation (8 tests)
- Empty/None DataFrame
- Missing required columns
- No intensity columns
- Empty/nonexistent samples
- Partial sample match

Test Class B: Core Functionality (6 tests)
- Single/multiple classes
- Single/multiple standards
- Sample order preservation
- Plot attributes
- Text color (black)

Test Class C: Data Edge Cases (5 tests)
- NaN values
- Zero intensities
- Large intensities
- Single sample
- Special characters

Test Class D: CSV Export (5 tests)
- Basic CSV structure
- Sample filtering
- Class filtering
- Column order
- Single standard

Test Class E: Integration (3 tests)
- Typical workflow
- Subset of samples
- Empty class filtering

Total: 27 tests
Coverage: All major code paths
Focus: Edge cases that matter + core functionality
"""