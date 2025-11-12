"""
Integration tests for edge cases using synthetic data.

Tests error handling, validation, and edge cases that are difficult
to test with real data. Uses controlled synthetic datasets.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lipidcruncher.core.services.internal_standards_visualization_service import InternalStandardsVisualizationService
from lipidcruncher.core.services.normalization_service import NormalizationService
from lipidcruncher.core.models.experiment import ExperimentConfig
from lipidcruncher.core.models.normalization import NormalizationConfig


# ============================================================================
# SYNTHETIC DATA FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_lipid_data():
    """Create synthetic lipid data with known properties."""
    lipids = []
    classes = ['PC', 'PE', 'TG', 'DG']
    
    for cls in classes:
        for i in range(5):
            lipids.append({
                'LipidMolec': f'{cls}(16:{i}_18:{i})',
                'ClassKey': cls,
                'CalcMass': 700.0 + i * 10,
                'BaseRt': 10.0 + i,
                'intensity[s1]': 10000 * (i + 1),
                'intensity[s2]': 11000 * (i + 1),
                'intensity[s3]': 9000 * (i + 1),
                'intensity[s4]': 12000 * (i + 1),
                'intensity[s5]': 13000 * (i + 1),
                'intensity[s6]': 8000 * (i + 1),
            })
    
    return pd.DataFrame(lipids)


@pytest.fixture
def synthetic_standards():
    """List of standard lipid names that exist in synthetic_lipid_data."""
    return ['PC(16:0_18:0)', 'PE(16:1_18:1)', 'TG(16:2_18:2)', 'DG(16:3_18:3)']


@pytest.fixture
def synthetic_protein_concentrations():
    """Synthetic protein concentrations for 6 samples."""
    return {
        's1': 2.5,
        's2': 2.8,
        's3': 2.3,
        's4': 2.6,
        's5': 2.9,
        's6': 2.4
    }


@pytest.fixture
def experiment_config_synthetic():
    """Experiment config for synthetic data (6 samples, 2 conditions)."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Condition1', 'Condition2'],
        number_of_samples_list=[3, 3]
    )


# ============================================================================
# TEST CLASS 1: ZERO AND NAN VALUES
# ============================================================================

class TestZeroAndNaNHandling:
    """Test handling of zero and NaN values in standards and data."""
    
    def test_standards_with_zero_intensities(self):
        """Test handling when standards have zero intensity values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(16:0_18:1)', 'TG(16:0_18:1)'],
            'ClassKey': ['PC', 'PE', 'TG'],
            'intensity[s1]': [0.0, 10000, 20000],
            'intensity[s2]': [0.0, 11000, 21000],
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 3
        pc_fig = [f for f in figs if 'PC' in f.layout.title.text][0]
        assert pc_fig is not None
    
    def test_standards_with_nan_values(self):
        """Test handling when standards have NaN values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [np.nan],
            'intensity[s3]': [5200.0]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2', 's3'])
        
        assert len(figs) == 1
        assert figs[0] is not None


# ============================================================================
# TEST CLASS 2: MISSING AND INVALID DATA
# ============================================================================

class TestMissingAndInvalidData:
    """Test handling of missing samples and invalid data."""
    
    def test_standards_not_in_dataset(self, synthetic_lipid_data):
        """Test error handling when custom standards don't exist in dataset."""
        available_lipids = synthetic_lipid_data['LipidMolec'].tolist()
        invalid_standards = ['PC(99:9_99:9)', 'PE(88:8_88:8)', 'FAKE(1:1_2:2)']
        
        valid = []
        invalid = []
        
        for std in invalid_standards:
            if std in available_lipids:
                valid.append(std)
            else:
                invalid.append(std)
        
        assert len(valid) == 0
        assert len(invalid) == len(invalid_standards)
        print(f"Correctly identified {len(invalid)} invalid standards")
    
    def test_single_sample_data(self):
        """Test with only one sample."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [5000.0]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1'])
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_empty_class_after_filtering(self):
        """Test when all lipids filtered out."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(16:0_18:1)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [5000.0, 6000.0],
            'intensity[s2]': [5100.0, 6100.0]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s999', 's888'])
        
        assert figs == []


# ============================================================================
# TEST CLASS 3: EXTREME VALUES
# ============================================================================

class TestExtremeValues:
    """Test handling of extreme intensity values."""
    
    def test_very_large_intensities(self):
        """Test with very large intensity values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e12],
            'intensity[s2]': [1.5e12]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_very_small_intensities(self):
        """Test with very small intensity values."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-6],
            'intensity[s2]': [1.5e-6]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 1
        assert figs[0] is not None
    
    def test_mixed_scale_intensities(self):
        """Test with intensities spanning many orders of magnitude."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PC(18:0_20:4)(d9)'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': [1e2, 1e8],
            'intensity[s2]': [1.5e2, 1.5e8]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 1
        assert figs[0] is not None


# ============================================================================
# TEST CLASS 4: SAMPLE ORDERING AND FILTERING
# ============================================================================

class TestSampleOrderingAndFiltering:
    """Test sample order preservation and filtering."""
    
    def test_custom_sample_order(self, synthetic_lipid_data):
        """Test that custom sample order is preserved in plots."""
        standards_df = synthetic_lipid_data[
            synthetic_lipid_data['LipidMolec'].isin(['PC(16:0_18:0)', 'PE(16:1_18:1)'])
        ].copy()
        
        custom_order = ['s3', 's1', 's6', 's2', 's5', 's4']
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(standards_df, custom_order)
        
        assert len(figs) > 0
        assert list(figs[0].layout.xaxis.categoryarray) == custom_order
    
    def test_subset_of_samples(self, synthetic_lipid_data):
        """Test plotting with subset of available samples."""
        standards_df = synthetic_lipid_data[
            synthetic_lipid_data['LipidMolec'].isin(['PC(16:0_18:0)', 'PE(16:1_18:1)'])
        ].copy()
        
        selected_samples = ['s1', 's3', 's5']
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(standards_df, selected_samples)
        
        assert len(figs) > 0
        assert list(figs[0].layout.xaxis.categoryarray) == selected_samples


# ============================================================================
# TEST CLASS 5: COEFFICIENT OF VARIATION
# ============================================================================

class TestCoefficientOfVariation:
    """Test standards with high coefficient of variation."""
    
    def test_high_cv_detection(self):
        """Test detection of standards with high CV%."""
        data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(16:0_18:1)(d9)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [5000.0, 1000.0],
            'intensity[s2]': [15000.0, 1050.0],
            'intensity[s3]': [10000.0, 950.0],
        })
        
        for lipid in data['LipidMolec']:
            row = data[data['LipidMolec'] == lipid]
            intensities = [row[f'intensity[s{i}]'].values[0] for i in range(1, 4)]
            
            mean = np.mean(intensities)
            std = np.std(intensities)
            cv = (std / mean) * 100
            
            if lipid == 'PC(16:0_18:1)(d7)':
                assert cv > 20
            else:
                assert cv < 20
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2', 's3'])
        
        assert len(figs) == 2


# ============================================================================
# TEST CLASS 6: SPECIAL CHARACTERS
# ============================================================================

class TestSpecialCharactersAndFormatting:
    """Test lipid names with special characters."""
    
    def test_special_characters_in_lipid_names(self):
        """Test with special characters in lipid names."""
        data = pd.DataFrame({
            'LipidMolec': [
                'PC(O-18:1/20:4)(d7)',
                'LPC(P-16:0)(d7)',
                'Cer(d18:1/24:1)(d7)'
            ],
            'ClassKey': ['PC', 'LPC', 'Cer'],
            'intensity[s1]': [5000.0, 3000.0, 4000.0],
            'intensity[s2]': [5100.0, 3100.0, 4100.0]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 3
        assert all(fig is not None for fig in figs)
    
    def test_long_lipid_names(self):
        """Test with very long lipid names."""
        data = pd.DataFrame({
            'LipidMolec': [
                'CardiolipinWithVeryLongNameForTesting(18:1_18:1_18:2_20:4)(d7)'
            ],
            'ClassKey': ['CL'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [5100.0]
        })
        
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(data, ['s1', 's2'])
        
        assert len(figs) == 1
        assert figs[0] is not None


# ============================================================================
# TEST CLASS 7: CSV EXPORT EDGE CASES
# ============================================================================

class TestCSVExportEdgeCases:
    """Test CSV export with edge cases."""
    
    def test_csv_with_many_samples(self, synthetic_lipid_data):
        """Test CSV export with many samples."""
        standards_df = synthetic_lipid_data[
            synthetic_lipid_data['ClassKey'] == 'PC'
        ].copy()
        
        samples = [f's{i}' for i in range(1, 7)]
        
        viz_service = InternalStandardsVisualizationService()
        csv_df = viz_service.prepare_csv_data(standards_df, 'PC', samples)
        
        expected_rows = 5 * 6
        assert len(csv_df) == expected_rows
        assert list(csv_df.columns) == ['LipidMolec', 'ClassKey', 'Sample', 'Intensity']
    
    def test_csv_with_single_sample(self, synthetic_lipid_data):
        """Test CSV export with single sample."""
        standards_df = synthetic_lipid_data[
            synthetic_lipid_data['ClassKey'] == 'PC'
        ].copy()
        
        viz_service = InternalStandardsVisualizationService()
        csv_df = viz_service.prepare_csv_data(standards_df, 'PC', ['s1'])
        
        assert len(csv_df) == 5
        assert all(csv_df['Sample'] == 's1')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])