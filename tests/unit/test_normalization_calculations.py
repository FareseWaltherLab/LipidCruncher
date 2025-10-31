"""
Unit tests for NormalizationService calculation correctness.

This test suite verifies that normalization formulas are implemented correctly
by using known input values and comparing results against hand-calculated expected
outputs. These tests are essential for ensuring scientific accuracy.

Test Coverage:
- Internal standards normalization: (Intensity_lipid / Intensity_standard) × Concentration_standard
- Protein normalization: Intensity_lipid / Protein_concentration  
- Combined normalization: Sequential application of both methods
- Numerical precision with extreme values

Why These Tests Matter:
Unlike edge case tests that verify error handling and boundary conditions,
these tests verify the core mathematical correctness of the normalization
algorithms. A bug in these formulas would produce scientifically invalid
results that could pass all edge case tests but lead to incorrect research
conclusions.

Test Strategy:
Each test uses simple, hand-verifiable values where the expected output can be
calculated manually. Tests verify results to high precision (1e-9 relative
tolerance) to catch any numerical errors or formula implementation mistakes.
"""
import pytest
import pandas as pd
import numpy as np
from src.lipidcruncher.core.services.normalization_service import NormalizationService
from src.lipidcruncher.core.models.normalization import NormalizationConfig
from src.lipidcruncher.core.models.experiment import ExperimentConfig


# ==================== Fixtures ====================

@pytest.fixture
def service():
    """NormalizationService instance."""
    return NormalizationService()


@pytest.fixture
def experiment_config():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[2, 2]
    )


# ==================== Test Class: Internal Standards Calculation Correctness ====================

class TestInternalStandardsCalculationCorrectness:
    """
    Tests that verify the internal standards normalization formula:
    Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard
    """
    
    def test_simple_internal_standards_calculation(self, service, experiment_config):
        """
        Test basic calculation with known values.
        
        Given:
        - PC lipid with intensity [1000, 2000, 3000, 4000]
        - PC standard with intensity [100, 100, 100, 100]
        - Standard concentration = 10.0
        
        Expected:
        - Result = (1000/100)*10, (2000/100)*10, (3000/100)*10, (4000/100)*10
        - Result = [100, 200, 300, 400]
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [2000.0],
            'intensity[s3]': [3000.0],
            'intensity[s4]': [4000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert - Check exact calculation
        expected_s1 = (1000.0 / 100.0) * 10.0  # = 100.0
        expected_s2 = (2000.0 / 100.0) * 10.0  # = 200.0
        expected_s3 = (3000.0 / 100.0) * 10.0  # = 300.0
        expected_s4 = (4000.0 / 100.0) * 10.0  # = 400.0
        
        assert result['concentration[s1]'].iloc[0] == pytest.approx(expected_s1, rel=1e-9)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(expected_s2, rel=1e-9)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(expected_s3, rel=1e-9)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(expected_s4, rel=1e-9)
    
    def test_internal_standards_with_varying_standard_intensities(
        self, service, experiment_config
    ):
        """
        Test calculation when standard intensities vary across samples.
        
        Given:
        - PE lipid with intensity [5000, 6000, 7000, 8000]
        - PE standard with intensity [200, 300, 400, 500]
        - Standard concentration = 20.0
        
        Expected:
        - s1: (5000/200)*20 = 500
        - s2: (6000/300)*20 = 400
        - s3: (7000/400)*20 = 350
        - s4: (8000/500)*20 = 320
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)'],
            'ClassKey': ['PE'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [6000.0],
            'intensity[s3]': [7000.0],
            'intensity[s4]': [8000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)(d9)'],
            'ClassKey': ['PE'],
            'intensity[s1]': [200.0],
            'intensity[s2]': [300.0],
            'intensity[s3]': [400.0],
            'intensity[s4]': [500.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PE'],
            internal_standards={'PE': 'PE(18:0_20:4)(d9)'},
            intsta_concentrations={'PE(18:0_20:4)(d9)': 20.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert
        assert result['concentration[s1]'].iloc[0] == pytest.approx(500.0, rel=1e-9)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(400.0, rel=1e-9)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(350.0, rel=1e-9)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(320.0, rel=1e-9)
    
    def test_internal_standards_multiple_lipids_same_class(
        self, service, experiment_config
    ):
        """
        Test that all lipids in same class use the same standard correctly.
        
        Given:
        - Two PC lipids with different intensities
        - One PC standard
        
        Expected:
        - Both lipids normalized using same standard
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(18:0_20:4)'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1500.0, 2500.0],
            'intensity[s3]': [2000.0, 3000.0],
            'intensity[s4]': [2500.0, 3500.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [500.0],
            'intensity[s2]': [500.0],
            'intensity[s3]': [500.0],
            'intensity[s4]': [500.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 5.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert - First lipid
        # s1: (1000/500)*5 = 10
        assert result[result['LipidMolec'] == 'PC(16:0_18:1)']['concentration[s1]'].iloc[0] == pytest.approx(10.0, rel=1e-9)
        
        # Assert - Second lipid
        # s1: (2000/500)*5 = 20
        assert result[result['LipidMolec'] == 'PC(18:0_20:4)']['concentration[s1]'].iloc[0] == pytest.approx(20.0, rel=1e-9)
    
    def test_internal_standards_different_classes_different_standards(
        self, service, experiment_config
    ):
        """
        Test that different classes use their own standards correctly.
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [1000.0, 2000.0],
            'intensity[s2]': [1000.0, 2000.0],
            'intensity[s3]': [1000.0, 2000.0],
            'intensity[s4]': [1000.0, 2000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)', 'PE(18:0_20:4)(d9)'],
            'ClassKey': ['PC', 'PE'],
            'intensity[s1]': [100.0, 200.0],  # Different standard intensities
            'intensity[s2]': [100.0, 200.0],
            'intensity[s3]': [100.0, 200.0],
            'intensity[s4]': [100.0, 200.0]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC', 'PE'],
            internal_standards={
                'PC': 'PC(16:0_18:1)(d7)',
                'PE': 'PE(18:0_20:4)(d9)'
            },
            intsta_concentrations={
                'PC(16:0_18:1)(d7)': 10.0,
                'PE(18:0_20:4)(d9)': 15.0  # Different concentrations
            }
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert - PC: (1000/100)*10 = 100
        pc_result = result[result['ClassKey'] == 'PC']
        assert pc_result['concentration[s1]'].iloc[0] == pytest.approx(100.0, rel=1e-9)
        
        # Assert - PE: (2000/200)*15 = 150
        pe_result = result[result['ClassKey'] == 'PE']
        assert pe_result['concentration[s1]'].iloc[0] == pytest.approx(150.0, rel=1e-9)


# ==================== Test Class: Protein Normalization Calculation Correctness ====================

class TestProteinNormalizationCalculationCorrectness:
    """
    Tests that verify the protein normalization formula:
    Concentration = Intensity_lipid / Protein_concentration
    """
    
    def test_simple_protein_normalization_calculation(self, service, experiment_config):
        """
        Test basic protein normalization with known values.
        
        Given:
        - Lipid with intensity [1000, 2000, 3000, 4000]
        - Protein concentrations [2.0, 4.0, 5.0, 10.0]
        
        Expected:
        - s1: 1000/2.0 = 500
        - s2: 2000/4.0 = 500
        - s3: 3000/5.0 = 600
        - s4: 4000/10.0 = 400
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [2000.0],
            'intensity[s3]': [3000.0],
            'intensity[s4]': [4000.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 4.0, 5.0, 10.0]
        })
        
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC'],
            protein_concentrations={'s1': 2.0, 's2': 4.0, 's3': 5.0, 's4': 10.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            protein_df=protein_data
        )
        
        # Assert
        assert result['concentration[s1]'].iloc[0] == pytest.approx(500.0, rel=1e-9)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(500.0, rel=1e-9)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(600.0, rel=1e-9)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(400.0, rel=1e-9)
    
    def test_protein_normalization_with_uniform_concentration(self, service, experiment_config):
        """
        Test protein normalization when all samples have same protein concentration.
        
        Given:
        - Lipid with intensity [100, 200, 300, 400]
        - All protein concentrations = 2.0
        
        Expected:
        - All values divided by 2.0: [50, 100, 150, 200]
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['TAG(16:0_18:1_18:2)'],
            'ClassKey': ['TAG'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [200.0],
            'intensity[s3]': [300.0],
            'intensity[s4]': [400.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 2.0, 2.0, 2.0]
        })
        
        config = NormalizationConfig(
            method='protein',
            selected_classes=['TAG'],
            protein_concentrations={'s1': 2.0, 's2': 2.0, 's3': 2.0, 's4': 2.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            protein_df=protein_data
        )
        
        # Assert
        assert result['concentration[s1]'].iloc[0] == pytest.approx(50.0, rel=1e-9)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(100.0, rel=1e-9)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(150.0, rel=1e-9)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(200.0, rel=1e-9)
    
    def test_protein_normalization_multiple_lipids(self, service, experiment_config):
        """
        Test that all lipids are normalized by protein correctly.
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(18:0_20:4)', 'PE(18:0_20:4)'],
            'ClassKey': ['PC', 'PC', 'PE'],
            'intensity[s1]': [1000.0, 1500.0, 2000.0],
            'intensity[s2]': [1200.0, 1800.0, 2400.0],
            'intensity[s3]': [1400.0, 2100.0, 2800.0],
            'intensity[s4]': [1600.0, 2400.0, 3200.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 3.0, 4.0, 5.0]
        })
        
        config = NormalizationConfig(
            method='protein',
            selected_classes=['PC', 'PE'],
            protein_concentrations={'s1': 2.0, 's2': 3.0, 's3': 4.0, 's4': 5.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            protein_df=protein_data
        )
        
        # Assert - Check first lipid (PC(16:0_18:1))
        # s1: 1000/2.0 = 500, s2: 1200/3.0 = 400, s3: 1400/4.0 = 350, s4: 1600/5.0 = 320
        pc1_result = result[result['LipidMolec'] == 'PC(16:0_18:1)']
        assert pc1_result['concentration[s1]'].iloc[0] == pytest.approx(500.0, rel=1e-9)
        assert pc1_result['concentration[s2]'].iloc[0] == pytest.approx(400.0, rel=1e-9)
        assert pc1_result['concentration[s3]'].iloc[0] == pytest.approx(350.0, rel=1e-9)
        assert pc1_result['concentration[s4]'].iloc[0] == pytest.approx(320.0, rel=1e-9)


# ==================== Test Class: Both Method Calculation Correctness ====================

class TestBothMethodCalculationCorrectness:
    """
    Tests that verify the combined normalization formula:
    Concentration = (Intensity_lipid / Intensity_standard) × Concentration_standard / Protein_concentration
    """
    
    def test_both_method_sequential_calculation(self, service, experiment_config):
        """
        Test that 'both' method correctly applies standards then protein.
        
        Given:
        - Lipid intensity = 1000
        - Standard intensity = 100
        - Standard concentration = 10
        - Protein concentration = 2
        
        Expected:
        Step 1 (standards): (1000/100) * 10 = 100
        Step 2 (protein): 100 / 2 = 50
        Final result: 50
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [1000.0],
            'intensity[s4]': [1000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            'intensity[s2]': [100.0],
            'intensity[s3]': [100.0],
            'intensity[s4]': [100.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 2.0, 2.0, 2.0]
        })
        
        config = NormalizationConfig(
            method='both',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0},
            protein_concentrations={'s1': 2.0, 's2': 2.0, 's3': 2.0, 's4': 2.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data,
            protein_df=protein_data
        )
        
        # Assert - All samples should be 50.0
        expected = 50.0  # (1000/100)*10 / 2.0
        assert result['concentration[s1]'].iloc[0] == pytest.approx(expected, rel=1e-9)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(expected, rel=1e-9)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(expected, rel=1e-9)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(expected, rel=1e-9)
    
    def test_both_method_with_varying_values(self, service, experiment_config):
        """
        Test 'both' method with all parameters varying.
        
        This ensures the order of operations is correct.
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)'],
            'ClassKey': ['PE'],
            'intensity[s1]': [5000.0],
            'intensity[s2]': [6000.0],
            'intensity[s3]': [7000.0],
            'intensity[s4]': [8000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PE(18:0_20:4)(d9)'],
            'ClassKey': ['PE'],
            'intensity[s1]': [200.0],
            'intensity[s2]': [250.0],
            'intensity[s3]': [300.0],
            'intensity[s4]': [400.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [2.0, 3.0, 4.0, 5.0]
        })
        
        config = NormalizationConfig(
            method='both',
            selected_classes=['PE'],
            internal_standards={'PE': 'PE(18:0_20:4)(d9)'},
            intsta_concentrations={'PE(18:0_20:4)(d9)': 20.0},
            protein_concentrations={'s1': 2.0, 's2': 3.0, 's3': 4.0, 's4': 5.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data,
            protein_df=protein_data
        )
        
        # Assert - Calculate expected values manually
        # s1: ((5000/200)*20) / 2.0 = 500/2.0 = 250
        # s2: ((6000/250)*20) / 3.0 = 480/3.0 = 160
        # s3: ((7000/300)*20) / 4.0 = 466.667/4.0 = 116.667
        # s4: ((8000/400)*20) / 5.0 = 400/5.0 = 80
        
        assert result['concentration[s1]'].iloc[0] == pytest.approx(250.0, rel=1e-6)
        assert result['concentration[s2]'].iloc[0] == pytest.approx(160.0, rel=1e-6)
        assert result['concentration[s3]'].iloc[0] == pytest.approx(116.667, rel=1e-3)
        assert result['concentration[s4]'].iloc[0] == pytest.approx(80.0, rel=1e-6)
    
    def test_both_method_matches_manual_chaining(self, service, experiment_config):
        """
        Test that 'both' method produces same result as manually chaining the two methods.
        """
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['TAG(16:0_18:1_18:2)'],
            'ClassKey': ['TAG'],
            'intensity[s1]': [10000.0],
            'intensity[s2]': [12000.0],
            'intensity[s3]': [14000.0],
            'intensity[s4]': [16000.0]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['TAG(16:0_18:1_18:2)(d5)'],
            'ClassKey': ['TAG'],
            'intensity[s1]': [1000.0],
            'intensity[s2]': [1000.0],
            'intensity[s3]': [1000.0],
            'intensity[s4]': [1000.0]
        })
        
        protein_data = pd.DataFrame({
            'Sample': ['s1', 's2', 's3', 's4'],
            'Concentration': [5.0, 6.0, 7.0, 8.0]
        })
        
        config_both = NormalizationConfig(
            method='both',
            selected_classes=['TAG'],
            internal_standards={'TAG': 'TAG(16:0_18:1_18:2)(d5)'},
            intsta_concentrations={'TAG(16:0_18:1_18:2)(d5)': 25.0},
            protein_concentrations={'s1': 5.0, 's2': 6.0, 's3': 7.0, 's4': 8.0}
        )
        
        # Act - Using 'both' method
        result_both = service.normalize(
            lipid_data,
            config_both,
            experiment_config,
            intsta_df=standard_data,
            protein_df=protein_data
        )
        
        # Act - Manual chaining
        config_standards = NormalizationConfig(
            method='internal_standard',
            selected_classes=['TAG'],
            internal_standards={'TAG': 'TAG(16:0_18:1_18:2)(d5)'},
            intsta_concentrations={'TAG(16:0_18:1_18:2)(d5)': 25.0}
        )
        temp = service.normalize(
            lipid_data,
            config_standards,
            experiment_config,
            intsta_df=standard_data
        )
        
        config_protein = NormalizationConfig(
            method='protein',
            selected_classes=['TAG'],
            protein_concentrations={'s1': 5.0, 's2': 6.0, 's3': 7.0, 's4': 8.0}
        )
        result_manual = service.normalize(
            temp,
            config_protein,
            experiment_config,
            protein_df=protein_data
        )
        
        # Assert - Results should match exactly
        pd.testing.assert_frame_equal(
            result_both[['concentration[s1]', 'concentration[s2]', 'concentration[s3]', 'concentration[s4]']],
            result_manual[['concentration[s1]', 'concentration[s2]', 'concentration[s3]', 'concentration[s4]']],
            rtol=1e-9
        )


# ==================== Test Class: Numerical Precision ====================

class TestNumericalPrecision:
    """
    Tests to ensure calculations maintain numerical precision.
    """
    
    def test_calculation_precision_with_large_numbers(self, service, experiment_config):
        """Test that large numbers don't lose precision."""
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e15],  # Very large number
            'intensity[s2]': [1e15],
            'intensity[s3]': [1e15],
            'intensity[s4]': [1e15]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e10],
            'intensity[s2]': [1e10],
            'intensity[s3]': [1e10],
            'intensity[s4]': [1e10]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert - Result should be (1e15 / 1e10) * 10 = 1e6
        expected = 1e6
        assert result['concentration[s1]'].iloc[0] == pytest.approx(expected, rel=1e-6)
    
    def test_calculation_precision_with_small_numbers(self, service, experiment_config):
        """Test that small numbers don't lose precision."""
        # Arrange
        lipid_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-5],  # Very small number
            'intensity[s2]': [1e-5],
            'intensity[s3]': [1e-5],
            'intensity[s4]': [1e-5]
        })
        
        standard_data = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [1e-3],
            'intensity[s2]': [1e-3],
            'intensity[s3]': [1e-3],
            'intensity[s4]': [1e-3]
        })
        
        config = NormalizationConfig(
            method='internal_standard',
            selected_classes=['PC'],
            internal_standards={'PC': 'PC(16:0_18:1)(d7)'},
            intsta_concentrations={'PC(16:0_18:1)(d7)': 10.0}
        )
        
        # Act
        result = service.normalize(
            lipid_data,
            config,
            experiment_config,
            intsta_df=standard_data
        )
        
        # Assert - Result should be (1e-5 / 1e-3) * 10 = 0.1
        expected = 0.1
        assert result['concentration[s1]'].iloc[0] == pytest.approx(expected, rel=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])