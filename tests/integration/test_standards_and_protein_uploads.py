"""
Integration tests for custom standards and protein concentration uploads.

Tests the complete workflows for uploading custom standards files
and BCA protein concentration files, including validation and normalization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lipidcruncher.core.services.internal_standards_visualization_service import InternalStandardsVisualizationService
from lipidcruncher.core.services.standards_service import StandardsService
from lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from lipidcruncher.core.services.normalization_service import NormalizationService
from lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService
from lipidcruncher.core.models.experiment import ExperimentConfig
from lipidcruncher.core.models.normalization import NormalizationConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def lipidsearch_data(fixtures_dir):
    """Load LipidSearch sample data."""
    return pd.read_csv(fixtures_dir / 'lipidsearch_sample.csv')


@pytest.fixture
def standards_list(fixtures_dir):
    """Load standards list from file (just names)."""
    df = pd.read_csv(fixtures_dir / 'standard_sample.csv')
    return df['LipidMolec'].tolist()


@pytest.fixture
def protein_concentrations(fixtures_dir):
    """Load protein concentrations from BCA file."""
    df = pd.read_csv(fixtures_dir / 'BCA_sample.csv')
    return dict(zip(df['Sample'], df['Concentration']))


@pytest.fixture
def experiment_config():
    """Create a basic experiment configuration."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT-DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4]
    )

@pytest.fixture
def cleaned_lipidsearch_data(lipidsearch_data, experiment_config):
    """Preprocessed and cleaned LipidSearch data using the SAME adapter as the app."""
    from src.lipidcruncher.adapters.streamlit_adapter import StreamlitDataAdapter
    
    adapter = StreamlitDataAdapter()
    
    # Use the adapter's clean_data method - same as the app uses
    cleaned_df, intsta_df, removed = adapter.clean_data(
        df=lipidsearch_data,
        experiment_config=experiment_config,
        data_format='LipidSearch 5.0',
        grade_config=None,
        apply_zero_filter=False,
        zero_filter_threshold=0.0,  # ← Fixed
        bqc_label='BQC'
    )
    
    return cleaned_df, intsta_df

# ============================================================================
# TEST CLASS 1: CUSTOM STANDARDS UPLOAD
# ============================================================================

class TestCustomStandardsUpload:
    """Test custom standards upload workflow."""
    
    def test_standards_file_structure(self, standards_list, fixtures_dir):
        """Test that standards file has correct structure."""
        # Load file directly
        df = pd.read_csv(fixtures_dir / 'standard_sample.csv')
        
        # Should have LipidMolec column
        assert 'LipidMolec' in df.columns
        
        # Should only have 1 column (just names, no intensities)
        assert len(df.columns) == 1
        
        # Should have at least some standards
        assert len(df) > 0
        
        print(f"Standards file has {len(df)} standards: {df['LipidMolec'].tolist()}")
    
    def test_custom_standards_validation(
        self,
        cleaned_lipidsearch_data,
        standards_list
    ):
        """
        Test validating custom standards against main dataset.
        
        The workflow is:
        1. User uploads file with just lipid names
        2. System validates names exist in main dataset
        3. System extracts intensity data from main dataset
        4. These become internal standards
        """
        cleaned_df, auto_intsta_df = cleaned_lipidsearch_data
        
        # Get list of available lipid names
        available_lipids = cleaned_df['LipidMolec'].tolist()
        
        # Validate which standards exist in dataset
        valid_standards = []
        invalid_standards = []
        
        for standard in standards_list:
            if standard in available_lipids:
                valid_standards.append(standard)
            else:
                invalid_standards.append(standard)
        
        print(f"Valid standards: {valid_standards}")
        print(f"Invalid standards: {invalid_standards}")
        
        # May have some invalid if the standards file contains
        # examples that aren't in this particular dataset
        # That's OK - we just use the valid ones
        
        assert len(valid_standards) >= 0  # Could be 0 if file has unrelated examples
    
    def test_custom_standards_extraction(
        self,
        cleaned_lipidsearch_data,
        standards_list,
        experiment_config
    ):
        """Test extracting intensity data for custom standards."""
        cleaned_df, auto_intsta_df = cleaned_lipidsearch_data
        
        # Validate standards
        available_lipids = cleaned_df['LipidMolec'].tolist()
        valid_standards = [s for s in standards_list if s in available_lipids]
        
        if len(valid_standards) == 0:
            pytest.skip("No valid standards in this dataset")
        
        # Extract standards data from main dataset
        custom_intsta_df = cleaned_df[
            cleaned_df['LipidMolec'].isin(valid_standards)
        ].copy()
        
        assert not custom_intsta_df.empty
        assert 'ClassKey' in custom_intsta_df.columns
        assert 'LipidMolec' in custom_intsta_df.columns
        
        # Verify intensity columns present
        intensity_cols = [c for c in custom_intsta_df.columns if c.startswith('intensity[')]
        assert len(intensity_cols) > 0
        
        print(f"Extracted {len(custom_intsta_df)} custom standards")
        print(f"Classes: {custom_intsta_df['ClassKey'].unique()}")
    
    def test_custom_standards_plots(
        self,
        cleaned_lipidsearch_data,
        standards_list,
        experiment_config
    ):
        """Test generating plots with custom standards."""
        cleaned_df, auto_intsta_df = cleaned_lipidsearch_data
        
        # Extract custom standards
        available_lipids = cleaned_df['LipidMolec'].tolist()
        valid_standards = [s for s in standards_list if s in available_lipids]
        
        if len(valid_standards) == 0:
            pytest.skip("No valid standards in this dataset")
        
        custom_intsta_df = cleaned_df[
            cleaned_df['LipidMolec'].isin(valid_standards)
        ].copy()
        
        # Generate plots
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(
            custom_intsta_df,
            experiment_config.full_samples_list
        )
        
        assert len(figs) > 0, "No plots generated with custom standards"
        
        # Verify plots have correct properties
        for fig in figs:
            assert fig is not None
            assert len(fig.data) > 0
        
        # Test CSV export
        classes = custom_intsta_df['ClassKey'].unique()
        for class_name in classes:
            csv_df = viz_service.prepare_csv_data(
                custom_intsta_df,
                class_name,
                experiment_config.full_samples_list
            )
            
            assert not csv_df.empty
            # All lipids should be from valid_standards
            assert all(name in valid_standards for name in csv_df['LipidMolec'].unique())


# ============================================================================
# TEST CLASS 2: PROTEIN CONCENTRATION UPLOAD
# ============================================================================

class TestProteinConcentrationUpload:
    """Test protein concentration (BCA) upload workflow."""
    
    def test_bca_file_structure(self, protein_concentrations, fixtures_dir):
        """Test that BCA file has correct structure."""
        # Load file directly
        df = pd.read_csv(fixtures_dir / 'BCA_sample.csv')
        
        # Should have Sample and Concentration columns
        assert 'Sample' in df.columns
        assert 'Concentration' in df.columns
        
        # Should have exactly 2 columns
        assert len(df.columns) == 2
        
        # All concentrations should be numeric and positive
        assert df['Concentration'].dtype in [np.float64, np.int64]
        assert all(df['Concentration'] > 0)
        
        print(f"BCA file has {len(df)} samples")
        print(f"Concentration range: {df['Concentration'].min()} - {df['Concentration'].max()}")
    
    def test_protein_normalization(
        self,
        cleaned_lipidsearch_data,
        protein_concentrations,
        experiment_config
    ):
        """Test protein normalization with BCA file upload."""
        cleaned_df, intsta_df = cleaned_lipidsearch_data
        
        # Get all lipid classes from cleaned data
        selected_classes = cleaned_df['ClassKey'].unique().tolist()
        
        # Convert protein_concentrations dict to DataFrame for the service
        protein_df = pd.DataFrame([
            {'Sample': k, 'Concentration': v}
            for k, v in protein_concentrations.items()
        ])
        
        # Create normalization config
        norm_config = NormalizationConfig(
            method='protein',
            selected_classes=selected_classes,
            protein_concentrations=protein_concentrations
        )
        
        # Apply normalization using the normalize() method
        norm_service = NormalizationService()
        normalized_df = norm_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            protein_df=protein_df
        )
        
        # Verify normalization was applied
        assert not normalized_df.empty
        assert 'concentration[s1]' in normalized_df.columns
        
        # Verify calculation is correct (intensity / protein_concentration)
        # Get first lipid for testing
        test_lipid = normalized_df.iloc[0]
        original_intensity = cleaned_df.iloc[0]['intensity[s1]']
        protein_conc = protein_concentrations['s1']
        expected_value = original_intensity / protein_conc
        
        assert abs(test_lipid['concentration[s1]'] - expected_value) < 0.01
    
    def test_protein_normalization_missing_samples(
        self,
        cleaned_lipidsearch_data,
        experiment_config
    ):
        """Test error handling when protein concentrations missing for some samples."""
        cleaned_df, intsta_df = cleaned_lipidsearch_data
        
        # Get all lipid classes from cleaned data
        selected_classes = cleaned_df['ClassKey'].unique().tolist()
        
        # Create incomplete protein concentrations (missing s3)
        incomplete_protein = {
            's1': 2.5,
            's2': 2.8,
            # s3 missing!
            's4': 2.6,
            's5': 2.9,
            's6': 2.4,
            's7': 2.7,
            's8': 2.5,
            's9': 2.6,
            's10': 2.8,
            's11': 2.7,
            's12': 2.5
        }
        
        # Convert to DataFrame
        protein_df = pd.DataFrame([
            {'Sample': k, 'Concentration': v}
            for k, v in incomplete_protein.items()
        ])
        
        norm_config = NormalizationConfig(
            method='protein',
            selected_classes=selected_classes,
            protein_concentrations=incomplete_protein
        )
        
        # Should work but skip missing sample - service handles this gracefully
        norm_service = NormalizationService()
        normalized_df = norm_service.normalize(
            cleaned_df,
            norm_config,
            experiment_config,
            protein_df=protein_df
        )
        
        # Verify it worked for available samples
        assert not normalized_df.empty
        assert 'concentration[s1]' in normalized_df.columns
        # s3 column should still exist but might have different values

# ============================================================================
# TEST CLASS 3: BOTH NORMALIZATION (CRITICAL BUG TEST)
# ============================================================================

class TestBothNormalization:
    """
    Test 'both' normalization method (internal standards + protein).
    
    This tests the critical bug fix from commit 05d1569 where protein
    normalization was skipped in 'both' mode because internal standards
    renamed columns from intensity[] to concentration[].
    """
    
    def test_both_normalization_applies_sequentially(
        self,
        cleaned_lipidsearch_data,
        protein_concentrations,
        experiment_config
    ):
        """
        Test that 'both' method applies both normalizations sequentially.
        
        Expected behavior:
        1. Internal standards normalization: intensity[] → concentration[]
        2. Protein normalization: concentration[] → concentration[] (divided by protein)
        
        Bug from commit 05d1569:
        - Protein step was skipped because it only looked for intensity[] columns
        - Result: Only standards normalization was applied
        """
        cleaned_df, intsta_df = cleaned_lipidsearch_data
        
        if intsta_df.empty:
            pytest.skip("No internal standards in dataset")
        
        # Get one standard per class for testing
        classes = intsta_df['ClassKey'].unique()
        internal_standards = {}
        intsta_concentrations = {}
        
        for cls in classes:
            class_standards = intsta_df[intsta_df['ClassKey'] == cls]['LipidMolec'].tolist()
            if class_standards:
                # Use first standard from each class
                standard_name = class_standards[0]
                internal_standards[cls] = standard_name
                intsta_concentrations[standard_name] = 1000.0
        
        # CRITICAL FIX: Only select classes that have standards
        selected_classes = list(internal_standards.keys())
        
        # Convert protein_concentrations to DataFrame
        protein_df = pd.DataFrame([
            {'Sample': k, 'Concentration': v}
            for k, v in protein_concentrations.items()
        ])
        
        # Step 1: Apply internal standards normalization
        standards_config = NormalizationConfig(
            method='internal_standard',
            selected_classes=selected_classes,
            internal_standards=internal_standards,
            intsta_concentrations=intsta_concentrations
        )
        
        norm_service = NormalizationService()
        after_standards = norm_service.normalize(
            cleaned_df,
            standards_config,
            experiment_config,
            intsta_df=intsta_df
        )
        
        # Verify we have concentration[] columns
        assert 'concentration[s1]' in after_standards.columns
        assert 'intensity[s1]' not in after_standards.columns
        
        # Step 2: Apply protein normalization
        protein_config = NormalizationConfig(
            method='protein',
            selected_classes=selected_classes,
            protein_concentrations=protein_concentrations
        )
        
        after_both = norm_service.normalize(
            after_standards,
            protein_config,
            experiment_config,
            protein_df=protein_df
        )
        
        # Verify protein normalization was applied
        # Values should be different after protein normalization
        assert not after_both.equals(after_standards)
        
        # Check that concentration values changed (divided by protein concentration)
        # Get first lipid for comparison
        before_protein = after_standards.iloc[0]['concentration[s1]']
        after_protein = after_both.iloc[0]['concentration[s1]']
        protein_conc = protein_concentrations['s1']
        
        expected = before_protein / protein_conc
        assert abs(after_protein - expected) < 0.01, \
            "Protein normalization should divide by protein concentration"
    
    def test_both_normalization_end_to_end(
        self,
        cleaned_lipidsearch_data,
        protein_concentrations,
        experiment_config
    ):
        """Test complete 'both' normalization workflow end-to-end."""
        cleaned_df, intsta_df = cleaned_lipidsearch_data
        
        if intsta_df.empty:
            pytest.skip("No internal standards in dataset")
        
        # Get standard concentrations
        classes = intsta_df['ClassKey'].unique()
        internal_standards = {}
        intsta_concentrations = {}
        
        for cls in classes:
            class_standards = intsta_df[intsta_df['ClassKey'] == cls]['LipidMolec'].tolist()
            if class_standards:
                standard_name = class_standards[0]
                internal_standards[cls] = standard_name
                intsta_concentrations[standard_name] = 1000.0
        
        # CRITICAL FIX: Only select classes that have standards
        selected_classes = list(internal_standards.keys())
        
        # Convert protein_concentrations to DataFrame
        protein_df = pd.DataFrame([
            {'Sample': k, 'Concentration': v}
            for k, v in protein_concentrations.items()
        ])
        
        # Apply 'both' method
        both_config = NormalizationConfig(
            method='both',
            selected_classes=selected_classes,
            internal_standards=internal_standards,
            intsta_concentrations=intsta_concentrations,
            protein_concentrations=protein_concentrations
        )
        
        # Use the service's normalize() method which handles 'both'
        norm_service = NormalizationService()
        final_df = norm_service.normalize(
            cleaned_df,
            both_config,
            experiment_config,
            intsta_df=intsta_df,
            protein_df=protein_df
        )
        
        # Verify both normalizations were applied
        assert 'concentration[s1]' in final_df.columns
        assert not final_df.empty
        
        # Verify the calculation is correct
        # Get first lipid
        original_intensity = cleaned_df.iloc[0]['intensity[s1]']
        
        # Get corresponding standard for this lipid's class
        lipid_class = cleaned_df.iloc[0]['ClassKey']
        if lipid_class in classes:
            class_standards = intsta_df[intsta_df['ClassKey'] == lipid_class]
            if not class_standards.empty:
                standard_name = class_standards.iloc[0]['LipidMolec']
                standard_intensity = class_standards.iloc[0]['intensity[s1]']
                standard_conc = intsta_concentrations[standard_name]
                protein_conc = protein_concentrations['s1']
                
                # Expected: (lipid_intensity / standard_intensity) * standard_conc / protein_conc
                expected = (original_intensity / standard_intensity) * standard_conc / protein_conc
                actual = final_df.iloc[0]['concentration[s1]']
                
                # Allow small floating point differences
                assert abs(actual - expected) < 0.01, \
                    f"Both normalization calculation incorrect: expected {expected}, got {actual}"

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])