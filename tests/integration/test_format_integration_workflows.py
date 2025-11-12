"""
Integration tests for consistency plots with real data formats.

Tests data loading, preprocessing, cleaning, and plot generation
with actual sample files in all three supported formats.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lipidcruncher.core.services.internal_standards_visualization_service import InternalStandardsVisualizationService
from lipidcruncher.core.services.data_cleaning_service import DataCleaningService
from lipidcruncher.core.services.format_preprocessing_service import FormatPreprocessingService
from lipidcruncher.core.models.experiment import ExperimentConfig


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
def generic_data(fixtures_dir):
    """Load Generic format sample data."""
    return pd.read_csv(fixtures_dir / 'generic_sample.csv')


@pytest.fixture
def metabolomics_data(fixtures_dir):
    """Load Metabolomics Workbench sample data."""
    with open(fixtures_dir / 'metabolomic_sample.csv', 'r') as f:
        return f.read()


@pytest.fixture
def experiment_config():
    """Create a basic experiment configuration."""
    return ExperimentConfig(
        n_conditions=3,
        conditions_list=['WT', 'ADGAT-DKO', 'BQC'],
        number_of_samples_list=[4, 4, 4]
    )


# ============================================================================
# TEST CLASS 1: LIPIDSEARCH FORMAT
# ============================================================================

class TestLipidSearchFormat:
    """Test complete workflow with real LipidSearch data."""
    
    def test_lipidsearch_format_end_to_end(
        self,
        lipidsearch_data,
        experiment_config
    ):
        """
        Test complete workflow with real LipidSearch data.
        
        Steps:
        1. Preprocess format (convert MeanArea -> intensity[])
        2. Clean data (grade filtering, extract standards)
        3. Generate consistency plots
        4. Verify plot properties
        5. Test CSV export
        """
        # Step 1: Preprocess format
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(
            lipidsearch_data,
            'lipidsearch'
        )

        if not success:
            raise ValueError(f"Preprocessing failed: {message}")

        
        # Verify preprocessing worked
        assert 'intensity[s1]' in preprocessed_df.columns
        assert 'LipidMolec' in preprocessed_df.columns
        assert 'ClassKey' in preprocessed_df.columns
        
        # Step 2: Clean data and extract standards
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            preprocessed_df,
            experiment_config,
            grade_config=None
        )
        cleaned_df, intsta_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Verify standards were extracted (LipidSearch has deuterated standards)
        assert not intsta_df.empty, "No internal standards detected in LipidSearch data"
        assert 'ClassKey' in intsta_df.columns
        
        # Check intensity columns exist
        intensity_cols = [c for c in intsta_df.columns if c.startswith('intensity[')]
        assert len(intensity_cols) > 0, "No intensity columns in standards"
        
        # Step 3: Get classes with standards
        classes_with_standards = sorted(intsta_df['ClassKey'].unique())
        assert len(classes_with_standards) > 0, "No classes with standards"
        
        print(f"Found standards for classes: {classes_with_standards}")
        
        # Step 4: Generate consistency plots
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(
            intsta_df,
            experiment_config.full_samples_list
        )
        
        assert len(figs) > 0, "No plots generated"
        assert len(figs) == len(classes_with_standards), "Plot count doesn't match class count"
        
        # Step 5: Verify plot properties
        for fig in figs:
            assert fig is not None
            assert len(fig.data) > 0, "Plot has no traces"
            
            # Check layout
            assert fig.layout.xaxis.title.text == 'Samples'
            assert fig.layout.yaxis.title.text == 'Raw Intensity'
            assert fig.layout.barmode == 'stack'
            
            # Check text color (should be black, not white)
            assert fig.layout.font.color == 'black'
        
        # Step 6: Test CSV export for each class
        for class_name in classes_with_standards:
            csv_df = viz_service.prepare_csv_data(
                intsta_df,
                class_name,
                experiment_config.full_samples_list
            )
            
            assert not csv_df.empty, f"CSV export failed for {class_name}"
            assert list(csv_df.columns) == ['LipidMolec', 'ClassKey', 'Sample', 'Intensity']
            assert all(csv_df['ClassKey'] == class_name)
            
            # Verify we have data for all samples
            samples_in_csv = set(csv_df['Sample'].unique())
            assert samples_in_csv.issubset(set(experiment_config.full_samples_list))
    
    def test_lipidsearch_with_condition_filtering(
        self,
        lipidsearch_data,
        experiment_config
    ):
        """Test generating plots with only selected conditions."""
        # Preprocess and clean
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(lipidsearch_data, 'lipidsearch')

        if not success:

            raise ValueError(f"Preprocessing failed: {message}")
        
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            preprocessed_df,
            experiment_config,
            grade_config=None
        )
        cleaned_df, intsta_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Select only WT condition (first 4 samples)
        wt_samples = experiment_config.individual_samples_list[0]
        
        # Generate plots with filtered samples
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(intsta_df, wt_samples)
        
        assert len(figs) > 0
        
        # Verify plots only show selected samples
        for fig in figs:
            # Check x-axis shows only WT samples
            assert list(fig.layout.xaxis.categoryarray) == wt_samples


# ============================================================================
# TEST CLASS 2: GENERIC FORMAT
# ============================================================================

class TestGenericFormat:
    """Test complete workflow with real Generic format data."""
    
    def test_generic_format_end_to_end(
        self,
        generic_data,
        experiment_config
    ):
        """
        Test complete workflow with real Generic format data.
        
        Steps:
        1. Preprocess format (add ClassKey, standardize names)
        2. Clean data
        3. Verify no auto-detected standards (Generic doesn't have them)
        4. Manually create standards for testing plots
        """
        # Step 1: Preprocess Generic format
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(
            generic_data,
            'generic'
        )

        if not success:
            raise ValueError(f"Preprocessing failed: {message}")

        
        # Verify preprocessing worked
        assert 'ClassKey' in preprocessed_df.columns, "ClassKey not added"
        assert 'LipidMolec' in preprocessed_df.columns
        
        # Verify lipid names standardized (should have parentheses)
        assert all('(' in name and ')' in name for name in preprocessed_df['LipidMolec'])
        
        # Verify intensity columns created
        intensity_cols = [c for c in preprocessed_df.columns if c.startswith('intensity[')]
        assert len(intensity_cols) > 0, "No intensity columns created"
        
        # Step 2: Clean data (no grade filtering for Generic)
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_generic_data(
            preprocessed_df,
            experiment_config
        )
        
        assert not cleaned_df.empty
        assert 'ClassKey' in cleaned_df.columns
        
        # Step 3: Generic format doesn't auto-detect standards
        # We would need custom standards upload for plotting
        # But we can verify the data structure is correct
        
        # Check we have multiple lipid classes
        classes = cleaned_df['ClassKey'].unique()
        assert len(classes) > 1, "Should have multiple lipid classes"
        
        print(f"Generic format has {len(classes)} lipid classes: {sorted(classes)[:5]}...")
    
    def test_generic_format_with_manual_standards(
        self,
        generic_data,
        experiment_config
    ):
        """Test Generic format with manually selected standards."""
        # Preprocess
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(generic_data, 'generic')

        if not success:

            raise ValueError(f"Preprocessing failed: {message}")
        
        # Clean
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_generic_data(
            preprocessed_df,
            experiment_config
        )
        
        # Manually select some lipids as "standards" for testing
        # (In real workflow, user would upload standards file)
        available_lipids = cleaned_df['LipidMolec'].tolist()
        
        # Pick first lipid from each of the first 2 classes as mock standards
        classes = sorted(cleaned_df['ClassKey'].unique())[:2]
        mock_standards = []
        
        for cls in classes:
            class_lipids = cleaned_df[cleaned_df['ClassKey'] == cls]['LipidMolec'].tolist()
            if class_lipids:
                mock_standards.append(class_lipids[0])
        
        # Create mock standards DataFrame
        mock_intsta_df = cleaned_df[cleaned_df['LipidMolec'].isin(mock_standards)].copy()
        
        # Generate plots
        viz_service = InternalStandardsVisualizationService()
        figs = viz_service.create_consistency_plots(
            mock_intsta_df,
            experiment_config.full_samples_list
        )
        
        assert len(figs) > 0, "No plots generated with manual standards"
        assert len(figs) == len(classes), "Should have one plot per class"


# ============================================================================
# TEST CLASS 3: METABOLOMICS WORKBENCH FORMAT
# ============================================================================

class TestMetabolomicsFormat:
    """Test complete workflow with real Metabolomics Workbench data."""
    
    def test_metabolomics_format_end_to_end(
        self,
        metabolomics_data,
        experiment_config
    ):
        """
        Test complete workflow with real Metabolomics Workbench data.
        
        Steps:
        1. Preprocess format (extract data section, standardize names)
        2. Verify data structure
        3. Clean data
        4. Test with manual standards selection
        """
        # Step 1: Preprocess MW format (text input)
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(
            metabolomics_data,
            'metabolomics_workbench'
        )

        if not success:
            raise ValueError(f"Preprocessing failed: {message}")

        
        # Verify data extracted
        assert not preprocessed_df.empty, "No data extracted from MW format"
        assert 'LipidMolec' in preprocessed_df.columns
        
        # Verify intensity columns created
        intensity_cols = [c for c in preprocessed_df.columns if c.startswith('intensity[')]
        assert len(intensity_cols) > 0, "No intensity columns created"
        
        # Verify lipid names standardized
        assert all('(' in name and ')' in name for name in preprocessed_df['LipidMolec'])
        
        # Step 2: Clean data
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_generic_data(  # MW uses generic cleaning
            preprocessed_df,
            experiment_config
        )
        
        assert not cleaned_df.empty
        
        # Verify we have lipid classes
        if 'ClassKey' in cleaned_df.columns:
            classes = cleaned_df['ClassKey'].unique()
            assert len(classes) > 0
            print(f"Metabolomics format has {len(classes)} lipid classes")


# ============================================================================
# TEST CLASS 4: DATA INTEGRITY
# ============================================================================

class TestDataIntegrity:
    """Test data integrity through the pipeline."""
    
    def test_sample_count_preserved(
        self,
        lipidsearch_data,
        experiment_config
    ):
        """Test that sample count is preserved through pipeline."""
        # Process data
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(lipidsearch_data, 'lipidsearch')

        if not success:

            raise ValueError(f"Preprocessing failed: {message}")
        
        # Count intensity columns
        intensity_cols = [c for c in preprocessed_df.columns if c.startswith('intensity[')]
        
        # Should match total samples in experiment config
        expected_samples = len(experiment_config.full_samples_list)
        assert len(intensity_cols) == expected_samples, \
            f"Sample count mismatch: {len(intensity_cols)} vs {expected_samples}"
    
    def test_lipid_count_after_cleaning(
        self,
        lipidsearch_data,
        experiment_config
    ):
        """Test that lipid counts are reasonable after cleaning."""
        # Process data
        preprocessing_service = FormatPreprocessingService()
        preprocessed_df, success, message = preprocessing_service.validate_and_preprocess(lipidsearch_data, 'lipidsearch')

        if not success:

            raise ValueError(f"Preprocessing failed: {message}")
        
        original_count = len(preprocessed_df)
        
        # Clean with standard grade filtering
        cleaning_service = DataCleaningService()
        cleaned_df = cleaning_service.clean_lipidsearch_data(
            preprocessed_df,
            experiment_config,
            grade_config=None
        )
        cleaned_df, intsta_df = cleaning_service.extract_internal_standards(cleaned_df)
        
        # Should have fewer lipids after cleaning (some filtered out)
        assert len(cleaned_df) < original_count
        assert len(cleaned_df) > 0, "All lipids filtered out!"
        
        # Standards should be a small subset
        if not intsta_df.empty:
            assert len(intsta_df) < len(cleaned_df)
            
        print(f"Original: {original_count}, Cleaned: {len(cleaned_df)}, Standards: {len(intsta_df)}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])