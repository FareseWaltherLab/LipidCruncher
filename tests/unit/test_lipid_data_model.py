"""
Unit tests for LipidMeasurement and LipidDataset models.
"""
import pytest
from pydantic import ValidationError
from src.lipidcruncher.core.models.lipid_data import LipidMeasurement, LipidDataset


class TestLipidMeasurement:
    """Test suite for LipidMeasurement model."""
    
    def test_valid_generic_measurement(self):
        """Test creating valid generic format measurement (no grade)."""
        measurement = LipidMeasurement(
            lipid_name='PC(16:0_18:1)',
            lipid_class='PC',
            values=[100.0, 150.0, 120.0],
            sample_ids=['s1', 's2', 's3']
        )
        
        assert measurement.lipid_name == 'PC(16:0_18:1)'
        assert measurement.lipid_class == 'PC'
        assert measurement.grade is None
        assert measurement.is_lipidsearch_format() == False
    
    def test_valid_lipidsearch_measurement(self):
        """Test creating valid LipidSearch format measurement."""
        measurement = LipidMeasurement(
            lipid_name='PE(18:0_20:4)',
            lipid_class='PE',
            values=[200.0, 250.0, 180.0],
            sample_ids=['s1', 's2', 's3'],
            grade='A',
            calc_mass=768.5543,
            base_rt=12.34,
            fa_key='18:0_20:4',
            sample_id_rate=95.5
        )
        
        assert measurement.grade == 'A'
        assert measurement.calc_mass == 768.5543
        assert measurement.base_rt == 12.34
        assert measurement.fa_key == '18:0_20:4'
        assert measurement.sample_id_rate == 95.5
        assert measurement.is_lipidsearch_format() == True
    
    def test_invalid_grade_raises_error(self):
        """Test that invalid grade raises error."""
        with pytest.raises(ValidationError, match="Grade must be A, B, C, or D"):
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0],
                sample_ids=['s1', 's2'],
                grade='E'
            )
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched values and sample_ids raises error."""
        with pytest.raises(ValidationError, match="must have same length"):
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0, 120.0],
                sample_ids=['s1', 's2']  # Only 2 samples for 3 values
            )
    
    def test_negative_calc_mass_raises_error(self):
        """Test that negative calc_mass raises error."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0],
                sample_ids=['s1', 's2'],
                calc_mass=-100.0
            )
    
    def test_negative_base_rt_raises_error(self):
        """Test that negative base_rt raises error."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0],
                sample_ids=['s1', 's2'],
                base_rt=-5.0
            )
    
    def test_to_series(self):
        """Test conversion to pandas Series."""
        measurement = LipidMeasurement(
            lipid_name='TAG(16:0_18:1_18:2)',
            lipid_class='TAG',
            values=[300.0, 350.0, 280.0],
            sample_ids=['s1', 's2', 's3']
        )
        
        series = measurement.to_series()
        assert series.name == 'TAG(16:0_18:1_18:2)'
        assert list(series.index) == ['s1', 's2', 's3']
        assert list(series.values) == [300.0, 350.0, 280.0]


class TestLipidDataset:
    """Test suite for LipidDataset model."""
    
    def test_empty_dataset(self):
        """Test creating empty dataset."""
        dataset = LipidDataset()
        assert dataset.count() == 0
        assert dataset.get_lipid_classes() == []
    
    def test_dataset_with_measurements(self):
        """Test creating dataset with measurements."""
        measurements = [
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0],
                sample_ids=['s1', 's2']
            ),
            LipidMeasurement(
                lipid_name='PE(18:0_20:4)',
                lipid_class='PE',
                values=[200.0, 250.0],
                sample_ids=['s1', 's2']
            )
        ]
        
        dataset = LipidDataset(measurements=measurements)
        assert dataset.count() == 2
        assert set(dataset.get_lipid_classes()) == {'PC', 'PE'}
    
    def test_filter_by_class(self):
        """Test filtering by lipid class."""
        measurements = [
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0],
                sample_ids=['s1']
            ),
            LipidMeasurement(
                lipid_name='PE(18:0_20:4)',
                lipid_class='PE',
                values=[200.0],
                sample_ids=['s1']
            ),
            LipidMeasurement(
                lipid_name='PC(18:0_18:1)',
                lipid_class='PC',
                values=[150.0],
                sample_ids=['s1']
            )
        ]
        
        dataset = LipidDataset(measurements=measurements)
        pc_only = dataset.filter_by_class(['PC'])
        
        assert pc_only.count() == 2
        assert pc_only.get_lipid_classes() == ['PC']
    
    def test_filter_by_grade(self):
        """Test filtering by grade (LipidSearch only)."""
        measurements = [
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0],
                sample_ids=['s1'],
                grade='A'
            ),
            LipidMeasurement(
                lipid_name='PE(18:0_20:4)',
                lipid_class='PE',
                values=[200.0],
                sample_ids=['s1'],
                grade='C'
            ),
            LipidMeasurement(
                lipid_name='TAG(16:0_18:1_18:2)',
                lipid_class='TAG',
                values=[300.0],
                sample_ids=['s1']
                # No grade - generic format
            )
        ]
        
        dataset = LipidDataset(measurements=measurements)
        ab_only = dataset.filter_by_grade(['A', 'B'])
        
        # Should keep grade A and the generic one (no grade)
        assert ab_only.count() == 2
    
    def test_get_lipidsearch_measurements(self):
        """Test filtering to only LipidSearch measurements."""
        measurements = [
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0],
                sample_ids=['s1'],
                grade='A'
            ),
            LipidMeasurement(
                lipid_name='PE(18:0_20:4)',
                lipid_class='PE',
                values=[200.0],
                sample_ids=['s1']
                # No grade - generic format
            )
        ]
        
        dataset = LipidDataset(measurements=measurements)
        lipidsearch_only = dataset.get_lipidsearch_measurements()
        
        assert lipidsearch_only.count() == 1
        assert lipidsearch_only.measurements[0].lipid_name == 'PC(16:0_18:1)'
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        measurements = [
            LipidMeasurement(
                lipid_name='PC(16:0_18:1)',
                lipid_class='PC',
                values=[100.0, 150.0],
                sample_ids=['s1', 's2']
            ),
            LipidMeasurement(
                lipid_name='PE(18:0_20:4)',
                lipid_class='PE',
                values=[200.0, 250.0],
                sample_ids=['s1', 's2']
            )
        ]
        
        dataset = LipidDataset(measurements=measurements)
        df = dataset.to_dataframe()
        
        assert df.shape == (2, 2)  # 2 lipids, 2 samples
        assert list(df.columns) == ['s1', 's2']
