"""Unit tests for StandardsService."""
import pytest
import pandas as pd
import numpy as np
from app.services.standards import (
    StandardsService,
    StandardsExtractionResult,
    StandardsValidationResult,
    StandardsProcessingResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_lipid_df():
    """Basic lipid DataFrame without standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PE(18:0_20:4)', 'TG(16:0_18:1_18:2)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [1000.0, 2000.0, 3000.0],
        'intensity[s2]': [1100.0, 2100.0, 3100.0],
        'intensity[s3]': [1200.0, 2200.0, 3200.0],
    })


@pytest.fixture
def df_with_deuterated_standards():
    """DataFrame with deuterated internal standards (d5, d7, d9 patterns)."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_15:0)(d7)', 'PC(18:0_18:1)',
            'PE(18:0_20:4)', 'PE(17:0_17:0)(d5)',
            'TG(16:0_18:1_18:2)', 'TG(15:0_15:0_15:0)(d9)',
        ],
        'ClassKey': ['PC', 'PC', 'PC', 'PE', 'PE', 'TG', 'TG'],
        'intensity[s1]': [1000.0, 500.0, 1100.0, 2000.0, 600.0, 3000.0, 700.0],
        'intensity[s2]': [1050.0, 520.0, 1150.0, 2050.0, 620.0, 3050.0, 720.0],
        'intensity[s3]': [1100.0, 540.0, 1200.0, 2100.0, 640.0, 3100.0, 740.0],
    })


@pytest.fixture
def df_with_istd_standards():
    """DataFrame with ISTD-labeled internal standards."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC_ISTD',
            'PE(18:0_20:4)', 'PE_ISTD',
        ],
        'ClassKey': ['PC', 'ISTD', 'PE', 'ISTD'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
    })


@pytest.fixture
def df_with_is_suffix():
    """DataFrame with _IS suffix standards."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_15:0)_IS',
            'PE(18:0_20:4)', 'PE(17:0_17:0)_IS',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
    })


@pytest.fixture
def df_with_splash_standards():
    """DataFrame with SPLASH lipidomix standards."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'SPLASH_PC',
            'PE(18:0_20:4)', 'SPLASH_PE',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
    })


@pytest.fixture
def df_with_is_marker():
    """DataFrame with (IS) marker standards."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_15:0)(IS)',
            'PE(18:0_20:4)', 'PE(17:0_17:0)(IS)',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
    })


@pytest.fixture
def df_with_class_istd():
    """DataFrame with Internal class key."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'Standard1',
            'PE(18:0_20:4)', 'Standard2',
        ],
        'ClassKey': ['PC', 'Internal', 'PE', 'Internal'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
    })


@pytest.fixture
def df_with_plus_d_pattern():
    """DataFrame with +D7 pattern standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)+D7', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0],
    })


@pytest.fixture
def df_with_cholesterol_standard():
    """DataFrame with cholesterol deuterated standard."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'Ch-D7', 'Cholesterol'],
        'ClassKey': ['PC', 'Cholesterol', 'Cholesterol'],
        'intensity[s1]': [1000.0, 500.0, 2000.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0],
    })


@pytest.fixture
def df_with_s_notation():
    """DataFrame with :(s) notation standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0):(s)', 'PE(18:0_20:4)'],
        'ClassKey': ['PC', 'PC', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0],
    })


@pytest.fixture
def df_with_mixed_standards():
    """DataFrame with multiple types of standards."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_15:0)(d7)',
            'PE(18:0_20:4)', 'PE_ISTD',
            'TG(16:0_18:1_18:2)', 'TG(15:0)_IS',
            'SM(d18:1_16:0)', 'SPLASH_SM',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'ISTD', 'TG', 'TG', 'SM', 'SM'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0, 3000.0, 700.0, 4000.0, 800.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0, 3100.0, 750.0, 4100.0, 850.0],
    })


@pytest.fixture
def standards_only_df():
    """DataFrame with only standards."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)', 'TG(15:0)(d9)'],
        'ClassKey': ['PC', 'PE', 'TG'],
        'intensity[s1]': [500.0, 600.0, 700.0],
        'intensity[s2]': [520.0, 620.0, 720.0],
    })


@pytest.fixture
def multi_sample_df():
    """DataFrame with many samples."""
    return pd.DataFrame({
        'LipidMolec': [
            'PC(16:0_18:1)', 'PC(15:0_15:0)(d7)',
            'PE(18:0_20:4)', 'PE(17:0_17:0)(d5)',
        ],
        'ClassKey': ['PC', 'PC', 'PE', 'PE'],
        'intensity[s1]': [1000.0, 500.0, 2000.0, 600.0],
        'intensity[s2]': [1100.0, 550.0, 2100.0, 650.0],
        'intensity[s3]': [1200.0, 600.0, 2200.0, 700.0],
        'intensity[s4]': [1300.0, 650.0, 2300.0, 750.0],
        'intensity[s5]': [1400.0, 700.0, 2400.0, 800.0],
    })


@pytest.fixture
def uploaded_standards_simple():
    """Simple uploaded standards file - just lipid names."""
    return pd.DataFrame({
        'LipidMolec': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
    })


@pytest.fixture
def uploaded_standards_with_class():
    """Uploaded standards file with ClassKey."""
    return pd.DataFrame({
        'Standard': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
        'Class': ['PC', 'PE'],
    })


@pytest.fixture
def uploaded_standards_complete():
    """Complete standards file with intensity values."""
    return pd.DataFrame({
        'Standard': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
        'Class': ['PC', 'PE'],
        'Sample1': [500.0, 600.0],
        'Sample2': [520.0, 620.0],
        'Sample3': [540.0, 640.0],
    })


# =============================================================================
# Test: StandardsExtractionResult Dataclass
# =============================================================================

class TestStandardsExtractionResult:
    """Tests for StandardsExtractionResult dataclass."""

    def test_creation_with_all_fields(self, basic_lipid_df, standards_only_df):
        """Test creating result with all fields."""
        result = StandardsExtractionResult(
            data_df=basic_lipid_df,
            standards_df=standards_only_df,
            standards_count=3,
            detection_patterns_matched=['lipid:(d\\d+)']
        )
        assert len(result.data_df) == 3
        assert len(result.standards_df) == 3
        assert result.standards_count == 3
        assert len(result.detection_patterns_matched) == 1

    def test_creation_with_empty_patterns(self, basic_lipid_df):
        """Test creating result with empty patterns list."""
        result = StandardsExtractionResult(
            data_df=basic_lipid_df,
            standards_df=pd.DataFrame(),
            standards_count=0,
            detection_patterns_matched=[]
        )
        assert result.detection_patterns_matched == []

    def test_default_patterns_field(self, basic_lipid_df):
        """Test that detection_patterns_matched defaults to empty list."""
        result = StandardsExtractionResult(
            data_df=basic_lipid_df,
            standards_df=pd.DataFrame(),
            standards_count=0
        )
        assert result.detection_patterns_matched == []


# =============================================================================
# Test: StandardsValidationResult Dataclass
# =============================================================================

class TestStandardsValidationResult:
    """Tests for StandardsValidationResult dataclass."""

    def test_valid_result(self):
        """Test creating valid result."""
        result = StandardsValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            valid_standards_count=5
        )
        assert result.is_valid is True
        assert result.valid_standards_count == 5

    def test_invalid_result_with_errors(self):
        """Test creating invalid result with errors."""
        result = StandardsValidationResult(
            is_valid=False,
            errors=["Missing column"],
            warnings=["Duplicates found"],
            valid_standards_count=0
        )
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_default_lists(self):
        """Test default empty lists."""
        result = StandardsValidationResult(is_valid=True)
        assert result.errors == []
        assert result.warnings == []
        assert result.valid_standards_count == 0


# =============================================================================
# Test: StandardsProcessingResult Dataclass
# =============================================================================

class TestStandardsProcessingResult:
    """Tests for StandardsProcessingResult dataclass."""

    def test_extract_mode_result(self, standards_only_df):
        """Test result for extract mode."""
        result = StandardsProcessingResult(
            standards_df=standards_only_df,
            duplicates_removed=0,
            standards_count=3,
            source_mode='extract'
        )
        assert result.source_mode == 'extract'
        assert result.standards_count == 3

    def test_complete_mode_result(self, standards_only_df):
        """Test result for complete mode."""
        result = StandardsProcessingResult(
            standards_df=standards_only_df,
            duplicates_removed=2,
            standards_count=3,
            source_mode='complete'
        )
        assert result.source_mode == 'complete'
        assert result.duplicates_removed == 2


# =============================================================================
# Test: detect_standards - Deuterated Standards
# =============================================================================

class TestDetectStandardsDeuterated:
    """Tests for detecting deuterated internal standards."""

    def test_detect_d5_pattern(self):
        """Test detection of (d5) pattern."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)(d5)'],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert result[0] is False or result.iloc[0] == False
        assert result[1] is True or result.iloc[1] == True

    def test_detect_d7_pattern(self):
        """Test detection of (d7) pattern."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)(d7)'],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_detect_d9_pattern(self):
        """Test detection of (d9) pattern."""
        df = pd.DataFrame({
            'LipidMolec': ['TG(16:0_18:1_18:2)', 'TG(15:0_15:0_15:0)(d9)'],
            'ClassKey': ['TG', 'TG'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_detect_d31_pattern(self):
        """Test detection of larger deuterium numbers."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(16:0_16:0)(d31)'],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_detect_plus_d_pattern(self, df_with_plus_d_pattern):
        """Test detection of +D7 pattern."""
        result = StandardsService.detect_standards(df_with_plus_d_pattern)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]

    def test_detect_minus_d_pattern(self):
        """Test detection of -D7 pattern."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)-D7'],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_detect_bracketed_d_pattern(self):
        """Test detection of [d7] pattern."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)[d7]'],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_detect_cholesterol_deuterated(self, df_with_cholesterol_standard):
        """Test detection of Ch-D7 cholesterol standard."""
        result = StandardsService.detect_standards(df_with_cholesterol_standard)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]


# =============================================================================
# Test: detect_standards - Other Patterns
# =============================================================================

class TestDetectStandardsOtherPatterns:
    """Tests for detecting other internal standard patterns."""

    def test_detect_istd_in_name(self, df_with_istd_standards):
        """Test detection of ISTD in lipid name."""
        result = StandardsService.detect_standards(df_with_istd_standards)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]

    def test_detect_is_suffix(self, df_with_is_suffix):
        """Test detection of _IS suffix."""
        result = StandardsService.detect_standards(df_with_is_suffix)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]

    def test_detect_splash_pattern(self, df_with_splash_standards):
        """Test detection of SPLASH standards."""
        result = StandardsService.detect_standards(df_with_splash_standards)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]

    def test_detect_is_marker(self, df_with_is_marker):
        """Test detection of (IS) marker."""
        result = StandardsService.detect_standards(df_with_is_marker)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]

    def test_detect_s_notation(self, df_with_s_notation):
        """Test detection of :(s) notation."""
        result = StandardsService.detect_standards(df_with_s_notation)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]

    def test_detect_class_istd(self, df_with_class_istd):
        """Test detection by ClassKey containing ISTD/Internal."""
        result = StandardsService.detect_standards(df_with_class_istd)
        assert not result.iloc[0]
        assert result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]


# =============================================================================
# Test: detect_standards - Edge Cases
# =============================================================================

class TestDetectStandardsEdgeCases:
    """Tests for edge cases in standard detection."""

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.detect_standards(pd.DataFrame())

    def test_none_dataframe_raises_error(self):
        """Test that None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.detect_standards(None)

    def test_missing_lipidmolec_column(self):
        """Test that missing LipidMolec column raises error."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        with pytest.raises(ValueError, match="LipidMolec"):
            StandardsService.detect_standards(df)

    def test_no_standards_detected(self, basic_lipid_df):
        """Test DataFrame with no standards."""
        result = StandardsService.detect_standards(basic_lipid_df)
        assert not result.any()

    def test_all_standards(self, standards_only_df):
        """Test DataFrame with only standards."""
        result = StandardsService.detect_standards(standards_only_df)
        assert result.all()

    def test_mixed_standards(self, df_with_mixed_standards):
        """Test DataFrame with mixed standards and non-standards."""
        result = StandardsService.detect_standards(df_with_mixed_standards)
        expected = [False, True, False, True, False, True, False, True]
        assert list(result) == expected

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)(D7)', 'PE(18:0)(d7)', 'istd_TG'],
            'ClassKey': ['PC', 'PE', 'TG'],
        })
        result = StandardsService.detect_standards(df)
        assert result.all()

    def test_missing_classkey_column(self):
        """Test detection works without ClassKey column."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0_15:0)(d7)'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_na_values_in_lipidmolec(self):
        """Test handling of NA values in LipidMolec."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', None, 'PC(15:0)(d7)', np.nan],
            'ClassKey': ['PC', 'PC', 'PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert not result.iloc[1]  # NA treated as not matching
        assert result.iloc[2]
        assert not result.iloc[3]

    def test_empty_string_lipidmolec(self):
        """Test handling of empty string LipidMolec."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', '', '  ', 'PC(15:0)(d7)'],
            'ClassKey': ['PC', 'PC', 'PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert not result.iloc[1]
        assert not result.iloc[2]
        assert result.iloc[3]


# =============================================================================
# Test: get_matched_patterns
# =============================================================================

class TestGetMatchedPatterns:
    """Tests for get_matched_patterns method."""

    def test_deuterated_pattern_match(self, df_with_deuterated_standards):
        """Test matching deuterated patterns."""
        matches = StandardsService.get_matched_patterns(df_with_deuterated_standards)
        assert 'PC(15:0_15:0)(d7)' in matches
        assert 'PE(17:0_17:0)(d5)' in matches
        assert 'TG(15:0_15:0_15:0)(d9)' in matches

    def test_multiple_patterns_per_lipid(self):
        """Test lipid matching multiple patterns."""
        df = pd.DataFrame({
            'LipidMolec': ['ISTD_PC(d7)'],
            'ClassKey': ['ISTD'],
        })
        matches = StandardsService.get_matched_patterns(df)
        assert 'ISTD_PC(d7)' in matches
        # Should match both ISTD and (d7)
        assert len(matches['ISTD_PC(d7)']) >= 2

    def test_no_matches(self, basic_lipid_df):
        """Test no patterns matched."""
        matches = StandardsService.get_matched_patterns(basic_lipid_df)
        assert len(matches) == 0

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty dict."""
        matches = StandardsService.get_matched_patterns(pd.DataFrame())
        assert matches == {}

    def test_none_dataframe(self):
        """Test None DataFrame returns empty dict."""
        matches = StandardsService.get_matched_patterns(None)
        assert matches == {}

    def test_missing_lipidmolec_column(self):
        """Test missing LipidMolec returns empty dict."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        matches = StandardsService.get_matched_patterns(df)
        assert matches == {}

    def test_class_pattern_match(self, df_with_class_istd):
        """Test class pattern matching."""
        matches = StandardsService.get_matched_patterns(df_with_class_istd)
        assert 'Standard1' in matches
        assert 'Standard2' in matches
        assert any('class:' in p for p in matches['Standard1'])


# =============================================================================
# Test: extract_standards
# =============================================================================

class TestExtractStandards:
    """Tests for extract_standards method."""

    def test_extract_deuterated_standards(self, df_with_deuterated_standards):
        """Test extracting deuterated standards."""
        result = StandardsService.extract_standards(df_with_deuterated_standards)

        assert isinstance(result, StandardsExtractionResult)
        assert result.standards_count == 3
        assert len(result.standards_df) == 3
        assert len(result.data_df) == 4

    def test_extract_preserves_columns(self, df_with_deuterated_standards):
        """Test that extraction preserves all columns."""
        result = StandardsService.extract_standards(df_with_deuterated_standards)

        assert 'LipidMolec' in result.standards_df.columns
        assert 'ClassKey' in result.standards_df.columns
        assert 'intensity[s1]' in result.standards_df.columns
        assert 'intensity[s2]' in result.standards_df.columns

    def test_extract_resets_index(self, df_with_deuterated_standards):
        """Test that extraction resets DataFrame index."""
        result = StandardsService.extract_standards(df_with_deuterated_standards)

        assert list(result.standards_df.index) == [0, 1, 2]
        assert list(result.data_df.index) == [0, 1, 2, 3]

    def test_extract_no_standards(self, basic_lipid_df):
        """Test extraction when no standards present."""
        result = StandardsService.extract_standards(basic_lipid_df)

        assert result.standards_count == 0
        assert result.standards_df.empty
        assert len(result.data_df) == 3

    def test_extract_all_standards(self, standards_only_df):
        """Test extraction when all rows are standards."""
        result = StandardsService.extract_standards(standards_only_df)

        assert result.standards_count == 3
        assert len(result.standards_df) == 3
        assert result.data_df.empty

    def test_extract_mixed_standards(self, df_with_mixed_standards):
        """Test extraction with multiple standard types."""
        result = StandardsService.extract_standards(df_with_mixed_standards)

        assert result.standards_count == 4
        assert len(result.data_df) == 4

    def test_extract_empty_dataframe(self):
        """Test extraction from empty DataFrame raises error."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.extract_standards(pd.DataFrame())

    def test_extract_none_dataframe(self):
        """Test extraction from None raises error."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.extract_standards(None)

    def test_extract_missing_lipidmolec(self):
        """Test extraction without LipidMolec column raises error."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        with pytest.raises(ValueError, match="LipidMolec"):
            StandardsService.extract_standards(df)

    def test_extract_patterns_tracked(self, df_with_deuterated_standards):
        """Test that matched patterns are tracked."""
        result = StandardsService.extract_standards(df_with_deuterated_standards)
        assert len(result.detection_patterns_matched) > 0


# =============================================================================
# Test: remove_standards_from_dataset
# =============================================================================

class TestRemoveStandardsFromDataset:
    """Tests for remove_standards_from_dataset method."""

    def test_remove_existing_standards(self, df_with_deuterated_standards, standards_only_df):
        """Test removing standards that exist in dataset."""
        # Create standards_df with just names
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
        })

        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            df_with_deuterated_standards, standards_df
        )

        assert len(removed) == 2
        assert 'PC(15:0_15:0)(d7)' in removed
        assert 'PE(17:0_17:0)(d5)' in removed
        assert len(filtered_df) == 5  # 7 - 2

    def test_remove_nonexistent_standards(self, basic_lipid_df):
        """Test removing standards that don't exist."""
        standards_df = pd.DataFrame({
            'LipidMolec': ['NonExistent1', 'NonExistent2'],
        })

        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            basic_lipid_df, standards_df
        )

        assert len(removed) == 0
        assert len(filtered_df) == 3

    def test_remove_empty_standards_df(self, basic_lipid_df):
        """Test with empty standards DataFrame."""
        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            basic_lipid_df, pd.DataFrame()
        )

        assert len(removed) == 0
        assert len(filtered_df) == 3

    def test_remove_none_standards_df(self, basic_lipid_df):
        """Test with None standards DataFrame."""
        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            basic_lipid_df, None
        )

        assert len(removed) == 0
        assert len(filtered_df) == 3

    def test_remove_empty_main_df(self):
        """Test with empty main DataFrame."""
        standards_df = pd.DataFrame({'LipidMolec': ['Standard1']})
        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            pd.DataFrame(), standards_df
        )

        assert len(removed) == 0
        assert filtered_df.empty

    def test_remove_none_main_df_raises(self):
        """Test that None main DataFrame raises error."""
        with pytest.raises(ValueError, match="None"):
            StandardsService.remove_standards_from_dataset(
                None, pd.DataFrame({'LipidMolec': ['S1']})
            )

    def test_remove_missing_lipidmolec_main(self):
        """Test missing LipidMolec in main raises error."""
        df = pd.DataFrame({'SomeCol': [1, 2]})
        standards_df = pd.DataFrame({'LipidMolec': ['S1']})
        with pytest.raises(ValueError, match="LipidMolec"):
            StandardsService.remove_standards_from_dataset(df, standards_df)

    def test_remove_missing_lipidmolec_standards(self, basic_lipid_df):
        """Test missing LipidMolec in standards raises error."""
        standards_df = pd.DataFrame({'SomeCol': [1, 2]})
        with pytest.raises(ValueError, match="LipidMolec"):
            StandardsService.remove_standards_from_dataset(basic_lipid_df, standards_df)

    def test_remove_preserves_other_columns(self, df_with_deuterated_standards):
        """Test that removal preserves all columns."""
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)(d7)'],
        })

        filtered_df, _ = StandardsService.remove_standards_from_dataset(
            df_with_deuterated_standards, standards_df
        )

        assert 'ClassKey' in filtered_df.columns
        assert 'intensity[s1]' in filtered_df.columns


# =============================================================================
# Test: validate_standards
# =============================================================================

class TestValidateStandards:
    """Tests for validate_standards method."""

    def test_valid_standards_df(self, standards_only_df):
        """Test validation of valid standards DataFrame."""
        result = StandardsService.validate_standards(standards_only_df)

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.valid_standards_count == 3

    def test_none_standards_df(self):
        """Test validation of None DataFrame."""
        result = StandardsService.validate_standards(None)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_empty_standards_df(self):
        """Test validation of empty DataFrame."""
        result = StandardsService.validate_standards(pd.DataFrame())

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_missing_lipidmolec_column(self):
        """Test validation without LipidMolec column."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        result = StandardsService.validate_standards(df)

        assert not result.is_valid
        assert any('LipidMolec' in e for e in result.errors)

    def test_duplicate_standards_warning(self):
        """Test warning for duplicate standards."""
        df = pd.DataFrame({
            'LipidMolec': ['Standard1', 'Standard1', 'Standard2'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        result = StandardsService.validate_standards(df)

        assert result.is_valid
        assert any('duplicate' in w.lower() for w in result.warnings)

    def test_empty_lipidmolec_warning(self):
        """Test warning for empty LipidMolec values."""
        df = pd.DataFrame({
            'LipidMolec': ['Standard1', '', 'Standard2'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        result = StandardsService.validate_standards(df)

        assert result.is_valid
        assert len(result.warnings) > 0

    def test_null_lipidmolec_warning(self):
        """Test warning for null LipidMolec values."""
        df = pd.DataFrame({
            'LipidMolec': ['Standard1', None, 'Standard2'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        result = StandardsService.validate_standards(df)

        assert result.is_valid
        assert len(result.warnings) > 0

    def test_missing_classkey_warning(self):
        """Test warning for missing ClassKey values."""
        df = pd.DataFrame({
            'LipidMolec': ['Standard1', 'Standard2'],
            'ClassKey': ['PC', None],
        })
        result = StandardsService.validate_standards(df)

        assert result.is_valid
        assert any('ClassKey' in w for w in result.warnings)

    def test_no_intensity_columns_warning(self):
        """Test warning when no intensity columns."""
        df = pd.DataFrame({
            'LipidMolec': ['Standard1', 'Standard2'],
            'ClassKey': ['PC', 'PE'],
        })
        result = StandardsService.validate_standards(df)

        assert result.is_valid
        assert any('intensity' in w.lower() for w in result.warnings)

    def test_check_existence_valid(self, df_with_deuterated_standards):
        """Test checking existence in cleaned data - valid case."""
        # Extract standards that actually exist in the df
        extraction = StandardsService.extract_standards(df_with_deuterated_standards)
        result = StandardsService.validate_standards(
            extraction.standards_df,
            cleaned_df=df_with_deuterated_standards,
            check_existence=True
        )
        # Standards exist in the main data
        assert result.is_valid

    def test_check_existence_missing(self, basic_lipid_df):
        """Test checking existence when standards not in data."""
        standards_df = pd.DataFrame({
            'LipidMolec': ['NonExistent1', 'NonExistent2'],
            'ClassKey': ['PC', 'PE'],
        })
        result = StandardsService.validate_standards(
            standards_df,
            cleaned_df=basic_lipid_df,
            check_existence=True
        )

        assert not result.is_valid
        assert any('not found' in e.lower() for e in result.errors)

    def test_valid_standards_count_excludes_duplicates(self):
        """Test that valid_standards_count excludes duplicates."""
        df = pd.DataFrame({
            'LipidMolec': ['S1', 'S1', 'S2', 'S2', 'S3'],
            'ClassKey': ['PC'] * 5,
        })
        result = StandardsService.validate_standards(df)

        assert result.valid_standards_count == 3  # unique count


# =============================================================================
# Test: validate_intensity_columns
# =============================================================================

class TestValidateIntensityColumns:
    """Tests for validate_intensity_columns method."""

    def test_valid_intensity_columns(self, standards_only_df):
        """Test validation with matching intensity columns."""
        result = StandardsService.validate_intensity_columns(
            standards_only_df, ['s1', 's2']
        )

        assert result.is_valid

    def test_missing_intensity_columns(self, standards_only_df):
        """Test validation with missing intensity columns."""
        result = StandardsService.validate_intensity_columns(
            standards_only_df, ['s1', 's2', 's3', 's4', 's5']
        )

        assert not result.is_valid
        assert any('missing' in e.lower() for e in result.errors)

    def test_extra_intensity_columns_warning(self, multi_sample_df):
        """Test warning for extra intensity columns."""
        result = StandardsService.validate_intensity_columns(
            multi_sample_df, ['s1', 's2']
        )

        assert result.is_valid
        assert any('extra' in w.lower() for w in result.warnings)

    def test_empty_standards_df(self):
        """Test validation with empty DataFrame."""
        result = StandardsService.validate_intensity_columns(
            pd.DataFrame(), ['s1', 's2']
        )

        assert not result.is_valid

    def test_none_standards_df(self):
        """Test validation with None DataFrame."""
        result = StandardsService.validate_intensity_columns(
            None, ['s1', 's2']
        )

        assert not result.is_valid

    def test_empty_expected_samples(self, standards_only_df):
        """Test validation with empty expected samples."""
        result = StandardsService.validate_intensity_columns(
            standards_only_df, []
        )

        assert result.is_valid  # No columns expected, any df is valid


# =============================================================================
# Test: process_standards_file - Extract Mode
# =============================================================================

class TestProcessStandardsFileExtractMode:
    """Tests for process_standards_file in extract mode."""

    def test_extract_mode_basic(self, uploaded_standards_simple, df_with_deuterated_standards):
        """Test basic extraction mode processing."""
        result = StandardsService.process_standards_file(
            uploaded_standards_simple,
            df_with_deuterated_standards,
            standards_in_dataset=True
        )

        assert isinstance(result, StandardsProcessingResult)
        assert result.source_mode == 'extract'
        assert result.standards_count == 2

    def test_extract_mode_with_classkey(self, uploaded_standards_with_class, df_with_deuterated_standards):
        """Test extraction mode with ClassKey column."""
        result = StandardsService.process_standards_file(
            uploaded_standards_with_class,
            df_with_deuterated_standards,
            standards_in_dataset=True
        )

        assert result.standards_count == 2
        assert 'ClassKey' in result.standards_df.columns

    def test_extract_mode_removes_duplicates(self, df_with_deuterated_standards):
        """Test that duplicates are removed."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)(d7)', 'PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
        })
        result = StandardsService.process_standards_file(
            uploaded_df,
            df_with_deuterated_standards,
            standards_in_dataset=True
        )

        assert result.duplicates_removed == 1
        assert result.standards_count == 2

    def test_extract_mode_missing_standards_error(self, basic_lipid_df):
        """Test error when standards not found in dataset."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['NonExistent1', 'NonExistent2'],
        })

        with pytest.raises(ValueError, match="not found"):
            StandardsService.process_standards_file(
                uploaded_df,
                basic_lipid_df,
                standards_in_dataset=True
            )

    def test_extract_mode_preserves_intensity(self, df_with_deuterated_standards):
        """Test that intensity values from dataset are preserved."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)(d7)'],
        })
        result = StandardsService.process_standards_file(
            uploaded_df,
            df_with_deuterated_standards,
            standards_in_dataset=True
        )

        assert 'intensity[s1]' in result.standards_df.columns
        assert result.standards_df['intensity[s1]'].iloc[0] == 500.0


# =============================================================================
# Test: process_standards_file - Complete Mode
# =============================================================================

class TestProcessStandardsFileCompleteMode:
    """Tests for process_standards_file in complete mode."""

    def test_complete_mode_basic(self, uploaded_standards_complete, basic_lipid_df):
        """Test basic complete mode processing."""
        result = StandardsService.process_standards_file(
            uploaded_standards_complete,
            basic_lipid_df,
            standards_in_dataset=False
        )

        assert result.source_mode == 'complete'
        assert result.standards_count == 2

    def test_complete_mode_missing_intensity_error(self, basic_lipid_df):
        """Test error when intensity columns missing."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['Standard1'],
            'ClassKey': ['PC'],
            # No intensity columns
        })

        with pytest.raises(ValueError, match="intensity"):
            StandardsService.process_standards_file(
                uploaded_df,
                basic_lipid_df,
                standards_in_dataset=False
            )

    def test_complete_mode_wrong_sample_count_error(self, basic_lipid_df):
        """Test error when wrong number of intensity columns."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['Standard1'],
            'ClassKey': ['PC'],
            'intensity[s1]': [100.0],
            # Missing s2 and s3
        })

        with pytest.raises(ValueError, match="missing"):
            StandardsService.process_standards_file(
                uploaded_df,
                basic_lipid_df,
                standards_in_dataset=False
            )


# =============================================================================
# Test: process_standards_file - Edge Cases
# =============================================================================

class TestProcessStandardsFileEdgeCases:
    """Edge case tests for process_standards_file."""

    def test_empty_uploaded_file(self, basic_lipid_df):
        """Test error with empty uploaded file."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.process_standards_file(
                pd.DataFrame(),
                basic_lipid_df,
                standards_in_dataset=True
            )

    def test_none_uploaded_file(self, basic_lipid_df):
        """Test error with None uploaded file."""
        with pytest.raises(ValueError, match="empty"):
            StandardsService.process_standards_file(
                None,
                basic_lipid_df,
                standards_in_dataset=True
            )

    def test_empty_cleaned_df(self):
        """Test error with empty cleaned dataset."""
        uploaded_df = pd.DataFrame({'LipidMolec': ['S1']})
        with pytest.raises(ValueError, match="empty"):
            StandardsService.process_standards_file(
                uploaded_df,
                pd.DataFrame(),
                standards_in_dataset=True
            )

    def test_none_cleaned_df(self):
        """Test error with None cleaned dataset."""
        uploaded_df = pd.DataFrame({'LipidMolec': ['S1']})
        with pytest.raises(ValueError, match="empty"):
            StandardsService.process_standards_file(
                uploaded_df,
                None,
                standards_in_dataset=True
            )

    def test_no_intensity_columns_in_cleaned(self):
        """Test error when cleaned data has no intensity columns."""
        uploaded_df = pd.DataFrame({'LipidMolec': ['S1']})
        cleaned_df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'],
            'ClassKey': ['PC'],
        })
        with pytest.raises(ValueError, match="intensity"):
            StandardsService.process_standards_file(
                uploaded_df,
                cleaned_df,
                standards_in_dataset=True
            )


# =============================================================================
# Test: _standardize_columns
# =============================================================================

class TestStandardizeColumns:
    """Tests for _standardize_columns method."""

    def test_first_column_to_lipidmolec(self):
        """Test first column renamed to LipidMolec."""
        df = pd.DataFrame({'AnyName': ['Lipid1', 'Lipid2']})
        result = StandardsService._standardize_columns(df, ['intensity[s1]'])

        assert 'LipidMolec' in result.columns

    def test_second_column_classkey_if_nonnumeric(self):
        """Test second column becomes ClassKey if non-numeric."""
        df = pd.DataFrame({
            'Name': ['Lipid1', 'Lipid2'],
            'Class': ['PC', 'PE'],
        })
        result = StandardsService._standardize_columns(df, [])

        assert 'ClassKey' in result.columns
        assert result['ClassKey'].tolist() == ['PC', 'PE']

    def test_second_column_intensity_if_numeric(self):
        """Test second column becomes intensity if numeric."""
        df = pd.DataFrame({
            'Name': ['Lipid1', 'Lipid2'],
            'Value': [100.0, 200.0],
        })
        result = StandardsService._standardize_columns(df, ['intensity[s1]'])

        assert 'intensity[s1]' in result.columns

    def test_infer_classkey_from_lipidmolec(self):
        """Test ClassKey inferred from LipidMolec."""
        df = pd.DataFrame({'Name': ['PC(16:0_18:1)', 'PE(18:0)']})
        result = StandardsService._standardize_columns(df, [])

        assert result['ClassKey'].tolist() == ['PC', 'PE']

    def test_empty_columns_error(self):
        """Test error when DataFrame has no columns."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="no columns"):
            StandardsService._standardize_columns(df, [])

    def test_preserves_intensity_column_order(self):
        """Test that intensity columns are assigned in order."""
        df = pd.DataFrame({
            'Name': ['L1'],
            'V1': [100],
            'V2': [200],
            'V3': [300],
        })
        expected_cols = ['intensity[s1]', 'intensity[s2]', 'intensity[s3]']
        result = StandardsService._standardize_columns(df, expected_cols)

        for col in expected_cols:
            assert col in result.columns


# =============================================================================
# Test: get_available_standards
# =============================================================================

class TestGetAvailableStandards:
    """Tests for get_available_standards method."""

    def test_basic_standards_list(self, standards_only_df):
        """Test getting list of standards."""
        standards = StandardsService.get_available_standards(standards_only_df)

        assert len(standards) == 3
        assert 'PC(15:0_15:0)(d7)' in standards

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty list."""
        standards = StandardsService.get_available_standards(pd.DataFrame())
        assert standards == []

    def test_none_dataframe(self):
        """Test None DataFrame returns empty list."""
        standards = StandardsService.get_available_standards(None)
        assert standards == []

    def test_missing_lipidmolec_column(self):
        """Test missing LipidMolec returns empty list."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        standards = StandardsService.get_available_standards(df)
        assert standards == []

    def test_unique_standards_only(self):
        """Test that only unique standards are returned."""
        df = pd.DataFrame({
            'LipidMolec': ['S1', 'S1', 'S2'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        standards = StandardsService.get_available_standards(df)

        assert len(standards) == 2


# =============================================================================
# Test: get_standards_by_class
# =============================================================================

class TestGetStandardsByClass:
    """Tests for get_standards_by_class method."""

    def test_basic_grouping(self, standards_only_df):
        """Test basic grouping by class."""
        by_class = StandardsService.get_standards_by_class(standards_only_df)

        assert 'PC' in by_class
        assert 'PE' in by_class
        assert 'TG' in by_class
        assert 'PC(15:0_15:0)(d7)' in by_class['PC']

    def test_multiple_standards_per_class(self):
        """Test multiple standards in same class."""
        df = pd.DataFrame({
            'LipidMolec': ['PC_IS1', 'PC_IS2', 'PE_IS1'],
            'ClassKey': ['PC', 'PC', 'PE'],
        })
        by_class = StandardsService.get_standards_by_class(df)

        assert len(by_class['PC']) == 2
        assert len(by_class['PE']) == 1

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty dict."""
        by_class = StandardsService.get_standards_by_class(pd.DataFrame())
        assert by_class == {}

    def test_none_dataframe(self):
        """Test None DataFrame returns empty dict."""
        by_class = StandardsService.get_standards_by_class(None)
        assert by_class == {}

    def test_missing_lipidmolec(self):
        """Test missing LipidMolec returns empty dict."""
        df = pd.DataFrame({'ClassKey': ['PC', 'PE']})
        by_class = StandardsService.get_standards_by_class(df)
        assert by_class == {}

    def test_infer_classkey_if_missing(self):
        """Test ClassKey inferred if missing."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0)(d7)', 'PE(17:0)(d5)'],
        })
        by_class = StandardsService.get_standards_by_class(df)

        assert 'PC' in by_class
        assert 'PE' in by_class


# =============================================================================
# Test: get_classes_with_standards
# =============================================================================

class TestGetClassesWithStandards:
    """Tests for get_classes_with_standards method."""

    def test_basic_classes(self, standards_only_df):
        """Test getting classes with standards."""
        classes = StandardsService.get_classes_with_standards(standards_only_df)

        assert isinstance(classes, set)
        assert 'PC' in classes
        assert 'PE' in classes
        assert 'TG' in classes

    def test_empty_dataframe(self):
        """Test empty DataFrame returns empty set."""
        classes = StandardsService.get_classes_with_standards(pd.DataFrame())
        assert classes == set()


# =============================================================================
# Test: get_classes_without_standards
# =============================================================================

class TestGetClassesWithoutStandards:
    """Tests for get_classes_without_standards method."""

    def test_missing_classes(self, basic_lipid_df, standards_only_df):
        """Test finding classes without standards."""
        # standards_only_df has PC, PE, TG
        # basic_lipid_df also has PC, PE, TG
        missing = StandardsService.get_classes_without_standards(
            basic_lipid_df, standards_only_df
        )
        assert len(missing) == 0

    def test_all_classes_missing(self, basic_lipid_df):
        """Test when no standards exist."""
        missing = StandardsService.get_classes_without_standards(
            basic_lipid_df, pd.DataFrame()
        )
        assert 'PC' in missing
        assert 'PE' in missing
        assert 'TG' in missing

    def test_some_classes_missing(self, basic_lipid_df):
        """Test when some classes have standards."""
        standards_df = pd.DataFrame({
            'LipidMolec': ['PC_IS'],
            'ClassKey': ['PC'],
        })
        missing = StandardsService.get_classes_without_standards(
            basic_lipid_df, standards_df
        )
        assert 'PC' not in missing
        assert 'PE' in missing
        assert 'TG' in missing

    def test_empty_data_df(self):
        """Test with empty data DataFrame."""
        missing = StandardsService.get_classes_without_standards(
            pd.DataFrame(), pd.DataFrame({'LipidMolec': ['S1'], 'ClassKey': ['PC']})
        )
        assert missing == set()

    def test_missing_classkey_in_data(self):
        """Test when data has no ClassKey column."""
        df = pd.DataFrame({'LipidMolec': ['L1', 'L2']})
        missing = StandardsService.get_classes_without_standards(
            df, pd.DataFrame({'LipidMolec': ['S1'], 'ClassKey': ['PC']})
        )
        assert missing == set()


# =============================================================================
# Test: suggest_standards_for_classes
# =============================================================================

class TestSuggestStandardsForClasses:
    """Tests for suggest_standards_for_classes method."""

    def test_class_specific_suggestion(self, standards_only_df):
        """Test suggesting class-specific standards."""
        suggestions = StandardsService.suggest_standards_for_classes(
            standards_only_df, ['PC', 'PE', 'TG']
        )

        assert suggestions['PC'] == 'PC(15:0_15:0)(d7)'
        assert suggestions['PE'] == 'PE(17:0_17:0)(d5)'

    def test_no_standards_available(self):
        """Test when no standards available."""
        suggestions = StandardsService.suggest_standards_for_classes(
            pd.DataFrame(), ['PC', 'PE']
        )

        assert suggestions['PC'] is None
        assert suggestions['PE'] is None

    def test_class_without_specific_standard(self, standards_only_df):
        """Test class without matching standard."""
        suggestions = StandardsService.suggest_standards_for_classes(
            standards_only_df, ['PC', 'NewClass']
        )

        assert suggestions['PC'] == 'PC(15:0_15:0)(d7)'
        assert suggestions['NewClass'] is None

    def test_empty_target_classes(self, standards_only_df):
        """Test with empty target classes list."""
        suggestions = StandardsService.suggest_standards_for_classes(
            standards_only_df, []
        )
        assert suggestions == {}


# =============================================================================
# Test: create_default_mapping
# =============================================================================

class TestCreateDefaultMapping:
    """Tests for create_default_mapping method."""

    def test_basic_mapping(self, basic_lipid_df, standards_only_df):
        """Test creating default class-to-standard mapping."""
        mapping = StandardsService.create_default_mapping(
            basic_lipid_df, standards_only_df
        )

        assert 'PC' in mapping
        assert 'PE' in mapping
        assert 'TG' in mapping

    def test_uses_class_specific_when_available(self, basic_lipid_df, standards_only_df):
        """Test that class-specific standards are used."""
        mapping = StandardsService.create_default_mapping(
            basic_lipid_df, standards_only_df
        )

        assert mapping['PC'] == 'PC(15:0_15:0)(d7)'
        assert mapping['PE'] == 'PE(17:0_17:0)(d5)'
        assert mapping['TG'] == 'TG(15:0)(d9)'

    def test_fallback_for_missing_class(self, standards_only_df):
        """Test fallback to first standard for classes without specific standard."""
        data_df = pd.DataFrame({
            'LipidMolec': ['NewClass(16:0)'],
            'ClassKey': ['NewClass'],
        })
        mapping = StandardsService.create_default_mapping(
            data_df, standards_only_df
        )

        # Should use first available standard
        assert mapping['NewClass'] in StandardsService.get_available_standards(standards_only_df)

    def test_no_standards_raises_error(self, basic_lipid_df):
        """Test error when no standards available."""
        with pytest.raises(ValueError, match="No internal standards"):
            StandardsService.create_default_mapping(basic_lipid_df, pd.DataFrame())

    def test_none_standards_raises_error(self, basic_lipid_df):
        """Test error when standards df is None."""
        with pytest.raises(ValueError, match="No internal standards"):
            StandardsService.create_default_mapping(basic_lipid_df, None)

    def test_empty_data_df(self, standards_only_df):
        """Test with empty data DataFrame returns empty mapping."""
        mapping = StandardsService.create_default_mapping(
            pd.DataFrame(), standards_only_df
        )
        assert mapping == {}

    def test_data_without_classkey(self, standards_only_df):
        """Test data without ClassKey column."""
        data_df = pd.DataFrame({'LipidMolec': ['L1', 'L2']})
        mapping = StandardsService.create_default_mapping(
            data_df, standards_only_df
        )
        assert mapping == {}


# =============================================================================
# Test: count_standards
# =============================================================================

class TestCountStandards:
    """Tests for count_standards method."""

    def test_count_in_mixed_df(self, df_with_deuterated_standards):
        """Test counting standards in mixed DataFrame."""
        count = StandardsService.count_standards(df_with_deuterated_standards)
        assert count == 3

    def test_count_zero_when_none(self, basic_lipid_df):
        """Test count is zero when no standards."""
        count = StandardsService.count_standards(basic_lipid_df)
        assert count == 0

    def test_count_all_standards(self, standards_only_df):
        """Test count when all are standards."""
        count = StandardsService.count_standards(standards_only_df)
        assert count == 3

    def test_count_empty_df(self):
        """Test count with empty DataFrame."""
        count = StandardsService.count_standards(pd.DataFrame())
        assert count == 0

    def test_count_none_df(self):
        """Test count with None DataFrame."""
        count = StandardsService.count_standards(None)
        assert count == 0

    def test_count_missing_lipidmolec(self):
        """Test count when LipidMolec missing."""
        df = pd.DataFrame({'SomeCol': [1, 2, 3]})
        count = StandardsService.count_standards(df)
        assert count == 0


# =============================================================================
# Test: has_standards
# =============================================================================

class TestHasStandards:
    """Tests for has_standards method."""

    def test_true_when_standards_present(self, df_with_deuterated_standards):
        """Test returns True when standards present."""
        assert StandardsService.has_standards(df_with_deuterated_standards)

    def test_false_when_no_standards(self, basic_lipid_df):
        """Test returns False when no standards."""
        assert not StandardsService.has_standards(basic_lipid_df)

    def test_false_for_empty_df(self):
        """Test returns False for empty DataFrame."""
        assert not StandardsService.has_standards(pd.DataFrame())

    def test_false_for_none_df(self):
        """Test returns False for None DataFrame."""
        assert not StandardsService.has_standards(None)


# =============================================================================
# Test: _infer_class
# =============================================================================

class TestInferClass:
    """Tests for _infer_class helper method."""

    def test_standard_format(self):
        """Test inference from standard lipid format."""
        assert StandardsService._infer_class('PC(16:0_18:1)') == 'PC'
        assert StandardsService._infer_class('PE(18:0_20:4)') == 'PE'
        assert StandardsService._infer_class('TG(16:0_18:1_18:2)') == 'TG'

    def test_no_parentheses(self):
        """Test when no parentheses in name."""
        assert StandardsService._infer_class('Cholesterol') == 'Cholesterol'
        assert StandardsService._infer_class('SPLASH') == 'SPLASH'

    def test_na_value(self):
        """Test handling of NA/None values."""
        assert StandardsService._infer_class(None) == 'Unknown'
        assert StandardsService._infer_class(np.nan) == 'Unknown'

    def test_complex_names(self):
        """Test complex lipid names."""
        assert StandardsService._infer_class('SM(d18:1_16:0)') == 'SM'
        assert StandardsService._infer_class('Cer(d18:1_24:0)') == 'Cer'


# =============================================================================
# Test: Type Handling
# =============================================================================

class TestTypeHandling:
    """Tests for proper type handling."""

    def test_detect_with_string_intensities(self):
        """Test detection works when intensity values are strings."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'PC(15:0)(d7)'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': ['1000', '500'],  # strings
        })
        result = StandardsService.detect_standards(df)
        assert result.iloc[1]

    def test_extract_with_mixed_numeric_types(self):
        """Test extraction with mixed int/float."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(15:0)(d7)'],
            'ClassKey': ['PC', 'PC'],
            'intensity[s1]': [1000, 500.5],  # int and float
        })
        result = StandardsService.extract_standards(df)
        assert result.standards_count == 1

    def test_validate_with_object_dtype(self):
        """Test validation with object dtype columns."""
        df = pd.DataFrame({
            'LipidMolec': pd.Series(['S1', 'S2'], dtype='object'),
            'ClassKey': pd.Series(['PC', 'PE'], dtype='object'),
        })
        result = StandardsService.validate_standards(df)
        assert result.is_valid

    def test_process_with_integer_intensity(self, df_with_deuterated_standards):
        """Test processing when intensity values are integers."""
        uploaded_df = pd.DataFrame({
            'LipidMolec': ['PC(15:0_15:0)(d7)'],
        })
        # Convert to int
        df = df_with_deuterated_standards.copy()
        for col in df.columns:
            if col.startswith('intensity['):
                df[col] = df[col].astype(int)

        result = StandardsService.process_standards_file(
            uploaded_df, df, standards_in_dataset=True
        )
        assert result.standards_count == 1


# =============================================================================
# Test: Boundary Conditions
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_single_row_df(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(15:0)(d7)'],
            'ClassKey': ['PC'],
            'intensity[s1]': [500.0],
        })
        result = StandardsService.extract_standards(df)
        assert result.standards_count == 1
        assert result.data_df.empty

    def test_single_column_df_valid(self):
        """Test with single column (LipidMolec only)."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PC(15:0)(d7)'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert result.iloc[1]

    def test_large_number_of_standards(self):
        """Test with many standards."""
        lipids = [f'Lipid_{i}' for i in range(100)]
        standards = [f'Standard_{i}(d7)' for i in range(50)]
        all_items = lipids + standards

        df = pd.DataFrame({
            'LipidMolec': all_items,
            'ClassKey': ['PC'] * len(all_items),
        })

        count = StandardsService.count_standards(df)
        assert count == 50

    def test_very_long_lipid_names(self):
        """Test with very long lipid names."""
        long_name = 'PC(' + '_'.join(['16:0'] * 10) + ')(d7)'
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', long_name],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert result.iloc[1]

    def test_special_characters_in_names(self):
        """Test lipid names with special characters."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0/18:1)', 'PE(P-18:0/20:4)', 'SM(d18:1;2O)(d7)'],
            'ClassKey': ['PC', 'PE', 'SM'],
        })
        result = StandardsService.detect_standards(df)
        assert not result.iloc[0]
        assert not result.iloc[1]
        assert result.iloc[2]

    def test_unicode_in_names(self):
        """Test handling of unicode characters."""
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0)', 'PE(d7)α', 'TG(d9)β'],
            'ClassKey': ['PC', 'PE', 'TG'],
        })
        result = StandardsService.detect_standards(df)
        # Should still detect the d7/d9 patterns
        assert result.iloc[1]
        assert result.iloc[2]

    def test_whitespace_in_names(self):
        """Test handling of whitespace in names."""
        df = pd.DataFrame({
            'LipidMolec': [' PC(16:0) ', '  PC(15:0)(d7)  '],
            'ClassKey': ['PC', 'PC'],
        })
        result = StandardsService.detect_standards(df)
        assert result.iloc[1]  # Should still match


# =============================================================================
# Test: Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests for realistic workflows."""

    def test_full_workflow_auto_detection(self, df_with_deuterated_standards):
        """Test full workflow: detect -> extract -> validate."""
        # Step 1: Extract standards
        extraction = StandardsService.extract_standards(df_with_deuterated_standards)

        assert extraction.standards_count == 3
        assert len(extraction.data_df) == 4

        # Step 2: Validate extracted standards
        validation = StandardsService.validate_standards(extraction.standards_df)

        assert validation.is_valid
        assert validation.valid_standards_count == 3

        # Step 3: Get standards by class
        by_class = StandardsService.get_standards_by_class(extraction.standards_df)

        assert 'PC' in by_class
        assert 'PE' in by_class
        assert 'TG' in by_class

    def test_full_workflow_uploaded_standards(self, df_with_deuterated_standards):
        """Test full workflow with uploaded standards file."""
        # Simulate uploaded file
        uploaded_df = pd.DataFrame({
            'StandardName': ['PC(15:0_15:0)(d7)', 'PE(17:0_17:0)(d5)'],
        })

        # Step 1: Process uploaded file
        result = StandardsService.process_standards_file(
            uploaded_df,
            df_with_deuterated_standards,
            standards_in_dataset=True
        )

        assert result.standards_count == 2

        # Step 2: Validate
        validation = StandardsService.validate_standards(result.standards_df)
        assert validation.is_valid

        # Step 3: Remove from main dataset
        filtered_df, removed = StandardsService.remove_standards_from_dataset(
            df_with_deuterated_standards,
            result.standards_df
        )

        assert len(removed) == 2
        assert len(filtered_df) == 5

    def test_workflow_with_no_standards(self, basic_lipid_df):
        """Test workflow when no standards in data."""
        # Check for standards
        has_stds = StandardsService.has_standards(basic_lipid_df)
        assert not has_stds

        # Extract should return empty standards
        extraction = StandardsService.extract_standards(basic_lipid_df)
        assert extraction.standards_count == 0
        assert len(extraction.data_df) == 3

        # Classes without standards
        missing = StandardsService.get_classes_without_standards(
            basic_lipid_df,
            extraction.standards_df
        )
        assert 'PC' in missing
        assert 'PE' in missing
        assert 'TG' in missing

    def test_workflow_multiple_standards_per_class(self):
        """Test workflow with multiple standards per class."""
        df = pd.DataFrame({
            'LipidMolec': [
                'PC(16:0_18:1)',
                'PC(15:0)(d5)',
                'PC(16:0)(d7)',
                'PC(17:0)(d9)',
                'PE(18:0_20:4)',
            ],
            'ClassKey': ['PC', 'PC', 'PC', 'PC', 'PE'],
            'intensity[s1]': [1000, 400, 500, 600, 2000],
        })

        extraction = StandardsService.extract_standards(df)
        by_class = StandardsService.get_standards_by_class(extraction.standards_df)

        assert len(by_class['PC']) == 3
        assert extraction.standards_count == 3

        # Suggestions should use first available
        suggestions = StandardsService.suggest_standards_for_classes(
            extraction.standards_df, ['PC', 'PE']
        )
        assert suggestions['PC'] is not None
        assert suggestions['PE'] is None  # No PE standards


# =============================================================================
# Standards Column Standardization Edge Cases (Bug Fix Tests)
# =============================================================================

class TestStandardsColumnStandardization:
    """
    Edge case tests for _standardize_columns method.

    These tests verify that ClassKey columns are correctly identified when
    standardizing uploaded standards files. The logic needs to distinguish
    between ClassKey columns (non-numeric text) and intensity columns (numeric).
    """

    def test_standardize_with_classkey_column(self):
        """Test standardization when ClassKey column is present."""
        df = pd.DataFrame({
            'Standard Name': ['PC(15:0)(d7)', 'PE(17:0)(d5)'],
            'Class': ['PC', 'PE'],  # ClassKey column (non-numeric)
            'Sample1': [500.0, 600.0],
            'Sample2': [520.0, 620.0],
        })
        expected_cols = ['intensity[s1]', 'intensity[s2]']

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert 'intensity[s1]' in result.columns
        assert 'intensity[s2]' in result.columns
        assert result['ClassKey'].iloc[0] == 'PC'

    def test_standardize_without_classkey_infers_from_name(self):
        """Test that ClassKey is inferred from LipidMolec when not provided."""
        df = pd.DataFrame({
            'Standard Name': ['PC(15:0)(d7)', 'PE(17:0)(d5)'],
            'Sample1': [500.0, 600.0],  # Numeric - intensity column
            'Sample2': [520.0, 620.0],
        })
        expected_cols = ['intensity[s1]', 'intensity[s2]']

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        # ClassKey should be inferred from lipid name
        assert result['ClassKey'].iloc[0] == 'PC'
        assert result['ClassKey'].iloc[1] == 'PE'

    def test_standardize_classkey_with_short_names(self):
        """Test ClassKey detection with typical short class names (PC, PE, TG)."""
        df = pd.DataFrame({
            'Lipid': ['PC(d7)', 'PE(d5)', 'TG(d9)'],
            'Class': ['PC', 'PE', 'TG'],  # Short alphabetic strings
            'Intensity1': [100.0, 200.0, 300.0],
        })
        expected_cols = ['intensity[s1]']

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'ClassKey' in result.columns
        assert set(result['ClassKey'].unique()) == {'PC', 'PE', 'TG'}

    def test_standardize_numeric_second_column_not_classkey(self):
        """Test that numeric second column is treated as intensity, not ClassKey."""
        df = pd.DataFrame({
            'Lipid': ['PC(d7)', 'PE(d5)'],
            'Conc1': [100.0, 200.0],  # Numeric - should be intensity
            'Conc2': [110.0, 210.0],
        })
        expected_cols = ['intensity[s1]', 'intensity[s2]']

        result = StandardsService._standardize_columns(df, expected_cols)

        # ClassKey should be inferred, not taken from Conc1
        assert 'ClassKey' in result.columns
        assert result['ClassKey'].iloc[0] == 'PC'  # Inferred from lipid name
        assert 'intensity[s1]' in result.columns
        assert 'intensity[s2]' in result.columns

    def test_standardize_mixed_classkey_values(self):
        """Test ClassKey with mixed-length class names."""
        df = pd.DataFrame({
            'Lipid': ['PC(d7)', 'CerG1(d5)', 'SM(d9)', 'Cholesterol(d7)'],
            'Class': ['PC', 'CerG1', 'SM', 'Chol'],  # Various lengths
            'Value': [100.0, 200.0, 300.0, 400.0],
        })
        expected_cols = ['intensity[s1]']

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'ClassKey' in result.columns
        assert 'PC' in result['ClassKey'].values
        assert 'CerG1' in result['ClassKey'].values

    def test_standardize_empty_dataframe_raises(self):
        """Test that empty DataFrame with no columns raises error."""
        df = pd.DataFrame()
        expected_cols = ['intensity[s1]']

        with pytest.raises(ValueError, match="no columns"):
            StandardsService._standardize_columns(df, expected_cols)

    def test_standardize_single_column_infers_classkey(self):
        """Test standardization with only lipid name column."""
        df = pd.DataFrame({
            'Lipid': ['PC(d7)', 'PE(d5)'],
        })
        expected_cols = []

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'LipidMolec' in result.columns
        assert 'ClassKey' in result.columns
        assert result['ClassKey'].iloc[0] == 'PC'

    def test_standardize_classkey_column_named_classkey(self):
        """Test when second column is explicitly named 'ClassKey'."""
        df = pd.DataFrame({
            'Lipid': ['PC(d7)', 'PE(d5)'],
            'ClassKey': ['PC', 'PE'],  # Explicitly named
            'Sample1': [100.0, 200.0],
        })
        expected_cols = ['intensity[s1]']

        result = StandardsService._standardize_columns(df, expected_cols)

        assert 'ClassKey' in result.columns
        assert result['ClassKey'].iloc[0] == 'PC'
        # Should have 3 columns: LipidMolec, ClassKey, intensity[s1]
        assert len(result.columns) == 3


class TestMetabolomicsWorkbenchParsing:
    """
    Edge case tests for Metabolomics Workbench format parsing.

    These tests verify correct handling of the MW format which is read as
    text (not CSV) and has a special structure with markers.
    """

    def test_detect_mw_format_with_markers(self):
        """Test detection of Metabolomics Workbench format by markers."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        mw_text = """#METABOLOMICS WORKBENCH
SUBJECT_TYPE
MS_METABOLITE_DATA_START
Samples	Sample1	Sample2	Sample3
Factors	Control	Control	Treatment
PC(16:0)	1000	1100	1200
PE(18:0)	2000	2100	2200
MS_METABOLITE_DATA_END
"""
        result = FormatDetectionService.detect_format(mw_text)
        assert result == DataFormat.METABOLOMICS_WORKBENCH

    def test_detect_mw_format_markers_in_wrong_order(self):
        """Test that markers in wrong order are not detected as MW format."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        bad_text = """MS_METABOLITE_DATA_END
Some data
MS_METABOLITE_DATA_START
"""
        result = FormatDetectionService.detect_format(bad_text)
        assert result == DataFormat.UNKNOWN

    def test_detect_mw_format_missing_start_marker(self):
        """Test that missing start marker is not detected as MW format."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        bad_text = """#METABOLOMICS WORKBENCH
Samples	Sample1	Sample2
PC(16:0)	1000	1100
MS_METABOLITE_DATA_END
"""
        result = FormatDetectionService.detect_format(bad_text)
        assert result == DataFormat.UNKNOWN

    def test_detect_mw_format_missing_end_marker(self):
        """Test that missing end marker is not detected as MW format."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        bad_text = """MS_METABOLITE_DATA_START
Samples	Sample1	Sample2
PC(16:0)	1000	1100
"""
        result = FormatDetectionService.detect_format(bad_text)
        assert result == DataFormat.UNKNOWN

    def test_detect_mw_format_empty_string(self):
        """Test that empty string is not detected as MW format."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        result = FormatDetectionService.detect_format("")
        assert result == DataFormat.UNKNOWN

    def test_detect_mw_format_with_extra_content(self):
        """Test MW detection with additional metadata before and after markers."""
        from app.services.format_detection import FormatDetectionService, DataFormat

        mw_text = """#METABOLOMICS WORKBENCH
#STUDY_ID ST000001
#ANALYSIS_TYPE MS
#VERSION 1
MS_METABOLITE_DATA_START
Samples	S1	S2
Factors	A	B
Lipid1	100	200
MS_METABOLITE_DATA_END
#END
"""
        result = FormatDetectionService.detect_format(mw_text)
        assert result == DataFormat.METABOLOMICS_WORKBENCH
