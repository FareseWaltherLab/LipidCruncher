"""Unit tests for non-lipid constants in app/constants.py.

Covers: routing enums, format mappings, internal standard patterns,
threshold constants, and resolve_format_enum.
"""
import re

import pytest
from app.constants import (
    # Routing enums
    Module,
    Page,
    PAGE_LANDING,
    PAGE_APP,
    MODULE_DATA_PROCESSING,
    MODULE_QC_ANALYSIS,
    # Format constants
    FORMAT_GENERIC,
    FORMAT_LIPIDSEARCH,
    FORMAT_METABOLOMICS_WORKBENCH,
    FORMAT_MSDIAL,
    FORMAT_OPTIONS,
    # Mappings
    get_format_display_to_enum,
    resolve_format_enum,
    # Internal standard patterns
    INTERNAL_STANDARD_CLASS_PATTERN,
    INTERNAL_STANDARD_LIPID_PATTERNS,
    # Thresholds
    COV_THRESHOLD_DEFAULT,
    LIPIDSEARCH_DETECTION_THRESHOLD,
    ZERO_FILTER_BQC_DEFAULT,
    ZERO_FILTER_NON_BQC_DEFAULT,
    # Grade filtering
    LIPIDSEARCH_DEFAULT_GRADES,
    LIPIDSEARCH_GRADE_OPTIONS,
    LIPIDSEARCH_RELAXED_GRADE_CLASSES,
    LIPIDSEARCH_RELAXED_GRADES,
    # Plotting
    CONDITION_COLORS,
)
from app.services.format_detection import DataFormat


# =============================================================================
# Routing Enums
# =============================================================================


class TestPageEnum:
    """Validate Page enum."""

    def test_values(self):
        assert Page.LANDING == "landing"
        assert Page.APP == "app"

    def test_is_str(self):
        assert isinstance(Page.LANDING, str)
        assert isinstance(Page.APP, str)

    def test_legacy_aliases(self):
        assert PAGE_LANDING == Page.LANDING
        assert PAGE_APP == Page.APP

    def test_membership(self):
        assert "landing" in [p.value for p in Page]
        assert "app" in [p.value for p in Page]

    def test_equality_with_str(self):
        assert Page.LANDING == "landing"
        assert Page.APP == "app"


class TestModuleEnum:
    """Validate Module enum."""

    def test_values(self):
        assert Module.DATA_PROCESSING == "Data Cleaning, Filtering, & Normalization"
        assert Module.QC_ANALYSIS == "Quality Check & Analysis"

    def test_is_str(self):
        assert isinstance(Module.DATA_PROCESSING, str)

    def test_legacy_aliases(self):
        assert MODULE_DATA_PROCESSING == Module.DATA_PROCESSING
        assert MODULE_QC_ANALYSIS == Module.QC_ANALYSIS


# =============================================================================
# Format Constants
# =============================================================================


class TestFormatConstants:
    """Validate format display strings and options."""

    def test_format_options_contains_all_formats(self):
        assert FORMAT_GENERIC in FORMAT_OPTIONS
        assert FORMAT_LIPIDSEARCH in FORMAT_OPTIONS
        assert FORMAT_MSDIAL in FORMAT_OPTIONS
        assert FORMAT_METABOLOMICS_WORKBENCH in FORMAT_OPTIONS

    def test_format_options_is_tuple(self):
        assert isinstance(FORMAT_OPTIONS, tuple)

    def test_format_options_length(self):
        assert len(FORMAT_OPTIONS) == 4

    def test_format_options_order(self):
        """Generic is first (default), specific formats follow."""
        assert FORMAT_OPTIONS[0] == FORMAT_GENERIC


# =============================================================================
# Format Mappings
# =============================================================================


class TestFormatMappings:
    """Validate format display → enum mappings."""

    def test_get_format_display_to_enum_all_formats(self):
        mapping = get_format_display_to_enum()
        assert mapping[FORMAT_LIPIDSEARCH] == DataFormat.LIPIDSEARCH
        assert mapping[FORMAT_MSDIAL] == DataFormat.MSDIAL
        assert mapping[FORMAT_GENERIC] == DataFormat.GENERIC
        assert mapping[FORMAT_METABOLOMICS_WORKBENCH] == DataFormat.METABOLOMICS_WORKBENCH

    def test_get_format_display_to_enum_length(self):
        mapping = get_format_display_to_enum()
        assert len(mapping) == 4

    def test_resolve_format_enum_from_string(self):
        assert resolve_format_enum('LipidSearch 5.0') == DataFormat.LIPIDSEARCH
        assert resolve_format_enum('MS-DIAL') == DataFormat.MSDIAL
        assert resolve_format_enum('Generic Format') == DataFormat.GENERIC

    def test_resolve_format_enum_passthrough(self):
        assert resolve_format_enum(DataFormat.LIPIDSEARCH) == DataFormat.LIPIDSEARCH
        assert resolve_format_enum(DataFormat.MSDIAL) == DataFormat.MSDIAL

    def test_resolve_format_enum_unknown_falls_back(self):
        assert resolve_format_enum('SomeUnknownFormat') == DataFormat.GENERIC


# =============================================================================
# Internal Standard Patterns
# =============================================================================


class TestInternalStandardPatterns:
    """Validate internal standard regex patterns detect expected markers."""

    def test_patterns_is_tuple(self):
        assert isinstance(INTERNAL_STANDARD_LIPID_PATTERNS, tuple)

    @pytest.mark.parametrize("lipid_name", [
        "PC 16:0_18:1(d5)",    # Deuterium label
        "PE 18:0_20:4(d7)",    # Deuterium label
        "Cer+D7 18:1/24:0",    # +D7 pattern
        "LPC-D7 18:1",         # -D7 pattern  (won't match -d\d+ since D is uppercase)
        "LPC-d7_18:1",         # Alternative deuterium
        "SM[d7] 18:1",         # Bracketed deuterium
        "Ch-D7",               # Cholesterol deuterated
        "PC ISTD 16:0",        # ISTD marker
        "SPLASH mix",          # SPLASH marker
        "PC 16:0(IS)",         # (IS) marker
        "PE 18:1_IS",          # _IS suffix
    ])
    def test_detects_known_standards(self, lipid_name):
        """Each known standard name matches at least one pattern."""
        matched = any(
            re.search(pattern, lipid_name)
            for pattern in INTERNAL_STANDARD_LIPID_PATTERNS
        )
        assert matched, f"No pattern matched '{lipid_name}'"

    @pytest.mark.parametrize("lipid_name", [
        "PC 16:0_18:1",
        "PE 34:1",
        "Cer 18:1;O2/24:0",
        "TG 16:0_18:1_18:2",
    ])
    def test_does_not_match_regular_lipids(self, lipid_name):
        """Regular lipid names should not be flagged as standards."""
        matched = any(
            re.search(pattern, lipid_name)
            for pattern in INTERNAL_STANDARD_LIPID_PATTERNS
        )
        assert not matched, f"Pattern incorrectly matched '{lipid_name}'"

    def test_class_pattern_matches_istd(self):
        assert re.search(INTERNAL_STANDARD_CLASS_PATTERN, "ISTD")

    def test_class_pattern_matches_internal(self):
        assert re.search(INTERNAL_STANDARD_CLASS_PATTERN, "Internal Standard")

    def test_class_pattern_no_match_regular(self):
        assert not re.search(INTERNAL_STANDARD_CLASS_PATTERN, "PC")


# =============================================================================
# Threshold Constants
# =============================================================================


class TestThresholds:
    """Validate threshold constants have sensible values."""

    def test_lipidsearch_threshold_positive(self):
        assert LIPIDSEARCH_DETECTION_THRESHOLD > 0

    def test_cov_threshold_in_range(self):
        assert 0 < COV_THRESHOLD_DEFAULT <= 100

    def test_zero_filter_defaults_in_range(self):
        assert 0 < ZERO_FILTER_NON_BQC_DEFAULT <= 100
        assert 0 < ZERO_FILTER_BQC_DEFAULT <= 100

    def test_bqc_stricter_than_non_bqc(self):
        """BQC samples should have a stricter (lower) threshold."""
        assert ZERO_FILTER_BQC_DEFAULT < ZERO_FILTER_NON_BQC_DEFAULT


# =============================================================================
# Grade Filtering
# =============================================================================


class TestGradeFiltering:
    """Validate LipidSearch grade filtering constants."""

    def test_grade_options_is_tuple(self):
        assert isinstance(LIPIDSEARCH_GRADE_OPTIONS, tuple)

    def test_all_grades_present(self):
        assert set(LIPIDSEARCH_GRADE_OPTIONS) == {'A', 'B', 'C', 'D'}

    def test_default_grades_subset_of_options(self):
        assert set(LIPIDSEARCH_DEFAULT_GRADES).issubset(set(LIPIDSEARCH_GRADE_OPTIONS))

    def test_relaxed_grades_superset_of_default(self):
        assert set(LIPIDSEARCH_DEFAULT_GRADES).issubset(set(LIPIDSEARCH_RELAXED_GRADES))

    def test_relaxed_classes_non_empty(self):
        assert len(LIPIDSEARCH_RELAXED_GRADE_CLASSES) > 0


# =============================================================================
# Plotting Constants
# =============================================================================


class TestPlottingConstants:
    """Validate plotting constants."""

    def test_condition_colors_is_tuple(self):
        assert isinstance(CONDITION_COLORS, tuple)

    def test_condition_colors_non_empty(self):
        assert len(CONDITION_COLORS) >= 10

    def test_condition_colors_are_hex(self):
        for color in CONDITION_COLORS:
            assert re.match(r'^#[0-9a-fA-F]{6}$', color), f"Invalid hex color: {color}"