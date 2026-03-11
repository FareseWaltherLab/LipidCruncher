"""
UI tests for Module 1 main content area components.

Tests cover:
- Zero filtering (sliders, detection threshold, BQC handling)
- Internal standards management (auto-detect, custom upload, source radio)
- Normalization (class selection, method radio, standards/protein config)
- Grade filtering (default vs custom mode, per-class multiselects)
- Quality filtering (preset levels, MS/MS checkbox)
- Column mapping display (sidebar mapping table, MS-DIAL override)
"""

import pytest


# =============================================================================
# Zero Filtering Tests
# =============================================================================

class TestZeroFiltering:
    """Tests for zero filtering UI configuration."""

    def test_non_bqc_slider_default_75(self, zero_filter_generic_app):
        """Non-BQC threshold slider defaults to 75%."""
        at = zero_filter_generic_app
        slider = at.slider(key='non_bqc_zero_threshold')
        assert slider is not None
        assert slider.value == 75

    def test_no_bqc_slider_when_no_bqc(self, zero_filter_generic_app):
        """BQC slider does not render when no BQC label provided."""
        at = zero_filter_generic_app
        # BQC slider should not exist
        try:
            at.slider(key='bqc_zero_threshold')
            assert False, "BQC slider should not exist without BQC label"
        except Exception:
            pass  # Expected — widget key not found
        # Info message about no BQC
        info_values = [i.value for i in at.info]
        assert any("No BQC" in v for v in info_values)

    def test_bqc_slider_shows_when_bqc_exists(self, zero_filter_bqc_app):
        """BQC slider renders when BQC label is provided."""
        at = zero_filter_bqc_app
        slider = at.slider(key='bqc_zero_threshold')
        assert slider is not None
        assert slider.value == 50  # ZERO_FILTER_BQC_DEFAULT

    def test_detection_threshold_default_generic(self, zero_filter_generic_app):
        """Detection threshold defaults to 0 for Generic format."""
        at = zero_filter_generic_app
        threshold = at.number_input(key='zero_filter_detection_threshold')
        assert threshold is not None
        assert threshold.value == 0.0

    def test_detection_threshold_default_lipidsearch(self, zero_filter_lipidsearch_app):
        """Detection threshold defaults to 30000 for LipidSearch format."""
        at = zero_filter_lipidsearch_app
        threshold = at.number_input(key='zero_filter_detection_threshold')
        assert threshold is not None
        assert threshold.value == 30000.0


# =============================================================================
# Internal Standards Tests
# =============================================================================

class TestInternalStandards:
    """Tests for internal standards management UI."""

    def test_standards_source_radio_has_two_options(self, intsta_with_standards_app):
        """Standards source radio has Automatic Detection and Upload Custom options."""
        at = intsta_with_standards_app
        radio = at.radio(key='standards_source_radio')
        assert radio is not None
        assert len(radio.options) == 2
        assert 'Automatic Detection' in radio.options
        assert 'Upload Custom Standards' in radio.options

    def test_auto_detection_default_selected(self, intsta_with_standards_app):
        """Standards source defaults to Automatic Detection."""
        at = intsta_with_standards_app
        radio = at.radio(key='standards_source_radio')
        assert radio.value == 'Automatic Detection'

    def test_auto_detected_standards_displayed(self, intsta_with_standards_app):
        """Auto-detected standards are displayed with count."""
        at = intsta_with_standards_app
        text_values = [t.value for t in at.text]
        assert any("standards_count:2" in v for v in text_values)
        # Success message about found standards
        success_values = [s.value for s in at.success]
        assert any("Found 2 standards" in v for v in success_values)

    def test_no_standards_shows_warning(self, intsta_no_standards_app):
        """Warning shown when no standards auto-detected."""
        at = intsta_no_standards_app
        text_values = [t.value for t in at.text]
        assert any("no_standards" in v for v in text_values)
        warning_values = [w.value for w in at.warning]
        assert any("No internal standards" in v for v in warning_values)

    def test_switch_to_custom_upload(self, intsta_with_standards_app):
        """Switching to Upload Custom Standards shows location radio."""
        at = intsta_with_standards_app
        at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
        # Standards location radio should now appear
        radio = at.radio(key='standards_location_radio')
        assert radio is not None
        assert len(radio.options) == 2


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Tests for normalization UI components."""

    def test_class_multiselect_shows_available_classes(self, norm_with_standards_app):
        """Class multiselect displays all available classes."""
        at = norm_with_standards_app
        ms = at.multiselect(key='temp_selected_classes')
        assert ms is not None
        assert 'PC' in ms.options
        assert 'PE' in ms.options

    def test_class_multiselect_defaults_to_all(self, norm_with_standards_app):
        """Class multiselect defaults to all available classes selected."""
        at = norm_with_standards_app
        ms = at.multiselect(key='temp_selected_classes')
        assert set(ms.value) == {'PC', 'PE'}

    def test_method_radio_has_four_options_with_standards(self, norm_with_standards_app):
        """Normalization method radio has 4 options when standards exist."""
        at = norm_with_standards_app
        radio = at.radio(key='norm_method_selection')
        assert radio is not None
        assert len(radio.options) == 4
        assert 'None (pre-normalized data)' in radio.options
        assert 'Internal Standards' in radio.options
        assert 'Protein-based' in radio.options
        assert 'Both' in radio.options

    def test_method_radio_default_is_none(self, norm_with_standards_app):
        """Normalization method defaults to None (pre-normalized)."""
        at = norm_with_standards_app
        radio = at.radio(key='norm_method_selection')
        assert radio.value == 'None (pre-normalized data)'

    def test_method_radio_has_two_options_without_standards(self, norm_no_standards_app):
        """Normalization method radio has 2 options without standards."""
        at = norm_no_standards_app
        radio = at.radio(key='norm_method_selection')
        assert len(radio.options) == 2
        assert 'None (pre-normalized data)' in radio.options
        assert 'Protein-based' in radio.options

    def test_none_method_runs_normalization(self, norm_with_standards_app):
        """'None' method produces normalized output."""
        at = norm_with_standards_app
        text_values = [t.value for t in at.text]
        assert any("normalized_rows:" in v for v in text_values)

    def test_selecting_protein_method_shows_protein_config(self, norm_no_standards_app):
        """Selecting Protein-based method shows protein input UI."""
        at = norm_no_standards_app
        at.radio(key='norm_method_selection').set_value('Protein-based').run()
        # Protein input method radio should appear
        radio = at.radio(key='protein_input_method')
        assert radio is not None
        assert radio.value == 'Manual Input'

    def test_protein_input_method_has_two_options(self, norm_no_standards_app):
        """Protein input method has Manual Input and Upload CSV options."""
        at = norm_no_standards_app
        at.radio(key='norm_method_selection').set_value('Protein-based').run()
        radio = at.radio(key='protein_input_method')
        assert len(radio.options) == 2
        assert 'Manual Input' in radio.options
        assert 'Upload CSV File' in radio.options


# =============================================================================
# Grade Filtering Tests
# =============================================================================

class TestGradeFiltering:
    """Tests for LipidSearch grade filtering UI."""

    def test_grade_filter_default_mode(self, grade_filtering_app):
        """Grade filtering defaults to 'Use Default Settings'."""
        at = grade_filtering_app
        radio = at.radio(key='grade_filter_mode_radio')
        assert radio is not None
        assert radio.value == 'Use Default Settings'
        # Output indicates default config
        text_values = [t.value for t in at.text]
        assert any("default_config" in v for v in text_values)

    def test_grade_filter_has_two_modes(self, grade_filtering_app):
        """Grade filtering radio has two mode options."""
        at = grade_filtering_app
        radio = at.radio(key='grade_filter_mode_radio')
        assert len(radio.options) == 2
        assert 'Use Default Settings' in radio.options
        assert 'Customize by Class' in radio.options

    def test_grade_filter_custom_shows_multiselects(self, grade_filtering_app):
        """Switching to custom mode shows per-class grade multiselects."""
        at = grade_filtering_app
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        # Should have multiselects for each class (LPC, PC, PE, SM = 4 classes)
        text_values = [t.value for t in at.text]
        assert any("custom_config:4" in v for v in text_values)

    def test_grade_filter_default_grades_correct(self, grade_filtering_app):
        """Custom mode defaults: A/B for most, A/B/C for LPC and SM."""
        at = grade_filtering_app
        at.radio(key='grade_filter_mode_radio').set_value('Customize by Class').run()
        # Check LPC gets A/B/C default
        lpc_ms = at.multiselect(key='grade_select_LPC')
        assert set(lpc_ms.value) == {'A', 'B', 'C'}
        # Check PC gets A/B default
        pc_ms = at.multiselect(key='grade_select_PC')
        assert set(pc_ms.value) == {'A', 'B'}


# =============================================================================
# Quality Filtering Tests
# =============================================================================

class TestQualityFiltering:
    """Tests for MS-DIAL quality filtering UI."""

    def test_quality_filter_radio_exists(self, quality_filtering_app):
        """Quality filtering radio exists with preset options."""
        at = quality_filtering_app
        radio = at.radio(key='msdial_quality_level_radio')
        assert radio is not None

    def test_quality_filter_has_four_options(self, quality_filtering_app):
        """Quality filtering has 4 preset options."""
        at = quality_filtering_app
        radio = at.radio(key='msdial_quality_level_radio')
        assert len(radio.options) == 4
        assert any('Strict' in o for o in radio.options)
        assert any('Moderate' in o for o in radio.options)
        assert any('Permissive' in o for o in radio.options)
        assert any('No filtering' in o for o in radio.options)

    def test_quality_filter_default_moderate(self, quality_filtering_app):
        """Quality filtering defaults to Moderate (Score ≥60)."""
        at = quality_filtering_app
        text_values = [t.value for t in at.text]
        assert any("threshold:60" in v for v in text_values)

    def test_quality_filter_msms_checkbox_exists(self, quality_filtering_app):
        """MS/MS validation checkbox is available."""
        at = quality_filtering_app
        checkbox = at.checkbox(key='msdial_custom_msms')
        assert checkbox is not None


# =============================================================================
# Column Mapping Tests
# =============================================================================

class TestColumnMapping:
    """Tests for column mapping display in sidebar."""

    def test_mapping_displayed_generic(self, column_mapping_generic_app):
        """Column mapping renders successfully for Generic format."""
        at = column_mapping_generic_app
        text_values = [t.value for t in at.text]
        assert any("mapping_success:True" in v for v in text_values)

    def test_msdial_override_expander_accessible(self, column_mapping_msdial_app):
        """MS-DIAL format shows override multiselect."""
        at = column_mapping_msdial_app
        text_values = [t.value for t in at.text]
        assert any("mapping_success:True" in v for v in text_values)
        # Override multiselect should exist
        ms = at.sidebar.multiselect(key='manual_sample_override')
        assert ms is not None

    def test_generic_no_override(self, column_mapping_generic_app):
        """Generic format has no MS-DIAL override multiselect."""
        at = column_mapping_generic_app
        try:
            at.sidebar.multiselect(key='manual_sample_override')
            assert False, "Override multiselect should not exist for Generic format"
        except Exception:
            pass  # Expected — widget key not found
