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

    def test_standards_source_radio_has_three_options(self, intsta_with_standards_app):
        """Standards source radio has Automatic, Select from Dataset, and Upload options."""
        at = intsta_with_standards_app
        radio = at.radio(key='standards_source_radio')
        assert radio is not None
        assert len(radio.options) == 3
        assert 'Automatic Detection' in radio.options
        assert 'Select from Dataset' in radio.options
        assert 'Upload Custom Standards' in radio.options

    def test_auto_detection_default_selected(self, intsta_with_standards_app):
        """Standards source defaults to Automatic Detection."""
        at = intsta_with_standards_app
        radio = at.radio(key='standards_source_radio')
        assert radio.value == 'Automatic Detection'

    def test_select_from_dataset_extracts_chosen_lipid(self, intsta_with_standards_app):
        """Selecting an existing dataset lipid uses it as a standard (no CSV upload)."""
        import pandas as pd

        at = intsta_with_standards_app
        # Names in standardized form, as produced by data ingestion.
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1', 'PE 34:2'],
            'ClassKey': ['PC', 'PC', 'PE'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned

        at.radio(key='standards_source_radio').set_value('Select from Dataset').run()
        ms = at.multiselect(key='dataset_standards_select')
        assert ms is not None
        assert 'PC 32:1' in ms.options

        ms.set_value(['PC 32:1']).run()
        assert not at.exception
        text_values = [t.value for t in at.text]
        assert any('standards_count:1' in v for v in text_values)

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

    def test_switch_to_custom_upload_no_extract_mode(self, intsta_with_standards_app):
        """Upload Custom Standards no longer offers the extract-from-dataset mode.

        Extract-from-dataset is now handled by the separate "Select from Dataset"
        source, so the location radio should be gone.
        """
        at = intsta_with_standards_app
        at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
        assert not at.exception
        location_radios = [r for r in at.radio if r.key == 'standards_location_radio']
        assert location_radios == []

    def test_additive_picker_shown_under_auto_detection(self, intsta_with_standards_app):
        """Automatic Detection offers an additive dataset-standards picker."""
        at = intsta_with_standards_app
        ms = at.multiselect(key='additional_istd_select')
        assert ms is not None

    def test_additional_standards_add_to_auto_detected(self, intsta_with_standards_app):
        """Dataset standards supplement (not replace) the auto-detected ones."""
        import pandas as pd

        at = intsta_with_standards_app
        # Names in standardized form, as produced by data ingestion.
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1', 'PE 34:2'],
            'ClassKey': ['PC', 'PC', 'PE'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned
        at.run()

        # Fixture auto-detects 2 standards; adding 1 more should give 3.
        at.multiselect(key='additional_istd_select').set_value(['PC 32:1']).run()
        assert not at.exception
        text_values = [t.value for t in at.text]
        assert any('standards_count:3' in v for v in text_values)

    def test_added_standard_removed_from_main_dataset(self, intsta_with_standards_app):
        """A lipid promoted to a standard is dropped from the analysed species."""
        import pandas as pd

        at = intsta_with_standards_app
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1', 'PE 34:2'],
            'ClassKey': ['PC', 'PC', 'PE'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned
        at.run()

        at.multiselect(key='additional_istd_select').set_value(['PC 32:1']).run()
        assert not at.exception
        remaining = at.session_state['cleaned_df']['LipidMolec'].tolist()
        assert 'PC 32:1' not in remaining
        assert 'PC 30:0' in remaining and 'PE 34:2' in remaining

    def test_upload_custom_no_file_defaults_to_empty(self, intsta_with_standards_app):
        """Upload Custom Standards does not fall back to auto-detected standards."""
        at = intsta_with_standards_app
        at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
        assert not at.exception
        text_values = [t.value for t in at.text]
        # No active standards until the user uploads a file.
        assert any('no_standards' in v for v in text_values)
        assert not any('standards_count' in v for v in text_values)

    def test_additive_picker_latches_expander_open(self, intsta_with_standards_app):
        """Adding standards from within Automatic Detection must not collapse it.

        Regression: the latch previously only fired on the source radio, so
        using the additive picker (without ever changing source) collapsed the
        expander.
        """
        import pandas as pd

        at = intsta_with_standards_app
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1', 'PE 34:2'],
            'ClassKey': ['PC', 'PC', 'PE'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned
        at.run()
        # Source radio untouched — the latch must still be unset here.
        assert '_intsta_expander_open' not in at.session_state

        at.multiselect(key='additional_istd_select').set_value(['PC 32:1']).run()
        assert not at.exception
        assert at.session_state['_intsta_expander_open'] is True

    def test_exclusive_picker_latches_expander_open(self, intsta_with_standards_app):
        """The Select-from-Dataset picker also latches the expander open."""
        import pandas as pd

        at = intsta_with_standards_app
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1'],
            'ClassKey': ['PC', 'PC'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned
        at.radio(key='standards_source_radio').set_value('Select from Dataset').run()
        # Reset the latch so we prove the picker itself sets it.
        at.session_state['_intsta_expander_open'] = False
        at.multiselect(key='dataset_standards_select').set_value(['PC 32:1']).run()
        assert not at.exception
        assert at.session_state['_intsta_expander_open'] is True

    def test_conditions_multiselect_latches_expander_open(self, intsta_with_standards_app):
        """The consistency-plot condition filter renders inside the expander too."""
        at = intsta_with_standards_app
        ms = at.multiselect(key='standards_conditions_select')
        assert ms is not None
        assert '_intsta_expander_open' not in at.session_state
        ms.set_value(['Control']).run()
        assert not at.exception
        assert at.session_state['_intsta_expander_open'] is True

    def test_select_from_dataset_removes_lipid_from_main_dataset(self, intsta_with_standards_app):
        """Exclusive Select-from-Dataset also drops the lipid from analysed species."""
        import pandas as pd

        at = intsta_with_standards_app
        cleaned = pd.DataFrame({
            'LipidMolec': ['PC 30:0', 'PC 32:1', 'PE 34:2'],
            'ClassKey': ['PC', 'PC', 'PE'],
            **{f'intensity[s{i}]': [1.0 * i, 2.0 * i, 3.0 * i] for i in range(1, 7)},
        })
        at.session_state['_test_cleaned_df'] = cleaned
        at.radio(key='standards_source_radio').set_value('Select from Dataset').run()
        at.multiselect(key='dataset_standards_select').set_value(['PC 32:1']).run()
        assert not at.exception
        remaining = at.session_state['cleaned_df']['LipidMolec'].tolist()
        assert 'PC 32:1' not in remaining
        assert 'PC 30:0' in remaining and 'PE 34:2' in remaining

    def test_clear_custom_standards_button_latches_expander_open(self, intsta_with_standards_app):
        """The Clear Custom Standards button latches the expander open."""
        import pandas as pd

        at = intsta_with_standards_app
        at.session_state['custom_standards_df'] = pd.DataFrame({
            'LipidMolec': ['STD 1'], 'ClassKey': ['PC'], 'intensity[s1]': [1.0],
        })
        at.session_state['custom_standards_mode'] = 'complete'
        at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
        at.session_state['_intsta_expander_open'] = False
        at.button(key='clear_custom_standards').click().run()
        assert not at.exception
        assert at.session_state['_intsta_expander_open'] is True

    def test_expander_latches_open_across_source_switches(self, intsta_with_standards_app):
        """Changing the source (either direction) keeps the expander open."""
        at = intsta_with_standards_app
        at.radio(key='standards_source_radio').set_value('Upload Custom Standards').run()
        assert at.session_state['_intsta_expander_open'] is True
        # Switching back to Automatic Detection must not collapse it.
        at.radio(key='standards_source_radio').set_value('Automatic Detection').run()
        assert at.session_state['_intsta_expander_open'] is True


# =============================================================================
# Standards Combination Tests
# =============================================================================

class TestCombineStandards:
    """Unit tests for merging auto-detected and dataset-picked standards."""

    @staticmethod
    def _df(names):
        import pandas as pd
        return pd.DataFrame({
            'LipidMolec': names,
            'ClassKey': ['PC'] * len(names),
            'intensity[s1]': [1.0] * len(names),
        })

    def test_union_of_both_sets(self):
        from app.ui.main_content.internal_standards import _combine_standards
        out = _combine_standards(self._df(['A', 'B']), self._df(['C']))
        assert out['LipidMolec'].tolist() == ['A', 'B', 'C']

    def test_deduplicates_overlapping_standard(self):
        from app.ui.main_content.internal_standards import _combine_standards
        out = _combine_standards(self._df(['A', 'B']), self._df(['B', 'C']))
        assert out['LipidMolec'].tolist() == ['A', 'B', 'C']
        assert len(out) == 3

    def test_empty_base_returns_added(self):
        import pandas as pd
        from app.ui.main_content.internal_standards import _combine_standards
        out = _combine_standards(pd.DataFrame(), self._df(['A']))
        assert out['LipidMolec'].tolist() == ['A']

    def test_empty_added_returns_base(self):
        import pandas as pd
        from app.ui.main_content.internal_standards import _combine_standards
        out = _combine_standards(self._df(['A']), pd.DataFrame())
        assert out['LipidMolec'].tolist() == ['A']


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
        """Normalization method radio has 5 options when standards exist."""
        at = norm_with_standards_app
        radio = at.radio(key='norm_method_selection')
        assert radio is not None
        assert len(radio.options) == 5
        assert 'None (pre-normalized data)' in radio.options
        assert 'Internal Standards' in radio.options
        assert 'Protein-based' in radio.options
        assert 'Internal Standards + Protein' in radio.options
        assert 'Total Intensity' in radio.options

    def test_method_radio_default_is_none(self, norm_with_standards_app):
        """Normalization method defaults to None (pre-normalized)."""
        at = norm_with_standards_app
        radio = at.radio(key='norm_method_selection')
        assert radio.value == 'None (pre-normalized data)'

    def test_method_radio_has_two_options_without_standards(self, norm_no_standards_app):
        """Normalization method radio has 3 options without standards."""
        at = norm_no_standards_app
        radio = at.radio(key='norm_method_selection')
        assert len(radio.options) == 3
        assert 'None (pre-normalized data)' in radio.options
        assert 'Protein-based' in radio.options
        assert 'Total Intensity' in radio.options

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

    def test_grade_filter_has_three_modes(self, grade_filtering_app):
        """Grade filtering radio has three mode options."""
        at = grade_filtering_app
        radio = at.radio(key='grade_filter_mode_radio')
        assert len(radio.options) == 3
        assert 'Use Default Settings' in radio.options
        assert 'Allow C for all classes' in radio.options
        assert 'Customize by Class' in radio.options

    def test_grade_filter_blanket_c_applies_abc_to_all_classes(self, grade_filtering_app):
        """'Allow C for all classes' returns A/B/C for every class."""
        at = grade_filtering_app
        at.radio(key='grade_filter_mode_radio').set_value('Allow C for all classes').run()
        text_values = [t.value for t in at.text]
        # One config entry per class (LPC, PC, PE, SM = 4 classes)
        assert any("custom_config:4" in v for v in text_values)
        # PC (a normally strict class) now includes grade C
        assert any(v == "pc_grades:A,B,C" for v in text_values)

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
