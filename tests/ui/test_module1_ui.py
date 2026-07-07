"""
UI tests for Module 1: Filter and Normalize.

Tests use Streamlit's AppTest framework with wrapper functions from conftest.py
to avoid importing main_app.py (which has st.set_page_config() at module level).

Tests across 10 groups covering:
1. Landing page navigation
2. Format selection
3. Sample data loading
4. Experiment definition
5. Confirm inputs / BQC
6. MS-DIAL data type selection
7. MS-DIAL override preservation (regression for d9451cf)
8. Back to Home navigation
9. Sample display names (sidebar-only feature)
10. MS-DIAL override + sample names (stale-names regression)
11. Regroup + confirm with sample names (blank-selection regression)
"""

from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    landing_page_script,
    format_and_upload_script,
    full_sidebar_script,
    app_page_script,
    msdial_data_type_script,
    override_preservation_script,
    override_reset_script,
)


# =============================================================================
# Group 1: Landing Page Navigation (3 tests)
# =============================================================================

class TestLandingPageNavigation:
    """Tests for the landing page and Start Crunching button."""

    def test_start_crunching_button_renders(self, landing_app):
        """Start Crunching button is visible on landing page."""
        at = landing_app
        assert len(at.button) >= 1
        assert "Start Crunching" in at.button[0].label

    def test_start_crunching_sets_page_to_app(self, landing_app):
        """Clicking Start Crunching navigates to app page."""
        at = landing_app
        at.button[0].click().run()
        assert at.session_state['page'] == 'app'

    def test_landing_page_initial_state(self, landing_app):
        """Landing page starts with page='landing'."""
        at = landing_app
        assert at.session_state['page'] == 'landing'


# =============================================================================
# Group 2: Format Selection (3 tests)
# =============================================================================

class TestFormatSelection:
    """Tests for the format selectbox in the sidebar."""

    def test_format_selectbox_has_four_options(self, format_upload_app):
        """Format selectbox offers all 4 supported formats."""
        at = format_upload_app
        options = at.sidebar.selectbox[0].options
        assert options == ['Generic Format', 'Metabolomics Workbench', 'LipidSearch', 'MS-DIAL']

    def test_format_default_is_generic(self, format_upload_app):
        """Default format selection is Generic."""
        at = format_upload_app
        assert at.sidebar.selectbox[0].value == 'Generic Format'

    def test_format_can_switch_to_msdial(self, format_upload_app):
        """User can switch format to MS-DIAL."""
        at = format_upload_app
        at.sidebar.selectbox[0].set_value('MS-DIAL').run()
        assert at.sidebar.selectbox[0].value == 'MS-DIAL'


# =============================================================================
# Group 3: Sample Data Loading (4 tests)
# =============================================================================

class TestSampleDataLoading:
    """Tests for loading sample datasets via the sidebar."""

    def test_load_generic_sample_data(self, format_upload_app):
        """Loading Generic sample data populates raw_df with 943 rows."""
        at = format_upload_app
        at.sidebar.button(key='load_sample').click().run()
        assert at.session_state['raw_df'].shape[0] == 943
        assert at.session_state['using_sample_data'] is True

    def test_load_lipidsearch_sample_data(self, format_upload_app):
        """Loading LipidSearch sample data produces valid DataFrame."""
        at = format_upload_app
        at.sidebar.selectbox[0].set_value('LipidSearch').run()
        at.sidebar.button(key='load_sample').click().run()
        raw_df = at.session_state['raw_df']
        assert raw_df is not None
        assert 'LipidMolec' in raw_df.columns

    def test_load_msdial_sample_data(self, format_upload_app):
        """Loading MS-DIAL sample data detects normalized data columns."""
        at = format_upload_app
        at.sidebar.selectbox[0].set_value('MS-DIAL').run()
        at.sidebar.button(key='load_sample').click().run()
        assert at.session_state['msdial_features'].has_normalized_data is True

    def test_load_metabolomics_workbench_sample_data(self, format_upload_app):
        """Loading Metabolomics Workbench sample data succeeds."""
        at = format_upload_app
        at.sidebar.selectbox[0].set_value('Metabolomics Workbench').run()
        at.sidebar.button(key='load_sample').click().run()
        assert at.session_state['raw_df'] is not None


# =============================================================================
# Group 4: Experiment Definition (4 tests)
# =============================================================================

class TestExperimentDefinition:
    """Tests for experiment configuration (conditions + samples)."""

    def test_sample_data_auto_populates_conditions(self, full_sidebar_app):
        """Loading sample data auto-populates condition count from metadata."""
        at = full_sidebar_app
        at.sidebar.button(key='load_sample').click().run()
        # Generic sample data has 3 conditions (WT, ADGAT-DKO, BQC)
        assert at.sidebar.number_input[0].value == 3

    def test_sample_data_auto_populates_condition_names(self, full_sidebar_app):
        """Loading sample data auto-populates condition names from metadata."""
        at = full_sidebar_app
        at.sidebar.button(key='load_sample').click().run()
        cond_input = at.text_input(key='cond_name_0')
        assert cond_input.value == 'WT'

    def test_sample_count_validation_blocks_progress(self, full_sidebar_app):
        """When sample counts don't match dataset, grouping/confirm are blocked."""
        at = full_sidebar_app
        at.sidebar.button(key='load_sample').click().run()
        # Override auto-populated value: set to 2 conditions so counts don't match
        at.sidebar.number_input[0].set_value(2).run()
        # 2 conditions x 4 samples = 8, but dataset has 12 samples
        # grouping_radio and confirm_checkbox should NOT be rendered
        grouping_radios = [r for r in at.sidebar.radio if r.key == 'grouping_radio']
        confirm_checkboxes = [c for c in at.sidebar.checkbox if c.key == 'confirm_checkbox']
        assert len(grouping_radios) == 0
        assert len(confirm_checkboxes) == 0

    def test_correct_sample_count_enables_grouping(self, generic_sidebar_app):
        """When sample counts match dataset (3x4=12), grouping and confirm appear."""
        at = generic_sidebar_app
        # generic_sidebar_app fixture already set 3 conditions x 4 samples = 12
        grouping_radio = at.sidebar.radio(key='grouping_radio')
        confirm_checkbox = at.sidebar.checkbox(key='confirm_checkbox')
        assert grouping_radio is not None
        assert confirm_checkbox is not None


# =============================================================================
# Group 5: Confirm Inputs / BQC (4 tests)
# =============================================================================

class TestConfirmInputsBQC:
    """Tests for BQC specification and input confirmation."""

    def test_bqc_radio_auto_populated_for_sample_data(self, generic_sidebar_app):
        """BQC radio auto-populates to 'Yes' for Generic sample data with BQC."""
        at = generic_sidebar_app
        assert at.sidebar.radio(key='bqc_radio').value == 'Yes'

    def test_confirm_checkbox_auto_checked_for_sample_data(self, generic_sidebar_app):
        """Confirm checkbox is auto-checked when loading sample data."""
        at = generic_sidebar_app
        assert at.sidebar.checkbox(key='confirm_checkbox').value is True

    def test_sample_data_auto_confirms_flow(self, generic_sidebar_app):
        """Loading sample data auto-populates and confirms, enabling flow."""
        at = generic_sidebar_app
        # Wrapper outputs "confirmed:3c" when experiment is returned
        text_values = [t.value for t in at.text]
        assert any("confirmed:3c" in v for v in text_values)

    def test_bqc_yes_shows_label_selector(self, generic_sidebar_app):
        """Setting BQC to 'Yes' shows the BQC label radio."""
        at = generic_sidebar_app
        at.sidebar.radio(key='bqc_radio').set_value('Yes').run()
        bqc_label_radio = at.sidebar.radio(key='bqc_label_radio')
        assert bqc_label_radio is not None

    def test_prefilled_bqc_radio_emits_no_default_value_warning(self, generic_sidebar_app):
        """A pre-filled BQC radio must not also be given an index default.

        Sample-data/alignment pre-fill sets 'bqc_radio' in session state; if the
        widget is also created with `index`, Streamlit warns that a keyed widget
        has both a default and a session-state value. The radio should render its
        pre-filled value with no such warning.
        """
        at = generic_sidebar_app
        offending = [
            w.value for w in at.warning
            if 'was created with a default value but also had its value set'
            in w.value
        ]
        assert offending == []


# =============================================================================
# Group 6: MS-DIAL Data Type Selection (3 tests)
# =============================================================================

class TestMSDIALDataTypeSelection:
    """Tests for MS-DIAL raw vs pre-normalized data type radio."""

    def test_data_type_radio_exists(self, msdial_data_type_app):
        """Data type radio is rendered when MS-DIAL features are present."""
        at = msdial_data_type_app
        radio = at.radio(key='msdial_data_type_radio')
        assert radio is not None

    def test_data_type_default_is_raw(self, msdial_data_type_app):
        """Default data type selection is raw intensity."""
        at = msdial_data_type_app
        radio = at.radio(key='msdial_data_type_radio')
        assert "Raw" in radio.value
        text_values = [t.value for t in at.text]
        assert any("use_normalized:False" in v for v in text_values)

    def test_data_type_can_switch_to_normalized(self, msdial_data_type_app):
        """User can switch to pre-normalized data."""
        at = msdial_data_type_app
        radio = at.radio(key='msdial_data_type_radio')
        # Find the pre-normalized option
        norm_option = [o for o in radio.options if "Pre-normalized" in o][0]
        radio.set_value(norm_option).run()
        text_values = [t.value for t in at.text]
        assert any("use_normalized:True" in v for v in text_values)


# =============================================================================
# Group 7: MS-DIAL Override Preservation (3 tests) — Regression for d9451cf
# =============================================================================

class TestMSDIALOverridePreservation:
    """Regression tests ensuring MS-DIAL sample override is preserved correctly."""

    def test_override_saved_to_session_state(self, msdial_features_dict):
        """Override samples persist in session state after run."""
        at = AppTest.from_function(override_preservation_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['msdial_features'] = msdial_features_dict
        at.session_state['_msdial_override_samples'] = ['s1', 's2', 's3']
        at.run()
        text_values = [t.value for t in at.text]
        assert any("override:['s1', 's2', 's3']" in v for v in text_values)

    def test_override_preserved_after_data_type_switch(self, msdial_features_dict):
        """Switching data type clears standardized_df and column_mapping but NOT the override."""
        at = AppTest.from_function(override_preservation_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['msdial_features'] = msdial_features_dict
        at.session_state['_msdial_override_samples'] = ['s1', 's2', 's3']
        at.session_state['standardized_df'] = 'placeholder'
        at.session_state['column_mapping'] = 'stale_mapping'
        at.session_state['n_intensity_cols'] = 99
        at.run()
        # Switch to pre-normalized (triggers on_data_type_change callback)
        radio = at.radio(key='msdial_data_type_radio')
        norm_option = [o for o in radio.options if "Pre-normalized" in o][0]
        radio.set_value(norm_option).run()
        text_values = [t.value for t in at.text]
        # Override should still be present
        assert any("override:['s1', 's2', 's3']" in v for v in text_values)
        # standardized_df, column_mapping, n_intensity_cols should all be cleared
        assert any("std_df_cleared:True" in v for v in text_values)
        assert any("col_mapping_cleared:True" in v for v in text_values)
        assert any("n_intensity_cleared:True" in v for v in text_values)

    def test_override_cleared_on_reset(self, msdial_features_dict):
        """reset_data_state() removes the override from session state."""
        at = AppTest.from_function(override_reset_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_msdial_override_samples'] = ['s1', 's2', 's3']
        at.run()
        # Verify override is present initially
        text_values = [t.value for t in at.text]
        assert any("override:['s1', 's2', 's3']" in v for v in text_values)
        # Click reset button
        at.button(key='reset_btn').click().run()
        # After reset + rerun, override should be gone
        text_values = [t.value for t in at.text]
        assert any("override:None" in v for v in text_values)


# =============================================================================
# Group 8: Back to Home (2 tests)
# =============================================================================

class TestBackToHome:
    """Tests for the Back to Home button on the app page."""

    def test_back_button_exists_when_no_data(self):
        """Back to Home button renders when no data is loaded."""
        at = AppTest.from_function(app_page_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['page'] = 'app'
        at.run()
        back_buttons = [b for b in at.button if "Back to Home" in b.label]
        assert len(back_buttons) >= 1

    def test_back_button_resets_to_landing(self):
        """Clicking Back to Home sets page to 'landing'."""
        at = AppTest.from_function(app_page_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['page'] = 'app'
        at.run()
        back_button = [b for b in at.button if "Back to Home" in b.label][0]
        back_button.click().run()
        assert at.session_state['page'] == 'landing'


# =============================================================================
# Group 9: Sample Display Names (sidebar-only feature)
# =============================================================================

class TestSampleNameEditor:
    """The sidebar seeds sample names from column headers and lets users edit them."""

    def test_seeds_names_from_column_headers(self, sample_name_editor_app):
        """Meaningful headers in column_mapping carry into session sample_names."""
        at = sample_name_editor_app
        assert not at.exception
        names = at.session_state['sample_names']
        assert names == {'s1': 'mouse liver #5', 's2': 'mouse liver #2'}

    def test_editor_round_trips_names_into_session(self, sample_name_editor_app):
        """The data editor writes the (seeded) names back to session on render."""
        at = sample_name_editor_app
        text_values = [t.value for t in at.text]
        assert any("mouse liver #5" in v for v in text_values)

    def test_editor_caption_present(self, sample_name_editor_app):
        """The editor explains the names carry to selectors/plots."""
        at = sample_name_editor_app
        captions = [c.value for c in at.caption]
        assert any("sample selectors" in c for c in captions)


# =============================================================================
# Group 10: MS-DIAL Override + Sample Names (regression)
# =============================================================================

class TestMSDIALOverrideSampleNames:
    """Dropping the blank via MS-DIAL override must refresh display names.

    Regression: sample_names was seeded once from the original mapping and not
    invalidated by the override, so the group table showed a stale, shifted map
    (blank still s1) even though the column mapping was correct.
    """

    def test_override_reseeds_names_without_blank(self, msdial_override_sample_names_app):
        at = msdial_override_sample_names_app
        assert not at.exception
        names = at.session_state['sample_names']
        # After dropping the blank, s1 is the first real sample, not "Blank".
        assert names['s1'] == 'fads2_1'
        assert 'Blank' not in names.values()
        assert names['s6'] == 'WT_3'


# =============================================================================
# Group 11: Regroup + confirm with sample names (blank-selection regression)
# =============================================================================

class TestRegroupConfirmWithNames:
    """Shuffling samples then confirming must advance, not reset the pickers.

    Regression: when sample display names were populated (e.g. MS-DIAL), the
    regroup multiselect's format_func read sample_names, which the regroup
    mutated mid-run. That reset the multiselects to empty and flipped
    grouping_complete/confirmed back to False, trapping the user in the
    reshuffle step instead of advancing.
    """

    def test_shuffle_then_confirm_advances(self):
        at = AppTest.from_function(full_sidebar_script, default_timeout=60)
        at.run()
        at.sidebar.selectbox[0].set_value('MS-DIAL').run()
        at.sidebar.button(key='load_sample').click().run()

        nconds = at.sidebar.number_input[0].value
        conds = [at.text_input(key=f'cond_name_{i}').value for i in range(nconds)]
        counts = [at.sidebar.number_input[i + 1].value for i in range(nconds)]
        ntot = sum(counts)
        # Reverse-order shuffle across all samples.
        shuffled = [f's{i}' for i in range(ntot, 0, -1)]

        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        idx = 0
        for c, n in zip(conds, counts):
            at.sidebar.multiselect(key=f'select_{c}').set_value(shuffled[idx:idx + n]).run()
            idx += n
        assert at.session_state['grouping_complete'] is True

        at.sidebar.checkbox(key='confirm_checkbox').set_value(True).run()
        # The grouping must stick and the app must advance.
        assert at.session_state['confirmed'] is True
        assert at.session_state['grouping_complete'] is True
        # Selections retained (not blanked).
        for c in conds:
            assert len(at.sidebar.multiselect(key=f'select_{c}').value) > 0
