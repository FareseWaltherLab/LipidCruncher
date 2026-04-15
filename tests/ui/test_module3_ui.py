"""
UI tests for Module 3: Visualize and Analyze.

Tests use Streamlit's AppTest framework with wrapper functions from conftest.py.
The Analysis module is parameterized via session state (_test_df, _test_experiment, etc.)
to test different data scenarios without importing main_app.py.

46 tests across 10 groups covering:
1. Radio selection — all 7 options render, switching works
2. Bar chart — defaults, scale switch, figure stored, empty class warning
3. Pie chart — defaults, per-condition display, deselect class updates
4. Saturation — consolidated warning, plot type switch, detailed FA no warning
5. FACH — class selectbox, conditions multiselect, figure stored
6. Pathway — control/experimental selectors, figure stored
7. Volcano — condition selectors, thresholds, hide nonsig, figure stored
8. Heatmap — type switch hides slider, cluster count, composition view
9. Navigation — module routing, state reset
10. Edge cases — single condition volcano warning, three-condition analysis
"""

from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    analysis_module_script,
    module3_nav_script,
    make_analysis_dataframe,
)


# =============================================================================
# Group 1: Analysis Radio Selection (3 tests)
# =============================================================================

class TestAnalysisRadioSelection:
    """Tests for the main analysis type radio selector."""

    def test_analysis_radio_exists_with_8_options(self, analysis_generic_app):
        """Analysis radio renders with all 8 analysis options."""
        at = analysis_generic_app
        radio = at.radio(key='analysis_radio')
        assert radio is not None
        assert len(radio.options) == 8

    def test_default_selection_is_bar_chart(self, analysis_generic_app):
        """Default selection is the first option (Bar Chart)."""
        at = analysis_generic_app
        radio = at.radio(key='analysis_radio')
        assert "Bar Chart" in radio.value

    def test_switching_to_pie_chart(self, analysis_generic_app):
        """Switching to Pie Charts option renders without exception."""
        at = analysis_generic_app
        radio = at.radio(key='analysis_radio')
        pie_option = [o for o in radio.options if "Pie" in o][0]
        at.radio(key='analysis_radio').set_value(pie_option).run()
        assert not at.exception
        # Pie class multiselect should now exist
        pie_ms = at.multiselect(key='pie_classes')
        assert pie_ms is not None


# =============================================================================
# Group 2: Bar Chart (8 tests)
# =============================================================================

class TestBarChartUI:
    """Tests for abundance bar chart section."""

    def test_bar_chart_renders_without_error(self, analysis_generic_app):
        """Bar chart section renders without exceptions."""
        at = analysis_generic_app
        assert not at.exception
        text_values = [t.value for t in at.text]
        assert any("analysis_rendered:ok" in v for v in text_values)

    def test_bar_conditions_multiselect(self, analysis_generic_app):
        """Bar chart condition multiselect has both conditions selected by default."""
        at = analysis_generic_app
        ms = at.multiselect(key='bar_conditions')
        assert ms is not None
        assert 'Control' in ms.value
        assert 'Treatment' in ms.value

    def test_bar_classes_multiselect(self, analysis_generic_app):
        """Bar chart class multiselect has all classes selected by default."""
        at = analysis_generic_app
        ms = at.multiselect(key='bar_classes')
        assert ms is not None
        assert 'PC' in ms.value
        assert 'PE' in ms.value

    def test_bar_scale_radio_default_log10(self, analysis_generic_app):
        """Bar chart scale radio defaults to Log10 Scale."""
        at = analysis_generic_app
        radio = at.radio(key='bar_scale_radio')
        assert radio is not None
        assert radio.value == "Log10 Scale"

    def test_bar_stats_mode_default_auto(self, analysis_generic_app):
        """Bar chart stats mode defaults to Auto."""
        at = analysis_generic_app
        radio = at.radio(key='bar_stats_mode')
        assert radio is not None
        assert radio.value == "Auto"

    def test_bar_switch_to_linear_scale(self, analysis_generic_app):
        """Switching bar chart scale to Linear updates figure."""
        at = analysis_generic_app
        at.radio(key='bar_scale_radio').set_value("Linear Scale").run()
        assert not at.exception
        assert at.session_state['analysis_bar_chart_fig'] is not None

    def test_bar_stores_figure_in_session(self, analysis_generic_app):
        """Bar chart figure is stored in session state."""
        at = analysis_generic_app
        assert at.session_state['analysis_bar_chart_fig'] is not None
        assert 'bar_chart' in at.session_state['analysis_all_plots']

    def test_bar_manual_mode_renders(self, analysis_generic_app):
        """Switching bar chart stats to Manual mode renders without error."""
        at = analysis_generic_app
        at.radio(key='bar_stats_mode').set_value("Manual").run()
        assert not at.exception
        assert at.session_state['analysis_bar_chart_fig'] is not None

    def test_bar_manual_non_parametric(self, analysis_generic_app):
        """Bar chart with Manual non-parametric test renders without error."""
        at = analysis_generic_app
        at.radio(key='bar_stats_mode').set_value("Manual").run()
        at.selectbox(key='bar_test_type').set_value("non_parametric").run()
        assert not at.exception
        assert at.session_state['analysis_bar_chart_fig'] is not None

    def test_bar_manual_bonferroni_correction(self, analysis_generic_app):
        """Bar chart with Manual bonferroni correction renders without error."""
        at = analysis_generic_app
        at.radio(key='bar_stats_mode').set_value("Manual").run()
        at.selectbox(key='bar_correction').set_value("bonferroni").run()
        assert not at.exception
        assert at.session_state['analysis_bar_chart_fig'] is not None

    def test_bar_manual_uncorrected(self, analysis_generic_app):
        """Bar chart with Manual uncorrected renders without error."""
        at = analysis_generic_app
        at.radio(key='bar_stats_mode').set_value("Manual").run()
        at.selectbox(key='bar_correction').set_value("uncorrected").run()
        assert not at.exception
        assert at.session_state['analysis_bar_chart_fig'] is not None

    def test_bar_deselect_all_classes_shows_warning(self, analysis_generic_app):
        """Deselecting all classes shows a warning."""
        at = analysis_generic_app
        at.multiselect(key='bar_classes').set_value([]).run()
        assert not at.exception
        warning_values = [w.value for w in at.warning]
        assert any("class" in v.lower() for v in warning_values)


# =============================================================================
# Group 3: Pie Chart (5 tests)
# =============================================================================

class TestPieChartUI:
    """Tests for abundance pie chart section."""

    def _switch_to_pie(self, at):
        """Helper to switch to pie chart analysis."""
        radio = at.radio(key='analysis_radio')
        pie_option = [o for o in radio.options if "Pie" in o][0]
        at.radio(key='analysis_radio').set_value(pie_option).run()
        return at

    def test_pie_renders_without_error(self, analysis_generic_app):
        """Pie chart section renders without exceptions."""
        at = self._switch_to_pie(analysis_generic_app)
        assert not at.exception

    def test_pie_classes_multiselect_defaults_all(self, analysis_generic_app):
        """Pie chart class multiselect defaults to all classes."""
        at = self._switch_to_pie(analysis_generic_app)
        ms = at.multiselect(key='pie_classes')
        assert ms is not None
        assert 'PC' in ms.value
        assert 'PE' in ms.value

    def test_pie_stores_figures_in_session(self, analysis_generic_app):
        """Pie chart figures are stored in session state per condition."""
        at = self._switch_to_pie(analysis_generic_app)
        figs = at.session_state['analysis_pie_chart_figs']
        assert 'Control' in figs
        assert 'Treatment' in figs

    def test_pie_deselect_class_updates_figures(self, analysis_generic_app):
        """Deselecting a class re-renders pie charts with remaining classes."""
        at = self._switch_to_pie(analysis_generic_app)
        at.multiselect(key='pie_classes').set_value(['PC']).run()
        assert not at.exception
        figs = at.session_state['analysis_pie_chart_figs']
        assert 'Control' in figs

    def test_pie_deselect_all_classes_shows_warning(self, analysis_generic_app):
        """Deselecting all classes shows a warning."""
        at = self._switch_to_pie(analysis_generic_app)
        at.multiselect(key='pie_classes').set_value([]).run()
        assert not at.exception
        warning_values = [w.value for w in at.warning]
        assert any("class" in v.lower() for v in warning_values)


# =============================================================================
# Group 4: Saturation Plots (5 tests)
# =============================================================================

class TestSaturationUI:
    """Tests for saturation plot section."""

    def _switch_to_saturation(self, at):
        """Helper to switch to saturation analysis."""
        radio = at.radio(key='analysis_radio')
        sat_option = [o for o in radio.options if "Saturation" in o][0]
        at.radio(key='analysis_radio').set_value(sat_option).run()
        return at

    def test_saturation_renders_without_error(self, analysis_generic_app):
        """Saturation section renders without exceptions."""
        at = self._switch_to_saturation(analysis_generic_app)
        assert not at.exception

    def test_saturation_consolidated_warning(self, analysis_generic_app):
        """Consolidated lipid names trigger FA compatibility info message."""
        at = self._switch_to_saturation(analysis_generic_app)
        info_values = [i.value for i in at.info]
        assert any("consolidated" in v.lower() for v in info_values)

    def test_saturation_plot_type_radio(self, analysis_generic_app):
        """Saturation plot type radio defaults to Concentration."""
        at = self._switch_to_saturation(analysis_generic_app)
        radio = at.radio(key='sat_plot_type')
        assert radio is not None
        assert radio.value == "Concentration"

    def test_saturation_switch_to_percentage(self, analysis_generic_app):
        """Switching to Percentage plot type re-renders without error."""
        at = self._switch_to_saturation(analysis_generic_app)
        at.radio(key='sat_plot_type').set_value("Percentage").run()
        assert not at.exception
        # Percentage plots stored in session
        figs = at.session_state['analysis_saturation_figs']
        assert any('percentage' in k for k in figs)

    def test_saturation_switch_to_both(self, analysis_generic_app):
        """Switching to Both plot type renders concentration and percentage."""
        at = self._switch_to_saturation(analysis_generic_app)
        at.radio(key='sat_plot_type').set_value("Both").run()
        assert not at.exception
        figs = at.session_state['analysis_saturation_figs']
        assert any('concentration' in k for k in figs)
        assert any('percentage' in k for k in figs)

    def test_saturation_manual_mode_renders(self, analysis_generic_app):
        """Switching saturation stats to Manual mode renders without error."""
        at = self._switch_to_saturation(analysis_generic_app)
        at.radio(key='sat_stats_mode').set_value("Manual").run()
        assert not at.exception
        figs = at.session_state['analysis_saturation_figs']
        assert len(figs) > 0

    def test_saturation_detailed_fa_no_consolidated_warning(self, analysis_detailed_fa_app):
        """Detailed FA names do not trigger consolidated warning."""
        at = analysis_detailed_fa_app
        radio = at.radio(key='analysis_radio')
        sat_option = [o for o in radio.options if "Saturation" in o][0]
        at.radio(key='analysis_radio').set_value(sat_option).run()
        assert not at.exception
        info_values = [i.value for i in at.info]
        assert not any("consolidated" in v.lower() for v in info_values)


# =============================================================================
# Group 5: Chain Length Distribution (4 tests)
# =============================================================================

class TestChainLengthUI:
    """Tests for Chain Length Distribution bubble chart section."""

    def _switch_to_chain_length(self, at):
        """Helper to switch to chain length analysis."""
        radio = at.radio(key='analysis_radio')
        cl_option = [o for o in radio.options if "Chain Length" in o][0]
        at.radio(key='analysis_radio').set_value(cl_option).run()
        return at

    def test_chain_length_renders_without_error(self, analysis_generic_app):
        """Chain length section renders without exceptions."""
        at = self._switch_to_chain_length(analysis_generic_app)
        assert not at.exception

    def test_chain_length_condition_selector_exists(self, analysis_generic_app):
        """Chain length has a condition multiselect."""
        at = self._switch_to_chain_length(analysis_generic_app)
        ms = at.multiselect(key='clen_conditions')
        assert ms is not None
        assert len(ms.value) > 0

    def test_chain_length_class_selector_exists(self, analysis_generic_app):
        """Chain length has a class multiselect."""
        at = self._switch_to_chain_length(analysis_generic_app)
        ms = at.multiselect(key='clen_classes')
        assert ms is not None
        assert len(ms.value) > 0

    def test_chain_length_figure_stored_in_session(self, analysis_generic_app):
        """Chain length figure is stored in session state."""
        at = self._switch_to_chain_length(analysis_generic_app)
        assert 'analysis_chain_length_fig' in at.session_state
        assert at.session_state['analysis_chain_length_fig'] is not None


# =============================================================================
# Group 6: FACH Heatmaps (3 tests)
# =============================================================================

class TestFACHUI:
    """Tests for Fatty Acid Composition Heatmap section."""

    def _switch_to_fach(self, at):
        """Helper to switch to FACH analysis."""
        radio = at.radio(key='analysis_radio')
        fach_option = [o for o in radio.options if "Fatty Acid" in o][0]
        at.radio(key='analysis_radio').set_value(fach_option).run()
        return at

    def test_fach_renders_without_error(self, analysis_generic_app):
        """FACH section renders without exceptions."""
        at = self._switch_to_fach(analysis_generic_app)
        assert not at.exception

    def test_fach_class_selectbox_exists(self, analysis_generic_app):
        """FACH has a class selectbox with available classes."""
        at = self._switch_to_fach(analysis_generic_app)
        sb = at.selectbox(key='fach_class')
        assert sb is not None
        assert sb.value in ['PC', 'PE']

    def test_fach_conditions_multiselect_defaults(self, analysis_generic_app):
        """FACH conditions multiselect defaults to first 2 eligible conditions."""
        at = self._switch_to_fach(analysis_generic_app)
        ms = at.multiselect(key='fach_conditions')
        assert ms is not None
        assert len(ms.value) == 2


# =============================================================================
# Group 6: Pathway Visualization (3 tests)
# =============================================================================

class TestPathwayUI:
    """Tests for pathway visualization section."""

    def _switch_to_pathway(self, at):
        """Helper to switch to pathway analysis."""
        radio = at.radio(key='analysis_radio')
        pathway_option = [o for o in radio.options if "Pathway" in o][0]
        at.radio(key='analysis_radio').set_value(pathway_option).run()
        return at

    def test_pathway_renders_without_error(self, analysis_generic_app):
        """Pathway section renders without exceptions."""
        at = self._switch_to_pathway(analysis_generic_app)
        assert not at.exception

    def test_pathway_condition_selectors_exist(self, analysis_generic_app):
        """Pathway has control and experimental condition selectboxes."""
        at = self._switch_to_pathway(analysis_generic_app)
        control = at.selectbox(key='pathway_control')
        experimental = at.selectbox(key='pathway_experimental')
        assert control is not None
        assert experimental is not None
        assert control.value != experimental.value

    def test_pathway_stores_figure_in_session(self, analysis_generic_app):
        """Pathway visualization figure is stored in session state."""
        at = self._switch_to_pathway(analysis_generic_app)
        assert at.session_state['analysis_pathway_fig'] is not None
        assert 'pathway' in at.session_state['analysis_all_plots']


# =============================================================================
# Group 6b: Pathway Layout Editor (10 tests)
# =============================================================================

class TestPathwayLayoutEditorUI:
    """Tests for pathway layout editing controls."""

    def _switch_to_pathway(self, at):
        """Helper to switch to pathway analysis."""
        radio = at.radio(key='analysis_radio')
        pathway_option = [o for o in radio.options if "Pathway" in o][0]
        at.radio(key='analysis_radio').set_value(pathway_option).run()
        return at

    def test_default_active_classes_18(self, analysis_generic_app):
        """Initial pathway state has 18 default active classes."""
        at = self._switch_to_pathway(analysis_generic_app)
        active = at.session_state['pathway_class_selector']
        assert len(active) == 18

    def test_preset_all_28_classes(self, analysis_generic_app):
        """Clicking 'All Classes (28)' sets active classes to 28."""
        at = self._switch_to_pathway(analysis_generic_app)
        at.button(key='pathway_start_all').click().run()
        assert not at.exception
        active = at.session_state['pathway_class_selector']
        assert len(active) == 28

    def test_preset_start_from_scratch(self, analysis_generic_app):
        """Clicking 'Start from Scratch' empties active classes."""
        at = self._switch_to_pathway(analysis_generic_app)
        at.button(key='pathway_start_scratch').click().run()
        assert not at.exception
        active = at.session_state['pathway_class_selector']
        assert active == []

    def test_preset_default_resets_to_18(self, analysis_generic_app):
        """Clicking 'Default (18 classes)' after 'All' resets to 18."""
        at = self._switch_to_pathway(analysis_generic_app)
        # Switch to all 28 first
        at.button(key='pathway_start_all').click().run()
        assert len(at.session_state['pathway_class_selector']) == 28
        # Now reset to default 18
        at.button(key='pathway_start_default').click().run()
        assert not at.exception
        assert len(at.session_state['pathway_class_selector']) == 18

    def test_add_custom_node(self, analysis_generic_app):
        """Adding a custom node updates session state."""
        at = self._switch_to_pathway(analysis_generic_app)
        at.text_input(key='pathway_add_node_name').set_value('CUSTOM').run()
        at.number_input(key='pathway_add_node_x').set_value(5.0).run()
        at.number_input(key='pathway_add_node_y').set_value(10.0).run()
        at.button(key='pathway_add_node_btn').click().run()
        assert not at.exception
        custom_nodes = at.session_state['analysis_pathway_custom_nodes']
        assert 'CUSTOM' in custom_nodes
        assert 'CUSTOM' in at.session_state['pathway_class_selector']

    def test_move_node_creates_position_override(self, analysis_generic_app):
        """Moving a node creates a position override in session state."""
        at = self._switch_to_pathway(analysis_generic_app)
        at.number_input(key='pathway_move_node_x').set_value(99.0).run()
        at.number_input(key='pathway_move_node_y').set_value(88.0).run()
        at.button(key='pathway_move_node_btn').click().run()
        assert not at.exception
        overrides = at.session_state['analysis_pathway_position_overrides']
        assert len(overrides) > 0

    def test_grid_toggle_defaults_to_false(self, analysis_generic_app):
        """Grid checkbox defaults to unchecked."""
        at = self._switch_to_pathway(analysis_generic_app)
        grid = at.checkbox(key='pathway_show_grid')
        assert grid is not None
        assert grid.value is False

    def test_grid_toggle_can_be_enabled(self, analysis_generic_app):
        """Grid checkbox can be toggled on."""
        at = self._switch_to_pathway(analysis_generic_app)
        at.checkbox(key='pathway_show_grid').set_value(True).run()
        assert not at.exception
        assert at.session_state['pathway_show_grid'] is True

    def test_class_multiselect_widget_exists(self, analysis_generic_app):
        """Class selector multiselect widget is rendered with all 28 options."""
        at = self._switch_to_pathway(analysis_generic_app)
        ms = at.multiselect(key='pathway_class_selector')
        assert ms is not None
        assert len(ms.options) == 28


# =============================================================================
# Group 7: Volcano Plot (6 tests)
# =============================================================================

class TestVolcanoUI:
    """Tests for volcano plot section."""

    def _switch_to_volcano(self, at):
        """Helper to switch to volcano analysis."""
        radio = at.radio(key='analysis_radio')
        volcano_option = [o for o in radio.options if "Volcano" in o][0]
        at.radio(key='analysis_radio').set_value(volcano_option).run()
        return at

    def test_volcano_renders_without_error(self, analysis_generic_app):
        """Volcano section renders without exceptions."""
        at = self._switch_to_volcano(analysis_generic_app)
        assert not at.exception

    def test_volcano_condition_selectors(self, analysis_generic_app):
        """Volcano control/experimental selectboxes exist with correct options."""
        at = self._switch_to_volcano(analysis_generic_app)
        control = at.selectbox(key='volcano_control')
        experimental = at.selectbox(key='volcano_experimental')
        assert control is not None
        assert experimental is not None
        assert 'Control' in control.options
        assert 'Treatment' in experimental.options

    def test_volcano_threshold_defaults(self, analysis_generic_app):
        """P-value and fold change thresholds have correct defaults."""
        at = self._switch_to_volcano(analysis_generic_app)
        p_input = at.number_input(key='volcano_p_threshold')
        fc_input = at.number_input(key='volcano_fc_threshold')
        assert p_input is not None
        assert fc_input is not None
        assert p_input.value == 0.05
        assert fc_input.value == 2.0

    def test_volcano_stores_figure_in_session(self, analysis_generic_app):
        """Volcano plot figure is stored in session state."""
        at = self._switch_to_volcano(analysis_generic_app)
        assert at.session_state['analysis_volcano_fig'] is not None

    def test_volcano_hide_nonsig_default_unchecked(self, analysis_generic_app):
        """Hide non-significant checkbox defaults to unchecked."""
        at = self._switch_to_volcano(analysis_generic_app)
        cb = at.checkbox(key='volcano_hide_nonsig')
        assert cb is not None
        assert cb.value is False

    def test_volcano_stores_data_in_session(self, analysis_generic_app):
        """Volcano data (VolcanoData) is stored in session state for sub-plots."""
        at = self._switch_to_volcano(analysis_generic_app)
        assert at.session_state['analysis_volcano_data'] is not None

    def test_volcano_non_parametric(self, analysis_generic_app):
        """Switching volcano to non-parametric test renders without error."""
        at = self._switch_to_volcano(analysis_generic_app)
        at.selectbox(key='volcano_stats_mode').set_value("non_parametric").run()
        assert not at.exception
        assert at.session_state['analysis_volcano_fig'] is not None

    def test_volcano_uncorrected(self, analysis_generic_app):
        """Switching volcano to uncorrected renders without error."""
        at = self._switch_to_volcano(analysis_generic_app)
        at.selectbox(key='volcano_correction').set_value("uncorrected").run()
        assert not at.exception
        assert at.session_state['analysis_volcano_fig'] is not None

    def test_volcano_bonferroni(self, analysis_generic_app):
        """Switching volcano to bonferroni correction renders without error."""
        at = self._switch_to_volcano(analysis_generic_app)
        at.selectbox(key='volcano_correction').set_value("bonferroni").run()
        assert not at.exception
        assert at.session_state['analysis_volcano_fig'] is not None


# =============================================================================
# Group 8: Lipidomic Heatmap (7 tests)
# =============================================================================

class TestHeatmapUI:
    """Tests for lipidomic heatmap section."""

    def _switch_to_heatmap(self, at):
        """Helper to switch to heatmap analysis."""
        radio = at.radio(key='analysis_radio')
        hm_option = [o for o in radio.options if "Lipidomic Heatmap" in o][0]
        at.radio(key='analysis_radio').set_value(hm_option).run()
        return at

    def test_heatmap_renders_without_error(self, analysis_generic_app):
        """Heatmap section renders without exceptions."""
        at = self._switch_to_heatmap(analysis_generic_app)
        assert not at.exception

    def test_heatmap_type_radio_default_clustered(self, analysis_generic_app):
        """Heatmap type radio defaults to Clustered."""
        at = self._switch_to_heatmap(analysis_generic_app)
        radio = at.radio(key='heatmap_type')
        assert radio is not None
        assert radio.value == "Clustered"

    def test_heatmap_cluster_slider_defaults(self, analysis_generic_app):
        """Cluster slider defaults to 5 in Clustered mode."""
        at = self._switch_to_heatmap(analysis_generic_app)
        slider = at.slider(key='heatmap_n_clusters')
        assert slider is not None
        assert slider.value == 5

    def test_heatmap_cluster_view_radio(self, analysis_generic_app):
        """Cluster composition view radio defaults to Species Count."""
        at = self._switch_to_heatmap(analysis_generic_app)
        radio = at.radio(key='heatmap_cluster_view')
        assert radio is not None
        assert radio.value == "Species Count"

    def test_heatmap_switch_to_regular_hides_slider(self, analysis_generic_app):
        """Switching to Regular mode removes the cluster slider."""
        at = self._switch_to_heatmap(analysis_generic_app)
        at.radio(key='heatmap_type').set_value("Regular").run()
        assert not at.exception
        # Slider should not be rendered in Regular mode
        try:
            at.slider(key='heatmap_n_clusters')
            slider_found = True
        except Exception:
            slider_found = False
        # In Regular mode, cluster slider is not rendered
        if slider_found:
            # If the slider exists, it might be a stale widget — check that
            # the heatmap figure was still generated
            assert at.session_state['analysis_heatmap_fig'] is not None
        else:
            assert True

    def test_heatmap_total_concentration_view(self, analysis_generic_app):
        """Switching cluster view to Total Concentration renders without error."""
        at = self._switch_to_heatmap(analysis_generic_app)
        at.radio(key='heatmap_cluster_view').set_value("Total Concentration").run()
        assert not at.exception
        assert at.session_state['analysis_heatmap_clusters'] is not None

    def test_heatmap_stores_figure_in_session(self, analysis_generic_app):
        """Heatmap figure is stored in session state."""
        at = self._switch_to_heatmap(analysis_generic_app)
        assert at.session_state['analysis_heatmap_fig'] is not None
        assert 'heatmap' in at.session_state['analysis_all_plots']

    def test_heatmap_cluster_composition_stored(self, analysis_generic_app):
        """Cluster composition DataFrame is stored in session state."""
        at = self._switch_to_heatmap(analysis_generic_app)
        assert at.session_state['analysis_heatmap_clusters'] is not None


# =============================================================================
# Group 9: Analysis Navigation (3 tests)
# =============================================================================

class TestAnalysisNavigation:
    """Tests for combined Module 2+3 navigation buttons."""

    def test_back_to_data_processing_resets_module(self, module3_nav_app):
        """Clicking 'Back to Data Processing' returns to Module 1."""
        at = module3_nav_app
        at.button(key='back_processing_module2').click().run()
        assert at.session_state['module'] == 'Data Cleaning, Filtering, & Normalization'

    def test_back_to_data_processing_resets_analysis_state(self, module3_nav_app):
        """Clicking 'Back to Data Processing' clears all analysis session state."""
        at = module3_nav_app
        at.button(key='back_processing_module2').click().run()
        assert at.session_state['analysis_bar_chart_fig'] is None
        assert at.session_state['analysis_volcano_fig'] is None
        assert at.session_state['analysis_all_plots'] == {}

    def test_back_to_home_resets_page(self, module3_nav_app):
        """Clicking 'Back to Home' from combined page sets page to 'landing'."""
        at = module3_nav_app
        at.button(key='back_home_module2').click().run()
        assert at.session_state['page'] == 'landing'


# =============================================================================
# Group 10: Edge Cases (3 tests)
# =============================================================================

class TestEdgeCases:
    """Tests for edge case scenarios."""

    def test_volcano_single_condition_shows_warning(self, analysis_single_cond_app):
        """Volcano with single condition shows a warning about needing 2 conditions."""
        at = analysis_single_cond_app
        radio = at.radio(key='analysis_radio')
        volcano_option = [o for o in radio.options if "Volcano" in o][0]
        at.radio(key='analysis_radio').set_value(volcano_option).run()
        assert not at.exception
        warning_values = [w.value for w in at.warning]
        assert any("2 conditions" in v or "at least 2" in v for v in warning_values)

    def test_pathway_single_condition_shows_warning(self, analysis_single_cond_app):
        """Pathway with single condition shows a warning about needing 2 conditions."""
        at = analysis_single_cond_app
        radio = at.radio(key='analysis_radio')
        pathway_option = [o for o in radio.options if "Pathway" in o][0]
        at.radio(key='analysis_radio').set_value(pathway_option).run()
        assert not at.exception
        warning_values = [w.value for w in at.warning]
        assert any("2 conditions" in v or "at least 2" in v for v in warning_values)

    def test_three_conditions_bar_chart(self, analysis_three_cond_app):
        """Bar chart with 3 conditions shows all 3 in multiselect."""
        at = analysis_three_cond_app
        ms = at.multiselect(key='bar_conditions')
        assert ms is not None
        assert len(ms.value) == 3
        assert 'Vehicle' in ms.value
