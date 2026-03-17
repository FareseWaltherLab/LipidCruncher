"""
UI tests for Module 3: Visualize and Analyze.

Tests use Streamlit's AppTest framework with wrapper functions from conftest.py.
The Analysis module is parameterized via session state (_test_df, _test_experiment, etc.)
to test different data scenarios without importing main_app.py.

25 tests across 7 groups covering:
1. Radio selection — all 7 options render, switching works
2. Bar chart — condition/class multiselects, scale radio, stats mode, plot stored
3. Pie chart — class multiselect, per-condition display
4. Saturation — consolidated warning, stats panel, plot type radio
5. Volcano — condition selectors, threshold inputs, label management
6. Heatmap — type radio, cluster slider, composition display
7. Navigation — module routing, state reset on navigation
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

    def test_analysis_radio_exists_with_7_options(self, analysis_generic_app):
        """Analysis radio renders with all 7 analysis options."""
        at = analysis_generic_app
        radio = at.radio(key='analysis_radio')
        assert radio is not None
        assert len(radio.options) == 7

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
# Group 2: Bar Chart (5 tests)
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


# =============================================================================
# Group 3: Pie Chart (3 tests)
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


# =============================================================================
# Group 4: Saturation Plots (3 tests)
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


# =============================================================================
# Group 5: Volcano Plot (4 tests)
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


# =============================================================================
# Group 6: Lipidomic Heatmap (4 tests)
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


# =============================================================================
# Group 7: Analysis Navigation (3 tests)
# =============================================================================

class TestAnalysisNavigation:
    """Tests for Module 3 navigation buttons."""

    def test_back_to_qc_resets_module(self, module3_nav_app):
        """Clicking 'Back to Quality Check' returns to Module 2."""
        at = module3_nav_app
        at.button(key='back_qc_module3').click().run()
        assert at.session_state['module'] == 'Quality Check & Analysis'

    def test_back_to_qc_resets_analysis_state(self, module3_nav_app):
        """Clicking 'Back to Quality Check' clears all analysis session state."""
        at = module3_nav_app
        at.button(key='back_qc_module3').click().run()
        assert at.session_state['analysis_bar_chart_fig'] is None
        assert at.session_state['analysis_volcano_fig'] is None
        assert at.session_state['analysis_all_plots'] == {}

    def test_back_to_home_resets_page(self, module3_nav_app):
        """Clicking 'Back to Home' from Module 3 sets page to 'landing'."""
        at = module3_nav_app
        at.button(key='back_home_module3').click().run()
        assert at.session_state['page'] == 'landing'
