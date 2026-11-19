"""
UI tests for the sidebar's "Color scheme" picker.

The choice recolors the figures on screen (and the copies kept for the PDF
report), survives reruns, and drops the figures of analysis pages not on
screen, so the report never mixes two schemes.
"""

from app.constants import CONDITION_PALETTES, DEFAULT_CONDITION_PALETTE
from tests.ui.test_lsi_report import _run_to_analysis, _show_analysis

_PICKER = '_condition_palette_select'


def _box_outlines(at):
    fig = at.session_state['qc_box_plot_fig2']
    return {t.name: t.line.color for t in fig.data if t.showlegend}


def _bar_outlines(at):
    fig = at.session_state['analysis_all_plots']['bar_chart']
    return [t.marker.line.color for t in fig.data if t.type == 'bar']


class TestColorSchemePicker:
    def test_defaults_to_classic(self):
        at = _run_to_analysis('None (pre-normalized data)')
        picker = at.sidebar.selectbox(key=_PICKER)
        assert picker.value == DEFAULT_CONDITION_PALETTE
        assert picker.options == list(CONDITION_PALETTES)
        classic = CONDITION_PALETTES['Classic']
        assert list(_box_outlines(at).values()) == list(classic[:len(_box_outlines(at))])

    def test_choice_recolors_quality_check_and_analysis(self):
        at = _run_to_analysis('None (pre-normalized data)')
        at.sidebar.selectbox(key=_PICKER).set_value('Soft').run()
        assert not at.exception, at.exception
        assert at.session_state['condition_palette'] == 'Soft'
        soft = CONDITION_PALETTES['Soft']
        outlines = _box_outlines(at)
        assert list(outlines.values()) == list(soft[:len(outlines)])
        assert _bar_outlines(at) == list(soft[:len(_bar_outlines(at))])

    def test_choice_survives_a_rerun(self):
        at = _run_to_analysis('None (pre-normalized data)')
        at.sidebar.selectbox(key=_PICKER).set_value('Colorblind-safe').run()
        at.run()
        assert at.sidebar.selectbox(key=_PICKER).value == 'Colorblind-safe'
        assert at.session_state['condition_palette'] == 'Colorblind-safe'

    def test_pages_not_shown_drop_their_figures(self):
        """The bar chart drawn in Classic must not reach the report once
        the scheme changes while another page is shown."""
        at = _run_to_analysis('None (pre-normalized data)')
        _show_analysis(at, 'Pie')
        assert 'bar_chart' in at.session_state['analysis_all_plots']
        at.sidebar.selectbox(key=_PICKER).set_value('Soft').run()
        assert not at.exception, at.exception
        assert 'bar_chart' not in at.session_state['analysis_all_plots']
