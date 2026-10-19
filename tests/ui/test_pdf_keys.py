"""Pins: which figures each analysis page leaves for the PDF report.

Each page owns its keys in ``analysis_all_plots`` and must hold exactly the
figures it currently draws: a selection change that removes a figure from
the page removes it from the report, and pages leave each other's keys
alone (the report collects every page the user visited).
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import (
    DEFAULT_TIMEOUT,
    analysis_module_script,
    make_analysis_dataframe,
)

PIE = {'pie_Control', 'pie_Treatment'}
SAT_CONC = {'sat_concentration_PC', 'sat_concentration_PE'}
SAT_PCT = {'sat_percentage_PC', 'sat_percentage_PE'}
VOLCANO = {'volcano', 'conc_vs_fc', 'lipid_distribution'}


@pytest.fixture
def at():
    from app.models.experiment import ExperimentConfig

    at = AppTest.from_function(analysis_module_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_test_df'] = make_analysis_dataframe(n_lipids=20, n_samples=6)
    at.session_state['_test_experiment'] = ExperimentConfig(
        n_conditions=2,
        conditions_list=['Control', 'Treatment'],
        number_of_samples_list=[3, 3],
    )
    at.session_state['_test_bqc_label'] = None
    at.session_state['_test_format_type'] = 'Generic Format'
    return at.run()


def _open(at, word):
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if word in o][0]).run()
    assert not at.exception
    return at


def _keys(at):
    return set(at.session_state['analysis_all_plots'])


def _set(at, key, value, widget='multiselect'):
    getattr(at, widget)(key=key).set_value(value).run()
    assert not at.exception


class TestEachPage:
    """Starting from a fresh session, so only the page's own keys exist."""

    def test_bar_chart(self, at):
        assert _keys(at) == {'bar_chart'}
        _set(at, 'bar_classes', [])
        assert _keys(at) == set()
        _set(at, 'bar_classes', ['PC'])
        assert _keys(at) == {'bar_chart'}

    def test_pie_charts(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Pie')
        assert _keys(at) == PIE
        _set(at, 'pie_classes', ['PC'])
        assert _keys(at) == PIE
        _set(at, 'pie_classes', [])
        assert _keys(at) == set()

    def test_saturation(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Saturation')
        assert _keys(at) == SAT_CONC
        _set(at, 'sat_plot_type', 'Percentage', 'radio')
        assert _keys(at) == SAT_PCT
        _set(at, 'sat_plot_type', 'Both', 'radio')
        assert _keys(at) == SAT_CONC | SAT_PCT
        _set(at, 'sat_classes', ['PC'])
        assert _keys(at) == {'sat_concentration_PC', 'sat_percentage_PC'}
        _set(at, 'sat_classes', [])
        assert _keys(at) == set()

    def test_chain_length(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Chain Length')
        assert _keys(at) == {'chain_length'}
        _set(at, 'clen_classes', [])
        assert _keys(at) == set()

    def test_fach(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Fatty Acid Composition')
        assert _keys(at) == {'fach'}
        _set(at, 'fach_conditions', [])
        assert _keys(at) == set()

    def test_pathway(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Pathway')
        assert _keys(at) == {'pathway'}

    def test_volcano(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Volcano')
        assert _keys(at) == VOLCANO
        _set(at, 'volcano_detail_lipids', [])
        assert _keys(at) == {'volcano', 'conc_vs_fc'}
        _set(at, 'volcano_classes', [])
        assert _keys(at) == set()

    def test_heatmap(self, at):
        at.session_state['analysis_all_plots'] = {}
        _open(at, 'Lipidomic Heatmap')
        assert _keys(at) == {'heatmap'}
        _set(at, 'heatmap_classes', [])
        assert _keys(at) == set()


def test_pages_keep_each_others_figures(at):
    """Every page visited stays in the report; changing one page's
    selection touches only that page's figures."""
    for word in ('Pie', 'Saturation', 'Chain Length', 'Fatty Acid Composition',
                 'Pathway', 'Volcano', 'Lipidomic Heatmap'):
        _open(at, word)
    everything = (
        {'bar_chart', 'chain_length', 'fach', 'pathway', 'heatmap'}
        | PIE | SAT_CONC | VOLCANO
    )
    assert _keys(at) == everything

    _set(at, 'heatmap_classes', [])
    assert _keys(at) == everything - {'heatmap'}
    _open(at, 'Bar Chart')
    _set(at, 'bar_classes', [])
    assert _keys(at) == everything - {'heatmap', 'bar_chart'}


def _publish_script():
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.ui.main_content.analysis._shared import _publish
    StreamlitAdapter.initialize_session_state()
    _publish(st.session_state['_key'], 'figure')


@pytest.mark.parametrize('key, accepted', [
    ('bar_chart', True), ('pie_Control', True), ('sat_percentage_PC', True),
    ('barchart', False), ('volcano_plot', False),
])
def test_publish_takes_only_keys_the_report_renders(key, accepted):
    at = AppTest.from_function(_publish_script, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['_key'] = key
    at.run()
    if accepted:
        assert not at.exception
        assert at.session_state['analysis_all_plots'] == {key: 'figure'}
    else:
        assert at.exception
        assert 'has no page' in at.exception[0].message
