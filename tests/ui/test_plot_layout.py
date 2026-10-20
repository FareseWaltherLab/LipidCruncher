"""
Plot layout rules every figure follows (R8, D12).

The figures hard-code black text, so their paper must be white too: left to
the theme, a dark Streamlit theme put black axis text on a dark paper. And
no plotting code may use ``titlefont``, which Plotly 6 removes.
"""

import re
from pathlib import Path

import plotly.graph_objects as go
import pytest

PLOTTING = Path(__file__).parents[2] / 'src' / 'app' / 'services' / 'plotting'


def themed(fig):
    """A figure that leaves both background and text to the theme."""
    return fig.layout.paper_bgcolor is None and fig.layout.font.color is None


def _white(fig):
    return (fig.layout.paper_bgcolor, fig.layout.plot_bgcolor) == ('white', 'white')


def test_qc_figures(qc_lipidsearch_app):
    at = qc_lipidsearch_app
    for key in ('qc_box_plot_fig1', 'qc_box_plot_fig2', 'qc_retention_time_plot',
                'qc_pca_plot'):
        assert _white(at.session_state[key]), key


def test_bqc_figure(qc_bqc_app):
    assert _white(qc_bqc_app.session_state['qc_bqc_plot'])


def test_internal_standards_figures(intsta_with_standards_app):
    figs = intsta_with_standards_app.session_state['standards_consistency_figs']
    assert figs and all(_white(f) for f in figs)


# Clustered and Regular leave background and text to the theme on purpose
# (their condition names follow its font colour); see the plotter's tests.
@pytest.mark.parametrize('heatmap_type', ['Grouped by Class', 'Aggregated'])
def test_heatmap_modes(analysis_generic_app, heatmap_type):
    at = analysis_generic_app
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if 'Lipidomic Heatmap' in o][0]).run()
    at.radio(key='heatmap_type').set_value(
        [o for o in at.radio(key='heatmap_type').options if heatmap_type in o][0]
    ).run()
    assert not at.exception
    assert _white(at.session_state['analysis_all_plots']['heatmap'])


def test_chain_length_text_is_black(analysis_generic_app):
    """Regression: its paper turned white while its text still followed the
    theme, which is white under a dark theme."""
    at = analysis_generic_app
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if 'Chain Length' in o][0]).run()
    fig = at.session_state['analysis_all_plots']['chain_length']
    assert _white(fig) and fig.layout.font.color == 'black'


def test_every_analysis_page(analysis_detailed_fa_app):
    """Each page's report figures (Plotly ones; the pathway is matplotlib)."""
    at = analysis_detailed_fa_app
    radio = at.radio(key='analysis_radio')
    for option in radio.options:
        at.radio(key='analysis_radio').set_value(option).run()
        assert not at.exception, option
        for key, fig in at.session_state['analysis_all_plots'].items():
            if isinstance(fig, go.Figure) and not (key == 'heatmap' and themed(fig)):
                assert _white(fig), (option, key)


def test_backgrounds_come_from_base_layout():
    """A plot setting its own colours is how three lost the white paper."""
    users = [
        f"{path.name}:{n}"
        for path in sorted(PLOTTING.rglob('*.py')) if path.name != '_shared.py'
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r'\b(plot|paper)_bgcolor\b', line)
    ]
    assert users == []


def test_no_titlefont():
    """Regression: fach.py used ``titlefont`` (deprecated; gone in Plotly 6)."""
    users = [
        f"{path.name}:{n}"
        for path in sorted(PLOTTING.rglob('*.py'))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r'\btitlefont\b', line)
    ]
    assert users == []
