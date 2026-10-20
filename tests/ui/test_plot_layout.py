"""
Plot layout rules every figure follows (R8, D12).

The figures hard-code black text, so their paper must be white too: left to
the theme, a dark Streamlit theme put black axis text on a dark paper. And
no plotting code may use ``titlefont``, which Plotly 6 removes.
"""

import re
from pathlib import Path

PLOTTING = Path(__file__).parents[2] / 'src' / 'app' / 'services' / 'plotting'


def test_box_plots_have_a_white_paper(qc_generic_app):
    at = qc_generic_app
    for key in ('qc_box_plot_fig1', 'qc_box_plot_fig2'):
        assert at.session_state[key].layout.paper_bgcolor == 'white', key


def test_chain_length_plot_has_a_white_paper(analysis_generic_app):
    at = analysis_generic_app
    radio = at.radio(key='analysis_radio')
    radio.set_value([o for o in radio.options if 'Chain Length' in o][0]).run()
    assert not at.exception
    fig = at.session_state['analysis_all_plots']['chain_length']
    assert fig.layout.paper_bgcolor == 'white'


def test_no_titlefont():
    """Regression: fach.py used ``titlefont`` (deprecated; gone in Plotly 6)."""
    users = [
        f"{path.name}:{n}"
        for path in sorted(PLOTTING.rglob('*.py'))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r'\btitlefont\b', line)
    ]
    assert users == []
