"""UI tests for the LipidSearch 5.2 dual-polarity alignment upload flow.

5.2 (per-file OriginalArea columns) requires the Alignment Setting file and is
merged into per-sample intensities; 5.0 (flat MeanArea) is untouched.
"""
from streamlit.testing.v1 import AppTest

from tests.ui.conftest import DEFAULT_TIMEOUT


# Alignment matching the dual-polarity df below: Control mouse 01 (s1-1/s1-2),
# Treated mouse 02 (s2-1/s2-2).
_ALIGNMENT_TEXT = (
    "*Parameters setting\n"
    "NormalizeType\tNONE\n"
    "\n"
    "*Target search job\n"
    "J\tm01n.raw\ts1-1\t*Control\tu1\n"
    "J\tm01p.raw\ts1-2\t*Control\tu2\n"
    "J\tm02n.raw\ts2-1\tTreated\tu3\n"
    "J\tm02p.raw\ts2-2\tTreated\tu4\n"
)


def _standardize_script():
    """Build a LipidSearch frame per st.session_state['_mode'] and run
    standardize_uploaded_data, emitting a result marker."""
    import streamlit as st
    import numpy as np
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.constants import FORMAT_LIPIDSEARCH
    from app.ui.sidebar.column_mapping import standardize_uploaded_data

    meta = {
        'LipidMolec': ['PC(16:0_18:1)', 'FA(15:0)'],
        'ClassKey': ['PC', 'FA'],
        'CalcMass': [757.5, 242.2],
        'BaseRt': [10.0, 3.0],
        'TotalGrade': ['A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0],
        'FAKey': ['(16:0_18:1)', '(15:0)'],
    }
    if st.session_state.get('_mode') == 'flat':
        df = pd.DataFrame({**meta, 'MeanArea[s1]': [1000.0, 2000.0],
                           'MeanArea[s2]': [1100.0, 2200.0]})
    else:
        df = pd.DataFrame({
            **meta,
            'OriginalArea[s1-1]': [np.nan, 300.0], 'OriginalArea[s1-2]': [200.0, np.nan],
            'OriginalArea[s2-1]': [np.nan, 330.0], 'OriginalArea[s2-2]': [220.0, np.nan],
        })

    result = standardize_uploaded_data(df, FORMAT_LIPIDSEARCH)
    if result is None:
        st.text("result:None")
    else:
        icols = [c for c in result.columns if c.startswith('intensity[')]
        st.text(f"result:{len(icols)}")


def _marker(at):
    for t in at.text:
        if t.value.startswith('result:'):
            return t.value.split(':', 1)[1]
    raise AssertionError("result marker not rendered")


class TestDualPolarityUpload:
    """5.2 dual-polarity data behaviour in standardize_uploaded_data."""

    def test_blocks_without_alignment(self):
        at = AppTest.from_function(_standardize_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_mode'] = 'dual'
        at.run()
        assert _marker(at) == "None"
        assert len(at.sidebar.error) > 0

    def test_merges_with_alignment(self):
        at = AppTest.from_function(_standardize_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_mode'] = 'dual'
        at.session_state['lipidsearch_alignment_text'] = _ALIGNMENT_TEXT
        at.run()
        # Two biological samples (mouse 01 Control, mouse 02 Treated).
        assert _marker(at) == "2"
        assert len(at.sidebar.error) == 0

    def test_flat_5_0_untouched(self):
        at = AppTest.from_function(_standardize_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['_mode'] = 'flat'
        at.run()
        # Flat MeanArea[s1]/[s2] -> intensity[s1]/[s2] via the normal path,
        # with no alignment required and no error.
        assert _marker(at) == "2"
        assert len(at.sidebar.error) == 0
