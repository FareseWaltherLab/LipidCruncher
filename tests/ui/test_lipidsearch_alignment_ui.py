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


def _prefill_script():
    """Pre-populate the experiment from an alignment's conditions/counts and
    emit the resulting widget-key values."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.ui.sidebar.file_upload import _populate_experiment_from_alignment
    _populate_experiment_from_alignment(['Control', 'Fumonisin B1'], [3, 3])

    st.text(f"c0:{st.session_state.get('cond_name_0')}")
    st.text(f"c1:{st.session_state.get('cond_name_1')}")
    st.text(f"n0:{st.session_state.get('n_samples_0')}")
    st.text(f"n1:{st.session_state.get('n_samples_1')}")
    meta = st.session_state.get('sample_data_experiment') or {}
    st.text(f"nc:{meta.get('n_conditions')}")


class TestExperimentPrefill:
    """Uploading an alignment pre-fills the experiment (conditions + counts)."""

    def test_prefills_condition_and_sample_widget_keys(self):
        at = AppTest.from_function(_prefill_script, default_timeout=DEFAULT_TIMEOUT).run()
        vals = {t.value.split(':', 1)[0]: t.value.split(':', 1)[1] for t in at.text}
        assert vals['c0'] == 'Control'
        assert vals['c1'] == 'Fumonisin B1'
        assert vals['n0'] == '3'
        assert vals['n1'] == '3'
        assert vals['nc'] == '2'
        # No StreamlitAPIException from setting widget-backed keys pre-render.
        assert len(at.exception) == 0


# One raw file searched by two jobs (ID_01) plus a pos/neg file pair (m02).
_NAMED_ALIGNMENT_TEXT = (
    "*Target search job\n"
    "J\tID_01.raw\ts1-1\tID\tu1\n"
    "J\tID_01.raw\ts1-2\tID\tu2\n"
    "J\tm02n.raw\ts2-1\tTreated\tu3\n"
    "J\tm02p.raw\ts2-2\tTreated\tu4\n"
)


def _sample_names_script():
    """Standardize a dual-polarity frame via its alignment, then run the
    sidebar sample grouping (which seeds and renders the Sample Names editor)
    and emit the resulting session sample_names."""
    import streamlit as st
    import numpy as np
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.constants import FORMAT_LIPIDSEARCH
    from app.services.lipidsearch_alignment import parse_alignment_file
    from app.ui.sidebar.column_mapping import standardize_uploaded_data
    from app.ui.sidebar.file_upload import _populate_experiment_from_alignment
    from app.ui.sidebar.sample_grouping import display_sample_grouping

    text = st.session_state['lipidsearch_alignment_text']
    if st.session_state.get('standardized_df') is None:
        alignment = parse_alignment_file(text)
        _populate_experiment_from_alignment(
            alignment.conditions, alignment.samples_per_condition
        )
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)', 'FA(15:0)'],
            'ClassKey': ['PC', 'FA'],
            'CalcMass': [757.5, 242.2],
            'BaseRt': [10.0, 3.0],
            'TotalGrade': ['A', 'A'],
            'TotalSmpIDRate(%)': [100.0, 100.0],
            'FAKey': ['(16:0_18:1)', '(15:0)'],
            'OriginalArea[s1-1]': [np.nan, 300.0], 'OriginalArea[s1-2]': [200.0, np.nan],
            'OriginalArea[s2-1]': [np.nan, 330.0], 'OriginalArea[s2-2]': [220.0, np.nan],
        })
        st.session_state.standardized_df = standardize_uploaded_data(
            df, FORMAT_LIPIDSEARCH
        )
    display_sample_grouping(st.session_state.standardized_df, FORMAT_LIPIDSEARCH)
    st.text(f"names:{st.session_state.get('sample_names')}")


class TestAlignmentSampleNames:
    """The alignment's raw filenames seed the editable sample display names.

    Regression: the merged column mapping holds only per-file tokens, so
    build_names_from_mapping named each alignment sample with the composite
    "OriginalArea[s1-1] + OriginalArea[s1-2]" string.
    """

    def test_seeds_editor_names_from_raw_filenames(self):
        at = AppTest.from_function(_sample_names_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['lipidsearch_alignment_text'] = _NAMED_ALIGNMENT_TEXT
        at.run()
        assert not at.exception
        # Seeded, then written back unchanged by the Sample Names editor.
        assert at.session_state['sample_names'] == {'s1': 'ID_01', 's2': 'm02'}

    def test_user_names_are_not_overwritten(self):
        at = AppTest.from_function(_sample_names_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['lipidsearch_alignment_text'] = _NAMED_ALIGNMENT_TEXT
        at.session_state['sample_names'] = {'s1': 'pooled', 's2': 'mouse 2'}
        at.run()
        assert not at.exception
        assert at.session_state['sample_names'] == {'s1': 'pooled', 's2': 'mouse 2'}


# Sample values 1..4 identify which raw file's data a column holds.
_SOFT_RESET_ALIGNMENT_TEXT = (
    "*Target search job\n"
    "J\talpha.raw\ts1-1\tA\tu1\nJ\tbeta.raw\ts1-2\tA\tu2\n"
    "J\tgamma.raw\ts2-1\tB\tu3\nJ\tdelta.raw\ts2-2\tB\tu4\n"
)


def _soft_reset_script():
    """Sample grouping over an alignment upload, with an optional soft reset
    (st.session_state['_soft']); emits each sample name's data value."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.constants import FORMAT_LIPIDSEARCH
    from app.services.lipidsearch_alignment import parse_alignment_file
    from app.ui.sidebar.column_mapping import standardize_uploaded_data
    from app.ui.sidebar.file_upload import _populate_experiment_from_alignment
    from app.ui.sidebar.sample_grouping import display_sample_grouping

    if st.session_state.pop('_soft', False):
        StreamlitAdapter.reset_to_experiment_setup()
    if st.session_state.get('experiment') is None:
        alignment = parse_alignment_file(st.session_state['lipidsearch_alignment_text'])
        _populate_experiment_from_alignment(
            alignment.conditions, alignment.samples_per_condition
        )
    if st.session_state.get('standardized_df') is None:
        df = pd.DataFrame({
            'LipidMolec': ['PC(16:0_18:1)'], 'ClassKey': ['PC'], 'CalcMass': [757.5],
            'BaseRt': [10.0], 'TotalGrade': ['A'], 'TotalSmpIDRate(%)': [100.0],
            'FAKey': ['(16:0_18:1)'],
            'OriginalArea[s1-1]': [1.0], 'OriginalArea[s1-2]': [2.0],
            'OriginalArea[s2-1]': [3.0], 'OriginalArea[s2-2]': [4.0],
        })
        st.session_state.standardized_df = standardize_uploaded_data(df, FORMAT_LIPIDSEARCH)
    display_sample_grouping(st.session_state.standardized_df, FORMAT_LIPIDSEARCH)
    sdf = st.session_state.standardized_df
    names = st.session_state.get('sample_names') or {}
    st.text("data:" + str(sorted(
        (name, float(sdf[f'intensity[{s}]'].iloc[0])) for s, name in names.items()
    )))


class TestAlignmentNamesAfterSoftReset:
    """Regression: a soft reset after a confirmed regroup kept the renumbered
    data but re-seeded names in alignment order, so 'alpha' labelled gamma's
    data (and a second regroup carried the mismatch through)."""

    _EXPECTED = "data:[('alpha', 1.0), ('beta', 2.0), ('delta', 4.0), ('gamma', 3.0)]"

    def test_names_stay_with_their_data(self):
        at = AppTest.from_function(_soft_reset_script, default_timeout=DEFAULT_TIMEOUT)
        at.session_state['lipidsearch_alignment_text'] = _SOFT_RESET_ALIGNMENT_TEXT
        at.run()
        assert at.text[0].value == self._EXPECTED
        # Regroup: swap the conditions, then confirm.
        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        at.sidebar.multiselect(key='select_A').set_value(['s3', 's4']).run()
        at.sidebar.multiselect(key='select_B').set_value(['s1', 's2']).run()
        at.sidebar.checkbox(key='confirm_checkbox').check().run()
        assert at.text[0].value == self._EXPECTED
        # Start Over, then regroup again and confirm.
        at.session_state['_soft'] = True
        at.run()
        assert at.text[0].value == self._EXPECTED
        at.sidebar.radio(key='grouping_radio').set_value('No').run()
        at.sidebar.multiselect(key='select_A').set_value(['s1', 's2']).run()
        at.sidebar.multiselect(key='select_B').set_value(['s3', 's4']).run()
        at.sidebar.checkbox(key='confirm_checkbox').check().run()
        assert not at.exception
        assert at.text[0].value == self._EXPECTED
