"""
UI tests for the AI Studio sidebar.

Uses Streamlit's AppTest to drive the Confirm Inputs flow and assert that
the data-dependent tabs lock/unlock in lockstep with the checkbox.
"""
from streamlit.testing.v1 import AppTest

DEFAULT_TIMEOUT = 15


def studio_confirm_flow_script():
    """Render the AI Studio sidebar with pre-seeded data so the flow reaches
    the Confirm Inputs checkbox using all default selections.

    Emits `snapshot_set:<bool>` where the bool mirrors whether
    `studio_applied_snapshot` is set — i.e. whether the data tabs are unlocked.
    """
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_studio_df_original') is None:
        df = pd.DataFrame({
            'LipidMolec': ['PC 34:1', 'PE 36:2'],
            'intensity[s1]': [1.0, 2.0],
            'intensity[s2]': [3.0, 4.0],
            'intensity[s3]': [5.0, 6.0],
            'intensity[s4]': [7.0, 8.0],
            'intensity[s5]': [9.0, 10.0],
            'intensity[s6]': [11.0, 12.0],
        })
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_df_original = df
        # Prefix '_pipeline_import' so the empty file_uploader on rerun isn't
        # mistaken for a user removing their file (which would wipe state).
        st.session_state.chat_studio_file_id = '_pipeline_import_test'

    from app.ui.ai_studio_page import _display_studio_sidebar
    _display_studio_sidebar()

    is_unlocked = st.session_state.get('studio_applied_snapshot') is not None
    st.text(f"snapshot_set:{is_unlocked}")


def _snapshot_set(at):
    for t in at.text:
        if t.value.startswith('snapshot_set:'):
            return t.value.endswith('True')
    raise AssertionError("snapshot_set marker not rendered")


class TestAIStudioConfirmInputs:
    """Confirm Inputs checkbox gates the data-dependent tabs."""

    def test_tabs_locked_before_confirm(self):
        at = AppTest.from_function(
            studio_confirm_flow_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        assert _snapshot_set(at) is False

    def test_checking_confirm_unlocks_tabs(self):
        at = AppTest.from_function(
            studio_confirm_flow_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert _snapshot_set(at) is True

    def test_unchecking_confirm_relocks_tabs(self):
        """Regression: unchecking the box must clear studio_applied_snapshot
        so the data tabs turn inactive again."""
        at = AppTest.from_function(
            studio_confirm_flow_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert _snapshot_set(at) is True

        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(False).run()
        assert _snapshot_set(at) is False


def sample_columns_picker_script():
    """Render only the Sample Columns block over a generic-format frame whose
    first column ('lipids') is text. No intensity[*]/concentration[*] columns
    exist, so the manual multiselect renders immediately.

    Emits `result:<bool>` reflecting whether the block returned a selection.
    """
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    df = pd.DataFrame({
        'lipids': ['PC 34:1', 'PE 36:2', 'SM 42:1'],   # text → not a sample
        'CalcMass': [758.5, 744.5, 700.5],             # numeric metadata name
        'Sample1': [1.0, 2.0, 3.0],
        'Sample2': [4.0, 5.0, 6.0],
    })
    from app.ui.ai_studio_page import _display_sample_columns_sidebar
    result = _display_sample_columns_sidebar(df)
    st.text(f"result:{result is not None}")


def _returned_selection(at):
    for t in at.text:
        if t.value.startswith('result:'):
            return t.value.endswith('True')
    raise AssertionError("result marker not rendered")


class TestSampleColumnNumericValidation:
    """Picking a non-numeric column as a sample must be rejected."""

    def test_non_numeric_pick_is_rejected(self):
        """Regression: selecting the text 'lipids' column produced no error
        and was accepted as a sample column."""
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.multiselect(
            key='studio_sample_cols_select',
        ).set_value(['lipids']).run()

        assert len(at.sidebar.error) > 0
        assert _returned_selection(at) is False

    def test_numeric_picks_are_accepted(self):
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.multiselect(
            key='studio_sample_cols_select',
        ).set_value(['Sample1', 'Sample2']).run()

        assert len(at.sidebar.error) == 0
        assert _returned_selection(at) is True


class TestSampleColumnPreselection:
    """Numeric columns are pre-selected so the user needn't hand-pick all."""

    def test_numeric_columns_are_preselected(self):
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        # Numeric, non-metadata columns pre-checked; text 'lipids' and the
        # lipidomics-named numeric 'CalcMass' are not.
        assert ms.value == ['Sample1', 'Sample2']

    def test_metadata_named_column_stays_selectable(self):
        """The lipidomics name list only nudges the default — it must not
        remove a column from the options (AI Studio accepts any input)."""
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert 'CalcMass' in ms.options
