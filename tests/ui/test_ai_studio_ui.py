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
    """The picker starts empty — the user hand-picks every sample column."""

    def test_no_columns_are_preselected(self):
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        # Nothing pre-checked: the default is an empty selection.
        assert ms.value == []

    def test_all_columns_stay_selectable(self):
        """Every column remains an option (AI Studio accepts any input)."""
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert 'CalcMass' in ms.options
        assert 'Sample1' in ms.options


def group_samples_script():
    """Render only the Group Samples block (manual reassignment path) over a
    4-sample / 2-condition setup."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.ui.ai_studio_page import _display_group_samples_sidebar
    result = _display_group_samples_sidebar(
        ['s1', 's2', 's3', 's4'], ['A', 'B'], [2, 2],
    )
    st.text(f"group_result:{result}")


class TestGroupSamplesNoDuplicates:
    """A fully-assigned sample drops out of the other conditions' options,
    so it can't be picked twice (no duplicate-error needed)."""

    def test_assigned_samples_leave_later_options(self):
        at = AppTest.from_function(
            group_samples_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='studio_grouping_radio').set_value('No').run()
        # Fully assign condition A; its samples must vanish from B's options.
        at.sidebar.multiselect(
            key='studio_select_0',
        ).set_value(['s1', 's2']).run()

        assert at.sidebar.multiselect(key='studio_select_1').options == [
            's3', 's4',
        ]
        # No duplicate-error is rendered for the manual flow.
        assert len(at.sidebar.error) == 0

    def test_regrouping_shows_old_to_new_name_table(self):
        """A completed manual regroup renders an old → new name table,
        mirroring the main pipeline's 'New sample order after regrouping:'."""
        at = AppTest.from_function(
            group_samples_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='studio_grouping_radio').set_value('No').run()
        at.sidebar.multiselect(
            key='studio_select_0',
        ).set_value(['s2', 's1']).run()
        at.sidebar.multiselect(
            key='studio_select_1',
        ).set_value(['s4', 's3']).run()

        name_tables = [
            t.value for t in at.sidebar.dataframe
            if 'updated name' in t.value.columns
        ]
        assert len(name_tables) == 1
        name_df = name_tables[0]
        assert list(name_df.columns) == ['old name', 'updated name', 'condition']
        assert list(name_df['updated name']) == ['s1', 's2', 's3', 's4']
        assert list(name_df['condition']) == ['A', 'A', 'B', 'B']
        assert set(name_df['old name']) == {'s1', 's2', 's3', 's4'}


def studio_preview_script():
    """Render the full AI Studio page with a processed df whose sample columns
    are renamed (intensity[s1..s2]) relative to the original (intensity[A..B]).
    """
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_studio_df_original') is None:
        original = pd.DataFrame({
            'lipids': ['PC 34:1'],
            'intensity[A]': [1.0],
            'intensity[B]': [2.0],
        })
        processed = original.rename(columns={
            'intensity[A]': 'intensity[s1]',
            'intensity[B]': 'intensity[s2]',
        })
        st.session_state.chat_studio_df_original = original
        st.session_state.chat_studio_df = processed
        st.session_state.chat_studio_file_id = '_pipeline_import_test'

    from app.ui.ai_studio_page import display_ai_studio_page
    display_ai_studio_page()


class TestDataPreviewReflectsReorder:
    """The Data Preview shows the processed (reordered + renamed) df."""

    def test_preview_uses_processed_columns(self):
        at = AppTest.from_function(
            studio_preview_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        all_columns = [
            col for d in at.dataframe for col in list(d.value.columns)
        ]
        assert 'intensity[s1]' in all_columns
        assert 'intensity[A]' not in all_columns


def mw_block_script():
    """Seed the detected format to Metabolomics Workbench and report whether
    the standard-format block fires."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    from app.services.format_detection import DataFormat
    from app.ui.ai_studio_page import _is_blocked_standard_format
    st.session_state.chat_studio_detected_format = (
        DataFormat.METABOLOMICS_WORKBENCH.value
    )
    st.text(f"blocked:{_is_blocked_standard_format()}")


class TestMetabolomicsWorkbenchBlocked:
    """A raw Metabolomics Workbench upload is recognized as MW and blocked,
    steering the user to the main pipeline.

    Regression: AI Studio passed a DataFrame to detect_format(), which only
    recognizes MW from its raw text (a plain read_csv of the multi-section
    file yields a garbage 'Unnamed' frame). MW was therefore misclassified as
    'Unknown', slipped past the block, and offered no usable sample columns.
    """

    def test_raw_mw_text_detects_as_mw_not_dataframe(self):
        import os
        import pandas as pd
        from app.services.format_detection import (
            DataFormat, FormatDetectionService,
        )
        from app.ui.ai_studio_page import _AI_STUDIO_BLOCKED_FORMATS

        mw_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'sample_datasets', 'mw_test_dataset.csv',
        )
        with open(mw_path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()

        # The fix: detect from the raw text → MW.
        assert (
            FormatDetectionService.detect_format(text)
            == DataFormat.METABOLOMICS_WORKBENCH
        )
        # The old behavior: detecting from the DataFrame → Unknown (not blocked).
        df = pd.read_csv(mw_path)
        assert (
            FormatDetectionService.detect_format(df) == DataFormat.UNKNOWN
        )
        # MW now belongs to the blocked set, so the advisory fires.
        assert (
            DataFormat.METABOLOMICS_WORKBENCH.value in _AI_STUDIO_BLOCKED_FORMATS
        )

    def test_mw_format_triggers_block(self):
        at = AppTest.from_function(
            mw_block_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        assert any(t.value == 'blocked:True' for t in at.text)


def pipeline_import_script():
    """Render the pipeline-import sidebar with all three pipeline stages
    seeded, and report the resulting Studio provenance.

    The QC frame deliberately differs from the normalized one: one lipid is
    dropped (high CoV) and sample s4 is removed (PCA outlier), which renumbers
    the survivors to s1..s3. `qc_experiment` reflects that; the pre-QC
    `experiment` still claims 4 samples.
    """
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('normalized_df') is None:
        st.session_state.standardized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1', 'PE 36:2'],
            'intensity[s1]': [1.0, 2.0], 'intensity[s2]': [3.0, 4.0],
            'intensity[s3]': [5.0, 6.0], 'intensity[s4]': [7.0, 8.0],
        })
        st.session_state.normalized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1', 'PE 36:2'],
            'concentration[s1]': [1.0, 2.0], 'concentration[s2]': [3.0, 4.0],
            'concentration[s3]': [5.0, 6.0], 'concentration[s4]': [7.0, 8.0],
        })
        st.session_state.qc_continuation_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            'concentration[s1]': [1.0], 'concentration[s2]': [3.0],
            'concentration[s3]': [5.0],
        })
        st.session_state.experiment = ExperimentConfig(
            conditions_list=['WT', 'KO'], number_of_samples_list=[2, 2],
        )
        st.session_state.qc_experiment = ExperimentConfig(
            conditions_list=['WT', 'KO'], number_of_samples_list=[2, 1],
        )

    from app.ui.ai_studio_page import _offer_pipeline_import_sidebar
    _offer_pipeline_import_sidebar()

    imported = st.session_state.get('chat_studio_df')
    st.text(f"source:{st.session_state.get('chat_studio_pipeline_source')}")
    st.text(f"rows:{len(imported) if imported is not None else -1}")
    studio_exp = st.session_state.get('chat_studio_experiment')
    st.text(
        "samples:"
        f"{studio_exp.number_of_samples_list if studio_exp else None}"
    )
    st.text(f"n_samples_1:{st.session_state.get('studio_n_samples_1')}")


def _marker(at, prefix):
    for t in at.text:
        if t.value.startswith(prefix):
            return t.value[len(prefix):]
    raise AssertionError(f"{prefix} marker not rendered")


class TestQCFilteredPipelineImport:
    """AI Studio can import the Quality Check module's output, not just the
    raw and normalized frames."""

    def test_qc_option_offered_when_qc_data_exists(self):
        at = AppTest.from_function(
            pipeline_import_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        options = at.sidebar.radio(key='ai_studio_pipeline_choice').options
        assert options[0].startswith('QC-filtered')
        assert any(o.startswith('Normalized') for o in options)
        assert any(o.startswith('Raw') for o in options)

    def test_importing_qc_data_sets_qc_provenance(self):
        at = AppTest.from_function(
            pipeline_import_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert _marker(at, 'source:') == 'qc_filtered'
        # The QC frame has one lipid; the normalized frame has two.
        assert _marker(at, 'rows:') == '1'

    def test_qc_import_pairs_with_the_qc_updated_experiment(self):
        """Regression: PCA sample removal drops columns and renumbers the
        survivors. Pairing the QC frame with the pre-QC `experiment` would
        tell the Studio there are 4 samples when the data holds 3, so the
        sample-column mapping silently misaligns."""
        at = AppTest.from_function(
            pipeline_import_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        # qc_experiment is [2, 1]; the stale pre-QC experiment is [2, 2].
        assert _marker(at, 'samples:') == '[2, 1]'
        # The sidebar prefill must agree with it.
        assert _marker(at, 'n_samples_1:') == '1'

    def test_normalized_import_keeps_the_pre_qc_experiment(self):
        """The QC-updated experiment must not leak into the other sources —
        the normalized frame still has all 4 samples."""
        at = AppTest.from_function(
            pipeline_import_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='ai_studio_pipeline_choice').set_value(
            'Normalized data (from pipeline)'
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert _marker(at, 'source:') == 'normalized'
        assert _marker(at, 'samples:') == '[2, 2]'

    def test_raw_import_still_works(self):
        at = AppTest.from_function(
            pipeline_import_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='ai_studio_pipeline_choice').set_value(
            'Raw uploaded data (from pipeline)'
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert _marker(at, 'source:') == 'raw'


def pipeline_import_no_qc_script():
    """Same as `pipeline_import_script` but the user never visited Module 2,
    so there is no QC frame to offer."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('normalized_df') is None:
        st.session_state.normalized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            'concentration[s1]': [1.0], 'concentration[s2]': [3.0],
        })

    from app.ui.ai_studio_page import _offer_pipeline_import_sidebar
    _offer_pipeline_import_sidebar()


def pipeline_import_excluded_script():
    """Sample s4 was excluded in the sidebar before Module 1: `experiment`
    and `processing_df` hold the 3 survivors, `standardized_df` still holds
    all 4 samples."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('standardized_df') is None:
        st.session_state.standardized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            'intensity[s1]': [1.0], 'intensity[s2]': [3.0],
            'intensity[s3]': [5.0], 'intensity[s4]': [7.0],
        })
        st.session_state.processing_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            'intensity[s1]': [1.0], 'intensity[s2]': [3.0],
            'intensity[s3]': [5.0],
        })
        st.session_state.experiment = ExperimentConfig(
            conditions_list=['WT', 'KO'], number_of_samples_list=[2, 1],
        )

    from app.ui.ai_studio_page import _offer_pipeline_import_sidebar
    _offer_pipeline_import_sidebar()

    imported = st.session_state.get('chat_studio_df')
    n_cols = (
        sum(c.startswith('intensity[') for c in imported.columns)
        if imported is not None else -1
    )
    st.text(f"intensity_cols:{n_cols}")


class TestRawImportAfterSidebarExclusion:
    def test_raw_import_matches_the_experiment(self):
        """Regression: the Raw import paired the full `standardized_df` with
        the post-exclusion `experiment`, so the Studio saw 4 sample columns
        for an experiment of 3."""
        at = AppTest.from_function(
            pipeline_import_excluded_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert _marker(at, 'intensity_cols:') == '3'


class TestQCOptionHiddenWithoutQCData:
    def test_no_qc_option_before_module2_runs(self):
        at = AppTest.from_function(
            pipeline_import_no_qc_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        options = at.sidebar.radio(key='ai_studio_pipeline_choice').options
        assert not any(o.startswith('QC-filtered') for o in options)
