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
    """A column without numbers can't be a sample column."""

    def test_text_columns_are_not_offered(self):
        """Regression: the text 'lipids' column could be picked as a sample
        column (the sandbox would have turned it into all-NaN)."""
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert 'lipids' not in ms.options

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
    """The picker starts from a guess the user reviews: the numeric columns
    whose names don't look like metadata."""

    def test_numeric_non_metadata_columns_are_preselected(self):
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert ms.value == ['Sample1', 'Sample2']

    def test_all_columns_stay_selectable(self):
        """Numeric metadata is not preselected but stays an option."""
        at = AppTest.from_function(
            sample_columns_picker_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        ms = at.sidebar.multiselect(key='studio_sample_cols_select')
        assert 'CalcMass' in ms.options
        assert 'Sample1' in ms.options


def assign_conditions_script():
    """Render only the Assign Conditions block over the sample names held
    in session state (default: KO-1, KO-2, WT-1, WT-2)."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    names = st.session_state.get('_test_names', ['KO-1', 'KO-2', 'WT-1', 'WT-2'])
    from app.ui.ai_studio_page import _display_assign_conditions_sidebar
    st.text(f"assignment:{_display_assign_conditions_sidebar(names)}")


def _edit_conditions(at, edits):
    """Edit Assign Conditions cells: edits maps row index → new condition."""
    at.session_state['studio_conditions_editor'] = {
        'edited_rows': {i: {'condition': c} for i, c in edits.items()},
        'added_rows': [], 'deleted_rows': [],
    }
    return at.run()


class TestAssignConditions:
    """One editable sample → condition table replaces the counts and the
    per-condition pickers."""

    def test_conditions_are_guessed_from_the_names(self):
        at = AppTest.from_function(
            assign_conditions_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        assert _marker_value(at, 'assignment:') == (
            "(['KO', 'WT'], [['KO-1', 'KO-2'], ['WT-1', 'WT-2']])"
        )

    def test_names_without_a_pattern_start_in_one_condition(self):
        at = AppTest.from_function(
            assign_conditions_script, default_timeout=DEFAULT_TIMEOUT,
        )
        at.session_state['_test_names'] = ['s1', 's2', 's3']
        at.run()
        assert _marker_value(at, 'assignment:') == (
            "(['Condition_1'], [['s1', 's2', 's3']])"
        )

    def test_editing_a_cell_regroups(self):
        """Samples need not be contiguous; conditions keep the order they
        first appear in."""
        at = AppTest.from_function(
            assign_conditions_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        _edit_conditions(at, {1: 'WT', 3: 'Control'})
        assert _marker_value(at, 'assignment:') == (
            "(['KO', 'WT', 'Control'], [['KO-1'], ['KO-2', 'WT-1'], ['WT-2']])"
        )

    def test_a_sample_without_a_condition_is_an_error(self):
        at = AppTest.from_function(
            assign_conditions_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        _edit_conditions(at, {0: '  '})
        assert _marker_value(at, 'assignment:') == 'None'
        assert any("needs a condition" in e.value for e in at.sidebar.error)

    def test_other_samples_start_from_a_fresh_guess(self):
        """Edits refer to rows of the old table, so a new sample list must
        not inherit them."""
        at = AppTest.from_function(
            assign_conditions_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        _edit_conditions(at, {0: 'Other'})
        at.session_state['_test_names'] = ['A_1', 'A_2', 'B_1', 'B_2']
        at.run()
        assert _marker_value(at, 'assignment:') == (
            "(['A', 'B'], [['A_1', 'A_2'], ['B_1', 'B_2']])"
        )


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


def test_page_states_how_the_data_is_used():
    """S8: the page never said that turns are logged on the server."""
    at = AppTest.from_function(
        studio_preview_script, default_timeout=DEFAULT_TIMEOUT,
    ).run()
    notice = [i.value for i in at.info if "How your data is used" in i.value]
    assert len(notice) == 1
    assert "saved on the LipidCruncher server" in notice[0]
    assert "first 3 rows" in notice[0]
    assert "does not use this data to train" in notice[0]


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
        from app.workflows.ai_studio import AIStudioWorkflow

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
            DataFormat.METABOLOMICS_WORKBENCH.value in AIStudioWorkflow.BLOCKED_FORMATS
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
    st.text(f"conditions_table:{st.session_state.get('studio_conditions')}")


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
        assert _marker(at, 'conditions_table:') == (
            "{'s1': 'WT', 's2': 'WT', 's3': 'KO'}"
        )

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
    and `processing_frame` hold the 3 survivors, `standardized_df` still
    holds all 4 samples."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    from app.models.sample_frame import SampleFrame
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('standardized_df') is None:
        st.session_state.standardized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            'intensity[s1]': [1.0], 'intensity[s2]': [3.0],
            'intensity[s3]': [5.0], 'intensity[s4]': [7.0],
        })
        st.session_state.experiment = ExperimentConfig(
            conditions_list=['WT', 'KO'], number_of_samples_list=[2, 1],
        )
        st.session_state.processing_frame = SampleFrame.from_upload(
            pd.DataFrame({
                'LipidMolec': ['PC 34:1'],
                'intensity[s1]': [1.0], 'intensity[s2]': [3.0],
                'intensity[s3]': [5.0],
            }),
            st.session_state.experiment,
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


def pipeline_import_qc_dropped_bqc_script():
    """Full AI Studio page after Quality Check's PCA removal dropped every
    BQC sample: `qc_experiment` has no BQC condition, while the sidebar's
    `bqc_label` (and the pre-QC `experiment`) still name it."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.models.experiment import ExperimentConfig
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('normalized_df') is None:
        st.session_state.standardized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            **{f'intensity[s{i}]': [float(i)] for i in range(1, 7)},
        })
        st.session_state.normalized_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            **{f'concentration[s{i}]': [float(i)] for i in range(1, 7)},
        })
        st.session_state.qc_continuation_df = pd.DataFrame({
            'LipidMolec': ['PC 34:1'],
            **{f'concentration[s{i}]': [float(i)] for i in range(1, 5)},
        })
        st.session_state.experiment = ExperimentConfig(
            conditions_list=['WT', 'KO', 'BQC'], number_of_samples_list=[2, 2, 2],
        )
        st.session_state.qc_experiment = ExperimentConfig(
            conditions_list=['WT', 'KO'], number_of_samples_list=[2, 2],
        )
        st.session_state.bqc_label = 'BQC'

    from app.ui.ai_studio_page import display_ai_studio_page
    display_ai_studio_page()
    st.text(f"bqc:{st.session_state.get('chat_studio_bqc_label')}")


class TestPipelineImportBQC:
    def test_qc_import_without_the_bqc_condition_has_no_bqc(self):
        """Regression: the sidebar's 'BQC' label was copied into the Studio's
        BQC radio, which falls back to its first option when the label is
        missing, so WT was silently treated as the BQC condition."""
        at = AppTest.from_function(
            pipeline_import_qc_dropped_bqc_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert not at.exception
        assert _marker(at, 'bqc:') == 'None'
        assert at.session_state['studio_bqc_radio'] == 'No'

    def test_normalized_import_keeps_bqc(self):
        at = AppTest.from_function(
            pipeline_import_qc_dropped_bqc_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='ai_studio_pipeline_choice').set_value(
            'Normalized data (from pipeline)'
        ).run()
        at.sidebar.button(key='ai_studio_import_pipeline').click().run()
        assert not at.exception
        assert _marker(at, 'bqc:') == 'BQC'


def data_questions_regroup_script():
    """Full AI Studio page over six samples whose values identify them, with
    the Data Questions panel replaced by a recorder that prints what it got.
    """
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_studio_df_original') is None:
        df = pd.DataFrame({'LipidMolec': ['PC 34:1']})
        for i in range(1, 7):
            df[f'intensity[s{i}]'] = [float(i)]  # value = original sample no.
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_df_original = df
        st.session_state.chat_studio_file_id = '_pipeline_import_test'
        st.session_state.chat_data_questions_session = 'stale session'

    import app.ui.ai_studio_page as page

    def _record(source_df):
        experiment = st.session_state.chat_studio_experiment
        first = experiment.individual_samples_list[0]
        values = [source_df[f'intensity[{s}]'].iloc[0] for s in first]
        st.text(f"dq_first_condition:{values}")
        st.text(f"dq_session:{st.session_state.chat_data_questions_session}")

    page.display_data_questions_chat = _record
    page.display_ai_studio_page()


def _marker_value(at, prefix):
    for t in at.text:
        if t.value.startswith(prefix):
            return t.value[len(prefix):]
    raise AssertionError(f"{prefix} marker not rendered")


class TestDataQuestionsMatchesTheExperiment:
    """Regression (Q3): Data Questions got the original, un-renamed frame
    with the experiment built on the renamed s1..sN. After a regroup the
    prompt's sample lists and every lc.* call pointed at other samples'
    columns, so group comparisons were silently wrong."""

    def test_regrouped_condition_reads_its_own_samples(self):
        at = AppTest.from_function(
            data_questions_regroup_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        # s1..s6 start in one condition; split them into A = s1, s3, s5 and
        # B = s2, s4, s6, so A's samples are renamed s1..s3.
        _edit_conditions(at, {0: 'A', 1: 'B', 2: 'A', 3: 'B', 4: 'A', 5: 'B'})
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()

        # Condition A holds the original s1, s3, s5, whatever they are called now.
        assert _marker_value(at, 'dq_first_condition:') == '[1.0, 3.0, 5.0]'

    def test_applying_an_experiment_restarts_the_data_questions_chat(self):
        at = AppTest.from_function(
            data_questions_regroup_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()

        # The old conversation referred to the old column names.
        assert _marker_value(at, 'dq_session:') == 'None'


def studio_exclusion_script():
    """The AI Studio sidebar over six named samples (WT_1..3, KO_1..3) whose
    values identify them, reaching Confirm Inputs with default selections:
    conditions WT and KO, guessed from the names."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_studio_df_original') is None:
        df = pd.DataFrame({'LipidMolec': ['PC 34:1']})
        for i, name in enumerate(['WT_1', 'WT_2', 'WT_3', 'KO_1', 'KO_2', 'KO_3']):
            df[f'intensity[{name}]'] = [float(i + 1)]
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_df_original = df
        st.session_state.chat_studio_file_id = '_pipeline_import_test'

    from app.ui.ai_studio_page import _display_studio_sidebar
    _display_studio_sidebar()

    exp = st.session_state.get('chat_studio_experiment')
    st.text(f"conditions:{exp.conditions_list if exp else None}")
    st.text(f"counts:{exp.number_of_samples_list if exp else None}")
    applied = st.session_state.get('chat_studio_df')
    st.text(
        "values:"
        f"{[applied[c].iloc[0] for c in applied.columns if c.startswith('intensity[')]}"
    )
    st.text(f"columns:{[c for c in applied.columns if c.startswith('intensity[')]}")
    st.text(f"bqc:{st.session_state.get('chat_studio_bqc_label')}")


def _exclusion_app(excluded):
    at = AppTest.from_function(
        studio_exclusion_script, default_timeout=DEFAULT_TIMEOUT,
    ).run()
    at.sidebar.multiselect(key='studio_exclude_samples').set_value(excluded).run()
    return at


class TestStudioSampleExclusion:
    """The AI Studio sidebar can exclude samples, as the main sidebar can."""

    def test_excluded_sample_is_dropped_and_survivors_renumbered(self):
        at = _exclusion_app(['WT_2'])
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert _marker_value(at, 'conditions:') == "['WT', 'KO']"
        assert _marker_value(at, 'counts:') == '[2, 3]'
        assert _marker_value(at, 'columns:') == str(
            [f'intensity[s{i}]' for i in range(1, 6)]
        )
        # WT_2 (value 2) is gone; WT_3 is the new s2.
        assert _marker_value(at, 'values:') == '[1.0, 3.0, 4.0, 5.0, 6.0]'

    def test_excluding_a_whole_condition_drops_it(self):
        at = _exclusion_app(['WT_1', 'WT_2', 'WT_3'])
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert _marker_value(at, 'conditions:') == "['KO']"
        assert _marker_value(at, 'values:') == '[4.0, 5.0, 6.0]'

    def test_excluding_the_bqc_condition_clears_bqc(self):
        at = AppTest.from_function(
            studio_exclusion_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='studio_bqc_radio').set_value('Yes').run()
        at.sidebar.radio(key='studio_bqc_label_radio').set_value('KO').run()
        at.sidebar.multiselect(key='studio_exclude_samples').set_value(
            ['KO_1', 'KO_2', 'KO_3']
        ).run()
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert _marker_value(at, 'conditions:') == "['WT']"
        assert _marker_value(at, 'bqc:') == 'None'
        assert any("continuing without BQC" in w.value for w in at.sidebar.warning)

    def test_fewer_than_two_remaining_blocks_confirm(self):
        at = _exclusion_app(['WT_1', 'WT_2', 'WT_3', 'KO_1', 'KO_2'])
        assert any("At least two samples" in e.value for e in at.sidebar.error)
        assert 'studio_confirm_checkbox' not in [c.key for c in at.sidebar.checkbox]

    def test_confirm_summary_matches_the_main_sidebar(self):
        """The summary used to read "s1-s2-s3 correspond to WT", which hid
        the dataset's own sample names and the exclusions."""
        at = _exclusion_app(['WT_2'])
        text = [m.value for m in at.sidebar.markdown]
        captions = [c.value for c in at.sidebar.caption]
        assert "5 samples in 2 conditions" in text
        assert "**WT** · 2 samples" in text
        assert "WT\\_1, WT\\_3" in captions
        assert "Excluded: WT\\_2" in captions


def exclusion_lifecycle_script():
    """Snapshot an exclusion for navigation, then clear it as a new upload
    or pipeline import does."""
    import streamlit as st
    from app.adapters.streamlit_adapter import StreamlitAdapter
    from app.ui.ai_studio_page import _clear_data_sessions
    StreamlitAdapter.initialize_session_state()

    st.session_state.studio_exclude_samples = ['WT_2']
    StreamlitAdapter.preserve_ai_studio_widgets()
    st.text(
        "preserved:"
        f"{st.session_state.get('_nav_preserved_studio_exclude_samples')}"
    )
    _clear_data_sessions()
    st.text(f"cleared:{'studio_exclude_samples' not in st.session_state}")


def test_exclusion_survives_navigation_and_resets_with_new_data():
    at = AppTest.from_function(
        exclusion_lifecycle_script, default_timeout=DEFAULT_TIMEOUT,
    ).run()
    assert _marker_value(at, 'preserved:') == "['WT_2']"
    assert _marker_value(at, 'cleared:') == 'True'


def transform_unlock_script():
    """Full AI Studio page over a long-format file (one row per lipid and
    sample), with the Transform and Data Questions panels replaced by
    recorders. The sidebar experiment is never confirmed."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    if st.session_state.get('chat_studio_df_original') is None:
        df = pd.DataFrame({
            'lipid': ['PC 34:1', 'PC 34:1', 'PE 36:2', 'PE 36:2'],
            'sample': ['A', 'B', 'A', 'B'],
            'area': [1.0, 2.0, 3.0, 4.0],
        })
        st.session_state.chat_studio_df = df
        st.session_state.chat_studio_df_original = df
        st.session_state.chat_studio_file_id = '_pipeline_import_test'

    import app.ui.ai_studio_page as page
    page.display_transform_chat = lambda df: st.text("transform_chat_shown")
    page.display_data_questions_chat = lambda df: st.text("dq_chat_shown")
    page.display_ai_studio_page()


def test_transform_opens_before_the_experiment_is_confirmed():
    """Regression: Transform to Generic was locked behind Confirm Inputs,
    though it only needs the uploaded file and a long-format file has no
    sample columns to define until it is transformed."""
    at = AppTest.from_function(
        transform_unlock_script, default_timeout=DEFAULT_TIMEOUT,
    ).run()
    assert not at.exception
    shown = [t.value for t in at.text]
    assert "transform_chat_shown" in shown
    # Data Questions still needs the experiment.
    assert "dq_chat_shown" not in shown


def sample_families_script():
    """Render the Sample Columns block over a LipidSearch-like export: two
    measurement families with a column per sample, plus per-lipid RT."""
    import streamlit as st
    import pandas as pd
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    df = pd.DataFrame({
        'LipidMolec': ['PC 34:1'], 'RT': [5.2],
        'Area[KO-1]': [1.0], 'Area[KO-2]': [2.0], 'Area[WT-1]': [3.0],
        'Height[KO-1]': [10.0], 'Height[KO-2]': [20.0], 'Height[WT-1]': [30.0],
    })
    from app.ui.ai_studio_page import _display_sample_columns_sidebar
    st.text(f"result:{_display_sample_columns_sidebar(df)}")


class TestSampleColumnFamilies:
    """With many per-sample measurements (Area[...], Height[...]), the user
    picks the measurement instead of every column."""

    def test_first_family_is_preselected(self):
        at = AppTest.from_function(
            sample_families_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        radio = at.sidebar.radio(key='studio_sample_family')
        assert radio.options == [
            'Area[…] (3 columns)', 'Height[…] (3 columns)', 'Pick columns myself',
        ]
        assert at.sidebar.multiselect(key='studio_sample_cols_select').value == [
            'Area[KO-1]', 'Area[KO-2]', 'Area[WT-1]',
        ]

    def test_choosing_a_family_selects_its_columns(self):
        at = AppTest.from_function(
            sample_families_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='studio_sample_family').set_value(
            'Height[…] (3 columns)'
        ).run()
        assert at.sidebar.multiselect(key='studio_sample_cols_select').value == [
            'Height[KO-1]', 'Height[KO-2]', 'Height[WT-1]',
        ]

    def test_pick_columns_myself_starts_empty(self):
        at = AppTest.from_function(
            sample_families_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        at.sidebar.radio(key='studio_sample_family').set_value(
            'Pick columns myself'
        ).run()
        assert at.sidebar.multiselect(key='studio_sample_cols_select').value == []
        assert _marker_value(at, 'result:') == 'None'


def nonstandard_upload_script():
    """Full AI Studio page with tests/fixtures/ai_studio_test_dataset.csv
    uploaded through the Studio's own uploader: a lipid table in no
    supported format (feature, class, RetentionTime, PrecursorMZ,
    QualityScore and Notes columns, then 15 samples KO-1 … WT-3), so it can
    only come in this way. AppTest cannot drive st.file_uploader, so the
    sidebar's uploader is replaced for the run by one returning the file;
    reading, format detection and the sidebar flow are the real ones. The
    Data Questions panel is a recorder."""
    import io
    import pathlib
    import streamlit as st
    import app
    from app.adapters.streamlit_adapter import StreamlitAdapter
    StreamlitAdapter.initialize_session_state()

    path = (
        pathlib.Path(app.__file__).parents[2]
        / 'tests' / 'fixtures' / 'ai_studio_test_dataset.csv'
    )
    data = path.read_bytes()
    upload = io.BytesIO(data)
    upload.name, upload.size = path.name, len(data)

    import app.ui.ai_studio_page as page

    def _record(df):
        exp = st.session_state.chat_studio_experiment
        st.text(f"dq_conditions:{exp.conditions_list}")
        st.text(f"dq_counts:{exp.number_of_samples_list}")
        st.text(f"dq_s4:{df['intensity[s4]'].iloc[0]}")

    page.display_data_questions_chat = _record
    page.display_transform_chat = lambda df: st.text("transform_chat_shown")
    st.sidebar.file_uploader = lambda *args, **kwargs: upload
    try:
        page.display_ai_studio_page()
    finally:
        del st.sidebar.file_uploader
    st.text(f"format:{st.session_state.get('chat_studio_detected_format')}")


class TestNonStandardUpload:
    """End to end over a file only the AI Studio accepts."""

    def _uploaded(self):
        at = AppTest.from_function(
            nonstandard_upload_script, default_timeout=DEFAULT_TIMEOUT,
        ).run()
        assert not at.exception
        return at

    def test_upload_is_accepted_and_transform_opens(self):
        at = self._uploaded()
        assert _marker_value(at, 'format:') == 'Unknown'
        assert not any("dataset detected" in w.value for w in at.sidebar.warning)
        assert "transform_chat_shown" in [t.value for t in at.text]

    def test_sidebar_guesses_the_samples_and_conditions(self):
        at = self._uploaded()
        samples = [
            f'{cond}-{i}'
            for cond in ['KO', 'KOWT', 'mut1', 'mut2', 'WT'] for i in (1, 2, 3)
        ]
        picker = at.sidebar.multiselect(key='studio_sample_cols_select')
        # Text and empty columns are not offered; numeric metadata is, but
        # is not preselected.
        assert 'feature' not in picker.options
        assert 'Notes' not in picker.options
        assert 'RetentionTime' in picker.options
        assert picker.value == samples

        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert _marker_value(at, 'dq_conditions:') == (
            "['KO', 'KOWT', 'mut1', 'mut2', 'WT']"
        )
        assert _marker_value(at, 'dq_counts:') == '[3, 3, 3, 3, 3]'

    def test_regrouped_samples_reach_data_questions(self):
        """Moving WT-3 into KO makes it KO's fourth sample, the new s4."""
        at = self._uploaded()
        _edit_conditions(at, {14: 'KO'})
        at.sidebar.checkbox(key='studio_confirm_checkbox').set_value(True).run()
        assert not at.exception
        assert _marker_value(at, 'dq_counts:') == '[4, 3, 3, 3, 2]'
        # WT-3's value in the first row of the file.
        assert _marker_value(at, 'dq_s4:') == '0.409898992'
