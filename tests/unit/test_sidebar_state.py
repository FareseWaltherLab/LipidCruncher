"""Unit tests for the adapter's sidebar regroup/exclusion memory
(``SidebarState``, inherited by ``StreamlitAdapter``).

The end-to-end behaviour (a regroup computed from the upload order, undone
by restoring it; exclusions that follow their sample through a regroup) is
pinned by the AppTests in tests/ui/test_sample_identity.py and
test_sidebar_state.py. These pin what each method writes.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.adapters.streamlit_adapter import StreamlitAdapter
from app.models.experiment import ExperimentConfig
from app.models.sample_frame import SampleFrame


class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def mock_st():
    with patch('app.adapters.sidebar_state.st') as st:
        st.session_state = MockSessionState()
        yield st


def _df(order=('a', 'b')):
    return pd.DataFrame({
        'LipidMolec': ['PC 34:1'],
        **{f'intensity[s{i + 1}]': [v] for i, v in enumerate(order)},
    })


class TestBeginRegroup:
    def test_first_call_snapshots_the_upload_order(self, mock_st):
        mock_st.session_state['sample_names'] = {'s1': 'ctrl', 's2': 'kd'}
        df = _df()

        base_df, base_names = StreamlitAdapter.begin_regroup(df)

        state = mock_st.session_state
        assert state['original_column_order'] == df.columns.tolist()
        pd.testing.assert_frame_equal(state['_pre_regroup_df'], df)
        assert state['_pre_regroup_df'] is not df, "snapshot must be a copy"
        assert state['_pre_regroup_sample_names'] == {'s1': 'ctrl', 's2': 'kd'}
        pd.testing.assert_frame_equal(base_df, df)
        assert base_names == {'s1': 'ctrl', 's2': 'kd'}

    def test_later_calls_return_the_snapshot_not_the_regrouped_data(self, mock_st):
        upload = _df(('a', 'b'))
        mock_st.session_state['sample_names'] = {'s1': 'ctrl', 's2': 'kd'}
        StreamlitAdapter.begin_regroup(upload)
        # A rerun arrives with already-regrouped data and remapped names.
        mock_st.session_state['sample_names'] = {'s1': 'kd', 's2': 'ctrl'}

        base_df, base_names = StreamlitAdapter.begin_regroup(_df(('b', 'a')))

        pd.testing.assert_frame_equal(base_df, upload)
        assert base_names == {'s1': 'ctrl', 's2': 'kd'}


class TestApplyAndUndoRegroup:
    @staticmethod
    def _regrouped():
        experiment = ExperimentConfig(
            n_conditions=2, conditions_list=['A', 'B'], number_of_samples_list=[1, 1],
        )
        return SampleFrame(
            df=_df(('b', 'a')), experiment=experiment,
            names={'s1': 'kd', 's2': 'ctrl'}, origin={'s1': 's2', 's2': 's1'},
        )

    def test_apply_records_names_origin_and_completion(self, mock_st):
        StreamlitAdapter.apply_regroup(self._regrouped())

        state = mock_st.session_state
        assert state['sample_names'] == {'s1': 'kd', 's2': 'ctrl'}
        assert state['regroup_origin'] == {'s1': 's2', 's2': 's1'}
        assert state['grouping_complete'] is True

    def test_undo_restores_the_snapshot_and_drops_it(self, mock_st):
        upload = _df(('a', 'b'))
        mock_st.session_state['sample_names'] = {'s1': 'ctrl', 's2': 'kd'}
        StreamlitAdapter.begin_regroup(upload)
        StreamlitAdapter.apply_regroup(self._regrouped())
        mock_st.session_state['_preserved_regroup_selections'] = {'A': ['s2']}

        restored = StreamlitAdapter.undo_regroup(_df(('b', 'a')))

        state = mock_st.session_state
        pd.testing.assert_frame_equal(restored, upload)
        pd.testing.assert_frame_equal(state['standardized_df'], upload)
        assert state['sample_names'] == {'s1': 'ctrl', 's2': 'kd'}
        assert state['regroup_origin'] is None
        assert state['grouping_complete'] is True
        assert state['_preserved_regroup_selections'] == {}
        for key in ('_pre_regroup_df', '_pre_regroup_sample_names',
                    'original_column_order'):
            assert key not in state

    def test_undo_without_a_snapshot_keeps_the_data(self, mock_st):
        df = _df()
        mock_st.session_state['standardized_df'] = 'untouched'

        assert StreamlitAdapter.undo_regroup(df) is df
        assert mock_st.session_state['standardized_df'] == 'untouched'
        assert mock_st.session_state['regroup_origin'] is None


class TestDropRegroupSnapshot:
    def test_restores_pre_regroup_names(self, mock_st):
        mock_st.session_state.update({
            '_pre_regroup_df': _df(), 'original_column_order': ['x'],
            '_pre_regroup_sample_names': {'s1': 'ctrl'},
            'sample_names': {'s1': 'remapped'},
        })

        StreamlitAdapter.drop_regroup_snapshot()

        state = mock_st.session_state
        assert state['sample_names'] == {'s1': 'ctrl'}
        assert '_pre_regroup_df' not in state
        assert 'original_column_order' not in state
        assert '_pre_regroup_sample_names' not in state

    def test_no_snapshot_is_a_noop(self, mock_st):
        mock_st.session_state['sample_names'] = {'s1': 'ctrl'}
        StreamlitAdapter.drop_regroup_snapshot()
        assert mock_st.session_state == {'sample_names': {'s1': 'ctrl'}}


class TestSetSampleNames:
    def test_not_regrouped_writes_only_the_names(self, mock_st):
        StreamlitAdapter.set_sample_names({'s1': 'edited'})
        assert mock_st.session_state == {'sample_names': {'s1': 'edited'}}

    def test_regrouped_also_stores_the_edit_under_the_upload_label(self, mock_st):
        mock_st.session_state.update({
            'regroup_origin': {'s1': 's2', 's2': 's1'},
            '_pre_regroup_sample_names': {'s1': 'ctrl', 's2': 'kd'},
        })

        StreamlitAdapter.set_sample_names({'s1': 'kd-edited', 's2': 'ctrl'})

        state = mock_st.session_state
        assert state['sample_names'] == {'s1': 'kd-edited', 's2': 'ctrl'}
        assert state['_pre_regroup_sample_names'] == {'s2': 'kd-edited', 's1': 'ctrl'}


class TestExcludedSamples:
    def test_save_keeps_exclusions_by_upload_label(self, mock_st):
        origin = {'s1': 's2', 's2': 's1', 's3': 's3'}

        StreamlitAdapter.save_excluded_samples(['s1', 's3'], origin)

        state = mock_st.session_state
        assert state['_preserved_excluded_samples'] == ['s2', 's3']
        assert state['_preserved_excluded_regroup_origin'] == origin

    def test_restore_prefers_the_live_widget_when_the_regroup_is_unchanged(self, mock_st):
        mock_st.session_state.update({
            'exclude_samples': ['s3'],
            '_preserved_excluded_samples': ['s1'],
            '_preserved_excluded_regroup_origin': None,
        })

        restored, origin = StreamlitAdapter.restore_excluded_samples()

        assert restored == ['s3']
        assert origin == {}
        assert mock_st.session_state['exclude_samples'] == ['s3']

    def test_restore_translates_saved_labels_after_a_regroup(self, mock_st):
        # Saved while not regrouped: the excluded sample was s2 at upload.
        mock_st.session_state.update({
            'exclude_samples': ['s2'],
            '_preserved_excluded_samples': ['s2'],
            '_preserved_excluded_regroup_origin': {},
            'regroup_origin': {'s1': 's2', 's2': 's1'},
        })

        restored, origin = StreamlitAdapter.restore_excluded_samples()

        assert restored == ['s1'], "upload s2 is now sidebar s1"
        assert origin == {'s1': 's2', 's2': 's1'}
        assert 'exclude_samples' not in mock_st.session_state, (
            "the stale widget key must go so the picker re-renders"
        )

    def test_restore_with_nothing_saved(self, mock_st):
        restored, origin = StreamlitAdapter.restore_excluded_samples()
        assert restored == []
        assert origin == {}


def test_adapter_exposes_the_sidebar_state_methods():
    for name in ('begin_regroup', 'apply_regroup', 'undo_regroup',
                 'drop_regroup_snapshot', 'set_sample_names',
                 'restore_excluded_samples', 'save_excluded_samples'):
        assert callable(getattr(StreamlitAdapter, name)), name
    assert not isinstance(StreamlitAdapter.begin_regroup, MagicMock)
