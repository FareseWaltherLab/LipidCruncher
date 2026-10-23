"""AIStudioWorkflow: the AI Studio's data setup, without Streamlit."""
from pathlib import Path

import pandas as pd
import pytest

from app.models.experiment import ExperimentConfig
from app.services.format_detection import DataFormat
from app.workflows.ai_studio import AIStudioWorkflow as W

SAMPLES = Path(__file__).parents[2] / 'sample_datasets'


def _exp(conditions, counts):
    return ExperimentConfig(conditions_list=conditions, number_of_samples_list=counts)


class TestBlockedStandardFormat:
    @pytest.mark.parametrize('fmt', [
        DataFormat.LIPIDSEARCH, DataFormat.MSDIAL, DataFormat.METABOLOMICS_WORKBENCH,
    ])
    def test_standard_formats_are_blocked(self, fmt):
        assert W.is_blocked_standard_format(fmt.value, None)

    def test_a_pipeline_import_is_never_blocked(self):
        assert not W.is_blocked_standard_format(DataFormat.LIPIDSEARCH.value, 'raw')

    @pytest.mark.parametrize('fmt', [DataFormat.GENERIC.value, None])
    def test_other_files_are_not_blocked(self, fmt):
        assert not W.is_blocked_standard_format(fmt, None)


class TestReadUpload:
    def test_comma_separated(self):
        df, fmt = W.read_upload(b"Lipid,WT_1,KO_1\nPC 34:1,1.0,2.0\n")
        assert list(df.columns) == ['Lipid', 'WT_1', 'KO_1']
        assert df['KO_1'].iloc[0] == 2.0
        assert fmt == DataFormat.GENERIC.value

    def test_tab_separated_with_a_csv_name(self):
        df, _ = W.read_upload(b"Lipid\tWT_1\tKO_1\nPC 34:1\t1.0\t2.0\n")
        assert list(df.columns) == ['Lipid', 'WT_1', 'KO_1']

    def test_lipidsearch(self):
        df, fmt = W.read_upload((SAMPLES / 'lipidsearch5_test_dataset.csv').read_bytes())
        assert fmt == DataFormat.LIPIDSEARCH.value
        assert not df.empty

    def test_workbench_is_recognised_from_its_text(self):
        _, fmt = W.read_upload((SAMPLES / 'mw_test_dataset.csv').read_bytes())
        assert fmt == DataFormat.METABOLOMICS_WORKBENCH.value

    def test_an_empty_table_has_no_format(self):
        df, fmt = W.read_upload(b"Lipid,WT_1\n")
        assert df.empty and fmt is None

    def test_an_unreadable_file_raises(self):
        with pytest.raises(pd.errors.EmptyDataError):
            W.read_upload(b"")


class TestRawImportFrame:
    def test_the_first_available_frame_wins(self):
        a, b, c = pd.DataFrame({'a': [1]}), pd.DataFrame({'b': [1]}), pd.DataFrame({'c': [1]})
        assert W.raw_import_frame(a, b, c) is a
        assert W.raw_import_frame(None, b, c) is b
        assert W.raw_import_frame(None, None, c) is c
        assert W.raw_import_frame(None, None, None) is None


class TestImportSetup:
    SIDEBAR = _exp(['WT', 'KO'], [3, 3])
    HANDOFF = _exp(['WT', 'KO', 'BQC'], [4, 4, 4])
    QC = _exp(['WT', 'KO', 'BQC'], [3, 4, 4])

    def _setup(self, source, handoff=HANDOFF, qc=QC):
        return W.import_setup(source, self.SIDEBAR, 'KO', handoff, 'BQC', qc)

    def test_raw_follows_the_sidebar(self):
        assert self._setup('raw') == (self.SIDEBAR, 'KO')

    def test_normalized_uses_the_handoff(self):
        assert self._setup('normalized') == (self.HANDOFF, 'BQC')

    def test_qc_filtered_uses_the_qc_experiment(self):
        assert self._setup('qc_filtered') == (self.QC, 'BQC')

    def test_before_quality_check_the_sidebar_is_used(self):
        assert self._setup('normalized', handoff=None, qc=None) == (self.SIDEBAR, 'KO')

    def test_a_bqc_condition_pca_removed_is_dropped(self):
        qc = _exp(['WT', 'KO'], [4, 4])
        assert self._setup('qc_filtered', qc=qc) == (qc, None)

    def test_without_an_experiment_the_label_is_kept(self):
        assert W.import_setup('raw', None, 'BQC', None, None, None) == (None, 'BQC')


class TestSampleColumns:
    def test_intensity_comes_before_concentration(self):
        df = pd.DataFrame(columns=['LipidMolec', 'concentration[s1]', 'intensity[s1]'])
        assert W.detect_sample_prefix(df) == 'intensity'
        assert W.detected_sample_columns(df) == ('intensity', ['intensity[s1]'])

    def test_concentration(self):
        df = pd.DataFrame(columns=['LipidMolec', 'concentration[s1]', 'concentration[s2]'])
        assert W.detected_sample_columns(df) == (
            'concentration', ['concentration[s1]', 'concentration[s2]'],
        )

    def test_neither(self):
        df = pd.DataFrame(columns=['Lipid', 'Area[WT_1]', 'intensity'])
        assert W.detect_sample_prefix(df) is None
        assert W.detected_sample_columns(df) == (None, [])


class TestDeriveExperiment:
    def test_one_condition_over_the_intensity_columns(self):
        df = pd.DataFrame({'LipidMolec': ['a'], 'intensity[s1]': [1], 'intensity[s2]': [2], 'x': [3]})
        assert W.derive_experiment(df) == _exp(['All Samples'], [2])

    def test_numeric_columns_without_intensity_ones(self):
        df = pd.DataFrame({'Lipid': ['a'], 'A': [1.0], 'B': [2], 'C': [3]})
        assert W.derive_experiment(df) == _exp(['All Samples'], [3])

    def test_at_least_one_sample(self):
        assert W.derive_experiment(pd.DataFrame({'Lipid': ['a']})) == _exp(['All Samples'], [1])


class TestApplyUserExperiment:
    DF = pd.DataFrame({
        'Lipid': ['PC 34:1'], 'WT_1': [1.0], 'KO_1': [2.0], 'WT_2': [3.0], 'KO_2': [4.0],
    })
    TO_COL = {c: c for c in ['WT_1', 'KO_1', 'WT_2', 'KO_2']}

    def test_conditions_become_contiguous_s_blocks(self):
        out, exp = W.apply_user_experiment(
            self.DF, ['WT', 'KO'], [['WT_1', 'WT_2'], ['KO_1', 'KO_2']], self.TO_COL, 'intensity',
        )
        assert list(out.columns) == [
            'Lipid', 'intensity[s1]', 'intensity[s2]', 'intensity[s3]', 'intensity[s4]',
        ]
        assert out.iloc[0, 1:].tolist() == [1.0, 3.0, 2.0, 4.0]
        assert exp == _exp(['WT', 'KO'], [2, 2])
        assert exp.samples_for('KO') == ['s3', 's4']

    def test_unpicked_columns_stay_unless_named_like_samples(self):
        df = pd.DataFrame({
            'LipidMolec': ['a'], 'note': ['x'],
            'intensity[s1]': [1], 'intensity[s2]': [2], 'intensity[s3]': [3],
        })
        out, exp = W.apply_user_experiment(
            df, ['C'], [['s2', 's3']],
            {'s2': 'intensity[s2]', 's3': 'intensity[s3]'}, 'intensity',
        )
        assert list(out.columns) == ['LipidMolec', 'note', 'intensity[s1]', 'intensity[s2]']
        assert out.iloc[0, 2:].tolist() == [2, 3]

    def test_the_output_prefix_is_used(self):
        df = pd.DataFrame({'LipidMolec': ['a'], 'concentration[s1]': [1.0], 'concentration[s2]': [2.0]})
        out, _ = W.apply_user_experiment(
            df, ['B', 'A'], [['s2'], ['s1']],
            {'s1': 'concentration[s1]', 's2': 'concentration[s2]'}, 'concentration',
        )
        assert out.to_dict('list') == {
            'LipidMolec': ['a'], 'concentration[s1]': [2.0], 'concentration[s2]': [1.0],
        }


class TestApplySnapshot:
    def test_equal_inputs_give_equal_snapshots(self):
        args = ('f', 'intensity', ['a', 'b'], ['WT'], [['a', 'b']], None)
        assert W.apply_snapshot(*args) == W.apply_snapshot(*args)
        hash(W.apply_snapshot(*args))

    def test_a_regroup_changes_it(self):
        one = W.apply_snapshot('f', 'intensity', ['a', 'b'], ['WT'], [['a', 'b']], None)
        two = W.apply_snapshot('f', 'intensity', ['a', 'b'], ['WT', 'KO'], [['a'], ['b']], None)
        assert one != two
