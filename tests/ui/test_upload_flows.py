"""Full-app tests for uploading your own data (sidebar/file_upload.py).

AppTest cannot upload files; the ``fake_uploads`` fixture hands the
uploaders a file instead, so these drive the real upload paths: reading,
error messages, Metabolomics Workbench text, and the LipidSearch 5.2
Alignment Setting step.
"""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP
from tests.ui.conftest import DEFAULT_TIMEOUT, FakeUpload

MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
SAMPLES = Path(__file__).parents[2] / 'sample_datasets'


def _app(format_prefix=None, uploads=None, **files):
    """Open the app, pick the format, then upload ``files`` (uploader label
    fragment -> FakeUpload) through ``uploads`` (the fake_uploads fixture)."""
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    if format_prefix is not None:
        select = at.sidebar.selectbox[0]
        select.set_value([o for o in select.options if o.startswith(format_prefix)][0]).run()
    if files:
        uploads.update(files)
        at.run()
    return at


def _errors(at):
    return [e.value for e in at.sidebar.error]


class TestUploadingADataset:

    def test_generic_csv_is_read_and_standardized(self, fake_uploads):
        at = _app(None, fake_uploads, dataset=FakeUpload(
            'mine.csv', (SAMPLES / 'generic_test_dataset.csv').read_bytes(),
        ))
        assert not at.exception
        assert any('uploaded successfully' in s.value for s in at.sidebar.success)
        assert at.session_state['raw_df'] is not None
        standardized = at.session_state['standardized_df']
        assert sum(c.startswith('intensity[') for c in standardized.columns) == 12

    def test_metabolomics_workbench_text_is_parsed(self, fake_uploads):
        at = _app('Metabolomics Workbench', fake_uploads, dataset=FakeUpload(
            'mw.csv', (SAMPLES / 'mw_test_dataset.csv').read_bytes(),
        ))
        assert not at.exception
        assert any('processed successfully' in s.value for s in at.sidebar.success)
        conditions = at.session_state['workbench_conditions']
        assert len(conditions) == 58
        assert 'Diet:Normal | BileAcid:water' in conditions.values()

    def test_empty_file_is_reported(self, fake_uploads):
        at = _app(None, fake_uploads, dataset=FakeUpload('empty.csv', b''))
        assert not at.exception
        assert any('appears to be empty' in e for e in _errors(at))
        assert at.session_state['raw_df'] is None

    def test_file_that_is_not_utf8_is_reported(self, fake_uploads):
        at = _app(None, fake_uploads, dataset=FakeUpload('latin1.csv', 'Lipid,Ä\nPC,1\n'.encode('latin-1')))
        assert not at.exception
        assert any('UTF-8' in e for e in _errors(at))

    def test_workbench_file_without_its_markers_is_reported(self, fake_uploads):
        at = _app('Metabolomics Workbench', fake_uploads, dataset=FakeUpload('plain.csv', b'Lipid,s1\nPC 34:1,1\n'))
        assert not at.exception
        assert any('section markers' in e for e in _errors(at))
        assert at.session_state['raw_df'] is None


class TestSampleData:

    def test_clear_and_upload_drops_the_sample(self):
        at = _app()
        at.sidebar.button(key='load_sample').click().run()
        assert at.session_state['using_sample_data'] is True
        [b for b in at.sidebar.button if b.label == 'Clear & Upload Your Data'][0].click().run()
        assert not at.exception
        assert at.session_state['using_sample_data'] is False
        assert at.session_state['raw_df'] is None


# Control mouse 01 (s1-1 negative, s1-2 positive), Treated mouse 02.
_ALIGNMENT = (
    "*Parameters setting\n"
    "NormalizeType\tNONE\n"
    "\n"
    "*Target search job\n"
    "J\tm01n.raw\ts1-1\t*Control\tu1\n"
    "J\tm01p.raw\ts1-2\t*Control\tu2\n"
    "J\tm02n.raw\ts2-1\tTreated\tu3\n"
    "J\tm02p.raw\ts2-2\tTreated\tu4\n"
)


def _dual_polarity_export() -> bytes:
    df = pd.DataFrame({
        'LipidMolec': ['PC(16:0_18:1)', 'FA(15:0)'],
        'ClassKey': ['PC', 'FA'],
        'CalcMass': [757.5, 242.2],
        'BaseRt': [10.0, 3.0],
        'TotalGrade': ['A', 'A'],
        'TotalSmpIDRate(%)': [100.0, 100.0],
        'FAKey': ['(16:0_18:1)', '(15:0)'],
        'OriginalArea[s1-1]': [None, 300.0], 'OriginalArea[s1-2]': [200.0, None],
        'OriginalArea[s2-1]': [None, 330.0], 'OriginalArea[s2-2]': [220.0, None],
    })
    return df.to_csv(sep='\t', index=False).encode()


class TestLipidSearchAlignment:
    """A 5.2 condition-grouped export needs its Alignment Setting file."""

    def test_asks_for_the_alignment_file(self, fake_uploads):
        at = _app('LipidSearch', fake_uploads, dataset=FakeUpload('ls52.csv', _dual_polarity_export()))
        assert not at.exception
        assert any('Alignment Setting file' in i.value for i in at.sidebar.info)
        assert 'lipidsearch_alignment_text' not in at.session_state

    def test_alignment_merges_runs_and_fills_in_the_experiment(self, fake_uploads):
        at = _app('LipidSearch', fake_uploads, dataset=FakeUpload('ls52.csv', _dual_polarity_export()))
        fake_uploads['Alignment Setting file'] = FakeUpload('align.txt', _ALIGNMENT.encode())
        at.run()
        at.run()
        assert not at.exception
        standardized = at.session_state['standardized_df']
        assert [c for c in standardized.columns if c.startswith('intensity[')] == [
            'intensity[s1]', 'intensity[s2]',
        ]
        assert at.session_state['cond_name_0'] == 'Control'
        assert at.session_state['cond_name_1'] == 'Treated'


class TestSwitchingFormatAfterUploading:

    def test_switch_to_workbench_after_uploading_reads_the_file(self, fake_uploads):
        """Regression: with a Workbench file uploaded under Generic, choosing
        Metabolomics Workbench parsed the file, then the format-change reset
        in the same run wiped what it had parsed, and the sidebar crashed
        reading the missing sample list."""
        at = _app(None, fake_uploads, dataset=FakeUpload(
            'mw.csv', (SAMPLES / 'mw_test_dataset.csv').read_bytes(),
        ))
        select = at.sidebar.selectbox[0]
        select.set_value('Metabolomics Workbench').run()
        assert not at.exception
        assert len(at.session_state['workbench_samples']) == 58
