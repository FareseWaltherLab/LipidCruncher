"""
LipidSearch lipids identified in both ionisation modes (M10).

The app keeps one row per lipid (better grade, then identification rate)
and never adds the two modes. It now says so on the page, with this file's
counts, and the processing help explains what it means for comparisons.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.constants import FORMAT_LIPIDSEARCH, PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')
SAMPLE = Path(__file__).parents[2] / 'sample_datasets' / 'lipidsearch5_test_dataset.csv'


def _module1():
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.selectbox[0].set_value(FORMAT_LIPIDSEARCH).run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception, at.exception
    return at


def _notes(at):
    return [i.value for i in at.main.info if i.value.startswith('Polarity:')]


@pytest.fixture
def two_mode_sample(tmp_path, monkeypatch):
    """The LipidSearch sample with ions: every row positive, three A-grade
    PC species also seen (B grade) in negative mode, two PC species only in
    negative mode."""
    df = pd.read_csv(SAMPLE)
    df['MainIon'] = 'M+H'
    pc = df.index[(df['ClassKey'] == 'PC') & (df['TotalGrade'] == 'A')]
    negative_twins = df.loc[pc[:3]].assign(MainIon='M+HCOO', TotalGrade='B')
    # Species of their own (the sample has several rows for some names).
    negative_only = df.loc[pc[3:5]].assign(
        MainIon='M+HCOO',
        LipidMolec=['PC(91:0_91:1)', 'PC(92:0_92:1)'],
        FAKey=['91:0_91:1', '92:0_92:1'],
    )
    pd.concat([df, negative_twins, negative_only]).to_csv(
        tmp_path / SAMPLE.name, index=False,
    )
    import app.ui.sidebar.file_upload as file_upload
    monkeypatch.setattr(file_upload, 'SAMPLE_DATA_DIR', tmp_path)


def test_note_names_the_counts(two_mode_sample):
    at = _module1()
    [note] = _notes(at)
    assert '3 lipid(s) were identified in both positive and negative mode' in note
    assert 'never added' in note
    assert ', 2 negative)' in note and 'PC (' in note


def test_no_note_without_ions():
    assert _notes(_module1()) == []


def test_help_explains_the_rule():
    from app.ui.content import get_processing_docs
    docs = get_processing_docs(FORMAT_LIPIDSEARCH)
    assert 'Lipids identified in both positive and negative mode' in docs
    assert 'never added' in docs
    assert 'detected in only one polarity' not in docs
