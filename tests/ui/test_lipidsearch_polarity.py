"""
LipidSearch lipids identified as several adducts or in both modes (M10).

The app keeps one row per lipid and never adds them. A class with its own
internal standard is quantified in the standard's adduct, and its species
seen only in other adducts are removed unless the user keeps them (power
user, 2026-10-22). The page says what happened, with this file's counts,
and the processing help explains the rule.
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
    return [i.value for i in at.main.info if i.value.startswith('Adducts:')]


@pytest.fixture
def two_mode_sample(tmp_path, monkeypatch):
    """The LipidSearch sample with ions: every row [M+H]+ (so is the PC
    standard), three A-grade PC species also seen (B grade) as [M+HCOO]-,
    two PC species only as [M+HCOO]-."""
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


def test_species_not_in_the_standards_adduct_are_removed(two_mode_sample):
    at = _module1()
    [note] = _notes(at)
    assert '2 seen only as other adducts removed (M+HCOO 2)' in note
    assert 'PC (M+H):' in note
    assert 'Keep species measured in other adducts' in note
    final = at.session_state.ingestion_result.cleaned_df
    assert not final['LipidMolec'].str.startswith('PC 91:0').any()
    assert set(final.loc[final['ClassKey'] == 'PC', 'MainIon']) == {'M+H'}


def test_the_user_can_keep_them(two_mode_sample):
    at = _module1()
    at.checkbox(key='keep_other_adducts_checkbox').check().run()
    assert not at.exception, at.exception
    [note] = _notes(at)
    assert '2 seen only as other adducts kept (M+HCOO 2)' in note
    final = at.session_state.ingestion_result.cleaned_df
    assert final['LipidMolec'].str.startswith('PC 91:0').any()


def test_power_users_choose_the_adduct_per_class(two_mode_sample):
    """Step 4: an adduct per class, for classes measured in more than one."""
    at = _module1()
    assert not [s for s in at.selectbox if s.key == 'adduct_select_PC']
    at.checkbox(key='customize_adducts_checkbox').check().run()
    select = at.selectbox(key='adduct_select_PC')
    assert select.options == ['Default', 'M+H', 'M+HCOO', 'Any adduct']
    select.set_value('M+HCOO').run()
    assert not at.exception, at.exception
    [note] = _notes(at)
    assert 'PC (M+HCOO, your choice):' in note
    final = at.session_state.ingestion_result.cleaned_df
    assert set(final.loc[final['ClassKey'] == 'PC', 'MainIon']) == {'M+HCOO'}


def test_no_note_without_ions():
    assert _notes(_module1()) == []


def test_help_explains_the_rule():
    from app.ui.content import get_processing_docs
    docs = get_processing_docs(FORMAT_LIPIDSEARCH)
    assert "quantified in the\nstandard's adduct" in docs
    assert 'never added' in docs
    assert 'detected in only one polarity' not in docs
