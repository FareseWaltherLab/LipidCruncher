"""
Module 1's "Final Filtered Data" is the data normalization runs on (M13).

The table and its CSV rendered before the internal-standards step, so a
lipid picked there as a standard (and removed from the data) was still in
the "final" download, which then disagreed with every later result.
"""

import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.constants import PAGE_APP

DEFAULT_TIMEOUT = 90
MAIN_APP = str(Path(__file__).parents[2] / 'src' / 'main_app.py')


@pytest.fixture
def module1(captured_downloads):
    at = AppTest.from_file(MAIN_APP, default_timeout=DEFAULT_TIMEOUT)
    at.session_state['page'] = PAGE_APP
    at.run()
    at.sidebar.button(key='load_sample').click().run()
    at.sidebar.checkbox(key='confirm_checkbox').check().run()
    assert not at.exception, at.exception
    return at


def _final_csv(captured):
    return pd.read_csv(io.BytesIO(captured['download_filtered_data']))


def test_standard_picked_from_the_dataset_is_not_in_the_final_data(
    module1, captured_downloads,
):
    at = module1
    at.radio(key='standards_source_radio').set_value('Select from Dataset').run()
    picker = at.multiselect(key='dataset_standards_select')
    pick = picker.options[0]
    picker.set_value([pick]).run()
    assert not at.exception, at.exception

    final = _final_csv(captured_downloads)
    assert pick not in final['LipidMolec'].tolist()
    assert final['LipidMolec'].tolist() == at.session_state['cleaned_df']['LipidMolec'].tolist()
    assert pick in at.session_state['intsta_df']['LipidMolec'].tolist()


def test_final_data_is_sorted_by_class_and_matches_normalization(module1, captured_downloads):
    at = module1
    final = _final_csv(captured_downloads)
    assert final['ClassKey'].tolist() == sorted(final['ClassKey'])
    assert sorted(final['LipidMolec']) == sorted(at.session_state['normalized_df']['LipidMolec'])
