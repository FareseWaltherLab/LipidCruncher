"""Checks that an AI answer table still matches the data it came from
(services/ai_chat/output_checks.py), for tables of any layout."""
from pathlib import Path

import pandas as pd
import pytest

from app.services.ai_chat.output_checks import check_lipid_table

_KO = ['KO-1', 'KO-2', 'KO-3']


@pytest.fixture(scope='module')
def studio_df():
    path = Path(__file__).parents[1] / 'fixtures' / 'ai_studio_test_dataset.csv'
    return pd.read_csv(path)


def _misaligned(df):
    """Names with a fresh index next to columns with the filtered one."""
    kept = df[df['QualityScore'] > 70]
    table = pd.DataFrame({'feature': kept['feature'].values})
    for col in _KO:
        table[col] = kept[col]
    return table


def test_misaligned_per_lipid_table_is_caught(studio_df):
    (message,) = check_lipid_table(_misaligned(studio_df), studio_df)
    assert "Values do not stay with their lipid" in message


def test_names_need_not_be_the_first_column_or_a_column(studio_df):
    kept = studio_df[studio_df['QualityScore'] > 70].reset_index(drop=True)
    good = kept[_KO + ['feature']]
    assert check_lipid_table(good, studio_df) == []
    assert check_lipid_table(good.set_index('feature'), studio_df) == []
    bad = _misaligned(studio_df)[_KO + ['feature']]
    assert check_lipid_table(bad.set_index('feature'), studio_df) != []


def test_summary_tables_are_not_checked(studio_df):
    means = studio_df.groupby('LipidClass')[_KO].mean().reset_index()
    assert check_lipid_table(means, studio_df) == []
    counts = pd.DataFrame({'n': [len(studio_df)]})
    assert check_lipid_table(counts, studio_df) == []
