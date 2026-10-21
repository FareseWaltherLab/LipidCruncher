"""
Checks that an AI-produced table still matches the data it came from.

Shared by the Transform to Generic and Data Questions tabs. Pure
functions over DataFrames — no Streamlit imports.
"""
from typing import List

import pandas as pd


def source_name_column(names: pd.Series, source_df: pd.DataFrame):
    """The source's text column that shares the most values with
    ``names`` (a table's lipid names); None if it has no text column."""
    return max(
        (c for c in source_df.columns if source_df[c].dtype == object),
        key=lambda c: names.isin(source_df[c].astype(str)).sum(),
        default=None,
    )


def check_values_stay_with_their_lipid(
    names: pd.Series,
    result_df: pd.DataFrame,
    source_df: pd.DataFrame,
    columns: List,
) -> List[str]:
    """Catch values that moved to another lipid.

    Generated code that filters rows and then builds ``result_df`` by
    index (``result['name'] = filtered['name'].values`` next to
    ``result[col] = filtered[col]``) lines the columns up by row label,
    so lipids get other lipids' values or none. The result looks like a
    valid table, so a schema check passes it.

    Compares each row of ``result_df`` (lipid name in ``names``) with the
    source row of the same lipid, over those of ``columns`` that the
    source has under the same name. Values may have changed (scaling,
    filling gaps), so only an exact copy of a *different* lipid's row is
    treated as moved. Lipids or columns that were renamed, and names that
    are not unique, are not checked.

    Returns:
        One error message when values moved, otherwise an empty list.
    """
    names = names.astype(str)
    name_col = source_name_column(names, source_df)
    columns = [c for c in columns if c in source_df.columns and c != name_col]
    if name_col is None or not columns:
        return []

    source_names = source_df[name_col].astype(str)
    unique = set(source_names[~source_names.duplicated(keep=False)])
    unique &= set(names[~names.duplicated(keep=False)])
    if not unique:
        return []

    def _rows(df, name_series):
        values = df[columns].apply(pd.to_numeric, errors='coerce')
        values.index = name_series.values
        return values

    source = _rows(source_df, source_names)
    output = _rows(result_df, names)

    def _key(row) -> tuple:
        return tuple('nan' if pd.isna(v) else f'{v:.10g}' for v in row)

    owner = {}
    for name, row in source.iterrows():
        if row.notna().sum() >= 2:
            owner.setdefault(_key(row), set()).add(name)

    moved, lost = [], []
    for name, row in output.iterrows():
        if name not in unique:
            continue
        own = source.loc[name]
        if row.isna().all() and own.notna().any():
            lost.append(name)
            continue
        key = _key(row)
        others = owner.get(key, set()) - {name}
        if others and key != _key(own):
            moved.append((name, sorted(others)[0]))

    if not moved:
        return []
    example = moved[0]
    message = (
        f"Values do not stay with their lipid: {len(moved)} row(s) carry "
        f"another lipid's values from the uploaded file (e.g. "
        f"'{example[0]}' has the values of '{example[1]}')"
    )
    if lost:
        message += (
            f", and {len(lost)} row(s) lost all their values "
            f"(e.g. '{lost[0]}')"
        )
    message += (
        ". This happens when columns of a filtered DataFrame are combined "
        "by index with a new one: call .reset_index(drop=True) on the "
        "filtered DataFrame before building result_df, or use .values "
        "for every column."
    )
    return [message]


def check_lipid_table(
    result_df: pd.DataFrame, source_df: pd.DataFrame,
) -> List[str]:
    """Run the identity check on an answer table of any layout.

    The lipid names are whichever text column (or the index) shares the
    most values with the source; the other columns are compared where
    the source has a column of the same name. Summary tables (means per
    condition, counts) share no such columns and pass unchecked.
    """
    table = (
        result_df if isinstance(result_df.index, pd.RangeIndex)
        else result_df.reset_index()
    )
    name_col = _table_name_column(table, source_df)
    if name_col is None:
        return []
    return check_values_stay_with_their_lipid(
        table[name_col], table, source_df,
        [c for c in table.columns if c != name_col],
    )


def _table_name_column(table: pd.DataFrame, source_df: pd.DataFrame):
    """The table's text column with the most values that are names in a
    source text column; None if no value matches."""
    best, best_count = None, 0
    for col in table.columns:
        if table[col].dtype != object:
            continue
        names = table[col].astype(str)
        source_col = source_name_column(names, source_df)
        if source_col is None:
            continue
        count = int(names.isin(source_df[source_col].astype(str)).sum())
        if count > best_count:
            best, best_count = col, count
    return best
