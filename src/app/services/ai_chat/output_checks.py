"""
Checks that an AI-produced answer still matches the data it came from.

Shared by the Transform to Generic, Data Questions and Custom
Visualization tabs. Pure functions over DataFrames — no Streamlit imports.
"""
import re
from typing import List, Tuple

import numpy as np
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


# A number standing on its own in prose: not part of a lipid name
# ('PC 16:0_18:1'), a sample name ('WT-1', 's3') or a word ('C18').
_PROSE_NUMBER = re.compile(
    r"(?<![\w.:/_\-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?([eE][-+]?\d+)?(%?)(?![\w:/_])"
)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def computed_values(stdout: str = "", result_df=None, figure=None) -> np.ndarray:
    """Every number a turn's code produced: printed, in its table, or
    plotted."""
    values = [float(v) for v in _ANY_NUMBER.findall(stdout or "")]
    if result_df is not None:
        table = result_df.reset_index()
        for col in table.columns:
            numbers = pd.to_numeric(table[col], errors="coerce")
            values.extend(numbers.dropna().astype(float))
    if figure is not None:
        for trace in figure.data:
            for axis in ("x", "y", "z", "values"):
                data = getattr(trace, axis, None)
                if data is not None:
                    numbers = pd.to_numeric(pd.Series(np.ravel(data)), errors="coerce")
                    values.extend(numbers.dropna().astype(float))
    return np.abs(np.asarray(values, dtype=float))


def numbers_not_computed(
    prose: str, computed: np.ndarray, context: str = "",
) -> Tuple[str, ...]:
    """Numbers the model stated that its code never produced.

    The explanation streams before the code runs, so a number in it is the
    model's claim, not a result. A stated number counts as computed when a
    computed value rounds to it (to the decimals it was written with; a
    percentage may also be a fraction). Numbers below 10 (list markers,
    "2 conditions") and numbers that appear in ``context`` (the question,
    the code) are not checked. For logging and measurement, not refusal.
    """
    context_numbers = {m.group(0) for m in _ANY_NUMBER.finditer(context or "")}
    unverified = []
    for match in _PROSE_NUMBER.finditer(prose or ""):
        whole, fraction, exponent, percent = match.groups()
        text = match.group(0)
        number = float(whole.replace(",", "") + (fraction or "") + (exponent or ""))
        if number < 10 and not fraction and not exponent:
            continue
        if text.rstrip("%") in context_numbers or whole.replace(",", "") in context_numbers:
            continue
        if exponent:
            tolerance = abs(number) * 5e-3
        else:
            tolerance = 0.5 * 10 ** -len((fraction or ".")[1:]) + 1e-12
        targets = [(number, tolerance)]
        if percent:
            targets.append((number / 100, tolerance / 100))
        if not any(np.any(np.abs(computed - t) <= tol) for t, tol in targets):
            unverified.append(text)
    return tuple(dict.fromkeys(unverified))
