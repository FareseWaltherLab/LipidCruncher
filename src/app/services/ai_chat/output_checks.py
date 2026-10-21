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


_UP_WORDS = ("up", "increase", "increased", "higher", "enriched", "elevated")
_DOWN_WORDS = ("down", "decrease", "decreased", "lower", "depleted", "reduced")


def _direction(name) -> str:
    """'up' / 'down' if a legend name states a direction, else ''."""
    words = set(re.findall(r"[a-z]+", str(name or "").lower()))
    up, down = bool(words & set(_UP_WORDS)), bool(words & set(_DOWN_WORDS))
    return "up" if up and not down else "down" if down and not up else ""


def check_direction_colors(figure) -> List[str]:
    """Catch a chart whose up/down coloring does not follow its values.

    Generated code that builds a per-bar color list and then re-sorts the
    data colors bars by another lipid's direction (seen 2026-10-21: a
    "colored by direction" top-10 bar chart with a falling lipid in the
    "Up in KO" color). When the legend names two directions, the points of
    each must all lie on one side of zero, on opposite sides; which side is
    "up" is not assumed. Covers one trace per direction and one trace with
    per-point colors plus legend-only traces naming the colors.

    Returns:
        One warning when the coloring contradicts the values, else [].
    """
    groups = {"up": [], "down": []}   # (label, value, legend name)
    legend = {}                       # color -> (direction, legend name)
    colored = []                      # traces with per-point colors
    for trace in figure.data:
        direction = _direction(getattr(trace, "name", None))
        color = getattr(getattr(trace, "marker", None), "color", None)
        labels, values = _labels_and_values(trace)
        if direction and isinstance(color, str):
            legend[color] = (direction, trace.name)
        if direction and values:
            groups[direction].extend(
                (label, v, trace.name) for label, v in zip(labels, values)
            )
        elif values and color is not None and not isinstance(color, str):
            colored.append((labels, values, list(color)))
    for labels, values, colors in colored:
        for label, value, color in zip(labels, values, colors):
            if color in legend:
                direction, name = legend[color]
                groups[direction].append((label, value, name))

    signs = {d: {np.sign(v) for _, v, _ in points if v} for d, points in groups.items()}
    if not signs["up"] or not signs["down"]:
        return []
    if len(signs["up"]) == 1 and len(signs["down"]) == 1 and signs["up"] != signs["down"]:
        return []
    # The side most "up" points are on; points of either direction on the
    # wrong side are miscolored.
    up_side = np.sign(sum(np.sign(v) for _, v, _ in groups["up"]) or 1)
    wrong = [(label, v, name) for d, points in groups.items()
             for label, v, name in points
             if v and np.sign(v) != (up_side if d == "up" else -up_side)]
    label, value, name = wrong[0]
    return [
        f"⚠️ The colors do not follow the direction of the values: "
        f"{len(wrong)} of {sum(len(p) for p in groups.values())} points are "
        f"colored as going the other way (e.g. '{label}' is colored "
        f"'{name}' with value {value:.3g}). Per-point colors were "
        f"probably computed before the data was re-sorted."
    ]


def _labels_and_values(trace):
    """A bar/scatter trace's category labels and numeric values."""
    x = list(trace.x) if getattr(trace, "x", None) is not None else []
    y = list(trace.y) if getattr(trace, "y", None) is not None else []
    numeric = pd.to_numeric(pd.Series(x, dtype=object), errors="coerce")
    if getattr(trace, "orientation", None) == "h" or (x and numeric.notna().all()):
        values, labels = x, y
    else:
        values, labels = y, x
    text = getattr(trace, "text", None)
    if not labels and text is not None and not isinstance(text, str):
        labels = list(text)
    values = pd.to_numeric(pd.Series(values, dtype=object), errors="coerce")
    keep = values.notna().to_numpy()
    labels = [str(lbl) for lbl, k in zip(labels or [""] * len(values), keep) if k]
    return labels, [float(v) for v in values[keep]]
