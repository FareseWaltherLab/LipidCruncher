"""Reset matrix: what every navigation / reset trigger does to session state.

Pins the CURRENT behaviour of the whole app (``main_app.py``) so the
session-state redesign (R1) can be checked against it: for each starting
stage x trigger, every session key is classified as

- ``cleared``  — set before, back to its default (or gone) after,
- ``kept``     — set before and after, same value,
- ``changed``  — set before and after, different value,
- ``added``    — default/absent before, set after.

The expectations live in ``reset_matrix.json`` next to this file. After an
intentional behaviour change, regenerate it with

    UPDATE_RESET_MATRIX=1 pytest tests/ui/test_reset_matrix.py

and review the diff: every moved key should be explainable.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.adapters.streamlit_adapter import SessionState
from app.constants import PAGE_APP

MAIN_APP = Path(__file__).parents[2] / "src" / "main_app.py"
GOLDEN = Path(__file__).parent / "reset_matrix.json"
UPDATE = os.environ.get("UPDATE_RESET_MATRIX") == "1"
TIMEOUT = 120

_DEFAULTS = {
    f.name: (f.default_factory() if f.default_factory is not dataclasses.MISSING
             else f.default)
    for f in dataclasses.fields(SessionState)
}


# ── value fingerprints ─────────────────────────────────────────────────────

def _has_figure(value) -> bool:
    if hasattr(value, "to_plotly_json") or hasattr(value, "savefig"):
        return True
    if isinstance(value, dict):
        return any(_has_figure(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_figure(v) for v in value)
    return False


def _fingerprint(value) -> str:
    """A stable digest of a session value.

    Figures count by presence only: they are rebuilt on reruns or served from
    st.cache_data depending on what ran before, so their content is not a
    stable signal of what a trigger did.
    """
    if _has_figure(value):
        payload = "figure"
    elif isinstance(value, pd.DataFrame):
        payload = (repr(list(value.columns))
                   + pd.util.hash_pandas_object(value, index=True)
                   .to_numpy().tobytes().hex())
    else:
        payload = repr(value)
        if " at 0x" in payload:  # object identity, not content
            payload = type(value).__name__
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _is_default(key, value) -> bool:
    if key not in _DEFAULTS:
        return False
    default = _DEFAULTS[key]
    if value is None or default is None:
        return value is default
    try:
        return bool(value == default)
    except (ValueError, TypeError):
        return False


def _snapshot(at) -> dict:
    """Every session key that holds a non-default value -> fingerprint."""
    state = at.session_state
    out = {}
    for key in state.filtered_state:
        if key.startswith("$$") or key.startswith("FormSubmitter"):
            continue
        value = state[key]
        if not _is_default(key, value):
            out[key] = _fingerprint(value)
    return out


def _classify(before: dict, after: dict) -> dict:
    return {
        "cleared": sorted(k for k in before if k not in after),
        "kept": sorted(k for k in before if k in after and before[k] == after[k]),
        "changed": sorted(k for k in before if k in after and before[k] != after[k]),
        "added": sorted(k for k in after if k not in before),
    }


# ── stages ─────────────────────────────────────────────────────────────────

def _module1() -> AppTest:
    """Generic sample data loaded and the experiment confirmed."""
    at = AppTest.from_file(str(MAIN_APP), default_timeout=TIMEOUT)
    at.session_state["page"] = PAGE_APP
    at.run()
    at.sidebar.button(key="load_sample").click().run()
    at.sidebar.checkbox(key="confirm_checkbox").check().run()
    assert not at.exception
    return at


def _module2() -> AppTest:
    """Module 1 done, Quality Check and analysis page rendered."""
    at = _module1()
    _button(at, "Next: Quality Check").click().run()
    assert not at.exception
    return at


STAGES = {"module1": _module1, "module2": _module2}


# ── triggers ───────────────────────────────────────────────────────────────

def _button(at, label_start):
    matches = [b for b in at.button if b.label.startswith(label_start)]
    assert matches, f"no button starting with {label_start!r}"
    return matches[0]


def _click_key(key):
    def _do(at):
        [b for b in at.button if b.key == key][0].click().run()
    return _do


def _rerun(at):
    """Control: a rerun with no interaction."""
    at.run()


def _next(at):
    _button(at, "Next: Quality Check").click().run()


def _reset_session(at):
    at.sidebar.button(key="reset_session").click().run()


def _switch_format(at):
    box = at.sidebar.selectbox[0]
    other = [o for o in box.options if o != box.value][0]
    box.set_value(other).run()


def _reload_sample(at):
    at.sidebar.button(key="load_sample").click().run()


def _uncheck_confirm(at):
    at.sidebar.checkbox(key="confirm_checkbox").uncheck().run()


def _exclude_first_sample(at):
    picker = at.sidebar.multiselect(key="exclude_samples")
    picker.set_value([picker.options[0]]).run()


TRIGGERS = {
    "module1": {
        "rerun": _rerun,
        "next": _next,
        "back_home": _click_key("back_home_module1"),
        "reset_session": _reset_session,
        "switch_format": _switch_format,
        "reload_sample": _reload_sample,
        "uncheck_confirm": _uncheck_confirm,
        "exclude_sample": _exclude_first_sample,
    },
    "module2": {
        "rerun": _rerun,
        "back_to_processing": _click_key("back_processing_module2"),
        "back_home": _click_key("back_home_module2"),
        "reset_session": _reset_session,
        "switch_format": _switch_format,
        "reload_sample": _reload_sample,
        "uncheck_confirm": _uncheck_confirm,
        "exclude_sample": _exclude_first_sample,
    },
}

CASES = [(stage, trigger) for stage in TRIGGERS for trigger in TRIGGERS[stage]]


def _run_case(stage, trigger) -> dict:
    at = STAGES[stage]()
    before = _snapshot(at)
    TRIGGERS[stage][trigger](at)
    assert not at.exception, at.exception
    return _classify(before, _snapshot(at))


@pytest.fixture(scope="module")
def golden():
    if UPDATE or not GOLDEN.exists():
        return {}
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("stage,trigger", CASES)
def test_reset_matrix(stage, trigger, golden):
    observed = _run_case(stage, trigger)
    name = f"{stage}/{trigger}"
    if UPDATE:
        _record(name, observed)
        return
    assert name in golden, f"{name} missing from {GOLDEN.name}; regenerate"
    assert observed == golden[name]


def _record(name, observed):
    """Merge one case into the golden file (safe under xdist via a lock)."""
    import fcntl
    import tempfile

    with open(Path(tempfile.gettempdir()) / "reset_matrix.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[name] = observed
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
