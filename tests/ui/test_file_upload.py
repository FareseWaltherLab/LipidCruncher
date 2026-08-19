"""Tests for delimiter-aware dataset reading in the sidebar file upload.

Regression coverage for LipidSearch 5.2, which exports tab-delimited data with a
.csv extension. Reading it as comma-separated collapsed the header into a single
column, producing "Missing required columns" errors.
"""
import io

import pandas as pd

from app.services.format_detection import FormatDetectionService, DataFormat
from app.ui.sidebar.file_upload import _read_tabular


# Minimal LipidSearch-shaped header + one data row.
_COLUMNS = [
    'LipidMolec', 'ClassKey', 'CalcMass', 'BaseRt',
    'TotalGrade', 'TotalSmpIDRate(%)', 'FAKey', 'MeanArea[s1]',
]
_ROW = ['PC(16:0)', 'PC', '700.5', '5.0', 'A', '100.00', '(16:0)', '12345.0']


def _content(sep: str) -> str:
    return sep.join(_COLUMNS) + '\n' + sep.join(_ROW) + '\n'


class _FakeUpload:
    """Mimics a Streamlit UploadedFile (BytesIO-like with getvalue)."""

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_read_tabular_parses_tab_delimited_path(tmp_path):
    """A tab-delimited .csv (LipidSearch 5.2) is parsed into separate columns."""
    f = tmp_path / "ls52.csv"
    f.write_text(_content('\t'))

    df = _read_tabular(f)

    assert list(df.columns) == _COLUMNS
    assert FormatDetectionService.detect_format(df) == DataFormat.LIPIDSEARCH


def test_read_tabular_parses_comma_delimited_path(tmp_path):
    """Legacy comma-delimited LipidSearch 5.0 files still parse unchanged."""
    f = tmp_path / "ls50.csv"
    f.write_text(_content(','))

    df = _read_tabular(f)

    assert list(df.columns) == _COLUMNS
    assert FormatDetectionService.detect_format(df) == DataFormat.LIPIDSEARCH


def test_read_tabular_parses_tab_delimited_upload():
    """Tab-delimited uploaded file (BytesIO-like) is parsed correctly."""
    upload = _FakeUpload(_content('\t').encode('utf-8'))

    df = _read_tabular(upload)

    assert list(df.columns) == _COLUMNS
    assert FormatDetectionService.detect_format(df) == DataFormat.LIPIDSEARCH


# ── Upload limit label ─────────────────────────────────────────────────
#
# The tooltip used to hardcode "Limit 800MB per file" while the real cap came
# from .streamlit/config.toml (or an env var in the container), so the two
# drifted apart: the tooltip claimed 800MB while the app served 200MB locally
# and 1GB in production.

def test_upload_limit_label_tracks_the_configured_cap(monkeypatch):
    import streamlit as st

    from app.ui.sidebar import file_upload

    for megabytes, expected in (
        (1024, '1GB'), (2048, '2GB'), (800, '800MB'), (200, '200MB'),
    ):
        monkeypatch.setattr(
            st, 'get_option', lambda name, mb=megabytes: (
                mb if name == 'server.maxUploadSize' else None
            ),
        )
        assert file_upload._upload_limit_text() == expected


def test_upload_limit_label_matches_the_real_option():
    """No monkeypatching: what the tooltip says is what the app enforces."""
    import streamlit as st

    from app.ui.sidebar import file_upload

    megabytes = st.get_option('server.maxUploadSize')
    label = file_upload._upload_limit_text()
    assert str(megabytes) in label or str(megabytes // 1024) in label
