"""
Which sample labels the app shows: original names or s1, s2, ...

One sidebar switch decides, app-wide, whether samples appear under their
original names (from the uploaded headers or the LipidSearch Alignment Setting
file, editable in the sidebar) or under LipidCruncher's standardized labels.
Every table, plot, picker and download asks here for the names of its label
space, and gets none when standardized labels are chosen. The data itself is
always keyed by s-label; only what is shown changes.

Label spaces (each keyed by the s-labels its data uses):
- ``'upload'``: Data Processing, Internal Standards, Normalization and the
  sidebar — ``sample_names``.
- ``'qc_input'``: Quality Check before PCA sample exclusion — the names
  frozen when Module 1 handed its data over (``qc_input_sample_names``).
- ``'qc'``: PCA and everything after it — the names re-keyed for the
  renumbering that exclusion causes (``qc_sample_names``).
"""

from typing import Callable, Dict, Optional

import streamlit as st

from app.ui.sample_names import sample_display_map

ORIGINAL = 'original'
STANDARDIZED = 'standardized'

_MODE_LABELS = {
    ORIGINAL: 'Original names',
    STANDARDIZED: 'Standardized labels (s1, s2, …)',
}
_MAP_KEYS = {
    'upload': 'sample_names',
    'qc_input': 'qc_input_sample_names',
    'qc': 'qc_sample_names',
}
_RADIO_KEY = '_sample_label_mode_radio'


def showing_original_names() -> bool:
    """True when samples are shown under their original names."""
    return st.session_state.get('sample_label_mode', ORIGINAL) == ORIGINAL


def names_for(space: str) -> Optional[Dict[str, str]]:
    """The ``{s-label -> name}`` map to display in a label space, or None.

    None when standardized labels are chosen or no names exist, so callers
    show the s-labels.
    """
    if not showing_original_names():
        return None
    return st.session_state.get(_MAP_KEYS[space]) or None


def labeler(space: str) -> Callable[[str], str]:
    """Return a function giving the shown text for an s-label in ``space``."""
    display = sample_display_map(names_for(space))
    return lambda label: display.get(label, label)


def render_sample_label_switch() -> None:
    """Sidebar radio choosing original names vs standardized labels.

    The choice lives in ``sample_label_mode`` rather than the widget key, so it
    survives pages where the sidebar is not drawn.
    """
    def _sync() -> None:
        st.session_state.sample_label_mode = st.session_state[_RADIO_KEY]

    options = [ORIGINAL, STANDARDIZED]
    current = st.session_state.get('sample_label_mode', ORIGINAL)
    st.sidebar.radio(
        'Show samples as',
        options,
        index=options.index(current) if current in options else 0,
        format_func=_MODE_LABELS.get,
        key=_RADIO_KEY,
        on_change=_sync,
        help=(
            "Applies everywhere: tables, plots, sample pickers, CSV downloads "
            "and the PDF report. Samples without an original name always show "
            "their standardized label."
        ),
    )
