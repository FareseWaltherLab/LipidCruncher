"""
Which sample labels the app shows: original names or s1, s2, ...

One sidebar switch decides, app-wide, whether samples appear under their
original names (from the uploaded headers or the LipidSearch Alignment Setting
file, editable in the sidebar) or under LipidCruncher's standardized labels.
Every table, plot, picker and download asks here for the names of its label
space, and gets none when standardized labels are chosen. The data itself is
always keyed by s-label; only what is shown changes.

Label spaces (each keyed by the s-labels its data uses):
- ``'upload'``: the sidebar — ``sample_names``, keyed by the full dataset's
  labels.
- ``'processing'``: Data Processing, Internal Standards and Normalization —
  the names of ``processing_frame``, re-keyed for the renumbering that the
  sidebar's sample exclusion causes (equal to ``sample_names`` when nothing
  is excluded).
- ``'qc_input'``: Quality Check before PCA sample exclusion — the names of
  ``qc_input_frame``, frozen when Module 1 handed its data over.
- ``'qc'``: PCA and everything after it — the names of ``qc_frame``,
  re-keyed for the renumbering that exclusion causes (the ``'qc_input'``
  names when nothing is excluded).
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
    names = {
        'upload': lambda: st.session_state.get('sample_names'),
        'processing': processing_names,
        'qc_input': qc_input_names,
        'qc': _qc_names,
    }[space]()
    return names or None


def processing_names() -> Optional[Dict[str, str]]:
    """Names keyed by Module 1's labels, whichever labels are shown.

    The sidebar's names when no processing frame exists (a page rendered
    without the sidebar).
    """
    frame = st.session_state.get('processing_frame')
    return frame.names if frame is not None else st.session_state.get('sample_names')


def qc_input_names() -> Optional[Dict[str, str]]:
    """Names keyed by the labels Quality Check received.

    Module 1's names when nothing was handed over (a page rendered without
    main_app's handoff).
    """
    frame = st.session_state.get('qc_input_frame')
    return frame.names if frame is not None else processing_names()


def _qc_names() -> Optional[Dict[str, str]]:
    frame = st.session_state.get('qc_frame')
    return frame.names if frame is not None else qc_input_names()


def display_names_for(space: str) -> Dict[str, str]:
    """``{s-label -> shown text}`` for the named samples in ``space``.

    The text tables and CSVs use, unique per sample; empty when standardized
    labels are chosen. Pass it to plots so they agree with the tables.
    """
    return sample_display_map(names_for(space))


def labeler(space: str) -> Callable[[str], str]:
    """Return a function giving the shown text for an s-label in ``space``."""
    display = display_names_for(space)
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
