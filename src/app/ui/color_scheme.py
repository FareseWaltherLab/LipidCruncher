"""
The condition color scheme every figure uses.

One sidebar picker chooses, app-wide, the palette that figures colored by
condition draw from: the Quality Check box plots and PCA, the internal
standards bars, the class concentration bar chart, the heatmap's condition
strip and the volcano page's individual-lipid plot. Each page passes
``current_palette()`` to the adapter, so the choice is part of every cache key.
"""

import streamlit as st

from app.constants import CONDITION_PALETTES, DEFAULT_CONDITION_PALETTE

_SELECT_KEY = '_condition_palette_select'


def current_palette() -> str:
    """The chosen condition color scheme (a CONDITION_PALETTES name)."""
    palette = st.session_state.get('condition_palette', DEFAULT_CONDITION_PALETTE)
    return palette if palette in CONDITION_PALETTES else DEFAULT_CONDITION_PALETTE


def render_color_scheme_picker() -> None:
    """Sidebar section, below Confirm Inputs, choosing the condition color
    scheme.

    The choice lives in ``condition_palette`` rather than the widget key, so it
    survives pages where the sidebar is not drawn.
    """
    def _sync() -> None:
        st.session_state.condition_palette = st.session_state[_SELECT_KEY]

    options = list(CONDITION_PALETTES)
    st.sidebar.subheader("Plot Colors")
    st.sidebar.selectbox(
        'Color scheme',
        options,
        index=options.index(current_palette()),
        key=_SELECT_KEY,
        on_change=_sync,
        help=(
            "Colors the conditions in every plot that is colored by condition, "
            "and in the PDF report. Colorblind-safe uses the Okabe-Ito palette."
        ),
    )
