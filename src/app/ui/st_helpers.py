"""
Reusable Streamlit UI helpers.

Eliminates repeated patterns across UI modules:
- Download button pairs (SVG + CSV)
- Section headers with consistent formatting
- Widget state persistence
"""

from typing import Optional

import pandas as pd
import streamlit as st

from app.ui.download_utils import (
    plotly_svg_download_button,
    matplotlib_svg_download_button,
    csv_download_button,
)


def display_export_buttons(
    fig,
    df: pd.DataFrame,
    svg_filename: str,
    csv_filename: str,
    svg_key: str,
    csv_key: str,
    *,
    is_matplotlib: bool = False,
) -> None:
    """Render a two-column row with SVG download + CSV download buttons.

    Args:
        fig: Plotly or Matplotlib figure.
        df: DataFrame for CSV export.
        svg_filename: Filename for the SVG download.
        csv_filename: Filename for the CSV download.
        svg_key: Unique widget key for the SVG button.
        csv_key: Unique widget key for the CSV button.
        is_matplotlib: If True, use matplotlib SVG export instead of Plotly.
    """
    col1, col2 = st.columns(2)
    with col1:
        if is_matplotlib:
            matplotlib_svg_download_button(fig, svg_filename, key=svg_key)
        else:
            plotly_svg_download_button(fig, svg_filename, key=svg_key)
    with col2:
        csv_download_button(df, csv_filename, key=csv_key)


def section_header(title: str) -> None:
    """Render a section header with a horizontal rule above it.

    Args:
        title: Header text including emoji prefix, e.g. "📊 Results".
    """
    st.markdown("---")
    st.markdown(f"#### {title}")


def results_header() -> None:
    """Render the standard '📈 Results' section header."""
    section_header("📈 Results")


def settings_header() -> None:
    """Render the standard '⚙️ Settings' section header."""
    section_header("⚙️ Settings")


def data_selection_header() -> None:
    """Render the standard '🎯 Data Selection' section header."""
    section_header("🎯 Data Selection")


def persist_widget(
    widget_key: str,
    preserved_key: str,
    default,
    options=None,
) -> object:
    """Restore a preserved widget value into session state before rendering.

    Call this BEFORE the widget that uses `widget_key`. After the widget,
    call `save_widget_value` to persist the current selection.

    Args:
        widget_key: The Streamlit widget key.
        preserved_key: The session state key where the value is persisted.
        default: Fallback value if nothing is preserved.
        options: If provided, the restored value must be in this list
            to be accepted (otherwise falls back to default).

    Returns:
        The value to use as the widget's initial value.
    """
    restored = st.session_state.get(preserved_key, default)
    if options is not None and restored not in options:
        restored = default
    return restored
