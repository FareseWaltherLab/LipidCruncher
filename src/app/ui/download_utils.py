"""
Reusable download helpers for Streamlit UI.

Provides SVG and CSV download buttons for Plotly figures,
Matplotlib figures, and DataFrames.
"""
import io
import logging

import streamlit as st
import pandas as pd

from app.ui.sample_names import name_samples_for_csv

logger = logging.getLogger(__name__)


def plotly_svg_download_button(fig, filename, key=None):
    """Create a download button for a Plotly figure as SVG.

    Args:
        fig: Plotly figure object.
        filename: Name for the downloaded file.
        key: Optional unique widget key.
    """
    try:
        svg_bytes = fig.to_image(format="svg")
    except (ValueError, OSError) as e:
        logger.error("SVG export failed (is kaleido installed?): %s", e)
        st.warning("SVG export unavailable. Install kaleido: `pip install kaleido`")
        return
    svg_string = svg_bytes.decode('utf-8')
    if not svg_string.startswith('<?xml'):
        svg_string = '<?xml version="1.0" encoding="utf-8"?>\n' + svg_string
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml",
        key=key,
    )


def matplotlib_svg_download_button(fig, filename, key=None):
    """Create a download button for a Matplotlib figure as SVG.

    Args:
        fig: Matplotlib figure object.
        filename: Name for the downloaded file.
        key: Optional unique widget key.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight')
        buf.seek(0)
        svg_string = buf.getvalue().decode('utf-8')
    except (ValueError, OSError) as e:
        logger.error("Matplotlib SVG export failed: %s", e)
        st.warning("SVG export failed. The figure may be in an invalid state.")
        return
    st.download_button(
        label="Download SVG",
        data=svg_string,
        file_name=filename,
        mime="image/svg+xml",
        key=key,
    )


def convert_df(df, sample_names=None):
    """Convert a DataFrame to CSV bytes for downloading.

    An informative index (a MultiIndex or any named level, such as the
    heatmap's LipidMolec/ClassKey row labels) is written out as leading
    columns; a plain positional index is dropped.

    Args:
        df: DataFrame to convert.
        sample_names: Optional ``{s-label -> name}`` map, keyed in the same
            label space as ``df``, used to name the per-sample columns.

    Returns:
        CSV-encoded bytes.
    """
    if isinstance(df.index, pd.MultiIndex) or any(
        name is not None for name in df.index.names
    ):
        df = df.reset_index()
    df = name_samples_for_csv(df, sample_names)
    return df.to_csv(index=False).encode('utf-8')


def csv_download_button(df, filename, key=None, on_click=None, sample_names=None):
    """Create a download button for a DataFrame as CSV.

    Args:
        df: DataFrame to download.
        filename: Name for the downloaded file.
        key: Optional unique widget key.
        on_click: Optional callback fired on download (clicking reruns the app).
        sample_names: Optional ``{s-label -> name}`` map for the per-sample
            columns; it must be keyed in the label space ``df`` is in.
    """
    st.download_button(
        label="Download CSV",
        data=convert_df(df, sample_names),
        file_name=filename,
        mime="text/csv",
        key=key,
        on_click=on_click,
    )
