"""
Reusable download helpers for Streamlit UI.

Provides SVG and CSV download buttons for Plotly figures,
Matplotlib figures, and DataFrames.
"""
import io
import logging

import streamlit as st
import pandas as pd

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


def convert_df(df):
    """Convert a DataFrame to CSV bytes for downloading.

    Args:
        df: DataFrame to convert.

    Returns:
        CSV-encoded bytes.
    """
    return df.to_csv(index=False).encode('utf-8')


def csv_download_button(df, filename, key=None):
    """Create a download button for a DataFrame as CSV.

    Args:
        df: DataFrame to download.
        filename: Name for the downloaded file.
        key: Optional unique widget key.
    """
    st.download_button(
        label="Download CSV",
        data=convert_df(df),
        file_name=filename,
        mime="text/csv",
        key=key,
    )
