"""
Internal Standards Management UI Components for LipidCruncher main content area.

This module contains:
- display_manage_internal_standards: Manage Internal Standards expander
  (auto-detect, upload custom, view consistency plots)
"""

import streamlit as st
import pandas as pd

from app.services.standards import StandardsService
from app.ui.standards_plots import display_standards_consistency_plots
from app.ui.content import STANDARDS_EXTRACT_HELP, STANDARDS_COMPLETE_HELP


# =============================================================================
# Helper Functions
# =============================================================================

def _fallback_standards(auto_detected_df: pd.DataFrame) -> pd.DataFrame:
    """Return auto-detected standards or empty DataFrame as fallback."""
    return auto_detected_df if auto_detected_df is not None else pd.DataFrame()


# =============================================================================
# Sub-Components
# =============================================================================

def _display_auto_detected_standards(auto_detected_df: pd.DataFrame) -> pd.DataFrame:
    """Display auto-detected standards with download button.

    Returns the active standards DataFrame.
    """
    # Clear custom standards when switching to automatic
    if st.session_state.custom_standards_df is not None:
        st.session_state.custom_standards_df = None
        st.session_state.custom_standards_mode = None

    if auto_detected_df is not None and not auto_detected_df.empty:
        st.success(f"✓ Found {len(auto_detected_df)} standards")
        st.dataframe(auto_detected_df, use_container_width=True)

        csv = auto_detected_df.to_csv(index=False)
        st.download_button(
            label="Download Detected Standards",
            data=csv,
            file_name="detected_standards.csv",
            mime="text/csv",
            key="download_auto_standards"
        )
        return auto_detected_df
    else:
        st.warning("No internal standards automatically detected in dataset.")
        return pd.DataFrame()


def _display_custom_upload(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """Display custom standards upload UI (extract from dataset or external).

    Returns the active standards DataFrame.
    """
    st.markdown("---")

    # Mode selection: standards in dataset or external
    st.markdown("**Are standards present in your main dataset?**")

    standards_location = st.radio(
        "Standards location:",
        options=[
            "Yes — Extract from dataset",
            "No — Uploading complete standards data"
        ],
        key="standards_location_radio",
        horizontal=True,
        label_visibility="collapsed"
    )

    use_extract_mode = "Yes" in standards_location
    current_mode = "extract" if use_extract_mode else "complete"

    # Clear custom standards if user switched between upload modes
    if (st.session_state.custom_standards_mode is not None and
        st.session_state.custom_standards_mode != current_mode and
        st.session_state.custom_standards_df is not None):
        st.session_state.custom_standards_df = None
        st.session_state.custom_standards_mode = None

    # Format guidance
    st.markdown("---")
    if use_extract_mode:
        st.markdown(STANDARDS_EXTRACT_HELP)
    else:
        st.markdown(STANDARDS_COMPLETE_HELP)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload standards CSV",
        type=['csv'],
        key="standards_file_uploader"
    )

    # Show preserved custom standards if file uploader is empty
    if uploaded_file is None and st.session_state.custom_standards_df is not None:
        return _display_preserved_custom_standards()

    # Process uploaded file
    if uploaded_file is not None:
        return _process_uploaded_standards(
            uploaded_file, cleaned_df, auto_detected_df,
            use_extract_mode, current_mode
        )

    # No file uploaded yet, use auto-detected as fallback
    if auto_detected_df is not None and not auto_detected_df.empty:
        st.info("Upload a CSV file or switch to Automatic Detection.")
        return auto_detected_df
    else:
        st.warning("No standards available. Please upload a standards file.")
        return pd.DataFrame()


def _display_preserved_custom_standards() -> pd.DataFrame:
    """Display previously uploaded custom standards with clear button."""
    st.success(f"✓ Using {len(st.session_state.custom_standards_df)} custom standards")
    st.dataframe(st.session_state.custom_standards_df, use_container_width=True)

    if st.button("Clear Custom Standards", key="clear_custom_standards"):
        st.session_state.custom_standards_df = None
        st.session_state.custom_standards_mode = None
        st.session_state.standards_source = "Automatic Detection"
        st.rerun()

    return st.session_state.custom_standards_df


def _process_uploaded_standards(
    uploaded_file,
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame,
    use_extract_mode: bool,
    current_mode: str
) -> pd.DataFrame:
    """Process an uploaded standards CSV file.

    Returns the active standards DataFrame.
    """
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        result = StandardsService.process_standards_file(
            uploaded_df=uploaded_df,
            cleaned_df=cleaned_df,
            standards_in_dataset=use_extract_mode
        )

        if result.standards_df is None or result.standards_df.empty:
            st.error("No valid standards found in uploaded file.")
            return _fallback_standards(auto_detected_df)

        st.session_state.custom_standards_df = result.standards_df
        st.session_state.custom_standards_mode = current_mode

        # For external standards, remove them from main dataset if present
        if not use_extract_mode:
            filtered_df, removed_lipids = StandardsService.remove_standards_from_dataset(
                cleaned_df, result.standards_df
            )
            if removed_lipids:
                st.session_state.cleaned_df = filtered_df
                preview = removed_lipids[:5]
                more = f"... and {len(removed_lipids) - 5} more" if len(removed_lipids) > 5 else ""
                st.warning(
                    f"⚠️ Removed {len(removed_lipids)} standard(s) from main dataset: "
                    f"{', '.join(preview)}{more}"
                )

        if result.duplicates_removed > 0:
            st.info(f"Removed {result.duplicates_removed} duplicate standard(s).")

        st.success(f"✓ Loaded {result.standards_count} custom standards (mode: {result.source_mode})")
        st.dataframe(result.standards_df, use_container_width=True)

        csv = result.standards_df.to_csv(index=False)
        st.download_button(
            label="Download Custom Standards",
            data=csv,
            file_name="custom_standards.csv",
            mime="text/csv",
            key="download_custom_standards"
        )

        return result.standards_df

    except ValueError as ve:
        st.error(str(ve))
        return _fallback_standards(auto_detected_df)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return _fallback_standards(auto_detected_df)


# =============================================================================
# UI Components
# =============================================================================

def display_manage_internal_standards(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Display the Manage Internal Standards expander.

    Allows users to:
    - Use automatically detected standards
    - Upload custom standards (from dataset or external)
    - Clear custom standards
    - View consistency plots for internal standards

    Args:
        cleaned_df: The cleaned data DataFrame (for extracting standards)
        auto_detected_df: Auto-detected internal standards DataFrame

    Returns:
        The active internal standards DataFrame to use for normalization
    """
    active_standards_df = pd.DataFrame()

    with st.expander("Manage Internal Standards", expanded=False):
        st.markdown("""
Auto-detection identifies deuterated standards (`(d5)`, `(d7)`, `(d9)`),
`ISTD`/`IS` markers in class names, and SPLASH LIPIDOMIX® patterns.
        """)

        # Initialize session state for standards management
        if 'standards_source' not in st.session_state:
            st.session_state.standards_source = "Automatic Detection"
        if 'custom_standards_df' not in st.session_state:
            st.session_state.custom_standards_df = None
        if 'custom_standards_mode' not in st.session_state:
            st.session_state.custom_standards_mode = None
        if 'original_auto_intsta_df' not in st.session_state:
            st.session_state.original_auto_intsta_df = auto_detected_df

        # Standards source selection
        st.markdown("##### 🏷️ Standards Source")

        standards_source = st.radio(
            "Standards source:",
            ["Automatic Detection", "Upload Custom Standards"],
            horizontal=True,
            key="standards_source_radio",
            label_visibility="collapsed"
        )
        st.session_state.standards_source = standards_source

        if standards_source == "Automatic Detection":
            active_standards_df = _display_auto_detected_standards(auto_detected_df)
        else:
            active_standards_df = _display_custom_upload(cleaned_df, auto_detected_df)

        # Show consistency plots if we have standards and experiment config
        if (active_standards_df is not None and
            not active_standards_df.empty and
            'experiment' in st.session_state and
            st.session_state.experiment is not None):
            display_standards_consistency_plots(
                intsta_df=active_standards_df,
                experiment=st.session_state.experiment
            )

    return active_standards_df
