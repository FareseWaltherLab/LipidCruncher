"""
Internal Standards Management UI Components for LipidCruncher main content area.

This module contains:
- display_manage_internal_standards: Manage Internal Standards expander
  (auto-detect, upload custom, view consistency plots)
"""

import logging

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

from app.services.standards import StandardsService
from app.ui.standards_plots import display_standards_consistency_plots
from app.ui.content import STANDARDS_COMPLETE_HELP
from app.ui.download_utils import csv_download_button
from app.ui.st_helpers import keep_intsta_expander_open, INTSTA_EXPANDER_KEY


# =============================================================================
# Helper Functions
# =============================================================================

def _fallback_standards(auto_detected_df: pd.DataFrame) -> pd.DataFrame:
    """Return auto-detected standards or empty DataFrame as fallback."""
    return auto_detected_df if auto_detected_df is not None else pd.DataFrame()


# Every interactive widget inside the Manage Internal Standards expander uses
# this as its on_change/on_click, so working in the section never collapses it.
_latch_expander_open = keep_intsta_expander_open


# =============================================================================
# Dataset-sourced standards helpers
# =============================================================================

def _extract_dataset_standards(cleaned_df: pd.DataFrame, selected_lipids: list):
    """Build a standards DataFrame from lipids already in the dataset.

    Returns (standards_df, error_message). Exactly one is non-None.
    """
    try:
        result = StandardsService.process_standards_file(
            uploaded_df=pd.DataFrame({'LipidMolec': selected_lipids}),
            cleaned_df=cleaned_df,
            standards_in_dataset=True,
        )
        return result.standards_df, None
    except ValueError as ve:
        logger.error("Dataset standards extraction error: %s", ve)
        return None, str(ve)


def _remove_dataset_standards_from_main(
    cleaned_df: pd.DataFrame,
    standards_df: pd.DataFrame,
) -> list:
    """Drop dataset-sourced standards from the main dataset.

    A lipid used as an internal standard should not also be analysed as a
    regular species, so it is removed from ``cleaned_df``. Safe to re-run:
    ``cleaned_df`` is rebuilt from the zero-filtering step on every rerun.

    Returns the list of removed lipid names.
    """
    if standards_df is None or standards_df.empty:
        return []
    if cleaned_df is None or cleaned_df.empty:
        return []

    filtered_df, removed = StandardsService.remove_standards_from_dataset(
        cleaned_df, standards_df
    )
    if removed:
        st.session_state.cleaned_df = filtered_df
    return removed


def _combine_standards(base_df: pd.DataFrame, added_df: pd.DataFrame) -> pd.DataFrame:
    """Union two standards DataFrames, de-duplicating on LipidMolec."""
    if base_df is None or base_df.empty:
        return added_df if added_df is not None else pd.DataFrame()
    if added_df is None or added_df.empty:
        return base_df

    combined = pd.concat([base_df, added_df], ignore_index=True)
    return combined.drop_duplicates(subset=['LipidMolec'], keep='first').reset_index(drop=True)


def _display_additional_dataset_standards(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Optional picker to add extra standards from the dataset.

    Additive: these supplement whatever standards are already active.
    Returns the added standards DataFrame (empty if none selected).
    """
    st.markdown("---")
    st.markdown("**➕ Add additional internal standards from the dataset**")
    st.caption(
        "For standards spiked into your samples that auto-detection can't "
        "recognize (e.g. unlabeled standards like `PG 14:0_14:0`). Selected "
        "lipids are used as standards and removed from the analysed species."
    )

    if cleaned_df is None or cleaned_df.empty or 'LipidMolec' not in cleaned_df.columns:
        st.info("No dataset available to select additional standards from.")
        return pd.DataFrame()

    all_lipids = sorted(cleaned_df['LipidMolec'].dropna().unique().tolist())

    preserved = st.session_state.get('additional_dataset_standards', [])
    default = [lip for lip in preserved if lip in all_lipids]

    selected = st.multiselect(
        "Additional standards from dataset:",
        options=all_lipids,
        default=default,
        key='additional_istd_select',
        on_change=_latch_expander_open,
    )
    st.session_state.additional_dataset_standards = selected

    if not selected:
        return pd.DataFrame()

    added_df, error = _extract_dataset_standards(cleaned_df, selected)
    if error:
        st.error(error)
        return pd.DataFrame()

    st.success(f"✓ Added {len(added_df)} standard(s) from the dataset.")
    return added_df


# =============================================================================
# Sub-Components
# =============================================================================

def _display_auto_detected_standards(
    auto_detected_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Display auto-detected standards, plus an optional additive picker.

    Returns the active standards DataFrame (auto-detected + any added).
    """
    # Clear custom standards when switching to automatic
    if st.session_state.custom_standards_df is not None:
        st.session_state.custom_standards_df = None
        st.session_state.custom_standards_mode = None

    if auto_detected_df is not None and not auto_detected_df.empty:
        st.success(f"✓ Found {len(auto_detected_df)} standards")
        st.dataframe(auto_detected_df, use_container_width=True)

        csv_download_button(
            auto_detected_df, "detected_standards.csv",
            key="download_auto_standards", on_click=_latch_expander_open,
        )
        base_df = auto_detected_df
    else:
        st.warning("No internal standards automatically detected in dataset.")
        base_df = pd.DataFrame()

    # Additional standards picked from the dataset (additive)
    added_df = _display_additional_dataset_standards(cleaned_df)
    if added_df is None or added_df.empty:
        return base_df

    removed = _remove_dataset_standards_from_main(cleaned_df, added_df)
    if removed:
        preview = ', '.join(removed[:5])
        more = f" and {len(removed) - 5} more" if len(removed) > 5 else ""
        st.info(
            f"Removed {len(removed)} lipid(s) from the analysed species "
            f"(now used as standards): {preview}{more}"
        )

    combined_df = _combine_standards(base_df, added_df)
    st.markdown("###### Active internal standards")
    st.dataframe(combined_df, use_container_width=True)
    csv_download_button(
        combined_df, "active_standards.csv",
        key="download_combined_standards", on_click=_latch_expander_open,
    )
    return combined_df


def _display_select_from_dataset(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """Pick lipids already present in the dataset to use as internal standards.

    Reuses the extract-from-dataset logic (intensities are pulled directly from
    the main dataset), but lets the user choose from a list instead of uploading
    a CSV of names.

    Returns the active standards DataFrame.
    """
    st.markdown("---")
    st.markdown(
        "Select one or more lipids already present in your dataset to use as "
        "internal standards — useful for unlabeled spiked standards (e.g. "
        "`PG 14:0_14:0`). Intensities are pulled directly from the data."
    )

    if cleaned_df is None or cleaned_df.empty or 'LipidMolec' not in cleaned_df.columns:
        st.warning("No dataset available to select standards from.")
        return pd.DataFrame()

    all_lipids = sorted(cleaned_df['LipidMolec'].dropna().unique().tolist())

    # Pre-select any previously chosen lipids that still exist
    preserved = st.session_state.get('selected_dataset_standards', [])
    default = [lip for lip in preserved if lip in all_lipids]

    selected = st.multiselect(
        "Lipids to use as internal standards:",
        options=all_lipids,
        default=default,
        key='dataset_standards_select',
        on_change=_latch_expander_open,
    )
    st.session_state.selected_dataset_standards = selected

    if not selected:
        st.info("Select at least one lipid to use as an internal standard.")
        return pd.DataFrame()

    standards_df, error = _extract_dataset_standards(cleaned_df, selected)
    if error:
        st.error(error)
        return _fallback_standards(auto_detected_df)

    st.session_state.custom_standards_df = standards_df
    st.session_state.custom_standards_mode = 'extract'

    removed = _remove_dataset_standards_from_main(cleaned_df, standards_df)
    if removed:
        preview = ', '.join(removed[:5])
        more = f" and {len(removed) - 5} more" if len(removed) > 5 else ""
        st.info(
            f"Removed {len(removed)} lipid(s) from the analysed species "
            f"(now used as standards): {preview}{more}"
        )

    st.success(f"✓ Using {len(standards_df)} selected standard(s) from the dataset.")
    st.dataframe(standards_df, use_container_width=True)
    csv_download_button(
        standards_df, "selected_standards.csv",
        key="download_selected_standards", on_click=_latch_expander_open,
    )

    return standards_df


def _display_custom_upload(
    cleaned_df: pd.DataFrame,
    auto_detected_df: pd.DataFrame
) -> pd.DataFrame:
    """Display external custom standards upload UI.

    Uploads complete external standards (with their own intensity values).
    To use standards already in the dataset, use the "Select from Dataset"
    source instead.

    Returns the active standards DataFrame.
    """
    st.markdown("---")

    current_mode = "complete"

    # Clear standards carried over from a different source/mode (e.g. extract)
    if (st.session_state.custom_standards_mode is not None and
        st.session_state.custom_standards_mode != current_mode and
        st.session_state.custom_standards_df is not None):
        st.session_state.custom_standards_df = None
        st.session_state.custom_standards_mode = None

    # Format guidance
    st.markdown(STANDARDS_COMPLETE_HELP)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload standards CSV",
        type=['csv'],
        key="standards_file_uploader",
        on_change=_latch_expander_open,
    )

    # Show preserved custom standards if file uploader is empty
    if uploaded_file is None and st.session_state.custom_standards_df is not None:
        return _display_preserved_custom_standards()

    # Process uploaded file
    if uploaded_file is not None:
        return _process_uploaded_standards(
            uploaded_file, cleaned_df, auto_detected_df,
            use_extract_mode=False, current_mode=current_mode,
        )

    # No file uploaded yet — do not fall back to auto-detected standards.
    # In this mode the user is providing their own standards, so nothing is
    # active until they upload (or switch back to Automatic Detection).
    st.info("Upload a standards CSV, or switch to Automatic Detection to use detected standards.")
    return pd.DataFrame()


def _display_preserved_custom_standards() -> pd.DataFrame:
    """Display previously uploaded custom standards with clear button."""
    st.success(f"✓ Using {len(st.session_state.custom_standards_df)} custom standards")
    st.dataframe(st.session_state.custom_standards_df, use_container_width=True)

    if st.button(
        "Clear Custom Standards",
        key="clear_custom_standards",
        on_click=_latch_expander_open,
    ):
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

        csv_download_button(
            result.standards_df, "custom_standards.csv",
            key="download_custom_standards", on_click=_latch_expander_open,
        )

        return result.standards_df

    except ValueError as ve:
        logger.error("Standards validation error: %s", ve)
        st.error(str(ve))
        return _fallback_standards(auto_detected_df)
    except UnicodeDecodeError as e:
        logger.error("Standards file encoding error: %s", e)
        st.error(
            "Could not read the standards file. Please ensure it is saved as a UTF-8 encoded CSV and try again."
        )
        return _fallback_standards(auto_detected_df)
    except pd.errors.EmptyDataError as e:
        logger.error("Empty standards file: %s", e)
        st.error("The uploaded standards file appears to be empty. Please check the file and try again.")
        return _fallback_standards(auto_detected_df)
    except (KeyError, pd.errors.ParserError) as e:
        logger.error("Standards file parse error: %s", e)
        st.error(
            "Could not parse the standards file. Please ensure it is a properly formatted CSV with consistent columns. "
            "If the issue persists after refreshing the app, contact abdih@mskcc.org."
        )
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

    # Keep the expander open once the user starts configuring standards, so a
    # rerun from any widget in this section doesn't collapse it mid-task.
    # Latched by each widget's on_change/on_click; starts collapsed on first load.
    with st.expander(
        "⚙️ Manage Internal Standards",
        expanded=st.session_state.get(INTSTA_EXPANDER_KEY, False),
    ):
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

        # Restore widget value from preserved session state (lost during module navigation)
        standards_options = [
            "Automatic Detection",
            "Select from Dataset",
            "Upload Custom Standards",
        ]
        if 'standards_source_radio' not in st.session_state:
            preserved_source = st.session_state.get('standards_source')
            if preserved_source in standards_options:
                st.session_state.standards_source_radio = preserved_source

        standards_source = st.radio(
            "Standards source:",
            standards_options,
            horizontal=True,
            key="standards_source_radio",
            label_visibility="collapsed",
            on_change=_latch_expander_open,
        )
        st.session_state.standards_source = standards_source

        if standards_source == "Automatic Detection":
            active_standards_df = _display_auto_detected_standards(
                auto_detected_df, cleaned_df
            )
        elif standards_source == "Select from Dataset":
            active_standards_df = _display_select_from_dataset(cleaned_df, auto_detected_df)
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
