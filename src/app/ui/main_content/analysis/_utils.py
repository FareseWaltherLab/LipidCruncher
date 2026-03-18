"""
Shared utility functions for the analysis module.

Provides fatty acid compatibility checks and consolidated lipid handling.
"""

from typing import List

import pandas as pd
import streamlit as st

from app.services.plotting.saturation_plot import SaturationPlotterService


def _check_fa_compatibility(df: pd.DataFrame) -> None:
    """Show a warning if lipid names lack detailed fatty acid composition."""
    if 'LipidMolec' not in df.columns:
        return
    sample = df['LipidMolec'].head(20)
    has_detailed = sample.str.contains('_').any()
    if not has_detailed:
        st.info(
            "⚠️ Your data appears to use consolidated lipid names "
            "(e.g., PC(34:1) instead of PC(16:0_18:1)). "
            "Saturation and pathway analyses work best with "
            "detailed fatty acid composition."
        )


def _display_consolidated_lipids(
    df: pd.DataFrame,
    selected_classes: List[str],
    key_prefix: str,
) -> List[str]:
    """Display consolidated lipid detection and exclusion UI.

    Returns:
        List of lipid names to exclude.
    """
    consolidated = SaturationPlotterService.identify_consolidated_lipids(
        df, selected_classes,
    )

    if not consolidated:
        return []

    st.markdown("---")
    st.markdown("#### ⚠️ Consolidated Format Lipids")

    # Build summary table
    summary_rows = []
    all_consolidated = []
    for cls, lipids in consolidated.items():
        total = len(df[df['ClassKey'] == cls])
        count = len(lipids)
        pct = (count / total * 100) if total > 0 else 0
        summary_rows.append({
            'Class': cls,
            'Total Lipids': total,
            'Consolidated': count,
            '% Consolidated': f"{pct:.1f}%",
        })
        for lip in lipids:
            all_consolidated.append(f"{lip} ({cls})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Consolidated', ascending=False)
    st.dataframe(summary_df, use_container_width=True)

    exclude_labels = st.multiselect(
        f"Select lipids to exclude ({len(all_consolidated)} detected):",
        all_consolidated,
        default=[],
        help="Exclude consolidated format lipids with multiple fatty acid chains.",
        key=f'{key_prefix}_exclude_consolidated',
    )

    if exclude_labels:
        st.success(f"{len(exclude_labels)} lipid(s) will be excluded.")

    # Extract lipid names from "lipid (class)" format
    excluded_names = []
    for label in exclude_labels:
        name = label.rsplit(' (', 1)[0]
        excluded_names.append(name)

    return excluded_names
