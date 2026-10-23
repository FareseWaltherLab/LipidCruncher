"""The sidebar's manual-regroup snapshot and sample-exclusion memory, part of
the StreamlitAdapter facade.

A manual regroup renumbers samples (intensity[s4] -> intensity[s1]) without
changing the column names, so it must always be computed from a snapshot of
the upload order (``_pre_regroup_df``, ``_pre_regroup_sample_names``,
``original_column_order``) and undone by restoring that snapshot. Sidebar
exclusions are remembered by each sample's upload label
(``_preserved_excluded_samples``) so a later regroup keeps excluding the same
samples. These methods are the only writers of those keys.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..models.sample_frame import SampleFrame


class SidebarState:
    """Regroup and exclusion state kept for the sidebar; StreamlitAdapter inherits it."""

    # ==================== Manual regroup ====================

    @staticmethod
    def begin_regroup(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, str]]]:
        """Snapshot the upload order the first time a regroup starts.

        Returns the base ``(df, sample_names)`` every regroup is computed
        from. Without the snapshot, each Streamlit rerun would re-apply the
        swap to already-swapped data and compound the display-name remap.

        Args:
            df: The sidebar's current standardized data.
        """
        if st.session_state.get('original_column_order') is None:
            st.session_state['original_column_order'] = df.columns.tolist()
            st.session_state['_pre_regroup_df'] = df.copy()
            st.session_state['_pre_regroup_sample_names'] = (
                st.session_state.get('sample_names')
            )
        return (
            st.session_state.get('_pre_regroup_df', df),
            st.session_state.get('_pre_regroup_sample_names'),
        )

    @staticmethod
    def apply_regroup(regrouped: SampleFrame) -> None:
        """Record a valid regroup: its renumbered names and each label's origin."""
        st.session_state['sample_names'] = regrouped.names
        st.session_state['regroup_origin'] = regrouped.origin
        st.session_state['grouping_complete'] = True

    @staticmethod
    def undo_regroup(df: pd.DataFrame) -> pd.DataFrame:
        """"Yes, properly grouped": restore the upload order and drop the snapshot.

        Returns the data to continue with: the pre-regroup snapshot when one
        exists (also committed to ``standardized_df`` now, since a run with
        Confirm unchecked returns before writing it), otherwise ``df``.
        """
        st.session_state['grouping_complete'] = True
        st.session_state['regroup_origin'] = None
        st.session_state['_preserved_regroup_selections'] = {}
        pre_regroup = st.session_state.get('_pre_regroup_df')
        if pre_regroup is None:
            return df
        df = pre_regroup.copy()
        st.session_state['standardized_df'] = df
        SidebarState.drop_regroup_snapshot()
        return df

    @staticmethod
    def drop_regroup_snapshot() -> None:
        """Forget the regroup snapshot, restoring the pre-regroup display names.

        For when the data it was taken from is replaced (the MS-DIAL
        raw/normalized toggle) as well as when a regroup is undone.
        """
        st.session_state.pop('_pre_regroup_df', None)
        st.session_state.pop('original_column_order', None)
        if '_pre_regroup_sample_names' in st.session_state:
            st.session_state['sample_names'] = st.session_state.pop(
                '_pre_regroup_sample_names'
            )

    @staticmethod
    def set_sample_names(names: Dict[str, str]) -> None:
        """Store edited display names, also under each sample's upload label
        while regrouped (the names are rebuilt from the snapshot every run)."""
        st.session_state['sample_names'] = names
        regroup_origin = st.session_state.get('regroup_origin')
        if regroup_origin and '_pre_regroup_sample_names' in st.session_state:
            st.session_state['_pre_regroup_sample_names'] = {
                regroup_origin.get(s, s): name for s, name in names.items()
            }

    # ==================== Sample exclusion ====================

    @staticmethod
    def restore_excluded_samples() -> Tuple[List[str], Dict[str, str]]:
        """The exclusion picker's starting selection, in current sidebar labels.

        Exclusions are kept by upload label; when the regroup changed since
        they were saved, the picker's own key holds stale labels and is
        dropped so it re-renders from the translated selection.

        Returns:
            ``(selected labels, regroup_origin)`` — the origin map is empty
            when the samples are not regrouped.
        """
        regroup_origin = st.session_state.get('regroup_origin') or {}
        saved_under = st.session_state.get('_preserved_excluded_regroup_origin') or {}
        if saved_under != regroup_origin:
            st.session_state.pop('exclude_samples', None)
        to_label = {origin: label for label, origin in regroup_origin.items()}
        restored = st.session_state.get('exclude_samples', [
            to_label.get(origin, origin)
            for origin in st.session_state.get('_preserved_excluded_samples') or []
        ])
        return restored, regroup_origin

    @staticmethod
    def save_excluded_samples(
        excluded: List[str], regroup_origin: Dict[str, str],
    ) -> None:
        """Remember the picked exclusions by upload label, and the regroup
        they were picked under."""
        st.session_state['_preserved_excluded_samples'] = [
            regroup_origin.get(s, s) for s in excluded
        ]
        st.session_state['_preserved_excluded_regroup_origin'] = regroup_origin
