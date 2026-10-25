"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
StreamlitAdapter is the one facade the UI calls. Its session state schema and
stage registry live in session_state.py; the sidebar's regroup and exclusion
memory in sidebar_state.py; its cached wrappers in module1_cache.py,
qc_cache.py and analysis_cache.py. It inherits all of them.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Optional

import pandas as pd
import streamlit as st

from app.constants import (
    PAGE_LANDING, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS,
    resolve_format_enum,
)

from ..services.format_detection import FormatDetectionService, DataFormat
from .analysis_cache import AnalysisCache
from .module1_cache import Module1Cache
from .qc_cache import QualityCheckCache
from .session_state import (  # noqa: F401 (re-exported)
    SessionState, _ROUTING_FIELDS, _ROUTING_WIDGETS, _STAGE_NAMES, _STAGES,
    _fingerprint, _inputs_key,
)
from .sidebar_state import SidebarState


class StreamlitAdapter(SidebarState, Module1Cache, QualityCheckCache, AnalysisCache):
    """
    Adapter for managing Streamlit session state and caching.

    Responsibilities:
    1. Initialize session state with default values
    2. Provide type-safe accessors for session state
    3. Provide cached wrappers for service calls (inherited)
    4. Handle UI data conversion to service models

    All methods are static - no instance state is stored.

    Session State Access Patterns
    =============================
    UI files should use exactly one of these three patterns per key:

    1. **Direct access** — for simple reads/writes of data keys owned by the
       current file. Use ``st.session_state['key']`` for reads that must exist
       and ``st.session_state['key'] = value`` for writes.  Use
       ``st.session_state.get('key')`` when the key may be ``None``. The key
       must belong to a stage in ``_STAGES`` (``session_state.py``), which is
       what clears it at the right time; a registry test scans the UI for
       keys that are not. Never clear keys by hand: use ``invalidate_from``
       or ``invalidate_results_from``.

    2. **``restore_widget_value`` / ``save_widget_value``** — for widget keys
       that must survive module navigation (Streamlit removes widget keys for
       non-rendered widgets).  Call ``restore_widget_value`` *before* the widget
       renders and ``save_widget_value`` in an ``on_change`` callback or
       immediately after the widget.  Pair each widget key with a
       ``_preserved_*`` session state key declared in ``SessionState``.

    3. **Widget ``key=`` parameter** — Streamlit widgets automatically read/write
       ``st.session_state[key]``.  Do *not* also write to the same key manually;
       let the widget own it.  Guard ``st.session_state[key] = default`` with
       ``if key not in st.session_state`` to avoid overwriting user selections.

    When to use which:
    - Data flowing between UI files → pattern 1 (direct access)
    - Widget state surviving module switches → pattern 2 (preserve/restore)
    - Widget with ``on_change`` callback → pattern 2 + pattern 3
    - Widget default that only needs to survive within the same page → pattern 3
    """

    # ==================== Session State Initialization ====================

    @staticmethod
    def initialize_session_state() -> None:
        """
        Initialize all session state variables with default values.

        Call this at the beginning of main() to ensure all required
        session state keys exist before they are accessed.
        Derives defaults from the SessionState dataclass.
        """
        defaults = SessionState()
        for f in fields(SessionState):
            if f.name not in st.session_state:
                st.session_state[f.name] = getattr(defaults, f.name)

    @staticmethod
    def invalidate_from(stage: str) -> None:
        """Clear ``stage`` and every later stage of ``_STAGES``.

        Fields go back to their defaults; widget, plain and prefixed keys are
        popped (re-created by their widgets, or re-initialized from
        ``SessionState`` on the next run).

        Args:
            stage: A ``_STAGES`` name ('upload', 'standardize', 'experiment',
                'processing', 'report', 'qc', 'analysis').
        """
        stages = _STAGES[_STAGE_NAMES.index(stage):]
        # Pop widget and plain keys first: Streamlit forbids assigning to a
        # key instantiated as a widget during the current run, but allows
        # deletion.
        for s in stages:
            for key in s.widgets | s.plain | {_inputs_key(s.name)}:
                st.session_state.pop(key, None)
        defaults = SessionState()
        for s in stages:
            for name in s.fields:
                st.session_state[name] = getattr(defaults, name)
        prefixes = tuple(p for s in stages for p in s.prefixes)
        if prefixes:
            field_names = {f.name for f in fields(SessionState)}
            for key in [k for k in list(st.session_state.keys())
                        if isinstance(k, str) and k.startswith(prefixes)
                        and k not in field_names]:
                st.session_state.pop(key, None)

    @staticmethod
    def track_inputs(stage: str, *inputs: Any) -> bool:
        """Drop what was computed from ``stage``'s previous inputs.

        Call where the stage reads its inputs, before it computes anything.
        When ``inputs`` differ from the last run's, the ``derived`` fields of
        ``stage`` and of every later stage go back to their defaults; the
        user's settings (other fields, widgets) are kept.

        Args:
            stage: A ``_STAGES`` name.
            *inputs: Everything the stage's results are computed from.

        Returns:
            True when results were dropped.
        """
        fingerprint = _fingerprint(inputs)
        previous = st.session_state.get(_inputs_key(stage))
        st.session_state[_inputs_key(stage)] = fingerprint
        if previous is None or previous == fingerprint:
            return False
        StreamlitAdapter.invalidate_results_from(stage)
        return True

    @staticmethod
    def invalidate_results_from(stage: str) -> None:
        """Drop the results of ``stage`` and every later stage, keeping settings.

        The ``derived`` fields go back to their defaults so the next run
        recomputes them; widgets and the other fields are untouched. For a
        callback that changes one of a stage's inputs itself (``track_inputs``
        covers inputs read on the run).

        Args:
            stage: A ``_STAGES`` name.
        """
        defaults = SessionState()
        for s in _STAGES[_STAGE_NAMES.index(stage):]:
            for name in s.derived:
                st.session_state[name] = getattr(defaults, name)

    @staticmethod
    def reset_data_state() -> None:
        """Reset all data-related session state for a fresh upload.

        Clears every stage and returns to Module 1; only ``page`` (app-level
        routing) is kept.
        """
        StreamlitAdapter.invalidate_from('upload')
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def reset_for_new_source(source_id: str) -> bool:
        """Reset data state when a different dataset source arrives.

        Uploading another file, or loading sample data, over a loaded dataset
        would otherwise leave every downstream frame (standardized, cleaned,
        normalized, QC, analysis) computed from the previous one. The source
        id is stable across reruns (an upload's file_id, or the sample-data
        format), so the same file surviving a rerun does not reset anything.

        Args:
            source_id: Identifier for the incoming dataset.

        Returns:
            True if a loaded dataset was reset.
        """
        if st.session_state.get('_dataset_source_id') == source_id:
            return False
        had_data = st.session_state.get('raw_df') is not None
        if had_data:
            StreamlitAdapter.reset_data_state()
            st.cache_data.clear()
        st.session_state['_dataset_source_id'] = source_id
        return had_data

    @staticmethod
    def reset_format_derived_state() -> None:
        """Clear everything derived from the upload when the format changes.

        Keeps the uploaded file (and the sample-data flags) so it is
        re-standardized under the newly selected format; standardization
        outputs, experiment setup and all downstream state are cleared, and
        the user returns to Module 1.
        """
        StreamlitAdapter.invalidate_from('standardize')
        # Same reason as in reset_to_experiment_setup(): an explicit False
        # beats the checkbox's cached value on a pre-fill rerun.
        st.session_state['confirm_checkbox'] = False
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def reset_to_experiment_setup() -> None:
        """Soft reset: keep the uploaded/standardized data, clear everything else.

        Clears experiment config, grouping, processing, normalization, QC and
        analysis state plus the ``@st.cache_data`` cache, then returns the user
        to the Define Experiment step — all via a rerun, so no page refresh is
        needed. Unlike ``reset_data_state()`` (a full wipe back to landing),
        this preserves the upload and standardization outputs so the user does
        not have to re-upload the file or re-map columns.
        """
        # A confirmed manual regroup leaves standardized_df renumbered, which no
        # longer matches the column mapping or the seeded sample names. Restore
        # the pre-regroup data, since the grouping itself is being reset.
        pre_regroup = st.session_state.get('_pre_regroup_df')
        StreamlitAdapter.invalidate_from('experiment')
        if pre_regroup is not None:
            st.session_state['standardized_df'] = pre_regroup
        # Force the confirm checkbox off. Popping its key is not enough: the
        # reset button renders before the confirm checkbox, and formats that
        # pre-fill the experiment (LipidSearch 5.2 alignment, Metabolomics
        # Workbench auto-detect) fire an extra st.rerun() in which Streamlit
        # re-creates the checkbox with identical parameters and restores its
        # cached 'checked' value. An explicit False in session state overrides
        # that. Safe to set: the widget is not instantiated yet this run.
        st.session_state['confirm_checkbox'] = False
        # Start from Module 1 so a re-confirmed experiment runs the pipeline
        # from the top rather than jumping into a stale QC/Analysis view.
        st.session_state['module'] = MODULE_DATA_PROCESSING
        # Drop cached computations (box plots, PCA, heatmaps, ...).
        st.cache_data.clear()

    # ==================== Navigation ====================

    @staticmethod
    def go_to_quality_check() -> None:
        """Next: Module 1 -> Quality Check & Analysis.

        QC and analysis start over, so Module 2/3 freeze the current Module 1
        output (see the qc_input_* fields) rather than reuse a previous one.
        """
        StreamlitAdapter.invalidate_from('qc')
        st.session_state['module'] = MODULE_QC_ANALYSIS

    @staticmethod
    def back_to_data_processing() -> None:
        """Back from Quality Check & Analysis to Module 1."""
        StreamlitAdapter.invalidate_from('qc')
        st.session_state['module'] = MODULE_DATA_PROCESSING

    @staticmethod
    def back_to_home() -> None:
        """Back to the landing page, dropping the dataset."""
        st.session_state['page'] = PAGE_LANDING
        StreamlitAdapter.reset_data_state()

    # ==================== Widget Preservation ====================

    @staticmethod
    def restore_widget_value(widget_key: str, preserved_key: str, default: Any) -> Any:
        """Restore a widget's value from its preserved session state key.

        When navigating between modules, Streamlit removes widget keys for
        non-rendered widgets. This method restores the value from a preserved
        key so the widget renders with the user's last selection.

        Args:
            widget_key: The Streamlit widget key (e.g., 'bqc_filter_choice').
            preserved_key: The session state key storing the preserved value
                          (e.g., '_preserved_bqc_filter_choice').
            default: Default value if neither key exists.

        Returns:
            The value to use for the widget's initial value/index.
        """
        if widget_key in st.session_state:
            return st.session_state[widget_key]
        preserved = st.session_state.get(preserved_key)
        if preserved is not None:
            return preserved
        return default

    @staticmethod
    def save_widget_value(widget_key: str, preserved_key: str) -> None:
        """Save a widget's current value to its preserved session state key.

        Call this after the widget renders (or in an on_change callback)
        to persist the value across module navigation.

        Args:
            widget_key: The Streamlit widget key to read from.
            preserved_key: The session state key to write to.
        """
        value = st.session_state.get(widget_key)
        if value is not None:
            st.session_state[preserved_key] = value

    # ==================== Service Wrappers ====================


    @staticmethod
    def detect_format_mismatch(df: pd.DataFrame, data_format: str) -> Optional[DataFormat]:
        """Return the format ``df`` looks like when it is not the selected one.

        None when detection agrees or is inconclusive (UNKNOWN). The dropdown
        alone decides how the file is read; this only feeds a warning.
        """
        detected = FormatDetectionService.detect_format(df)
        if detected in (DataFormat.UNKNOWN, resolve_format_enum(data_format)):
            return None
        return detected
