"""
Streamlit adapter layer.

Bridge between UI and services - handles session state management and caching.
StreamlitAdapter is the one facade the UI calls. Its session state schema and
stage registry live in session_state.py; its cached wrappers live in
module1_cache.py, qc_cache.py and analysis_cache.py, which it inherits.
"""
from __future__ import annotations

import uuid
from dataclasses import fields
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from app.constants import (
    PAGE_LANDING, MODULE_DATA_PROCESSING, MODULE_QC_ANALYSIS,
    resolve_format_enum,
)

from ..models.experiment import ExperimentConfig
from ..models.isotope_tracing import IsotopeCorrectionConfig
from ..services.format_detection import FormatDetectionService, DataFormat
from ..workflows.isotope_tracing import IsotopeTracingWorkflow, IsotopeTracingResult
from .analysis_cache import AnalysisCache
from .module1_cache import Module1Cache
from .qc_cache import QualityCheckCache
from .session_state import (  # noqa: F401 (re-exported)
    SessionState, _ROUTING_FIELDS, _ROUTING_WIDGETS, _STAGE_NAMES, _STAGES,
    _fingerprint, _inputs_key,
    _MAIN_APP_NAV_WIDGET_KEYS, _MAIN_APP_NAV_WIDGET_PREFIXES,
    _AI_STUDIO_NAV_WIDGET_KEYS, _AI_STUDIO_NAV_WIDGET_PREFIXES, _NAV_PRESERVE_PREFIX,
)


class _UncachedResult(Exception):
    """Carries a result out of a cached function without it being cached."""

    def __init__(self, result):
        super().__init__()
        self.result = result


def _normalize_lc_question(question: str) -> str:
    """Canonical cache key for a LipidCruncher-questions prompt."""
    return " ".join(question.split()).lower()


@st.cache_resource
def _lc_answer_cache() -> Dict[str, str]:
    """Process-wide cache of canonical answers to LipidCruncher app questions.

    Keyed by normalized question text. Shared across sessions because these
    answers describe the application itself and do not depend on user data,
    so reuse is both correct and desirable. Doubles as a fallback when the
    AI service is unavailable.
    """
    return {}


@st.cache_resource(show_spinner=False)
def _isocorrector_available() -> bool:
    """Whether R + IsoCorrectoR can run here, probed once per process.

    The probe starts Rscript (~0.7 s), which every rerun of the isotope
    tracing page used to pay. R comes with the image, so the answer cannot
    change under a running process.
    """
    from ..services.isotope_correction import check_isocorrector_available
    return check_isocorrector_available()


class StreamlitAdapter(Module1Cache, QualityCheckCache, AnalysisCache):
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
       ``st.session_state.get('key')`` when the key may be ``None``.

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
                'processing', 'report', 'ai_studio', 'isotope', 'qc',
                'analysis').
        """
        StreamlitAdapter._clear(_STAGES[_STAGE_NAMES.index(stage):])

    @staticmethod
    def clear_stage(stage: str) -> None:
        """Clear one stage only, for state no other stage depends on (the
        AI Studio, the isotope tracing page)."""
        StreamlitAdapter._clear((_STAGES[_STAGE_NAMES.index(stage)],))

    @staticmethod
    def _clear(stages) -> None:
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
        defaults = SessionState()
        for s in _STAGES[_STAGE_NAMES.index(stage):]:
            for name in s.derived:
                st.session_state[name] = getattr(defaults, name)
        return True

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

    @staticmethod
    def _snapshot_widgets(static_keys, prefixes) -> None:
        """Copy each named/prefixed widget key into a parallel
        ``_nav_preserved_*`` key. Shared by the per-page preserve helpers
        below. Streamlit garbage-collects widget keys whose widgets don't
        render in the current run, so without this snapshot the user
        would lose their sidebar configuration on every cross-page hop.
        """
        keys = list(static_keys)
        keys += [
            k for k in list(st.session_state.keys())
            if isinstance(k, str) and k.startswith(prefixes)
        ]
        for key in keys:
            if key in st.session_state:
                st.session_state[f'{_NAV_PRESERVE_PREFIX}{key}'] = (
                    st.session_state[key]
                )

    @staticmethod
    def preserve_main_app_widgets() -> None:
        """Snapshot main-app sidebar widget keys before the user navigates
        to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        StreamlitAdapter._snapshot_widgets(
            _MAIN_APP_NAV_WIDGET_KEYS, _MAIN_APP_NAV_WIDGET_PREFIXES,
        )
        # The file uploader cannot be snapshotted; it comes back empty.
        if st.session_state.get('raw_df') is not None:
            st.session_state['upload_kept_across_pages'] = True

    @staticmethod
    def preserve_ai_studio_widgets() -> None:
        """Snapshot AI Studio sidebar widget keys before the user
        navigates to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        StreamlitAdapter._snapshot_widgets(
            _AI_STUDIO_NAV_WIDGET_KEYS, _AI_STUDIO_NAV_WIDGET_PREFIXES,
        )

    @staticmethod
    def restore_navigation_widgets() -> None:
        """Restore any widget key that was preserved by either
        ``preserve_main_app_widgets()`` or ``preserve_ai_studio_widgets()``
        and is currently absent from session_state. Generic over both
        pages because each page's widget keys have unique prefixes — keys
        not matching any rendered widget on the current page are simply
        unused. If Streamlit kept the widget alive, the live value wins.
        """
        prefix_len = len(_NAV_PRESERVE_PREFIX)
        for key in list(st.session_state.keys()):
            if not isinstance(key, str):
                continue
            if not key.startswith(_NAV_PRESERVE_PREFIX):
                continue
            target = key[prefix_len:]
            if target not in st.session_state:
                st.session_state[target] = st.session_state[key]

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

    # ==================== Cached Isotope Tracing Wrapper ====================

    @staticmethod
    def isocorrector_available() -> bool:
        """Whether the isotope tracing page can run (R + IsoCorrectoR)."""
        return _isocorrector_available()

    @staticmethod
    def run_isotope_correction(
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> IsotopeTracingResult:
        """Isotope tracing workflow, cached on success.

        Keyed on the three input DataFrames and the config, so re-running with
        unchanged inputs skips the (slow) R subprocess. A run that failed in R
        (a timeout, a crash, inputs IsoCorrectoR rejected) is not cached, so
        retrying really re-runs it.

        Args:
            measurement_df: Measurement file.
            molecule_df: Molecule file.
            element_df: Element file.
            config: IsoCorrectoR settings.

        Returns:
            IsotopeTracingResult — success with the correction, or failure with
            validation_errors / error_message.
        """
        try:
            return StreamlitAdapter._run_isotope_correction_cached(
                measurement_df, molecule_df, element_df, config
            )
        except _UncachedResult as uncached:
            return uncached.result

    @staticmethod
    @st.cache_data(
        show_spinner="Running IsoCorrectoR...",
        hash_funcs={IsotopeCorrectionConfig: hash},
    )
    def _run_isotope_correction_cached(
        measurement_df: pd.DataFrame,
        molecule_df: pd.DataFrame,
        element_df: pd.DataFrame,
        config: IsotopeCorrectionConfig,
    ) -> IsotopeTracingResult:
        result = IsotopeTracingWorkflow.run(
            measurement_df, molecule_df, element_df, config
        )
        if result.error_message:
            raise _UncachedResult(result)  # Streamlit does not cache exceptions
        return result

    # ==================== AI Chat Wrappers (NOT cached — stateful) ====================

    @staticmethod
    def stream_ai_transform(
        user_message: str,
        session,
        source_df: pd.DataFrame,
    ):
        """Stream one turn of the AI data transformation chat.

        Returns (explanation_generator, finalize). The UI streams the
        explanation (st.write_stream), then calls finalize() to execute the
        code and obtain a TransformStepResult.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.stream_transform_message(
            user_message=user_message,
            session=session,
            source_df=source_df,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
            keep_columns=st.session_state.get('chat_transform_kept_columns') or (),
            sample_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
        )

    @staticmethod
    def keep_ai_transform_columns(session, source_df: pd.DataFrame):
        """Re-run the last transform keeping the columns it refused only
        for their metadata-like names, and remember them for later turns."""
        from ..workflows.ai_chat import AIChatWorkflow
        kept = tuple(st.session_state.get('chat_transform_kept_columns') or ()) + tuple(
            st.session_state.get('chat_transform_refused_metadata') or ()
        )
        st.session_state.chat_transform_kept_columns = kept
        return AIChatWorkflow.keep_transform_columns(
            session=session,
            source_df=source_df,
            keep_columns=kept,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
            sample_columns=st.session_state.get(
                'chat_studio_sample_columns_original'
            ),
        )

    @staticmethod
    def stream_ai_viz(
        user_message: str,
        session,
        analysis_df: pd.DataFrame,
        experiment: ExperimentConfig,
    ):
        """Stream one turn of the AI visualization chat.

        Returns (explanation_generator, finalize). The UI streams the
        explanation, then calls finalize() for a VizStepResult.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.stream_viz_message(
            user_message=user_message,
            session=session,
            analysis_df=analysis_df,
            experiment=experiment,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_renamed'
            ),
        )

    @staticmethod
    def stream_ai_data_questions(
        user_message: str,
        session,
        source_df: pd.DataFrame,
    ):
        """Stream one turn of the AI data questions chat.

        Returns (explanation_generator, finalize). The UI streams the
        explanation, then calls finalize() for a DataQuestionsStepResult.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.stream_data_questions_message(
            user_message=user_message,
            session=session,
            source_df=source_df,
            numeric_columns=st.session_state.get(
                'chat_studio_sample_columns_renamed'
            ),
            experiment=st.session_state.get('chat_studio_experiment'),
        )

    @staticmethod
    def stream_ai_lc_questions(
        user_message: str,
        session,
    ):
        """Stream one turn of the LipidCruncher questions chat.

        Returns (session_with_user_message, text_chunk_generator). The UI
        consumes the generator (e.g. via st.write_stream) and appends the
        assistant message from the accumulated text.
        """
        from ..workflows.ai_chat import AIChatWorkflow
        return AIChatWorkflow.stream_lc_questions_message(
            user_message=user_message,
            session=session,
        )

    @staticmethod
    def get_cached_lc_answer(question: str) -> Optional[str]:
        """Return a cached answer for a LipidCruncher-questions prompt, if any."""
        return _lc_answer_cache().get(_normalize_lc_question(question))

    @staticmethod
    def set_cached_lc_answer(question: str, answer: str) -> None:
        """Store a canonical answer for a LipidCruncher-questions prompt."""
        _lc_answer_cache()[_normalize_lc_question(question)] = answer

    # ------------------------------------------------------------------
    # AI Studio interaction logging (super-user preview)
    # ------------------------------------------------------------------

    @staticmethod
    def _ai_log_session_id() -> str:
        """Return an anonymous per-Streamlit-session id, creating it once.

        Groups all turns from one browser session together in the logs
        without identifying the user.
        """
        sid = st.session_state.get("ai_log_session_id")
        if sid is None:
            sid = uuid.uuid4().hex
            st.session_state["ai_log_session_id"] = sid
        return sid

    # ------------------------------------------------------------------
    # AI Studio usage limits
    # ------------------------------------------------------------------

    @staticmethod
    def _ai_client_id() -> Optional[str]:
        """The visitor's address, for the per-client AI Studio limit.

        Behind the AWS load balancer every connection comes from the
        balancer, so ``st.context.ip_address`` is the same for everyone.
        The balancer appends the address it saw to ``X-Forwarded-For``, so
        the last entry is the one a visitor cannot forge (anything before
        it is whatever the visitor sent). That holds for exactly one proxy
        in front of the app; reached directly, the header is the visitor's
        to set, and only the session and global limits really hold.
        Locally there is neither, and the per-client limit is skipped.
        """
        forwarded = st.context.headers.get("X-Forwarded-For") or ""
        last = forwarded.rsplit(",", 1)[-1].strip()
        return last or st.context.ip_address

    @staticmethod
    def ai_usage_refusal(runs_code: bool = True) -> Optional[str]:
        """Why this visitor may not start an AI turn now, or None."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        return UsageLimitService.refusal(
            StreamlitAdapter._ai_log_session_id(),
            StreamlitAdapter._ai_client_id(),
            runs_code=runs_code,
        )

    @staticmethod
    def start_ai_turn(runs_code: bool = True) -> Optional[str]:
        """Count one AI turn against every limit; the refusal, or None."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        return UsageLimitService.start_turn(
            StreamlitAdapter._ai_log_session_id(),
            StreamlitAdapter._ai_client_id(),
            runs_code=runs_code,
        )

    @staticmethod
    def record_ai_execution() -> None:
        """Count one execution of AI-generated code for this session."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        UsageLimitService.record_execution(
            StreamlitAdapter._ai_log_session_id(),
        )

    @staticmethod
    def log_ai_turn(
        *,
        tab: str,
        question: str,
        answer: str,
        code: Optional[str] = None,
        produced_table: bool = False,
        produced_figure: bool = False,
        error: Optional[str] = None,
        history=None,
        df: Optional[pd.DataFrame] = None,
        experiment: Optional[ExperimentConfig] = None,
        figure=None,
        unverified_numbers=(),
    ) -> str:
        """Log one completed AI Studio turn. Returns the turn_id so the UI
        can attach a later 👍/👎 feedback record to it.

        ``history`` is the chat's messages *before* this turn; ``df`` /
        ``experiment`` describe the data the tab ran against (the upload's
        name and detected format come from session state); ``figure`` is
        the Plotly figure the turn produced, if any.
        """
        from ..models.chat import AIConfig
        from ..services.ai_chat.interaction_logger import InteractionLoggingService
        turn_id = InteractionLoggingService.new_turn_id()
        InteractionLoggingService.log_turn(
            turn_id=turn_id,
            session_id=StreamlitAdapter._ai_log_session_id(),
            tab=tab,
            question=question,
            answer=answer,
            code=code,
            produced_table=produced_table,
            produced_figure=produced_figure,
            error=error,
            model=AIConfig().model,
            dataset=InteractionLoggingService.describe_dataset(
                df,
                source=(
                    st.session_state.get("chat_studio_pipeline_source")
                    or st.session_state.get("chat_studio_file_id")
                ),
                data_format=st.session_state.get("chat_studio_detected_format"),
                experiment=experiment,
            ),
            history=history,
            figure=InteractionLoggingService.summarize_figure(figure),
            unverified_numbers=unverified_numbers,
        )
        return turn_id

    @staticmethod
    def log_ai_feedback(*, turn_id: str, verdict: str, note: str = "") -> None:
        """Log a 👍/👎 signal for a previously-logged turn."""
        from ..services.ai_chat.interaction_logger import InteractionLoggingService
        InteractionLoggingService.log_feedback(
            turn_id=turn_id,
            session_id=StreamlitAdapter._ai_log_session_id(),
            verdict=verdict,
            note=note,
        )
