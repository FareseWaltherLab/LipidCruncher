"""
The AI Studio's part of the adapter: its session state, its reset stage, the
sidebar snapshots kept across a visit to another page, and the AI chat,
answer cache, usage limit and logging calls. StreamlitAdapter inherits
AIStudioAdapter; SessionState inherits AIStudioState.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ..models.experiment import ExperimentConfig


@dataclass
class AIStudioState:
    """The AI Studio's session state fields (see SessionState)."""

    # --- AI Chat (owner: ai_chat/chat_panel.py) ---
    chat_transform_session: Optional[Any] = None
    chat_viz_session: Optional[Any] = None
    chat_data_questions_session: Optional[Any] = None
    # About the app rather than the data, so no reset clears it (see
    # _ROUTING_FIELDS); the AI Studio's Back to Home does.
    chat_lc_questions_session: Optional[Any] = None
    chat_mode: Optional[str] = None
    chat_transform_result: Optional[pd.DataFrame] = None
    chat_transform_report: Optional[Any] = None
    # Sample columns with metadata-like names: refused by the last transform
    # turn / confirmed by the user as samples for this conversation.
    chat_transform_refused_metadata: Tuple[str, ...] = ()
    chat_transform_kept_columns: Tuple[str, ...] = ()
    chat_viz_figures: Dict[str, Any] = field(default_factory=dict)

    # --- AI Studio (owner: ai_studio_page.py) ---
    # `chat_studio_df_original` holds the dataframe with the user's
    # original sample column names (what they uploaded / imported). It is
    # what the data preview, Data Questions, and Transform tabs read.
    # `chat_studio_df` holds a reordered/renamed copy with sample columns
    # normalized to `intensity[s1..sN]` per condition — required by the
    # Custom Visualization tab's plotters because ExperimentConfig
    # generates its sample labels as s1..sN.
    chat_studio_df: Optional[pd.DataFrame] = None
    chat_studio_df_original: Optional[pd.DataFrame] = None
    chat_studio_experiment: Optional[Any] = None
    # Tracks where the AI Studio data came from. None for direct upload,
    # 'normalized' or 'raw' for the two pipeline-import options. Used to
    # hard-lock Custom Visualization when the source is 'raw' regardless
    # of whether the columns happen to satisfy the validation check.
    chat_studio_pipeline_source: Optional[str] = None
    # Detected format string (DataFormat.value) for the uploaded file.
    # Used to block the AI Studio flow when a known standard format
    # (LipidSearch, MS-DIAL) is uploaded directly — those should be
    # processed through the main pipeline first.
    chat_studio_detected_format: Optional[str] = None
    # Names of the columns in `chat_studio_df_original` that the user
    # (or pipeline import) marked as sample-intensity columns. Used by
    # the sandbox to pre-coerce these to numeric before exposing the
    # DataFrame to AI-generated code, preventing TypeError on
    # arithmetic when the raw CSV inferred string dtypes.
    chat_studio_sample_columns_original: Optional[List[str]] = None
    # Names of the same columns AFTER the apply-step rename to
    # `{intensity|concentration}[s1..sN]`. Used when the Custom Viz
    # tab pre-coerces `chat_studio_df` (the renamed copy).
    chat_studio_sample_columns_renamed: Optional[List[str]] = None


# The 'ai_studio' stage of _STAGES (see _Stage).
AI_STUDIO_KEYS = dict(
    fields=frozenset({
        'chat_transform_session', 'chat_viz_session',
        'chat_data_questions_session', 'chat_mode', 'chat_transform_result',
        'chat_transform_report', 'chat_transform_refused_metadata',
        'chat_transform_kept_columns',
        'chat_viz_figures',
        'chat_studio_df', 'chat_studio_df_original', 'chat_studio_experiment',
        'chat_studio_pipeline_source', 'chat_studio_detected_format',
        'chat_studio_sample_columns_original',
        'chat_studio_sample_columns_renamed',
    }),
    widgets=frozenset({
        'chat_input', 'chat_apply_transform', 'chat_download_transform',
        'chat_download_removed_lipids', 'chat_keep_metadata_columns',
        'chat_input_questions', 'chat_input_transform',
        'chat_input_lc_questions', 'chat_retry_lc', 'chat_input_viz',
        'ai_studio_file_uploader', 'ai_studio_import_pipeline',
        'ai_studio_pipeline_choice', 'studio_sample_family',
        'studio_sample_cols_select', 'studio_conditions_editor',
        'studio_exclude_samples', 'studio_sample_cols_radio',
        'studio_bqc_radio', 'studio_bqc_label_radio',
        'studio_confirm_checkbox',
        'chat_retry_dq', 'chat_retry_viz', 'chat_retry_transform',
    }),
    # The data the tabs work on and what they derived from it, cleared
    # with it (AQ13: these used to survive a reset that cleared
    # chat_studio_df).
    plain=frozenset({
        'studio_conditions', 'studio_applied_snapshot',
        'chat_studio_file_id', 'chat_studio_upload_kept',
        'chat_studio_bqc_label',
        'chat_questions_result_df', 'chat_transform_diagnostic',
        'chat_viz_validation', 'chat_turn_ids',
        'chat_pending_prompt_dq', 'chat_pending_prompt_transform',
        'chat_pending_prompt_viz', 'chat_pending_prompt_lc',
        'chat_lc_retry_input',
        'chat_dq_last_error', 'chat_dq_failed_input',
        'chat_viz_last_error', 'chat_viz_failed_input',
        'chat_transform_last_error', 'chat_transform_failed_input',
    }),
    # Per-turn feedback widgets (_interaction_log.py).
    prefixes=('fb_', 'note_', 'send_note_'),
)


# Sidebar widget keys that need a snapshot when the user navigates to a
# different page (main app ↔ AI Studio). Without this, Streamlit garbage-
# collects keys whose widgets did not render in the current run, and the
# user loses their experiment configuration on every round-trip.
_MAIN_APP_NAV_WIDGET_KEYS = (
    'data_format_select', 'n_conditions',
    'bqc_radio', 'bqc_label_radio', 'confirm_checkbox',
    'grouping_radio', 'manual_sample_override',
)
_MAIN_APP_NAV_WIDGET_PREFIXES = (
    'cond_name_', 'n_samples_', 'select_',
)
# The Assign Conditions table is not listed: Streamlit refuses to restore a
# data_editor through session state, so its values are kept in the plain
# 'studio_conditions' key instead, which navigation does not drop.
_AI_STUDIO_NAV_WIDGET_KEYS = (
    'studio_sample_cols_radio', 'studio_sample_family',
    'studio_sample_cols_select',
    'studio_bqc_radio', 'studio_bqc_label_radio',
    'studio_exclude_samples', 'studio_confirm_checkbox',
)
_AI_STUDIO_NAV_WIDGET_PREFIXES = ()
_NAV_PRESERVE_PREFIX = '_nav_preserved_'


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


class AIStudioAdapter:
    """The AI Studio's adapter calls; StreamlitAdapter inherits them."""

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

    @classmethod
    def preserve_main_app_widgets(cls) -> None:
        """Snapshot main-app sidebar widget keys before the user navigates
        to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        cls._snapshot_widgets(
            _MAIN_APP_NAV_WIDGET_KEYS, _MAIN_APP_NAV_WIDGET_PREFIXES,
        )
        # The file uploader cannot be snapshotted; it comes back empty.
        if st.session_state.get('raw_df') is not None:
            st.session_state['upload_kept_across_pages'] = True

    @classmethod
    def preserve_ai_studio_widgets(cls) -> None:
        """Snapshot AI Studio sidebar widget keys before the user
        navigates to a different page. Restored on re-entry by
        ``restore_navigation_widgets()``."""
        cls._snapshot_widgets(
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

    @classmethod
    def ai_usage_refusal(cls, runs_code: bool = True) -> Optional[str]:
        """Why this visitor may not start an AI turn now, or None."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        return UsageLimitService.refusal(
            cls._ai_log_session_id(),
            cls._ai_client_id(),
            runs_code=runs_code,
        )

    @classmethod
    def start_ai_turn(cls, runs_code: bool = True) -> Optional[str]:
        """Count one AI turn against every limit; the refusal, or None."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        return UsageLimitService.start_turn(
            cls._ai_log_session_id(),
            cls._ai_client_id(),
            runs_code=runs_code,
        )

    @classmethod
    def record_ai_execution(cls) -> None:
        """Count one execution of AI-generated code for this session."""
        from ..services.ai_chat.usage_limits import UsageLimitService
        UsageLimitService.record_execution(
            cls._ai_log_session_id(),
        )

    @classmethod
    def log_ai_turn(
        cls,
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
            session_id=cls._ai_log_session_id(),
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

    @classmethod
    def log_ai_feedback(cls, *, turn_id: str, verdict: str, note: str = "") -> None:
        """Log a 👍/👎 signal for a previously-logged turn."""
        from ..services.ai_chat.interaction_logger import InteractionLoggingService
        InteractionLoggingService.log_feedback(
            turn_id=turn_id,
            session_id=cls._ai_log_session_id(),
            verdict=verdict,
            note=note,
        )
