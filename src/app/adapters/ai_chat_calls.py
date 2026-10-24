"""The AI Studio tabs' calls into the AI chat workflow, the usage limits
and the interaction log, part of the StreamlitAdapter facade.

The data and navigation state of the Studio page is AIStudioAdapter
(ai_studio.py); this is what a chat turn needs: the stream_ai_* wrappers
the tabs call, the shared answer cache of the LipidCruncher Questions tab,
the anonymous session id, the per-client address and the turn log.
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from ..models.experiment import ExperimentConfig


def _normalize_lc_question(question: str) -> str:
    """Canonical cache key for a LipidCruncher-questions prompt."""
    return " ".join(question.split()).lower()


@st.cache_resource
def _lc_answer_cache() -> Dict[str, str]:
    """Process-wide cache of canonical answers to LipidCruncher app questions.

    Keyed by normalized question text. Shared across sessions because these
    answers describe the application itself and do not depend on user data,
    so reuse is both correct and desirable.
    """
    return {}


class AIChatCalls:
    """The AI Studio tabs' AI calls; StreamlitAdapter inherits them."""

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
            bqc_label=st.session_state.get('chat_studio_bqc_label'),
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
            bqc_label=st.session_state.get('chat_studio_bqc_label'),
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
