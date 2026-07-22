"""
Structured logging of AI Studio interactions.

Emits one JSON record per turn (and per user feedback event) to a
dedicated logger that writes to stdout. In the hosted AWS deployment the
container's stdout is shipped to CloudWatch Logs by the ``awslogs`` driver,
so these records are retrievable with Logs Insights without any additional
infrastructure or persistent volume.

Pure — no Streamlit. The UI/adapter layer supplies the fields (including an
anonymous per-session id) and owns any session state. Emission failures are
swallowed: logging must never break a chat turn.
"""
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

# Records reference this schema version so the offline eval/analysis tooling
# can evolve the shape later without silently mis-parsing old logs.
SCHEMA_VERSION = 1

_LOGGER_NAME = "ai_studio.interactions"


def _get_logger() -> logging.Logger:
    """Return the interaction logger, configuring a stdout handler once.

    ``propagate = False`` keeps these JSON lines out of the app's ordinary
    (human-readable) log stream so the two don't interleave.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not getattr(logger, "_ai_studio_configured", False):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._ai_studio_configured = True
    return logger


def _emit(record: dict) -> None:
    try:
        _get_logger().info(json.dumps(record, default=str))
    except Exception:  # pragma: no cover - defensive; must never break a turn
        logging.getLogger(__name__).exception(
            "Failed to emit AI Studio interaction log"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InteractionLoggingService:
    """Emits AI Studio turns and feedback as structured JSON log lines."""

    @staticmethod
    def new_turn_id() -> str:
        """A fresh id linking a turn record to its later feedback record."""
        return uuid.uuid4().hex

    @staticmethod
    def build_turn_record(
        *,
        turn_id: str,
        session_id: str,
        tab: str,
        question: str,
        answer: str,
        code: Optional[str] = None,
        produced_table: bool = False,
        produced_figure: bool = False,
        error: Optional[str] = None,
    ) -> dict:
        """Build the turn record dict (pure — no I/O). Broken out so it can
        be unit-tested without capturing stdout."""
        return {
            "event": "ai_turn",
            "schema": SCHEMA_VERSION,
            "ts": _now_iso(),
            "turn_id": turn_id,
            "session_id": session_id,
            "tab": tab,
            "question": question,
            "answer": answer,
            "code": code or None,
            "produced_table": produced_table,
            "produced_figure": produced_figure,
            "error": error,
        }

    @staticmethod
    def build_feedback_record(
        *,
        turn_id: str,
        session_id: str,
        verdict: str,
        note: str = "",
    ) -> dict:
        """Build the feedback record dict (pure — no I/O)."""
        return {
            "event": "ai_feedback",
            "schema": SCHEMA_VERSION,
            "ts": _now_iso(),
            "turn_id": turn_id,
            "session_id": session_id,
            "verdict": verdict,
            "note": note,
        }

    @staticmethod
    def log_turn(
        *,
        turn_id: str,
        session_id: str,
        tab: str,
        question: str,
        answer: str,
        code: Optional[str] = None,
        produced_table: bool = False,
        produced_figure: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Emit one turn record (question, answer, tab, code, outcome)."""
        _emit(InteractionLoggingService.build_turn_record(
            turn_id=turn_id,
            session_id=session_id,
            tab=tab,
            question=question,
            answer=answer,
            code=code,
            produced_table=produced_table,
            produced_figure=produced_figure,
            error=error,
        ))

    @staticmethod
    def log_feedback(
        *,
        turn_id: str,
        session_id: str,
        verdict: str,
        note: str = "",
    ) -> None:
        """Emit one feedback record ('up' / 'down') for a prior turn."""
        _emit(InteractionLoggingService.build_feedback_record(
            turn_id=turn_id,
            session_id=session_id,
            verdict=verdict,
            note=note,
        ))
