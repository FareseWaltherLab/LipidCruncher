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
#   1 — question / answer / code / outcome flags
#   2 — adds model, dataset descriptor, prior conversation, figure JSON
SCHEMA_VERSION = 2

# Size guards. The awslogs driver splits a stdout line above 256 KB into
# several CloudWatch events, which would break the JSON, so every optional
# blob is capped to keep a whole record comfortably under that.
HISTORY_MAX_MESSAGES = 20        # last N prior messages (10 turns)
HISTORY_MAX_CHARS = 1000         # per prior message
FIGURE_MAX_BYTES = 120_000       # serialized Plotly figure
DATASET_MAX_COLUMNS = 60         # column names listed in the descriptor

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
    def describe_dataset(
        df,
        *,
        source: Optional[str] = None,
        data_format: Optional[str] = None,
        experiment=None,
    ) -> Optional[dict]:
        """Describe the dataset a turn ran against — shape and column names,
        never the values. ``experiment`` (an ExperimentConfig) adds the
        condition → sample-count map when the tab has one."""
        if df is None:
            return None
        columns = [str(c) for c in df.columns]
        desc = {
            "source": source,
            "format": data_format,
            "rows": int(len(df)),
            "n_columns": len(columns),
            "columns": columns[:DATASET_MAX_COLUMNS],
        }
        if experiment is not None:
            desc["conditions"] = dict(zip(
                experiment.conditions_list,
                experiment.number_of_samples_list,
            ))
        return desc

    @staticmethod
    def serialize_figure(fig) -> Optional[str]:
        """Plotly figure → JSON string, or None if absent / too large to fit
        in one log event (the flag ``produced_figure`` still records that a
        figure existed)."""
        if fig is None:
            return None
        try:
            text = fig.to_json()
        except Exception:
            return None
        if len(text.encode("utf-8")) > FIGURE_MAX_BYTES:
            return None
        return text

    @staticmethod
    def _compact_history(history) -> Optional[list]:
        """Prior messages as ``[{role, content}]``, last N only, each
        truncated. Accepts objects with ``.role``/``.content`` or dicts."""
        if not history:
            return None
        out = []
        for m in list(history)[-HISTORY_MAX_MESSAGES:]:
            role = m["role"] if isinstance(m, dict) else m.role
            content = m["content"] if isinstance(m, dict) else m.content
            content = content or ""
            if len(content) > HISTORY_MAX_CHARS:
                content = content[:HISTORY_MAX_CHARS] + "…"
            out.append({"role": role, "content": content})
        return out

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
        model: Optional[str] = None,
        dataset: Optional[dict] = None,
        history=None,
        figure_json: Optional[str] = None,
    ) -> dict:
        """Build the turn record dict (pure — no I/O). Broken out so it can
        be unit-tested without capturing stdout.

        ``history`` is the conversation *before* this turn (so the record is
        self-contained when read back); ``dataset`` comes from
        ``describe_dataset``; ``figure_json`` from ``serialize_figure``."""
        return {
            "event": "ai_turn",
            "schema": SCHEMA_VERSION,
            "ts": _now_iso(),
            "turn_id": turn_id,
            "session_id": session_id,
            "tab": tab,
            "model": model,
            "question": question,
            "answer": answer,
            "code": code or None,
            "produced_table": produced_table,
            "produced_figure": produced_figure,
            "error": error,
            "dataset": dataset,
            "history": InteractionLoggingService._compact_history(history),
            "figure_json": figure_json,
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
        model: Optional[str] = None,
        dataset: Optional[dict] = None,
        history=None,
        figure_json: Optional[str] = None,
    ) -> None:
        """Emit one turn record (question, answer, tab, code, outcome, plus
        the model, dataset descriptor, prior conversation and figure)."""
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
            model=model,
            dataset=dataset,
            history=history,
            figure_json=figure_json,
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
