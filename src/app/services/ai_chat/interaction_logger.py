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
#   3 — `figure` (trace types, names, point counts, titles) replaces
#       `figure_json`: the plotted values are the user's data
#   4 — adds `unverified_numbers`: numbers the answer states that the
#       turn's code never produced (Data Questions, Visualization)
SCHEMA_VERSION = 4

# Size guards. The awslogs driver splits a stdout line above 256 KB into
# several CloudWatch events, which would break the JSON (and the pull script
# then drops the turn), so every blob is capped to keep a whole record
# comfortably under that. Anything cut is named in the record's `truncated`.
HISTORY_MAX_MESSAGES = 20        # last N prior messages (10 turns)
HISTORY_MAX_CHARS = 1000         # per prior message
FIGURE_MAX_TRACES = 50           # traces listed in the figure summary
DATASET_MAX_COLUMNS = 60         # column names listed in the descriptor
TEXT_MAX_BYTES = 20_000          # question / answer / code / error / note,
                                 # as serialized (non-ASCII escapes to 6 bytes)
RECORD_MAX_BYTES = 200_000       # last resort: drop figure, then history

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


def _cap_text(record: dict, field: str) -> None:
    text = record.get(field)
    if not text:
        return
    size = len(json.dumps(text))
    if size > TEXT_MAX_BYTES:
        record[field] = text[:len(text) * TEXT_MAX_BYTES // size] + "…"
        record.setdefault("truncated", []).append(field)


def _fit(record: dict, fields) -> dict:
    """Cap ``fields``, then drop the figure and history if still too big."""
    for field in fields:
        _cap_text(record, field)
    for field in ("figure", "history"):
        if len(json.dumps(record, default=str)) <= RECORD_MAX_BYTES:
            break
        if record.get(field) is not None:
            record[field] = None
            record.setdefault("truncated", []).append(field)
    return record


def _emit(record: dict) -> None:
    try:
        _get_logger().info(json.dumps(record, default=str))
    except Exception:  # pragma: no cover - defensive; must never break a turn
        logging.getLogger(__name__).exception(
            "Failed to emit AI Studio interaction log"
        )


def _trace_points(trace) -> Optional[int]:
    """How many points a trace plots (its first data array's length)."""
    for attr in ("x", "y", "values", "z", "r", "lat"):
        value = getattr(trace, attr, None)
        if value is not None:
            try:
                return len(value)
            except TypeError:
                continue
    return None


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
        """Describe the dataset a turn ran against: its shape and column
        names (which can carry sample names), not the table's values.
        ``experiment`` (an ExperimentConfig) adds the condition →
        sample-count map when the tab has one.

        The rest of the turn record is not value-free: the answer, printed
        output and earlier messages can quote numbers from the data."""
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
    def summarize_figure(fig) -> Optional[dict]:
        """What a Plotly figure showed, without the plotted values: each
        trace's type, name and point count, and the titles. The values (and
        hover text) are the user's data, which the logs leave out."""
        if fig is None:
            return None
        try:
            layout = fig.layout
            traces = [
                {"type": t.type, "name": t.name, "points": _trace_points(t)}
                for t in fig.data
            ]
            return {
                "n_traces": len(traces),
                "traces": traces[:FIGURE_MAX_TRACES],
                "title": layout.title.text,
                "x_title": layout.xaxis.title.text,
                "y_title": layout.yaxis.title.text,
            }
        except Exception:
            return None

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
        figure: Optional[dict] = None,
        unverified_numbers=(),
    ) -> dict:
        """Build the turn record dict (pure — no I/O). Broken out so it can
        be unit-tested without capturing stdout.

        ``history`` is the conversation *before* this turn (so the record is
        self-contained when read back); ``dataset`` comes from
        ``describe_dataset``; ``figure`` from ``summarize_figure``."""
        return _fit({
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
            "figure": figure,
            "unverified_numbers": list(unverified_numbers)[:50],
        }, ("question", "answer", "code", "error"))

    @staticmethod
    def build_feedback_record(
        *,
        turn_id: str,
        session_id: str,
        verdict: str,
        note: str = "",
    ) -> dict:
        """Build the feedback record dict (pure — no I/O)."""
        return _fit({
            "event": "ai_feedback",
            "schema": SCHEMA_VERSION,
            "ts": _now_iso(),
            "turn_id": turn_id,
            "session_id": session_id,
            "verdict": verdict,
            "note": note,
        }, ("note",))

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
        figure: Optional[dict] = None,
        unverified_numbers=(),
    ) -> None:
        """Emit one turn record (question, answer, tab, code, outcome, plus
        the model, dataset descriptor, prior conversation and a summary of
        the figure)."""
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
            figure=figure,
            unverified_numbers=unverified_numbers,
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
