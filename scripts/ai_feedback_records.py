"""
Pure helpers for the AI Studio feedback viewer: decode pulled log lines,
join ``ai_feedback`` records to their ``ai_turn`` by ``turn_id``, and
rebuild the conversation that led up to a turn.

No Streamlit, no AWS — so it is unit-testable (tests/unit/test_ai_feedback_records.py).
"""
import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class Turn:
    """One ``ai_turn`` record plus every feedback event logged against it."""
    record: dict
    feedback: List[dict] = field(default_factory=list)

    @property
    def turn_id(self) -> str:
        return self.record["turn_id"]

    @property
    def verdict(self) -> str:
        """Thumb state after the latest feedback event: 'up', 'down' or
        'none'. Every feedback event carries the thumb as it stood when it
        was sent, so the last one is the current state."""
        if not self.feedback:
            return "none"
        return self.feedback[-1].get("verdict") or "none"

    @property
    def notes(self) -> List[str]:
        return [fb["note"] for fb in self.feedback if fb.get("note")]


def parse_records(lines: Iterable[str]) -> List[dict]:
    """JSON-decode log lines. Blank or non-JSON lines are skipped, and exact
    duplicate lines (overlapping pulls) are collapsed."""
    seen = set()
    out = []
    for line in lines:
        line = line.strip()
        if not line or line in seen:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or "event" not in rec:
            continue
        seen.add(line)
        out.append(rec)
    return out


def join_feedback(records: Iterable[dict]) -> List[Turn]:
    """Group records into Turns (chronological). Feedback whose turn is not
    in the pull (e.g. pulled with ``--since`` after the turn) is dropped."""
    turns = {}
    for rec in records:
        if rec.get("event") == "ai_turn":
            turns[rec["turn_id"]] = Turn(rec)
    for rec in records:
        if rec.get("event") == "ai_feedback" and rec.get("turn_id") in turns:
            turns[rec["turn_id"]].feedback.append(rec)
    for t in turns.values():
        t.feedback.sort(key=lambda r: r.get("ts", ""))
    return sorted(turns.values(), key=lambda t: t.record.get("ts", ""))


def conversation_for(turn: Turn, turns: List[Turn]) -> List[dict]:
    """Messages *before* ``turn`` as ``[{role, content}]``.

    Schema-2 records carry their own ``history``. Older records don't, so
    the conversation is rebuilt from earlier turns in the same session and
    tab — an approximation (it ignores chat resets) but the best available.
    """
    rec = turn.record
    if rec.get("history") is not None:
        return list(rec["history"])
    out = []
    for other in turns:
        o = other.record
        if (o.get("session_id") == rec.get("session_id")
                and o.get("tab") == rec.get("tab")
                and o.get("ts", "") < rec.get("ts", "")):
            out.append({"role": "user", "content": o.get("question", "")})
            out.append({"role": "assistant", "content": o.get("answer", "")})
    return out
