"""Unit tests for the feedback-viewer join helpers (scripts/ai_feedback_records.py)."""
import json

from ai_feedback_records import Turn, conversation_for, join_feedback, parse_records


def _turn(tid, ts, session="s", tab="visualization", **extra):
    rec = {"event": "ai_turn", "schema": 2, "ts": ts, "turn_id": tid,
           "session_id": session, "tab": tab, "question": f"q-{tid}",
           "answer": f"a-{tid}", "history": None}
    rec.update(extra)
    return rec


def _fb(tid, ts, verdict, note=""):
    return {"event": "ai_feedback", "ts": ts, "turn_id": tid,
            "session_id": "s", "verdict": verdict, "note": note}


class TestParseRecords:
    def test_skips_blank_and_non_json_and_dedupes(self):
        good = json.dumps(_turn("t1", "2026-09-01T00:00:00"))
        lines = ["", "not json", good, good, json.dumps({"no_event": 1})]
        recs = parse_records(lines)
        assert len(recs) == 1
        assert recs[0]["turn_id"] == "t1"


class TestJoinFeedback:
    def test_attaches_feedback_in_time_order_and_sorts_turns(self):
        recs = [
            _fb("t2", "2026-09-01T00:05:00", "up", "nice"),
            _turn("t2", "2026-09-01T00:02:00"),
            _turn("t1", "2026-09-01T00:01:00"),
            _fb("t2", "2026-09-01T00:03:00", "down"),
            _fb("orphan", "2026-09-01T00:04:00", "up"),
        ]
        turns = join_feedback(recs)
        assert [t.turn_id for t in turns] == ["t1", "t2"]
        t2 = turns[1]
        assert [fb["verdict"] for fb in t2.feedback] == ["down", "up"]
        assert t2.verdict == "up"          # latest event wins
        assert t2.notes == ["nice"]
        assert turns[0].verdict == "none"
        assert turns[0].notes == []

    def test_comment_without_thumb_is_verdict_none(self):
        turns = join_feedback([
            _turn("t1", "2026-09-01T00:01:00"),
            _fb("t1", "2026-09-01T00:02:00", "none", "just a comment"),
        ])
        assert turns[0].verdict == "none"
        assert turns[0].notes == ["just a comment"]


class TestConversationFor:
    def test_uses_embedded_history_when_present(self):
        hist = [{"role": "user", "content": "earlier"}]
        turns = join_feedback([
            _turn("t0", "2026-09-01T00:00:00"),
            _turn("t1", "2026-09-01T00:01:00", history=hist),
        ])
        assert conversation_for(turns[1], turns) == hist

    def test_rebuilds_from_same_session_and_tab_for_schema1(self):
        turns = join_feedback([
            _turn("t0", "2026-09-01T00:00:00"),
            _turn("other-tab", "2026-09-01T00:00:30", tab="lc_questions"),
            _turn("other-session", "2026-09-01T00:00:40", session="z"),
            _turn("t1", "2026-09-01T00:01:00"),
            _turn("later", "2026-09-01T00:02:00"),
        ])
        target = next(t for t in turns if t.turn_id == "t1")
        assert conversation_for(target, turns) == [
            {"role": "user", "content": "q-t0"},
            {"role": "assistant", "content": "a-t0"},
        ]
