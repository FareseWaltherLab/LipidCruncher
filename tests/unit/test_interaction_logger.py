"""Unit tests for the AI Studio interaction logger."""
import json
import logging

from app.services.ai_chat.interaction_logger import (
    InteractionLoggingService,
    SCHEMA_VERSION,
    _LOGGER_NAME,
)


class _CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _capture():
    handler = _CapturingHandler()
    logging.getLogger(_LOGGER_NAME).addHandler(handler)
    return handler


class TestBuildRecords:
    def test_turn_record_has_expected_fields(self):
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t1",
            session_id="s1",
            tab="data_questions",
            question="How many species?",
            answer="There are 24.",
            code="result_df = df",
            produced_table=True,
        )
        assert rec["event"] == "ai_turn"
        assert rec["schema"] == SCHEMA_VERSION
        assert rec["turn_id"] == "t1"
        assert rec["session_id"] == "s1"
        assert rec["tab"] == "data_questions"
        assert rec["question"] == "How many species?"
        assert rec["answer"] == "There are 24."
        assert rec["code"] == "result_df = df"
        assert rec["produced_table"] is True
        assert rec["produced_figure"] is False
        assert rec["error"] is None
        # ts present and the whole record is JSON-serializable
        assert rec["ts"]
        json.loads(json.dumps(rec))

    def test_empty_code_normalizes_to_none(self):
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t", session_id="s", tab="transform",
            question="q", answer="a", code="",
        )
        assert rec["code"] is None

    def test_feedback_record_has_expected_fields(self):
        rec = InteractionLoggingService.build_feedback_record(
            turn_id="t1", session_id="s1", verdict="up", note="great",
        )
        assert rec["event"] == "ai_feedback"
        assert rec["turn_id"] == "t1"
        assert rec["verdict"] == "up"
        assert rec["note"] == "great"
        json.loads(json.dumps(rec))


class TestEmission:
    def test_log_turn_emits_one_parseable_json_line(self):
        handler = _capture()
        try:
            InteractionLoggingService.log_turn(
                turn_id="t9", session_id="s9", tab="visualization",
                question="plot it", answer="here", produced_figure=True,
            )
        finally:
            logging.getLogger(_LOGGER_NAME).removeHandler(handler)

        assert len(handler.records) == 1
        payload = json.loads(handler.records[0])
        assert payload["event"] == "ai_turn"
        assert payload["turn_id"] == "t9"
        assert payload["produced_figure"] is True

    def test_log_feedback_emits_one_parseable_json_line(self):
        handler = _capture()
        try:
            InteractionLoggingService.log_feedback(
                turn_id="t9", session_id="s9", verdict="down",
            )
        finally:
            logging.getLogger(_LOGGER_NAME).removeHandler(handler)

        assert len(handler.records) == 1
        payload = json.loads(handler.records[0])
        assert payload["event"] == "ai_feedback"
        assert payload["verdict"] == "down"

    def test_new_turn_ids_are_unique(self):
        ids = {InteractionLoggingService.new_turn_id() for _ in range(100)}
        assert len(ids) == 100
