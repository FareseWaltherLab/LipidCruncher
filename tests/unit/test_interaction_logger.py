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


class TestSchema2Context:
    """Schema 2 adds the model, a dataset descriptor, the prior conversation
    and the figure, so a record can be reviewed on its own."""

    def test_schema_version_is_2(self):
        assert SCHEMA_VERSION == 2

    def test_turn_record_carries_model_dataset_history_figure(self):
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t", session_id="s", tab="visualization",
            question="q", answer="a",
            model="claude-x",
            dataset={"rows": 3},
            history=[{"role": "user", "content": "earlier"}],
            figure_json='{"data": []}',
        )
        assert rec["model"] == "claude-x"
        assert rec["dataset"] == {"rows": 3}
        assert rec["history"] == [{"role": "user", "content": "earlier"}]
        assert rec["figure_json"] == '{"data": []}'
        json.loads(json.dumps(rec))

    def test_context_fields_default_to_none(self):
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t", session_id="s", tab="lc_questions",
            question="q", answer="a",
        )
        assert rec["model"] is None
        assert rec["dataset"] is None
        assert rec["history"] is None
        assert rec["figure_json"] is None

    def test_history_accepts_message_objects_and_is_capped(self):
        from app.models.chat import ChatMessage
        from app.services.ai_chat.interaction_logger import (
            HISTORY_MAX_CHARS, HISTORY_MAX_MESSAGES,
        )
        long_text = "x" * (HISTORY_MAX_CHARS + 50)
        msgs = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant",
                        content=f"m{i}")
            for i in range(HISTORY_MAX_MESSAGES + 5)
        ] + [ChatMessage(role="user", content=long_text)]
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t", session_id="s", tab="data_questions",
            question="q", answer="a", history=msgs,
        )
        hist = rec["history"]
        # only the most recent messages survive, oldest dropped
        assert len(hist) == HISTORY_MAX_MESSAGES
        assert hist[0]["content"] == "m6"
        # over-long content is truncated with a marker
        assert hist[-1]["content"] == "x" * HISTORY_MAX_CHARS + "…"
        assert hist[-1]["role"] == "user"

    def test_empty_history_is_none(self):
        rec = InteractionLoggingService.build_turn_record(
            turn_id="t", session_id="s", tab="lc_questions",
            question="q", answer="a", history=(),
        )
        assert rec["history"] is None

    def test_describe_dataset_shape_columns_and_conditions(self):
        import pandas as pd
        from app.models.experiment import ExperimentConfig
        df = pd.DataFrame({"LipidMolec": ["a", "b"], "s1": [1, 2], "s2": [3, 4]})
        exp = ExperimentConfig(
            conditions_list=["WT", "KO"], number_of_samples_list=[1, 1],
        )
        desc = InteractionLoggingService.describe_dataset(
            df, source="run.csv_123", data_format="Generic", experiment=exp,
        )
        assert desc == {
            "source": "run.csv_123",
            "format": "Generic",
            "rows": 2,
            "n_columns": 3,
            "columns": ["LipidMolec", "s1", "s2"],
            "conditions": {"WT": 1, "KO": 1},
        }
        json.loads(json.dumps(desc))

    def test_describe_dataset_caps_column_names_and_handles_none(self):
        import pandas as pd
        from app.services.ai_chat.interaction_logger import DATASET_MAX_COLUMNS
        df = pd.DataFrame({f"c{i}": [0] for i in range(DATASET_MAX_COLUMNS + 10)})
        desc = InteractionLoggingService.describe_dataset(df)
        assert desc["n_columns"] == DATASET_MAX_COLUMNS + 10
        assert len(desc["columns"]) == DATASET_MAX_COLUMNS
        assert "conditions" not in desc
        assert InteractionLoggingService.describe_dataset(None) is None

    def test_serialize_figure_small_ok_large_dropped_none_passthrough(self):
        import plotly.graph_objects as go
        from app.services.ai_chat.interaction_logger import FIGURE_MAX_BYTES
        small = go.Figure(data=[go.Bar(x=["a", "b"], y=[1, 2])])
        text = InteractionLoggingService.serialize_figure(small)
        assert text and json.loads(text)["data"][0]["type"] == "bar"

        big = go.Figure(data=[go.Scatter(y=list(range(FIGURE_MAX_BYTES // 4)))])
        assert InteractionLoggingService.serialize_figure(big) is None

        assert InteractionLoggingService.serialize_figure(None) is None

    def test_log_turn_emits_context_fields(self):
        handler = _capture()
        try:
            InteractionLoggingService.log_turn(
                turn_id="t2", session_id="s2", tab="visualization",
                question="plot", answer="done", model="claude-x",
                dataset={"rows": 1}, history=[{"role": "user", "content": "hi"}],
                figure_json="{}",
            )
        finally:
            logging.getLogger(_LOGGER_NAME).removeHandler(handler)
        payload = json.loads(handler.records[0])
        assert payload["schema"] == 2
        assert payload["model"] == "claude-x"
        assert payload["dataset"] == {"rows": 1}
        assert payload["history"] == [{"role": "user", "content": "hi"}]
        assert payload["figure_json"] == "{}"


class TestRecordsFitOneLogEvent:
    """Regression (AQ12): only history was capped, so one "print the table"
    answer made a 359 KB record. awslogs splits lines over 256 KB, the
    pieces no longer parse, and the pull script silently drops the turn."""

    def _turn(self, **kw):
        args = dict(turn_id="t", session_id="s", tab="data_questions",
                    question="q", answer="a")
        args.update(kw)
        return InteractionLoggingService.build_turn_record(**args)

    def test_long_answer_and_code_are_capped_and_marked(self):
        rec = self._turn(answer="x" * 400_000, code="y" * 50_000)
        assert len(rec["answer"]) < 25_000
        assert len(rec["code"]) < 25_000
        assert rec["truncated"] == ["answer", "code"]
        assert len(json.dumps(rec)) < 256 * 1024

    def test_whole_record_stays_under_the_line_limit(self):
        history = [{"role": "user", "content": "é" * 1000}] * 20
        rec = self._turn(
            question="é" * 30_000, answer="é" * 30_000, code="é" * 30_000,
            error="é" * 30_000, history=history,
            figure_json='{"data": "' + "1" * 119_000 + '"}',
        )
        assert len(json.dumps(rec)) < 256 * 1024
        assert rec["figure_json"] is None
        assert "figure_json" in rec["truncated"]

    def test_small_records_are_untouched(self):
        rec = self._turn(answer="There are 24.")
        assert rec["answer"] == "There are 24."
        assert "truncated" not in rec

    def test_feedback_note_is_capped(self):
        rec = InteractionLoggingService.build_feedback_record(
            turn_id="t", session_id="s", verdict="down", note="n" * 400_000,
        )
        assert len(json.dumps(rec)) < 256 * 1024
        assert rec["truncated"] == ["note"]
