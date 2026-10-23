"""
Renders scripts/feedback_viewer.py with AppTest against a small log file
mixing schema-1 and schema-2 records, so a change to the record shape or
the viewer surfaces here rather than the first time someone opens it.
"""
import json

import plotly.graph_objects as go
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

# Absolute: AppTest resolves a relative path against this file, not the cwd.
VIEWER = str(Path(__file__).parents[2] / 'scripts' / 'feedback_viewer.py')


@pytest.fixture
def log_file(tmp_path):
    fig = go.Figure(data=[go.Bar(x=["PC", "PE"], y=[3, 5])])
    records = [
        {"event": "ai_turn", "schema": 1, "ts": "2026-09-01T10:00:00+00:00",
         "turn_id": "t1", "session_id": "s1", "tab": "lc_questions",
         "question": "What is Generic?", "answer": "A plain CSV format.",
         "code": None, "produced_table": False, "produced_figure": False,
         "error": None},
        {"event": "ai_turn", "schema": 2, "ts": "2026-09-01T10:05:00+00:00",
         "turn_id": "t2", "session_id": "s1", "tab": "visualization",
         "model": "claude-x", "question": "Species per class",
         "answer": "Here is a bar chart.", "code": "fig = px.bar(df)",
         "produced_table": False, "produced_figure": True, "error": None,
         "dataset": {"source": "run.csv_10", "format": "Generic", "rows": 8,
                     "n_columns": 4, "columns": ["LipidMolec", "ClassKey", "s1", "s2"],
                     "conditions": {"WT": 1, "KO": 1}},
         "history": [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}],
         "figure_json": fig.to_json()},
        {"event": "ai_feedback", "ts": "2026-09-01T10:06:00+00:00",
         "turn_id": "t2", "session_id": "s1", "verdict": "down",
         "note": "Wrong axis label"},
    ]
    path = tmp_path / "logs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _run(path):
    at = AppTest.from_file(VIEWER, default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(path)).run()
    return at


def test_overview_renders_metrics_and_table(log_file):
    at = _run(log_file)
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Turns"] == "2"
    assert metrics["👎"] == "1"
    assert metrics["Comments"] == "1"
    assert len(at.dataframe) == 1
    assert "Wrong axis label" in at.dataframe[0].value["comment"].tolist()


def test_filter_only_comments(log_file):
    at = _run(log_file)
    at.sidebar.checkbox[0].check().run()   # "Only with comments"
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Turns"] == "1"


def test_detail_renders_conversation_figure_code_and_dataset(log_file):
    """AppTest cannot click a dataframe row, so render the detail pane
    directly for the schema-2 turn."""
    def script(path):
        import sys
        sys.path.insert(0, "scripts")
        import streamlit as st
        from ai_feedback_records import join_feedback, parse_records
        from feedback_viewer import _detail
        with open(path) as fh:
            turns = join_feedback(parse_records(fh))
        st.session_state["turns"] = turns
        _detail(turns[1], turns)

    at = AppTest.from_function(script, args=(str(log_file),), default_timeout=30)
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Wrong axis label" in text
    assert "Here is a bar chart." in text
    assert "hello" in text                       # earlier conversation
    assert "fig = px.bar(df)" in [c.value for c in at.code]
    assert "run.csv_10" in " ".join(m.value for m in at.markdown)
    assert len(at.get("plotly_chart")) == 1


def test_schema1_figure_turn_explains_missing_figure(log_file):
    def script(path):
        import sys
        sys.path.insert(0, "scripts")
        from ai_feedback_records import join_feedback, parse_records
        from feedback_viewer import _detail
        with open(path) as fh:
            turns = join_feedback(parse_records(fh))
        old = turns[0]
        old.record["produced_figure"] = True   # schema-1 turn that drew a figure
        _detail(old, turns)

    at = AppTest.from_function(script, args=(str(log_file),), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 0
    assert any("schema-1" in c.value for c in at.caption)


def test_schema3_figure_turn_shows_the_summary(log_file):
    """Schema 3 logs what the figure showed, not its values."""
    def script(path):
        import sys
        sys.path.insert(0, "scripts")
        from ai_feedback_records import join_feedback, parse_records
        from feedback_viewer import _detail
        with open(path) as fh:
            turns = join_feedback(parse_records(fh))
        turn = turns[1]
        turn.record["schema"] = 3
        del turn.record["figure_json"]
        turn.record["figure"] = {
            "n_traces": 1, "title": "Species per class",
            "traces": [{"type": "bar", "name": "WT", "points": 2}],
        }
        _detail(turn, turns)

    at = AppTest.from_function(script, args=(str(log_file),), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 0
    assert (
        "Figure (values not logged) — title: Species per class; "
        "1 traces: bar 'WT' (2 points)"
    ) in [c.value for c in at.caption]


def test_missing_file_shows_hint(tmp_path):
    at = _run(tmp_path / "nope.jsonl")
    assert not at.exception
    assert any("pull_ai_logs" in i.value for i in at.info)
