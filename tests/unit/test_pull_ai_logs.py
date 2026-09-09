"""Unit tests for scripts/pull_ai_logs.py argument handling (no AWS calls)."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

import pull_ai_logs


class TestSinceToEpochMs:
    def test_relative_units(self):
        now = datetime.now(timezone.utc).timestamp() * 1000
        assert abs(pull_ai_logs.since_to_epoch_ms("2h") - (now - 2 * 3600_000)) < 5_000
        assert abs(pull_ai_logs.since_to_epoch_ms("30m") - (now - 30 * 60_000)) < 5_000

    def test_iso_date_is_utc(self):
        assert pull_ai_logs.since_to_epoch_ms("2026-09-01") == int(
            datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000
        )


class TestLogGroupIsNotBakedIn:
    def test_missing_log_group_is_an_error(self, monkeypatch):
        monkeypatch.delenv("AI_STUDIO_LOG_GROUP", raising=False)
        with patch("sys.argv", ["pull_ai_logs.py"]), \
             patch.object(pull_ai_logs, "fetch_messages") as fetch:
            with pytest.raises(SystemExit):
                pull_ai_logs.main()
        fetch.assert_not_called()

    def test_env_var_supplies_log_group(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AI_STUDIO_LOG_GROUP", "/ecs/some-task")
        out = tmp_path / "o.jsonl"
        with patch("sys.argv", ["pull_ai_logs.py", "--out", str(out)]), \
             patch.object(pull_ai_logs, "fetch_messages", return_value=[]) as fetch:
            pull_ai_logs.main()
        assert fetch.call_args.args[0] == "/ecs/some-task"
        assert out.exists()
