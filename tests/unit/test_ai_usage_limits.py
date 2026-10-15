"""AI Studio usage limits (S5).

The per-session message and execution caps lived in Streamlit session state:
"Reset Chat", starting any tab's conversation and reloading the page all set
them back to zero, and nothing limited a visitor across sessions or the app
as a whole. The counters now live in the app process."""
from collections import Counter
from types import SimpleNamespace

import pytest

from app.services.ai_chat import usage_limits
from app.services.ai_chat.usage_limits import UsageLimitService


@pytest.fixture(autouse=True)
def fresh_counts(monkeypatch):
    monkeypatch.setattr(usage_limits, "_counts", Counter())
    monkeypatch.setattr(usage_limits, "_day", None)
    monkeypatch.setattr(usage_limits, "_today", lambda: "2026-10-15")


def _use(n, session="s", client="1.2.3.4", runs_code=True):
    for _ in range(n):
        assert UsageLimitService.start_turn(session, client, runs_code) is None


class TestSessionLimit:
    def test_refused_after_the_session_cap(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 3)
        _use(3)
        reason = UsageLimitService.start_turn("s", "1.2.3.4")
        assert "This session has used its 3 AI messages" in reason
        assert UsageLimitService.refusal("s", "1.2.3.4") == reason

    def test_another_session_is_unaffected(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 3)
        _use(3)
        assert UsageLimitService.start_turn("other", "1.2.3.4") is None

    def test_refused_turns_are_not_counted(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 1)
        monkeypatch.setattr(usage_limits, "GLOBAL_DAILY_MESSAGES", 3)
        _use(1)
        for _ in range(5):
            assert UsageLimitService.start_turn("s", "1.2.3.4") is not None
        _use(1, session="b", client="5.6.7.8")
        _use(1, session="c", client="5.6.7.8")

    def test_execution_cap_applies_only_to_code_running_tabs(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "SESSION_EXECUTIONS", 2)
        for _ in range(2):
            UsageLimitService.record_execution("s")
        assert "run generated code 2 times" in UsageLimitService.start_turn(
            "s", "1.2.3.4", runs_code=True)
        assert UsageLimitService.start_turn(
            "s", "1.2.3.4", runs_code=False) is None


class TestClientLimit:
    def test_survives_a_new_session(self, monkeypatch):
        """A reload is a new Streamlit session from the same address."""
        monkeypatch.setattr(usage_limits, "CLIENT_DAILY_MESSAGES", 3)
        _use(2, session="first")
        _use(1, session="after-reload")
        reason = UsageLimitService.start_turn("third", "1.2.3.4")
        assert "today's limit of 3 AI messages" in reason
        assert UsageLimitService.start_turn("third", "5.6.7.8") is None

    def test_unknown_client_skips_only_the_client_limit(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "CLIENT_DAILY_MESSAGES", 1)
        monkeypatch.setattr(usage_limits, "SESSION_MESSAGES", 2)
        _use(2, client=None)
        assert "This session" in UsageLimitService.start_turn("s", None)


class TestGlobalLimit:
    def test_counts_every_client(self, monkeypatch):
        monkeypatch.setattr(usage_limits, "GLOBAL_DAILY_MESSAGES", 3)
        _use(1, session="a", client="1.1.1.1")
        _use(1, session="b", client="2.2.2.2")
        _use(1, session="c", client=None)
        reason = UsageLimitService.start_turn("d", "4.4.4.4")
        assert "usage limit for all users" in reason


def test_counts_start_again_the_next_utc_day(monkeypatch):
    monkeypatch.setattr(usage_limits, "GLOBAL_DAILY_MESSAGES", 1)
    _use(1)
    assert UsageLimitService.start_turn("s", "1.2.3.4") is not None
    monkeypatch.setattr(usage_limits, "_today", lambda: "2026-10-16")
    assert UsageLimitService.start_turn("s", "1.2.3.4") is None


class TestClientId:
    """Behind the ALB every connection comes from the balancer; it appends
    the address it saw to X-Forwarded-For, so only the last entry counts."""

    def _client_id(self, monkeypatch, headers, ip_address=None):
        import app.adapters.streamlit_adapter as adapter_module
        from app.adapters.streamlit_adapter import StreamlitAdapter
        monkeypatch.setattr(
            adapter_module.st, "context",
            SimpleNamespace(headers=headers, ip_address=ip_address),
        )
        return StreamlitAdapter._ai_client_id()

    def test_last_forwarded_entry_wins_over_a_forged_one(self, monkeypatch):
        assert self._client_id(
            monkeypatch,
            {"X-Forwarded-For": "6.6.6.6, 203.0.113.7"},
            ip_address="10.0.0.5",
        ) == "203.0.113.7"

    def test_single_entry(self, monkeypatch):
        assert self._client_id(
            monkeypatch, {"X-Forwarded-For": "203.0.113.7"},
            ip_address="10.0.0.5",
        ) == "203.0.113.7"

    def test_falls_back_to_the_connection_address(self, monkeypatch):
        assert self._client_id(
            monkeypatch, {}, ip_address="10.0.0.5") == "10.0.0.5"

    def test_none_locally(self, monkeypatch):
        assert self._client_id(monkeypatch, {}, ip_address=None) is None
