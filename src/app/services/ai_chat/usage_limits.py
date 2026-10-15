"""
AI Studio usage limits: per browser session, per client, and for the app.

Every AI Studio turn calls the Anthropic API (up to four round trips with
tools) on the deployment's key, and may run generated code. The counters
live here, in the app process, not in Streamlit session state, so nothing a
visitor does in the browser ("Reset Chat", opening another tab, starting a
new conversation, reloading the page) hands them a fresh allowance:

- **session** — a conversation-length cap across all four tabs; a reload
  starts a new session;
- **client** — a daily cap per client address, which survives reloads (the
  adapter supplies the address; see ``StreamlitAdapter._ai_client_id``);
- **global** — a daily cap for every visitor together.

All counts are per UTC day and held in memory. That makes them a guard
against one visitor running up the bill, not a spending limit: they start
again when the process restarts and are not shared between tasks. The
spend limit set on the Anthropic workspace is the real backstop.

Pure Python — no Streamlit imports.
"""
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

SESSION_MESSAGES = 50
SESSION_EXECUTIONS = 20
CLIENT_DAILY_MESSAGES = 150
GLOBAL_DAILY_MESSAGES = 1000

_lock = threading.Lock()
_day: Optional[str] = None
_counts: Counter = Counter()  # (scope, key, kind) -> count, for _day only


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _roll_over() -> None:
    """Start a new day's counts. Caller holds ``_lock``."""
    global _day
    today = _today()
    if today != _day:
        _day = today
        _counts.clear()


def _refusal(
    session_id: str, client_id: Optional[str], runs_code: bool,
) -> Optional[str]:
    """Why a turn may not start now, or None. Caller holds ``_lock``."""
    if _counts['global', '', 'messages'] >= GLOBAL_DAILY_MESSAGES:
        return (
            "The AI Studio has reached today's usage limit for all users. "
            "It resets at midnight UTC."
        )
    if (client_id is not None
            and _counts['client', client_id, 'messages']
            >= CLIENT_DAILY_MESSAGES):
        return (
            f"You have reached today's limit of {CLIENT_DAILY_MESSAGES} AI "
            f"messages. It resets at midnight UTC."
        )
    if _counts['session', session_id, 'messages'] >= SESSION_MESSAGES:
        return (
            f"This session has used its {SESSION_MESSAGES} AI messages. "
            f"Reload the page to start a new session."
        )
    if (runs_code
            and _counts['session', session_id, 'executions']
            >= SESSION_EXECUTIONS):
        return (
            f"This session has run generated code {SESSION_EXECUTIONS} "
            f"times, its limit. Reload the page to start a new session."
        )
    return None


class UsageLimitService:
    """Process-wide AI Studio usage counters. Static methods only."""

    @staticmethod
    def refusal(
        session_id: str, client_id: Optional[str], runs_code: bool = True,
    ) -> Optional[str]:
        """Why a turn may not start now, or None. Counts nothing.

        Args:
            session_id: Anonymous id of the browser session.
            client_id: The client's address, or None when unknown (then
                only the session and global limits apply).
            runs_code: Whether the tab executes generated code, so the
                session's execution limit applies too.
        """
        with _lock:
            _roll_over()
            return _refusal(session_id, client_id, runs_code)

    @staticmethod
    def start_turn(
        session_id: str, client_id: Optional[str], runs_code: bool = True,
    ) -> Optional[str]:
        """Count one message if every limit allows it.

        Check and count happen under one lock, so simultaneous turns cannot
        all slip under a limit. A turn is counted when it starts, failed or
        not: a failed turn may still have cost tokens.

        Returns:
            None if the turn may go ahead (and was counted), else the
            refusal to show the user.
        """
        with _lock:
            _roll_over()
            reason = _refusal(session_id, client_id, runs_code)
            if reason is None:
                _counts['global', '', 'messages'] += 1
                if client_id is not None:
                    _counts['client', client_id, 'messages'] += 1
                _counts['session', session_id, 'messages'] += 1
            return reason

    @staticmethod
    def record_execution(session_id: str) -> None:
        """Count one execution of generated code for the session."""
        with _lock:
            _roll_over()
            _counts['session', session_id, 'executions'] += 1
