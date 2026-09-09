"""
Shared helpers for AI Studio interaction logging + feedback UI.

`record_turn` logs a completed turn (question, answer, tab, code, outcome)
and remembers its turn_id against the assistant message so `render_feedback`
can draw a 👍/👎 widget next to that message and log the user's verdict.

Kept in one place so each of the four tab UIs wires it in with one line.
"""
import streamlit as st

from app.adapters.streamlit_adapter import StreamlitAdapter

# session_state keys owned by this module
_TURN_IDS = "chat_turn_ids"          # {assistant msg timestamp: turn_id}
_FEEDBACK_LOGGED = "ai_feedback_logged"  # {turn_id: "up" | "down"}


def record_turn(
    tab: str,
    question: str,
    answer: str,
    *,
    assistant_msg=None,
    code=None,
    produced_table: bool = False,
    produced_figure: bool = False,
    error=None,
    history=None,
    df=None,
    experiment=None,
    figure=None,
) -> None:
    """Log one turn. If ``assistant_msg`` is given, map its timestamp to the
    turn_id so the feedback widget can attach to it. ``history`` is the
    session's messages before this turn; ``df`` / ``experiment`` / ``figure``
    give the record its data and figure context. Never raises — logging
    must not break a chat turn."""
    try:
        turn_id = StreamlitAdapter.log_ai_turn(
            tab=tab,
            question=question,
            answer=answer,
            code=code,
            produced_table=produced_table,
            produced_figure=produced_figure,
            error=error,
            history=history,
            df=df,
            experiment=experiment,
            figure=figure,
        )
        if assistant_msg is not None:
            store = st.session_state.setdefault(_TURN_IDS, {})
            store[assistant_msg.timestamp] = turn_id
    except Exception:
        pass


def render_feedback(msg) -> None:
    """Draw a 👍/👎 widget for an assistant message that has a logged
    turn_id, and log the verdict when the user picks (or changes) it."""
    if msg.role != "assistant":
        return
    turn_id = st.session_state.get(_TURN_IDS, {}).get(msg.timestamp)
    if turn_id is None:
        return

    # Thumbs: log only when the verdict changes, so reruns don't re-emit.
    selection = st.feedback("thumbs", key=f"fb_{turn_id}")
    if selection is not None:
        verdict = "up" if selection == 1 else "down"
        logged = st.session_state.setdefault(_FEEDBACK_LOGGED, {})
        if logged.get(turn_id) != verdict:
            logged[turn_id] = verdict
            try:
                StreamlitAdapter.log_ai_feedback(
                    turn_id=turn_id, verdict=verdict
                )
            except Exception:
                pass

    # Optional free-text comment, tucked behind a compact popover so it
    # doesn't clutter the transcript. A comment can stand alone or ride
    # along with whatever thumb is currently set.
    with st.popover("💬 Comment"):
        note = st.text_area(
            "Leave a comment about this answer",
            key=f"note_{turn_id}",
            placeholder="What was right or wrong? Anything worth flagging?",
        )
        if st.button("Send comment", key=f"send_note_{turn_id}"):
            text = (note or "").strip()
            if not text:
                st.warning("Type a comment first.")
            else:
                sel = st.session_state.get(f"fb_{turn_id}")
                verdict = (
                    "up" if sel == 1 else "down" if sel == 0 else "none"
                )
                try:
                    StreamlitAdapter.log_ai_feedback(
                        turn_id=turn_id, verdict=verdict, note=text,
                    )
                    st.toast("Comment saved. Thank you!")
                except Exception:
                    pass
