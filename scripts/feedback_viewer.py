"""
Local viewer for AI Studio user feedback.

    python scripts/pull_ai_logs.py            # fetch records from CloudWatch
    streamlit run scripts/feedback_viewer.py  # browse them

Shows every logged turn with its thumbs verdict and comments, and for the
selected turn the full context: the conversation leading up to it, the
question and answer, the generated code, the figure (from schema 3 only a
summary of it: the plotted values are not logged), and the dataset it ran
against. Runs entirely on your machine; nothing here is deployed.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_feedback_records import conversation_for, join_feedback, parse_records  # noqa: E402

VERDICT_ICON = {"up": "👍", "down": "👎", "none": "·"}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float):
    with open(path) as fh:
        return join_feedback(parse_records(fh))


def _fmt_ts(ts: str) -> str:
    return (ts or "")[:16].replace("T", " ")


def _sidebar():
    st.sidebar.header("AI Studio feedback")
    path = st.sidebar.text_input("Log file", "ai_studio_logs.jsonl")
    st.sidebar.caption("Refresh with `python scripts/pull_ai_logs.py`")
    return path


def _filters(turns):
    tabs = sorted({t.record.get("tab", "") for t in turns})
    st.sidebar.subheader("Filter")
    sel_tabs = st.sidebar.multiselect("Tab", tabs, default=tabs)
    sel_verdicts = st.sidebar.multiselect(
        "Verdict", ["up", "down", "none"], default=["up", "down", "none"],
        format_func=lambda v: f"{VERDICT_ICON[v]} {v}",
    )
    only_comments = st.sidebar.checkbox("Only with comments")
    only_errors = st.sidebar.checkbox("Only errored turns")
    query = st.sidebar.text_input("Search question / answer / comments")

    q = query.lower().strip()
    out = []
    for t in turns:
        r = t.record
        if r.get("tab") not in sel_tabs or t.verdict not in sel_verdicts:
            continue
        if only_comments and not t.notes:
            continue
        if only_errors and not r.get("error"):
            continue
        if q and q not in " ".join(
            [r.get("question", ""), r.get("answer", ""), *t.notes]
        ).lower():
            continue
        out.append(t)
    return out


def _metrics(turns):
    cols = st.columns(5)
    cols[0].metric("Turns", len(turns))
    cols[1].metric("👍", sum(t.verdict == "up" for t in turns))
    cols[2].metric("👎", sum(t.verdict == "down" for t in turns))
    cols[3].metric("Comments", sum(len(t.notes) for t in turns))
    cols[4].metric("Errors", sum(bool(t.record.get("error")) for t in turns))


def _table(turns):
    """Newest first; returns the selected Turn or None."""
    rows = [{
        "when": _fmt_ts(t.record.get("ts")),
        "tab": t.record.get("tab"),
        "verdict": VERDICT_ICON[t.verdict],
        "comment": " | ".join(t.notes),
        "question": t.record.get("question", ""),
        "error": "⚠️" if t.record.get("error") else "",
    } for t in reversed(turns)]
    event = st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={"question": st.column_config.TextColumn(width="large")},
    )
    sel = event.selection.rows
    return list(reversed(turns))[sel[0]] if sel else None


def _figure_summary(fig: dict) -> str:
    """One line for a schema-3 figure summary (no values were logged)."""
    traces = []
    for t in fig.get("traces", []):
        label = t.get("type") or "trace"
        if t.get("name"):
            label += f" '{t['name']}'"
        if t.get("points") is not None:
            label += f" ({t['points']} points)"
        traces.append(label)
    hidden = fig.get("n_traces", len(traces)) - len(traces)
    if hidden > 0:
        traces.append(f"{hidden} more")
    titles = ", ".join(
        f"{k}: {fig[key]}" for k, key in
        (("title", "title"), ("x", "x_title"), ("y", "y_title"))
        if fig.get(key)
    )
    return (f"Figure (values not logged) — {titles + '; ' if titles else ''}"
            f"{fig.get('n_traces', len(traces))} traces: {', '.join(traces)}")


def _detail(turn, all_turns):
    r = turn.record
    st.divider()
    head = st.columns([3, 1])
    head[0].subheader(f"{VERDICT_ICON[turn.verdict]} {r.get('question', '')}")
    head[1].caption(
        f"{_fmt_ts(r.get('ts'))} · {r.get('tab')} · "
        f"{r.get('model') or 'model n/a'}\n\n"
        f"session {r.get('session_id', '')[:8]} · turn {turn.turn_id[:8]}"
    )

    if turn.feedback:
        st.markdown("**Feedback**")
        for fb in turn.feedback:
            note = f" — {fb['note']}" if fb.get("note") else ""
            st.markdown(
                f"- {_fmt_ts(fb.get('ts'))} "
                f"{VERDICT_ICON.get(fb.get('verdict'), '·')}{note}"
            )

    if r.get("error"):
        st.error(r["error"])

    left, right = st.columns([3, 2])
    with left:
        history = conversation_for(turn, all_turns)
        if history:
            with st.expander(f"Earlier conversation ({len(history)} messages)"):
                for m in history:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
        with st.chat_message("user"):
            st.markdown(r.get("question", ""))
        with st.chat_message("assistant"):
            st.markdown(r.get("answer") or "_(empty answer)_")
            if r.get("figure_json"):  # schema 2 logged the whole figure
                st.plotly_chart(pio.from_json(r["figure_json"]),
                                width="stretch")
            elif r.get("figure"):
                st.caption(_figure_summary(r["figure"]))
            elif r.get("produced_figure"):
                st.caption(
                    "A figure was produced but was not logged "
                    + ("(schema-1 record, before figure logging)."
                       if r.get("schema", 1) < 2 else "(too large)."))
            if r.get("code"):
                with st.expander("Generated code"):
                    st.code(r["code"], language="python")
    with right:
        ds = r.get("dataset")
        if ds:
            st.markdown("**Dataset**")
            st.write(f"{ds.get('source') or '?'} · {ds.get('format') or '?'} · "
                     f"{ds.get('rows')} rows × {ds.get('n_columns')} columns")
            if ds.get("conditions"):
                st.write("Conditions:", ds["conditions"])
            with st.expander("Columns"):
                st.write(ds.get("columns"))
        elif r.get("schema", 1) < 2:
            st.caption("No dataset context (schema-1 record).")
        flags = []
        if r.get("produced_table"):
            flags.append("table")
        if r.get("produced_figure"):
            flags.append("figure")
        st.caption("Produced: " + (", ".join(flags) or "text only"))


def main():
    st.set_page_config(page_title="AI Studio feedback", layout="wide",
                       page_icon="💬")
    path = _sidebar()
    p = Path(path)
    if not p.exists():
        st.info(f"No log file at `{path}`. Run `python scripts/pull_ai_logs.py` first.")
        return
    turns = _load(str(p), p.stat().st_mtime)
    shown = _filters(turns)
    _metrics(shown)
    if not shown:
        st.info("No turns match the current filters.")
        return
    selected = _table(shown)
    if selected is None:
        st.caption("Select a row to see the full context.")
        return
    _detail(selected, turns)


if __name__ == "__main__":
    main()
