#!/usr/bin/env python3
"""
Pull AI Studio interaction records (``ai_turn`` / ``ai_feedback``) out of
CloudWatch into a JSONL file that scripts/feedback_viewer.py reads.

    python scripts/pull_ai_logs.py                    # everything → ai_studio_logs.jsonl
    python scripts/pull_ai_logs.py --since 7d         # last 7 days
    python scripts/pull_ai_logs.py --since 2026-09-01 --out /tmp/logs.jsonl

Uses the AWS CLI, signed in to the account that owns the integration
deployment. One record per line; the file is overwritten.
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_GROUP = "/ecs/lipidcruncher-integration-task"
REGION = "us-east-1"
FILTER = '{ $.event = "ai_*" }'


def since_to_epoch_ms(text: str) -> int:
    """'7d' / '12h' / '90m' relative, or an ISO date/datetime (UTC)."""
    m = re.fullmatch(r"(\d+)([dhm])", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "h": timedelta(hours=n),
                 "m": timedelta(minutes=n)}[unit]
        start = datetime.now(timezone.utc) - delta
    else:
        start = datetime.fromisoformat(text)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
    return int(start.timestamp() * 1000)


def fetch_messages(log_group: str, region: str, start_ms=None) -> list:
    cmd = [
        "aws", "logs", "filter-log-events",
        "--log-group-name", log_group,
        "--region", region,
        "--filter-pattern", FILTER,
        "--query", "events[].message",
        "--output", "json",
    ]
    if start_ms is not None:
        cmd += ["--start-time", str(start_ms)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"aws CLI failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout or "[]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--since", help="7d, 12h, 90m, or an ISO date (UTC)")
    ap.add_argument("--out", default="ai_studio_logs.jsonl")
    ap.add_argument("--log-group", default=LOG_GROUP)
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args()

    start_ms = since_to_epoch_ms(args.since) if args.since else None
    messages = fetch_messages(args.log_group, args.region, start_ms)

    counts = {"ai_turn": 0, "ai_feedback": 0, "skipped": 0}
    with Path(args.out).open("w") as fh:
        for msg in messages:
            try:
                rec = json.loads(msg)
            except json.JSONDecodeError:
                counts["skipped"] += 1
                continue
            counts[rec.get("event", "skipped")] = (
                counts.get(rec.get("event", "skipped"), 0) + 1
            )
            fh.write(json.dumps(rec) + "\n")

    print(f"Wrote {args.out}: {counts['ai_turn']} turns, "
          f"{counts['ai_feedback']} feedback events"
          + (f", {counts['skipped']} unparseable lines skipped"
             if counts["skipped"] else ""))


if __name__ == "__main__":
    main()
