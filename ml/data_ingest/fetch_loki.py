#!/usr/bin/env python3
"""
fetch_loki.py – Export Loki logs to JSON/CSV for ML pipeline.

Usage:
  python fetch_loki.py \
    --host http://localhost:3100 \
    --query '{service="sample-app"}' \
    --start 2024-06-01T00:00:00Z \
    --end   2024-06-02T00:00:00Z \
    --out   ../data/logs.csv

THESIS NOTE: Adjust --query to match your Loki label set.
For large time windows, use --batch-hours to paginate.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


def fetch_loki_range(
    host: str, query: str, start: datetime, end: datetime, limit: int = 5000
) -> list[dict]:
    """
    Fetch log entries from Loki using the query_range API.
    Returns a flat list of parsed log dicts.
    """
    url = f"{host}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": int(start.timestamp() * 1e9),  # nanoseconds
        "end": int(end.timestamp() * 1e9),
        "limit": limit,
        "direction": "forward",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success":
        raise RuntimeError(f"Loki error: {data}")

    records = []
    for stream in data["data"]["result"]:
        labels = stream["stream"]
        for ts_ns, line in stream["values"]:
            ts = datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc)
            # Try to parse structured JSON log line
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"message": line}

            records.append({
                "timestamp": ts,
                "raw": line,
                **labels,
                **parsed,
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="Export Loki logs to CSV")
    parser.add_argument("--host", default="http://localhost:3100")
    parser.add_argument("--query", default='{service="sample-app"}')
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch-hours", type=float, default=1.0,
                        help="Fetch in batches of N hours to avoid hitting limits")
    parser.add_argument("--out", default="../data/logs.csv")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    all_records: list[dict] = []
    batch_delta = timedelta(hours=args.batch_hours)
    cursor = start

    while cursor < end:
        batch_end = min(cursor + batch_delta, end)
        print(f"  Fetching {cursor.isoformat()} → {batch_end.isoformat()} ...", end=" ", flush=True)
        try:
            records = fetch_loki_range(args.host, args.query, cursor, batch_end, args.limit)
            all_records.extend(records)
            print(f"{len(records)} lines")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
        cursor = batch_end

    if not all_records:
        print("No logs fetched. Check Loki connection, query, and time range.")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    out = args.out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if out.endswith(".parquet"):
        df.to_parquet(out, engine="pyarrow")
    else:
        df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} log entries → {out}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
