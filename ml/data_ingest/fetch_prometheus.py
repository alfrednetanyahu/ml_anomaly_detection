#!/usr/bin/env python3
"""
fetch_prometheus.py – Export Prometheus metrics to CSV/Parquet for ML pipeline.

Usage:
  python fetch_prometheus.py \
    --host http://localhost:9090 \
    --start 2024-06-01T00:00:00Z \
    --end   2024-06-02T00:00:00Z \
    --step  60 \
    --out   ../data/metrics.parquet
"""

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

# ── Metrics to export ─────────────────────────────────────────────────────────
# key → Prometheus query string
QUERIES: dict[str, str] = {
    # System
    "cpu_busy": "1 - avg(rate(node_cpu_seconds_total{mode='idle'}[5m]))",
    "mem_used_ratio": "1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
    "disk_io_read_bps": "rate(node_disk_read_bytes_total[5m])",
    "disk_io_write_bps": "rate(node_disk_written_bytes_total[5m])",
    "net_rx_bps": "rate(node_network_receive_bytes_total{device='ens5'}[5m])",
    "net_tx_bps": "rate(node_network_transmit_bytes_total{device='ens5'}[5m])",
    # App
    "request_rate": "sum(rate(app_http_requests_total[5m]))",
    "error_rate": "sum(rate(app_http_requests_total{status=~'5..'}[5m]))",
    "error_ratio": (
        "sum(rate(app_http_requests_total{status=~'5..'}[5m])) / "
        "(sum(rate(app_http_requests_total[5m])) + 0.001)"
    ),
    "p50_latency": (
        "histogram_quantile(0.50, sum by(le)(rate(app_http_request_duration_seconds_bucket[5m])))"
    ),
    "p95_latency": (
        "histogram_quantile(0.95, sum by(le)(rate(app_http_request_duration_seconds_bucket[5m])))"
    ),
    "p99_latency": (
        "histogram_quantile(0.99, sum by(le)(rate(app_http_request_duration_seconds_bucket[5m])))"
    ),
}


def fetch_range(host: str, query: str, start: datetime, end: datetime, step: int) -> pd.Series:
    """Return a time-indexed Series for a single PromQL query."""
    url = f"{host}/api/v1/query_range"
    params = {
        "query": query,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": step,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success":
        raise RuntimeError(f"Prometheus error: {data}")

    results = data["data"]["result"]
    if not results:
        return pd.Series(dtype=float, name=query)

    # Take first result (scalar-like queries return one series)
    values = results[0]["values"]
    idx = pd.to_datetime([v[0] for v in values], unit="s", utc=True)
    vals = pd.array([float(v[1]) for v in values])
    return pd.Series(vals, index=idx)


def main():
    parser = argparse.ArgumentParser(description="Export Prometheus metrics to CSV/Parquet")
    parser.add_argument("--host", default="http://localhost:9090")
    parser.add_argument("--start", required=True, help="ISO8601 start time, e.g. 2024-06-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO8601 end time")
    parser.add_argument("--step", type=int, default=60, help="Step in seconds (default: 60)")
    parser.add_argument("--out", default="../data/metrics.parquet")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))

    frames = {}
    for name, query in QUERIES.items():
        print(f"  Fetching: {name} ...", end=" ", flush=True)
        try:
            s = fetch_range(args.host, query, start, end, args.step)
            frames[name] = s
            print(f"{len(s)} points")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    if not frames:
        print("No data fetched. Check Prometheus connection and time range.")
        sys.exit(1)

    df = pd.DataFrame(frames)
    df.index.name = "timestamp"
    df = df.sort_index().ffill()

    out = args.out
    if out.endswith(".parquet"):
        df.to_parquet(out, engine="pyarrow")
    else:
        df.to_csv(out)

    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {out}")


if __name__ == "__main__":
    main()
