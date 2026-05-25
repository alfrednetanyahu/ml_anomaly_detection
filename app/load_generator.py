#!/usr/bin/env python3
"""
load_generator.py – Simple load generator for thesis experiments.

Usage:
  # Normal load
  python load_generator.py --host http://localhost:8000 --mode normal --duration 300

  # Anomaly: high error rate (Scenario 1)
  python load_generator.py --host http://localhost:8000 --mode errors --duration 120

  # Anomaly: high latency (Scenario 2)
  python load_generator.py --host http://localhost:8000 --mode slow --duration 120

  # Anomaly: log-only error burst (Scenario 3)
  python load_generator.py --host http://localhost:8000 --mode burst --duration 120

THESIS NOTE: Replace with Locust for more sophisticated load profiles.
Record start/end times and save them to ml/data_ingest/incidents.json
as ground-truth labels for the evaluation script.
"""

import argparse
import json
import pathlib
import random
import sys
import time
from datetime import datetime, timezone

import requests


def generate_load(host: str, mode: str, duration: int, rps: float) -> dict:
    """Send requests to host for duration seconds at rps, return ok/error counts."""
    end_time = time.time() + duration
    interval = 1.0 / rps
    counts = {"ok": 0, "error": 0}

    print(f"[{datetime.now().isoformat()}] Starting '{mode}' load for {duration}s at {rps} rps")

    while time.time() < end_time:
        t0 = time.time()
        try:
            if mode == "normal":
                endpoint = random.choices(
                    ["/", "/work", "/health"], weights=[1, 8, 1]
                )[0]
                r = requests.get(f"{host}{endpoint}", timeout=10)
            elif mode == "errors":
                r = requests.get(f"{host}/error?error_rate=0.8", timeout=10)
            elif mode == "slow":
                delay = random.uniform(3.0, 7.0)
                r = requests.get(f"{host}/slow?delay={delay}", timeout=20)
            elif mode == "burst":
                # Scenario 3: log-only burst — hits /log-burst which emits error
                # log entries without returning HTTP 5xx, so Prometheus metrics
                # stay flat while Loki sees a keyword spike.
                r = requests.get(f"{host}/log-burst?count=10", timeout=10)
            else:
                print(f"Unknown mode: {mode}", file=sys.stderr)
                sys.exit(1)

            if r.status_code >= 500:
                counts["error"] += 1
            else:
                counts["ok"] += 1
        except requests.exceptions.RequestException as e:
            counts["error"] += 1
            print(f"Request failed: {e}", file=sys.stderr)

        elapsed = time.time() - t0
        time.sleep(max(0, interval - elapsed))

    print(f"[{datetime.now().isoformat()}] Done. ok={counts['ok']} errors={counts['error']}")
    return counts


def main():
    """Parse args, run load, optionally record the incident window."""
    parser = argparse.ArgumentParser(description="Thesis load generator")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["normal", "errors", "slow", "burst"], default="normal")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--rps", type=float, default=5.0, help="Requests per second")
    parser.add_argument("--record-incident", action="store_true",
                        help="Append incident window to ml/data_ingest/incidents.json")
    args = parser.parse_args()

    start_ts = datetime.now(timezone.utc).isoformat()
    counts = generate_load(args.host, args.mode, args.duration, args.rps)
    end_ts = datetime.now(timezone.utc).isoformat()

    if args.record_incident and args.mode != "normal":
        incident = {
            "start": start_ts,
            "end": end_ts,
            "type": args.mode,
            "description": f"Injected {args.mode} anomaly via load generator",
            "requests": counts,
        }
        path = pathlib.Path(__file__).parent.parent / "ml" / "data_ingest" / "incidents.json"
        content = path.read_text().strip() if path.exists() else ""
        existing = json.loads(content) if content else []
        existing.append(incident)
        path.write_text(json.dumps(existing, indent=2))
        print(f"Incident recorded to {path}")


if __name__ == "__main__":
    main()
