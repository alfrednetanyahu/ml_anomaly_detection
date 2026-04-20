#!/usr/bin/env python3
"""
evaluate_vs_rules.py – Compare rule-based alerts vs ML anomaly detection.

Inputs:
  - incidents.json          : Ground-truth incident windows (from load_generator.py)
  - metrics_anomalies.csv   : Output of metrics_isolation_forest.py
  - logs_anomalies.csv      : Output of logs_isolation_forest.py
  - rule_alerts.csv         : Prometheus/Alertmanager alerts exported to CSV
                              (or simulated from alert_rules.yml thresholds)

Output:
  - ml/output/evaluation_report.json
  - ml/output/evaluation_report.txt

Usage:
  python evaluate_vs_rules.py \
    --incidents ../../data_ingest/incidents.json \
    --metrics-anomalies ../output/metrics_anomalies.csv \
    --logs-anomalies    ../output/logs_anomalies.csv \
    --rule-alerts       ../output/rule_alerts.csv \
    --resolution        60

THESIS NOTE: This is the core evaluation chapter script. Extend by:
  - Computing AUC-ROC / average precision if you have soft scores
  - Adding per-incident breakdown (which model caught which incident)
  - Implementing combined model (OR / AND / score fusion)
  - Adding time-to-detection analysis per detection method
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ── Ground truth helpers ──────────────────────────────────────────────────────

def load_incidents(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_ground_truth_series(incidents: list[dict], index: pd.DatetimeIndex) -> pd.Series:
    """Return a binary Series (1 = anomaly) aligned to the given time index."""
    gt = pd.Series(0, index=index, name="ground_truth")
    for inc in incidents:
        start = pd.Timestamp(inc["start"]).tz_convert("UTC")
        end = pd.Timestamp(inc["end"]).tz_convert("UTC")
        gt.loc[(index >= start) & (index <= end)] = 1
    return gt


# ── Metrics loading ────────────────────────────────────────────────────────────

def load_anomaly_series(csv_path: str, label_col: str = "is_anomaly") -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[label_col].rename(Path(csv_path).stem)


def load_rule_alerts(csv_path: str, resolution_s: int) -> pd.Series:
    """
    Load Prometheus/Alertmanager alerts and convert to a binary time series.

    Expected CSV columns: starts_at, ends_at (ISO8601 strings).
    You can export these from Alertmanager or simulate them using
    threshold logic on the raw metrics CSV.

    THESIS NOTE: To generate this from raw metrics without Alertmanager,
    apply the alert rule thresholds manually:
        df["rule_alert"] = (df["cpu_busy"] > 0.80).astype(int)
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"  [WARN] Rule alerts file not found: {csv_path} – returning empty series.")
        return pd.Series(dtype=int, name="rule_based")

    df = pd.read_csv(csv_path, parse_dates=["starts_at", "ends_at"])
    # We'll build a per-minute series spanning the whole file
    start = df["starts_at"].min().floor("min")
    end = df["ends_at"].max().ceil("min")
    idx = pd.date_range(start, end, freq=f"{resolution_s}s", tz="UTC")
    s = pd.Series(0, index=idx, name="rule_based")
    for _, row in df.iterrows():
        s.loc[(idx >= row["starts_at"]) & (idx <= row["ends_at"])] = 1
    return s


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def classification_report(gt: pd.Series, pred: pd.Series, name: str) -> dict:
    aligned = pd.DataFrame({"gt": gt, "pred": pred}).dropna()
    tp = int(((aligned["gt"] == 1) & (aligned["pred"] == 1)).sum())
    fp = int(((aligned["gt"] == 0) & (aligned["pred"] == 1)).sum())
    fn = int(((aligned["gt"] == 1) & (aligned["pred"] == 0)).sum())
    tn = int(((aligned["gt"] == 0) & (aligned["pred"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "detector": name,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_positive": int(aligned["pred"].sum()),
        "total_ground_truth": int(aligned["gt"].sum()),
    }


def time_to_detection(incidents: list[dict], pred: pd.Series, name: str) -> list[dict]:
    """
    For each incident, compute how many seconds after incident start the
    detector first fires (or None if it never fires during the window).
    """
    results = []
    for inc in incidents:
        start = pd.Timestamp(inc["start"]).tz_convert("UTC")
        end = pd.Timestamp(inc["end"]).tz_convert("UTC")
        window = pred.loc[(pred.index >= start) & (pred.index <= end)]
        first_detection = window[window == 1].index.min() if (window == 1).any() else None
        ttd_s = (first_detection - start).total_seconds() if first_detection else None
        results.append({
            "incident_start": inc["start"],
            "incident_type": inc.get("type", "unknown"),
            "detector": name,
            "detected": first_detection is not None,
            "ttd_seconds": ttd_s,
        })
    return results


# ── Combined detector ─────────────────────────────────────────────────────────

def combine_detectors(*series, mode: str = "OR") -> pd.Series:
    """
    Combine multiple binary anomaly series.
    mode: 'OR'  – anomaly if any detector fires
          'AND' – anomaly only if all detectors fire
    """
    df = pd.concat(series, axis=1).dropna()
    if mode == "OR":
        return (df.sum(axis=1) >= 1).astype(int).rename("combined_OR")
    else:
        return (df.sum(axis=1) == len(series)).astype(int).rename("combined_AND")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate rule-based vs ML anomaly detection")
    parser.add_argument("--incidents", default="../../data_ingest/incidents.json")
    parser.add_argument("--metrics-anomalies", default="../output/metrics_anomalies.csv")
    parser.add_argument("--logs-anomalies", default="../output/logs_anomalies.csv")
    parser.add_argument("--rule-alerts", default="../output/rule_alerts.csv",
                        help="CSV with starts_at/ends_at columns from Alertmanager")
    parser.add_argument("--resolution", type=int, default=60,
                        help="Time resolution in seconds for alignment (default: 60)")
    parser.add_argument("--out-dir", default="../output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading incidents ...")
    incidents = load_incidents(args.incidents)
    print(f"  {len(incidents)} incidents loaded")

    print("Loading detector outputs ...")
    metrics_series = load_anomaly_series(args.metrics_anomalies)
    logs_series = load_anomaly_series(args.logs_anomalies)
    rules_series = load_rule_alerts(args.rule_alerts, args.resolution)

    # ── Align all series to common index ─────────────────────────────────────
    all_idx = metrics_series.index.union(logs_series.index)
    if len(rules_series):
        all_idx = all_idx.union(rules_series.index)

    gt = build_ground_truth_series(incidents, all_idx)
    metrics_aligned = metrics_series.reindex(all_idx, fill_value=0)
    logs_aligned = logs_series.reindex(all_idx, fill_value=0)
    rules_aligned = rules_series.reindex(all_idx, fill_value=0) if len(rules_series) else pd.Series(0, index=all_idx, name="rule_based")

    combined_or = combine_detectors(metrics_aligned, logs_aligned, mode="OR")
    combined_and = combine_detectors(metrics_aligned, logs_aligned, mode="AND")

    # ── Classification metrics ────────────────────────────────────────────────
    print("\n=== Classification Performance ===")
    reports = []
    for name, series in [
        ("metrics_isolation_forest", metrics_aligned),
        ("logs_isolation_forest", logs_aligned),
        ("rule_based", rules_aligned),
        ("combined_OR", combined_or),
        ("combined_AND", combined_and),
    ]:
        r = classification_report(gt, series, name)
        reports.append(r)
        print(f"\n[{name}]")
        print(f"  Precision : {r['precision']:.3f}")
        print(f"  Recall    : {r['recall']:.3f}")
        print(f"  F1        : {r['f1']:.3f}")
        print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}")

    # ── Time-to-detection ─────────────────────────────────────────────────────
    print("\n=== Time-to-Detection ===")
    ttd_results = []
    for name, series in [
        ("metrics_isolation_forest", metrics_aligned),
        ("logs_isolation_forest", logs_aligned),
        ("rule_based", rules_aligned),
        ("combined_OR", combined_or),
    ]:
        ttd = time_to_detection(incidents, series, name)
        ttd_results.extend(ttd)
        detected = [t for t in ttd if t["detected"]]
        if detected:
            mean_ttd = np.mean([t["ttd_seconds"] for t in detected])
            print(f"  [{name}] detected {len(detected)}/{len(incidents)}, mean TTD={mean_ttd:.0f}s")
        else:
            print(f"  [{name}] detected 0/{len(incidents)}")

    # ── Save report ────────────────────────────────────────────────────────────
    report = {
        "summary": reports,
        "time_to_detection": ttd_results,
        "incidents": incidents,
    }
    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved to {json_path}")

    # Human-readable text report
    txt_path = out_dir / "evaluation_report.txt"
    with open(txt_path, "w") as f:
        f.write("Thesis Anomaly Detection – Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        for r in reports:
            f.write(f"Detector: {r['detector']}\n")
            f.write(f"  Precision : {r['precision']:.3f}\n")
            f.write(f"  Recall    : {r['recall']:.3f}\n")
            f.write(f"  F1        : {r['f1']:.3f}\n")
            f.write(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}  TN={r['tn']}\n\n")
    print(f"Text report saved to {txt_path}")


if __name__ == "__main__":
    main()
