#!/usr/bin/env python3
"""
logs_isolation_forest.py – Unsupervised anomaly detection on log data.

Pipeline:
  1. Load logs CSV/Parquet (from fetch_loki.py)
  2. Extract features from raw log lines
  3. Aggregate features into time windows (default: 1 minute)
  4. Train IsolationForest on a normal window
  5. Score and label anomalies, save to CSV

Usage:
  python logs_isolation_forest.py \
    --data ../data/logs.csv \
    --train-end 2024-06-01T08:00:00Z \
    --window 60 \
    --contamination 0.02 \
    --out ../output/logs_anomalies.csv

THESIS NOTE: Log anomaly detection is an open research area. Extend this by:
  - Using Drain/LogParse for template extraction (log clustering)
  - Word embeddings (TF-IDF, fastText) on message text for semantic features
  - Sequence-based models (LSTM) on log event sequences
  - Online/streaming mode for real-time detection
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Keyword lists ─────────────────────────────────────────────────────────────
ERROR_KEYWORDS = [
    "error", "exception", "traceback", "fatal", "critical",
    "fail", "failed", "failure", "crash", "panic",
]
WARN_KEYWORDS = ["warn", "warning", "deprecated", "timeout", "retry", "slow"]
LEVEL_MAP = {"debug": 0, "info": 1, "warning": 2, "warn": 2, "error": 3, "critical": 4}

# ── Feature extraction per log line ──────────────────────────────────────────

def extract_line_features(row: pd.Series) -> dict:
    """
    Extract numerical features from a single log line.

    THESIS NOTE: Add more features:
      - HTTP status code (parsed from message)
      - Parsed numeric fields (duration_ms, bytes, etc.)
      - Log template ID (requires Drain or similar clustering)
    """
    msg = str(row.get("message", row.get("msg", row.get("raw", "")))).lower()
    level_str = str(row.get("level", "info")).lower()

    return {
        "level_numeric": LEVEL_MAP.get(level_str, 1),
        "msg_length": len(msg),
        "error_keyword_count": sum(kw in msg for kw in ERROR_KEYWORDS),
        "warn_keyword_count": sum(kw in msg for kw in WARN_KEYWORDS),
        "has_traceback": int("traceback" in msg or "exception" in msg),
        "has_timeout": int("timeout" in msg or "timed out" in msg),
        "numeric_count": len(re.findall(r"\d+\.\d+|\d+", msg)),
    }


def build_windowed_features(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Aggregate per-line features into fixed time windows.

    Returns a DataFrame indexed by window timestamp.
    """
    line_feats = df.apply(extract_line_features, axis=1, result_type="expand")
    line_feats.index = df["timestamp"] if "timestamp" in df.columns else df.index
    line_feats.index = pd.to_datetime(line_feats.index, utc=True)

    # ── Aggregate within each window ─────────────────────────────────────────
    agg = line_feats.resample(window).agg(
        log_count=("level_numeric", "count"),
        mean_level=("level_numeric", "mean"),
        max_level=("level_numeric", "max"),
        total_msg_length=("msg_length", "sum"),
        error_kw_sum=("error_keyword_count", "sum"),
        warn_kw_sum=("warn_keyword_count", "sum"),
        traceback_count=("has_traceback", "sum"),
        timeout_count=("has_timeout", "sum"),
        mean_numeric_count=("numeric_count", "mean"),
    )

    # ── Derived features ──────────────────────────────────────────────────────
    agg["error_rate_in_window"] = agg["error_kw_sum"] / (agg["log_count"] + 1e-6)
    agg["log_count_delta"] = agg["log_count"].diff().fillna(0)

    # Rolling log burst indicator
    agg["log_count_roll_mean_5"] = agg["log_count"].rolling(5, min_periods=1).mean()
    agg["log_count_roll_std_5"] = agg["log_count"].rolling(5, min_periods=1).std().fillna(0)
    agg["log_burst_z"] = (
        (agg["log_count"] - agg["log_count_roll_mean_5"])
        / (agg["log_count_roll_std_5"] + 1e-6)
    )

    return agg.fillna(0)


# ── Training & scoring ────────────────────────────────────────────────────────

def train_and_score(
    agg: pd.DataFrame,
    train_end,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> pd.DataFrame:
    if not isinstance(train_end, pd.Timestamp):
        train_end = pd.Timestamp(train_end)
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize("UTC")
    mask_train = agg.index < train_end
    if mask_train.sum() == 0:
        n_train = max(1, int(len(agg) * 0.7))
        mask_train = pd.Series(False, index=agg.index)
        mask_train.iloc[:n_train] = True
        print(
            f"WARNING: train_end={train_end} is before all data "
            f"({agg.index[0]} → {agg.index[-1]}). "
            f"Falling back to first {n_train}/{len(agg)} rows as training. "
            "Set TRAIN_END_ISO to a timestamp within the data range for experiments.",
            file=sys.stderr,
        )

    X_train = agg[mask_train].values
    X_all = agg.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled)

    scores = clf.decision_function(X_all_scaled)
    labels = clf.predict(X_all_scaled)

    result = agg.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = (labels == -1).astype(int)
    result["in_train"] = mask_train.astype(int)

    n_anomalies = result["is_anomaly"].sum()
    print(f"Training windows : {mask_train.sum()}")
    print(f"Total windows    : {len(result)}")
    print(f"Anomalies flagged: {n_anomalies} ({100*n_anomalies/len(result):.1f}%)")

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(result: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, col in zip(axes, ["log_count", "error_kw_sum", "anomaly_score"]):
        ax.plot(result.index, result[col], linewidth=0.8, label=col)
        if col != "anomaly_score":
            anomalies = result[result["is_anomaly"] == 1]
            ax.scatter(anomalies.index, anomalies[col], color="red", s=10, zorder=5)
        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=8)

    # Mark anomaly windows as red vertical bands
    for ts, row in result[result["is_anomaly"] == 1].iterrows():
        for ax in axes:
            ax.axvspan(ts, ts + pd.Timedelta("1min"), alpha=0.15, color="red")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Log Anomaly Detection – Isolation Forest")
    plt.tight_layout()
    fig.savefig(out_dir / "logs_anomalies_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Log anomaly detection with IsolationForest")
    parser.add_argument("--data", default="../data/logs.csv")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--window", default="1min",
                        help="Aggregation window, e.g. 1min, 5min (default: 1min)")
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", default="../output/logs_anomalies.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"])

    print(f"Loaded {len(df)} log entries")

    print(f"Aggregating into {args.window} windows ...")
    agg = build_windowed_features(df, args.window)
    print(f"  → {len(agg)} windows, {len(agg.columns)} features")

    from datetime import datetime, timezone
    train_end = datetime.fromisoformat(args.train_end.replace("Z", "+00:00"))
    result = train_and_score(
        agg, train_end, args.contamination, args.n_estimators, args.random_state
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(f"Results saved to {out_path}")

    plot_results(result, out_path.parent)


if __name__ == "__main__":
    main()
