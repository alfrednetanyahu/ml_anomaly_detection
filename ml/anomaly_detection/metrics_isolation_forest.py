#!/usr/bin/env python3
"""
metrics_isolation_forest.py – Unsupervised anomaly detection on Prometheus metrics.

Pipeline:
  1. Load metrics CSV/Parquet (from fetch_prometheus.py)
  2. Feature engineering (rolling stats, time features, deltas)
  3. Train IsolationForest on a "normal" training window
  4. Score full dataset; label anomalies
  5. Save results + plots to ml/output/

Usage:
  python metrics_isolation_forest.py \
    --data ../data/metrics.parquet \
    --train-end 2024-06-01T08:00:00Z \
    --contamination 0.02 \
    --out ../output/metrics_anomalies.csv

THESIS NOTE: This is the baseline ML model. Extend by:
  - Trying LSTM Autoencoder for temporal patterns
  - Adding more engineered features (FFT components, lag features)
  - Hyperparameter search for contamination and n_estimators
  - Ensembling metrics + log anomaly scores
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Feature engineering ───────────────────────────────────────────────────────

FEATURE_COLS = [
    "cpu_busy",
    "mem_used_ratio",
    "request_rate",
    "error_ratio",
    "p95_latency",
    "p99_latency",
]

ROLLING_WINDOWS = [5, 15, 30]  # in samples (multiply by step to get seconds)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix used for IsolationForest.

    THESIS NOTE: Add domain-specific features here:
      - Fourier components of periodic metrics (daily/hourly seasonality)
      - Correlation-based features between metric pairs
      - Rate of change (second derivative)
    """
    feat = pd.DataFrame(index=df.index)

    # ── Time-based features ──────────────────────────────────────────────────
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    for col in FEATURE_COLS:
        if col not in df.columns:
            continue

        s = df[col].fillna(0)
        feat[col] = s

        # Delta (first difference)
        feat[f"{col}_delta"] = s.diff().fillna(0)

        # Rolling statistics
        for w in ROLLING_WINDOWS:
            feat[f"{col}_roll_mean_{w}"] = s.rolling(w, min_periods=1).mean()
            feat[f"{col}_roll_std_{w}"] = s.rolling(w, min_periods=1).std().fillna(0)

    return feat.fillna(0)


# ── Model training & scoring ──────────────────────────────────────────────────

def train_and_score(
    df: pd.DataFrame,
    train_end: datetime,
    contamination: float,
    n_estimators: int,
    random_state: int,
) -> pd.DataFrame:
    features = engineer_features(df)

    mask_train = features.index < train_end
    if mask_train.sum() == 0:
        # train_end precedes all data (e.g. sparse data window not yet filled).
        # Fall back to first 70 % of available rows so the script can still run.
        n_train = max(1, int(len(features) * 0.7))
        mask_train = pd.Series(False, index=features.index)
        mask_train.iloc[:n_train] = True
        print(
            f"WARNING: train_end={train_end} is before all data "
            f"({features.index[0]} → {features.index[-1]}). "
            f"Falling back to first {n_train}/{len(features)} rows as training. "
            "Set TRAIN_END_ISO to a timestamp within the data range for experiments.",
            file=sys.stderr,
        )

    X_train = features[mask_train].values
    X_all = features.values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    # Train IsolationForest
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled)

    # Score: negative_outlier_factor (lower = more anomalous)
    scores = clf.decision_function(X_all_scaled)
    labels = clf.predict(X_all_scaled)  # 1 = normal, -1 = anomaly

    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = (labels == -1).astype(int)
    result["in_train"] = mask_train.astype(int)

    n_anomalies = result["is_anomaly"].sum()
    print(f"Training samples : {mask_train.sum()}")
    print(f"Total samples    : {len(result)}")
    print(f"Anomalies flagged: {n_anomalies} ({100*n_anomalies/len(result):.1f}%)")

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(result: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot anomaly score over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, col in zip(axes, ["cpu_busy", "p95_latency", "error_ratio"]):
        if col not in result.columns:
            continue
        ax.plot(result.index, result[col], label=col, linewidth=0.8)
        anomalies = result[result["is_anomaly"] == 1]
        ax.scatter(
            anomalies.index, anomalies[col],
            color="red", s=10, zorder=5, label="anomaly"
        )
        ax.set_ylabel(col)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Metrics Anomaly Detection – Isolation Forest")
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_anomalies_timeseries.png", dpi=150)
    plt.close(fig)

    # Anomaly score histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(result["anomaly_score"], bins=50, color="steelblue", edgecolor="white")
    ax2.axvline(
        result.loc[result["is_anomaly"] == 1, "anomaly_score"].max(),
        color="red", linestyle="--", label="anomaly threshold"
    )
    ax2.set_xlabel("Anomaly Score (lower = more anomalous)")
    ax2.set_ylabel("Count")
    ax2.set_title("Anomaly Score Distribution")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(out_dir / "metrics_score_distribution.png", dpi=150)
    plt.close(fig2)

    print(f"Plots saved to {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Metrics anomaly detection with IsolationForest")
    parser.add_argument("--data", default="../data/metrics.parquet")
    parser.add_argument("--train-end", required=True,
                        help="ISO8601 timestamp: data before this is 'normal' training window")
    parser.add_argument("--contamination", type=float, default=0.02,
                        help="Expected fraction of anomalies in training data (default: 0.02)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", default="../output/metrics_anomalies.csv")
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    if len(df) == 0:
        print("No metrics data fetched — skipping anomaly detection.")
        sys.exit(1)

    train_end = datetime.fromisoformat(args.train_end.replace("Z", "+00:00"))
    result = train_and_score(
        df, train_end, args.contamination, args.n_estimators, args.random_state
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(f"Results saved to {out_path}")

    plot_results(result, out_path.parent)


if __name__ == "__main__":
    main()
