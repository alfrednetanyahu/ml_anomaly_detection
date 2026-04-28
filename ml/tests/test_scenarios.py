"""
test_scenarios.py – End-to-end detection tests for the three EXPERIMENTS.md scenarios.

Each test builds synthetic data that mirrors the experiment protocol
(15-min baseline → inject anomaly → check detector) and asserts the
expected detection behaviour described in EXPERIMENTS.md.

Scenario 1 – High Error Rate   : error_ratio spikes to ~80%  → metrics IF detects
Scenario 2 – High Latency      : p95/p99 latency spikes 20–40× → metrics IF detects
Scenario 3 – Log Error Burst   : error keyword flood, no metric change
                                  → logs IF detects, rule-based misses,
                                    combined_OR recovers the gap

IsolationForest calibration note
---------------------------------
Production runs use contamination=0.02 (few anomalies expected in a steady-state
window).  Scenario tests use contamination=0.15 and n_estimators=200 to verify
that these patterns ARE detectable — not to reproduce production thresholds.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from evaluate_vs_rules import classification_report, combine_detectors
from logs_isolation_forest import build_windowed_features
from logs_isolation_forest import train_and_score as logs_train_and_score
from metrics_isolation_forest import FEATURE_COLS
from metrics_isolation_forest import train_and_score as metrics_train_and_score

# ── Protocol constants ────────────────────────────────────────────────────────
# 90-min clean baseline + 30-min anomaly window = 2-hour total run.
# TRAIN_END_ISO is set to the anomaly injection start — as required by the
# experiment protocol ("note the start timestamp" → set TRAIN_END_ISO).
_N_TRAIN = 90
_N_ANOMALY = 30
_N_TOTAL = _N_TRAIN + _N_ANOMALY

_TRAIN_END = datetime(2024, 6, 1, 11, 30, tzinfo=timezone.utc)

# Higher contamination + more trees for scenario detectability checks.
# These are NOT production hyperparameters.
_SCENARIO_IF = {"contamination": 0.15, "n_estimators": 200, "random_state": 42}


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _normal_metrics_df(rng) -> pd.DataFrame:
    """Baseline: all features uniformly distributed in a narrow low-noise range."""
    idx = pd.date_range("2024-06-01 10:00", periods=_N_TOTAL, freq="1min", tz="UTC")
    df = pd.DataFrame(index=idx)
    for col in FEATURE_COLS:
        df[col] = rng.uniform(0.01, 0.05, size=_N_TOTAL)
    return df


def _inject_spike(df: pd.DataFrame, col: str, val: float) -> pd.DataFrame:
    """Replace the last _N_ANOMALY rows of `col` with `val` ± small jitter."""
    rng = np.random.default_rng(99)
    df = df.copy()
    df.iloc[-_N_ANOMALY:, df.columns.get_loc(col)] = val + rng.uniform(-0.03, 0.03, size=_N_ANOMALY)
    return df


def _metrics_scores(df: pd.DataFrame) -> pd.DataFrame:
    return metrics_train_and_score(df, _TRAIN_END, **_SCENARIO_IF)


def _detection_rate(result: pd.DataFrame) -> float:
    """Fraction of the anomaly window flagged."""
    return result.iloc[-_N_ANOMALY:]["is_anomaly"].mean()


def _false_positive_rate(result: pd.DataFrame) -> float:
    """Fraction of the training window incorrectly flagged."""
    return result.iloc[:_N_TRAIN]["is_anomaly"].mean()


# ── Scenario 1 – High Error Rate ─────────────────────────────────────────────

def test_scenario1_error_ratio_spike_detected():
    """
    Scenario 1: 80% error_ratio (load_generator --mode errors) is detected by
    metrics IF.  Mirrors EXPERIMENTS.md: 'Expected ML detection: Score spike in
    error_ratio and error_kw_sum features.'
    """
    df = _inject_spike(_normal_metrics_df(np.random.default_rng(1)), "error_ratio", 0.80)
    result = _metrics_scores(df)
    assert _detection_rate(result) >= 0.5, (
        f"Scenario 1: ≥50% of anomaly window should be flagged, "
        f"got {_detection_rate(result):.1%}"
    )


def test_scenario1_clean_baseline_has_low_false_positive_rate():
    """Training window should stay mostly clean when anomaly is only in test window."""
    df = _inject_spike(_normal_metrics_df(np.random.default_rng(1)), "error_ratio", 0.80)
    result = _metrics_scores(df)
    assert _false_positive_rate(result) < 0.30, (
        f"Scenario 1: too many false positives in training window "
        f"({_false_positive_rate(result):.1%})"
    )


# ── Scenario 2 – High Latency ─────────────────────────────────────────────────

def test_scenario2_latency_spike_detected():
    """
    Scenario 2: 3–7s latency (load_generator --mode slow) spikes both p95 and p99 —
    they are correlated in real slow scenarios.  Mirrors EXPERIMENTS.md: 'Expected
    ML detection: Score spike in p95_latency, p99_latency, p95_latency_delta.'
    """
    df = _normal_metrics_df(np.random.default_rng(2))
    df = _inject_spike(df, "p95_latency", 5.0)
    df = _inject_spike(df, "p99_latency", 8.0)
    result = _metrics_scores(df)
    assert _detection_rate(result) >= 0.5, (
        f"Scenario 2: ≥50% of anomaly window should be flagged, "
        f"got {_detection_rate(result):.1%}"
    )


def test_scenario2_clean_baseline_has_low_false_positive_rate():
    df = _normal_metrics_df(np.random.default_rng(2))
    df = _inject_spike(df, "p95_latency", 5.0)
    df = _inject_spike(df, "p99_latency", 8.0)
    result = _metrics_scores(df)
    assert _false_positive_rate(result) < 0.30


# ── Scenario 3 – Log Error Burst ─────────────────────────────────────────────

def _make_scenario3_raw_logs() -> pd.DataFrame:
    """
    Build raw log lines matching the Scenario 3 injection from EXPERIMENTS.md:
      Training  : 90 min of realistic varied normal logs (~4/min via random arrival)
      Anomaly   : 30 min of error keyword flood (~20/min, mirroring
                  `docker exec sample-app ... seq 1 200 error messages`)

    Using random timestamps and varied message content in the training window
    ensures StandardScaler sees non-zero variance across all features.
    """
    rng = np.random.default_rng(7)

    # Training window: random arrival times within 10:00–11:30 (not fixed intervals)
    # so log_count and total_msg_length vary between 1-minute windows.
    n_train = _N_TRAIN * 4
    train_epoch_start = pd.Timestamp("2024-06-01 10:00", tz="UTC").timestamp()
    train_epoch_end   = pd.Timestamp("2024-06-01 11:30", tz="UTC").timestamp()
    train_ts = sorted(rng.uniform(train_epoch_start, train_epoch_end, size=n_train))
    train_times = pd.to_datetime(train_ts, unit="s", utc=True)

    endpoints = ["/health", "/api/v1/users", "/api/v1/items", "/metrics"]
    methods   = ["GET", "POST", "PUT"]
    durations = rng.integers(5, 500, size=n_train)
    train_df = pd.DataFrame({
        "timestamp": train_times,
        "message": [
            f"processed {rng.choice(methods)} {rng.choice(endpoints)} in {d}ms"
            for d in durations
        ],
        # 3 % of training logs are warnings — gives slight variance to mean_level
        "level": rng.choice(["info", "warning"], p=[0.97, 0.03], size=n_train).tolist(),
    })

    # Anomaly window: dense error flood with timeout + retry + exception keywords
    n_anomaly = _N_ANOMALY * 20
    anomaly_times = pd.date_range("2024-06-01 11:30", periods=n_anomaly, freq="3s", tz="UTC")
    retry_nums = rng.integers(1, 10, size=n_anomaly)
    anomaly_df = pd.DataFrame({
        "timestamp": anomaly_times,
        "message": [
            f"DB connection timeout - retry {r} failed, exception in handler"
            for r in retry_nums
        ],
        "level": ["error"] * n_anomaly,
    })

    return pd.concat([train_df, anomaly_df]).reset_index(drop=True)


@pytest.fixture(scope="module")
def scenario3_windowed():
    return build_windowed_features(_make_scenario3_raw_logs(), "1min")


@pytest.fixture(scope="module")
def scenario3_result(scenario3_windowed):
    return logs_train_and_score(scenario3_windowed, _TRAIN_END, **_SCENARIO_IF)


def test_scenario3_error_kw_sum_spikes_in_anomaly_window(scenario3_windowed):
    """
    Validate the feature engineering captures the burst: error_kw_sum should be
    ≥5× higher in the anomaly window than in the baseline.
    """
    train_mean  = scenario3_windowed.loc[
        scenario3_windowed.index < pd.Timestamp(_TRAIN_END), "error_kw_sum"
    ].mean()
    anomaly_mean = scenario3_windowed.loc[
        scenario3_windowed.index >= pd.Timestamp(_TRAIN_END), "error_kw_sum"
    ].mean()
    assert anomaly_mean > train_mean + 5, (
        f"error_kw_sum should be much higher in anomaly window "
        f"(train={train_mean:.2f}, anomaly={anomaly_mean:.2f})"
    )


def test_scenario3_log_burst_detected_by_logs_IF(scenario3_result):
    """
    Scenario 3: error keyword flood → logs IF flags the anomaly window.
    Mirrors EXPERIMENTS.md: 'Expected ML detection: Spike in error_kw_sum,
    traceback_count, log_burst_z.'
    """
    anomaly_window = scenario3_result.loc[
        scenario3_result.index >= pd.Timestamp(_TRAIN_END)
    ]
    detection_rate = anomaly_window["is_anomaly"].mean()
    assert detection_rate >= 0.5, (
        f"Scenario 3: ≥50% of anomaly window should be flagged, got {detection_rate:.1%}"
    )


def test_scenario3_training_window_mostly_clean(scenario3_result):
    """Baseline window (normal varied logs) should not be mostly flagged."""
    train_window = scenario3_result.loc[
        scenario3_result.index < pd.Timestamp(_TRAIN_END)
    ]
    false_positive_rate = train_window["is_anomaly"].mean()
    assert false_positive_rate < 0.30, (
        f"Scenario 3: too many false positives in training window "
        f"({false_positive_rate:.1%})"
    )


# ── Scenario 3 ML advantage: rule-based misses log-only anomalies ─────────────

def test_scenario3_rule_based_misses_log_burst():
    """
    No Prometheus rule covers log-only anomalies (EXPERIMENTS.md: 'Expected
    rule-based alert: None').  Rule-based series stays zero throughout.
    """
    idx = pd.date_range("2024-06-01 10:00", periods=_N_TOTAL, freq="1min", tz="UTC")
    rule_based = pd.Series(0, index=idx, name="rule_based")
    gt = pd.Series(0, index=idx, name="ground_truth")
    gt.iloc[-_N_ANOMALY:] = 1

    report = classification_report(gt, rule_based, "rule_based")
    assert report["recall"] == 0.0


def test_scenario3_combined_OR_recovers_log_burst():
    """
    combined_OR = metrics_IF | logs_IF catches Scenario 3 even when metrics_IF
    sees nothing — the key ML advantage over pure threshold rules described in
    EXPERIMENTS.md: 'This scenario is most illustrative of ML's advantage.'
    """
    idx = pd.date_range("2024-06-01 10:00", periods=_N_TOTAL, freq="1min", tz="UTC")
    metrics_anomaly = pd.Series(0, index=idx, name="metrics_IF")
    logs_anomaly    = pd.Series(0, index=idx, name="logs_IF")
    logs_anomaly.iloc[-_N_ANOMALY:] = 1

    gt = pd.Series(0, index=idx, name="ground_truth")
    gt.iloc[-_N_ANOMALY:] = 1

    combined = combine_detectors(metrics_anomaly, logs_anomaly, mode="OR")
    report = classification_report(gt, combined, "combined_OR")
    assert report["recall"] == 1.0
