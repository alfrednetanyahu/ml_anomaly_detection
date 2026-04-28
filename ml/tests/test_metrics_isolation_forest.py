from datetime import datetime, timezone

import numpy as np
import pandas as pd

from metrics_isolation_forest import (
    engineer_features, train_and_score, FEATURE_COLS, ROLLING_WINDOWS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_metrics_df(n=120, start="2024-06-01"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    for col in FEATURE_COLS:
        df[col] = rng.uniform(0, 1, size=n)
    return df


def _train_end(offset_min=70):
    """Return a UTC datetime 'offset_min' minutes after the default df start."""
    return datetime(2024, 6, 1, offset_min // 60, offset_min % 60, tzinfo=timezone.utc)


_IF_PARAMS = {"contamination": 0.05, "n_estimators": 10, "random_state": 0}


# ── engineer_features ─────────────────────────────────────────────────────────

def test_engineer_features_output_length():
    df = _make_metrics_df(60)
    assert len(engineer_features(df)) == 60


def test_engineer_features_no_nans():
    df = _make_metrics_df(60)
    feat = engineer_features(df)
    assert feat.isnull().sum().sum() == 0


def test_engineer_features_no_nans_with_missing_values():
    df = _make_metrics_df(60)
    df.iloc[5, 0] = np.nan
    feat = engineer_features(df)
    assert feat.isnull().sum().sum() == 0


def test_engineer_features_time_columns_present():
    feat = engineer_features(_make_metrics_df(30))
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert col in feat.columns, f"Missing time feature: {col}"


def test_engineer_features_time_values_bounded():
    feat = engineer_features(_make_metrics_df(30))
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert feat[col].between(-1, 1).all(), f"{col} values out of [-1, 1]"


def test_engineer_features_delta_columns_present():
    df = _make_metrics_df(60)
    feat = engineer_features(df)
    for col in FEATURE_COLS:
        assert f"{col}_delta" in feat.columns, f"Missing delta: {col}_delta"


def test_engineer_features_rolling_columns_present():
    df = _make_metrics_df(60)
    feat = engineer_features(df)
    for col in FEATURE_COLS:
        for w in ROLLING_WINDOWS:
            assert f"{col}_roll_mean_{w}" in feat.columns
            assert f"{col}_roll_std_{w}" in feat.columns


def test_engineer_features_missing_column_skipped():
    df = _make_metrics_df(60).drop(columns=["cpu_busy"])
    feat = engineer_features(df)
    assert "cpu_busy" not in feat.columns
    assert "cpu_busy_delta" not in feat.columns
    for w in ROLLING_WINDOWS:
        assert f"cpu_busy_roll_mean_{w}" not in feat.columns


def test_engineer_features_first_delta_is_zero():
    df = _make_metrics_df(10)
    feat = engineer_features(df)
    assert feat["cpu_busy_delta"].iloc[0] == 0.0


def test_engineer_features_rolling_mean_shape_matches_input():
    df = _make_metrics_df(50)
    feat = engineer_features(df)
    assert len(feat) == len(df)


# ── train_and_score ───────────────────────────────────────────────────────────

def test_train_and_score_output_columns():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    for col in ["anomaly_score", "is_anomaly", "in_train"]:
        assert col in result.columns


def test_train_and_score_binary_is_anomaly():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    assert set(result["is_anomaly"].unique()).issubset({0, 1})


def test_train_and_score_binary_in_train():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    assert set(result["in_train"].unique()).issubset({0, 1})


def test_train_and_score_training_mask_count():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    assert result["in_train"].sum() == 70


def test_train_and_score_preserves_original_columns():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    for col in FEATURE_COLS:
        assert col in result.columns


def test_train_and_score_anomaly_score_is_float():
    df = _make_metrics_df(100)
    result = train_and_score(df, _train_end(70), **_IF_PARAMS)
    assert result["anomaly_score"].dtype.kind == "f"


def test_train_and_score_contamination_respected():
    # With contamination=0.10, roughly 10% of total samples should be anomalies
    n = 200
    df = _make_metrics_df(n)
    result = train_and_score(df, _train_end(140), contamination=0.10,
                             n_estimators=50, random_state=42)
    anomaly_rate = result["is_anomaly"].mean()
    # IsolationForest contamination sets the decision threshold, not an exact count;
    # allow generous tolerance
    assert 0.05 <= anomaly_rate <= 0.20


def test_train_and_score_result_length_matches_input():
    df = _make_metrics_df(80)
    result = train_and_score(df, _train_end(56), contamination=0.02,
                             n_estimators=10, random_state=0)
    assert len(result) == len(df)
