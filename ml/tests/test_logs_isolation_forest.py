from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from logs_isolation_forest import extract_line_features, build_windowed_features, train_and_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(message="all good", level="info"):
    return pd.Series({"message": message, "level": level})


def _make_logs_df(n=120, freq="30s", start="2024-06-01"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": idx,
        "message":   ["normal log " + str(i) for i in range(n)],
        "level":     rng.choice(["info", "info", "info", "error"], size=n),
    })


# ── extract_line_features ─────────────────────────────────────────────────────

def test_extract_info_level_numeric():
    assert extract_line_features(_row("hello", "info"))["level_numeric"] == 1


def test_extract_error_level_numeric():
    assert extract_line_features(_row("bad", "error"))["level_numeric"] == 3


def test_extract_critical_level_numeric():
    assert extract_line_features(_row("!", "critical"))["level_numeric"] == 4


def test_extract_debug_level_numeric():
    assert extract_line_features(_row("x", "debug"))["level_numeric"] == 0


def test_extract_unknown_level_defaults_to_info():
    assert extract_line_features(_row("x", "verbose"))["level_numeric"] == 1


def test_extract_error_keyword_in_message():
    feat = extract_line_features(_row("an error occurred"))
    assert feat["error_keyword_count"] >= 1


def test_extract_multiple_error_keywords():
    feat = extract_line_features(_row("fatal crash: exception raised"))
    assert feat["error_keyword_count"] >= 2


def test_extract_warn_keyword_in_message():
    feat = extract_line_features(_row("retry after timeout"))
    assert feat["warn_keyword_count"] >= 2


def test_extract_has_traceback_true():
    assert extract_line_features(_row("traceback (most recent call)"))["has_traceback"] == 1


def test_extract_has_traceback_via_exception():
    assert extract_line_features(_row("exception in handler"))["has_traceback"] == 1


def test_extract_has_traceback_false_for_normal():
    assert extract_line_features(_row("processed request"))["has_traceback"] == 0


def test_extract_has_timeout_true():
    assert extract_line_features(_row("connection timeout"))["has_timeout"] == 1


def test_extract_has_timeout_via_timed_out():
    assert extract_line_features(_row("request timed out"))["has_timeout"] == 1


def test_extract_has_timeout_false_for_normal():
    assert extract_line_features(_row("request complete"))["has_timeout"] == 0


def test_extract_msg_length():
    msg = "hello world"
    assert extract_line_features(_row(msg))["msg_length"] == len(msg)


def test_extract_numeric_count():
    feat = extract_line_features(_row("response 200 in 123.45ms, 3 retries"))
    assert feat["numeric_count"] >= 3


def test_extract_falls_back_to_raw_field():
    row = pd.Series({"raw": "fallback from raw field"})
    feat = extract_line_features(row)
    assert feat["msg_length"] == len("fallback from raw field")


def test_extract_falls_back_to_msg_field():
    row = pd.Series({"msg": "from msg field"})
    feat = extract_line_features(row)
    assert feat["msg_length"] == len("from msg field")


def test_extract_all_keys_present():
    feat = extract_line_features(_row())
    expected = {"level_numeric", "msg_length", "error_keyword_count",
                "warn_keyword_count", "has_traceback", "has_timeout", "numeric_count"}
    assert expected == set(feat.keys())


# ── build_windowed_features ───────────────────────────────────────────────────

def test_build_windowed_returns_dataframe():
    df = _make_logs_df(60)
    assert isinstance(build_windowed_features(df, "1min"), pd.DataFrame)


def test_build_windowed_has_required_columns():
    df = _make_logs_df(60)
    agg = build_windowed_features(df, "1min")
    required = {
        "log_count", "mean_level", "max_level", "total_msg_length",
        "error_kw_sum", "warn_kw_sum", "traceback_count", "timeout_count",
        "error_rate_in_window", "log_count_delta", "log_burst_z",
        "log_count_roll_mean_5", "log_count_roll_std_5",
    }
    missing = required - set(agg.columns)
    assert not missing, f"Missing columns: {missing}"


def test_build_windowed_no_nans():
    df = _make_logs_df(120)
    agg = build_windowed_features(df, "1min")
    assert agg.isnull().sum().sum() == 0


def test_build_windowed_log_count_sums_to_input_rows():
    df = _make_logs_df(60)
    agg = build_windowed_features(df, "1min")
    assert agg["log_count"].sum() == 60


def test_build_windowed_error_kw_sum_nonzero_for_error_logs():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-06-01T00:00:00Z", "2024-06-01T00:00:30Z"]),
        "message":   ["an error occurred", "normal log"],
        "level":     ["error", "info"],
    })
    agg = build_windowed_features(df, "1min")
    assert agg["error_kw_sum"].sum() >= 1


def test_build_windowed_index_is_datetimeindex():
    df = _make_logs_df(30)
    agg = build_windowed_features(df, "1min")
    assert isinstance(agg.index, pd.DatetimeIndex)


def test_build_windowed_coarser_window_fewer_rows():
    df = _make_logs_df(120)
    agg_1m  = build_windowed_features(df, "1min")
    agg_5m  = build_windowed_features(df, "5min")
    assert len(agg_5m) < len(agg_1m)


def test_build_windowed_traceback_count():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-06-01T00:00:10Z", "2024-06-01T00:00:40Z"]),
        "message":   ["traceback occurred", "normal"],
        "level":     ["error", "info"],
    })
    agg = build_windowed_features(df, "1min")
    assert agg["traceback_count"].sum() == 1


# ── train_and_score (logs) ────────────────────────────────────────────────────

_TRAIN_END = datetime(2024, 6, 1, 0, 45, tzinfo=timezone.utc)
_IF_PARAMS = {"contamination": 0.05, "n_estimators": 10, "random_state": 0}


@pytest.fixture(scope="module")
def windowed_agg():
    return build_windowed_features(_make_logs_df(120), "1min")


def test_logs_train_and_score_output_columns(windowed_agg):
    result = train_and_score(windowed_agg, _TRAIN_END, **_IF_PARAMS)
    for col in ["anomaly_score", "is_anomaly", "in_train"]:
        assert col in result.columns


def test_logs_train_and_score_binary_labels(windowed_agg):
    result = train_and_score(windowed_agg, _TRAIN_END, **_IF_PARAMS)
    assert set(result["is_anomaly"].unique()).issubset({0, 1})
    assert set(result["in_train"].unique()).issubset({0, 1})


def test_logs_train_and_score_result_length(windowed_agg):
    result = train_and_score(windowed_agg, _TRAIN_END, **_IF_PARAMS)
    assert len(result) == len(windowed_agg)


def test_logs_train_and_score_score_is_float(windowed_agg):
    result = train_and_score(windowed_agg, _TRAIN_END, **_IF_PARAMS)
    assert result["anomaly_score"].dtype.kind == "f"


def test_logs_train_and_score_training_mask(windowed_agg):
    result = train_and_score(windowed_agg, _TRAIN_END, **_IF_PARAMS)
    expected_train = (windowed_agg.index < pd.Timestamp(_TRAIN_END)).sum()
    assert result["in_train"].sum() == expected_train
