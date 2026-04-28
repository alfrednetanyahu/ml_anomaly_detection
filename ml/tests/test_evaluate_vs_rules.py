import pandas as pd
import pytest

from evaluate_vs_rules import (
    build_ground_truth_series,
    classification_report,
    combine_detectors,
    load_anomaly_series,
    load_rule_alerts,
    time_to_detection,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _idx(start="2024-06-01 10:00", periods=60, freq="1min"):
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


INCIDENTS = [
    {"start": "2024-06-01T10:10:00Z", "end": "2024-06-01T10:20:00Z", "type": "errors"},
    {"start": "2024-06-01T10:40:00Z", "end": "2024-06-01T10:50:00Z", "type": "slow"},
]


# ── build_ground_truth_series ─────────────────────────────────────────────────

def test_ground_truth_zeros_before_any_incident():
    idx = _idx("2024-06-01 10:00", periods=9)
    gt = build_ground_truth_series(INCIDENTS, idx)
    assert (gt == 0).all()


def test_ground_truth_ones_inside_incident():
    idx = _idx("2024-06-01 10:10", periods=11)
    gt = build_ground_truth_series(INCIDENTS, idx)
    assert (gt == 1).all()


def test_ground_truth_zeros_between_incidents():
    idx = _idx("2024-06-01 10:21", periods=18)
    gt = build_ground_truth_series(INCIDENTS, idx)
    assert (gt == 0).all()


def test_ground_truth_mixed_values_over_full_hour():
    idx = _idx("2024-06-01 10:00", periods=60)
    gt = build_ground_truth_series(INCIDENTS, idx)
    assert gt.sum() > 0
    assert gt.sum() < len(idx)


def test_ground_truth_empty_incidents_all_zero():
    idx = _idx(periods=30)
    gt = build_ground_truth_series([], idx)
    assert (gt == 0).all()


def test_ground_truth_returns_series_with_same_index():
    idx = _idx(periods=20)
    gt = build_ground_truth_series(INCIDENTS, idx)
    assert isinstance(gt, pd.Series)
    assert len(gt) == len(idx)


# ── classification_report ─────────────────────────────────────────────────────

def _clf_report(gt_vals, pred_vals, name="det"):
    idx = _idx(periods=len(gt_vals))
    gt   = pd.Series(gt_vals, index=idx)
    pred = pd.Series(pred_vals, index=idx)
    return classification_report(gt, pred, name)


def test_perfect_detection():
    r = _clf_report([0,0,1,1,1,0], [0,0,1,1,1,0])
    assert r["precision"] == 1.0
    assert r["recall"]    == 1.0
    assert r["f1"]        == 1.0
    assert r["tp"] == 3
    assert r["fp"] == 0
    assert r["fn"] == 0


def test_no_detections():
    r = _clf_report([0,0,1,1,0,0], [0,0,0,0,0,0])
    assert r["recall"] == 0.0
    assert r["fn"] == 2
    assert r["tp"] == 0


def test_all_false_positives():
    r = _clf_report([0,0,0,0,0,0], [1,1,1,1,1,1])
    assert r["precision"] == 0.0
    assert r["tp"] == 0
    assert r["fp"] == 6


def test_partial_detection():
    r = _clf_report([0,0,1,1,0,0,0,0], [0,0,1,0,0,0,0,0])
    assert r["tp"] == 1
    assert r["fn"] == 1
    assert r["fp"] == 0
    assert r["precision"] == 1.0
    assert r["recall"]    == pytest.approx(0.5)
    assert r["f1"]        == pytest.approx(2/3, abs=1e-4)  # rounded to 4dp in source


def test_no_positives_in_gt_or_pred():
    r = _clf_report([0,0,0,0], [0,0,0,0])
    assert r["tp"] == 0
    assert r["precision"] == 0.0
    assert r["recall"]    == 0.0
    assert r["f1"]        == 0.0


def test_report_contains_required_keys():
    r = _clf_report([0,1], [1,0])
    for key in ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "detector"]:
        assert key in r


def test_report_detector_name():
    r = _clf_report([0,0], [0,0], name="my_detector")
    assert r["detector"] == "my_detector"


# ── time_to_detection ─────────────────────────────────────────────────────────

def _pred_series(values, start="2024-06-01 10:10", freq="1min"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="det")


def test_ttd_detects_at_incident_start():
    # First timestep fires — TTD should be 0
    pred = _pred_series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    results = time_to_detection([INCIDENTS[0]], pred, "det")
    assert results[0]["detected"] is True
    assert results[0]["ttd_seconds"] == pytest.approx(0.0)


def test_ttd_detects_after_three_minutes():
    # idx[0]=10:10, idx[3]=10:13, delay = 3 min = 180s
    pred = _pred_series([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    results = time_to_detection([INCIDENTS[0]], pred, "det")
    assert results[0]["detected"] is True
    assert results[0]["ttd_seconds"] == pytest.approx(180.0)


def test_ttd_not_detected():
    pred = _pred_series([0] * 11)
    results = time_to_detection([INCIDENTS[0]], pred, "det")
    assert results[0]["detected"] is False
    assert results[0]["ttd_seconds"] is None


def test_ttd_multiple_incidents():
    idx = _idx("2024-06-01 10:00", periods=60)
    pred = pd.Series(0, index=idx, name="det")
    pred.loc[pd.Timestamp("2024-06-01 10:12", tz="UTC")] = 1  # inside incident 1
    pred.loc[pd.Timestamp("2024-06-01 10:42", tz="UTC")] = 1  # inside incident 2
    results = time_to_detection(INCIDENTS, pred, "det")
    assert len(results) == 2
    assert all(r["detected"] for r in results)


def test_ttd_result_contains_required_keys():
    pred = _pred_series([1] * 11)
    results = time_to_detection([INCIDENTS[0]], pred, "det")
    for key in ["incident_start", "incident_type", "detector", "detected", "ttd_seconds"]:
        assert key in results[0]


def test_ttd_records_incident_type():
    pred = _pred_series([1] * 11)
    results = time_to_detection([INCIDENTS[0]], pred, "det")
    assert results[0]["incident_type"] == "errors"


# ── combine_detectors ─────────────────────────────────────────────────────────

def _series(values, name):
    idx = _idx(periods=len(values))
    return pd.Series(values, index=idx, name=name)


def test_combine_or_fires_when_either_fires():
    s1 = _series([1, 0, 0, 0, 0], "a")
    s2 = _series([0, 0, 1, 0, 0], "b")
    result = combine_detectors(s1, s2, mode="OR")
    assert result.iloc[0] == 1
    assert result.iloc[2] == 1
    assert result.iloc[1] == 0


def test_combine_or_zero_when_neither_fires():
    s1 = _series([0, 0, 0], "a")
    s2 = _series([0, 0, 0], "b")
    result = combine_detectors(s1, s2, mode="OR")
    assert (result == 0).all()


def test_combine_and_fires_only_when_both_fire():
    s1 = _series([1, 1, 0, 0], "a")
    s2 = _series([1, 0, 1, 0], "b")
    result = combine_detectors(s1, s2, mode="AND")
    assert result.iloc[0] == 1   # both fire
    assert result.iloc[1] == 0   # only s1
    assert result.iloc[2] == 0   # only s2
    assert result.iloc[3] == 0   # neither


def test_combine_or_name():
    s1, s2 = _series([0], "a"), _series([0], "b")
    assert combine_detectors(s1, s2, mode="OR").name == "combined_OR"


def test_combine_and_name():
    s1, s2 = _series([0], "a"), _series([0], "b")
    assert combine_detectors(s1, s2, mode="AND").name == "combined_AND"


def test_combine_three_detectors_or():
    idx = _idx(periods=4)
    s1 = pd.Series([1, 0, 0, 0], index=idx, name="a")
    s2 = pd.Series([0, 1, 0, 0], index=idx, name="b")
    s3 = pd.Series([0, 0, 1, 0], index=idx, name="c")
    result = combine_detectors(s1, s2, s3, mode="OR")
    assert list(result.values) == [1, 1, 1, 0]


def test_combine_three_detectors_and():
    idx = _idx(periods=4)
    s1 = pd.Series([1, 1, 0, 0], index=idx, name="a")
    s2 = pd.Series([1, 0, 1, 0], index=idx, name="b")
    s3 = pd.Series([1, 0, 0, 0], index=idx, name="c")
    result = combine_detectors(s1, s2, s3, mode="AND")
    assert list(result.values) == [1, 0, 0, 0]


# ── load_anomaly_series ───────────────────────────────────────────────────────

def test_load_anomaly_series_values(tmp_path):
    idx = pd.date_range("2024-06-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame({"is_anomaly": [0, 1, 0, 0, 1]}, index=idx)
    csv_path = tmp_path / "metrics_anomalies.csv"
    df.to_csv(csv_path)

    s = load_anomaly_series(str(csv_path))
    assert len(s) == 5
    assert s.sum() == 2


def test_load_anomaly_series_localizes_naive_index(tmp_path):
    idx = pd.date_range("2024-06-01", periods=3, freq="1min")  # tz-naive
    df = pd.DataFrame({"is_anomaly": [0, 0, 1]}, index=idx)
    csv_path = tmp_path / "anom.csv"
    df.to_csv(csv_path)

    s = load_anomaly_series(str(csv_path))
    assert s.index.tz is not None


def test_load_anomaly_series_stem_as_name(tmp_path):
    idx = pd.date_range("2024-06-01", periods=2, freq="1min", tz="UTC")
    df = pd.DataFrame({"is_anomaly": [0, 1]}, index=idx)
    csv_path = tmp_path / "metrics_anomalies_20240601.csv"
    df.to_csv(csv_path)

    s = load_anomaly_series(str(csv_path))
    assert s.name == "metrics_anomalies_20240601"


# ── load_rule_alerts ──────────────────────────────────────────────────────────

def test_load_rule_alerts_missing_file_returns_empty():
    s = load_rule_alerts("/nonexistent/path/alerts.csv", resolution_s=60)
    assert isinstance(s, pd.Series)
    assert len(s) == 0


def test_load_rule_alerts_from_csv(tmp_path):
    csv_path = tmp_path / "rule_alerts.csv"
    csv_path.write_text(
        "starts_at,ends_at\n"
        "2024-06-01T10:10:00+00:00,2024-06-01T10:20:00+00:00\n"
    )
    s = load_rule_alerts(str(csv_path), resolution_s=60)
    assert isinstance(s, pd.Series)
    assert len(s) > 0


def test_load_rule_alerts_name(tmp_path):
    csv_path = tmp_path / "rule_alerts.csv"
    csv_path.write_text(
        "starts_at,ends_at\n"
        "2024-06-01T10:10:00+00:00,2024-06-01T10:10:59+00:00\n"
    )
    s = load_rule_alerts(str(csv_path), resolution_s=60)
    assert s.name == "rule_based"
