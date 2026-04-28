from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fetch_prometheus import fetch_range, QUERIES


def _prom_response(values, status="success"):
    return {
        "status": status,
        "data": {
            "result": [{"metric": {}, "values": values}] if values else [],
        },
    }


@patch("fetch_prometheus.requests.get")
def test_fetch_range_returns_series(mock_get):
    ts = [1717200000, 1717200060, 1717200120]
    vals = ["0.1", "0.2", "0.3"]
    mock_get.return_value.json.return_value = _prom_response(list(zip(ts, vals)))
    mock_get.return_value.raise_for_status = MagicMock()

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)
    result = fetch_range("http://localhost:9090", "up", start, end, 60)

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert list(result.values) == pytest.approx([0.1, 0.2, 0.3])


@patch("fetch_prometheus.requests.get")
def test_fetch_range_index_is_utc_datetimes(mock_get):
    ts = [1717200000, 1717200060]
    mock_get.return_value.json.return_value = _prom_response(list(zip(ts, ["1.0", "2.0"])))
    mock_get.return_value.raise_for_status = MagicMock()

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)
    result = fetch_range("http://localhost:9090", "up", start, end, 60)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.tz is not None


@patch("fetch_prometheus.requests.get")
def test_fetch_range_empty_result_returns_empty_series(mock_get):
    mock_get.return_value.json.return_value = _prom_response([])
    mock_get.return_value.raise_for_status = MagicMock()

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)
    result = fetch_range("http://localhost:9090", "up", start, end, 60)

    assert isinstance(result, pd.Series)
    assert len(result) == 0


@patch("fetch_prometheus.requests.get")
def test_fetch_range_raises_on_prometheus_error(mock_get):
    mock_get.return_value.json.return_value = {"status": "error", "error": "bad query"}
    mock_get.return_value.raise_for_status = MagicMock()

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="Prometheus error"):
        fetch_range("http://localhost:9090", "bad_query{}", start, end, 60)


@patch("fetch_prometheus.requests.get")
def test_fetch_range_calls_correct_endpoint(mock_get):
    mock_get.return_value.json.return_value = _prom_response([[1717200000, "0.5"]])
    mock_get.return_value.raise_for_status = MagicMock()

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)
    fetch_range("http://prom:9090", "cpu_query", start, end, 30)

    url = mock_get.call_args[0][0]
    params = mock_get.call_args[1]["params"]
    assert url == "http://prom:9090/api/v1/query_range"
    assert params["query"] == "cpu_query"
    assert params["step"] == 30
    assert params["start"] == start.timestamp()
    assert params["end"] == end.timestamp()


@patch("fetch_prometheus.requests.get")
def test_fetch_range_http_error_propagates(mock_get):
    mock_get.return_value.raise_for_status.side_effect = Exception("HTTP 500")

    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end   = datetime(2024, 6, 1, 1, tzinfo=timezone.utc)

    with pytest.raises(Exception, match="HTTP 500"):
        fetch_range("http://localhost:9090", "up", start, end, 60)


def test_queries_dict_has_system_metrics():
    system_metrics = {"cpu_busy", "mem_used_ratio", "disk_io_read_bps", "net_rx_bps"}
    assert system_metrics.issubset(set(QUERIES.keys()))


def test_queries_dict_has_app_metrics():
    app_metrics = {"request_rate", "error_rate", "error_ratio", "p50_latency", "p95_latency", "p99_latency"}
    assert app_metrics.issubset(set(QUERIES.keys()))


def test_queries_values_are_nonempty_strings():
    for name, query in QUERIES.items():
        assert isinstance(query, str), f"Query for '{name}' is not a string"
        assert len(query.strip()) > 0, f"Query for '{name}' is empty"
