import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from fetch_loki import fetch_loki_range


def _loki_response(streams, status="success"):
    return {"status": status, "data": {"result": streams}}


def _ts_ns(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1e9))


_T0 = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
_T1 = datetime(2024, 6, 1, 10, 1, tzinfo=timezone.utc)


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_parses_json_line(mock_get):
    line = json.dumps({"message": "hello world", "level": "info"})
    stream = {"stream": {"service": "app"}, "values": [[_ts_ns(_T0), line]]}
    mock_get.return_value.json.return_value = _loki_response([stream])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", '{service="app"}', _T0, _T1)

    assert len(records) == 1
    assert records[0]["message"] == "hello world"
    assert records[0]["level"] == "info"
    assert records[0]["service"] == "app"
    assert "timestamp" in records[0]


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_fallback_non_json(mock_get):
    line = "plain text log line"
    stream = {"stream": {"service": "app"}, "values": [[_ts_ns(_T0), line]]}
    mock_get.return_value.json.return_value = _loki_response([stream])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", '{service="app"}', _T0, _T1)

    assert len(records) == 1
    assert records[0]["message"] == line
    assert records[0]["raw"] == line


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_empty_result(mock_get):
    mock_get.return_value.json.return_value = _loki_response([])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", '{service="app"}', _T0, _T1)

    assert records == []


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_raises_on_loki_error(mock_get):
    mock_get.return_value.json.return_value = {"status": "error", "error": "parse error"}
    mock_get.return_value.raise_for_status = MagicMock()

    with pytest.raises(RuntimeError, match="Loki error"):
        fetch_loki_range("http://localhost:3100", "bad_query", _T0, _T1)


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_merges_stream_labels_into_records(mock_get):
    stream = {
        "stream": {"service": "app", "env": "prod"},
        "values": [[_ts_ns(_T0), '{"msg": "ok"}']],
    }
    mock_get.return_value.json.return_value = _loki_response([stream])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", "q", _T0, _T1)

    assert records[0]["service"] == "app"
    assert records[0]["env"] == "prod"


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_multiple_streams(mock_get):
    stream_a = {"stream": {"service": "a"}, "values": [[_ts_ns(_T0), '{"msg": "a"}']]}
    stream_b = {"stream": {"service": "b"}, "values": [[_ts_ns(_T1), '{"msg": "b"}']]}
    mock_get.return_value.json.return_value = _loki_response([stream_a, stream_b])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", "q", _T0, _T1)

    assert len(records) == 2
    services = {r["service"] for r in records}
    assert services == {"a", "b"}


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_timestamp_is_datetime(mock_get):
    stream = {"stream": {}, "values": [[_ts_ns(_T0), '{"msg": "x"}']]}
    mock_get.return_value.json.return_value = _loki_response([stream])
    mock_get.return_value.raise_for_status = MagicMock()

    records = fetch_loki_range("http://localhost:3100", "q", _T0, _T1)

    assert isinstance(records[0]["timestamp"], datetime)
    assert records[0]["timestamp"].tzinfo is not None


@patch("fetch_loki.requests.get")
def test_fetch_loki_range_uses_nanosecond_timestamps(mock_get):
    mock_get.return_value.json.return_value = _loki_response([])
    mock_get.return_value.raise_for_status = MagicMock()

    fetch_loki_range("http://localhost:3100", "q", _T0, _T1, limit=100)

    params = mock_get.call_args[1]["params"]
    assert params["start"] == int(_T0.timestamp() * 1e9)
    assert params["end"]   == int(_T1.timestamp() * 1e9)
    assert params["limit"] == 100
