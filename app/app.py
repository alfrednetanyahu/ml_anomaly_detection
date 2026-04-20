"""
Sample application for thesis anomaly detection experiments.

Endpoints:
  GET /         – healthy, fast response
  GET /work     – simulates normal CPU/DB work with jitter
  GET /slow     – artificial delay (anomaly scenario)
  GET /error    – forced 500 errors (anomaly scenario)
  GET /health   – liveness probe
  GET /metrics  – Prometheus metrics (via prometheus_client)

THESIS NOTE: Extend this app to simulate more realistic microservice
scenarios (e.g., downstream HTTP calls, database queries, message queue
processing) to create richer telemetry for the ML pipeline.
"""

import logging
import os
import random
import time

import structlog
from flask import Flask, jsonify, request
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# ─── Structured logging ───────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log = structlog.get_logger("sample-app")

# ─── Prometheus metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "app_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "app_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
ERROR_COUNTER = Counter(
    "app_errors_total",
    "Total application errors",
    ["endpoint", "error_type"],
)

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


def record_metrics(endpoint: str, status: int, duration: float):
    REQUEST_COUNT.labels(
        method=request.method, endpoint=endpoint, status=str(status)
    ).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    start = time.time()
    log.info("request", endpoint="/", method=request.method)
    data = {"message": "OK", "service": "sample-app"}
    record_metrics("/", 200, time.time() - start)
    return jsonify(data), 200


@app.route("/work")
def work():
    """Simulate a normal workload with slight randomness."""
    start = time.time()
    # Simulate variable processing time (normal distribution)
    sleep_time = max(0, random.gauss(mu=0.05, sigma=0.02))
    time.sleep(sleep_time)
    log.info("request", endpoint="/work", duration_ms=round(sleep_time * 1000, 2))
    record_metrics("/work", 200, time.time() - start)
    return jsonify({"result": "done", "duration_ms": round(sleep_time * 1000, 2)}), 200


@app.route("/slow")
def slow():
    """
    Simulate a slow endpoint – use this to generate anomalous latency.
    Latency is drawn from a heavy-tail distribution to create outliers.

    THESIS NOTE: Trigger this endpoint in load scripts to create
    labeled anomaly windows for ground truth.
    """
    start = time.time()
    delay = float(request.args.get("delay", random.uniform(2.0, 8.0)))
    log.warning("slow_request", endpoint="/slow", delay_s=delay)
    time.sleep(delay)
    record_metrics("/slow", 200, time.time() - start)
    return jsonify({"message": "slow response", "delay_s": delay}), 200


@app.route("/error")
def error():
    """
    Simulate random 500 errors.
    error_rate query param controls probability (0-1, default 1.0).

    THESIS NOTE: Use this to create labeled error anomaly windows.
    """
    start = time.time()
    error_rate = float(request.args.get("error_rate", 1.0))
    if random.random() < error_rate:
        ERROR_COUNTER.labels(endpoint="/error", error_type="forced_500").inc()
        log.error(
            "forced_error",
            endpoint="/error",
            error_type="forced_500",
            status=500,
        )
        record_metrics("/error", 500, time.time() - start)
        return jsonify({"error": "Simulated internal server error"}), 500
    record_metrics("/error", 200, time.time() - start)
    return jsonify({"message": "no error this time"}), 200


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(REGISTRY), 200, {"Content-Type": CONTENT_TYPE_LATEST}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("APP_ENV", "production") == "development"
    log.info("starting", port=port, debug=debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
