#!/usr/bin/env bash
# collect_and_detect.sh
#
# Automates README steps 6-8: export telemetry → anomaly detection → evaluate.
# All outputs are isolated in a per-run timestamped directory so repeated
# cron executions never overwrite each other.
#
# Usage:
#   TH_IP=<ec2-ip> ./collect_and_detect.sh
#
# Cron example (every 2 hours):
#   0 */2 * * * TH_IP=1.2.3.4 /path/to/scripts/collect_and_detect.sh
#
# Environment variables (all optional except TH_IP or individual host vars):
#   TH_IP              EC2 / host IP  (used to derive default Prometheus/Loki URLs)
#   PROMETHEUS_HOST    Full URL override, e.g. http://1.2.3.4:9090
#   LOKI_HOST          Full URL override, e.g. http://1.2.3.4:3100
#   ALERTMANAGER_HOST  Optional: set to enable rule-based comparison, e.g. http://1.2.3.4:9093
#                      Not derived from TH_IP — must be set explicitly when Alertmanager is deployed
#   LOKI_QUERY         LogQL query  (default: {service="sample-app"})
#   WINDOW_HOURS       Hours of telemetry to collect per run  (default: 2)
#   TRAIN_RATIO        Fraction of window used as training    (default: 0.7)
#   TRAIN_END_ISO      Anomaly injection start time (ISO8601).  Data before this
#                      timestamp is treated as "normal" training data.
#                      REQUIRED for experiments — set to the exact moment you ran
#                      `load_generator.py --mode errors/slow/burst`.
#                      When unset, auto-computed as start + TRAIN_RATIO * WINDOW.
#   INCIDENTS_FILE     Path to incidents.json written by load_generator.py
#                      --record-incident (default: ml/data_ingest/incidents.json).
#                      Evaluation step is skipped when this file is absent.

set -euo pipefail

# ── Resolve canonical paths ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(cd "$SCRIPT_DIR/../ml" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
TH_IP="${TH_IP:-}"

if [[ -n "$TH_IP" ]]; then
    PROMETHEUS_HOST="${PROMETHEUS_HOST:-http://${TH_IP}:9090}"
    LOKI_HOST="${LOKI_HOST:-http://${TH_IP}:3100}"
else
    PROMETHEUS_HOST="${PROMETHEUS_HOST:-}"
    LOKI_HOST="${LOKI_HOST:-}"
fi
# Alertmanager is optional — set ALERTMANAGER_HOST explicitly to enable rule-based comparison
ALERTMANAGER_HOST="${ALERTMANAGER_HOST:-}"

if [[ -z "$PROMETHEUS_HOST" || -z "$LOKI_HOST" ]]; then
    echo "ERROR: Set TH_IP, or set PROMETHEUS_HOST and LOKI_HOST individually." >&2
    exit 1
fi

LOKI_QUERY="${LOKI_QUERY:-}"
[[ -z "$LOKI_QUERY" ]] && LOKI_QUERY='{service="sample-app"}'
WINDOW_HOURS="${WINDOW_HOURS:-2}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
TRAIN_END_ISO="${TRAIN_END_ISO:-}"
INCIDENTS_FILE="${INCIDENTS_FILE:-$ML_DIR/data_ingest/incidents.json}"

# ── Timestamps & time window ──────────────────────────────────────────────────
RUN_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
END_EPOCH="$(date -u +%s)"
END_ISO="$(date -u -d "@${END_EPOCH}"   +'%Y-%m-%dT%H:%M:%SZ')"
START_EPOCH="$(( END_EPOCH - WINDOW_HOURS * 3600 ))"
START_ISO="$(date -u -d "@${START_EPOCH}" +'%Y-%m-%dT%H:%M:%SZ')"

_TRAIN_END_AUTO=0
if [[ -z "$TRAIN_END_ISO" ]]; then
    TRAIN_EPOCH="$(awk "BEGIN { printf \"%.0f\", ${START_EPOCH} + ${WINDOW_HOURS} * 3600 * ${TRAIN_RATIO} }")"
    TRAIN_END_ISO="$(date -u -d "@${TRAIN_EPOCH}" +'%Y-%m-%dT%H:%M:%SZ')"
    _TRAIN_END_AUTO=1
fi

# ── Output layout ─────────────────────────────────────────────────────────────
# Raw telemetry lands in data/ with a timestamp suffix so it can be reprocessed.
# Everything derived from one run lives in output/run_<RUN_TS>/ — this also
# keeps the hardcoded PNG names from the anomaly scripts unique per run.
DATA_DIR="$ML_DIR/data"
RUN_DIR="$ML_DIR/output/run_${RUN_TS}"
LOG_DIR="$ML_DIR/logs"

METRICS_FILE="$DATA_DIR/metrics_${RUN_TS}.parquet"
LOGS_FILE="$DATA_DIR/logs_${RUN_TS}.csv"
METRICS_ANOMALIES="$RUN_DIR/metrics_anomalies.csv"
LOGS_ANOMALIES="$RUN_DIR/logs_anomalies.csv"
RULE_ALERTS_FILE="$RUN_DIR/rule_alerts.csv"

mkdir -p "$DATA_DIR" "$RUN_DIR" "$LOG_DIR"

# ── Per-run log ───────────────────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/run_${RUN_TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "=== collect_and_detect  run=${RUN_TS} ==="
log "  Window    : ${START_ISO}  →  ${END_ISO}  (${WINDOW_HOURS}h)"
log "  Train-end : ${TRAIN_END_ISO}$([[ "$_TRAIN_END_AUTO" -eq 1 ]] && echo " (auto — set TRAIN_END_ISO to anomaly injection time for experiments)")"
log "  Incidents : ${INCIDENTS_FILE}"
log "  Prometheus: ${PROMETHEUS_HOST}"
log "  Loki      : ${LOKI_HOST}"
log "  Run dir   : ${RUN_DIR}"

# ── Python environment ────────────────────────────────────────────────────────
VENV="$ML_DIR/.venv"
REQUIREMENTS="$ML_DIR/requirements.txt"

if [[ ! -f "$VENV/bin/activate" ]]; then
    log "  Venv not found — creating $VENV ..."
    python3 -m venv "$VENV"
    log "  Installing requirements from $REQUIREMENTS ..."
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$REQUIREMENTS"
    log "  Venv ready."
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
log "  Python    : $VENV (venv)"

cd "$ML_DIR"

# ── Step 6: Export telemetry ──────────────────────────────────────────────────
log "--- [6a] Fetching Prometheus metrics ---"
python3 data_ingest/fetch_prometheus.py \
    --host  "$PROMETHEUS_HOST" \
    --start "$START_ISO" \
    --end   "$END_ISO" \
    --out   "$METRICS_FILE"

log "--- [6b] Fetching Loki logs ---"
LOGS_OK=1
python3 data_ingest/fetch_loki.py \
    --host  "$LOKI_HOST" \
    --query "$LOKI_QUERY" \
    --start "$START_ISO" \
    --end   "$END_ISO" \
    --out   "$LOGS_FILE" || LOGS_OK=0

if [[ "$LOGS_OK" -eq 0 ]]; then
    log "WARNING: No logs fetched — skipping log anomaly detection and evaluation."
fi

# ── Step 7: Anomaly detection ─────────────────────────────────────────────────
# PNGs (metrics_anomalies_timeseries.png, metrics_score_distribution.png,
# logs_anomalies_timeseries.png) are written to the parent dir of --out, which
# is $RUN_DIR — so they are unique per run.
log "--- [7a] Metrics IsolationForest ---"
METRICS_OK=1
python3 anomaly_detection/metrics_isolation_forest.py \
    --data      "$METRICS_FILE" \
    --train-end "$TRAIN_END_ISO" \
    --out       "$METRICS_ANOMALIES" || METRICS_OK=0
if [[ "$METRICS_OK" -eq 0 ]]; then
    log "WARNING: Metrics anomaly detection failed — check data above."
fi

log "--- [7b] Logs IsolationForest ---"
if [[ "$LOGS_OK" -eq 1 ]]; then
    python3 anomaly_detection/logs_isolation_forest.py \
        --data      "$LOGS_FILE" \
        --train-end "$TRAIN_END_ISO" \
        --out       "$LOGS_ANOMALIES"
else
    log "  (skipped — no log data)"
fi

# ── Step 6c: Fetch Alertmanager rule alerts ───────────────────────────────────
log "--- [6c] Fetching Alertmanager rule alerts ---"
if [[ -n "$ALERTMANAGER_HOST" ]]; then
    python3 - "$ALERTMANAGER_HOST" "$START_ISO" "$END_ISO" "$RULE_ALERTS_FILE" <<'PYEOF' \
        || log "  [WARN] Could not fetch Alertmanager alerts — rule-based comparison skipped."
import sys, csv, requests
from datetime import datetime, timezone

host, start_iso, end_iso, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
end   = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

resp = requests.get(f"{host}/api/v2/alerts", timeout=10)
resp.raise_for_status()

rows = []
for a in resp.json():
    s = datetime.fromisoformat(a["startsAt"].replace("Z", "+00:00"))
    e = datetime.fromisoformat(a["endsAt"].replace("Z",   "+00:00"))
    if e > start and s < end:
        rows.append({"starts_at": a["startsAt"], "ends_at": a["endsAt"]})

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["starts_at", "ends_at"])
    w.writeheader()
    w.writerows(rows)

print(f"  {len(rows)} alert interval(s) → {out}")
PYEOF
else
    log "  [WARN] ALERTMANAGER_HOST not set — rule-based comparison skipped."
fi

# ── Step 8: Evaluate vs rule-based ────────────────────────────────────────────
log "--- [8] Evaluating vs rule-based alerts ---"
if [[ "$METRICS_OK" -eq 0 ]]; then
    log "  (skipped — no metrics anomaly output)"
elif [[ ! -f "$INCIDENTS_FILE" ]]; then
    log "  (skipped — no incidents file at $INCIDENTS_FILE)"
    log "  Run experiments first: python app/load_generator.py --mode errors --record-incident"
else
    EVAL_LOGS_ARG=""
    if [[ "$LOGS_OK" -eq 1 ]]; then
        EVAL_LOGS_ARG="--logs-anomalies $LOGS_ANOMALIES"
    fi
    # shellcheck disable=SC2086
    python3 evaluation/evaluate_vs_rules.py \
        --incidents         "$INCIDENTS_FILE" \
        --metrics-anomalies "$METRICS_ANOMALIES" \
        --rule-alerts       "$RULE_ALERTS_FILE" \
        $EVAL_LOGS_ARG \
        --out-dir           "$RUN_DIR"
fi

log "=== Run ${RUN_TS} complete ==="
log "  Raw data  : ${METRICS_FILE}"
[[ "$LOGS_OK" -eq 1 ]] && log "            : ${LOGS_FILE}"
log "  Anomalies : ${RUN_DIR}/metrics_anomalies.csv"
[[ "$LOGS_OK" -eq 1 ]] && log "            : ${RUN_DIR}/logs_anomalies.csv"
log "  Eval      : ${RUN_DIR}/evaluation_report.json"
log "  Log       : ${LOG_FILE}"
