#!/usr/bin/env bash
# run_experiment.sh
#
# Full experiment runner: baseline → anomaly injection → recovery → collect/detect.
# Cycles automatically through the three EXPERIMENTS.md scenarios so repeated
# cron executions cover all three over time.
#
# Usage (manual):
#   TH_IP=<ec2-ip> APP_HOST=http://<ec2-ip>:8000 ./run_experiment.sh
#   TH_IP=<ec2-ip> APP_HOST=http://<ec2-ip>:8000 SCENARIO=slow ./run_experiment.sh
#
# Cron example (every 2 hours – cycles errors → slow → burst → errors → …):
#   0 */2 * * * TH_IP=<ec2-ip> APP_HOST=http://<ec2-ip>:8000 \
#               /path/to/scripts/run_experiment.sh
#
# Environment variables:
#   TH_IP              EC2 / host IP (used to derive Prometheus/Loki URLs)
#   APP_HOST           Sample-app base URL  (default: http://$TH_IP:8000)
#   SCENARIO           Force a specific scenario: errors | slow | burst
#                      When unset, cycles automatically via scenario_state.txt
#   BASELINE_DURATION  Seconds of normal load before injection  (default: 900)
#   ANOMALY_DURATION   Seconds of anomaly injection             (default: 600)
#   RECOVERY_DURATION  Seconds of normal load after injection   (default: 900)
#   RPS                Requests per second for load generator   (default: 5)
#   WINDOW_HOURS       Telemetry window passed to collect_and_detect.sh (default: 1)
#   PROMETHEUS_HOST    Override Prometheus URL
#   LOKI_HOST          Override Loki URL
#   ALERTMANAGER_HOST  Optional — enables rule-based comparison

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="$(cd "$SCRIPT_DIR/../ml" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/../app" && pwd)"

# ── Configuration ─────────────────────────────────────────────────────────────
TH_IP="${TH_IP:-}"
APP_HOST="${APP_HOST:-}"
[[ -z "$APP_HOST" && -n "$TH_IP" ]] && APP_HOST="http://${TH_IP}:8000"

if [[ -z "$APP_HOST" ]]; then
    echo "ERROR: Set APP_HOST or TH_IP." >&2
    exit 1
fi

BASELINE_DURATION="${BASELINE_DURATION:-900}"
ANOMALY_DURATION="${ANOMALY_DURATION:-600}"
RECOVERY_DURATION="${RECOVERY_DURATION:-900}"
RPS="${RPS:-5}"
WINDOW_HOURS="${WINDOW_HOURS:-1}"

# ── Scenario cycling ──────────────────────────────────────────────────────────
SCENARIOS=(errors slow burst)
STATE_FILE="$ML_DIR/data/scenario_state.txt"

if [[ -n "${SCENARIO:-}" ]]; then
    # Explicit override — don't advance the counter.
    true
else
    mkdir -p "$ML_DIR/data"
    IDX=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
    SCENARIO="${SCENARIOS[$((IDX % 3))]}"
    echo $(( IDX + 1 )) > "$STATE_FILE"
fi

# ── Python environment ────────────────────────────────────────────────────────
VENV="$ML_DIR/.venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: ML venv not found at $VENV. Run collect_and_detect.sh once first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
PYTHON="$VENV/bin/python"

# ── Logging ───────────────────────────────────────────────────────────────────
RUN_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
LOG_DIR="$ML_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_${RUN_TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "=== run_experiment  run=${RUN_TS}  scenario=${SCENARIO} ==="
log "  App       : ${APP_HOST}"
log "  Baseline  : ${BASELINE_DURATION}s  Anomaly: ${ANOMALY_DURATION}s  Recovery: ${RECOVERY_DURATION}s"
log "  Log       : ${LOG_FILE}"

# ── Step 1: Baseline load ─────────────────────────────────────────────────────
log "--- [1] Baseline load (${BASELINE_DURATION}s normal) ---"
"$PYTHON" "$APP_DIR/load_generator.py" \
    --host     "$APP_HOST" \
    --mode     normal \
    --duration "$BASELINE_DURATION" \
    --rps      "$RPS"

# ── Step 2: Anomaly injection ─────────────────────────────────────────────────
# Record TRAIN_END_ISO *before* starting injection so the training window
# contains only baseline data — as required by EXPERIMENTS.md protocol.
TRAIN_END_ISO="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
log "--- [2] Anomaly injection: ${SCENARIO} (${ANOMALY_DURATION}s) ---"
log "  TRAIN_END_ISO=${TRAIN_END_ISO}  (boundary between baseline and anomaly)"
"$PYTHON" "$APP_DIR/load_generator.py" \
    --host            "$APP_HOST" \
    --mode            "$SCENARIO" \
    --duration        "$ANOMALY_DURATION" \
    --rps             "$RPS" \
    --record-incident

# ── Step 3: Recovery load ─────────────────────────────────────────────────────
log "--- [3] Recovery load (${RECOVERY_DURATION}s normal) ---"
"$PYTHON" "$APP_DIR/load_generator.py" \
    --host     "$APP_HOST" \
    --mode     normal \
    --duration "$RECOVERY_DURATION" \
    --rps      "$RPS"

# ── Step 4: Collect telemetry and run anomaly detection ───────────────────────
log "--- [4] Collecting telemetry and running detection ---"
TRAIN_END_ISO="$TRAIN_END_ISO" \
WINDOW_HOURS="$WINDOW_HOURS" \
TH_IP="${TH_IP:-}" \
PROMETHEUS_HOST="${PROMETHEUS_HOST:-}" \
LOKI_HOST="${LOKI_HOST:-}" \
ALERTMANAGER_HOST="${ALERTMANAGER_HOST:-}" \
    "$SCRIPT_DIR/collect_and_detect.sh"

log "=== Experiment ${RUN_TS} complete (scenario=${SCENARIO}) ==="
