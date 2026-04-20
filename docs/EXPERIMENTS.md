# EXPERIMENTS.md – Reproducing Thesis Experiments

This document describes how to reproduce the three anomaly injection scenarios
used in the thesis evaluation.

---

## Setup Checklist

- [ ] EC2 instance running (`terraform apply`)
- [ ] Monitoring stack running (`docker-compose up -d`)
- [ ] Prometheus scraping app and node exporter (`http://<IP>:9090/targets`)
- [ ] Grafana connected (`http://<IP>:3000`)
- [ ] Baseline load generator running for ≥ 30 min before injecting anomalies

---

## Experiment Protocol

For each scenario:
1. Run normal load for **15 minutes** to establish a baseline.
2. Note the **start timestamp** (UTC).
3. Inject the anomaly.
4. Note the **end timestamp**.
5. Continue normal load for **15 minutes** after.
6. Record the incident in `ml/data_ingest/incidents.json` using `--record-incident`.

---

## Scenario 1 – High Error Rate

**Goal**: Evaluate detection of a spike in HTTP 5xx errors.

```bash
# Baseline
python app/load_generator.py --host http://<IP>:8000 --mode normal --duration 900

# Anomaly injection (10 min, 80% error rate)
python app/load_generator.py \
  --host http://<IP>:8000 \
  --mode errors \
  --duration 600 \
  --record-incident

# Recovery
python app/load_generator.py --host http://<IP>:8000 --mode normal --duration 900
```

**Expected rule-based alert**: `HighErrorRate` fires within ~1 min (5m rate window).  
**Expected ML detection**: Score spike in `error_ratio` and `error_kw_sum` features.

---

## Scenario 2 – High Latency / Slow Endpoint

**Goal**: Evaluate detection of p95 latency degradation.

```bash
# Baseline
python app/load_generator.py --host http://<IP>:8000 --mode normal --duration 900

# Anomaly injection (10 min, random 3–7s delays)
python app/load_generator.py \
  --host http://<IP>:8000 \
  --mode slow \
  --duration 600 \
  --record-incident

# Recovery
python app/load_generator.py --host http://<IP>:8000 --mode normal --duration 900
```

**Expected rule-based alert**: `SlowResponseTime` fires when p95 > 2s.  
**Expected ML detection**: Score spike in `p95_latency`, `p99_latency`, `p95_latency_delta`.

---

## Scenario 3 – Burst of Error Logs (Log-only anomaly)

**Goal**: Inject error logs without triggering HTTP 5xx (e.g., internal warning flood).

Modify the sample app or call the `/error` endpoint with `error_rate=0` while
adding log noise via a helper script (TODO: add `log_noise.py`).

Alternatively, SSH into the container and manually `echo` error lines:

```bash
docker exec sample-app sh -c \
  'for i in $(seq 1 200); do
     echo "{\"level\":\"error\",\"message\":\"Simulated DB connection timeout\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}";
     sleep 0.5;
   done'
```

**Expected rule-based alert**: None (no Prometheus rule covers log bursts by default).  
**Expected ML detection**: Spike in `error_kw_sum`, `traceback_count`, `log_burst_z`.  
This scenario is most illustrative of ML's advantage over pure metric rules.

---

## Ground Truth Recording

All scenarios are recorded by `load_generator.py --record-incident` into:

```
ml/data_ingest/incidents.json
```

You can also manually edit this file to add exact timestamps observed in Grafana.

---

## Suggested Experiment Matrix

| Scenario | Duration | Rule fires? | ML detects? | Notes |
|----------|----------|-------------|-------------|-------|
| High error rate | 10 min | Yes (HighErrorRate) | Expected yes | Classic scenario |
| High latency | 10 min | Yes (SlowResponseTime) | Expected yes | Metrics correlated |
| Log error burst | 10 min | No | Expected yes | ML advantage |
| Combined (errors + slow) | 10 min | Yes (both) | Expected yes | Fusion test |
| Low-and-slow (5% errors) | 30 min | Borderline | TBD | Hard case |

---

## THESIS NOTE

Document in your thesis the exact timestamps and reproduce the evaluation using
`evaluate_vs_rules.py`. Discuss cases where the ML model fires early (true
early detection) or late (after the rule), and cases where it produces false
positives during normal load spikes.
