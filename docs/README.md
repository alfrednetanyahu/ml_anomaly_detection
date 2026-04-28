# ML-based Anomaly Detection in DevOps Telemetry

> Master's Thesis Project  
> "Machine Learning-based Anomaly Detection in DevOps Telemetry: Design, Implementation, and Comparison with Rule-based Monitoring"

---

## Repository Structure

```
thesis-repo/
├── terraform/              # AWS infrastructure (VPC, EC2)
├── infra/                  # Docker Compose monitoring stack
│   ├── docker-compose.yml
│   ├── prometheus/         # Prometheus config + alert rules
│   ├── loki/               # Loki config
│   ├── promtail/           # Promtail (log shipper) config
│   ├── otel/               # OpenTelemetry Collector config
│   └── grafana/            # Datasources, dashboards
├── app/                    # Sample Flask app + load generator
├── ml/                     # Python ML pipeline
│   ├── data_ingest/        # Scripts to export data from Prometheus/Loki
│   ├── anomaly_detection/  # IsolationForest implementations
│   ├── evaluation/         # Rule-based vs ML comparison
│   └── notebooks/          # Jupyter notebooks
└── docs/                   # This README + EXPERIMENTS.md
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Terraform | ≥ 1.3 |
| AWS CLI | ≥ 2.x (configured with credentials) |
| Docker + Docker Compose | ≥ 24.x |
| Python | ≥ 3.11 |
| An AWS key pair | – |

---

## Quick Start

### 1. Provision AWS Infrastructure

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars  # edit key_pair_name, region
terraform init
terraform apply
```

Note the `instance_public_ip` output.

### 2. SSH into the EC2 Instance

```bash
# Use the ssh_command output from Terraform
ssh -i ~/.ssh/<your-key>.pem ubuntu@<PUBLIC_IP>
```

### 3. Start the Monitoring Stack

```bash
cd /opt/thesis/infra
cp .env.example .env          # edit passwords
docker-compose up -d
```

### 4. Verify Services

| Service | URL |
|---------|-----|
| Sample App | `http://$TH_IP:8000` |
| Prometheus | `http://$TH_IP:9090` |
| Grafana | `http://$TH_IP:3000` (admin/changeme) |
| Loki | `http://$TH_IP:3100` |

### 5. Generate Load & Anomalies

```bash
# From your local machine or the EC2 instance
cd app
pip install requests

# Normal load (5 min)
python load_generator.py --host http://$TH_IP:8000 --mode normal --duration 300

# Anomaly: high error rate (2 min, recorded as incident)
python load_generator.py --host http://$TH_IP:8000 --mode errors --duration 120 --record-incident

# Anomaly: high latency (2 min, recorded as incident)
python load_generator.py --host http://$TH_IP:8000 --mode slow --duration 120 --record-incident
```

### 6. Export Telemetry Data

```bash
cd ml
pip install -r requirements.txt
mkdir -p data

# Export metrics (adjust time window to your experiment)
python data_ingest/fetch_prometheus.py \
  --host http://$TH_IP:9090 \
  --start 2024-06-01T00:00:00Z \
  --end   2024-06-01T12:00:00Z \
  --out   data/metrics.parquet

# Export logs
python data_ingest/fetch_loki.py \
  --host http://$TH_IP:3100 \
  --query '{service="sample-app"}' \
  --start 2024-06-01T00:00:00Z \
  --end   2024-06-01T12:00:00Z \
  --out   data/logs.csv
```

### 7. Run Anomaly Detection

```bash
# Metrics
python anomaly_detection/metrics_isolation_forest.py \
  --data data/metrics.parquet \
  --train-end 2024-06-01T08:00:00Z \
  --out output/metrics_anomalies.csv

# Logs
python anomaly_detection/logs_isolation_forest.py \
  --data data/logs.csv \
  --train-end 2024-06-01T08:00:00Z \
  --out output/logs_anomalies.csv
```

### 8. Evaluate vs Rule-based Alerts

```bash
python evaluation/evaluate_vs_rules.py \
  --incidents data_ingest/incidents.json \
  --metrics-anomalies output/metrics_anomalies.csv \
  --logs-anomalies    output/logs_anomalies.csv \
  --out-dir output/
```

Reports are written to `ml/output/evaluation_report.json` and `.txt`.

---

## Notebooks

```bash
cd ml
jupyter notebook notebooks/
```

- `01_metrics_feature_engineering.ipynb` – EDA, feature engineering, IsolationForest
- `02_logs_analysis.ipynb` – Log feature extraction, windowing, scoring  *(TODO)*
- `03_evaluation_comparison.ipynb` – Full comparison table and charts  *(TODO)*

---

## Teardown

```bash
cd terraform && terraform destroy
```

---

## See Also

- [EXPERIMENTS.md](EXPERIMENTS.md) – Reproducing thesis experiment scenarios
