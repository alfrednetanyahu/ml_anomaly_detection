# ML Anomaly Detection in DevOps Telemetry – Thesis Repository

See [docs/README.md](docs/README.md) for full setup instructions.

## One-line Summary

```
terraform apply → docker-compose up → python ml pipeline → evaluation report
```

## Tech Stack

- **Infrastructure**: AWS EC2 (Terraform)
- **Metrics**: Prometheus + Node Exporter
- **Logs**: Loki + Promtail
- **Telemetry routing**: OpenTelemetry Collector
- **Visualisation**: Grafana
- **Sample app**: Python/Flask
- **ML**: scikit-learn IsolationForest (Python)
