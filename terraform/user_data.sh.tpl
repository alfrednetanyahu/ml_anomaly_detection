#!/bin/bash
set -euxo pipefail

# ── System update ──────────────────────────────────────────────────────────
apt-get update -y
apt-get upgrade -y

# ── Docker ─────────────────────────────────────────────────────────────────
apt-get install -y ca-certificates curl gnupg lsb-release git python3-pip

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker
usermod -aG docker ubuntu

# ── Docker Compose (standalone v2 alias) ──────────────────────────────────
ln -sf /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose

# ── Node exporter (for Prometheus system metrics) ─────────────────────────
NODE_EXPORTER_VERSION="1.8.1"
cd /tmp
curl -fsSL "https://github.com/prometheus/node_exporter/releases/download/v$${NODE_EXPORTER_VERSION}/node_exporter-$${NODE_EXPORTER_VERSION}.linux-amd64.tar.gz" \
  | tar -xz
mv node_exporter-$${NODE_EXPORTER_VERSION}.linux-amd64/node_exporter /usr/local/bin/
useradd -rs /bin/false node_exporter || true
cat > /etc/systemd/system/node_exporter.service <<'EOF'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now node_exporter

# ── Clone repository ──────────────────────────────────────────────────────
REPO_DIR="/opt/thesis"
git clone ${github_repo_url} "$REPO_DIR" || true
chown -R ubuntu:ubuntu "$REPO_DIR"

# ── Start monitoring stack ────────────────────────────────────────────────
# Uncomment to auto-start on boot:
# cd "$REPO_DIR/infra" && docker-compose up -d

echo "Bootstrap complete. Run: cd /opt/thesis/infra && docker-compose up -d"
