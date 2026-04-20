output "instance_public_ip" {
  description = "Public IP of the EC2 monitoring node"
  value       = aws_instance.monitoring.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.monitoring.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_instance.monitoring.public_ip}"
}

output "grafana_url" {
  description = "Grafana dashboard URL"
  value       = "http://${aws_instance.monitoring.public_ip}:3000"
}

output "prometheus_url" {
  description = "Prometheus UI URL"
  value       = "http://${aws_instance.monitoring.public_ip}:9090"
}

output "app_url" {
  description = "Sample application URL"
  value       = "http://${aws_instance.monitoring.public_ip}:8000"
}
