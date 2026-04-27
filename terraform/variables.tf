variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "eu-north-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "Name of an existing AWS key pair for SSH access"
  type        = string
  default     = "secscla-key"
}

variable "public_key_path" {
  description = "Path to the local SSH public key file to import into AWS"
  type        = string
  default     = "~/.ssh/secsla-key.pub"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "thesis-anomaly"
}

variable "github_repo_url" {
  description = "GitHub URL of this repository (for cloning in user data)"
  type        = string
variable "deploy_key_private" {
  description = "Private SSH deploy key for cloning the GitHub repository"
  type        = string
  sensitive   = true
}

variable "allowed_cidr" {
  description = "CIDR block allowed to access EC2 (e.g., your IP). Use 0.0.0.0/0 for open (not recommended)."
  type        = string
  default     = "0.0.0.0/0"
}
