variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
  default     = "agentic1111"
}

variable "region" {
  description = "GCP region (must have c2 instances for AVX-512)"
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for compute instance"
  default     = "us-central1-a"
}

variable "admin_cidr" {
  description = "CIDR range for internal service access (API, debug, OVMS). Set to your VPN or office IP."
  type        = string
}
