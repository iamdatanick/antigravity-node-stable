terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# --- Network ---
resource "google_compute_network" "vpc" {
  name                    = "antigravity-vpc"
  auto_create_subnetworks = true
}

resource "google_compute_firewall" "allow_services" {
  name    = "allow-antigravity-services"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "1055", "4055", "8080", "9001"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["antigravity-node"]
}

# --- Compute Instance ---
# c2-standard-8: 8 vCPU, 32GB RAM, Intel Cascade Lake (AVX-512 guaranteed)
resource "google_compute_instance" "pilot" {
  name         = "antigravity-v14-pilot"
  machine_type = "c2-standard-8"
  zone         = var.zone
  tags         = ["antigravity-node"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 100
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = google_compute_network.vpc.name
    access_config {}
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -e

    echo "--- [Phase 0] System Prep ---"
    apt-get update
    apt-get install -y ca-certificates curl gnupg lsb-release s3cmd

    echo "--- [Phase 0] Installing Docker ---"
    mkdir -m 0755 -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    usermod -aG docker ubuntu

    echo "--- [VG-101] AVX-512 Check ---"
    if grep -q avx512 /proc/cpuinfo; then
        echo "HARDWARE PASS: AVX-512 Detected." > /etc/motd
    else
        echo "HARDWARE FAIL: AVX-512 NOT DETECTED." > /etc/motd
    fi

    echo "--- [CG-104] Creating Persistence Layer ---"
    BASE_DIR="/home/ubuntu/antigravity"
    mkdir -p $BASE_DIR/data/{ceph,ceph_conf,etcd}
    mkdir -p $BASE_DIR/{scripts,src,config,models}
    chown -R ubuntu:ubuntu $BASE_DIR

    echo "--- Bootstrap Complete ---"
  EOT
}
