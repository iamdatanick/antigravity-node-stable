output "ssh_command" {
  value = "ssh ubuntu@${google_compute_instance.pilot.network_interface.0.access_config.0.nat_ip}"
}

output "ui_url" {
  value = "http://${google_compute_instance.pilot.network_interface.0.access_config.0.nat_ip}:1055"
}

output "health_url" {
  value = "http://${google_compute_instance.pilot.network_interface.0.access_config.0.nat_ip}:8080/health"
}

output "ovms_url" {
  value = "http://${google_compute_instance.pilot.network_interface.0.access_config.0.nat_ip}:9001"
}
