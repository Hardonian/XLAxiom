output "kms_key_id" {
  description = "Resource ID of the SLSA signer Cloud KMS key"
  value       = google_kms_crypto_key.slsa_signer.id
}

output "bigquery_dataset_id" {
  description = "Fully qualified BigQuery telemetry dataset"
  value       = google_bigquery_dataset.xlaxiom_telemetry.id
}

output "evidence_bucket_uri" {
  description = "GCS Evidence Bucket URI"
  value       = google_storage_bucket.evidence_bucket.url
}

output "service_account_email" {
  description = "Service Account email for CI/CD runners"
  value       = google_service_account.xlaxiom_runner.email
}

