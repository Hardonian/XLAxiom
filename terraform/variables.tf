variable "project_id" {
  description = "The Google Cloud Platform Project ID"
  type        = string
}

variable "region" {
  description = "The GCP region for regional resources"
  type        = string
  default     = "us-central1"
}

variable "location" {
  description = "Location for Cloud KMS and BigQuery (global or multi-region)"
  type        = string
  default     = "global"
}

variable "dataset_location" {
  description = "BigQuery dataset location"
  type        = string
  default     = "US"
}

