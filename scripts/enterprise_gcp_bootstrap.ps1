<#
.SYNOPSIS
    XLAxiom Enterprise GCP Bootstrap Script for PowerShell.
.DESCRIPTION
    Provisions Cloud KMS KeyRing, BigQuery Telemetry Dataset, GCS Evidence Bucket, and Looker Studio views.
#>

[CmdletBinding()]
param (
    [string]$ProjectId = $(gcloud config get-value project 2>$null),
    [string]$Location = "global",
    [string]$BqLocation = "US"
)

if (-not $ProjectId) {
    Write-Error "GCP Project ID is required. Pass -ProjectId or set active gcloud project."
    exit 1
}

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host " XLAxiom Enterprise GCP Bootstrapper (PowerShell)" -ForegroundColor Cyan
Write-Host " Project: $ProjectId | KMS Location: $Location | BQ: $BqLocation" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

# 1. Enable Services
Write-Host ">>> Enabling required GCP services..." -ForegroundColor Yellow
gcloud services enable `
    tpu.googleapis.com `
    cloudkms.googleapis.com `
    bigquery.googleapis.com `
    storage.googleapis.com `
    aiplatform.googleapis.com `
    logging.googleapis.com `
    --project=$ProjectId --quiet

# 2. Provision KMS Keyring and CryptoKey
Write-Host ">>> Provisioning Cloud KMS Keyring and CryptoKey..." -ForegroundColor Yellow
$null = gcloud kms keyrings describe xlaxiom-keyring --location=$Location --project=$ProjectId 2>&1
if ($LASTEXITCODE -ne 0) {
    gcloud kms keyrings create xlaxiom-keyring --location=$Location --project=$ProjectId
}

$null = gcloud kms keys describe slsa-signer --keyring=xlaxiom-keyring --location=$Location --project=$ProjectId 2>&1
if ($LASTEXITCODE -ne 0) {
    gcloud kms keys create slsa-signer `
        --keyring=xlaxiom-keyring `
        --location=$Location `
        --purpose=asymmetric-signing `
        --default-algorithm=rsa-sign-pss-3072-sha256 `
        --protection-level=hsm `
        --project=$ProjectId
}

# 3. Create GCS Evidence Bucket
$bucketName = "xlaxiom-evidence-$ProjectId"
Write-Host ">>> Ensuring GCS bucket gs://$bucketName exists..." -ForegroundColor Yellow
$null = gcloud storage buckets describe "gs://$bucketName" 2>&1
if ($LASTEXITCODE -ne 0) {
    gcloud storage buckets create "gs://$bucketName" --location=$BqLocation --project=$ProjectId --uniform-bucket-level-access
}

# 4. Create BigQuery Telemetry Dataset
Write-Host ">>> Provisioning BigQuery telemetry dataset..." -ForegroundColor Yellow
bq --location=$BqLocation mk --dataset --project_id=$ProjectId --description "XLAxiom Telemetry" xlaxiom_telemetry 2>$null

# 5. Deploy Looker Studio & Analytics Views
Write-Host ">>> Deploying Looker Studio analytical SQL views..." -ForegroundColor Yellow
$viewsSql = Get-Content "sql/bigquery_views.sql" -Raw
$resolvedSql = $viewsSql.Replace("{PROJECT_ID}", $ProjectId)
$tempSql = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tempSql -Value $resolvedSql -Encoding UTF8
bq query --use_legacy_sql=false --project_id=$ProjectId < $tempSql 2>$null
Remove-Item $tempSql -Force -ErrorAction SilentlyContinue

Write-Host "`nXLAxiom Enterprise GCP infrastructure successfully bootstrapped!" -ForegroundColor Green

