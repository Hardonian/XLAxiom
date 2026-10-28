<#
.SYNOPSIS
    XLAxiom Multi-Generation Cloud TPU Fleet Evidence Collection Automation for Windows/PowerShell.
.DESCRIPTION
    Automates provisioning, execution, artifact retrieval, and strict teardown across TPU v4, v5e, v5p, and v6e (Trillium).
#>

[CmdletBinding()]
param (
    [string]$ProjectId = $(gcloud config get-value project 2>$null)
)

if (-not $ProjectId) {
    Write-Error "No GCP Project ID detected. Set via 'gcloud config set project <ID>' or pass -ProjectId."
    exit 1
}

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host " XLAxiom Multi-Generation Cloud TPU Fleet Evidence Collector" -ForegroundColor Cyan
Write-Host " Target GCP Project: $ProjectId" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

$fleetConfigs = @(
    @{ Gen = "v4";  Accel = "v4-8";         Zone = "us-central2-b"; OutDir = "evidence/tpu-v4" },
    @{ Gen = "v5e"; Accel = "v5litepod-4";  Zone = "us-south1-a";   OutDir = "evidence/tpu-v5e" },
    @{ Gen = "v5p"; Accel = "v5p-8";        Zone = "us-east5-a";    OutDir = "evidence/tpu-v5p" },
    @{ Gen = "v6e"; Accel = "v6e-4";        Zone = "us-central1-ba"; OutDir = "evidence/tpu-v6e" }
)

# Enable Cloud TPU API
gcloud services enable tpu.googleapis.com --project="$ProjectId" --quiet

foreach ($item in $fleetConfigs) {
    $gen = $item.Gen
    $accel = $item.Accel
    $zone = $item.Zone
    $outDir = $item.OutDir
    $vmName = "xlaxiom-$gen-collector"

    Write-Host "`n>>> Processing Generation: $gen ($accel) in $zone..." -ForegroundColor Yellow
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    }

    $vmCreated = $false
    try {
        Write-Host "Provisioning Cloud TPU VM '$vmName'..."
        $createOut = gcloud compute tpus tpu-vm create $vmName `
            --zone=$zone `
            --accelerator-type=$accel `
            --version="tpu-ubuntu2204-base" `
            --project=$ProjectId `
            --quiet 2>&1

        if ($LASTEXITCODE -eq 0) {
            $vmCreated = $true
            Write-Host "Executing XLAxiom test suite on remote TPU VM..."
            $remoteCmd = @"
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="`$HOME/.local/bin:`$PATH"
git clone https://github.com/Hardonian/XLAxiom.git ~/XLAxiom
cd ~/XLAxiom
uv sync --extra dev
uv run python -m xlaxiom.cli cloud tpu-run --accelerator $accel --cases 50 --output ~/results.json
"@
            gcloud compute tpus tpu-vm ssh $vmName --zone=$zone --project=$ProjectId --command=$remoteCmd

            Write-Host "Downloading verification artifacts..."
            gcloud compute tpus tpu-vm scp "${vmName}:~/results.json" "$outDir/tpu-results.json" --zone=$zone --project=$ProjectId
            Write-Host "Evidence collected for $gen." -ForegroundColor Green
        } else {
            Write-Warning "Could not provision $accel in $zone (quota or capacity unavailable). Skipping."
        }
    }
    finally {
        if ($vmCreated) {
            Write-Host "Tearing down Cloud TPU VM '$vmName' in $zone to prevent idle billing..." -ForegroundColor Magenta
            gcloud compute tpus tpu-vm delete $vmName --zone=$zone --project=$ProjectId --quiet
        }
    }
}

Write-Host "`nMulti-generation TPU evidence collection routine complete." -ForegroundColor Green

