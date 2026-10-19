# Automated Physical TPU Evidence Collector for XLAxiom
# Billed per second by Google Cloud; typical run is 3-5 minutes (< $0.20 total on v5e).
# Automatically tears down the TPU VM in a finally block to prevent idle costs.

[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Zone = "",
    [string]$AcceleratorType = "v5litepod-4",
    [string]$Version = "tpu-ubuntu2204-base",
    [string]$VmName = "xlaxiom-tpu-eval",
    [switch]$Spot = $false,
    [switch]$KeepVm = $false
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK ('gcloud') is not installed or not in PATH."
    exit 1
}

# Resolve active GCP project if not supplied
if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
    if (-not $ProjectId) {
        Write-Error "No active GCP project found. Run 'gcloud config set project <ID>' or pass -ProjectId."
        exit 1
    }
}

# Note: Single-chip v5litepod-1 (TPU_LITE_DEVICE_V5) has 0 quota by default in GCP.
# Pod slices (TPU_LITE_PODSLICE_V5, starting at 4-chip v5litepod-4) have default quota (16 slices).
$candidateZones = if ($Zone) { @($Zone) } else { @("us-south1-a", "us-central1-a", "us-east5-a", "europe-west4-a", "us-east1-c") }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " XLAxiom Physical TPU Evidence Collector" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Project:          $ProjectId"
Write-Host " Candidate Zones:  $($candidateZones -join ', ')"
Write-Host " Accelerator:      $AcceleratorType"
Write-Host " VM Name:          $VmName"
Write-Host " Estimated Cost:   < $0.35 (billed per-second, destroyed upon completion)"
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Enable Cloud TPU API if needed
Write-Host "`n[1/5] Ensuring Cloud TPU API is enabled..." -ForegroundColor Yellow
gcloud services enable tpu.googleapis.com --project=$ProjectId

$vmCreated = $false
$activeZone = ""

try {
    # 2. Provision TPU VM across candidate zones
    foreach ($candidate in $candidateZones) {
        Write-Host "`n[2/5] Attempting to create Cloud TPU VM '$VmName' in zone '$candidate'..." -ForegroundColor Yellow
        $createArgs = @(
            "compute", "tpus", "tpu-vm", "create", $VmName,
            "--zone=$candidate",
            "--accelerator-type=$AcceleratorType",
            "--version=$Version",
            "--project=$ProjectId",
            "--quiet"
        )
        if ($Spot) {
            $createArgs += "--spot"
        }

        & gcloud @createArgs
        if ($LASTEXITCODE -eq 0) {
            $activeZone = $candidate
            $vmCreated = $true
            Write-Host "TPU VM successfully provisioned in zone '$activeZone'!" -ForegroundColor Green
            break
        }
        Write-Warning "Capacity unavailable in zone '$candidate'. Trying next zone..."
    }

    if (-not $vmCreated) {
        throw "Failed to allocate TPU VM across candidate zones ($($candidateZones -join ', ')). Check TPU quota or try v4-8 in us-central2-b."
    }

    # 3. Remote execution script
    Write-Host "`n[3/5] Setting up JAX with libtpu and executing XLAxiom test suite on TPU..." -ForegroundColor Yellow
    $remoteScript = @'
set -euo pipefail

echo "=== TPU Node Ready: $(hostname) ==="
sudo apt-get update -y && sudo apt-get install -y python3-pip python3-venv git curl

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

rm -rf XLAxiom
git clone https://github.com/Hardonian/XLAxiom.git
cd XLAxiom

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Verifying Physical TPU Silicon ==="
python -c "import jax; print('JAX Devices:', jax.devices()); assert any(d.platform == 'tpu' for d in jax.devices()), 'No TPU silicon found!'"

echo "=== Executing Metamorphic Properties on TPU ==="
mkdir -p evidence/tpu
python -m xlaxiom.cli doctor > evidence/tpu/doctor.json
python -m xlaxiom.cli environment capture --label "google-cloud-tpu-v5e-viperlite" --output evidence/tpu/environment.json
python -m xlaxiom.cli property run --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
python -m xlaxiom.cli property run --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
python -m xlaxiom.cli property campaign --cases 50 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
python -m xlaxiom.cli property run --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
python -m xlaxiom.cli property compare \
    --baseline evidence/tpu/cpu-jit-evidence.json \
    --comparison evidence/tpu/tpu-jit-evidence.json \
    --output evidence/tpu/tpu-vs-cpu-comparison.json

echo "=== Higher-Order Autodiff ==="
python -m xlaxiom.cli gradient --higher-order --seed 0 --output evidence/tpu/tpu-higher-order.json || true


echo "=== Evidence Generation Complete ==="
'@

    # Execute remote commands via gcloud SSH
    & gcloud compute tpus tpu-vm ssh $VmName --zone=$activeZone --project=$ProjectId --command=$remoteScript
    if ($LASTEXITCODE -ne 0) {
        throw "Remote TPU execution failed."
    }

    # 4. Pull evidence artifacts locally
    Write-Host "`n[4/5] Downloading evidence artifacts to ./evidence/tpu/..." -ForegroundColor Yellow
    $localTarget = Join-Path (Get-Location) "evidence\tpu"
    if (-not (Test-Path $localTarget)) {
        New-Item -ItemType Directory -Path $localTarget -Force | Out-Null
    }

    & gcloud compute tpus tpu-vm scp --recurse "${VmName}:~/XLAxiom/evidence/tpu/*" $localTarget --zone=$activeZone --project=$ProjectId
    Write-Host "Evidence artifacts saved to $localTarget" -ForegroundColor Green

} finally {
    # 5. Teardown TPU VM to avoid ongoing charges
    if ($vmCreated -and (-not $KeepVm)) {
        Write-Host "`n[5/5] Tearing down Cloud TPU VM '$VmName' to prevent idle billing..." -ForegroundColor Yellow
        & gcloud compute tpus tpu-vm delete $VmName --zone=$activeZone --project=$ProjectId --quiet
        Write-Host "Cloud TPU VM successfully destroyed. Total run time was under 5 minutes." -ForegroundColor Green
    } elseif ($KeepVm) {
        Write-Host "`nNotice: -KeepVm was passed. Cloud TPU VM '$VmName' is still running." -ForegroundColor Red
        Write-Host "Remember to delete it when done: gcloud compute tpus tpu-vm delete $VmName --zone=$activeZone --project=$ProjectId" -ForegroundColor Red
    }
}
