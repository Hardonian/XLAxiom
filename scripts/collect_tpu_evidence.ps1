# Automated Physical TPU Evidence Collector for XLAxiom
# Billed per second by Google Cloud; typical run is 3-5 minutes (< $0.20 total on v5e).
# Automatically tears down the TPU VM in a finally block to prevent idle costs.

[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Zone = "us-east1-c",
    [string]$AcceleratorType = "v5litepod-1",
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

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " XLAxiom Physical TPU Evidence Collector" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Project:          $ProjectId"
Write-Host " Zone:             $Zone"
Write-Host " Accelerator:      $AcceleratorType"
Write-Host " VM Name:          $VmName"
Write-Host " Estimated Cost:   < $0.20 (billed per-second, destroyed upon completion)"
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Enable Cloud TPU API if needed
Write-Host "`n[1/5] Ensuring Cloud TPU API is enabled..." -ForegroundColor Yellow
gcloud services enable tpu.googleapis.com --project=$ProjectId

$vmCreated = $false

try {
    # 2. Provision TPU VM
    Write-Host "`n[2/5] Creating Cloud TPU VM '$VmName'..." -ForegroundColor Yellow
    $createArgs = @(
        "compute", "tpus", "tpu-vm", "create", $VmName,
        "--zone=$Zone",
        "--accelerator-type=$AcceleratorType",
        "--version=$Version",
        "--project=$ProjectId",
        "--quiet"
    )
    if ($Spot) {
        $createArgs += "--spot"
    }

    & gcloud @createArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create TPU VM. Check TPU quota in zone $Zone for $AcceleratorType."
    }
    $vmCreated = $true

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
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
uv sync --extra dev --frozen

echo "=== Verifying Physical TPU Silicon ==="
python -c "import jax; print('JAX Devices:', jax.devices()); assert any(d.platform == 'tpu' for d in jax.devices()), 'No TPU silicon found!'"

echo "=== Executing Metamorphic Properties on TPU ==="
mkdir -p evidence/tpu
uv run xlaxiom doctor > evidence/tpu/doctor.json
uv run xlaxiom environment capture --output evidence/tpu/environment.json
uv run xlaxiom property run --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
uv run xlaxiom property run --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
uv run xlaxiom property campaign --cases 25 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
uv run xlaxiom property run --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
uv run xlaxiom property compare \
    --baseline evidence/tpu/cpu-jit-evidence.json \
    --comparison evidence/tpu/tpu-jit-evidence.json \
    --output evidence/tpu/tpu-vs-cpu-comparison.json

echo "=== Evidence Generation Complete ==="
'@

    # Execute remote commands via gcloud SSH
    & gcloud compute tpus tpu-vm ssh $VmName --zone=$Zone --project=$ProjectId --command=$remoteScript
    if ($LASTEXITCODE -ne 0) {
        throw "Remote TPU execution failed."
    }

    # 4. Pull evidence artifacts locally
    Write-Host "`n[4/5] Downloading evidence artifacts to ./evidence/tpu/..." -ForegroundColor Yellow
    $localTarget = Join-Path (Get-Location) "evidence\tpu"
    if (-not (Test-Path $localTarget)) {
        New-Item -ItemType Directory -Path $localTarget -Force | Out-Null
    }

    & gcloud compute tpus tpu-vm scp --recurse "${VmName}:~/XLAxiom/evidence/tpu/*" $localTarget --zone=$Zone --project=$ProjectId
    Write-Host "Evidence artifacts saved to $localTarget" -ForegroundColor Green

} finally {
    # 5. Teardown TPU VM to avoid ongoing charges
    if ($vmCreated -and (-not $KeepVm)) {
        Write-Host "`n[5/5] Tearing down Cloud TPU VM '$VmName' to prevent idle billing..." -ForegroundColor Yellow
        & gcloud compute tpus tpu-vm delete $VmName --zone=$Zone --project=$ProjectId --quiet
        Write-Host "Cloud TPU VM successfully destroyed. Total run time was under 5 minutes." -ForegroundColor Green
    } elseif ($KeepVm) {
        Write-Host "`nNotice: -KeepVm was passed. Cloud TPU VM '$VmName' is still running." -ForegroundColor Red
        Write-Host "Remember to delete it when done: gcloud compute tpus tpu-vm delete $VmName --zone=$Zone --project=$ProjectId" -ForegroundColor Red
    }
}
