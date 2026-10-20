# Automated Physical TPU Evidence Collector for XLAxiom
# Billed per second by Google Cloud; typical run is bounded to limit v5e cost.
# Automatically tears down the TPU VM in a finally block to prevent idle costs.

[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Zone = "",
    [string]$AcceleratorType = "v5litepod-4",
    [string]$Version = "tpu-ubuntu2204-base",
    [string]$VmName = "xlaxiom-tpu-eval",
    [ValidateRange(1, 120)]
    [int]$RemoteTimeoutMinutes = 20,
    [switch]$Spot = $false,
    [switch]$KeepVm = $false
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK ('gcloud') is not installed or not in PATH."
    exit 1
}

# On Windows, gcloud's TPU SSH command delegates to bundled PuTTY. Provision the
# existing Compute Engine OpenSSH key directly so native ssh.exe can be used without
# depending on gcloud's SSH side effects or project-wide metadata changes.
$isWindowsPlatform = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
if ($isWindowsPlatform) {
    $sshUser = $env:USERNAME -replace "[^A-Za-z0-9_-]", "_"
    $sshKey = Join-Path $env:USERPROFILE ".ssh\google_compute_engine"
    $sshPublicKeyPath = "$sshKey.pub"
    if (-not (Test-Path $sshKey) -or -not (Test-Path $sshPublicKeyPath)) {
        throw "Compute Engine OpenSSH key pair not found at '$sshKey'."
    }
    $sshPublicKey = (Get-Content $sshPublicKeyPath -Raw).Trim()
    $sshMetadata = "${sshUser}:$sshPublicKey"
}

function Invoke-NativeCommandWithTimeout {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [ValidateRange(1, 7200)]
        [int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    # ProcessStartInfo.ArgumentList preserves every argument boundary exactly. This
    # matters on Windows: PowerShell background jobs serialize string arrays, which
    # can collapse OpenSSH's `-o value` pairs into an invalid single command line.
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Command
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $Arguments) {
        $startInfo.ArgumentList.Add($argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "$Operation failed to start '$Command'."
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            $process.Kill($true)
            $process.WaitForExit()
            throw "$Operation timed out after $TimeoutSeconds seconds."
        }
        $process.WaitForExit()
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        if ($stdout) { Write-Host $stdout.TrimEnd() }
        if ($stderr) { Write-Host $stderr.TrimEnd() }
        if ($process.ExitCode -ne 0) {
            throw "$Operation failed: $Command exited with code $($process.ExitCode)."
        }
    } finally {
        $process.Dispose()
    }
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
$costCeiling = if ($Spot) { "< `$0.46 Spot" } else { "< `$1.90 on-demand" }
Write-Host " Estimated Cost:   $costCeiling (20-minute remote ceiling; destroyed upon completion)"
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
        if ($isWindowsPlatform) {
            $createArgs += "--metadata=ssh-keys=$sshMetadata"
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
rm -rf evidence/tpu
mkdir -p evidence/tpu
python -m xlaxiom.cli doctor > evidence/tpu/doctor.json
python -m xlaxiom.cli environment capture --label "google-cloud-tpu-v5e-viperlite" --output evidence/tpu/environment.json
python -m xlaxiom.cli property run --x64 --backend jax-tpu-eager --output evidence/tpu/tpu-eager-evidence.json
python -m xlaxiom.cli property run --x64 --backend jax-tpu-jit --output evidence/tpu/tpu-jit-evidence.json
python -m xlaxiom.cli property campaign --x64 --cases 50 --backend jax-tpu-jit --output evidence/tpu/tpu-campaign.json
python -m xlaxiom.cli property run --x64 --backend jax-jit --output evidence/tpu/cpu-jit-evidence.json
python -m xlaxiom.cli property compare \
    --baseline evidence/tpu/cpu-jit-evidence.json \
    --comparison evidence/tpu/tpu-jit-evidence.json \
    --output evidence/tpu/tpu-vs-cpu-comparison.json

echo "=== Higher-Order Autodiff ==="
python -m xlaxiom.cli gradient --higher-order --seed 0 --output evidence/tpu/tpu-higher-order.json


echo "=== Evidence Generation Complete ==="
'@

    # Windows gcloud hardcodes a bundled PuTTY client which cannot consume
    # OpenSSH safety options reliably. Use the OS OpenSSH client there.
    if ($isWindowsPlatform) {
        $sshCommand = (Get-Command ssh.exe -ErrorAction Stop).Source
        $externalIp = (& gcloud compute tpus tpu-vm describe $VmName `
                --zone=$activeZone `
                --project=$ProjectId `
                --format="value(networkEndpoints[0].accessConfig.externalIp)").Trim()
        if (-not $externalIp) {
            throw "TPU VM has no external IP for native OpenSSH."
        }
        $sshTarget = "${sshUser}@${externalIp}"
        $sshBaseArgs = @(
            "-T", "-i", $sshKey,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=NUL",
            "-o", "GlobalKnownHostsFile=NUL",
            "-o", "ConnectTimeout=30",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"
        )
        $probeArgs = $sshBaseArgs + @($sshTarget, "true")
        $sshArgs = $sshBaseArgs + @($sshTarget, $remoteScript)
    } else {
        $sshCommand = "gcloud"
        $gcloudSshBaseArgs = @(
            "compute", "tpus", "tpu-vm", "ssh", $VmName,
            "--zone=$activeZone",
            "--project=$ProjectId",
            "--quiet",
            "--strict-host-key-checking=no",
            "--ssh-flag=-o BatchMode=yes",
            "--ssh-flag=-o ConnectTimeout=30",
            "--ssh-flag=-o ConnectionAttempts=3",
            "--ssh-flag=-o ServerAliveInterval=15",
            "--ssh-flag=-o ServerAliveCountMax=4"
        )
        $probeArgs = $gcloudSshBaseArgs + "--command=true"
        $sshArgs = $gcloudSshBaseArgs + "--command=$remoteScript"
    }

    # TPU creation can report READY before sshd and the propagated project key are
    # usable. Probe first so the evidence run does not fail on that short race.
    $sshReady = $false
    for ($attempt = 1; $attempt -le 12; $attempt++) {
        Write-Host "Waiting for TPU SSH readiness (attempt $attempt/12)..."
        try {
            Invoke-NativeCommandWithTimeout `
                -Command $sshCommand `
                -Arguments $probeArgs `
                -TimeoutSeconds 45 `
                -Operation "TPU SSH readiness probe"
            $sshReady = $true
            break
        } catch {
            if ($attempt -eq 12) {
                throw "TPU SSH did not become ready after 12 attempts: $($_.Exception.Message)"
            }
            Start-Sleep -Seconds 15
        }
    }

    if (-not $sshReady) {
        throw "TPU SSH readiness probe ended without a usable connection."
    }

    Invoke-NativeCommandWithTimeout `
        -Command $sshCommand `
        -Arguments $sshArgs `
        -TimeoutSeconds ($RemoteTimeoutMinutes * 60) `
        -Operation "Remote TPU execution"

    # 4. Pull evidence artifacts locally
    Write-Host "`n[4/5] Downloading evidence artifacts to ./evidence/tpu/..." -ForegroundColor Yellow
    $localTarget = Join-Path (Get-Location) "evidence\tpu"
    if (-not (Test-Path $localTarget)) {
        New-Item -ItemType Directory -Path $localTarget -Force | Out-Null
    }

    if ($isWindowsPlatform) {
        $scpCommand = (Get-Command scp.exe -ErrorAction Stop).Source
        $scpArgs = @(
            "-i", $sshKey,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=NUL",
            "-o", "GlobalKnownHostsFile=NUL",
            "-o", "ConnectTimeout=30",
            "${sshTarget}:~/XLAxiom/evidence/tpu/*.json", $localTarget
        )
    } else {
        $scpCommand = "gcloud"
        $scpArgs = @(
            "compute", "tpus", "tpu-vm", "scp", "--recurse",
            "${VmName}:~/XLAxiom/evidence/tpu/*", $localTarget,
            "--zone=$activeZone",
            "--project=$ProjectId",
            "--quiet",
            "--strict-host-key-checking=no",
            "--scp-flag=-o BatchMode=yes",
            "--scp-flag=-o ConnectTimeout=30",
            "--scp-flag=-o ConnectionAttempts=3",
            "--scp-flag=-o ServerAliveInterval=15",
            "--scp-flag=-o ServerAliveCountMax=4"
        )
    }
    Invoke-NativeCommandWithTimeout `
        -Command $scpCommand `
        -Arguments $scpArgs `
        -TimeoutSeconds 300 `
        -Operation "TPU evidence download"
    Write-Host "Evidence artifacts saved to $localTarget" -ForegroundColor Green

} finally {
    # 5. Teardown TPU VM to avoid ongoing charges
    if ($vmCreated -and (-not $KeepVm)) {
        Write-Host "`n[5/5] Tearing down Cloud TPU VM '$VmName' to prevent idle billing..." -ForegroundColor Yellow
        & gcloud compute tpus tpu-vm delete $VmName --zone=$activeZone --project=$ProjectId --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Error "TPU VM deletion failed; delete '$VmName' manually to stop billing."
        }
        Write-Host "Cloud TPU VM successfully destroyed." -ForegroundColor Green
    } elseif ($KeepVm) {
        Write-Host "`nNotice: -KeepVm was passed. Cloud TPU VM '$VmName' is still running." -ForegroundColor Red
        Write-Host "Remember to delete it when done: gcloud compute tpus tpu-vm delete $VmName --zone=$activeZone --project=$ProjectId" -ForegroundColor Red
    }
}
