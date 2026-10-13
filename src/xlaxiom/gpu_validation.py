"""Bounded GPU validation for XLAxiom on real, explicitly selected CUDA devices.

Every path here is capability-gated and honest:

* A requested accelerator that cannot initialize is reported as UNSUPPORTED or
  BLOCKED. It is never silently replaced by a CPU device.
* Device placement is verified from the output array's own ``.device`` attribute
  after ``block_until_ready()``, not from the environment variable that was set.
* All allocations are bounded; there is no unbounded allocation or benchmark sweep.
* At most one test worker runs per GPU.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from .hardware import (
    CudaVariant,
    DeviceState,
    PhysicalGpu,
    environment_manifest,
    idle_assessment,
    is_idle,
    select_variant,
)

# The host may export LD_LIBRARY_PATH pointing at an older system CUDA toolkit. That
# shadows the cuSPARSE/cuBLAS shipped inside the JAX CUDA wheels and makes the plugin
# fail with "Unable to load cuSPARSE". The fix is per-process and never touches the host.
SHADOWING_LIBRARY_PATH = "LD_LIBRARY_PATH"
# JAX preallocates most device memory by default. Validation and bounded campaigns must
# coexist with the host's own workloads, so every XLAxiom CUDA child opts out.
JAX_GPU_PREALLOCATE = "XLA_PYTHON_CLIENT_PREALLOCATE"

PROBE_SHAPE = (32, 32)
PROBE_ELEMENTS = PROBE_SHAPE[0] * PROBE_SHAPE[1]

VARIANT_EXTRA: dict[CudaVariant, str] = {
    CudaVariant.CUDA12: "cuda12",
    CudaVariant.CUDA13: "cuda13",
}


def gpu_environment_dir() -> str:
    """Return the host-local directory that contains isolated CUDA probe environments.

    ``XLAXIOM_GPU_ENV_DIR`` is an explicit operator override. Otherwise use the
    platform's per-user cache location rather than a path tied to one developer's
    home directory.
    """
    configured = os.environ.get("XLAXIOM_GPU_ENV_DIR")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    home = os.path.expanduser("~")
    system = platform.system()
    if system == "Windows":
        cache_root = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
    elif system == "Darwin":
        cache_root = os.path.join(home, "Library", "Caches")
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache"))
    return os.path.join(os.path.expanduser(cache_root), "xlaxiom-gpu")


def variant_python(variant: CudaVariant) -> str:
    """Return the isolated interpreter for a CUDA wheel variant on this platform."""
    interpreter = ("Scripts", "python.exe") if platform.system() == "Windows" else ("bin", "python")
    return os.path.join(gpu_environment_dir(), f"{variant.value}-probe", *interpreter)


@dataclass
class ProbeCheck:
    """One verifiable step of the GPU validation sequence."""

    name: str
    status: str
    detail: str = ""
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceValidation:
    """The complete, honest validation record for one device."""

    device: dict[str, Any]
    variant: str | None
    state: str
    reason: str
    checks: list[ProbeCheck] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "variant": self.variant,
            "state": self.state,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
            "environment": self.environment,
            "measurements": self.measurements,
            "diagnostics": self.diagnostics,
        }


def clean_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a CUDA child environment isolated from host library and memory defaults."""
    env = dict(os.environ)
    if SHADOWING_LIBRARY_PATH in env:
        # Recorded, not silenced: the reason is captured in the validation report.
        env.pop(SHADOWING_LIBRARY_PATH, None)
    if extra:
        env.update(extra)
    env[JAX_GPU_PREALLOCATE] = "false"
    return env


def probe_source() -> str:
    """The bounded probe program, executed in an isolated interpreter.

    Kept as source so it can be audited and reused verbatim in the report.
    """
    return f'''
"""Bounded GPU probe: eager + JIT, explicit placement, no CPU fallback."""
import json
import sys

import jax
import jax.numpy as jnp

result = {{
    "jax": jax.__version__,
    "jax_devices": [str(device) for device in jax.devices()],
    "checks": [],
}}


def record(name, ok, detail="", device=None):
    result["checks"].append({{"name": name, "status": "pass" if ok else "fail",
                              "detail": detail, "device": device}})
    return ok


try:
    import jaxlib

    result["jaxlib"] = jaxlib.__version__
except ImportError:
    result["jaxlib"] = None

# 1. GPU devices must actually be present; a CPU fallback is a hard failure.
try:
    gpu_devices = jax.devices("gpu")
    record("jax_devices_gpu", bool(gpu_devices), str(gpu_devices))
except Exception as exc:
    record("jax_devices_gpu", False, repr(exc))
    print(json.dumps(result, sort_keys=True))
    sys.exit(0)

if not gpu_devices:
    print(json.dumps(result, sort_keys=True))
    sys.exit(0)

device = gpu_devices[0]
result["selected_device"] = str(device)

# 2. Bounded eager execution with explicit placement.
try:
    with jax.default_device(device):
        values = (jnp.arange({PROBE_ELEMENTS}, dtype=jnp.float32).reshape({PROBE_SHAPE}) + 1.0)
        values = values.block_until_ready()
    actual = str(values.device)
    record("eager_execution", values.device.platform == "gpu", f"shape={{values.shape}}", actual)
    result["eager_device"] = actual
    result["eager_checksum"] = float(jnp.asarray(values).sum())
except Exception as exc:
    record("eager_execution", False, repr(exc))

# 3. Bounded JIT execution, synchronized, with placement verified from the output.
try:
    def program(matrix):
        return (matrix @ matrix).sum()

    with jax.default_device(device):
        compiled = jax.jit(program)
        placed = jax.device_put(values, device)
        output = compiled(placed).block_until_ready()
    actual = str(output.device)
    record("jit_execution", output.device.platform == "gpu", f"value={{float(output)}}", actual)
    result["jit_device"] = actual
    result["jit_value"] = float(output)
except Exception as exc:
    record("jit_execution", False, repr(exc))

# 4. Explicit placement of a fresh array onto the selected device.
try:
    with jax.default_device(device):
        placed = jax.device_put(jnp.ones({PROBE_SHAPE}, dtype=jnp.float32), device)
    placed = placed.block_until_ready()
    record("explicit_placement", str(placed.device) == str(device), str(placed.device))
except Exception as exc:
    record("explicit_placement", False, repr(exc))

print(json.dumps(result, sort_keys=True))
'''


def _python_for(variant: CudaVariant) -> str:
    return variant_python(variant)


def environment_exists(variant: CudaVariant) -> bool:
    return os.path.isfile(_python_for(variant))


def run_probe(
    device: PhysicalGpu,
    variant: CudaVariant,
    ordinal: int,
    timeout_s: float = 180.0,
) -> DeviceValidation:
    """Run the bounded probe against one device in one isolated environment.

    The device is selected through ``CUDA_VISIBLE_DEVICES`` in CUDA's own ordinal
    space, so the host driver and every other GPU workload are untouched.
    """
    validation = DeviceValidation(
        device=device.to_dict(),
        variant=variant.value,
        state=DeviceState.INCONCLUSIVE.value,
        reason="",
        environment={"ordinal": ordinal, "variant": variant.value},
    )
    if not environment_exists(variant):
        validation.state = DeviceState.BLOCKED.value
        validation.reason = (
            f"isolated environment for {variant.value} is not installed at {_python_for(variant)}"
        )
        return validation
    environment = clean_environment({"CUDA_VISIBLE_DEVICES": str(ordinal)})
    command = [_python_for(variant), "-c", probe_source()]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=environment,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except subprocess.TimeoutExpired:
        validation.state = DeviceState.BLOCKED.value
        validation.reason = f"probe exceeded {timeout_s}s"
        return validation
    validation.environment.update(
        {
            "command": "python -c <bounded-probe>",
            "cuda_visible_devices": str(ordinal),
            "ld_library_path_removed": SHADOWING_LIBRARY_PATH in os.environ,
            "exit_code": completed.returncode,
        }
    )
    payload: dict[str, Any] = {}
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                import json

                payload = json.loads(line)
            except ValueError:
                continue
            break
    if not payload:
        validation.state = DeviceState.BLOCKED.value
        validation.reason = "probe produced no parsable observation"
        validation.diagnostics = {
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
        return validation
    validation.checks = [
        ProbeCheck(
            name=entry["name"],
            status="pass" if entry["status"] == "pass" else "fail",
            detail=str(entry.get("detail", "")),
            device=entry.get("device"),
        )
        for entry in payload.get("checks", [])
    ]
    validation.environment.update(
        {
            "jax": payload.get("jax"),
            "jaxlib": payload.get("jaxlib"),
            "jax_devices": payload.get("jax_devices", []),
            "selected_device": payload.get("selected_device"),
        }
    )
    if payload.get("jit_value") is not None:
        validation.measurements = {
            "eager_checksum": payload.get("eager_checksum"),
            "jit_value": payload.get("jit_value"),
            "eager_device": payload.get("eager_device"),
            "jit_device": payload.get("jit_device"),
        }
    failed = [check for check in validation.checks if check.status != "pass"]
    if completed.returncode != 0 and not payload.get("checks"):
        validation.state = DeviceState.BLOCKED.value
        validation.reason = f"probe exited {completed.returncode} before observing devices"
        validation.diagnostics = {"stderr": completed.stderr[-2000:]}
        return validation
    if not validation.checks:
        validation.state = DeviceState.UNSUPPORTED.value
        validation.reason = "no GPU device was visible to the isolated runtime"
        validation.diagnostics = {"stderr": completed.stderr[-2000:]}
        return validation
    if failed:
        first = failed[0]
        validation.state = DeviceState.UNSUPPORTED.value
        validation.reason = f"{first.name} failed: {first.detail}"
        validation.diagnostics = {
            "failed_checks": [check.name for check in failed],
            "stderr": completed.stderr[-2000:],
        }
        return validation
    validation.state = DeviceState.PASS.value
    validation.reason = (
        f"{len(validation.checks)} bounded checks passed on {device.name} "
        f"under jax[{variant.value}]"
    )
    return validation


def validate_device(
    device: PhysicalGpu,
    timeout_s: float = 180.0,
    allow_busy: bool = False,
    max_used_mib: int = 256,
) -> DeviceValidation:
    """Validate one physical device: capability gate, then idleness gate, then probe.

    ``allow_busy`` is an explicit operator override. It is recorded in the evidence and
    still requires ample free VRAM, so a bounded probe can never evict an external
    workload. The default is deliberately strict.
    """
    variant, reason = select_variant(device)
    if variant is None:
        return DeviceValidation(
            device=device.to_dict(),
            variant=None,
            state=DeviceState.UNSUPPORTED.value,
            reason=reason,
            environment={"cuda12_supported": False, "cuda13_supported": False},
        )
    assessment = idle_assessment(device)
    idle = is_idle(device, max_used_mib=max_used_mib)
    if not idle and not allow_busy:
        return DeviceValidation(
            device=device.to_dict(),
            variant=variant.value,
            state=DeviceState.BLOCKED.value,
            reason=(
                "device holds an external workload "
                f"({', '.join(device.active_processes) or 'processes present'}); "
                f"{assessment['used_vram_mib']}MiB in use, "
                f"{assessment['utilization_percent']}% utilization"
            ),
            environment={"idleness": assessment, "selection_reason": reason},
        )
    if not idle:
        # Explicit override path: demand clearly available idle capacity.
        if device.free_vram_mib < 2048 or device.utilization_percent > 5.0:
            return DeviceValidation(
                device=device.to_dict(),
                variant=variant.value,
                state=DeviceState.BLOCKED.value,
                reason=(
                    "operator override declined: insufficient idle capacity "
                    f"({device.free_vram_mib}MiB free, "
                    f"{device.utilization_percent}% utilization)"
                ),
                environment={"idleness": assessment, "selection_reason": reason},
            )
        assessment = {**assessment, "operator_override": True}
    validation = run_probe(device, variant, device.ordinal, timeout_s)
    validation.environment["selection_reason"] = reason
    validation.environment["idleness"] = assessment
    return validation


def validate_all(
    devices: list[PhysicalGpu],
    timeout_s: float = 180.0,
    allow_busy: bool = False,
    max_used_mib: int = 256,
) -> dict[str, Any]:
    """Validate every visible device and summarise the honest aggregate state."""
    results = [
        validate_device(device, timeout_s, allow_busy=allow_busy, max_used_mib=max_used_mib)
        for device in devices
    ]
    states = [result.state for result in results]
    return {
        "report": "XLAxiom EPYC CUDA hardware validation",
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "environment": environment_manifest(),
        "device_count": len(results),
        "results": [result.to_dict() for result in results],
        "summary": {
            "pass": states.count(DeviceState.PASS.value),
            "unsupported": states.count(DeviceState.UNSUPPORTED.value),
            "blocked": states.count(DeviceState.BLOCKED.value),
            "inconclusive": states.count(DeviceState.INCONCLUSIVE.value),
        },
        "notes": [
            "No CPU fallback was used or reported as a GPU result.",
            "CUDA runtime ordinal order may differ from nvidia-smi index order; both are recorded.",
            "Host drivers, CUDA installations, and services were not modified.",
        ],
    }


def failure_summary(validation: DeviceValidation) -> str:
    """A compact, portable diagnostic string for a non-passing device."""
    if validation.state == DeviceState.PASS.value:
        return f"PASS: {validation.reason}"
    parts = [f"{validation.state}: {validation.reason}"]
    for check in validation.checks:
        if check.status != "pass":
            parts.append(f"  - {check.name}: {check.detail}")
    diagnostics = validation.diagnostics
    if diagnostics.get("stderr"):
        tail = " ".join(diagnostics["stderr"].split())[:400]
        parts.append(f"  - stderr: {tail}")
    return "\n".join(parts)
