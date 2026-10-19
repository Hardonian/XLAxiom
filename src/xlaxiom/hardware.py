"""Real CUDA/GPU hardware discovery, capability gating, and isolated GPU validation.

This module is deliberately capability-only. It never mutates the host driver, never
changes a system CUDA installation, and never silently substitutes a CPU device for a
requested accelerator. A device that is absent, busy, or incompatible is reported as
UNSUPPORTED or BLOCKED; it is never reported as a CPU result.

Ordinal space warning: the CUDA runtime ordinal order is not guaranteed to match the
``nvidia-smi`` / NVML index order. Every selection path in this module therefore goes
through the CUDA runtime's own ordinal space, and the recorded evidence always carries
both the CUDA ordinal and the physical device name so the two can be reconciled.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

# Per official JAX installation documentation (https://docs.jax.dev/en/latest/installation.html):
#   - CUDA 12 wheels support NVIDIA GPUs with SM version 5.2 (Maxwell) or newer.
#   - CUDA 13 wheels support NVIDIA GPUs with SM version 7.5 or newer.
#   - Driver must be >= 525 for CUDA 12 on Linux and >= 580 for CUDA 13 on Linux.
CUDA12_MIN_SM = (5, 2)
CUDA13_MIN_SM = (7, 5)
CUDA12_MIN_DRIVER = 525
CUDA13_MIN_DRIVER = 580


class DeviceState(StrEnum):
    PASS = "PASS"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CudaVariant(StrEnum):
    CUDA12 = "cuda12"
    CUDA13 = "cuda13"


def _sm_tuple(text: str) -> tuple[int, int]:
    parts = text.split(".")
    return (int(parts[0]), int(parts[1])) if len(parts) >= 2 else (int(parts[0]), 0)


def redact_uuid(raw: str) -> str:
    """Return a stable, non-sensitive suffix for a GPU UUID.

    The raw UUID is never recorded. A salted digest keeps device identity comparable
    across runs on the same machine without exposing the hardware identifier.
    """
    salt = os.environ.get("XLAXIOM_DEVICE_SALT", "xlaxiom-local-evidence")
    digest = hashlib.sha256(f"{salt}:{raw}".encode()).hexdigest()
    return f"dev-{digest[:12]}"


def _nvidia_smi_csv(query: str) -> list[list[str]]:
    """Query nvidia-smi without letting LD_LIBRARY_PATH break it."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.strip().splitlines():
        if line.strip():
            rows.append([cell.strip() for cell in line.split(",")])
    return rows


def _nvidia_smi_fields(fields: str) -> list[dict[str, str]]:
    """Query nvidia-smi and return one dict per GPU, keyed by requested field name."""
    rows = _nvidia_smi_csv(fields)
    names = [name.strip() for name in fields.split(",")]
    records: list[dict[str, str]] = []
    for row in rows:
        if len(row) != len(names):
            continue
        records.append(dict(zip(names, row, strict=True)))
    return records


@dataclass(frozen=True)
class PhysicalGpu:
    """One physical GPU as reported by the driver, keyed by CUDA ordinal."""

    ordinal: int
    name: str
    uuid_redacted: str
    compute_capability: str
    sm: tuple[int, int]
    driver_version: str
    cuda_version: str
    total_vram_mib: int
    used_vram_mib: int
    free_vram_mib: int
    utilization_percent: float
    active_processes: tuple[str, ...] = ()
    smi_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sm"] = list(self.sm)
        return data


def _visible_gpu_name(ordinal_names: list[str]) -> str | None:
    """The single device name the runtime exposes when exactly one GPU is visible.

    When ``CUDA_VISIBLE_DEVICES`` pins one GPU the runtime reports one name, so the
    NVML listing cannot be reordered by it. Matching that name is the only reliable way
    to attribute evidence to the correct physical device.
    """
    if len(ordinal_names) == 1:
        return ordinal_names[0]
    return None


def enumerate_physical_gpus() -> list[PhysicalGpu]:
    """Report every visible GPU, keyed by CUDA ordinal, with redacted identity.

    ``nvidia-smi`` reports NVML index order. The CUDA runtime may enumerate in a
    different order, so the ordinal is resolved by matching the device name that the
    CUDA runtime reports against the NVML listing. When the runtime order cannot be
    determined, the NVML index is used and the evidence records that fact.
    """
    fields = "index,name,uuid,compute_cap,memory.total,memory.used,memory.free,utilization.gpu"
    records = _nvidia_smi_fields(fields)
    if not records:
        return []
    driver_version = _driver_version()
    cuda_version = _driver_cuda_version()
    processes = _gpu_processes()
    devices: list[PhysicalGpu] = []
    for record in records:
        try:
            smi_index = int(record["index"])
            compute_cap = record["compute_cap"]
            device = PhysicalGpu(
                ordinal=-1,
                name=record["name"],
                uuid_redacted=redact_uuid(record["uuid"]),
                compute_capability=compute_cap,
                sm=_sm_tuple(compute_cap),
                driver_version=driver_version,
                cuda_version=cuda_version,
                total_vram_mib=int(float(record["memory.total"])),
                used_vram_mib=int(float(record["memory.used"])),
                free_vram_mib=int(float(record["memory.free"])),
                utilization_percent=float(record["utilization.gpu"]),
                active_processes=tuple(processes.get(record["uuid"], ())),
                smi_index=smi_index,
            )
        except (ValueError, KeyError):
            continue
        devices.append(device)
    devices.sort(key=lambda item: item.smi_index or 0)
    ordinal_names = _cuda_runtime_names()
    visible_name = _visible_gpu_name(ordinal_names)
    result: list[PhysicalGpu] = []
    if visible_name is not None:
        # Only one device is exposed to this process (CUDA_VISIBLE_DEVICES). Match it by
        # the name the runtime reports so the evidence attributes the right physical GPU.
        match = next(
            (
                device
                for device in devices
                if device.name.strip().lower() == visible_name.strip().lower()
            ),
            None,
        )
        if match is not None:
            result = [
                PhysicalGpu(
                    ordinal=0,
                    name=match.name,
                    uuid_redacted=match.uuid_redacted,
                    compute_capability=match.compute_capability,
                    sm=match.sm,
                    driver_version=match.driver_version,
                    cuda_version=match.cuda_version,
                    total_vram_mib=match.total_vram_mib,
                    used_vram_mib=match.used_vram_mib,
                    free_vram_mib=match.free_vram_mib,
                    utilization_percent=match.utilization_percent,
                    active_processes=match.active_processes,
                    smi_index=match.smi_index,
                )
            ]
            return result
    if ordinal_names and len(ordinal_names) == len(devices):
        remaining = list(devices)
        for ordinal, cuda_name in enumerate(ordinal_names):
            index = next(
                (
                    position
                    for position, device in enumerate(remaining)
                    if device.name.strip().lower() == cuda_name.strip().lower()
                ),
                None,
            )
            if index is None:
                continue
            device = remaining.pop(index)
            result.append(
                PhysicalGpu(
                    ordinal=ordinal,
                    name=device.name,
                    uuid_redacted=device.uuid_redacted,
                    compute_capability=device.compute_capability,
                    sm=device.sm,
                    driver_version=device.driver_version,
                    cuda_version=device.cuda_version,
                    total_vram_mib=device.total_vram_mib,
                    used_vram_mib=device.used_vram_mib,
                    free_vram_mib=device.free_vram_mib,
                    utilization_percent=device.utilization_percent,
                    active_processes=device.active_processes,
                    smi_index=device.smi_index,
                )
            )
        return result
    # The runtime order is unknown: keep NVML order and say so explicitly.
    return [
        PhysicalGpu(
            ordinal=device.smi_index if device.smi_index is not None else position,
            name=device.name,
            uuid_redacted=device.uuid_redacted,
            compute_capability=device.compute_capability,
            sm=device.sm,
            driver_version=device.driver_version,
            cuda_version=device.cuda_version,
            total_vram_mib=device.total_vram_mib,
            used_vram_mib=device.used_vram_mib,
            free_vram_mib=device.free_vram_mib,
            utilization_percent=device.utilization_percent,
            active_processes=device.active_processes,
            smi_index=device.smi_index,
        )
        for position, device in enumerate(devices)
    ]


def _driver_version() -> str:
    records = _nvidia_smi_fields("driver_version")
    return records[0]["driver_version"] if records else ""


def _driver_cuda_version() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [executable], capture_output=True, text=True, timeout=30, env=dict(os.environ)
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in completed.stdout.splitlines():
        if "CUDA Version" in line:
            return line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
    return ""


def _gpu_processes() -> dict[str, tuple[str, ...]]:
    """Map a GPU UUID to a redacted description of the processes holding it."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ),
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    mapping: dict[str, list[str]] = {}
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 3:
            continue
        raw_uuid, pid, name = cells[0], cells[1], cells[2]
        # Keep only the executable leaf and drop any arguments so no command line,
        # trace identifier, or session detail can leak into portable evidence.
        leaf = name.strip().split()[0] if name.strip() else "unknown"
        leaf = leaf.rsplit("/", 1)[-1] if "/" in leaf else leaf
        mapping.setdefault(raw_uuid, []).append(f"pid{pid}:{leaf}")
    return {key: tuple(value) for key, value in mapping.items()}


def _cuda_runtime_names() -> list[str]:
    """Device names in CUDA runtime ordinal order, via the cudart JAX will bind to."""
    import ctypes
    import glob

    patterns = [
        "/home/scott/.cache/xlaxiom-gpu/*/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so*",
        "/usr/local/cuda/lib64/libcudart.so*",
        "/usr/local/cuda-*/lib64/libcudart.so*",
    ]
    for pattern in patterns:
        for library in sorted(glob.glob(pattern)):
            try:
                handle = ctypes.CDLL(library)
                handle.cudaGetDeviceCount.restype = ctypes.c_int
                count = ctypes.c_int(0)
                if handle.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value <= 0:
                    continue
                names: list[str] = []
                for index in range(count.value):
                    # cudaDeviceProp begins with `char name[256]`.
                    buffer = ctypes.create_string_buffer(65536)
                    if handle.cudaGetDeviceProperties(buffer, index) != 0:
                        names = []
                        break
                    names.append(buffer.value.split(b"\x00")[0].decode("utf-8", "replace"))
                if len(names) == count.value:
                    return names
            except (OSError, AttributeError):
                continue
    return []


def sm_supported(sm: tuple[int, int], variant: CudaVariant) -> bool:
    return sm >= (CUDA13_MIN_SM if variant is CudaVariant.CUDA13 else CUDA12_MIN_SM)


def driver_supported(driver_version: str, variant: CudaVariant) -> bool:
    if not driver_version:
        return False
    try:
        major = int(driver_version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major >= (CUDA13_MIN_DRIVER if variant is CudaVariant.CUDA13 else CUDA12_MIN_DRIVER)


def select_variant(device: PhysicalGpu) -> tuple[CudaVariant | None, str]:
    """Choose the newest compatible JAX CUDA wheel variant for one device."""
    if sm_supported(device.sm, CudaVariant.CUDA13) and driver_supported(
        device.driver_version, CudaVariant.CUDA13
    ):
        return CudaVariant.CUDA13, (
            f"SM {device.compute_capability} >= {CUDA13_MIN_SM[0]}.{CUDA13_MIN_SM[1]} and "
            f"driver {device.driver_version} >= {CUDA13_MIN_DRIVER}"
        )
    if sm_supported(device.sm, CudaVariant.CUDA12) and driver_supported(
        device.driver_version, CudaVariant.CUDA12
    ):
        reason = f"SM {device.compute_capability} >= {CUDA12_MIN_SM[0]}.{CUDA12_MIN_SM[1]}"
        if not sm_supported(device.sm, CudaVariant.CUDA13):
            reason += (
                f"; CUDA 13 requires SM >= {CUDA13_MIN_SM[0]}.{CUDA13_MIN_SM[1]}, "
                "so CUDA 12 applies"
            )
        else:
            reason += f"; driver {device.driver_version} < {CUDA13_MIN_DRIVER}"
        return CudaVariant.CUDA12, reason
    return None, (
        f"SM {device.compute_capability} is below the CUDA 12 minimum "
        f"{CUDA12_MIN_SM[0]}.{CUDA12_MIN_SM[1]}"
    )


def is_idle(device: PhysicalGpu, max_used_mib: int = 256, max_utilization: float = 5.0) -> bool:
    """Conservative idleness test: little VRAM held and no measurable compute load.

    A device holding an external process, or already using VRAM above the budget, is
    treated as busy. This is deliberately strict: an idle capacity claim must be
    clearly justified before any bounded workload is allowed to run.
    """
    if device.active_processes:
        return False
    if device.utilization_percent > max_utilization:
        return False
    return device.used_vram_mib <= max_used_mib


def idle_assessment(device: PhysicalGpu) -> dict[str, Any]:
    """Classify a device as idle or busy against an external workload."""
    return {
        "used_vram_mib": device.used_vram_mib,
        "free_vram_mib": device.free_vram_mib,
        "utilization_percent": device.utilization_percent,
        "active_processes": list(device.active_processes),
        "external_workload": bool(device.active_processes),
    }


def library_path_risk() -> dict[str, Any]:
    """Detect a host ``LD_LIBRARY_PATH`` that can shadow the JAX CUDA wheels.

    JAX's pip wheels bundle their own CUDA libraries. A host ``LD_LIBRARY_PATH``
    pointing at a system CUDA toolkit takes precedence and shadows them, so plugin
    initialization fails with an error such as ``Unable to load cuSPARSE`` that names a
    library the wheel actually ships. This is reported so the operator sees the cause
    instead of debugging it from scratch; nothing is changed on the host.
    """
    value = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in value.split(":") if entry]
    risky = [entry for entry in entries if "cuda" in entry.lower()]
    return {
        "shadowing_risk": bool(risky),
        "cuda_entries": risky,
        "guidance": (
            "Clear LD_LIBRARY_PATH for the JAX process only (for example "
            "'env -u LD_LIBRARY_PATH ...'). Do not change the host driver, the system "
            "CUDA installation, or any running service."
        )
        if risky
        else "no CUDA directory is present in LD_LIBRARY_PATH",
    }


def ordinal_ordering_risk(devices: list[PhysicalGpu]) -> dict[str, Any]:
    """Report whether CUDA runtime ordinal order differs from nvidia-smi index order.

    The two orderings are not guaranteed to match. Selecting a device by
    ``CUDA_VISIBLE_DEVICES`` uses the CUDA runtime ordinal, so an operator who reads
    ``nvidia-smi`` and picks index 1 can silently target a different GPU. Every
    artifact records both identifiers so the mapping is never assumed.
    """
    if len(devices) < 2:
        return {"differs": False, "note": "fewer than two devices; ordering is unambiguous"}
    differs = [device.ordinal != device.smi_index for device in devices]
    return {
        "differs": any(differs),
        "cuda_ordinals": [device.ordinal for device in devices],
        "nvidia_smi_indices": [device.smi_index for device in devices],
        "names": [device.name for device in devices],
        "note": (
            "CUDA runtime ordinal order differs from nvidia-smi index order on this "
            "host. CUDA_VISIBLE_DEVICES uses the ordinal, not the nvidia-smi index."
        )
        if any(differs)
        else "CUDA ordinal order matches nvidia-smi index order on this host",
    }


def environment_manifest() -> dict[str, Any]:
    """A portable, secret-free record of the host and JAX environment."""
    import sys

    manifest: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "kernel": platform.version(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "nvidia_driver": _driver_version(),
        "cuda_version": _driver_cuda_version(),
        "ld_library_path_present": "LD_LIBRARY_PATH" in os.environ,
    }
    if "LD_LIBRARY_PATH" in os.environ:
        manifest["ld_library_path_risk"] = library_path_risk()
    try:
        import numpy

        manifest["numpy"] = numpy.__version__
    except ImportError:
        manifest["numpy"] = None
    try:
        import jax
        import jaxlib

        manifest["jax"] = jax.__version__
        manifest["jaxlib"] = jaxlib.__version__
    except ImportError:
        manifest["jax"] = None
        manifest["jaxlib"] = None
    return manifest


def discover_apple_silicon() -> dict[str, Any]:
    """Detect Apple Silicon (Darwin ARM64) capabilities and Metal/MPS status."""
    is_darwin = platform.system() == "Darwin"
    is_arm = platform.machine() in {"arm64", "aarch64"}
    mps_available = False
    if is_darwin and is_arm:
        try:
            import torch

            mps_available = bool(torch.backends.mps.is_available())
        except (ImportError, AttributeError):
            pass
    return {
        "is_apple_silicon": is_darwin and is_arm,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "metal_supported": is_darwin and is_arm,
        "mps_available": mps_available,
    }


def discover_rocm_devices() -> list[dict[str, Any]]:
    """Detect AMD ROCm (HIP) accelerators via rocm-smi."""
    executable = shutil.which("rocm-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, "--showid", "--json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if completed.returncode == 0:
            data = json.loads(completed.stdout)
            return [{"device_id": str(k), "info": v} for k, v in data.items()]
    except Exception:
        pass
    return []


def discover_tpu_devices() -> dict[str, Any]:
    """Detect Google Cloud TPU accelerators via JAX runtime and environment variables."""
    devices: list[str] = []
    tpu_available = False
    chip_type = "unknown"
    error: str | None = None
    try:
        import jax

        tpu_devs = [d for d in jax.devices() if getattr(d, "platform", None) == "tpu"]
        if tpu_devs:
            tpu_available = True
            devices = [str(d) for d in tpu_devs]
            # Infer chip type from device description or env
            desc = devices[0].lower()
            if "v5e" in desc or "viperlite" in desc:
                chip_type = "v5e"
            elif "v4" in desc:
                chip_type = "v4"
            elif "v3" in desc:
                chip_type = "v3"
            elif "v2" in desc:
                chip_type = "v2"
            else:
                chip_type = "tpu-generic"
    except Exception as exc:
        error = repr(exc)

    tpu_name_env = os.environ.get("TPU_NAME")
    return {
        "tpu_available": tpu_available,
        "device_count": len(devices),
        "devices": devices,
        "chip_type": chip_type,
        "tpu_name_env": tpu_name_env,
        "error": error,
    }


def hardware_report() -> dict[str, Any]:
    """A complete non-destructive hardware inventory with per-device compatibility."""
    devices = enumerate_physical_gpus()
    entries = []
    for device in devices:
        variant, reason = select_variant(device)
        entries.append(
            {
                "device": device.to_dict(),
                "selected_variant": variant.value if variant else None,
                "selection_reason": reason,
                "idleness": idle_assessment(device),
                "cuda12_supported": sm_supported(device.sm, CudaVariant.CUDA12),
                "cuda13_supported": sm_supported(device.sm, CudaVariant.CUDA13),
            }
        )
    return {
        "report": "XLAxiom CUDA hardware inventory",
        "environment": environment_manifest(),
        "device_count": len(entries),
        "devices": entries,
        "apple_silicon": discover_apple_silicon(),
        "rocm_devices": discover_rocm_devices(),
        "tpu_devices": discover_tpu_devices(),
        "note": (
            "CUDA runtime ordinal order may differ from nvidia-smi index order; the "
            "ordinal and nvidia-smi index are both recorded for reconciliation."
        ),
    }

