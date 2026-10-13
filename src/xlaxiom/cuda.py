"""Capability-only CUDA discovery; never silently substitutes CPU.

A requested accelerator that is absent, busy, or numerically unsupported is reported
as UNSUPPORTED or BLOCKED. It is never reported as, or replaced by, a CPU result.
"""

from __future__ import annotations

from typing import Any

from .hardware import (
    CudaVariant,
    enumerate_physical_gpus,
    hardware_report,
    select_variant,
)


def cuda_capability() -> dict[str, Any]:
    """Report the real CUDA capability of the JAX runtime in this process.

    This answers "can the runtime that is actually installed execute on a GPU", which
    is the question that matters for evidence. It does not consult the driver: a GPU
    visible to the driver but unsupported by the installed jaxlib is UNSUPPORTED.
    """
    try:
        import jax
        import jaxlib
    except ImportError as exc:
        return {"supported": False, "status": "unsupported", "reason": str(exc)}
    try:
        devices = jax.devices("gpu")
    except Exception as exc:
        return {
            "supported": False,
            "status": "unsupported",
            "reason": repr(exc),
            "jaxlib": jaxlib.__version__,
        }
    if not devices:
        return {
            "supported": False,
            "status": "unsupported",
            "reason": "no JAX CUDA device",
            "jaxlib": jaxlib.__version__,
        }
    return {
        "supported": True,
        "status": "available",
        "jaxlib": jaxlib.__version__,
        "devices": [str(device) for device in devices],
    }


def driver_capability() -> dict[str, Any]:
    """Report every physical GPU the driver can see, with compatibility gating.

    Compatibility uses the official JAX rules: CUDA 12 wheels require SM 5.2 or newer,
    CUDA 13 wheels require SM 7.5 or newer, and the Linux driver must be at least 525
    for CUDA 12 and 580 for CUDA 13.
    """
    return hardware_report()


def recommended_variant() -> dict[str, Any]:
    """The newest CUDA wheel variant compatible with the host's GPUs.

    Returns ``None`` when no visible GPU meets even the CUDA 12 minimum.
    """
    devices = enumerate_physical_gpus()
    if not devices:
        return {"variant": None, "reason": "no NVIDIA GPU is visible to the driver"}

    cuda12_result: dict[str, Any] | None = None
    for device in devices:
        variant, reason = select_variant(device)
        if variant is CudaVariant.CUDA13:
            return {
                "variant": CudaVariant.CUDA13.value,
                "device": device.name,
                "reason": reason,
            }
        if variant is CudaVariant.CUDA12 and cuda12_result is None:
            cuda12_result = {
                "variant": CudaVariant.CUDA12.value,
                "device": device.name,
                "reason": reason,
            }

    if cuda12_result is not None:
        return cuda12_result

    return {
        "variant": None,
        "reason": "no visible GPU meets the CUDA 12 minimum of SM 5.2",
    }
