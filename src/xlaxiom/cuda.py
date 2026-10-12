"""Capability-only CUDA discovery; never silently substitutes CPU."""

from __future__ import annotations

from typing import Any


def cuda_capability() -> dict[str, Any]:
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
