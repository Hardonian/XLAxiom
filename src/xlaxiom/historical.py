"""Explicit trusted adapters for documented historical regressions."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HistoricalObservation:
    issue: str
    adapter: str
    jax_version: str
    jaxlib_version: str
    platform: str
    input_float32: float
    reference_gradient: float
    arcsinh_gradient: float
    acosh_gradient: float
    classification: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def replay_jax_40634() -> HistoricalObservation:
    """Replay the fixed large-float32 gradient observation without loading executable case data."""
    import jax
    import jax.numpy as jnp
    import jaxlib

    value = jnp.float32(1e22)
    arcsinh = float(jax.jit(jax.grad(jnp.arcsinh))(value).block_until_ready())
    acosh = float(jax.jit(jax.grad(jnp.arccosh))(value).block_until_ready())
    reference = 1.0 / math.hypot(1.0, 1e22)
    resolved = np.isclose(arcsinh, reference, rtol=2e-5, atol=0.0) and np.isclose(
        acosh, reference, rtol=2e-5, atol=0.0
    )
    return HistoricalObservation(
        issue="https://github.com/jax-ml/jax/issues/40634",
        adapter="jax-40634-asinh-acosh-large-float32-v1",
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        platform=platform.platform(),
        input_float32=float(value),
        reference_gradient=reference,
        arcsinh_gradient=arcsinh,
        acosh_gradient=acosh,
        classification=(
            "resolved_historical_regression"
            if resolved
            else "known_historical_regression_reproduced"
        ),
    )
