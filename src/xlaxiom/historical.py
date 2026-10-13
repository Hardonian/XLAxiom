"""Explicit trusted adapters for documented historical regressions."""

from __future__ import annotations

import json
import math
import platform
import sys
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


def hyperbolic_gradient_study() -> dict[str, object]:
    """Per-operation eager/JIT gradient observations with analytic float64 references."""
    import jax
    import jax.numpy as jnp
    import jaxlib

    cases = (("arcsinh", 1e22), ("arcsinh", -1e22), ("acosh", 1e22))
    observations: list[dict[str, object]] = []
    for operation, raw_value in cases:
        value = jnp.float32(raw_value)
        function = getattr(jnp, operation)
        eager = float(jax.grad(function)(value).block_until_ready())
        compiled = float(jax.jit(jax.grad(function))(value).block_until_ready())
        if operation == "arcsinh":
            reference = 1.0 / math.hypot(1.0, abs(raw_value))
        else:
            reference = 1.0 / (math.sqrt(raw_value - 1.0) * math.sqrt(raw_value + 1.0))
        passed = np.isclose(eager, reference, rtol=2e-5, atol=0.0) and np.isclose(
            compiled, reference, rtol=2e-5, atol=0.0
        )
        observations.append(
            {
                "operation": operation,
                "input": raw_value,
                "eager_gradient": eager,
                "jit_gradient": compiled,
                "analytic_reference": reference,
                "rtol": 2e-5,
                "classification": "pass" if passed else "analytic_discrepancy",
            }
        )
    return {
        "study": "XLAxiom Historical Regression Study 001 — JAX Hyperbolic Gradient Stability",
        "source": "https://github.com/jax-ml/jax/issues/40634",
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": platform.platform(),
        "x64_enabled": bool(getattr(jax.config, "jax_enable_x64", False)),
        "observations": observations,
    }


def expected_hyperbolic_status(jax_version: str) -> dict[tuple[str, float], str] | None:
    """Released-version expectations from independently executed historical evidence.

    Returning ``None`` intentionally fails the compatibility test for an uncharacterized
    version instead of silently treating changed numerical behavior as a pass.
    """
    known = {
        "0.10.2": {
            ("arcsinh", 1e22): "analytic_discrepancy",
            ("arcsinh", -1e22): "analytic_discrepancy",
            ("acosh", 1e22): "analytic_discrepancy",
        },
        "0.11.1": {
            ("arcsinh", 1e22): "analytic_discrepancy",
            ("arcsinh", -1e22): "analytic_discrepancy",
            ("acosh", 1e22): "analytic_discrepancy",
        },
        "0.11.2": {
            ("arcsinh", 1e22): "analytic_discrepancy",
            ("arcsinh", -1e22): "analytic_discrepancy",
            ("acosh", 1e22): "pass",
        },
    }
    return known.get(jax_version)


def main(argv: list[str] | None = None) -> int:
    if argv == ["_study"]:
        print(json.dumps(hyperbolic_gradient_study(), sort_keys=True))
        return 0
    raise SystemExit("historical module is invoked by xlaxiom historical")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


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
