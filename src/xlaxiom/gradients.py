"""Bounded declarative scalar AD cases with independent analytic references."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

GradientOperation = Literal["tanh", "exp", "arcsinh", "acosh"]


@dataclass(frozen=True)
class GradientCase:
    identifier: str
    operation: GradientOperation
    value: float
    dtype: Literal["float32", "float64"] = "float32"
    seed: int = 0

    def validate(self) -> None:
        if not math.isfinite(self.value) or self.operation == "acosh" and self.value < 1:
            raise ValueError("gradient case has an invalid mathematical domain")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_gradient_case(seed: int) -> GradientCase:
    """Generate one domain-valid scalar case, including meaningful float32 edge magnitudes."""
    operations: tuple[GradientOperation, ...] = ("tanh", "exp", "arcsinh", "acosh")
    operation = operations[seed % len(operations)]
    values = (0.0, 1e-6, 1.0, 10.0, 1e22)
    value = values[(seed // len(operations)) % len(values)]
    if operation in {"tanh", "exp"}:
        value = min(value, 10.0)
    if operation == "acosh":
        value = max(value, 1.0)
    return GradientCase(f"gradient-{operation}-{seed}", operation, value, seed=seed)


def analytic_gradient(case: GradientCase) -> float:
    case.validate()
    x = case.value
    if case.operation == "tanh":
        return 1.0 / math.cosh(x) ** 2
    if case.operation == "exp":
        return math.exp(x)
    if case.operation == "arcsinh":
        return 1.0 / math.hypot(1.0, abs(x))
    return 1.0 / (math.sqrt(x - 1.0) * math.sqrt(x + 1.0)) if x > 1 else math.inf


def run_gradient_case(case: GradientCase, rtol: float = 2e-5) -> dict[str, object]:
    """Compare eager and JIT reverse-mode gradients against the analytic reference."""
    case.validate()
    import jax
    import jax.numpy as jnp

    function = getattr(jnp, case.operation)
    value = jnp.asarray(case.value, dtype=getattr(jnp, case.dtype))
    eager = float(jax.grad(function)(value).block_until_ready())
    compiled = float(jax.jit(jax.grad(function))(value).block_until_ready())
    reference = analytic_gradient(case)
    valid_reference = math.isfinite(reference)
    matches = (
        valid_reference
        and np.isclose(eager, reference, rtol=rtol, atol=0)
        and np.isclose(compiled, reference, rtol=rtol, atol=0)
    )
    return {
        "case": case.to_dict(),
        "oracle": "analytic-v1",
        "reference": reference,
        "eager": eager,
        "jit": compiled,
        "rtol": rtol,
        "classification": "pass" if matches else "gradient_discrepancy",
    }
