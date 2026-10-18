"""Bounded declarative scalar AD cases with independent analytic references."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .execution import doctor

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
        # The derivative is singular at exactly one, so stay inside the
        # differentiable part of the domain for an analytic comparison.
        value = max(value, 1.0001)
    return GradientCase(f"gradient-{operation}-{seed}", operation, value, seed=seed)


def analytic_gradient(case: GradientCase) -> float:
    """Return the reference derivative at the scalar actually presented to JAX."""
    case.validate()
    x = float(np.asarray(case.value, dtype=np.dtype(case.dtype)))
    if case.operation == "tanh":
        return 1.0 / math.cosh(x) ** 2
    if case.operation == "exp":
        return math.exp(x)
    if case.operation == "arcsinh":
        return 1.0 / math.hypot(1.0, abs(x))
    return 1.0 / (math.sqrt(x - 1.0) * math.sqrt(x + 1.0)) if x > 1 else math.inf


def analytic_hessian(case: GradientCase) -> float:
    """Return the reference second derivative (scalar Hessian) at the input scalar."""
    case.validate()
    x = float(np.asarray(case.value, dtype=np.dtype(case.dtype)))
    if case.operation == "tanh":
        return -2.0 * math.tanh(x) / (math.cosh(x) ** 2)
    if case.operation == "exp":
        return math.exp(x)
    if case.operation == "arcsinh":
        denom = (1.0 + x * x) ** 1.5
        return -x / denom
    if case.operation == "acosh":
        if x <= 1.0:
            return math.inf
        denom = (x * x - 1.0) ** 1.5
        return -x / denom
    return math.nan


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
        "reference_input": float(np.asarray(case.value, dtype=np.dtype(case.dtype))),
        "eager": eager,
        "jit": compiled,
        "rtol": rtol,
        "classification": "pass" if matches else "gradient_discrepancy",
    }


def run_gradient_campaign(cases: int, seed: int = 0) -> dict[str, Any]:
    """Run bounded deterministic analytic-gradient cases and segregate known evidence."""
    if not 1 <= cases <= 500:
        raise ValueError("gradient campaign cases must be between 1 and 500")
    results = [run_gradient_case(generate_gradient_case(seed + offset)) for offset in range(cases)]
    coverage: dict[str, int] = {}
    counts = {"pass": 0, "known_historical_discrepancy": 0, "candidate_discrepancy": 0}
    known: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for result in results:
        case = result["case"]
        assert isinstance(case, dict)
        operation = str(case["operation"])
        coverage[operation] = coverage.get(operation, 0) + 1
        if result["classification"] == "pass":
            counts["pass"] += 1
        elif operation == "arcsinh" and abs(float(case["value"])) >= 1e20:
            # Documented upstream behavior from JAX issue #40634: arcsinh gradients at
            # extreme float32 magnitudes remain discrepant across released versions.
            # Segregated as known, never reported as a new XLAxiom finding.
            counts["known_historical_discrepancy"] += 1
            known.append(result)
        else:
            counts["candidate_discrepancy"] += 1
            candidates.append(result)
    return {
        "campaign_version": "gradient-v1",
        "configuration": {"cases": cases, "seed": seed, "oracle": "analytic-v1"},
        "environment": doctor(),
        "operation_coverage": coverage,
        "counts": counts,
        "known_historical_observations": known,
        "candidate_anomalies": candidates,
    }


def compute_jvp(operation: str, x: float, v: float, dtype: str = "float32") -> tuple[float, float]:
    """Evaluate forward-mode Jacobian-vector product (JVP) using JAX."""
    import jax
    import jax.numpy as jnp

    fn = getattr(jnp, operation)
    x_val = jnp.asarray(x, dtype=getattr(jnp, dtype))
    v_val = jnp.asarray(v, dtype=getattr(jnp, dtype))
    primals_out, tangents_out = jax.jvp(fn, (x_val,), (v_val,))
    return float(primals_out.block_until_ready()), float(tangents_out.block_until_ready())


def compute_vjp(
    operation: str, x: float, cotangent: float, dtype: str = "float32"
) -> tuple[float, float]:
    """Evaluate reverse-mode vector-Jacobian product (VJP) using JAX."""
    import jax
    import jax.numpy as jnp

    fn = getattr(jnp, operation)
    x_val = jnp.asarray(x, dtype=getattr(jnp, dtype))
    cot_val = jnp.asarray(cotangent, dtype=getattr(jnp, dtype))
    primals_out, vjp_fn = jax.vjp(fn, x_val)
    (grad_val,) = vjp_fn(cot_val)
    return float(primals_out.block_until_ready()), float(grad_val.block_until_ready())


def compute_hvp(operation: str, x: float, v: float, dtype: str = "float32") -> float:
    """Evaluate Hessian-vector product (HVP) via forward-over-reverse autodiff."""
    import jax
    import jax.numpy as jnp

    fn = getattr(jnp, operation)
    x_val = jnp.asarray(x, dtype=getattr(jnp, dtype))
    v_val = jnp.asarray(v, dtype=getattr(jnp, dtype))
    # HVP is the directional derivative of the gradient
    _, hvp_out = jax.jvp(jax.grad(fn), (x_val,), (v_val,))
    return float(hvp_out.block_until_ready())


def run_higher_order_gradient_case(
    case: GradientCase, v: float = 1.0, rtol: float = 1e-4
) -> dict[str, object]:
    """Validate higher-order autodiff (HVP) against analytic Hessian references."""
    case.validate()
    hvp_val = compute_hvp(case.operation, case.value, v, dtype=case.dtype)
    ref_hess = analytic_hessian(case)
    expected_hvp = ref_hess * v

    # Finite difference second derivative approximation for triangulation
    eps = 1e-4 if case.dtype == "float64" else 1e-2
    x = case.value
    import jax
    import jax.numpy as jnp

    fn = getattr(jnp, case.operation)
    g_plus = float(jax.grad(fn)(jnp.asarray(x + eps, dtype=getattr(jnp, case.dtype))))
    g_minus = float(jax.grad(fn)(jnp.asarray(x - eps, dtype=getattr(jnp, case.dtype))))
    fd_hvp = ((g_plus - g_minus) / (2.0 * eps)) * v

    matches = math.isfinite(expected_hvp) and np.isclose(
        hvp_val, expected_hvp, rtol=rtol, atol=1e-5
    )

    return {
        "case": case.to_dict(),
        "tangent": v,
        "analytic_hessian": ref_hess,
        "expected_hvp": expected_hvp,
        "observed_hvp": hvp_val,
        "finite_difference_hvp": fd_hvp,
        "rtol": rtol,
        "classification": "pass" if matches else "higher_order_discrepancy",
    }
