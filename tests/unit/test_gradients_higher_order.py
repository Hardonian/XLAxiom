"""Unit tests for higher-order automatic differentiation (Priority 64)."""

import pytest

from xlaxiom.gradients import (
    GradientCase,
    compute_jvp,
    compute_vjp,
    run_higher_order_gradient_case,
)


def test_jvp_and_vjp_consistency() -> None:
    # Test for tanh at x = 0.5 with tangent/cotangent = 2.0
    x = 0.5
    v = 2.0
    primal_jvp, tangent_out = compute_jvp("tanh", x, v, dtype="float32")
    primal_vjp, cotangent_out = compute_vjp("tanh", x, v, dtype="float32")

    assert pytest.approx(primal_jvp, rel=1e-5) == primal_vjp
    assert pytest.approx(tangent_out, rel=1e-5) == cotangent_out


def test_analytic_hessian_and_hvp() -> None:
    # Case: tanh(1.0)
    case = GradientCase("test-tanh", "tanh", 1.0, dtype="float32")
    result = run_higher_order_gradient_case(case, v=1.5, rtol=1e-3)

    assert result["classification"] == "pass"
    assert pytest.approx(result["observed_hvp"], rel=1e-3) == result["expected_hvp"]
    assert pytest.approx(result["observed_hvp"], rel=5e-2) == result["finite_difference_hvp"]


def test_hvp_exp() -> None:
    # For exp(x), f''(x) = exp(x)
    case = GradientCase("test-exp", "exp", 1.2, dtype="float32")
    result = run_higher_order_gradient_case(case, v=1.0, rtol=1e-3)
    assert result["classification"] == "pass"
