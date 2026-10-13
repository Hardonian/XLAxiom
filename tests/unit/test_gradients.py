import pytest

from xlaxiom.gradients import (
    GradientCase,
    analytic_gradient,
    generate_gradient_case,
    run_gradient_case,
)


def test_gradient_generator_is_domain_valid() -> None:
    assert all(generate_gradient_case(seed).operation for seed in range(20))
    with pytest.raises(ValueError):
        GradientCase("bad", "acosh", 0.5).validate()


def test_analytic_gradient_oracles() -> None:
    assert analytic_gradient(GradientCase("t", "tanh", 0.0)) == 1.0
    assert analytic_gradient(GradientCase("a", "arcsinh", 1e22)) == 1e-22


def test_generalized_gradient_case_runs() -> None:
    result = run_gradient_case(GradientCase("t", "tanh", 1.0))
    assert result["classification"] == "pass"
