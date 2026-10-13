import numpy as np
import pytest

from xlaxiom.gradients import (
    GradientCase,
    analytic_gradient,
    generate_gradient_case,
    run_gradient_campaign,
    run_gradient_case,
)


def test_gradient_generator_is_domain_valid() -> None:
    assert all(
        run_gradient_case(generate_gradient_case(seed))["classification"]
        in {"pass", "gradient_discrepancy"}
        for seed in range(20)
    )
    with pytest.raises(ValueError):
        GradientCase("bad", "acosh", 0.5).validate()


def test_analytic_gradient_oracles() -> None:
    assert analytic_gradient(GradientCase("t", "tanh", 0.0)) == 1.0
    assert np.isclose(analytic_gradient(GradientCase("a", "arcsinh", 1e22)), 1e-22)


def test_generalized_gradient_case_runs() -> None:
    result = run_gradient_case(GradientCase("t", "tanh", 1.0))
    assert result["classification"] == "pass"


def test_gradient_campaign_segregates_known_historical_observations() -> None:
    result = run_gradient_campaign(20)
    assert set(result["operation_coverage"]) == {"tanh", "exp", "arcsinh", "acosh"}
    assert result["environment"]["jax"] is not None
    # The lockfile resolves JAX 0.10.2 on Python 3.11 and 0.11.2 on Python 3.12+. The
    # acosh(+1e22) gradient was fixed in 0.11.2, so it is a candidate on 0.10.2 only.
    import jax

    acosh_is_candidate = jax.__version__ < "0.11.2"
    expected_known = 1
    expected_candidates = 3 if acosh_is_candidate else 2
    expected_operations = {"tanh", "acosh"} if acosh_is_candidate else {"tanh"}
    assert result["counts"]["known_historical_discrepancy"] == expected_known
    assert result["counts"]["candidate_discrepancy"] == expected_candidates
    assert {candidate["case"]["operation"] for candidate in result["candidate_anomalies"]} == (
        expected_operations
    )
