from xlaxiom.historical import (
    hyperbolic_gradient_study,
    replay_jax_40634,
)


def test_current_packaged_jax_reproduces_known_arcsinh_issue() -> None:
    result = replay_jax_40634()
    assert result.classification in {
        "known_historical_regression_reproduced",
        "resolved_historical_regression",
    }
    if result.classification == "known_historical_regression_reproduced":
        assert result.arcsinh_gradient == 0.0
    else:
        assert result.arcsinh_gradient > 0
    study = hyperbolic_gradient_study()
    observed = {
        (item["operation"], item["input"]): item["classification"] for item in study["observations"]
    }
    assert observed
