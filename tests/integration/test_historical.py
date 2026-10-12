from xlaxiom.historical import replay_jax_40634


def test_current_packaged_jax_reproduces_known_arcsinh_issue() -> None:
    result = replay_jax_40634()
    assert result.classification == "known_historical_regression_reproduced"
    assert result.arcsinh_gradient == 0.0
    assert result.acosh_gradient > 0
