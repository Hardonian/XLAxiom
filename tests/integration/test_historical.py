from xlaxiom.historical import replay_jax_40634


def test_current_jax_resolves_large_float32_asinh_acosh_gradient() -> None:
    result = replay_jax_40634()
    assert result.classification == "resolved_historical_regression"
    assert result.arcsinh_gradient > 0
    assert result.acosh_gradient > 0
