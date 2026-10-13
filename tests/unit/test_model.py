import pytest

from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.model import ValidationError, Workload, _broadcast_shape, materialize_inputs


def test_generation_is_deterministic_and_canonical() -> None:
    first = generate(GenerationConfig(seed=99))
    second = generate(GenerationConfig(seed=99))
    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert (materialize_inputs(first)["x"] == materialize_inputs(second)["x"]).all()


def test_rejects_forward_reference() -> None:
    data = generate(GenerationConfig()).canonical_dict()
    data["nodes"][0]["inputs"] = ["future", "y"]
    try:
        Workload.from_dict(data)
    except ValidationError as exc:
        assert "forward" in str(exc)
    else:
        raise AssertionError("invalid workload accepted")


def test_generator_covers_rank_three_and_integer_safe_motifs() -> None:
    rank_three = generate(GenerationConfig(seed=4, max_rank=3))
    integer = generate(GenerationConfig(seed=1, dtype="int32"))

    assert max(len(spec.shape) for spec in rank_three.inputs) == 3
    assert all(spec.dtype == "int32" for spec in integer.inputs)

    with pytest.raises(ValueError, match="unsupported generation dtype"):
        generate(GenerationConfig(dtype="float16"))


def test_materialization_supports_explicit_float_edge_strategies() -> None:
    signed_zero = generate(GenerationConfig(seed=7, input_mode="signed_zero"))
    nonfinite = generate(GenerationConfig(seed=8, input_mode="nonfinite"))

    signed_values = materialize_inputs(signed_zero)["x"].reshape(-1)
    nonfinite_values = materialize_inputs(nonfinite)["x"].reshape(-1)

    assert signed_values[0] == 0 and signed_values[1] == 0
    assert signed_values[0].view("u4") != signed_values[1].view("u4")
    assert pytest.approx(float("inf")) == float(nonfinite_values[1])
    assert nonfinite_values[0] != nonfinite_values[0]


def test_generator_rejects_nonfinite_integer_inputs() -> None:
    with pytest.raises(ValueError, match="requires a floating-point dtype"):
        generate(GenerationConfig(dtype="int32", input_mode="nonfinite"))


def test_extreme_finite_input_strategy_controls_overflow_and_underflow_edges() -> None:
    workload = generate(GenerationConfig(seed=9, input_mode="extreme_finite"))
    values = materialize_inputs(workload)["x"].reshape(-1)

    assert np.isfinite(values[:6]).all()
    assert abs(values[0]) > 1e30
    assert 0 < abs(values[4]) < 1e-40


def test_broadcast_shape_valid() -> None:
    assert _broadcast_shape() == ()
    assert _broadcast_shape((2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), (2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), (1, 3)) == (2, 3)
    assert _broadcast_shape((1, 3), (2, 1)) == (2, 3)
    assert _broadcast_shape((5, 1, 4), (1, 3, 1)) == (5, 3, 4)
    assert _broadcast_shape((2, 1, 4), (3, 1), (1,)) == (2, 3, 4)
    assert _broadcast_shape((), (2, 3), ()) == (2, 3)


def test_broadcast_shape_incompatible() -> None:
    with pytest.raises(ValidationError, match="incompatible broadcast shapes"):
        _broadcast_shape((2, 3), (4, 3))

    with pytest.raises(ValidationError, match="incompatible broadcast shapes"):
        _broadcast_shape((5, 2, 4), (5, 3, 4))
