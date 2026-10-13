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
