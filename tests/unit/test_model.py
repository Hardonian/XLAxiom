from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.model import ValidationError, Workload, materialize_inputs


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
