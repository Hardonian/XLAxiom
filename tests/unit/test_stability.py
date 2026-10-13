from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.stability import check_stability


def test_numpy_stability_reports_consistent_repeated_execution() -> None:
    record = check_stability(generate(GenerationConfig(seed=3)), "numpy", runs=3)

    assert record["classification"] == "pass"
    assert len(record["executions"]) == 3
    assert len(record["pairwise"]) == 2
