from xlaxiom.metamorphic import build_property
from xlaxiom.mutation import evaluate_property_sharpness, generate_workload_mutants


def test_generate_workload_mutants_creates_valid_mutants() -> None:
    pair = build_property("add-commutativity-v1", seed=42)
    mutants = generate_workload_mutants(pair.right)
    assert len(mutants) > 0
    for mutant in mutants:
        assert mutant.workload.identifier.startswith(pair.right.identifier)
        mutant.workload.validate()


def test_evaluate_property_sharpness() -> None:
    report = evaluate_property_sharpness("add-commutativity-v1", seed=42, backend="numpy")
    assert report.property_identifier == "add-commutativity-v1"
    assert report.total_mutants > 0
    assert report.killed_mutants > 0
    assert report.sharpness_score > 0.0

    report_dict = report.to_dict()
    assert report_dict["property_identifier"] == "add-commutativity-v1"
    assert "sharpness_score" in report_dict
    assert len(report_dict["results"]) == report.total_mutants


def test_evaluate_property_sharpness_subtraction() -> None:
    report = evaluate_property_sharpness("subtraction-as-addition-v1", seed=42, backend="numpy")
    assert report.total_mutants > 0
    # Every mutant injected should be killed by subtraction-as-addition
    assert report.killed_mutants == report.total_mutants
    assert report.sharpness_score == 1.0
