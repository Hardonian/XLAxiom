from xlaxiom.comparison import Classification
from xlaxiom.metamorphic import build_property_artifact
from xlaxiom.model import InputSpec, Node, Workload
from xlaxiom.relational import (
    build_relational_report,
    minimize_relational,
    run_relational_comparison,
)
from xlaxiom.storage import RelationalArtifact, load_relational_case, save_relational_case


def test_property_artifact_round_trip_and_execution(tmp_path) -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=4, backend="numpy")
    path = save_relational_case(tmp_path / "relation.json", artifact)

    restored = load_relational_case(path)
    comparison = run_relational_comparison(restored)

    assert restored.to_dict() == artifact.to_dict()
    assert comparison.classification == Classification.PASS


def test_relational_minimizer_preserves_a_pair_discrepancy() -> None:
    inputs = (InputSpec("x", (4, 4)),)
    left = Workload(
        identifier="left-double-neg",
        seed=8,
        inputs=inputs,
        nodes=(Node("neg", ("x",), "n1"), Node("neg", ("n1",), "out")),
        outputs=("out",),
    )
    right = Workload(
        identifier="right-neg",
        seed=8,
        inputs=inputs,
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )
    artifact = RelationalArtifact(
        identifier="synthetic-relational-discrepancy",
        left=left,
        right=right,
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value,
        backend="numpy",
        provenance="test-only",
    )

    outcome = minimize_relational(artifact, max_evaluations=12)

    assert (
        run_relational_comparison(outcome.artifact).classification
        == Classification.NUMERICAL_DISCREPANCY
    )
    assert outcome.artifact.minimization is not None
    assert outcome.left.reduced_nodes < outcome.left.original_nodes


def test_relational_report_identifies_both_workloads() -> None:
    artifact = build_property_artifact("reshape-roundtrip-v1", seed=9, backend="numpy")
    report = build_relational_report(artifact, run_relational_comparison(artifact))

    observed = report["observed_result"]
    assert report["artifact_kind"] == "relational-case"
    assert observed["left_workload_sha256"] == artifact.left.digest()
    assert observed["right_workload_sha256"] == artifact.right.digest()
