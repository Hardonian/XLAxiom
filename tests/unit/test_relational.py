"""Additional unit coverage for relational artifact execution and minimization.

These tests target the honest-failure paths of the relational pipeline: unsupported
backends, non-reproducible predicates, minimization of a passing relation, and the
report/bundle surfaces. They complement the round-trip and discrepancy tests in
test_relational.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from xlaxiom.comparison import Classification
from xlaxiom.metamorphic import build_property_artifact
from xlaxiom.model import InputSpec, Node, Workload
from xlaxiom.relational import (
    build_relational_report,
    minimize_relational,
    reproduce_relational_isolated,
    run_relational_comparison,
    write_relational_report,
)
from xlaxiom.storage import (
    ArtifactError,
    RelationalArtifact,
    load_relational_case,
    save_relational_case,
)


def _discrepant_artifact() -> RelationalArtifact:
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
    return RelationalArtifact(
        identifier="synthetic-relational-discrepancy",
        left=left,
        right=right,
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value,
        backend="numpy",
        provenance="test-only",
    )


def test_unsupported_backend_raises_artifact_error() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=1, backend="numpy")
    with pytest.raises(ArtifactError, match="unsupported relational backend"):
        run_relational_comparison(
            RelationalArtifact(
                identifier=artifact.identifier,
                left=artifact.left,
                right=artifact.right,
                backend="not-a-backend",
                provenance="test-only",
            )
        )


def test_all_declared_backends_execute_a_passing_relation() -> None:
    for backend in ("numpy", "jax-eager", "jax-jit"):
        artifact = build_property_artifact("add-commutativity-v1", seed=3, backend=backend)
        comparison = run_relational_comparison(artifact)
        assert comparison.classification == Classification.PASS, backend


def test_minimize_relational_rejects_a_non_reproducible_predicate() -> None:
    artifact = _discrepant_artifact()
    tampered = RelationalArtifact(
        identifier=artifact.identifier,
        left=artifact.left,
        right=artifact.right,
        # A declared expectation that the workloads do not currently produce.
        expected_classification="shape_mismatch",
        backend="numpy",
        provenance="test-only",
    )
    with pytest.raises(ArtifactError, match="not currently reproducible"):
        minimize_relational(tampered, max_evaluations=4)


def test_minimize_relational_rejects_a_passing_relation() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=5, backend="numpy")
    with pytest.raises(ArtifactError, match="requires a currently failing relation"):
        minimize_relational(artifact, max_evaluations=4)


def test_minimize_relational_shrinks_both_sides() -> None:
    outcome = minimize_relational(_discrepant_artifact(), max_evaluations=16)

    # The left side (double negation) reduces; the right side is already a single
    # node and stays minimal.
    assert outcome.left.reduced_nodes < outcome.left.original_nodes
    assert outcome.right.reduced_nodes <= outcome.right.original_nodes
    assert outcome.duration_ms >= 0
    minimization = outcome.artifact.minimization
    assert minimization is not None
    assert minimization["original_left_sha256"]
    assert minimization["original_right_sha256"]
    assert minimization["preserved_signature"]
    # The minimized pair must still fail the same way it did before reduction.
    preserved = run_relational_comparison(outcome.artifact)
    assert preserved.classification == Classification.NUMERICAL_DISCREPANCY


def test_reproduce_relational_isolated_rejects_invalid_budgets() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=2, backend="numpy")
    with pytest.raises(ArtifactError, match="must be positive"):
        reproduce_relational_isolated(artifact, attempts=0)
    with pytest.raises(ArtifactError, match="must be positive"):
        reproduce_relational_isolated(artifact, timeout_s=0)


def test_reproduce_relational_isolated_records_blocked_observations() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=6, backend="numpy")
    with patch("xlaxiom.relational.subprocess.run", side_effect=OSError("no interpreter")):
        outcome = reproduce_relational_isolated(artifact, attempts=2, timeout_s=5.0)

    assert outcome["reproduced"] is False
    assert outcome["classification"] is None
    assert len(outcome["observations"]) == 2
    assert {item["state"] for item in outcome["observations"]} == {"blocked"}


def test_reproduce_relational_isolated_records_nonzero_exit_as_blocked() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=7, backend="numpy")

    class _Completed:
        returncode = 1
        stderr = "Traceback: import failure"
        stdout = ""

    with patch("xlaxiom.relational.subprocess.run", return_value=_Completed()):
        outcome = reproduce_relational_isolated(artifact, attempts=1, timeout_s=5.0)

    assert outcome["reproduced"] is False
    assert outcome["observations"][0]["state"] == "blocked"
    assert "import failure" in outcome["observations"][0]["error"]


def test_reproduce_relational_isolated_records_unparseable_output_as_inconclusive() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=8, backend="numpy")

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = "not json at all"

    with patch("xlaxiom.relational.subprocess.run", return_value=_Completed()):
        outcome = reproduce_relational_isolated(artifact, attempts=1, timeout_s=5.0)

    assert outcome["reproduced"] is False
    assert outcome["observations"][0]["state"] == "inconclusive"


def test_reproduce_relational_isolated_reports_a_single_agreed_classification() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=9, backend="numpy")

    class _Completed:
        returncode = 0
        stderr = ""
        stdout = json.dumps({"classification": Classification.PASS.value})

    with patch("xlaxiom.relational.subprocess.run", return_value=_Completed()):
        outcome = reproduce_relational_isolated(artifact, attempts=3, timeout_s=5.0)

    assert outcome["reproduced"] is True
    assert outcome["classification"] == Classification.PASS.value
    assert len(outcome["observations"]) == 3


def test_reproduce_relational_isolated_does_not_claim_reproduction_on_disagreement() -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=10, backend="numpy")
    payloads = [
        json.dumps({"classification": Classification.PASS.value}),
        json.dumps({"classification": Classification.NUMERICAL_DISCREPANCY.value}),
    ]

    class _Completed:
        def __init__(self, index: int) -> None:
            self.returncode = 0
            self.stderr = ""
            self.stdout = payloads[index]

    calls = {"n": 0}

    def fake_run(*_args: object, **_kwargs: object) -> _Completed:
        index = calls["n"]
        calls["n"] += 1
        return _Completed(index)

    with patch("xlaxiom.relational.subprocess.run", side_effect=fake_run):
        outcome = reproduce_relational_isolated(artifact, attempts=2, timeout_s=5.0)

    assert outcome["reproduced"] is False
    assert outcome["classification"] is None


def test_write_relational_report_emits_json_and_markdown(tmp_path) -> None:
    artifact = build_property_artifact("add-commutativity-v1", seed=11, backend="numpy")
    comparison = run_relational_comparison(artifact)

    json_path, markdown_path = write_relational_report(tmp_path, artifact, comparison)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["upstream_candidate"] is False
    assert report["issue_draft"]["do_not_file_automatically"] is True
    markdown = markdown_path.read_text(encoding="utf-8")
    assert artifact.identifier in markdown
    assert "No upstream issue is filed by XLAxiom." in markdown


def test_relational_report_flags_a_discrepancy_as_an_upstream_candidate() -> None:
    artifact = _discrepant_artifact()
    comparison = run_relational_comparison(artifact)

    report = build_relational_report(artifact, comparison)

    assert report["upstream_candidate"] is True
    assert artifact.identifier in report["issue_draft"]["title"]


def test_saved_relational_artifact_survives_a_round_trip(tmp_path) -> None:
    artifact = _discrepant_artifact()
    path = save_relational_case(tmp_path / "relation.json", artifact)
    restored = load_relational_case(path)

    assert restored.to_dict() == artifact.to_dict()
    assert restored.minimization is None
