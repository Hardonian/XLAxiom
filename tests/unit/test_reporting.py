import json

import numpy as np

from xlaxiom.comparison import Classification, ComparisonResult
from xlaxiom.execution import ExecutionResult, ExecutionStatus
from xlaxiom.generation import synthetic_control
from xlaxiom.model import ComparisonPolicy
from xlaxiom.pipeline import FaultInjection
from xlaxiom.reporting import build_report, write_report
from xlaxiom.storage import CaseArtifact


def _make_sample_comparison(
    classification: Classification = Classification.NUMERICAL_DISCREPANCY,
) -> ComparisonResult:
    policy = ComparisonPolicy(atol=1e-5, rtol=1e-5)
    left = ExecutionResult(
        "left", "eager", ExecutionStatus.OK, (np.array([1.0], dtype=np.float32),)
    )
    right = ExecutionResult(
        "right", "eager", ExecutionStatus.OK, (np.array([2.0], dtype=np.float32),)
    )
    return ComparisonResult(
        classification=classification,
        explanation="Values differed beyond tolerance.",
        policy=policy,
        left=left.summary(),
        right=right.summary(),
    )


def test_build_report_non_fault_upstream_candidate() -> None:
    artifact = CaseArtifact(synthetic_control(seed=42), fault=None)
    comparison = _make_sample_comparison(Classification.NUMERICAL_DISCREPANCY)

    report = build_report(artifact, comparison)

    assert report["report_version"] == "1.0"
    assert "timestamp_utc" in report
    assert "python" in report["environment"]
    assert "platform" in report["environment"]
    assert "numpy" in report["environment"]
    assert report["upstream_candidate"] is True
    assert "differential evidence only" in report["interpretation"]
    assert report["issue_draft"]["do_not_file_automatically"] is True
    assert "Differential observation: numerical_discrepancy" in report["issue_draft"]["title"]


def test_build_report_synthetic_fault_not_upstream_candidate() -> None:
    fault = FaultInjection(output_offset=1.0, label="injected_sin_fault")
    artifact = CaseArtifact(synthetic_control(seed=42), fault=fault)
    comparison = _make_sample_comparison(Classification.NUMERICAL_DISCREPANCY)

    report = build_report(artifact, comparison)

    assert report["upstream_candidate"] is False
    assert "synthetic control" in report["interpretation"]
    expected_limitations = "Not suitable for filing while this case is synthetic."
    assert report["issue_draft"]["limitations"] == expected_limitations


def test_build_report_non_discrepancy_classification_not_upstream_candidate() -> None:
    artifact = CaseArtifact(synthetic_control(seed=42), fault=None)
    comparison = _make_sample_comparison(Classification.PASS)

    report = build_report(artifact, comparison)

    assert report["upstream_candidate"] is False


def test_write_report_creates_json_and_markdown(tmp_path) -> None:
    artifact = CaseArtifact(synthetic_control(seed=42), fault=None)
    comparison = _make_sample_comparison(Classification.NUMERICAL_DISCREPANCY)

    case_path = tmp_path / "reduced-case.json"
    json_path, markdown_path = write_report(tmp_path, artifact, comparison, case_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert json_path.name == "report.json"
    assert markdown_path.name == "report.md"

    # Verify JSON content
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["report_version"] == "1.0"
    assert "comparison" in data["observed_result"]
    assert data["observed_result"]["comparison"]["classification"] == "numerical_discrepancy"

    # Verify Markdown content
    md_text = markdown_path.read_text(encoding="utf-8")
    assert "# XLAxiom evidence report" in md_text
    assert artifact.workload.identifier in md_text
    assert "numerical_discrepancy" in md_text
    assert "No upstream issue is filed by XLAxiom." in md_text
    assert f"xlaxiom reproduce --case {case_path}" in md_text

    issue_draft = data["issue_draft"]
    assert issue_draft["case_sha256"] == artifact.to_dict()["sha256"]
    assert issue_draft["workload_sha256"] == artifact.workload.digest()

    html_path = tmp_path / "report.html"
    assert html_path.exists()
    html_text = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "XLAxiom Evidence Report" in html_text
    assert "NUMERICAL DISCREPANCY" in html_text
    assert f"xlaxiom reproduce --case {case_path}" in html_text
    assert "Strict non-filing notice" in html_text
