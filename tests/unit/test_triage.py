from pathlib import Path

from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.storage import CaseArtifact, save_case
from xlaxiom.triage import format_triage_issue


def test_format_triage_issue_generates_markdown(tmp_path: Path) -> None:
    case_path = tmp_path / "test_case.json"
    workload = generate(GenerationConfig(seed=42))
    case = CaseArtifact(
        workload=workload,
        comparison_mode="jax-eager",
        expected_classification="pass",
        provenance="local-test",
    )
    save_case(case_path, case)

    issue_md = format_triage_issue(case_path, target_tracker="jax")
    assert "### Description: Numerical discrepancy observed in" in issue_md
    assert "#### Reproduction Verdict & Classification" in issue_md
    assert "#### Numerical Error Metrics" in issue_md
    assert "#### Environment Provenance" in issue_md
    assert "#### Standalone Reproduction Script" in issue_md
    assert "import jax" in issue_md
    assert "import numpy as np" in issue_md


def test_format_triage_issue_supports_pytorch_tracker(tmp_path: Path) -> None:
    case_path = tmp_path / "test_case.json"
    workload = generate(GenerationConfig(seed=12))
    case = CaseArtifact(
        workload=workload,
        comparison_mode="jax-eager",
        expected_classification="pass",
        provenance="local-test",
    )
    save_case(case_path, case)

    issue_md = format_triage_issue(case_path, target_tracker="pytorch")
    assert "PYTORCH" in issue_md


def test_format_triage_issue_supports_tpu_tracker(tmp_path: Path) -> None:
    case_path = tmp_path / "test_case_tpu.json"
    workload = generate(GenerationConfig(seed=24))
    case = CaseArtifact(
        workload=workload,
        comparison_mode="jax-eager",
        expected_classification="pass",
        provenance="tpu-eval",
    )
    save_case(case_path, case)

    issue_md = format_triage_issue(case_path, target_tracker="tpu")
    assert "Google Cloud TPU / OpenXLA" in issue_md
    assert "#### StableHLO MLIR Dialect Representation" in issue_md
    assert "module @xlaxiom" in issue_md
    assert "backend='tpu'" in issue_md

