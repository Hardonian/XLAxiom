from pathlib import Path

import pytest

from xlaxiom.bisect import BisectResult, BisectStep, bisect_versions
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.storage import CaseArtifact, save_case


def test_bisect_versions_binary_search(tmp_path: Path) -> None:
    case_path = tmp_path / "test_case.json"
    workload = generate(GenerationConfig(seed=1))
    case = CaseArtifact(
        workload=workload,
        comparison_mode="jax-eager",
        expected_classification="pass",
        provenance="test",
    )
    save_case(case_path, case)

    versions = ["0.4.30", "0.4.31", "0.4.32", "0.4.33", "0.4.34"]

    # Mock evaluator where 0.4.30, 0.4.31, 0.4.32 pass, but 0.4.33 and 0.4.34 fail
    def mock_evaluator(version: str, index: int, path: Path, timeout_s: float) -> BisectStep:
        classification = (
            "pass" if version in {"0.4.30", "0.4.31", "0.4.32"} else "numerical_discrepancy"
        )
        return BisectStep(
            version=version,
            index=index,
            status="ok",
            classification=classification,
            duration_s=0.5,
        )

    result = bisect_versions(case_path, versions, evaluator=mock_evaluator)
    assert isinstance(result, BisectResult)
    assert result.first_bad_version == "0.4.33"
    assert result.last_good_version == "0.4.32"
    assert "jax==0.4.33" in result.reproduction_command
    assert result.steps_executed <= 4


def test_bisect_versions_all_pass(tmp_path: Path) -> None:
    case_path = tmp_path / "test_case.json"
    workload = generate(GenerationConfig(seed=2))
    case = CaseArtifact(
        workload=workload,
        comparison_mode="jax-eager",
        expected_classification="pass",
        provenance="test",
    )
    save_case(case_path, case)

    versions = ["0.4.30", "0.4.31", "0.4.32"]

    def mock_evaluator(version: str, index: int, path: Path, timeout_s: float) -> BisectStep:
        return BisectStep(
            version=version,
            index=index,
            status="ok",
            classification="pass",
            duration_s=0.2,
        )

    result = bisect_versions(case_path, versions, evaluator=mock_evaluator)
    assert result.first_bad_version is None
    assert result.last_good_version == "0.4.30"


def test_bisect_versions_invalid_args(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bisect_versions(tmp_path / "nonexistent.json", ["0.4.30", "0.4.31"])

    case_path = tmp_path / "case.json"
    case_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two versions"):
        bisect_versions(case_path, ["0.4.30"])
