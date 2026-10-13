import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.historical_matrix import run_hyperbolic_matrix, validate_version


def test_validate_version_valid() -> None:
    valid_versions = [
        "0.4.25",
        "0.4.35",
        "0.4.35.dev20240101",
        "0.4.35+cuda12",
        "0.5.0a1",
        "1.0.0-rc1",
    ]
    for version in valid_versions:
        validate_version(version)  # Should not raise


def test_validate_version_invalid() -> None:
    invalid_versions = [
        "--evil-flag",
        "0.4.25 --extra-index-url http://evil.com",
        "0.4.25; rm -rf /",
        "0.4.25\n--other-flag",
        "0.4.25\0",
        "-0.4.25",
        "",
        " 0.4.25",
    ]
    for version in invalid_versions:
        with pytest.raises(ValueError, match="Invalid version string"):
            validate_version(version)


def test_run_hyperbolic_matrix_invalid_version_raises(tmp_path) -> None:
    output_path = str(tmp_path / "matrix_report.json")
    versions = ["0.4.25", "--malicious-flag"]

    with patch("subprocess.run") as mock_run:
        with pytest.raises(ValueError, match="Invalid version string"):
            run_hyperbolic_matrix(versions, output_path)

    # subprocess.run should never be called if validation fails
    mock_run.assert_not_called()


def test_run_hyperbolic_matrix_success(tmp_path) -> None:
    output_path = str(tmp_path / "matrix_report.json")
    versions = ["0.4.25", "0.4.35"]

    mock_completed_process = MagicMock(spec=subprocess.CompletedProcess)
    mock_completed_process.returncode = 0
    mock_completed_process.stderr = "some logs"
    mock_completed_process.stdout = 'unrelated log line\n{"jax": "0.4.25", "status": "ok"}'

    with patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        report = run_hyperbolic_matrix(versions, output_path)

    assert mock_run.call_count == len(versions)
    assert report["report_version"] == "1.0"
    assert len(report["results"]) == 2

    first_result = report["results"][0]
    assert first_result["requested_jax"] == "0.4.25"
    assert first_result["exit_code"] == 0
    assert first_result["stderr"] == "some logs"
    assert first_result["study"] == {"jax": "0.4.25", "status": "ok"}
    assert "duration_s" in first_result

    # Check file was saved correctly
    with open(output_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == report


def test_run_hyperbolic_matrix_failure(tmp_path) -> None:
    output_path = str(tmp_path / "failed_report.json")
    versions = ["0.4.20"]

    mock_completed_process = MagicMock(spec=subprocess.CompletedProcess)
    mock_completed_process.returncode = 1
    mock_completed_process.stderr = "Error: package not found"
    mock_completed_process.stdout = ""

    with patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        report = run_hyperbolic_matrix(versions, output_path)

    assert mock_run.call_count == 1
    assert len(report["results"]) == 1

    result = report["results"][0]
    assert result["requested_jax"] == "0.4.20"
    assert result["exit_code"] == 1
    assert result["stderr"] == "Error: package not found"
    assert "study" not in result
