import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.matrix import (
    MatrixEnvironment,
    _run_environment,
    _worker,
    load_config,
    main,
    run_matrix,
)


def test_load_config_valid(tmp_path: Path) -> None:
    config_file = tmp_path / "matrix_config.json"
    config_content = {
        "environments": [
            {"name": "jax-0.4.30", "jax": "0.4.30", "timeout_s": 60.0},
            {"name": "jax-0.4.35", "jax": "0.4.35"},
        ]
    }
    config_file.write_text(json.dumps(config_content), encoding="utf-8")

    envs = load_config(config_file)
    assert len(envs) == 2
    assert envs[0] == MatrixEnvironment(name="jax-0.4.30", jax="0.4.30", timeout_s=60.0)
    assert envs[1] == MatrixEnvironment(name="jax-0.4.35", jax="0.4.35", timeout_s=90.0)


def test_load_config_empty(tmp_path: Path) -> None:
    config_file = tmp_path / "empty_config.json"
    config_file.write_text(json.dumps({"environments": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix requires uniquely named environments"):
        load_config(config_file)


def test_load_config_duplicate_names(tmp_path: Path) -> None:
    config_file = tmp_path / "dup_config.json"
    config_content = {
        "environments": [
            {"name": "env1", "jax": "0.4.30"},
            {"name": "env1", "jax": "0.4.35"},
        ]
    }
    config_file.write_text(json.dumps(config_content), encoding="utf-8")

    with pytest.raises(ValueError, match="matrix requires uniquely named environments"):
        load_config(config_file)


@patch("subprocess.run")
def test_run_environment_success(mock_run: MagicMock, tmp_path: Path) -> None:
    env = MatrixEnvironment(name="test_env", jax="0.4.30", timeout_s=30.0)
    case_path = tmp_path / "case.json"

    comparison_dict = {"classification": "pass", "max_abs_diff": 0.0}
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"some logs...\n{json.dumps(comparison_dict)}\n",
        stderr="",
    )

    result = _run_environment(env, case_path)

    assert result["worker_status"] == "ok"
    assert result["exit_code"] == 0
    assert result["comparison"] == comparison_dict
    assert result["environment"]["name"] == "test_env"

    mock_run.assert_called_once()
    called_args = mock_run.call_args[0][0]
    assert called_args[0] == "uv"
    assert "jax==0.4.30" in called_args


@patch("subprocess.run")
def test_run_environment_failure_non_zero_exit(mock_run: MagicMock, tmp_path: Path) -> None:
    env = MatrixEnvironment(name="test_env", jax="0.4.30")
    case_path = tmp_path / "case.json"

    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="Process failed without json output",
        stderr="Error details",
    )

    result = _run_environment(env, case_path)

    assert result["worker_status"] == "failed"
    assert result["exit_code"] == 1
    assert "comparison" not in result
    assert result["stderr"] == "Error details"


@patch("subprocess.run")
def test_run_environment_timeout(mock_run: MagicMock, tmp_path: Path) -> None:
    env = MatrixEnvironment(name="test_env", jax="0.4.30", timeout_s=10.0)
    case_path = tmp_path / "case.json"

    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd="uv run ...", timeout=10.0, output="partial stdout", stderr="timeout stderr"
    )

    result = _run_environment(env, case_path)

    assert result["worker_status"] == "timeout"
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "timeout stderr"
    assert result["environment"]["name"] == "test_env"


@patch("xlaxiom.matrix.load_case")
@patch("xlaxiom.matrix.load_config")
@patch("xlaxiom.matrix._run_environment")
@patch("xlaxiom.matrix.save_json")
def test_run_matrix_end_to_end(
    mock_save_json: MagicMock,
    mock_run_env: MagicMock,
    mock_load_config: MagicMock,
    mock_load_case: MagicMock,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    case_path = tmp_path / "case.json"
    output_path = tmp_path / "results.json"

    env1 = MatrixEnvironment(name="env1", jax="0.4.30")
    env2 = MatrixEnvironment(name="env2", jax="0.4.35")
    mock_load_config.return_value = (env1, env2)

    res1 = {
        "environment": {"name": "env1", "jax": "0.4.30", "timeout_s": 90.0},
        "worker_status": "ok",
        "comparison": {"classification": "pass"},
    }
    res2 = {
        "environment": {"name": "env2", "jax": "0.4.35", "timeout_s": 90.0},
        "worker_status": "ok",
        "comparison": {"classification": "numerical_discrepancy"},
    }
    mock_run_env.side_effect = [res1, res2]

    report = run_matrix(config_path, case_path, output_path)

    mock_load_case.assert_called_once_with(case_path.resolve())
    mock_load_config.assert_called_once_with(config_path)
    assert len(report["results"]) == 2
    assert "pairwise" in report
    assert len(report["pairwise"]) == 1
    assert report["pairwise"][0]["older"] == "env1"
    assert report["pairwise"][0]["newer"] == "env2"
    assert report["pairwise"][0]["classification"] == "new_regression"

    mock_save_json.assert_called_once_with(output_path, report)


@patch("xlaxiom.matrix.load_case")
@patch("xlaxiom.pipeline.run_comparison")
def test_worker_execution(
    mock_run_comparison: MagicMock,
    mock_load_case: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_case = MagicMock()
    mock_case.workload = "test_workload"
    mock_case.comparison_mode = "eager"
    mock_case.fault = None
    mock_load_case.return_value = mock_case

    mock_result = MagicMock()
    mock_result.to_dict.return_value = {"status": "success", "metric": 0.99}
    mock_run_comparison.return_value = mock_result

    exit_code = _worker("/path/to/case.json")

    assert exit_code == 0
    mock_load_case.assert_called_once_with("/path/to/case.json")
    mock_run_comparison.assert_called_once_with("test_workload", "eager", None)

    captured = capsys.readouterr()
    printed_json = json.loads(captured.out.strip())
    assert printed_json == {"metric": 0.99, "status": "success"}


@patch("xlaxiom.matrix._worker")
def test_main_worker_invocation(mock_worker: MagicMock) -> None:
    mock_worker.return_value = 0
    code = main(["_worker", "/path/to/case.json"])
    assert code == 0
    mock_worker.assert_called_once_with("/path/to/case.json")


def test_main_invalid_invocation() -> None:
    with pytest.raises(SystemExit, match="matrix module is invoked by xlaxiom matrix"):
        main([])

    with pytest.raises(SystemExit, match="matrix module is invoked by xlaxiom matrix"):
        main(["invalid_arg"])


def test_matrix_environment_version_validation_valid() -> None:
    env = MatrixEnvironment(name="valid_env", jax="0.4.30")
    assert env.jax == "0.4.30"
    env2 = MatrixEnvironment(name="valid_env2", jax="0.4.35-dev1")
    assert env2.jax == "0.4.35-dev1"


def test_matrix_environment_version_validation_invalid() -> None:
    invalid_versions = [
        "--version",
        "0.4.30; rm -rf /",
        "0.4.30 --help",
        " 0.4.30",
        "-jax",
        "0.4.30\n",
        "",
    ]
    for version in invalid_versions:
        with pytest.raises(ValueError, match="Invalid or unsafe JAX version specifier"):
            MatrixEnvironment(name="bad_env", jax=version)
