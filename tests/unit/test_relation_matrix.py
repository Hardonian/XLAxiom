from pathlib import Path
from unittest.mock import MagicMock, patch

from xlaxiom.matrix import MatrixEnvironment, run_relation_matrix


@patch("xlaxiom.matrix.save_json")
@patch("xlaxiom.matrix._run_relation_environment")
@patch("xlaxiom.matrix.load_config")
@patch("xlaxiom.matrix.load_relational_case")
def test_relation_matrix_uses_checked_relational_artifacts(
    checked_case: MagicMock,
    environments: MagicMock,
    run_environment: MagicMock,
    save: MagicMock,
    tmp_path: Path,
) -> None:
    environments.return_value = (MatrixEnvironment("first", "0.5.0"),)
    run_environment.return_value = {"environment": {"name": "first"}, "worker_status": "ok"}

    report = run_relation_matrix(
        tmp_path / "matrix.json", tmp_path / "relation.json", tmp_path / "out.json"
    )

    checked_case.assert_called_once()
    assert report["kind"] == "relational"
    save.assert_called_once()
