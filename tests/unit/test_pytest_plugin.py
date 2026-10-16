from pathlib import Path
from typing import Any

import pytest

from xlaxiom.model import Workload


def test_workload_factory_fixture(xlaxiom_workload_factory: Any) -> None:
    workload = xlaxiom_workload_factory(seed=99)
    assert isinstance(workload, Workload)
    assert workload.seed == 99


def test_evidence_dir_fixture(xlaxiom_evidence_dir: Path) -> None:
    assert xlaxiom_evidence_dir.is_dir()
    test_file = xlaxiom_evidence_dir / "test.txt"
    test_file.write_text("ok", encoding="utf-8")
    assert test_file.exists()


@pytest.mark.xlaxiom(rtol=1e-5, atol=1e-6)
def test_marked_test_execution() -> None:
    # Confirms custom marker works seamlessly in test execution
    assert True
