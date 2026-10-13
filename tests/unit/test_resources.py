from importlib.metadata import metadata
from importlib.resources import files

import pytest

from xlaxiom.model import InputSpec, Node, ValidationError, Workload


def test_distribution_exposes_pep561_typing_marker() -> None:
    marker = files("xlaxiom").joinpath("py.typed")

    assert marker.is_file()
    assert "Typing :: Typed" in metadata("xlaxiom").get_all("Classifier", failobj=[])


def test_broadcasting_explosion_is_rejected_before_materialization() -> None:
    workload = Workload(
        "explosive",
        0,
        (InputSpec("x", (100_000, 1)), InputSpec("y", (1, 100_000))),
        (Node("add", ("x", "y"), "out"),),
        ("out",),
    )
    with pytest.raises(ValidationError, match="allocation budget"):
        workload.validate()


def test_mismatched_matmul_is_rejected_symbolically() -> None:
    workload = Workload(
        "bad-matmul",
        0,
        (InputSpec("x", (2, 3)), InputSpec("y", (4, 2))),
        (Node("matmul", ("x", "y"), "out"),),
        ("out",),
    )
    with pytest.raises(ValidationError, match="contracting"):
        workload.validate()
