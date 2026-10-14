from importlib.metadata import metadata
from importlib.resources import files

import pytest

from xlaxiom.model import InputSpec, Node, ResourceLimits, ValidationError, Workload


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


def test_dtype_promotion_cannot_bypass_the_intermediate_allocation_limit() -> None:
    """``int32 + float32`` is a float64 NumPy result, not a four-byte tensor."""
    workload = Workload(
        "promoted-result",
        0,
        (InputSpec("integers", (1,), "int32"), InputSpec("floats", (1,), "float32")),
        (Node("add", ("integers", "floats"), "out"),),
        ("out",),
    )
    limits = ResourceLimits(
        max_tensor_bytes=4,
        max_intermediate_bytes=4,
        max_cumulative_bytes=16,
    )

    with pytest.raises(ValidationError, match="intermediate out exceeds tensor allocation budget"):
        workload.validate(limits)
