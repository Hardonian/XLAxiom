"""Unit tests for symbolic dimension modeling and dynamic shapes (Priority 62)."""

import pytest

from xlaxiom.model import InputSpec, Node, ValidationError, Workload
from xlaxiom.symbolic import (
    DynamicInputSpec,
    SymbolicDim,
    infer_symbolic_broadcast,
    instantiate_workload,
    resolve_symbolic_shape,
)


def test_symbolic_dim_validation() -> None:
    dim = SymbolicDim("batch", min_val=1, max_val=128)
    assert dim.name == "batch"
    assert dim.validate_value(64) == 64

    with pytest.raises(ValidationError, match="out of bounds"):
        dim.validate_value(256)

    with pytest.raises(ValidationError, match="must be an int"):
        dim.validate_value("64")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="valid identifier"):
        SymbolicDim("123bad")

    with pytest.raises(ValidationError, match="cannot be less than min_val"):
        SymbolicDim("dim", min_val=100, max_val=10)


def test_dynamic_input_spec() -> None:
    spec = DynamicInputSpec(name="x", shape=("batch", 32, "seq_len"), dtype="float32")
    resolved = spec.instantiate({"batch": 4, "seq_len": 128})
    assert resolved == (4, 32, 128)

    with pytest.raises(ValidationError, match="missing binding"):
        spec.instantiate({"batch": 4})


def test_resolve_symbolic_shape() -> None:
    batch = SymbolicDim("batch", 1, 64)
    shape = resolve_symbolic_shape(
        (batch, 16, "hidden"),
        {"batch": 8, "hidden": 256},
        dims_meta={"hidden": SymbolicDim("hidden", 64, 512)},
    )
    assert shape == (8, 16, 256)

    with pytest.raises(ValidationError, match="unbound symbolic dimension"):
        resolve_symbolic_shape((batch, "missing"), {"batch": 8})


def test_infer_symbolic_broadcast() -> None:
    # Matching symbolic dimensions
    assert infer_symbolic_broadcast(("batch", 1, "dim"), ("batch", 32, 1)) == (
        "batch",
        32,
        "dim",
    )
    # Broadcast with constant 1
    assert infer_symbolic_broadcast((1, "dim"), ("batch", "dim")) == ("batch", "dim")

    # Incompatible symbolic dimensions
    with pytest.raises(ValidationError, match="incompatible symbolic broadcast"):
        infer_symbolic_broadcast(("dim1",), ("dim2",))

    # Incompatible constant vs symbolic
    with pytest.raises(ValidationError, match="cannot broadcast symbolic dimension"):
        infer_symbolic_broadcast((4,), ("batch",))


def test_instantiate_workload() -> None:
    workload = Workload(
        identifier="symbolic-case",
        seed=42,
        inputs=(
            InputSpec("x", shape=("batch", 16), dtype="float32"),  # type: ignore[arg-type]
            InputSpec("y", shape=("batch", 16), dtype="float32"),  # type: ignore[arg-type]
        ),
        nodes=(Node(op="add", inputs=("x", "y"), output="z"),),
        outputs=("z",),
    )

    concrete = instantiate_workload(
        workload,
        bindings={"batch": 8},
        dims_meta={"batch": SymbolicDim("batch", 1, 32)},
    )

    assert concrete.identifier == "symbolic-case-instantiated"
    assert concrete.inputs[0].shape == (8, 16)
    assert concrete.inputs[1].shape == (8, 16)
    assert concrete.metadata["symbolic_bindings"] == "{'batch': 8}"
