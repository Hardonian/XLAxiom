"""Unit tests for StableHLO MLIR syntax verification and roundtrip validation."""

from __future__ import annotations

from xlaxiom.model import InputSpec, Node, Workload
from xlaxiom.stablehlo import export_stablehlo, verify_stablehlo_roundtrip


def _sample_workload() -> Workload:
    inp1 = InputSpec("x", (2, 3), "float32")
    inp2 = InputSpec("y", (2, 3), "float32")
    node1 = Node("add", ("x", "y"), "added")
    node2 = Node("tanh", ("added",), "out")
    return Workload(
        identifier="stablehlo-verify-workload",
        seed=101,
        inputs=(inp1, inp2),
        nodes=(node1, node2),
        outputs=("out",),
    )


def test_verify_stablehlo_roundtrip_valid() -> None:
    workload = _sample_workload()
    mlir = export_stablehlo(workload)
    assert "stablehlo.add" in mlir
    assert "stablehlo.tanh" in mlir

    audit = verify_stablehlo_roundtrip(workload)
    assert audit["verified"] is True
    assert audit["syntax_valid"] is True
    assert audit["return_valid"] is True
    assert audit["ssa_count"] == 4  # %arg0, %arg1, %added, %out
    assert "stablehlo" in audit["dialects_used"]
    assert audit["workload_digest"] == workload.digest()
