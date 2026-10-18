"""Unit tests for StableHLO dialect exporter (Priority 63)."""

from xlaxiom.model import InputSpec, Node, Workload
from xlaxiom.stablehlo import export_stablehlo


def test_export_stablehlo_elementwise() -> None:
    workload = Workload(
        identifier="elementwise-test",
        seed=1,
        inputs=(
            InputSpec("x", shape=(4, 16), dtype="float32"),
            InputSpec("y", shape=(4, 16), dtype="float32"),
        ),
        nodes=(
            Node(op="add", inputs=("x", "y"), output="sum"),
            Node(op="exp", inputs=("sum",), output="exp_out"),
        ),
        outputs=("exp_out",),
    )

    mlir = export_stablehlo(workload)
    assert "module @xlaxiom" in mlir
    assert (
        "func.func @main(%arg0: tensor<4x16xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x16xf32>"
        in mlir
    )
    assert "%sum = stablehlo.add %arg0, %arg1 : tensor<4x16xf32>" in mlir
    assert "%exp_out = stablehlo.exponential %sum : tensor<4x16xf32>" in mlir
    assert "return %exp_out : tensor<4x16xf32>" in mlir


def test_export_stablehlo_matmul_and_reshape() -> None:
    workload = Workload(
        identifier="matmul-test",
        seed=2,
        inputs=(
            InputSpec("a", shape=(2, 4), dtype="float32"),
            InputSpec("b", shape=(4, 8), dtype="float32"),
        ),
        nodes=(
            Node(op="matmul", inputs=("a", "b"), output="c"),
            Node(op="reshape", inputs=("c",), output="c_flat", params={"shape": [16]}),
        ),
        outputs=("c_flat",),
    )

    mlir = export_stablehlo(workload)
    assert "stablehlo.dot_general" in mlir
    assert "contracting_dims = [1] x [0]" in mlir
    assert "stablehlo.reshape" in mlir
    assert "return %c_flat : tensor<16xf32>" in mlir


def test_export_stablehlo_reduce() -> None:
    workload = Workload(
        identifier="reduce-test",
        seed=3,
        inputs=(InputSpec("x", shape=(8, 16), dtype="float32"),),
        nodes=(Node(op="sum", inputs=("x",), output="total"),),
        outputs=("total",),
    )

    mlir = export_stablehlo(workload)
    assert "stablehlo.reduce" in mlir
    assert "reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>)" in mlir
    assert "stablehlo.return" in mlir
