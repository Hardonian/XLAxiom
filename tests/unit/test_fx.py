"""Unit tests for PyTorch FX graph adapter (Priority 67)."""

import pytest

try:
    import torch
    import torch.fx

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from xlaxiom.fx import workload_from_fx, workload_to_fx
from xlaxiom.model import InputSpec, Node, Workload


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for FX tests")
def test_fx_export_and_execution() -> None:
    workload = Workload(
        identifier="fx-test-case",
        seed=1,
        inputs=(
            InputSpec("x", shape=(2, 4), dtype="float32"),
            InputSpec("y", shape=(2, 4), dtype="float32"),
        ),
        nodes=(
            Node(op="add", inputs=("x", "y"), output="sum"),
            Node(op="tanh", inputs=("sum",), output="out"),
        ),
        outputs=("out",),
    )

    gm = workload_to_fx(workload)
    assert isinstance(gm, torch.fx.GraphModule)

    # Execute gm with concrete PyTorch tensors
    tx = torch.ones(2, 4)
    ty = torch.ones(2, 4) * 2
    res = gm(tx, ty)
    expected = torch.tanh(tx + ty)
    assert torch.allclose(res, expected)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for FX tests")
def test_fx_ingestion() -> None:
    # Build a simple FX graph via symbolic tracing of a Python function
    def sample_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sin(a * b)

    gm = torch.fx.symbolic_trace(sample_fn)
    workload = workload_from_fx(
        gm,
        input_shapes={"a": (4, 4), "b": (4, 4)},
        identifier="ingested-sin-mul",
    )

    assert workload.identifier == "ingested-sin-mul"
    assert len(workload.inputs) == 2
    assert len(workload.nodes) == 2
    assert workload.nodes[0].op == "mul"
    assert workload.nodes[1].op == "sin"
