"""Deterministic valid workload generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import InputSpec, Node, Workload


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 0
    max_rank: int = 2
    max_dim: int = 8
    max_nodes: int = 8
    dtype: str = "float32"


def generate(config: GenerationConfig) -> Workload:
    """Systematically vary bounded elementwise, broadcast, reduction, and matmul motifs."""
    rng = np.random.default_rng(config.seed)

    def dim() -> int:
        return int(rng.integers(2, config.max_dim + 1))

    motif = config.seed % 4
    if motif == 0:
        shape = (dim(), dim())
        inputs = (InputSpec("x", shape, config.dtype), InputSpec("y", shape, config.dtype))
        nodes = [Node("add", ("x", "y"), "n0"), Node("tanh", ("n0",), "out")]
    elif motif == 1:
        rows, columns = dim(), dim()
        inputs = (
            InputSpec("x", (rows, 1), config.dtype),
            InputSpec("y", (1, columns), config.dtype),
        )
        nodes = [
            Node("add", ("x", "y"), "n0"),
            Node("mean", ("n0",), "out", {"axis": [0], "keepdims": False}),
        ]
    elif motif == 2:
        rows, inner, columns = dim(), dim(), dim()
        inputs = (
            InputSpec("x", (rows, inner), config.dtype),
            InputSpec("y", (inner, columns), config.dtype),
        )
        nodes = [
            Node("matmul", ("x", "y"), "n0"),
            Node("transpose", ("n0",), "n1", {"axes": [1, 0]}),
            Node("sum", ("n1",), "out", {"axis": [0], "keepdims": False}),
        ]
    else:
        first, second = dim(), dim()
        inputs = (InputSpec("x", (first, second), config.dtype),)
        nodes = [
            Node("reshape", ("x",), "n0", {"shape": [first * second]}),
            Node("sin", ("n0",), "n1"),
            Node("reshape", ("n1",), "out", {"shape": [first, second]}),
        ]
    workload = Workload(
        identifier=f"generated-{config.seed}",
        seed=config.seed,
        inputs=tuple(inputs),
        nodes=tuple(nodes),
        outputs=("out",),
        metadata={"generator": "bounded-v1"},
    )
    workload.validate()
    return workload


def synthetic_control(seed: int = 0) -> Workload:
    """A deterministic, deliberately reducible workload for pipeline verification only."""
    workload = Workload(
        identifier=f"synthetic-control-{seed}",
        seed=seed,
        inputs=(
            InputSpec("x", (8, 8)),
            InputSpec("zero", (8, 8), value_mode="zeros"),
            InputSpec("unused", (8, 8)),
        ),
        nodes=(
            Node("add", ("x", "zero"), "a"),
            Node("tanh", ("a",), "b"),
            Node("neg", ("b",), "c"),
            Node("neg", ("c",), "out"),
            Node("mul", ("unused", "unused"), "dead"),
        ),
        outputs=("out",),
        metadata={"kind": "synthetic-control", "warning": "not a real compiler anomaly"},
    )
    workload.validate()
    return workload
