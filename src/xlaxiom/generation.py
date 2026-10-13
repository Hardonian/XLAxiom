"""Deterministic valid workload generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .model import DTYPES, InputSpec, Node, Workload


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 0
    max_rank: int = 2
    max_dim: int = 8
    max_nodes: int = 8
    dtype: str = "float32"
    input_mode: Literal[
        "random", "zeros", "ones", "finite_edges", "extreme_finite", "signed_zero", "nonfinite"
    ] = "finite_edges"

    def validate(self) -> None:
        if not 1 <= self.max_rank <= 6:
            raise ValueError("max_rank must be between 1 and 6")
        if not 2 <= self.max_dim <= 32:
            raise ValueError("max_dim must be between 2 and 32")
        if not 1 <= self.max_nodes <= 64:
            raise ValueError("max_nodes must be between 1 and 64")
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported generation dtype {self.dtype!r}")
        if self.input_mode not in {
            "random",
            "zeros",
            "ones",
            "finite_edges",
            "extreme_finite",
            "signed_zero",
            "nonfinite",
        }:
            raise ValueError(f"unsupported generation input mode {self.input_mode!r}")
        if self.dtype.startswith("int") and self.input_mode in {"signed_zero", "nonfinite"}:
            raise ValueError(f"input mode {self.input_mode!r} requires a floating-point dtype")


def generate(config: GenerationConfig) -> Workload:
    """Systematically vary bounded elementwise, broadcast, reduction, and matmul motifs."""
    config.validate()
    rng = np.random.default_rng(config.seed)

    def dim() -> int:
        return int(rng.integers(2, config.max_dim + 1))

    def input_spec(name: str, shape: tuple[int, ...]) -> InputSpec:
        return InputSpec(name, shape, config.dtype, config.input_mode)

    if config.dtype.startswith("int"):
        motif = config.seed % 2
    else:
        motif = config.seed % (6 if config.max_rank >= 3 else 4)
    inputs: tuple[InputSpec, ...]
    nodes: list[Node]
    if motif == 0:
        shape = (dim(), dim())
        inputs = (input_spec("x", shape), input_spec("y", shape))
        tail = "abs" if config.dtype.startswith("int") else "tanh"
        nodes = [Node("add", ("x", "y"), "n0"), Node(tail, ("n0",), "out")]
    elif motif == 1 and config.dtype.startswith("int"):
        rows, inner, columns = dim(), dim(), dim()
        inputs = (
            input_spec("x", (rows, inner)),
            input_spec("y", (inner, columns)),
        )
        nodes = [Node("matmul", ("x", "y"), "out")]
    elif motif == 1:
        rows, columns = dim(), dim()
        inputs = (
            input_spec("x", (rows, 1)),
            input_spec("y", (1, columns)),
        )
        nodes = [
            Node("add", ("x", "y"), "n0"),
            Node("mean", ("n0",), "out", {"axis": [0], "keepdims": False}),
        ]
    elif motif == 2:
        rows, inner, columns = dim(), dim(), dim()
        inputs = (
            input_spec("x", (rows, inner)),
            input_spec("y", (inner, columns)),
        )
        nodes = [
            Node("matmul", ("x", "y"), "n0"),
            Node("transpose", ("n0",), "n1", {"axes": [1, 0]}),
            Node("sum", ("n1",), "out", {"axis": [0], "keepdims": False}),
        ]
    elif motif == 3:
        first, second = dim(), dim()
        inputs = (input_spec("x", (first, second)),)
        nodes = [
            Node("reshape", ("x",), "n0", {"shape": [first * second]}),
            Node("sin", ("n0",), "n1"),
            Node("reshape", ("n1",), "out", {"shape": [first, second]}),
        ]
    elif motif == 4:
        batch, rows, columns = dim(), dim(), dim()
        inputs = (
            input_spec("x", (batch, rows, 1)),
            input_spec("y", (1, rows, columns)),
        )
        nodes = [
            Node("add", ("x", "y"), "n0"),
            Node("mean", ("n0",), "out", {"axis": [1], "keepdims": False}),
        ]
    else:
        batch, rows, inner, columns = dim(), dim(), dim(), dim()
        inputs = (
            input_spec("x", (batch, rows, inner)),
            input_spec("y", (1, inner, columns)),
        )
        nodes = [
            Node("matmul", ("x", "y"), "n0"),
            Node("transpose", ("n0",), "n1", {"axes": [0, 2, 1]}),
            Node("sum", ("n1",), "out", {"axis": [1], "keepdims": False}),
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
