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
    """Make a bounded graph; its seed is all that is needed to recreate inputs."""
    rng = np.random.default_rng(config.seed)
    rank = int(rng.integers(1, config.max_rank + 1))
    shape = tuple(int(rng.integers(2, config.max_dim + 1)) for _ in range(rank))
    inputs = (
        InputSpec("x", shape, config.dtype),
        InputSpec("y", shape, config.dtype),
        InputSpec("unused", shape, config.dtype),
    )
    nodes: list[Node] = [Node("add", ("x", "y"), "n0")]
    source = "n0"
    unary = ("sin", "tanh", "abs", "neg")
    for index in range(1, max(2, config.max_nodes - 1)):
        op = unary[int(rng.integers(0, len(unary)))]
        output = f"n{index}"
        nodes.append(Node(op, (source,), output))
        source = output
    # A valid deliberately dead branch exercises graph-branch minimization.
    nodes.append(Node("mul", ("unused", "unused"), "dead"))
    workload = Workload(
        identifier=f"generated-{config.seed}",
        seed=config.seed,
        inputs=inputs,
        nodes=tuple(nodes),
        outputs=(source,),
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
