"""Closed operation metadata shared by validation and execution.

Keeping arity, execution spelling, and structural behavior in one place prevents a
new operation from being accepted by the portable workload schema but interpreted
differently by an execution adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

OperationKind = Literal["elementwise", "matmul", "reshape", "transpose", "reduction"]


@dataclass(frozen=True)
class OperationSpec:
    """Static contract for one operation in the safe declarative registry."""

    arity: int
    api_name: str
    kind: OperationKind


_OPERATION_SPECS = {
    "add": OperationSpec(2, "add", "elementwise"),
    "sub": OperationSpec(2, "subtract", "elementwise"),
    "mul": OperationSpec(2, "multiply", "elementwise"),
    "div": OperationSpec(2, "divide", "elementwise"),
    "matmul": OperationSpec(2, "matmul", "matmul"),
    "neg": OperationSpec(1, "negative", "elementwise"),
    "sin": OperationSpec(1, "sin", "elementwise"),
    "cos": OperationSpec(1, "cos", "elementwise"),
    "exp": OperationSpec(1, "exp", "elementwise"),
    "tanh": OperationSpec(1, "tanh", "elementwise"),
    "abs": OperationSpec(1, "abs", "elementwise"),
    "reshape": OperationSpec(1, "reshape", "reshape"),
    "transpose": OperationSpec(1, "transpose", "transpose"),
    "sum": OperationSpec(1, "sum", "reduction"),
    "mean": OperationSpec(1, "mean", "reduction"),
}

# A read-only mapping is an important part of the closed-registry guarantee: workload
# data can select an operation, but cannot extend the operations the host will run.
OPERATION_SPECS: Mapping[str, OperationSpec] = MappingProxyType(_OPERATION_SPECS)
OPS = frozenset(OPERATION_SPECS)
UNARY_OPS = frozenset(
    name for name, spec in OPERATION_SPECS.items() if spec.kind == "elementwise" and spec.arity == 1
)
BINARY_OPS = frozenset(
    name for name, spec in OPERATION_SPECS.items() if spec.kind == "elementwise" and spec.arity == 2
) | {"matmul"}
