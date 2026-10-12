"""Versioned, declarative workload model and safe operation registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = "1.0"
DTYPES = {"float32", "float64", "int32", "int64"}
UNARY_OPS = {"neg", "sin", "cos", "exp", "tanh", "abs"}
BINARY_OPS = {"add", "sub", "mul", "div", "matmul"}
OPS = UNARY_OPS | BINARY_OPS | {"reshape", "transpose", "sum", "mean"}


class ValidationError(ValueError):
    """A declarative workload violates the bounded schema."""


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    value_mode: Literal["random", "zeros", "ones"] = "random"

    def validate(self, max_elements: int) -> None:
        if not self.name.isidentifier():
            raise ValidationError(f"invalid input name: {self.name!r}")
        if self.dtype not in DTYPES:
            raise ValidationError(f"unsupported dtype: {self.dtype}")
        if any(not isinstance(d, int) or d < 1 for d in self.shape):
            raise ValidationError(f"input {self.name} has invalid shape {self.shape}")
        if int(np.prod(self.shape, dtype=np.int64)) > max_elements:
            raise ValidationError(f"input {self.name} exceeds element budget")


@dataclass(frozen=True)
class Node:
    op: str
    inputs: tuple[str, ...]
    output: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate_basic(self) -> None:
        if self.op not in OPS:
            raise ValidationError(f"operation {self.op!r} is not in the safe registry")
        if not self.output.isidentifier():
            raise ValidationError(f"invalid output name: {self.output!r}")
        expected = 1 if self.op in UNARY_OPS | {"reshape", "transpose", "sum", "mean"} else 2
        if len(self.inputs) != expected:
            raise ValidationError(f"{self.op} expects {expected} inputs")


@dataclass(frozen=True)
class ComparisonPolicy:
    identifier: str = "dtype-aware-v1"
    rtol: float = 2e-5
    atol: float = 2e-6
    equal_nan: bool = True
    check_signed_zero: bool = True


@dataclass(frozen=True)
class Workload:
    identifier: str
    seed: int
    inputs: tuple[InputSpec, ...]
    nodes: tuple[Node, ...]
    outputs: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    policy: ComparisonPolicy = field(default_factory=ComparisonPolicy)
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self, max_elements: int = 100_000) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema version {self.schema_version}")
        if not self.identifier or len(self.nodes) > 64 or len(self.inputs) > 16:
            raise ValidationError("identifier is missing or workload complexity exceeds limit")
        known: dict[str, tuple[int, ...]] = {}
        for spec in self.inputs:
            spec.validate(max_elements)
            if spec.name in known:
                raise ValidationError(f"duplicate value name {spec.name}")
            known[spec.name] = spec.shape
        for node in self.nodes:
            node.validate_basic()
            if node.output in known or any(name not in known for name in node.inputs):
                raise ValidationError(
                    f"node {node.output} has duplicate or forward input reference"
                )
            known[node.output] = infer_shape(node, known)
        if not self.outputs or any(name not in known for name in self.outputs):
            raise ValidationError("workload must name at least one defined output")

    def canonical_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Dataclass conversion preserves tuples as tuples; JSON makes them canonical lists.
        return data

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workload:
        try:
            policy = ComparisonPolicy(**data.get("policy", {}))
            workload = cls(
                identifier=str(data["identifier"]),
                seed=int(data["seed"]),
                inputs=tuple(
                    InputSpec(
                        name=str(item["name"]),
                        shape=tuple(int(v) for v in item["shape"]),
                        dtype=str(item.get("dtype", "float32")),
                        value_mode=cast(
                            Literal["random", "zeros", "ones"], item.get("value_mode", "random")
                        ),
                    )
                    for item in data["inputs"]
                ),
                nodes=tuple(
                    Node(
                        op=str(item["op"]),
                        inputs=tuple(str(v) for v in item["inputs"]),
                        output=str(item["output"]),
                        params=dict(item.get("params", {})),
                    )
                    for item in data["nodes"]
                ),
                outputs=tuple(str(v) for v in data["outputs"]),
                schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
                policy=policy,
                metadata={str(k): str(v) for k, v in data.get("metadata", {}).items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed workload: {exc}") from exc
        workload.validate()
        return workload


def infer_shape(node: Node, known: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    shapes = [known[name] for name in node.inputs]
    try:
        if node.op in {"add", "sub", "mul", "div"}:
            return tuple(np.broadcast_shapes(*shapes))
        if node.op == "matmul":
            return tuple(np.matmul(np.empty(shapes[0]), np.empty(shapes[1])).shape)
        if node.op in UNARY_OPS:
            return shapes[0]
        if node.op == "reshape":
            shape = tuple(int(v) for v in node.params["shape"])
            if int(np.prod(shape)) != int(np.prod(shapes[0])) or any(v < 1 for v in shape):
                raise ValidationError("reshape does not preserve element count")
            return shape
        if node.op == "transpose":
            axes = tuple(
                int(v) for v in node.params.get("axes", tuple(reversed(range(len(shapes[0])))))
            )
            if sorted(axes) != list(range(len(shapes[0]))):
                raise ValidationError("transpose axes are not a permutation")
            return tuple(shapes[0][axis] for axis in axes)
        if node.op in {"sum", "mean"}:
            axes_value = node.params.get("axis")
            axes = (
                tuple(range(len(shapes[0])))
                if axes_value is None
                else tuple(int(v) for v in axes_value)
            )
            normalized = tuple(axis % len(shapes[0]) for axis in axes)
            if len(set(normalized)) != len(normalized):
                raise ValidationError("duplicate reduction axis")
            if node.params.get("keepdims", False):
                return tuple(1 if i in normalized else dim for i, dim in enumerate(shapes[0]))
            return tuple(dim for i, dim in enumerate(shapes[0]) if i not in normalized)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"invalid {node.op} parameters: {exc}") from exc
    raise ValidationError(f"shape inference unavailable for {node.op}")


def materialize_inputs(workload: Workload) -> dict[str, NDArray[np.generic]]:
    """Generate deterministic bounded inputs without evaluating workload-provided code."""
    values: dict[str, NDArray[np.generic]] = {}
    seed_sequence = np.random.SeedSequence(workload.seed)
    for spec, child_seed in zip(
        workload.inputs, seed_sequence.spawn(len(workload.inputs)), strict=True
    ):
        dtype = np.dtype(spec.dtype)
        if spec.value_mode == "zeros":
            value = np.zeros(spec.shape, dtype=dtype)
        elif spec.value_mode == "ones":
            value = np.ones(spec.shape, dtype=dtype)
        else:
            rng = np.random.default_rng(child_seed)
            if np.issubdtype(dtype, np.integer):
                value = rng.integers(-8, 9, size=spec.shape, dtype=dtype)
            else:
                value = rng.normal(0.0, 1.0, size=spec.shape).astype(dtype)
                # Reproducible edge values, without forcing invalid values into every operation.
                flat = value.reshape(-1)
                if flat.size >= 6:
                    flat[:6] = np.array([0.0, -0.0, 1e-20, 1e10, 1.0, -1.0], dtype=dtype)
        values[spec.name] = value
    return values
