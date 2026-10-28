"""Versioned, declarative workload model and safe operation registry."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

# Backward-compatible registry views; new code should import operation metadata directly.
from .operations import BINARY_OPS as BINARY_OPS
from .operations import OPERATION_SPECS
from .operations import OPS as OPS
from .operations import UNARY_OPS as UNARY_OPS

try:
    import ml_dtypes  # noqa: F401

    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False

LOW_PRECISION_DTYPES: set[str] = {"float16"}
if HAS_ML_DTYPES:
    LOW_PRECISION_DTYPES |= {"bfloat16", "float8_e4m3fn", "float8_e5m2"}

SCHEMA_VERSION = "1.0"
DTYPES = {"float32", "float64", "int32", "int64"} | LOW_PRECISION_DTYPES


def _get_finfo(dtype: np.dtype[Any]) -> Any:
    """Return floating point limits, supporting standard NumPy and ml_dtypes."""
    try:
        return np.finfo(dtype)
    except (ValueError, TypeError):
        if HAS_ML_DTYPES:
            import ml_dtypes

            return ml_dtypes.finfo(dtype)
        raise


class ValidationError(ValueError):
    """A declarative workload violates the bounded schema."""


def _is_plain_int(value: object) -> bool:
    """Return whether ``value`` is an integer JSON value, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _integer_sequence(value: object, field: str) -> tuple[int, ...]:
    """Read a JSON integer array without silently coercing malformed values."""
    if not isinstance(value, (list, tuple)) or not all(_is_plain_int(item) for item in value):
        raise ValidationError(f"{field} must be an array of integers")
    return tuple(cast(int, item) for item in value)


@dataclass(frozen=True)
class ResourceLimits:
    """Conservative, pre-materialization limits for one declarative workload."""

    max_rank: int = 6
    max_elements_per_tensor: int = 100_000
    max_tensor_bytes: int = 4_000_000
    max_intermediate_bytes: int = 16_000_000
    max_cumulative_bytes: int = 32_000_000


DEFAULT_LIMITS = ResourceLimits()


def _checked_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension < 1 or result > (2**63 - 1) // dimension:
            raise ValidationError("shape element count overflows the supported bound")
        result *= dimension
    return result


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    value_mode: Literal[
        "random", "zeros", "ones", "finite_edges", "extreme_finite", "signed_zero", "nonfinite"
    ] = "random"

    def validate(self, limits: ResourceLimits) -> None:
        if not self.name.isidentifier():
            raise ValidationError(f"invalid input name: {self.name!r}")
        if self.dtype not in DTYPES:
            raise ValidationError(f"unsupported dtype: {self.dtype}")
        if self.value_mode not in {
            "random",
            "zeros",
            "ones",
            "finite_edges",
            "extreme_finite",
            "signed_zero",
            "nonfinite",
        }:
            raise ValidationError(f"unsupported input value mode: {self.value_mode}")
        if self.value_mode == "nonfinite" and not self.dtype.startswith("float"):
            raise ValidationError("nonfinite input mode requires a floating-point dtype")
        if self.value_mode == "signed_zero" and not self.dtype.startswith("float"):
            raise ValidationError("signed_zero input mode requires a floating-point dtype")
        if len(self.shape) > limits.max_rank or any(
            not _is_plain_int(d) or d < 1 for d in self.shape
        ):
            raise ValidationError(f"input {self.name} has invalid shape {self.shape}")
        elements = _checked_product(self.shape)
        if (
            elements > limits.max_elements_per_tensor
            or elements * np.dtype(self.dtype).itemsize > limits.max_tensor_bytes
        ):
            raise ValidationError(f"input {self.name} exceeds element budget")


@dataclass(frozen=True)
class Node:
    op: str
    inputs: tuple[str, ...]
    output: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate_basic(self) -> None:
        spec = OPERATION_SPECS.get(self.op)
        if spec is None:
            raise ValidationError(f"operation {self.op!r} is not in the safe registry")
        if not self.output.isidentifier():
            raise ValidationError(f"invalid output name: {self.output!r}")
        if not isinstance(self.params, dict):
            raise ValidationError(f"{self.op} parameters must be an object")
        if len(self.inputs) != spec.arity:
            raise ValidationError(f"{self.op} expects {spec.arity} inputs")


@dataclass(frozen=True)
class ComparisonPolicy:
    identifier: str = "dtype-aware-v1"
    rtol: float = 2e-5
    atol: float = 2e-6
    equal_nan: bool = True
    check_signed_zero: bool = True

    def validate(self) -> None:
        if not self.identifier or not math.isfinite(self.rtol) or not math.isfinite(self.atol):
            raise ValidationError("comparison policy must have finite tolerances and an identifier")
        if self.rtol < 0 or self.atol < 0:
            raise ValidationError("comparison tolerances must be non-negative")


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

    def validate(self, limits: ResourceLimits = DEFAULT_LIMITS) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"unsupported schema version {self.schema_version}")
        if not self.identifier or len(self.nodes) > 64 or len(self.inputs) > 16:
            raise ValidationError("identifier is missing or workload complexity exceeds limit")
        self.policy.validate()
        known_shapes: dict[str, tuple[int, ...]] = {}
        known_dtypes: dict[str, np.dtype[Any]] = {}
        cumulative_bytes = 0
        for spec in self.inputs:
            spec.validate(limits)
            if spec.name in known_shapes:
                raise ValidationError(f"duplicate value name {spec.name}")
            dtype = np.dtype(spec.dtype)
            size = _checked_product(spec.shape) * dtype.itemsize
            cumulative_bytes += size
            known_shapes[spec.name] = spec.shape
            known_dtypes[spec.name] = dtype
        for node in self.nodes:
            node.validate_basic()
            if node.output in known_shapes or any(name not in known_shapes for name in node.inputs):
                raise ValidationError(
                    f"node {node.output} has duplicate or forward input reference"
                )
            shape = infer_shape(node, known_shapes)
            dtype = infer_dtype(node, known_dtypes)
            if len(shape) > limits.max_rank:
                raise ValidationError(f"intermediate {node.output} exceeds rank budget")
            allocation = _checked_product(shape) * dtype.itemsize
            if (
                _checked_product(shape) > limits.max_elements_per_tensor
                or allocation > limits.max_tensor_bytes
            ):
                raise ValidationError(
                    f"intermediate {node.output} exceeds tensor allocation budget"
                )
            if allocation > limits.max_intermediate_bytes:
                raise ValidationError(
                    f"intermediate {node.output} exceeds intermediate allocation budget"
                )
            cumulative_bytes += allocation
            if cumulative_bytes > limits.max_cumulative_bytes:
                raise ValidationError("workload exceeds cumulative allocation budget")
            known_shapes[node.output] = shape
            known_dtypes[node.output] = dtype
        if not self.outputs or any(name not in known_shapes for name in self.outputs):
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
            if not isinstance(data, dict):
                raise ValidationError("workload must be a JSON object")
            if not _is_plain_int(data["seed"]):
                raise ValidationError("workload seed must be an integer")
            policy = ComparisonPolicy(**data.get("policy", {}))
            workload = cls(
                identifier=data["identifier"],
                seed=data["seed"],
                inputs=tuple(
                    InputSpec(
                        name=item["name"],
                        shape=_integer_sequence(item["shape"], "input shape"),
                        dtype=item.get("dtype", "float32"),
                        value_mode=cast(
                            Literal[
                                "random",
                                "zeros",
                                "ones",
                                "finite_edges",
                                "extreme_finite",
                                "signed_zero",
                                "nonfinite",
                            ],
                            item.get("value_mode", "random"),
                        ),
                    )
                    for item in data["inputs"]
                ),
                nodes=tuple(
                    Node(
                        op=item["op"],
                        inputs=tuple(item["inputs"]),
                        output=item["output"],
                        params=item.get("params", {}),
                    )
                    for item in data["nodes"]
                ),
                outputs=tuple(data["outputs"]),
                schema_version=data.get("schema_version", SCHEMA_VERSION),
                policy=policy,
                metadata=data.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed workload: {exc}") from exc
        workload.validate()
        return workload


def infer_shape(node: Node, known: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    shapes = [known[name] for name in node.inputs]
    spec = OPERATION_SPECS.get(node.op)
    if spec is None:
        raise ValidationError(f"shape inference unavailable for {node.op}")
    try:
        if spec.kind == "elementwise":
            _require_no_params(node)
            return shapes[0] if spec.arity == 1 else _broadcast_shape(*shapes)
        if spec.kind == "matmul":
            _require_no_params(node)
            return _matmul_shape(shapes[0], shapes[1])
        if spec.kind == "reshape":
            _require_parameter_keys(node, {"shape"}, required={"shape"})
            shape = _integer_sequence(node.params["shape"], "reshape shape")
            if _checked_product(shape) != _checked_product(shapes[0]) or any(v < 1 for v in shape):
                raise ValidationError("reshape does not preserve element count")
            return shape
        if spec.kind == "transpose":
            _require_parameter_keys(node, {"axes"}, required=set())
            axes = (
                tuple(reversed(range(len(shapes[0]))))
                if "axes" not in node.params
                else _integer_sequence(node.params["axes"], "transpose axes")
            )
            if sorted(axes) != list(range(len(shapes[0]))):
                raise ValidationError("transpose axes are not a permutation")
            return tuple(shapes[0][axis] for axis in axes)
        if spec.kind == "reduction":
            _require_parameter_keys(node, {"axis", "keepdims"}, required=set())
            keepdims = node.params.get("keepdims", False)
            if not isinstance(keepdims, bool):
                raise ValidationError("reduction keepdims must be a boolean")
            rank = len(shapes[0])
            axes_value = node.params.get("axis")
            axes = _reduction_axes(axes_value, rank)
            normalized = tuple(axis + rank if axis < 0 else axis for axis in axes)
            if len(set(normalized)) != len(normalized):
                raise ValidationError("duplicate reduction axis")
            if keepdims:
                return tuple(1 if i in normalized else dim for i, dim in enumerate(shapes[0]))
            return tuple(dim for i, dim in enumerate(shapes[0]) if i not in normalized)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"invalid {node.op} parameters: {exc}") from exc
    raise ValidationError(f"shape inference unavailable for {node.op}")


_DTYPE_CACHE: dict[tuple[str, tuple[str, ...], bool], np.dtype[Any]] = {}


def infer_dtype(node: Node, known: dict[str, np.dtype[Any]]) -> np.dtype[Any]:
    """Return the conservative NumPy result dtype for a validated operation.

    The schema must account for the widest portable result, not just the input item
    width.  For example, NumPy promotes an ``int32``/``float32`` add to ``float64``
    and widens an ``int32`` reduction to ``int64``.  Tracking that result prevents a
    workload from bypassing an allocation limit through dtype promotion.
    """
    spec = OPERATION_SPECS.get(node.op)
    if spec is None:
        raise ValidationError(f"dtype inference unavailable for {node.op}")
    input_dtypes = tuple(known[name] for name in node.inputs)
    if spec.kind in {"reshape", "transpose"}:
        return input_dtypes[0]
    if spec.kind == "matmul":
        return np.dtype(np.result_type(*input_dtypes))

    keepdims = bool(node.params.get("keepdims", False))
    cache_key = (node.op, tuple(d.name for d in input_dtypes), keepdims)
    cached = _DTYPE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    operation = getattr(np, spec.api_name)
    probes = tuple(np.empty((), dtype=dtype) for dtype in input_dtypes)
    try:
        if spec.kind == "reduction":
            result = operation(probes[0], axis=None, keepdims=keepdims)
        else:
            result = operation(*probes)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid {node.op} dtype combination: {exc}") from exc
    res_dtype = np.asarray(result).dtype
    _DTYPE_CACHE[cache_key] = res_dtype
    return res_dtype


def _require_parameter_keys(node: Node, allowed: set[str], required: set[str]) -> None:
    """Reject unexpected or missing parameters before an execution backend sees them."""
    keys = set(node.params)
    unknown = keys - allowed
    missing = required - keys
    if unknown:
        raise ValidationError(f"{node.op} has unsupported parameters: {sorted(unknown)}")
    if missing:
        raise ValidationError(f"{node.op} is missing required parameters: {sorted(missing)}")


def _require_no_params(node: Node) -> None:
    _require_parameter_keys(node, set(), required=set())


def _reduction_axes(value: object, rank: int) -> tuple[int, ...]:
    """Validate a reduction axis exactly as the execution backends will receive it."""
    if value is None:
        return tuple(range(rank))
    if _is_plain_int(value):
        axes: tuple[int, ...] = (cast(int, value),)
    else:
        axes = _integer_sequence(value, "reduction axis")
    if rank == 0:
        raise ValidationError("a scalar reduction cannot name an explicit axis")
    if not axes:
        raise ValidationError("reduction axis must not be empty")
    if any(axis < -rank or axis >= rank for axis in axes):
        raise ValidationError(f"reduction axis must be between {-rank} and {rank - 1}")
    return axes


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    result: tuple[int, ...] = ()
    for shape in shapes:
        if not result:
            result = shape
            continue
        if result == shape or not shape:
            continue
        r_len = len(result)
        s_len = len(shape)
        if r_len == s_len:
            out = list(result)
            for i in range(r_len):
                r = result[i]
                s = shape[i]
                if r == s or s == 1:
                    continue
                if r == 1:
                    out[i] = s
                else:
                    raise ValidationError(f"incompatible broadcast shapes: {shapes}")
            result = tuple(out)
            continue

        out_len = r_len if r_len > s_len else s_len
        out = [0] * out_len
        for i in range(1, out_len + 1):
            r = result[-i] if i <= r_len else 1
            s = shape[-i] if i <= s_len else 1
            if r == s:
                out[-i] = r
            elif r == 1:
                out[-i] = s
            elif s == 1:
                out[-i] = r
            else:
                raise ValidationError(f"incompatible broadcast shapes: {shapes}")
        result = tuple(out)
    return result


def _matmul_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    if not left or not right:
        raise ValidationError("matmul requires tensors with rank at least one")
    left_contract = left[-1]
    right_contract = right[0] if len(right) == 1 else right[-2]
    if left_contract != right_contract:
        raise ValidationError(
            f"matmul contracting dimensions differ: {left_contract} and {right_contract}"
        )
    left_batch = left[:-2] if len(left) > 1 else ()
    right_batch = right[:-2] if len(right) > 1 else ()
    batch = _broadcast_shape(left_batch, right_batch)
    left_matrix = () if len(left) == 1 else (left[-2],)
    right_matrix = () if len(right) == 1 else (right[-1],)
    return batch + left_matrix + right_matrix


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
                if spec.value_mode == "finite_edges":
                    flat = value.reshape(-1)
                    edges = np.asarray([0, 1, -1, 8], dtype=dtype)
                    flat[: min(flat.size, len(edges))] = edges[: flat.size]
            else:
                value = rng.normal(0.0, 1.0, size=spec.shape).astype(dtype)
                flat = value.reshape(-1)
                info = _get_finfo(dtype)
                if spec.value_mode == "finite_edges":
                    max_half = float(info.max) / 2.0
                    tiny_val = float(info.tiny)
                    edges = np.array(
                        [0.0, -0.0, tiny_val, min(1e10, max_half), 1.0, -1.0],
                        dtype=dtype,
                    )
                    flat[: min(flat.size, len(edges))] = edges[: flat.size]
                elif spec.value_mode == "extreme_finite":
                    edges = np.array(
                        [
                            info.max / 4,
                            -info.max / 4,
                            info.tiny,
                            -info.tiny,
                            info.smallest_subnormal,
                            -info.smallest_subnormal,
                        ],
                        dtype=dtype,
                    )
                    flat[: min(flat.size, len(edges))] = edges[: flat.size]
                elif spec.value_mode == "signed_zero":
                    edges = np.array([0.0, -0.0, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
                    flat[: min(flat.size, len(edges))] = edges[: flat.size]
                elif spec.value_mode == "nonfinite":
                    edges = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0], dtype=dtype)
                    flat[: min(flat.size, len(edges))] = edges[: flat.size]
        values[spec.name] = value
    return values
