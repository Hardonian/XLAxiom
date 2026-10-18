"""Symbolic dimension and dynamic shape modeling for compiler testing.

Enables compilation verification across parametric batch sizes, variable sequence
lengths, and dynamic rank constraints, detecting compiler loop-bound and polyhedral
optimizations bugs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest

from .model import ValidationError, Workload


@dataclass(frozen=True)
class SymbolicDim:
    """A bounded symbolic dimension in a dynamic workload."""

    name: str
    min_val: int = 1
    max_val: int = 4096

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValidationError(
                f"symbolic dimension name must be a valid identifier: {self.name!r}"
            )
        if self.min_val < 1:
            raise ValidationError(
                f"symbolic dimension min_val must be at least 1, got {self.min_val}"
            )
        if self.max_val < self.min_val:
            raise ValidationError(
                f"symbolic dimension max_val ({self.max_val}) cannot be less than "
                f"min_val ({self.min_val})"
            )

    def validate_value(self, value: int) -> int:
        """Verify that a concrete integer binding satisfies the dimension bounds."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(
                f"dimension {self.name!r} binding must be an int, got {type(value)}"
            )
        if not (self.min_val <= value <= self.max_val):
            raise ValidationError(
                f"dimension {self.name!r} value {value} out of bounds "
                f"[{self.min_val}, {self.max_val}]"
            )
        return value


@dataclass(frozen=True)
class DynamicInputSpec:
    """Input tensor specification allowing symbolic dimension names."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "float32"
    value_mode: str = "finite_edges"

    def instantiate(self, bindings: dict[str, int]) -> tuple[int, ...]:
        """Resolve symbolic dimensions to a concrete integer shape tuple."""
        concrete: list[int] = []
        for dim in self.shape:
            if isinstance(dim, int):
                concrete.append(dim)
            elif isinstance(dim, str):
                if dim not in bindings:
                    raise ValidationError(f"missing binding for symbolic dimension {dim!r}")
                concrete.append(bindings[dim])
            else:
                raise ValidationError(f"invalid dimension type {type(dim)}")
        return tuple(concrete)


def resolve_symbolic_shape(
    shape: Sequence[int | str | SymbolicDim],
    bindings: dict[str, int],
    dims_meta: dict[str, SymbolicDim] | None = None,
) -> tuple[int, ...]:
    """Resolve a symbolic shape sequence against concrete dimension bindings."""
    resolved: list[int] = []
    meta = dims_meta or {}
    for d in shape:
        if isinstance(d, int):
            resolved.append(d)
        elif isinstance(d, SymbolicDim):
            if d.name not in bindings:
                raise ValidationError(f"unbound symbolic dimension: {d.name!r}")
            val = bindings[d.name]
            d.validate_value(val)
            resolved.append(val)
        elif isinstance(d, str):
            if d not in bindings:
                raise ValidationError(f"unbound symbolic dimension: {d!r}")
            val = bindings[d]
            if d in meta:
                meta[d].validate_value(val)
            elif val < 1:
                raise ValidationError(f"dimension {d!r} must be positive, got {val}")
            resolved.append(val)
        else:
            raise ValidationError(f"unexpected dimension object {d!r}")
    return tuple(resolved)


def infer_symbolic_broadcast(
    left: Sequence[int | str],
    right: Sequence[int | str],
) -> tuple[int | str, ...]:
    """Infer symbolic broadcast shape between two shapes containing symbolic identifiers."""
    out: list[int | str] = []
    for l_dim, r_dim in zip_longest(reversed(left), reversed(right), fillvalue=1):
        if l_dim == r_dim:
            out.append(l_dim)
        elif l_dim == 1:
            out.append(r_dim)
        elif r_dim == 1:
            out.append(l_dim)
        elif isinstance(l_dim, str) and isinstance(r_dim, str):
            raise ValidationError(
                f"incompatible symbolic broadcast between {l_dim!r} and {r_dim!r}"
            )
        elif isinstance(l_dim, str) or isinstance(r_dim, str):
            raise ValidationError(
                f"cannot broadcast symbolic dimension with distinct constant: {l_dim} vs {r_dim}"
            )
        else:
            raise ValidationError(f"incompatible broadcast shapes: {l_dim} vs {r_dim}")
    out.reverse()
    return tuple(out)


def instantiate_workload(
    workload: Workload,
    bindings: dict[str, int],
    dims_meta: dict[str, SymbolicDim] | None = None,
) -> Workload:
    """Create a concrete static Workload from a workload with symbolic dimension metadata."""
    from .model import InputSpec, Node

    new_inputs: list[InputSpec] = []
    for inp in workload.inputs:
        new_shape = resolve_symbolic_shape(inp.shape, bindings, dims_meta)
        new_inputs.append(
            InputSpec(
                name=inp.name,
                shape=new_shape,
                dtype=inp.dtype,
                value_mode=inp.value_mode,
            )
        )

    new_nodes: list[Node] = []
    for node in workload.nodes:
        params = dict(node.params)
        if node.op == "reshape" and "shape" in params:
            params["shape"] = list(resolve_symbolic_shape(params["shape"], bindings, dims_meta))
        new_nodes.append(
            Node(
                op=node.op,
                inputs=node.inputs,
                output=node.output,
                params=params,
            )
        )

    concrete = Workload(
        identifier=f"{workload.identifier}-instantiated",
        seed=workload.seed,
        inputs=tuple(new_inputs),
        nodes=tuple(new_nodes),
        outputs=workload.outputs,
        policy=workload.policy,
        metadata={**workload.metadata, "symbolic_bindings": str(bindings)},
    )
    concrete.validate()
    return concrete
