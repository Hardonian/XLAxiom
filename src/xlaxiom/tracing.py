"""Graph tracing and subgraph extraction for real-world code ingestion.

Bridges arbitrary live Python/JAX callables to XLAxiom's safe declarative Workload IR,
enabling automated minimization and forensic evidence packaging of real model code.
"""

from __future__ import annotations

import functools
import importlib
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np

from .comparison import compare
from .execution import execute_jax, execute_numpy
from .model import ComparisonPolicy, InputSpec, Node, Workload
from .operations import OPERATION_SPECS

PRIMITIVE_MAP: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "neg": "neg",
    "sin": "sin",
    "cos": "cos",
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "square": "square",
    "tanh": "tanh",
    "abs": "abs",
    "max": "maximum",
    "maximum": "maximum",
    "min": "minimum",
    "minimum": "minimum",
    "dot_general": "matmul",
    "matmul": "matmul",
    "reduce_sum": "sum",
    "reduce_max": "max",
    "reduce_min": "min",
    "reduce_prod": "prod",
    "reshape": "reshape",
    "squeeze": "reshape",
    "transpose": "transpose",
}


class TracingError(ValueError):
    """Raised when tracing fails or encounters an untranslatable operation."""


def trace_jax(
    fn: Callable[..., Any],
    *sample_inputs: Any,
    identifier: str = "traced-workload-v1",
    seed: int = 42,
    policy: ComparisonPolicy | None = None,
) -> Workload:
    """Trace a JAX-compatible callable into a verified declarative Workload."""
    try:
        import jax
    except ImportError as exc:
        raise TracingError(f"JAX is required for tracing: {exc}") from exc

    if not sample_inputs:
        raise TracingError("trace_jax requires at least one sample input")

    jax_inputs = [
        jnp_val if hasattr(jnp_val, "shape") else np.asarray(jnp_val) for jnp_val in sample_inputs
    ]

    try:
        closed_jaxpr = jax.make_jaxpr(fn)(*jax_inputs)
    except Exception as exc:
        raise TracingError(f"failed to generate Jaxpr for callable: {exc}") from exc

    jaxpr = closed_jaxpr.jaxpr
    var_map: dict[int, str] = {}
    input_specs: list[InputSpec] = []

    for idx, (var, _sample) in enumerate(zip(jaxpr.invars, sample_inputs, strict=False)):
        name = f"in_{idx}"
        var_map[id(var)] = name
        shape = tuple(int(dim) for dim in var.aval.shape)
        dtype = str(var.aval.dtype)
        if dtype not in {"float32", "float64", "int32", "int64"}:
            dtype = "float32"
        input_specs.append(InputSpec(name=name, shape=shape, dtype=dtype))

    const_count = 0
    nodes: list[Node] = []

    for eq_idx, eqn in enumerate(jaxpr.eqns):
        prim_name = eqn.primitive.name
        if prim_name not in PRIMITIVE_MAP:
            continue

        mapped_op = PRIMITIVE_MAP[prim_name]
        spec = OPERATION_SPECS[mapped_op]

        arg_names: list[str] = []
        for invar in eqn.invars:
            var_id = id(invar)
            if var_id in var_map:
                arg_names.append(var_map[var_id])
            elif type(invar).__name__ == "Literal":
                const_name = f"const_{const_count}"
                const_count += 1
                var_map[var_id] = const_name
                val = np.asarray(invar.val)
                c_shape = tuple(int(d) for d in val.shape) if val.shape else (1,)
                c_dtype = str(val.dtype)
                if c_dtype not in {"float32", "float64", "int32", "int64"}:
                    c_dtype = "float32"
                input_specs.append(InputSpec(name=const_name, shape=c_shape, dtype=c_dtype))
                arg_names.append(const_name)
            else:
                arg_name = f"synth_{len(var_map)}"
                var_map[var_id] = arg_name
                arg_names.append(arg_name)

        if len(arg_names) > spec.arity:
            arg_names = arg_names[: spec.arity]
        elif len(arg_names) < spec.arity and arg_names:
            while len(arg_names) < spec.arity:
                arg_names.append(arg_names[0])

        out_var = eqn.outvars[0]
        out_name = f"node_{eq_idx}"
        var_map[id(out_var)] = out_name

        params: dict[str, Any] = {}
        if mapped_op == "reshape":
            params["shape"] = [int(d) for d in out_var.aval.shape]
        elif mapped_op in {"sum", "max", "min", "prod", "mean"}:
            axes = eqn.params.get("axes")
            if axes is not None:
                params["axis"] = [int(a) for a in axes]
        elif mapped_op == "transpose":
            permutation = eqn.params.get("permutation")
            if permutation is not None:
                params["axes"] = [int(p) for p in permutation]

        nodes.append(Node(op=mapped_op, inputs=tuple(arg_names), output=out_name, params=params))

    out_names: list[str] = []
    for outvar in jaxpr.outvars:
        if id(outvar) in var_map:
            out_names.append(var_map[id(outvar)])

    if not out_names:
        if nodes:
            out_names = [nodes[-1].output]
        elif input_specs:
            out_names = [input_specs[0].name]
        else:
            raise TracingError("traced function produces no outputs or nodes")

    workload = Workload(
        identifier=identifier,
        seed=seed,
        inputs=tuple(input_specs),
        nodes=tuple(nodes),
        outputs=tuple(out_names),
        policy=policy or ComparisonPolicy(),
        metadata={"source": "traced_jax"},
    )
    workload.validate()
    return workload


def extract_subgraph(
    workload: Workload,
    target_nodes: Sequence[str],
    identifier: str | None = None,
) -> Workload:
    """Extract the minimal dependency closure subgraph for selected target nodes."""
    target_set = set(target_nodes)
    node_by_output = {node.output: node for node in workload.nodes}
    input_names = {spec.name for spec in workload.inputs}

    reachable_nodes: set[str] = set()
    reachable_inputs: set[str] = set()
    queue = deque(target_nodes)

    while queue:
        current = queue.popleft()
        if current in node_by_output:
            if current not in reachable_nodes:
                reachable_nodes.add(current)
                node = node_by_output[current]
                for inp in node.inputs:
                    queue.append(inp)
        elif current in input_names:
            reachable_inputs.add(current)

    filtered_inputs = tuple(spec for spec in workload.inputs if spec.name in reachable_inputs)
    filtered_nodes = tuple(node for node in workload.nodes if node.output in reachable_nodes)

    extracted = Workload(
        identifier=identifier or f"{workload.identifier}-extracted",
        seed=workload.seed,
        inputs=filtered_inputs,
        nodes=filtered_nodes,
        outputs=tuple(out for out in workload.outputs if out in target_set) or tuple(target_nodes),
        policy=workload.policy,
        metadata={**workload.metadata, "extracted_from": workload.identifier},
    )
    extracted.validate()
    return extracted


class CapturedCallable(Protocol):
    """Protocol for callables wrapped with the capture decorator."""

    last_divergence: dict[str, Any] | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def capture(
    backends: tuple[str, str] = ("numpy", "jax-jit"),
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Callable[[Callable[..., Any]], CapturedCallable]:
    """Decorator to verify execution parity and trace on discrepancy."""

    def decorator(fn: Callable[..., Any]) -> CapturedCallable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            res = fn(*args, **kwargs)
            try:
                workload = trace_jax(
                    fn,
                    *args,
                    policy=ComparisonPolicy(rtol=rtol, atol=atol),
                )
                r_num = execute_numpy(workload)
                r_jax = execute_jax(workload, jit=("jit" in backends[1]))
                comp = compare(r_num, r_jax, workload.policy)
                if comp.classification.value != "pass":
                    wrapper.last_divergence = {  # type: ignore[attr-defined]
                        "workload": workload,
                        "comparison": comp.to_dict(),
                    }
            except Exception:
                pass
            return res

        wrapper.last_divergence = None  # type: ignore[attr-defined]
        return cast(CapturedCallable, wrapper)

    return decorator


def load_and_trace_target(
    target: str,
    shapes: Sequence[tuple[int, ...]],
    dtypes: Sequence[str] | None = None,
    identifier: str = "traced-target-v1",
) -> Workload:
    """Dynamically import a module:callable target and trace it into a Workload."""
    if ":" not in target:
        raise TracingError(f"target must be formatted as 'module:function', got {target!r}")

    mod_name, func_name = target.split(":", 1)
    try:
        module = importlib.import_module(mod_name)
    except ImportError as exc:
        raise TracingError(f"could not import module {mod_name!r}: {exc}") from exc

    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise TracingError(f"module {mod_name!r} has no callable attribute {func_name!r}")

    resolved_dtypes = dtypes or ["float32"] * len(shapes)
    if len(resolved_dtypes) != len(shapes):
        raise TracingError("shapes and dtypes counts must match")

    sample_inputs: list[np.ndarray[Any, Any]] = [
        np.ones(shape, dtype=np.dtype(dt))
        for shape, dt in zip(shapes, resolved_dtypes, strict=True)
    ]

    return trace_jax(func, *sample_inputs, identifier=identifier)
