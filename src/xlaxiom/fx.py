"""PyTorch FX Graph ingestion and export adapter for differential compiler testing.

Translates between PyTorch 2.0 FX graph representations (torch.fx.GraphModule) and
declarative XLAxiom Workloads, allowing arbitrary PyTorch exported subgraphs to be
cross-tested against JAX/XLA, NumPy references, and metamorphic invariant suites.
"""

from __future__ import annotations

import operator
from typing import Any

from .model import InputSpec, Node, ValidationError, Workload

_TORCH_OP_MAP: dict[Any, str] = {
    operator.add: "add",
    operator.sub: "sub",
    operator.mul: "mul",
    operator.truediv: "div",
}


def _get_op_name(target: Any) -> str | None:
    if target in _TORCH_OP_MAP:
        return _TORCH_OP_MAP[target]
    name = getattr(target, "__name__", str(target)).lower()
    if name in (
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "maximum",
        "minimum",
        "abs",
        "neg",
        "negate",
        "sqrt",
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
        "matmul",
        "reshape",
    ):
        aliases = {"subtract": "sub", "multiply": "mul", "divide": "div", "negate": "neg"}
        return aliases.get(name, name)
    return None


def workload_from_fx(
    graph_module: Any,
    input_shapes: dict[str, tuple[int, ...]],
    input_dtypes: dict[str, str] | None = None,
    identifier: str = "fx-imported-workload",
    seed: int = 0,
) -> Workload:
    """Ingest a PyTorch torch.fx.GraphModule into an XLAxiom Workload."""
    dtypes = input_dtypes or {}
    inputs: list[InputSpec] = []
    nodes: list[Node] = []
    outputs: list[str] = []

    for fx_node in graph_module.graph.nodes:
        if fx_node.op == "placeholder":
            name = fx_node.name
            if name not in input_shapes:
                raise ValidationError(f"missing input shape specification for placeholder {name!r}")
            inputs.append(
                InputSpec(
                    name=name,
                    shape=input_shapes[name],
                    dtype=dtypes.get(name, "float32"),
                )
            )

        elif fx_node.op == "call_function":
            op_name = _get_op_name(fx_node.target)
            if op_name is None:
                raise ValidationError(f"unsupported PyTorch FX target function: {fx_node.target}")
            arg_names = [arg.name if hasattr(arg, "name") else str(arg) for arg in fx_node.args]
            nodes.append(Node(op=op_name, inputs=tuple(arg_names), output=fx_node.name))

        elif fx_node.op == "output":
            # fx_node.args[0] is the return value or tuple of values
            arg = fx_node.args[0]
            if isinstance(arg, (tuple, list)):
                outputs.extend(item.name if hasattr(item, "name") else str(item) for item in arg)
            elif hasattr(arg, "name"):
                outputs.append(arg.name)
            elif arg is not None:
                outputs.append(str(arg))

    if not outputs:
        raise ValidationError("FX graph has no output node")

    workload = Workload(
        identifier=identifier,
        seed=seed,
        inputs=tuple(inputs),
        nodes=tuple(nodes),
        outputs=tuple(outputs),
        metadata={"source": "torch.fx"},
    )
    workload.validate()
    return workload


def workload_to_fx(workload: Workload) -> Any:
    """Export an XLAxiom Workload to a runnable PyTorch torch.fx.GraphModule."""
    import torch
    import torch.fx

    workload.validate()
    graph = torch.fx.Graph()
    env: dict[str, torch.fx.Node] = {}

    for inp in workload.inputs:
        env[inp.name] = graph.placeholder(inp.name)

    op_to_torch: dict[str, Any] = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "maximum": torch.maximum,
        "minimum": torch.minimum,
        "abs": torch.abs,
        "neg": torch.neg,
        "sqrt": torch.sqrt,
        "exp": torch.exp,
        "log": torch.log,
        "sin": torch.sin,
        "cos": torch.cos,
        "tanh": torch.tanh,
        "matmul": torch.matmul,
    }

    for node in workload.nodes:
        in_nodes = [env[name] for name in node.inputs]
        if node.op in op_to_torch:
            fn = op_to_torch[node.op]
            env[node.output] = graph.call_function(fn, tuple(in_nodes))
        elif node.op == "reshape":
            shape = tuple(node.params["shape"])
            env[node.output] = graph.call_function(torch.reshape, (in_nodes[0], shape))
        else:
            raise ValidationError(f"cannot translate op {node.op!r} to torch.fx")

    out_nodes = [env[name] for name in workload.outputs]
    if len(out_nodes) == 1:
        graph.output(out_nodes[0])
    else:
        graph.output(tuple(out_nodes))

    return torch.fx.GraphModule(torch.nn.Module(), graph)
