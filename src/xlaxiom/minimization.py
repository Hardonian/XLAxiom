"""Deterministic, failure-preserving reduction for declarative workloads."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from time import perf_counter

from .comparison import Classification
from .model import InputSpec, Node, Workload


@dataclass(frozen=True)
class MinimizationStats:
    original_nodes: int
    reduced_nodes: int
    original_elements: int
    reduced_elements: int
    candidates_evaluated: int
    accepted_reductions: int
    duration_ms: float
    budget_exhausted: bool
    preserved_classification: str
    preserved_signature: str


@dataclass(frozen=True)
class MinimizationResult:
    workload: Workload
    stats: MinimizationStats


def _elements(workload: Workload) -> int:
    result = 0
    for spec in workload.inputs:
        amount = 1
        for dimension in spec.shape:
            amount *= dimension
        result += amount
    return result


def _prune_dead(workload: Workload) -> Workload:
    required = set(workload.outputs)
    nodes: list[Node] = []
    for node in reversed(workload.nodes):
        if node.output in required:
            nodes.append(node)
            required.update(node.inputs)
    nodes.reverse()
    inputs = tuple(spec for spec in workload.inputs if spec.name in required)
    return replace(workload, inputs=inputs, nodes=tuple(nodes))


def _replace_value(workload: Workload, old: str, new: str) -> Workload:
    nodes = tuple(
        replace(node, inputs=tuple(new if value == old else value for value in node.inputs))
        for node in workload.nodes
    )
    outputs = tuple(new if value == old else value for value in workload.outputs)
    return _prune_dead(replace(workload, nodes=nodes, outputs=outputs))


def _identity_simplifications(workload: Workload) -> Iterable[Workload]:
    zero_inputs = {spec.name for spec in workload.inputs if spec.value_mode == "zeros"}
    by_output = {node.output: node for node in workload.nodes}
    for node in workload.nodes:
        if node.op == "add" and node.inputs[1] in zero_inputs:
            yield _replace_value(workload, node.output, node.inputs[0])
        if node.op == "add" and node.inputs[0] in zero_inputs:
            yield _replace_value(workload, node.output, node.inputs[1])
        if node.op == "neg" and node.inputs[0] in by_output:
            previous = by_output[node.inputs[0]]
            if previous.op == "neg":
                yield _replace_value(workload, node.output, previous.inputs[0])


def _shape_simplification(workload: Workload) -> Iterable[Workload]:
    """Conservative rank-preserving size reduction for elementwise graphs only."""
    unsupported = {"matmul", "reshape"}
    if any(node.op in unsupported for node in workload.nodes):
        return
    if not any(any(dimension > 1 for dimension in spec.shape) for spec in workload.inputs):
        return
    shrunk = tuple(replace(spec, shape=tuple(1 for _ in spec.shape)) for spec in workload.inputs)
    yield replace(workload, inputs=shrunk)


def _value_simplifications(workload: Workload) -> Iterable[Workload]:
    for index, spec in enumerate(workload.inputs):
        if spec.value_mode == "random":
            inputs = list(workload.inputs)
            inputs[index] = InputSpec(spec.name, spec.shape, spec.dtype, "zeros")
            yield replace(workload, inputs=tuple(inputs))


def minimize(
    workload: Workload,
    predicate: Callable[[Workload], object],
    expected: object,
    max_evaluations: int = 64,
) -> MinimizationResult:
    """Greedily accept only valid candidates that preserve the exact observation class."""
    started = perf_counter()
    current = workload
    evaluated = 0
    accepted = 0
    exhausted = False
    while True:
        candidates: list[Workload] = [_prune_dead(current)]
        candidates.extend(_identity_simplifications(current))
        candidates.extend(_shape_simplification(current))
        candidates.extend(_value_simplifications(current))
        changed = False
        seen: set[str] = {current.digest()}
        for candidate in candidates:
            if evaluated >= max_evaluations:
                exhausted = True
                break
            try:
                candidate.validate()
            except Exception:
                continue
            digest = candidate.digest()
            if digest in seen:
                continue
            seen.add(digest)
            evaluated += 1
            if predicate(candidate) == expected:
                current = candidate
                accepted += 1
                changed = True
                break
        if exhausted or not changed:
            break
    stats = MinimizationStats(
        len(workload.nodes),
        len(current.nodes),
        _elements(workload),
        _elements(current),
        evaluated,
        accepted,
        (perf_counter() - started) * 1000,
        exhausted,
        expected.value if isinstance(expected, Classification) else str(expected),
        str(expected),
    )
    return MinimizationResult(current, stats)
