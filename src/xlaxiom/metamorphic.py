"""Curated, declarative metamorphic properties for numerical execution evidence.

The checked-in corpus only selects from this fixed builder registry.  Its JSON data is
never evaluated as code, and a malformed or drifted manifest fails closed instead of
silently enabling a new property.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from .comparison import Classification, compare
from .execution import ExecutionResult, execute_jax, execute_numpy
from .model import InputSpec, Node, Workload

CORPUS_VERSION = "1.0"
RELATION_OUTPUTS_EQUAL = "outputs_equal"


class MetamorphicPropertyError(ValueError):
    """The curated property corpus or a requested property is invalid."""


@dataclass(frozen=True)
class MetamorphicProperty:
    """One reviewed metamorphic relation exposed by the curated corpus."""

    identifier: str
    title: str
    description: str
    relation: str


@dataclass(frozen=True)
class PropertyWorkloads:
    """Two safe declarative workloads whose outputs must satisfy one relation."""

    left: Workload
    right: Workload


PropertyBuilder = Callable[[int], PropertyWorkloads]


def _dimensions(seed: int) -> tuple[int, int, int]:
    if not 0 <= seed < 2**63:
        raise MetamorphicPropertyError("seed must be between 0 and 2**63 - 1")
    return 2 + seed % 7, 2 + (seed // 7) % 7, 2 + (seed // 49) % 7


def _workload(
    identifier: str,
    seed: int,
    inputs: tuple[InputSpec, ...],
    nodes: tuple[Node, ...],
    outputs: tuple[str, ...],
    side: str,
) -> Workload:
    workload = Workload(
        identifier=f"{identifier}-{side}-{seed}",
        seed=seed,
        inputs=inputs,
        nodes=nodes,
        outputs=outputs,
        metadata={
            "corpus": "curated-v1",
            "kind": "metamorphic-property",
            "property": identifier,
            "side": side,
        },
    )
    workload.validate()
    return workload


def _add_commutativity(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)), InputSpec("y", (rows, columns)))
    return PropertyWorkloads(
        _workload(
            "add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("x", "y"), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("y", "x"), "out"),),
            ("out",),
            "right",
        ),
    )


def _transpose_involution(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)),)
    return PropertyWorkloads(
        _workload("transpose-involution-v1", seed, inputs, (), ("x",), "left"),
        _workload(
            "transpose-involution-v1",
            seed,
            inputs,
            (
                Node("transpose", ("x",), "transposed", {"axes": [1, 0]}),
                Node("transpose", ("transposed",), "out", {"axes": [1, 0]}),
            ),
            ("out",),
            "right",
        ),
    )


def _reshape_roundtrip(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)),)
    return PropertyWorkloads(
        _workload("reshape-roundtrip-v1", seed, inputs, (), ("x",), "left"),
        _workload(
            "reshape-roundtrip-v1",
            seed,
            inputs,
            (
                Node("reshape", ("x",), "flat", {"shape": [rows * columns]}),
                Node("reshape", ("flat",), "out", {"shape": [rows, columns]}),
            ),
            ("out",),
            "right",
        ),
    )


def _matmul_transpose(seed: int) -> PropertyWorkloads:
    rows, inner, columns = _dimensions(seed)
    inputs = (InputSpec("x", (rows, inner)), InputSpec("y", (inner, columns)))
    return PropertyWorkloads(
        _workload(
            "matmul-transpose-v1",
            seed,
            inputs,
            (
                Node("matmul", ("x", "y"), "product"),
                Node("transpose", ("product",), "out", {"axes": [1, 0]}),
            ),
            ("out",),
            "left",
        ),
        _workload(
            "matmul-transpose-v1",
            seed,
            inputs,
            (
                Node("transpose", ("y",), "y_transposed", {"axes": [1, 0]}),
                Node("transpose", ("x",), "x_transposed", {"axes": [1, 0]}),
                Node("matmul", ("y_transposed", "x_transposed"), "out"),
            ),
            ("out",),
            "right",
        ),
    )


_BUILDERS: dict[str, PropertyBuilder] = {
    "add-commutativity-v1": _add_commutativity,
    "transpose-involution-v1": _transpose_involution,
    "reshape-roundtrip-v1": _reshape_roundtrip,
    "matmul-transpose-v1": _matmul_transpose,
}


def _corpus_payload() -> dict[str, Any]:
    try:
        resource = files("xlaxiom").joinpath("corpus", "curated-v1.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise MetamorphicPropertyError(f"cannot read curated corpus: {exc}") from exc
    if not isinstance(payload, dict):
        raise MetamorphicPropertyError("curated corpus must be a JSON object")
    return payload


def curated_properties() -> tuple[MetamorphicProperty, ...]:
    """Load and validate the reviewed property portion of the packaged corpus."""
    payload = _corpus_payload()
    if payload.get("corpus_version") != CORPUS_VERSION:
        raise MetamorphicPropertyError("unsupported curated corpus version")
    entries = payload.get("metamorphic_properties")
    if not isinstance(entries, list):
        raise MetamorphicPropertyError("curated corpus properties must be a list")
    properties: list[MetamorphicProperty] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise MetamorphicPropertyError("curated corpus property must be an object")
        try:
            property_definition = MetamorphicProperty(
                identifier=str(entry["identifier"]),
                title=str(entry["title"]),
                description=str(entry["description"]),
                relation=str(entry["relation"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MetamorphicPropertyError(f"malformed curated property: {exc}") from exc
        if (
            not property_definition.identifier
            or not property_definition.title
            or not property_definition.description
            or property_definition.relation != RELATION_OUTPUTS_EQUAL
        ):
            raise MetamorphicPropertyError("curated property has unsupported fields")
        properties.append(property_definition)
    identifiers = tuple(property_definition.identifier for property_definition in properties)
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(_BUILDERS):
        raise MetamorphicPropertyError(
            "curated corpus does not exactly match the safe builder registry"
        )
    return tuple(properties)


def build_property(identifier: str, seed: int) -> PropertyWorkloads:
    """Build a reviewed property pair; arbitrary corpus data cannot select code."""
    if identifier not in {
        property_definition.identifier for property_definition in curated_properties()
    }:
        raise MetamorphicPropertyError(f"unknown curated property {identifier!r}")
    return _BUILDERS[identifier](seed)


def _execute(workload: Workload, backend: str) -> ExecutionResult:
    if backend == "numpy":
        return execute_numpy(workload)
    if backend == "jax-eager":
        return execute_jax(workload, requested_platform="cpu")
    if backend == "jax-jit":
        return execute_jax(workload, jit=True, requested_platform="cpu")
    raise MetamorphicPropertyError(f"unsupported property backend {backend!r}")


def run_property(identifier: str, seed: int = 0, backend: str = "jax-jit") -> dict[str, Any]:
    """Execute both sides of one property and record its numerical observation."""
    property_definition = next(
        (
            property_definition
            for property_definition in curated_properties()
            if property_definition.identifier == identifier
        ),
        None,
    )
    if property_definition is None:
        raise MetamorphicPropertyError(f"unknown curated property {identifier!r}")
    workloads = build_property(identifier, seed)
    observation = compare(
        _execute(workloads.left, backend), _execute(workloads.right, backend), workloads.left.policy
    )
    return {
        "evidence_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "kind": "metamorphic-property-evidence",
        "property": {
            "identifier": property_definition.identifier,
            "title": property_definition.title,
            "description": property_definition.description,
            "relation": property_definition.relation,
        },
        "seed": seed,
        "backend": backend,
        "left_workload_sha256": workloads.left.digest(),
        "right_workload_sha256": workloads.right.digest(),
        "comparison": observation.to_dict(),
        "interpretation": (
            "A passing property confirms this relation for the recorded bounded input and "
            "backend only; a discrepancy is an observation requiring ordinary evidence review."
        ),
    }


def run_curated_properties(
    seed: int = 0, backend: str = "jax-jit", identifiers: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Run all, or a named subset, of the curated property corpus."""
    selected = identifiers or tuple(
        property_definition.identifier for property_definition in curated_properties()
    )
    records = [run_property(identifier, seed, backend) for identifier in selected]
    counts = {classification.value: 0 for classification in Classification}
    for record in records:
        counts[str(record["comparison"]["classification"])] += 1
    return {
        "evidence_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "kind": "metamorphic-corpus-evidence",
        "seed": seed,
        "backend": backend,
        "requested_properties": list(selected),
        "counts": counts,
        "properties": records,
        "warning": (
            "This corpus contains reviewed metamorphic relations, not newly discovered compiler "
            "findings. It does not establish correctness beyond the recorded executions."
        ),
    }
