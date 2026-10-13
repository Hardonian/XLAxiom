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

from .comparison import Classification, ComparisonResult, compare
from .execution import ExecutionResult, execute_jax, execute_numpy
from .model import ComparisonPolicy, InputSpec, Node, Workload
from .storage import RelationalArtifact

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
    requires_x64: bool = False


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
    # Reviewed properties exercise the same deterministic finite-edge inputs unless
    # the relation intentionally requests signed-zero or non-finite observations.
    normalized_inputs = tuple(
        spec
        if spec.value_mode != "random"
        else InputSpec(spec.name, spec.shape, spec.dtype, "finite_edges")
        for spec in inputs
    )
    workload = Workload(
        identifier=f"{identifier}-{side}-{seed}",
        seed=seed,
        inputs=normalized_inputs,
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


def _negation_involution(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)),)
    return PropertyWorkloads(
        _workload("negation-involution-v1", seed, inputs, (), ("x",), "left"),
        _workload(
            "negation-involution-v1",
            seed,
            inputs,
            (Node("neg", ("x",), "negated"), Node("neg", ("negated",), "out")),
            ("out",),
            "right",
        ),
    )


def _subtraction_as_addition(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)), InputSpec("y", (rows, columns)))
    return PropertyWorkloads(
        _workload(
            "subtraction-as-addition-v1",
            seed,
            inputs,
            (Node("sub", ("x", "y"), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "subtraction-as-addition-v1",
            seed,
            inputs,
            (Node("neg", ("y",), "negated"), Node("add", ("x", "negated"), "out")),
            ("out",),
            "right",
        ),
    )


def _absolute_idempotence(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns)),)
    return PropertyWorkloads(
        _workload(
            "absolute-idempotence-v1",
            seed,
            inputs,
            (Node("abs", ("x",), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "absolute-idempotence-v1",
            seed,
            inputs,
            (Node("abs", ("x",), "first"), Node("abs", ("first",), "out")),
            ("out",),
            "right",
        ),
    )


def _integer_add_commutativity(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (
        InputSpec("x", (rows, columns), dtype="int32"),
        InputSpec("y", (rows, columns), dtype="int32"),
    )


def _float64_add_commutativity(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (
        InputSpec("x", (rows, columns), dtype="float64"),
        InputSpec("y", (rows, columns), dtype="float64"),
    )
    return PropertyWorkloads(
        _workload(
            "float64-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("x", "y"), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "float64-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("y", "x"), "out"),),
            ("out",),
            "right",
        ),
    )


def _mixed_dtype_add_commutativity(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (
        InputSpec("integer_values", (rows, columns), dtype="int32"),
        InputSpec("floating_values", (rows, columns), dtype="float32"),
    )
    return PropertyWorkloads(
        _workload(
            "mixed-dtype-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("integer_values", "floating_values"), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "mixed-dtype-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("floating_values", "integer_values"), "out"),),
            ("out",),
            "right",
        ),
    )


def _signed_zero_negation_involution(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns), value_mode="signed_zero"),)
    return PropertyWorkloads(
        _workload(
            "signed-zero-negation-involution-v1", seed, inputs, (), ("x",), "left"
        ),
        _workload(
            "signed-zero-negation-involution-v1",
            seed,
            inputs,
            (Node("neg", ("x",), "negated"), Node("neg", ("negated",), "out")),
            ("out",),
            "right",
        ),
    )


def _nonfinite_absolute_idempotence(seed: int) -> PropertyWorkloads:
    rows, columns, _ = _dimensions(seed)
    inputs = (InputSpec("x", (rows, columns), value_mode="nonfinite"),)
    return PropertyWorkloads(
        _workload(
            "nonfinite-absolute-idempotence-v1",
            seed,
            inputs,
            (Node("abs", ("x",), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "nonfinite-absolute-idempotence-v1",
            seed,
            inputs,
            (Node("abs", ("x",), "first"), Node("abs", ("first",), "out")),
            ("out",),
            "right",
        ),
    )
    return PropertyWorkloads(
        _workload(
            "integer-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("x", "y"), "out"),),
            ("out",),
            "left",
        ),
        _workload(
            "integer-add-commutativity-v1",
            seed,
            inputs,
            (Node("add", ("y", "x"), "out"),),
            ("out",),
            "right",
        ),
    )


_BUILDERS: dict[str, PropertyBuilder] = {
    "add-commutativity-v1": _add_commutativity,
    "transpose-involution-v1": _transpose_involution,
    "reshape-roundtrip-v1": _reshape_roundtrip,
    "matmul-transpose-v1": _matmul_transpose,
    "negation-involution-v1": _negation_involution,
    "subtraction-as-addition-v1": _subtraction_as_addition,
    "absolute-idempotence-v1": _absolute_idempotence,
    "integer-add-commutativity-v1": _integer_add_commutativity,
    "float64-add-commutativity-v1": _float64_add_commutativity,
    "mixed-dtype-add-commutativity-v1": _mixed_dtype_add_commutativity,
    "signed-zero-negation-involution-v1": _signed_zero_negation_involution,
    "nonfinite-absolute-idempotence-v1": _nonfinite_absolute_idempotence,
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
        requires_x64 = entry.get("requires_x64", False)
        try:
            property_definition = MetamorphicProperty(
                identifier=str(entry["identifier"]),
                title=str(entry["title"]),
                description=str(entry["description"]),
                relation=str(entry["relation"]),
                requires_x64=requires_x64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MetamorphicPropertyError(f"malformed curated property: {exc}") from exc
        if (
            not property_definition.identifier
            or not property_definition.title
            or not property_definition.description
            or property_definition.relation != RELATION_OUTPUTS_EQUAL
            or not isinstance(property_definition.requires_x64, bool)
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


def build_property_artifact(
    identifier: str, seed: int = 0, backend: str = "jax-jit"
) -> RelationalArtifact:
    """Materialize a reviewed property as a portable relational artifact."""
    workloads = build_property(identifier, seed)
    return RelationalArtifact(
        identifier=f"{identifier}-{seed}",
        left=workloads.left,
        right=workloads.right,
        expected_classification=Classification.PASS.value,
        backend=backend,
        provenance="curated-metamorphic-property",
    )


def _execute(workload: Workload, backend: str) -> ExecutionResult:
    if backend == "numpy":
        return execute_numpy(workload)
    if backend == "jax-eager":
        return execute_jax(workload, requested_platform="cpu")
    if backend == "jax-jit":
        return execute_jax(workload, jit=True, requested_platform="cpu")
    raise MetamorphicPropertyError(f"unsupported property backend {backend!r}")


def _triage(comparison: Any) -> str:
    if comparison.classification == Classification.PASS:
        return "pass"
    if comparison.classification == Classification.NUMERICAL_DISCREPANCY:
        return "candidate_metamorphic_violation"
    return "execution_or_environment_failure"


def _x64_enabled(result: ExecutionResult) -> bool:
    return bool(result.environment and result.environment.get("jax_enable_x64") == "true")


def run_property(
    identifier: str,
    seed: int = 0,
    backend: str = "jax-jit",
    policy: ComparisonPolicy | None = None,
) -> dict[str, Any]:
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
    selected_policy = policy or workloads.left.policy
    left = _execute(workloads.left, backend)
    right = _execute(workloads.right, backend)
    if property_definition.requires_x64 and backend != "numpy" and not (
        _x64_enabled(left) and _x64_enabled(right)
    ):
        observation = ComparisonResult(
            Classification.ENVIRONMENT_FAILURE,
            "the float64 relation requires JAX x64; rerun the property command with --x64",
            selected_policy,
            left=left.summary(),
            right=right.summary(),
        )
    else:
        observation = compare(left, right, selected_policy)
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
        "requires_x64": property_definition.requires_x64,
        "left_workload_sha256": workloads.left.digest(),
        "right_workload_sha256": workloads.right.digest(),
        "comparison": observation.to_dict(),
        "triage": _triage(observation),
        "interpretation": (
            "A passing property confirms this relation for the recorded bounded input and "
            "backend only; a discrepancy is an observation requiring ordinary evidence review."
        ),
    }


def run_curated_properties(
    seed: int = 0,
    backend: str = "jax-jit",
    identifiers: tuple[str, ...] | None = None,
    policy: ComparisonPolicy | None = None,
) -> dict[str, Any]:
    """Run all, or a named subset, of the curated property corpus."""
    selected = identifiers or tuple(
        property_definition.identifier for property_definition in curated_properties()
    )
    records = [run_property(identifier, seed, backend, policy) for identifier in selected]
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


def run_property_campaign(
    cases: int,
    seed: int = 0,
    backend: str = "jax-jit",
    identifiers: tuple[str, ...] | None = None,
    policy: ComparisonPolicy | None = None,
) -> dict[str, Any]:
    """Run deterministic property shards across sequential seeds under a fixed bound."""
    if not 1 <= cases <= 500:
        raise MetamorphicPropertyError("property campaign cases must be between 1 and 500")
    selected = identifiers or tuple(
        property_definition.identifier for property_definition in curated_properties()
    )
    records = [
        run_curated_properties(seed + offset, backend, selected, policy) for offset in range(cases)
    ]
    counts = {classification.value: 0 for classification in Classification}
    candidates: list[dict[str, Any]] = []
    for record in records:
        for classification, count in record["counts"].items():
            counts[classification] += int(count)
        for property_record in record["properties"]:
            if property_record["comparison"]["classification"] != Classification.PASS.value:
                candidates.append(property_record)
    return {
        "evidence_version": "1.0",
        "corpus_version": CORPUS_VERSION,
        "kind": "metamorphic-property-campaign-evidence",
        "seed": seed,
        "cases": cases,
        "backend": backend,
        "requested_properties": list(selected),
        "counts": counts,
        "candidate_observations": candidates,
        "shards": records,
    }


def property_coverage(seed: int = 0) -> dict[str, Any]:
    """Report operations, dtypes, ranks, and edge-value coverage of reviewed pairs."""
    operations: set[str] = set()
    dtypes: set[str] = set()
    ranks: set[int] = set()
    for property_definition in curated_properties():
        pair = build_property(property_definition.identifier, seed)
        for workload in (pair.left, pair.right):
            operations.update(node.op for node in workload.nodes)
            dtypes.update(spec.dtype for spec in workload.inputs)
            ranks.update(len(spec.shape) for spec in workload.inputs)
    return {
        "corpus_version": CORPUS_VERSION,
        "kind": "metamorphic-property-coverage",
        "property_count": len(curated_properties()),
        "operations": sorted(operations),
        "dtypes": sorted(dtypes),
        "ranks": sorted(ranks),
        "edge_value_strategy": (
            "random inputs deterministically include signed zero and magnitude edges"
        ),
    }
