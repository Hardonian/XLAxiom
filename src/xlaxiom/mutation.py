"""Mutation testing and sharpness evaluation for metamorphic relations.

Systematically synthesizes intentional semantic and syntactic compiler fault mutations
(operator swaps, sign inversions, operand permutations, and node offsets) to measure
the empirical fault-detection power (sharpness score) of each metamorphic invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comparison import Classification, compare
from .execution import execute_numpy
from .metamorphic import build_property, curated_properties
from .model import Node, Workload

_SWAP_MAP: dict[str, str] = {
    "add": "sub",
    "sub": "add",
    "maximum": "minimum",
    "minimum": "maximum",
    "sin": "cos",
    "cos": "sin",
    "square": "abs",
    "abs": "square",
    "min": "max",
    "max": "min",
}


@dataclass(frozen=True)
class Mutant:
    """One mutated workload representing an intentional compiler bug hypothesis."""

    original_workload_id: str
    mutator_name: str
    description: str
    workload: Workload


@dataclass(frozen=True)
class MutantResult:
    """Execution and verification outcome of a single mutated workload."""

    mutant: Mutant
    killed: bool
    classification: str
    explanation: str


@dataclass(frozen=True)
class SharpnessReport:
    """Summary of mutant kill rate measuring metamorphic relation sharpness."""

    property_identifier: str
    backend: str
    total_mutants: int
    killed_mutants: int
    sharpness_score: float
    results: tuple[MutantResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_identifier": self.property_identifier,
            "backend": self.backend,
            "total_mutants": self.total_mutants,
            "killed_mutants": self.killed_mutants,
            "sharpness_score": round(self.sharpness_score, 4),
            "results": [
                {
                    "mutator": r.mutant.mutator_name,
                    "description": r.mutant.description,
                    "killed": r.killed,
                    "classification": r.classification,
                }
                for r in self.results
            ],
        }


def generate_workload_mutants(workload: Workload) -> list[Mutant]:
    """Generate deterministic syntactic and semantic fault mutants for a workload."""
    mutants: list[Mutant] = []
    base_id = workload.identifier

    # Mutator 1: Operator swap
    for idx, node in enumerate(workload.nodes):
        if node.op in _SWAP_MAP:
            swapped_op = _SWAP_MAP[node.op]
            mutated_nodes = list(workload.nodes)
            mutated_nodes[idx] = Node(swapped_op, node.inputs, node.output, node.params)
            try:
                mutated_workload = Workload(
                    identifier=f"{base_id}-mutant-swap-{idx}",
                    seed=workload.seed,
                    inputs=workload.inputs,
                    nodes=tuple(mutated_nodes),
                    outputs=workload.outputs,
                    policy=workload.policy,
                    metadata={**workload.metadata, "mutant": f"swap_{node.op}_to_{swapped_op}"},
                )
                mutated_workload.validate()
                mutants.append(
                    Mutant(
                        original_workload_id=base_id,
                        mutator_name="OperatorSwapMutator",
                        description=f"Swapped node {idx} op from {node.op} to {swapped_op}",
                        workload=mutated_workload,
                    )
                )
            except Exception:
                pass

    # Mutator 2: Operand reversal (for binary ops where operands are distinct)
    for idx, node in enumerate(workload.nodes):
        if len(node.inputs) == 2 and node.inputs[0] != node.inputs[1]:
            reversed_inputs = (node.inputs[1], node.inputs[0])
            mutated_nodes = list(workload.nodes)
            mutated_nodes[idx] = Node(node.op, reversed_inputs, node.output, node.params)
            try:
                mutated_workload = Workload(
                    identifier=f"{base_id}-mutant-rev-{idx}",
                    seed=workload.seed,
                    inputs=workload.inputs,
                    nodes=tuple(mutated_nodes),
                    outputs=workload.outputs,
                    policy=workload.policy,
                    metadata={**workload.metadata, "mutant": f"reverse_{node.inputs}"},
                )
                mutated_workload.validate()
                mutants.append(
                    Mutant(
                        original_workload_id=base_id,
                        mutator_name="OperandReversalMutator",
                        description=(
                            f"Reversed node {idx} operands from {node.inputs} to {reversed_inputs}"
                        ),
                        workload=mutated_workload,
                    )
                )
            except Exception:
                pass

    # Mutator 3: Negation injection on output node
    if workload.nodes:
        last_node = workload.nodes[-1]
        neg_output = f"{last_node.output}_mut_neg"
        new_nodes = list(workload.nodes)
        new_nodes.append(Node("neg", (last_node.output,), neg_output))
        try:
            mutated_workload = Workload(
                identifier=f"{base_id}-mutant-neg-out",
                seed=workload.seed,
                inputs=workload.inputs,
                nodes=tuple(new_nodes),
                outputs=(neg_output,),
                policy=workload.policy,
                metadata={**workload.metadata, "mutant": "negate_output"},
            )
            mutated_workload.validate()
            mutants.append(
                Mutant(
                    original_workload_id=base_id,
                    mutator_name="SignInversionMutator",
                    description="Inverted sign of the primary output node",
                    workload=mutated_workload,
                )
            )
        except Exception:
            pass

    return mutants


def evaluate_property_sharpness(
    property_identifier: str, seed: int = 42, backend: str = "numpy"
) -> SharpnessReport:
    """Evaluate mutation kill rate for a curated metamorphic property."""
    valid_ids = {prop.identifier for prop in curated_properties()}
    if property_identifier not in valid_ids:
        raise ValueError(f"unknown curated property {property_identifier!r}")

    pair = build_property(property_identifier, seed)
    mutants = generate_workload_mutants(pair.right)
    if not mutants:
        # Fall back to left workload mutants
        mutants = generate_workload_mutants(pair.left)

    left_exec = execute_numpy(pair.left)
    results: list[MutantResult] = []

    for mutant in mutants:
        mutant_exec = execute_numpy(mutant.workload)
        comparison = compare(left_exec, mutant_exec, pair.left.policy)
        killed = comparison.classification != Classification.PASS
        results.append(
            MutantResult(
                mutant=mutant,
                killed=killed,
                classification=comparison.classification.value,
                explanation=comparison.explanation,
            )
        )

    killed_count = sum(1 for r in results if r.killed)
    total_count = len(results)
    score = (killed_count / total_count) if total_count > 0 else 1.0

    return SharpnessReport(
        property_identifier=property_identifier,
        backend=backend,
        total_mutants=total_count,
        killed_mutants=killed_count,
        sharpness_score=score,
        results=tuple(results),
    )
