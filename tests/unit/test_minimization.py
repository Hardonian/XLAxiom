from __future__ import annotations

from xlaxiom.comparison import Classification
from xlaxiom.minimization import (
    MinimizationResult,
    MinimizationStats,
    _elements,
    _identity_simplifications,
    _prune_dead,
    _replace_value,
    _shape_simplification,
    _value_simplifications,
    minimize,
)
from xlaxiom.model import InputSpec, Node, Workload


def test_elements_counts_total_input_elements() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 3), "float32", "random"),
            InputSpec("y", (4, 1, 2), "float32", "zeros"),
        ),
        nodes=(Node("add", ("x", "y"), "z"),),
        outputs=("z",),
    )
    # (2 * 3) + (4 * 1 * 2) = 6 + 8 = 14
    assert _elements(workload) == 14


def test_prune_dead_removes_unused_nodes_and_inputs() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 2), "float32", "random"),
            InputSpec("unused_in", (2, 2), "float32", "zeros"),
        ),
        nodes=(
            Node("add", ("x", "unused_in"), "used_node"),
            Node("neg", ("x",), "dead_node"),
        ),
        outputs=("used_node",),
    )
    pruned = _prune_dead(workload)
    assert len(pruned.nodes) == 1
    assert pruned.nodes[0].output == "used_node"
    assert [spec.name for spec in pruned.inputs] == ["x", "unused_in"]

    # Completely unused input case
    workload_dead_input = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 2), "float32", "random"),
            InputSpec("orphan", (2, 2), "float32", "zeros"),
        ),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )
    pruned_orphan = _prune_dead(workload_dead_input)
    assert len(pruned_orphan.inputs) == 1
    assert pruned_orphan.inputs[0].name == "x"


def test_replace_value_substitutes_inputs_and_outputs() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 2), "float32", "random"),
            InputSpec("y", (2, 2), "float32", "zeros"),
        ),
        nodes=(
            Node("add", ("x", "y"), "t1"),
            Node("neg", ("t1",), "t2"),
        ),
        outputs=("t2",),
    )
    # Replace t1 with x
    replaced = _replace_value(workload, "t1", "x")
    # node add becomes dead and is pruned because t1 is no longer needed
    assert len(replaced.nodes) == 1
    assert replaced.nodes[0].inputs == ("x",)
    assert replaced.nodes[0].output == "t2"
    assert replaced.outputs == ("t2",)


def test_identity_simplifications_add_zero_and_double_neg() -> None:
    # Test add zero simplification (left and right)
    workload_add = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 2), "float32", "random"),
            InputSpec("z", (2, 2), "float32", "zeros"),
        ),
        nodes=(
            Node("add", ("x", "z"), "out_right"),
            Node("add", ("z", "x"), "out_left"),
        ),
        outputs=("out_right", "out_left"),
    )
    simplifications = list(_identity_simplifications(workload_add))
    assert len(simplifications) == 2

    # Test double neg simplification neg(neg(x))
    workload_neg = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(InputSpec("x", (2, 2), "float32", "random"),),
        nodes=(
            Node("neg", ("x",), "n1"),
            Node("neg", ("n1",), "n2"),
        ),
        outputs=("n2",),
    )
    neg_simplifications = list(_identity_simplifications(workload_neg))
    assert len(neg_simplifications) == 1
    # Replacing n1 with x should yield a workload where n2 takes x directly and n1 is pruned
    simplified = neg_simplifications[0]
    assert len(simplified.nodes) == 0  # neg(x) replaced by x in output n2 -> pruned
    assert simplified.outputs == ("x",)


def test_shape_simplification_reduces_rank_preserving_shapes() -> None:
    workload_elementwise = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(InputSpec("x", (4, 4), "float32", "random"),),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )
    shrunk = list(_shape_simplification(workload_elementwise))
    assert len(shrunk) == 1
    assert shrunk[0].inputs[0].shape == (1, 1)

    # Skipped when unsupported ops like matmul or reshape are present
    workload_matmul = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (4, 4), "float32", "random"),
            InputSpec("y", (4, 4), "float32", "random"),
        ),
        nodes=(Node("matmul", ("x", "y"), "out"),),
        outputs=("out",),
    )
    assert list(_shape_simplification(workload_matmul)) == []

    # Skipped when shapes are already (1, 1)
    workload_ones = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(InputSpec("x", (1, 1), "float32", "random"),),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )
    assert list(_shape_simplification(workload_ones)) == []


def test_value_simplifications_converts_random_to_zeros() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (2, 2), "float32", "random"),
            InputSpec("y", (2, 2), "float32", "zeros"),
        ),
        nodes=(Node("add", ("x", "y"), "out"),),
        outputs=("out",),
    )
    value_simps = list(_value_simplifications(workload))
    assert len(value_simps) == 1
    assert value_simps[0].inputs[0].value_mode == "zeros"


def test_minimize_happy_path_and_stats() -> None:
    # Workload with a double negation and random input
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(
            InputSpec("x", (4, 4), "float32", "random"),
            InputSpec("z", (4, 4), "float32", "zeros"),
        ),
        nodes=(
            Node("neg", ("x",), "n1"),
            Node("neg", ("n1",), "n2"),
            Node("add", ("n2", "z"), "out"),
        ),
        outputs=("out",),
    )

    # Predicate expects Classification.NUMERICAL_DISCREPANCY
    def check_fn(candidate: Workload) -> Classification:
        return Classification.NUMERICAL_DISCREPANCY

    result = minimize(
        workload,
        predicate=check_fn,
        expected=Classification.NUMERICAL_DISCREPANCY,
        max_evaluations=10,
    )

    assert isinstance(result, MinimizationResult)
    assert isinstance(result.stats, MinimizationStats)
    assert result.stats.reduced_nodes < result.stats.original_nodes
    assert result.stats.preserved_classification == Classification.NUMERICAL_DISCREPANCY.value
    assert result.stats.preserved_signature == str(Classification.NUMERICAL_DISCREPANCY)
    assert result.stats.duration_ms >= 0


def test_minimize_exhausts_max_evaluations() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(InputSpec("x", (4, 4), "float32", "random"),),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )

    result = minimize(
        workload,
        predicate=lambda _: "match",
        expected="match",
        max_evaluations=0,
    )

    assert result.stats.budget_exhausted is True
    assert result.stats.candidates_evaluated == 0


def test_minimize_handles_invalid_candidate_gracefully() -> None:
    workload = Workload(
        identifier="test_workload",
        seed=42,
        inputs=(InputSpec("x", (2, 2), "float32", "random"),),
        nodes=(Node("neg", ("x",), "out"),),
        outputs=("out",),
    )

    result = minimize(
        workload,
        predicate=lambda _: "ok",
        expected="ok",
        max_evaluations=5,
    )
    assert result.workload is not None
