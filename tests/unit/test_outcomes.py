from __future__ import annotations

from xlaxiom.comparison import Classification
from xlaxiom.execution import ExecutionResult, ExecutionStatus
from xlaxiom.outcomes import compare_outcomes


def test_matching_non_ok_statuses_are_outcome_equivalent() -> None:
    left = ExecutionResult("jax-cpu", "jit", ExecutionStatus.COMPILATION_FAILURE)
    right = ExecutionResult("jax-cpu", "eager", ExecutionStatus.COMPILATION_FAILURE)

    result = compare_outcomes(left, right)

    assert result.classification == Classification.PASS
    assert "compilation_failure" in result.explanation


def test_different_statuses_are_fail_closed() -> None:
    left = ExecutionResult("jax-cpu", "jit", ExecutionStatus.COMPILATION_FAILURE)
    right = ExecutionResult("jax-cpu", "eager", ExecutionStatus.EXECUTION_FAILURE)

    result = compare_outcomes(left, right)

    assert result.classification == Classification.INCONCLUSIVE
