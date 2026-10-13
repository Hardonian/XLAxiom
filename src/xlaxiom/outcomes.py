"""Fail-closed equivalence checks for execution outcomes, including exceptions."""

from __future__ import annotations

from .comparison import Classification, ComparisonResult
from .execution import ExecutionResult, ExecutionStatus
from .model import ComparisonPolicy


def compare_outcomes(
    left: ExecutionResult, right: ExecutionResult, policy: ComparisonPolicy | None = None
) -> ComparisonResult:
    """Compare only terminal execution categories, never exception message text.

    This is deliberately a separate operation from numerical comparison.  Two successful
    executions are status-equivalent but their values remain unassessed; callers must run
    the ordinary numerical comparator for that.  Two matching non-OK statuses are also
    equivalent at this layer, while a one-sided or category mismatch is inconclusive.
    """
    selected_policy = policy or ComparisonPolicy(identifier="outcome-equivalence-v1")
    if left.status == right.status:
        state = "completed" if left.status == ExecutionStatus.OK else "failed"
        return ComparisonResult(
            Classification.PASS,
            f"both executions {state} with terminal status {left.status.value}",
            selected_policy,
            left=left.summary(),
            right=right.summary(),
        )
    return ComparisonResult(
        Classification.INCONCLUSIVE,
        "terminal execution statuses differ; exception equivalence is not established",
        selected_policy,
        left=left.summary(),
        right=right.summary(),
    )
