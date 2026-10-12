import numpy as np

from xlaxiom.comparison import Classification, compare
from xlaxiom.execution import ExecutionResult, ExecutionStatus
from xlaxiom.model import ComparisonPolicy


def test_nan_and_signed_zero_semantics_are_explicit() -> None:
    policy = ComparisonPolicy(atol=0, rtol=0, check_signed_zero=True)
    left = ExecutionResult(
        "a", "eager", ExecutionStatus.OK, (np.array([np.nan, -0.0], dtype=np.float32),)
    )
    right = ExecutionResult(
        "b", "eager", ExecutionStatus.OK, (np.array([np.nan, 0.0], dtype=np.float32),)
    )
    result = compare(left, right, policy)
    assert result.classification == Classification.NUMERICAL_DISCREPANCY
    assert result.metrics[0].signed_zero_mismatches == 1
