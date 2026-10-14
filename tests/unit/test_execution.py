"""Contract tests for the public Array API execution adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from xlaxiom.execution import ExecutionStatus, execute_array_api, execute_numpy
from xlaxiom.model import InputSpec, Node, Workload


class _NumpyNamespaceProxy:
    """A minimal NumPy-compatible namespace with opt-in missing operations."""

    def __init__(self, missing: set[str] | None = None) -> None:
        self._missing = missing or set()

    def __getattr__(self, name: str) -> Any:
        if name in self._missing:
            raise AttributeError(f"operation {name!r} is unavailable")
        return getattr(np, name)


class _NeverUsedNamespace:
    """Makes accidental backend access before workload validation visible."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"invalid workloads must not access API operation {name!r}")


def _workload() -> Workload:
    return Workload(
        identifier="array-api-adapter-contract",
        seed=17,
        inputs=(
            InputSpec("x", (2, 3), "float32"),
            InputSpec("y", (2, 3), "float32"),
        ),
        nodes=(
            Node("add", ("x", "y"), "added"),
            Node("sin", ("added",), "sine"),
            Node("reshape", ("sine",), "reshaped", {"shape": [3, 2]}),
            Node("transpose", ("reshaped",), "transposed", {"axes": [1, 0]}),
            Node("sum", ("transposed",), "out", {"axis": 1}),
        ),
        outputs=("out",),
    )


def test_array_api_proxy_matches_numpy_for_a_valid_workload() -> None:
    workload = _workload()

    result = execute_array_api(workload, _NumpyNamespaceProxy(), backend="numpy-proxy")
    numpy_result = execute_numpy(workload)

    assert result.status == ExecutionStatus.OK
    assert result.backend == "numpy-proxy"
    assert result.mode == "eager"
    assert result.error is None
    assert numpy_result.status == ExecutionStatus.OK
    assert len(result.outputs) == len(numpy_result.outputs)
    for actual, expected in zip(result.outputs, numpy_result.outputs, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
        )


def test_array_api_reports_a_missing_required_operation_as_unsupported() -> None:
    result = execute_array_api(
        _workload(), _NumpyNamespaceProxy(missing={"sin"}), backend="limited-api"
    )

    assert result.status == ExecutionStatus.UNSUPPORTED_BACKEND
    assert result.backend == "limited-api"
    assert result.outputs == ()
    assert result.error is not None
    assert "sin" in result.error.lower()


def test_array_api_preserves_invalid_workload_status_before_accessing_backend() -> None:
    invalid = Workload(
        identifier="array-api-invalid-workload",
        seed=1,
        inputs=(InputSpec("x", (2, 2)),),
        nodes=(Node("neg", ("not_defined",), "out"),),
        outputs=("out",),
    )

    result = execute_array_api(invalid, _NeverUsedNamespace(), backend="unreachable-api")

    assert result.status == ExecutionStatus.INVALID_WORKLOAD
    assert result.backend == "unreachable-api"
    assert result.error is not None
