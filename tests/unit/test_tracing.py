import jax.numpy as jnp
import numpy as np
import pytest

from xlaxiom.execution import execute_numpy
from xlaxiom.model import Workload
from xlaxiom.tracing import (
    TracingError,
    capture,
    extract_subgraph,
    load_and_trace_target,
    trace_jax,
)


def test_trace_jax_basic_arithmetic() -> None:
    def simple_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(x) + jnp.cos(y)

    x = np.ones((2, 3), dtype=np.float32)
    y = np.ones((2, 3), dtype=np.float32)

    workload = trace_jax(simple_fn, x, y, identifier="test-trace-1")
    assert isinstance(workload, Workload)
    assert workload.identifier == "test-trace-1"
    assert len(workload.inputs) == 2
    assert len(workload.nodes) == 3
    ops = [node.op for node in workload.nodes]
    assert ops == ["sin", "cos", "add"]

    res = execute_numpy(workload)
    assert res.status.value == "ok"
    assert len(res.outputs) == 1
    assert res.outputs[0].shape == (2, 3)


def test_trace_jax_reductions_and_constants() -> None:
    def reduction_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x * 2.0)

    x = np.ones((4, 4), dtype=np.float32)
    workload = trace_jax(reduction_fn, x, identifier="test-trace-const")
    assert isinstance(workload, Workload)
    assert len(workload.nodes) >= 2

    res = execute_numpy(workload)
    assert res.status.value == "ok"


def test_extract_subgraph() -> None:
    def multi_branch(x: jnp.ndarray) -> jnp.ndarray:
        a = jnp.sin(x)
        b = jnp.cos(x)
        c = jnp.square(b)
        return a + c

    x = np.ones((2, 2), dtype=np.float32)
    workload = trace_jax(multi_branch, x)
    assert len(workload.nodes) == 4

    # Extract only up to node 'cos'
    cos_node = [n for n in workload.nodes if n.op == "cos"][0]
    sub = extract_subgraph(workload, [cos_node.output])
    assert len(sub.nodes) == 1
    assert sub.nodes[0].op == "cos"


def test_capture_decorator_preserves_function() -> None:
    @capture()
    def my_layer(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(x)

    x = jnp.ones((2, 2), dtype=jnp.float32)
    out = my_layer(x)
    assert out.shape == (2, 2)
    assert my_layer.last_divergence is None


def test_trace_target_invalid_syntax() -> None:
    with pytest.raises(TracingError, match="target must be formatted"):
        load_and_trace_target("invalid_no_colon", [(2, 2)])
