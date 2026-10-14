import numpy as np
import pytest

from xlaxiom.execution import ExecutionStatus, execute_jax, execute_numpy
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.model import ValidationError, Workload, _broadcast_shape, materialize_inputs


def test_generation_is_deterministic_and_canonical() -> None:
    first = generate(GenerationConfig(seed=99))
    second = generate(GenerationConfig(seed=99))
    assert first.canonical_json() == second.canonical_json()
    assert first.digest() == second.digest()
    assert (materialize_inputs(first)["x"] == materialize_inputs(second)["x"]).all()


def test_rejects_forward_reference() -> None:
    data = generate(GenerationConfig()).canonical_dict()
    data["nodes"][0]["inputs"] = ["future", "y"]
    try:
        Workload.from_dict(data)
    except ValidationError as exc:
        assert "forward" in str(exc)
    else:
        raise AssertionError("invalid workload accepted")


def test_generator_covers_rank_three_and_integer_safe_motifs() -> None:
    rank_three = generate(GenerationConfig(seed=4, max_rank=3))
    integer = generate(GenerationConfig(seed=1, dtype="int32"))

    assert max(len(spec.shape) for spec in rank_three.inputs) == 3
    assert all(spec.dtype == "int32" for spec in integer.inputs)

    with pytest.raises(ValueError, match="unsupported generation dtype"):
        generate(GenerationConfig(dtype="float16"))


def test_materialization_supports_explicit_float_edge_strategies() -> None:
    signed_zero = generate(GenerationConfig(seed=7, input_mode="signed_zero"))
    nonfinite = generate(GenerationConfig(seed=8, input_mode="nonfinite"))

    signed_values = materialize_inputs(signed_zero)["x"].reshape(-1)
    nonfinite_values = materialize_inputs(nonfinite)["x"].reshape(-1)

    assert signed_values[0] == 0 and signed_values[1] == 0
    assert signed_values[0].view("u4") != signed_values[1].view("u4")
    assert pytest.approx(float("inf")) == float(nonfinite_values[1])
    assert nonfinite_values[0] != nonfinite_values[0]


def test_generator_rejects_nonfinite_integer_inputs() -> None:
    with pytest.raises(ValueError, match="requires a floating-point dtype"):
        generate(GenerationConfig(dtype="int32", input_mode="nonfinite"))


def test_extreme_finite_input_strategy_controls_overflow_and_underflow_edges() -> None:
    workload = generate(GenerationConfig(seed=9, input_mode="extreme_finite"))
    values = materialize_inputs(workload)["x"].reshape(-1)

    assert np.isfinite(values[:6]).all()
    assert abs(values[0]) > 1e30
    assert 0 < abs(values[2]) < 1e-30


def test_broadcast_shape_valid() -> None:
    assert _broadcast_shape() == ()
    assert _broadcast_shape((2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), (2, 3)) == (2, 3)
    assert _broadcast_shape((2, 3), (1, 3)) == (2, 3)
    assert _broadcast_shape((1, 3), (2, 1)) == (2, 3)
    assert _broadcast_shape((5, 1, 4), (1, 3, 1)) == (5, 3, 4)
    assert _broadcast_shape((2, 1, 4), (3, 1), (1,)) == (2, 3, 4)
    assert _broadcast_shape((), (2, 3), ()) == (2, 3)


def test_broadcast_shape_incompatible() -> None:
    with pytest.raises(ValidationError, match="incompatible broadcast shapes"):
        _broadcast_shape((2, 3), (4, 3))

    with pytest.raises(ValidationError, match="incompatible broadcast shapes"):
        _broadcast_shape((5, 2, 4), (5, 3, 4))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"axis": [2]}, "reduction axis must be between"),
        ({"axis": [-3]}, "reduction axis must be between"),
        ({"axis": [0, 0]}, "duplicate reduction axis"),
        ({"axis": [0], "keepdims": 1}, "keepdims must be a boolean"),
        ({"axis": [0], "unexpected": True}, "unsupported parameters"),
    ],
)
def test_rejects_malformed_reduction_parameters_before_execution(params, message) -> None:
    data = generate(GenerationConfig(seed=1)).canonical_dict()
    data["nodes"] = [
        {
            "op": "sum",
            "inputs": ["x"],
            "output": "out",
            "params": params,
        }
    ]
    data["outputs"] = ["out"]

    with pytest.raises(ValidationError, match=message):
        Workload.from_dict(data)


def test_rejects_malformed_operation_parameter_types_without_coercion() -> None:
    data = generate(GenerationConfig(seed=1)).canonical_dict()
    data["nodes"] = [
        {
            "op": "reshape",
            "inputs": ["x"],
            "output": "out",
            "params": {"shape": ["4"]},
        }
    ]
    data["outputs"] = ["out"]

    with pytest.raises(ValidationError, match="reshape shape must be an array of integers"):
        Workload.from_dict(data)


def test_generator_preferred_op_support() -> None:
    for op in ("maximum", "minimum", "sqrt", "square", "log", "prod", "max", "min", "matmul"):
        workload = generate(GenerationConfig(seed=42, preferred_op=op))
        workload.validate()
        assert any(node.op == op for node in workload.nodes)

    with pytest.raises(ValueError, match="unsupported preferred operation 'not_an_op'"):
        generate(GenerationConfig(preferred_op="not_an_op"))


def test_expanded_operations_execute_numpy_and_jax() -> None:
    for op in ("maximum", "minimum", "sqrt", "square", "log", "prod", "max", "min"):
        # For sqrt and log, use positive inputs
        workload = generate(GenerationConfig(seed=10, preferred_op=op, input_mode="ones"))
        np_res = execute_numpy(workload)
        assert np_res.status == ExecutionStatus.OK
        assert len(np_res.outputs) == 1
        jax_res = execute_jax(workload, jit=True)
        assert jax_res.status == ExecutionStatus.OK
        assert len(jax_res.outputs) == 1
        np.testing.assert_allclose(
            np.asarray(np_res.outputs[0], dtype=np.float64),
            np.asarray(jax_res.outputs[0], dtype=np.float64),
            rtol=1e-5,
            atol=1e-6,
        )


def test_generate_boundary_suite() -> None:
    from xlaxiom.generation import generate_boundary_suite
    workload = generate(GenerationConfig(seed=7, dtype="float32"))
    suite = generate_boundary_suite(workload)
    assert len(suite) == 6
    modes = [w.metadata["boundary_mode"] for w in suite]
    assert modes == ["zeros", "ones", "finite_edges", "extreme_finite", "signed_zero", "nonfinite"]
    for variant in suite:
        variant.validate()
        inputs = materialize_inputs(variant)
        assert len(inputs) == len(variant.inputs)

