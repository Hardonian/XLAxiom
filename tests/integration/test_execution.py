from xlaxiom.execution import ExecutionStatus, execute_jax
from xlaxiom.generation import GenerationConfig, generate


def test_jax_execution_records_requested_cpu_device() -> None:
    result = execute_jax(generate(GenerationConfig(seed=2)), jit=True)
    assert result.status == ExecutionStatus.OK
    assert result.device is not None and "cpu" in result.device.lower()
    assert result.environment is not None
    assert result.environment["requested_platform"] == "cpu"


def test_generated_motifs_execute_with_numpy_and_jax() -> None:
    for seed in range(4):
        result = execute_jax(generate(GenerationConfig(seed=seed)), jit=True)
        assert result.status == ExecutionStatus.OK
