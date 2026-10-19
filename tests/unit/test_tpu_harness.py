"""Unit tests for Cloud TPU discovery, topology extraction, and emulator harness."""

from __future__ import annotations

from xlaxiom.execution import execute_jax
from xlaxiom.model import InputSpec, Node, Workload
from xlaxiom.tpu_harness import (
    audit_tpu_workload_equivalence,
    extract_tpu_topology,
    parse_tpu_device_string,
    tpu_hardware_emulator,
)


def _sample_workload() -> Workload:
    inp = InputSpec("x", (2, 2), "float32")
    node = Node("square", ("x",), "square_out")
    return Workload(
        identifier="tpu-test-workload",
        seed=42,
        inputs=(inp,),
        nodes=(node,),
        outputs=("square_out",),
    )



def test_parse_tpu_device_string() -> None:
    raw = "TPU_0(process=0,(0,0,0,0))"
    coord = parse_tpu_device_string(raw)
    assert coord.ordinal == 0
    assert coord.process_id == 0
    assert coord.coords == (0, 0, 0, 0)
    assert coord.raw_str == raw

    raw_alt = "TPU_3(process=1,(1,1,0,0))"
    coord_alt = parse_tpu_device_string(raw_alt)
    assert coord_alt.ordinal == 3
    assert coord_alt.process_id == 1
    assert coord_alt.coords == (1, 1, 0, 0)


def test_extract_tpu_topology_fallback() -> None:
    # On an environment without physical TPU, extract_tpu_topology returns zero devices
    topology = extract_tpu_topology()
    assert topology.is_simulated is False
    assert topology.num_devices >= 0


def test_tpu_hardware_emulator() -> None:
    workload = _sample_workload()
    with tpu_hardware_emulator(chip_type="v5e", num_chips=4) as sim_topo:
        assert sim_topo.is_simulated is True
        assert sim_topo.chip_type == "v5e"
        assert sim_topo.num_devices == 4
        assert sim_topo.mesh_shape == (2, 2)
        assert len(sim_topo.coordinates) == 4

        # Within the emulator, execute_jax with requested_platform="tpu" must succeed
        res = execute_jax(workload, jit=True, requested_platform="tpu")
        assert res.status.value == "ok"
        assert "tpu" in res.backend
        assert res.device is not None


def test_audit_tpu_workload_equivalence() -> None:
    workload = _sample_workload()
    audit = audit_tpu_workload_equivalence(workload, emulate_if_absent=True)
    assert audit["equivalent"] is True
    assert audit["cpu_status"] == "ok"
    assert audit["tpu_status"] == "ok"
    assert audit["classification"] == "pass"

