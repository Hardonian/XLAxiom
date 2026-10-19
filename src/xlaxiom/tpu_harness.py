"""Cloud TPU device discovery, topology extraction, and execution harness.

Provides discovery and extraction of Google Cloud TPU pod topologies (v5e ViperLite,
v4), structured device coordinate parsing, and a deterministic TPUHardwareEmulator
for offline test environments without physical TPU quota.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

from .execution import ExecutionResult, ExecutionStatus, execute_jax
from .hardware import discover_tpu_devices
from .model import Workload


@dataclass(frozen=True)
class TPUDeviceCoordinate:
    """Parsed hardware coordinates of a physical or simulated TPU core."""

    ordinal: int
    process_id: int
    coords: tuple[int, ...]
    raw_str: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TPUTopologyInfo:
    """Detailed Google Cloud TPU topology specification."""

    chip_type: str
    num_devices: int
    mesh_shape: tuple[int, ...]
    coordinates: list[TPUDeviceCoordinate]
    is_simulated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "chip_type": self.chip_type,
            "num_devices": self.num_devices,
            "mesh_shape": list(self.mesh_shape),
            "coordinates": [c.to_dict() for c in self.coordinates],
            "is_simulated": self.is_simulated,
        }


_TPU_PATTERN = re.compile(
    r"TPU_(?P<ordinal>\d+)\(process=(?P<process>\d+),\((?P<coords>[0-9,]+)\)\)",
    re.IGNORECASE,
)


def parse_tpu_device_string(device_str: str) -> TPUDeviceCoordinate:
    """Parse JAX TPU device descriptor into structured coordinates.

    Example: ``TPU_0(process=0,(0,0,0,0))`` -> ordinal=0, process_id=0, coords=(0,0,0,0).
    """
    match = _TPU_PATTERN.search(device_str)
    if not match:
        # Fallback for alternative JAX formatting
        return TPUDeviceCoordinate(
            ordinal=0,
            process_id=0,
            coords=(0,),
            raw_str=device_str,
        )
    ordinal = int(match.group("ordinal"))
    process_id = int(match.group("process"))
    coords = tuple(int(x.strip()) for x in match.group("coords").split(",") if x.strip())
    return TPUDeviceCoordinate(
        ordinal=ordinal,
        process_id=process_id,
        coords=coords,
        raw_str=device_str,
    )


def extract_tpu_topology() -> TPUTopologyInfo:
    """Extract physical TPU pod topology from the active JAX runtime if present."""
    probe = discover_tpu_devices()
    if not probe["tpu_available"] or not probe["devices"]:
        return TPUTopologyInfo(
            chip_type="none",
            num_devices=0,
            mesh_shape=(),
            coordinates=[],
            is_simulated=False,
        )

    coords = [parse_tpu_device_string(d) for d in probe["devices"]]
    num_devices = len(coords)

    # Infer mesh shape
    mesh_shape: tuple[int, ...]
    if num_devices == 4:
        mesh_shape = (2, 2)
    elif num_devices == 8:
        mesh_shape = (2, 2, 2)
    elif num_devices == 16:
        mesh_shape = (2, 4, 2)
    else:
        mesh_shape = (num_devices,)

    return TPUTopologyInfo(
        chip_type=probe["chip_type"],
        num_devices=num_devices,
        mesh_shape=mesh_shape,
        coordinates=coords,
        is_simulated=False,
    )


@contextmanager
def tpu_hardware_emulator(
    chip_type: str = "v5e",
    num_chips: int = 4,
) -> Generator[TPUTopologyInfo, None, None]:
    """Context manager to emulate TPU hardware execution locally.

    Intercepts JAX execution requests for ``tpu`` and executes via CPU JIT while
    attributing execution to simulated TPU device coordinates and mesh topology.
    """
    sim_coords = []
    mesh_shape: tuple[int, ...]
    if num_chips == 4:
        mesh_shape = (2, 2)
        grid = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)]
    else:
        mesh_shape = (num_chips,)
        grid = [(i, 0, 0, 0) for i in range(num_chips)]

    for ord_idx, c in enumerate(grid[:num_chips]):
        sim_coords.append(
            TPUDeviceCoordinate(
                ordinal=ord_idx,
                process_id=0,
                coords=c,
                raw_str=f"TPU_{ord_idx}(process=0,({','.join(map(str, c))}))",
            )
        )

    sim_topology = TPUTopologyInfo(
        chip_type=chip_type,
        num_devices=num_chips,
        mesh_shape=mesh_shape,
        coordinates=sim_coords,
        is_simulated=True,
    )

    import os
    prev_env = os.environ.get("XLAXIOM_EMULATE_TPU")
    os.environ["XLAXIOM_EMULATE_TPU"] = "1"
    try:
        yield sim_topology
    finally:
        if prev_env is None:
            os.environ.pop("XLAXIOM_EMULATE_TPU", None)
        else:
            os.environ["XLAXIOM_EMULATE_TPU"] = prev_env



def audit_tpu_workload_equivalence(
    workload: Workload,
    emulate_if_absent: bool = True,
) -> dict[str, Any]:
    """Execute differential comparison between CPU JIT and TPU JIT for a workload."""
    from .comparison import Classification, compare

    cpu_res = execute_jax(workload, jit=True, requested_platform="cpu")

    probe = discover_tpu_devices()
    if probe["tpu_available"]:
        tpu_res = execute_jax(workload, jit=True, requested_platform="tpu")
        topology = extract_tpu_topology()
    elif emulate_if_absent:
        with tpu_hardware_emulator() as sim_topo:
            tpu_res = execute_jax(workload, jit=True, requested_platform="tpu")
            topology = sim_topo
    else:
        tpu_res = ExecutionResult(
            backend="jax-tpu",
            mode="jit",
            status=ExecutionStatus.UNSUPPORTED_BACKEND,
            error="no physical TPU device present and emulation disabled",
        )
        topology = extract_tpu_topology()

    comparison = compare(cpu_res, tpu_res, policy=workload.policy)
    return {
        "workload_sha256": workload.digest(),
        "topology": topology.to_dict(),
        "cpu_status": cpu_res.status.value,
        "tpu_status": tpu_res.status.value,
        "classification": comparison.classification.value,
        "metrics": [asdict(m) for m in comparison.metrics] if comparison.metrics else [],
        "equivalent": comparison.classification == Classification.PASS,
    }


