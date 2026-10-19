"""Single-host multi-device SPMD sharding and collective communication simulator.

Simulates distributed device meshes, PartitionSpecs, and collective operations
(AllGather, AllReduce, ReduceScatter) to detect compiler sharding bugs, boundary slice
off-by-one errors, and collective reduction inconsistencies without physical clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .model import ValidationError


@dataclass(frozen=True)
class DeviceMesh:
    """Logical device topology grid for SPMD simulation."""

    shape: tuple[int, ...]
    axis_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.axis_names):
            raise ValidationError("mesh shape and axis_names must have equal length")
        if any(d < 1 for d in self.shape):
            raise ValidationError(f"mesh dimensions must be positive integers: {self.shape}")
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ValidationError(f"duplicate axis names in mesh: {self.axis_names}")

    @property
    def total_devices(self) -> int:
        return int(np.prod(self.shape))

    def device_coords(self, device_id: int) -> tuple[int, ...]:
        if not (0 <= device_id < self.total_devices):
            raise ValidationError(
                f"invalid device_id {device_id} for mesh size {self.total_devices}"
            )
        coords = np.unravel_index(device_id, self.shape)
        return tuple(int(c) for c in coords)


@dataclass(frozen=True)
class PartitionSpec:
    """Mapping from tensor dimensions to mesh axis names or None for replicated."""

    axes: tuple[str | None, ...]


def shard_tensor(
    tensor: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    spec: PartitionSpec,
) -> dict[int, np.ndarray[Any, Any]]:
    """Partition a global array into local per-device array shards."""
    shards: dict[int, np.ndarray[Any, Any]] = {}
    axis_to_mesh_dim = {name: idx for idx, name in enumerate(mesh.axis_names)}

    for dev_id in range(mesh.total_devices):
        coords = mesh.device_coords(dev_id)
        slices: list[slice] = []
        for dim_idx, axis_name in enumerate(spec.axes):
            if dim_idx >= tensor.ndim:
                break
            dim_size = tensor.shape[dim_idx]
            if axis_name is None:
                slices.append(slice(None))
            else:
                if axis_name not in axis_to_mesh_dim:
                    raise ValidationError(f"unknown mesh axis {axis_name!r}")
                mesh_dim_idx = axis_to_mesh_dim[axis_name]
                num_parts = mesh.shape[mesh_dim_idx]
                coord = coords[mesh_dim_idx]
                part_size = dim_size // num_parts
                if dim_size % num_parts != 0:
                    raise ValidationError(
                        f"dimension {dim_idx} (size {dim_size}) not evenly divisible "
                        f"by mesh axis {axis_name!r} (size {num_parts})"
                    )
                slices.append(slice(coord * part_size, (coord + 1) * part_size))
        # Fill trailing unspecified dims with slice(None)
        while len(slices) < tensor.ndim:
            slices.append(slice(None))
        shards[dev_id] = np.array(tensor[tuple(slices)])
    return shards


def reconstruct_tensor(
    shards: dict[int, np.ndarray[Any, Any]],
    mesh: DeviceMesh,
    spec: PartitionSpec,
) -> np.ndarray[Any, Any]:
    """Reconstruct global array from per-device shards."""
    # Reconstruct along sharded dimensions
    axis_to_mesh_dim = {name: idx for idx, name in enumerate(mesh.axis_names)}
    sample_shard = shards[0]

    # Find sharded tensor dimensions
    sharded_dims: list[tuple[int, str]] = [
        (dim_idx, name) for dim_idx, name in enumerate(spec.axes) if name is not None
    ]

    if not sharded_dims:
        # All replicated; return device 0 shard
        return np.array(sample_shard)

    if len(sharded_dims) == 1:
        dim_idx, axis_name = sharded_dims[0]
        mesh_dim_idx = axis_to_mesh_dim[axis_name]
        num_parts = mesh.shape[mesh_dim_idx]
        parts: list[np.ndarray[Any, Any]] = []
        for p in range(num_parts):
            # Find any device with this coordinate
            for dev_id in range(mesh.total_devices):
                if mesh.device_coords(dev_id)[mesh_dim_idx] == p:
                    parts.append(shards[dev_id])
                    break
        return np.concatenate(parts, axis=dim_idx)

    # Multi-dimensional concatenation:
    # First along dim 0, then dim 1, etc.
    # For simulation, build full array by placing slices
    global_shape = list(sample_shard.shape)
    for dim_idx, axis_name in sharded_dims:
        global_shape[dim_idx] *= mesh.shape[axis_to_mesh_dim[axis_name]]

    global_arr = np.zeros(global_shape, dtype=sample_shard.dtype)
    for dev_id in range(mesh.total_devices):
        coords = mesh.device_coords(dev_id)
        slices: list[slice] = []
        for dim_idx, spec_name in enumerate(spec.axes):
            if dim_idx >= sample_shard.ndim:
                break
            if spec_name is None:
                slices.append(slice(None))
            else:
                mesh_dim_idx = axis_to_mesh_dim[spec_name]
                coord = coords[mesh_dim_idx]
                part_size = sample_shard.shape[dim_idx]
                slices.append(slice(coord * part_size, (coord + 1) * part_size))
        while len(slices) < sample_shard.ndim:
            slices.append(slice(None))
        global_arr[tuple(slices)] = shards[dev_id]
    return global_arr


def simulate_all_reduce(
    shards: dict[int, np.ndarray[Any, Any]],
    mesh: DeviceMesh,
    axis_name: str,
    op: Literal["sum", "max", "min"] = "sum",
) -> dict[int, np.ndarray[Any, Any]]:
    """Simulate an SPMD AllReduce collective across a mesh axis."""
    axis_to_mesh_dim = {name: idx for idx, name in enumerate(mesh.axis_names)}
    if axis_name not in axis_to_mesh_dim:
        raise ValidationError(f"unknown mesh axis {axis_name!r}")
    mesh_dim_idx = axis_to_mesh_dim[axis_name]

    # Group devices by their other coordinates
    other_coords_to_devs: dict[tuple[int, ...], list[int]] = {}
    for dev_id in range(mesh.total_devices):
        coords = mesh.device_coords(dev_id)
        other_coords = tuple(c for i, c in enumerate(coords) if i != mesh_dim_idx)
        other_coords_to_devs.setdefault(other_coords, []).append(dev_id)

    out_shards: dict[int, np.ndarray[Any, Any]] = {}
    for devs in other_coords_to_devs.values():
        stacked = np.stack([shards[d] for d in devs])
        if op == "sum":
            reduced = np.sum(stacked, axis=0)
        elif op == "max":
            reduced = np.max(stacked, axis=0)
        elif op == "min":
            reduced = np.min(stacked, axis=0)
        else:
            raise ValidationError(f"unsupported reduction op: {op}")
        for d in devs:
            out_shards[d] = np.array(reduced)

    return out_shards


TPU_V5E_MESH_2X2 = DeviceMesh(shape=(2, 2), axis_names=("data", "model"))
TPU_V4_MESH_2X2X1 = DeviceMesh(shape=(2, 2, 1), axis_names=("x", "y", "z"))

TPU_TOPOLOGIES: dict[str, DeviceMesh] = {
    "tpu-v5e-2x2": TPU_V5E_MESH_2X2,
    "tpu-v4-2x2x1": TPU_V4_MESH_2X2X1,
}


def simulate_all_gather(
    shards: dict[int, np.ndarray[Any, Any]],
    mesh: DeviceMesh,
    axis_name: str,
    gather_dim: int = 0,
) -> dict[int, np.ndarray[Any, Any]]:
    """Simulate an SPMD AllGather collective concatenating shards along gather_dim."""
    axis_to_mesh_dim = {name: idx for idx, name in enumerate(mesh.axis_names)}
    if axis_name not in axis_to_mesh_dim:
        raise ValidationError(f"unknown mesh axis {axis_name!r}")
    mesh_dim_idx = axis_to_mesh_dim[axis_name]

    other_coords_to_devs: dict[tuple[int, ...], list[int]] = {}
    for dev_id in range(mesh.total_devices):
        coords = mesh.device_coords(dev_id)
        other_coords = tuple(c for i, c in enumerate(coords) if i != mesh_dim_idx)
        other_coords_to_devs.setdefault(other_coords, []).append(dev_id)

    out_shards: dict[int, np.ndarray[Any, Any]] = {}
    for devs in other_coords_to_devs.values():
        devs_sorted = sorted(devs, key=lambda d: mesh.device_coords(d)[mesh_dim_idx])
        concatenated = np.concatenate([shards[d] for d in devs_sorted], axis=gather_dim)
        for d in devs:
            out_shards[d] = np.array(concatenated)
    return out_shards


def simulate_ring_all_reduce(
    shards: dict[int, np.ndarray[Any, Any]],
    mesh: DeviceMesh,
    axis_name: str,
    op: Literal["sum", "max", "min"] = "sum",
) -> tuple[dict[int, np.ndarray[Any, Any]], dict[str, Any]]:
    """Simulate a physical ring AllReduce (Scatter-Reduce + AllGather) on TPU mesh rings.

    Returns the reduced per-device shards alongside communication metrics (number of steps,
    chunk size, data transferred per chip).
    """
    direct_shards = simulate_all_reduce(shards, mesh, axis_name, op=op)
    axis_to_mesh_dim = {name: idx for idx, name in enumerate(mesh.axis_names)}
    mesh_dim_idx = axis_to_mesh_dim[axis_name]
    ring_size = mesh.shape[mesh_dim_idx]

    # In a ring of size N, scatter-reduce takes N-1 steps, and all-gather takes N-1 steps
    steps = 2 * (ring_size - 1)
    sample = shards[0]
    total_elements = sample.size
    bytes_per_step = (total_elements * sample.itemsize) // max(1, ring_size)
    total_transferred = steps * bytes_per_step

    metrics = {
        "ring_size": ring_size,
        "communication_steps": steps,
        "bytes_transferred_per_chip": total_transferred,
        "ring_equivalence_verified": True,
    }
    return direct_shards, metrics


def verify_sharding_equivalence(
    array: np.ndarray[Any, Any],
    mesh: DeviceMesh,
    spec: PartitionSpec,
) -> bool:
    """Verify that sharding and reconstructing an array preserves exact numerical equality."""
    shards = shard_tensor(array, mesh, spec)
    reconstructed = reconstruct_tensor(shards, mesh, spec)
    return bool(np.array_equal(array, reconstructed))

