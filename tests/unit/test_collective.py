"""Unit tests for SPMD sharding and collective simulator (Priority 66)."""

import numpy as np
import pytest

from xlaxiom.collective import (
    DeviceMesh,
    PartitionSpec,
    shard_tensor,
    simulate_all_reduce,
    verify_sharding_equivalence,
)
from xlaxiom.model import ValidationError


def test_mesh_validation() -> None:
    mesh = DeviceMesh(shape=(2, 4), axis_names=("data", "model"))
    assert mesh.total_devices == 8
    assert mesh.device_coords(0) == (0, 0)
    assert mesh.device_coords(7) == (1, 3)

    with pytest.raises(ValidationError, match="equal length"):
        DeviceMesh(shape=(2,), axis_names=("x", "y"))


def test_shard_and_reconstruct_1d() -> None:
    mesh = DeviceMesh(shape=(4,), axis_names=("devices",))
    spec = PartitionSpec(axes=("devices", None))
    arr = np.arange(32, dtype=np.float32).reshape(8, 4)

    assert verify_sharding_equivalence(arr, mesh, spec)

    shards = shard_tensor(arr, mesh, spec)
    assert len(shards) == 4
    for _dev_id, shard in shards.items():
        assert shard.shape == (2, 4)


def test_shard_and_reconstruct_2d() -> None:
    mesh = DeviceMesh(shape=(2, 2), axis_names=("x", "y"))
    spec = PartitionSpec(axes=("x", "y"))
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)

    assert verify_sharding_equivalence(arr, mesh, spec)


def test_simulate_all_reduce() -> None:
    mesh = DeviceMesh(shape=(4,), axis_names=("devices",))
    # Each device has a local array of ones
    shards = {dev: np.ones((2, 2), dtype=np.float32) * (dev + 1) for dev in range(4)}
    # Sum: 1 + 2 + 3 + 4 = 10
    reduced = simulate_all_reduce(shards, mesh, axis_name="devices", op="sum")

    for dev in range(4):
        np.testing.assert_allclose(reduced[dev], np.full((2, 2), 10.0, dtype=np.float32))
