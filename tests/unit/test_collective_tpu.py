"""Unit tests for TPU-native SPMD collective simulations and ring AllReduce."""

from __future__ import annotations

import numpy as np

from xlaxiom.collective import (
    TPU_TOPOLOGIES,
    TPU_V4_MESH_2X2X1,
    TPU_V5E_MESH_2X2,
    PartitionSpec,
    shard_tensor,
    simulate_all_gather,
    simulate_ring_all_reduce,
    verify_sharding_equivalence,
)


def test_tpu_topologies_presets() -> None:
    assert TPU_V5E_MESH_2X2.total_devices == 4
    assert TPU_V5E_MESH_2X2.shape == (2, 2)
    assert TPU_V5E_MESH_2X2.axis_names == ("data", "model")

    assert TPU_V4_MESH_2X2X1.total_devices == 4
    assert TPU_V4_MESH_2X2X1.shape == (2, 2, 1)

    assert "tpu-v5e-2x2" in TPU_TOPOLOGIES
    assert "tpu-v4-2x2x1" in TPU_TOPOLOGIES


def test_tpu_v5e_sharding_equivalence() -> None:
    mesh = TPU_V5E_MESH_2X2
    spec = PartitionSpec(("data", "model"))
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert verify_sharding_equivalence(arr, mesh, spec) is True


def test_tpu_ring_all_reduce() -> None:
    mesh = TPU_V5E_MESH_2X2
    spec = PartitionSpec(("data", None))
    arr = np.ones((8, 8), dtype=np.float32)
    shards = shard_tensor(arr, mesh, spec)

    reduced_shards, metrics = simulate_ring_all_reduce(shards, mesh, axis_name="data", op="sum")
    assert metrics["ring_size"] == 2
    assert metrics["communication_steps"] == 2
    assert metrics["ring_equivalence_verified"] is True

    # Along 'data' axis of size 2, 2 shards of 1s summed = 2s
    for shard in reduced_shards.values():
        assert np.allclose(shard, 2.0)


def test_tpu_all_gather() -> None:
    mesh = TPU_V5E_MESH_2X2
    spec = PartitionSpec(("data", None))
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    shards = shard_tensor(arr, mesh, spec)

    gathered = simulate_all_gather(shards, mesh, axis_name="data", gather_dim=0)
    for _dev_id, g_arr in gathered.items():
        assert g_arr.shape == (4, 4)
