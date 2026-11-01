"""Unit tests for dependency-free safetensors reader/writer and sharding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xlaxiom.safetensors_io import (
    INDEX_FILENAME,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    read_checkpoint,
    read_safetensors,
    write_safetensors,
)


def test_bfloat16_bit_roundtrip() -> None:
    # Test special values: 0, 1.0, -1.0, 2.0, subnormals, infinity, nan
    special = np.array([0.0, -0.0, 1.0, -1.0, 2.0, 0.5, np.inf, -np.inf], dtype=np.float32)
    bits = float32_to_bfloat16_bits(special)
    reconstructed = bfloat16_bits_to_float32(bits)
    np.testing.assert_array_equal(reconstructed[:6], special[:6])
    assert np.isposinf(reconstructed[6])
    assert np.isneginf(reconstructed[7])

    # NaN handling
    nan_arr = np.array([np.nan], dtype=np.float32)
    nan_bits = float32_to_bfloat16_bits(nan_arr)
    nan_recon = bfloat16_bits_to_float32(nan_bits)
    assert np.isnan(nan_recon[0])


def test_safetensors_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    tensors = {
        "weight_f32": np.random.randn(4, 8).astype(np.float32),
        "weight_f16": np.random.randn(2, 3).astype(np.float16),
        "weight_i32": np.arange(10, dtype=np.int32),
        "mask": np.array([True, False, True, True]),
    }
    metadata = {"format": "pt", "author": "xlaxiom"}

    write_safetensors(path, tensors, metadata=metadata)
    loaded = read_safetensors(path)

    assert set(loaded.keys()) == set(tensors.keys())
    np.testing.assert_array_equal(loaded["weight_f32"], tensors["weight_f32"])
    np.testing.assert_array_equal(loaded["weight_f16"], tensors["weight_f16"])
    np.testing.assert_array_equal(loaded["weight_i32"], tensors["weight_i32"])
    np.testing.assert_array_equal(loaded["mask"], tensors["mask"])


def test_safetensors_bfloat16_write_read(tmp_path: Path) -> None:
    path = tmp_path / "bf16.safetensors"
    orig_f32 = np.array([[1.0, 2.5], [3.25, 4.125]], dtype=np.float32)

    write_safetensors(path, {"layer.weight": orig_f32}, bf16=["layer.weight"])
    loaded = read_safetensors(path)

    assert "layer.weight" in loaded
    assert loaded["layer.weight"].dtype == np.float32
    # bf16 can represent these numbers exactly
    np.testing.assert_allclose(loaded["layer.weight"], orig_f32, atol=1e-6)


def test_read_sharded_checkpoint(tmp_path: Path) -> None:
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    index_file = tmp_path / INDEX_FILENAME

    w1 = np.ones((2, 2), dtype=np.float32)
    w2 = np.full((3, 3), 2.0, dtype=np.float32)

    write_safetensors(shard1, {"layer1.weight": w1})
    write_safetensors(shard2, {"layer2.weight": w2})

    index_data = {
        "metadata": {"total_size": 1024},
        "weight_map": {
            "layer1.weight": "model-00001-of-00002.safetensors",
            "layer2.weight": "model-00002-of-00002.safetensors",
        },
    }
    index_file.write_text(json.dumps(index_data), encoding="utf-8")

    # Load via index file
    loaded_from_index = read_checkpoint(index_file)
    assert set(loaded_from_index.keys()) == {"layer1.weight", "layer2.weight"}
    np.testing.assert_array_equal(loaded_from_index["layer1.weight"], w1)
    np.testing.assert_array_equal(loaded_from_index["layer2.weight"], w2)

    # Load via directory path containing index
    loaded_from_dir = read_checkpoint(tmp_path)
    assert set(loaded_from_dir.keys()) == {"layer1.weight", "layer2.weight"}


def test_read_directory_single_or_multi_shards(tmp_path: Path) -> None:
    # Directory with multiple *.safetensors without an index file
    s1 = tmp_path / "part_a.safetensors"
    s2 = tmp_path / "part_b.safetensors"
    write_safetensors(s1, {"a": np.array([1, 2])})
    write_safetensors(s2, {"b": np.array([3, 4])})

    merged = read_checkpoint(tmp_path)
    assert set(merged.keys()) == {"a", "b"}


def test_corrupt_safetensors_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.safetensors"
    bad_file.write_bytes(b"short")
    with pytest.raises(ValueError, match="shorter than 8 bytes"):
        read_safetensors(bad_file)

    bad_header = tmp_path / "bad_hdr.safetensors"
    bad_header.write_bytes((100).to_bytes(8, "little") + b"not valid json")
    with pytest.raises(ValueError, match="invalid safetensors header length|corrupt"):
        read_safetensors(bad_header)

