"""Dependency-free ``.safetensors`` reader/writer with bfloat16 and sharded-checkpoint support.

The reference ``safetensors.numpy`` loader cannot read ``BF16`` tensors (NumPy has no native
bfloat16), yet bf16 is the default dtype of most modern LLM checkpoints. This module parses the
(simple, documented) safetensors container directly, widening ``BF16`` to ``float32`` losslessly.

Format: ``[u64 little-endian header length][JSON header][raw tensor bytes]``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

INDEX_FILENAME = "model.safetensors.index.json"
_MAX_HEADER_BYTES = 100 * 1024 * 1024

_DTYPES: dict[str, np.dtype[Any]] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}
_BF16 = "BF16"
_SAFETENSORS_NAMES = {np.dtype(v).newbyteorder("<"): k for k, v in _DTYPES.items()}


def bfloat16_bits_to_float32(bits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Widen raw bfloat16 bit patterns (``uint16``) to ``float32`` (exact)."""
    widened = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    result: np.ndarray[Any, Any] = widened.view(np.float32)
    return result


def float32_to_bfloat16_bits(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Round ``float32`` values to bfloat16 bit patterns (round-to-nearest-even, NaN-safe)."""
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    rounded = ((u32 + rounding) >> 16).astype(np.uint16)
    nan_bits = ((u32 >> 16) | np.uint32(0x0040)).astype(np.uint16)
    result: np.ndarray[Any, Any] = np.where(np.isnan(f32), nan_bits, rounded).astype(np.uint16)
    return result


def _parse_header_from_file(handle: Any, path: Path) -> dict[str, Any]:
    len_bytes = handle.read(8)
    if len(len_bytes) < 8:
        raise ValueError(f"'{path}' is not a safetensors file (shorter than 8 bytes).")
    header_len = int.from_bytes(len_bytes, "little")
    if header_len > _MAX_HEADER_BYTES:
        raise ValueError(f"'{path}' has an invalid safetensors header length ({header_len}).")
    header_raw = handle.read(header_len)
    if len(header_raw) < header_len:
        raise ValueError(f"'{path}' has an invalid safetensors header length ({header_len}).")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"'{path}' has a corrupt safetensors JSON header: {err}") from err
    if not isinstance(header, dict):
        raise ValueError(f"'{path}' safetensors header must be a JSON object.")
    return header


def read_safetensors(path: str | Path) -> dict[str, np.ndarray[Any, Any]]:
    """Read every tensor of a ``.safetensors`` file into NumPy arrays.

    ``BF16`` tensors are returned as ``float32``. Unsupported dtypes (e.g. FP8) and malformed
    files raise ``ValueError`` with the offending tensor name.
    """
    p = Path(path)
    with p.open("rb") as f:
        header = _parse_header_from_file(f, p)
        data = np.fromfile(f, dtype=np.uint8)
    tensors: dict[str, np.ndarray[Any, Any]] = {}

    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype_name = str(info["dtype"])
            shape = tuple(int(s) for s in info["shape"])
            begin, end = (int(o) for o in info["data_offsets"])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"'{p}': malformed header entry for tensor '{name}'.") from err

        if dtype_name == _BF16:
            itemsize = 2
        elif dtype_name in _DTYPES:
            itemsize = _DTYPES[dtype_name].itemsize
        else:
            raise ValueError(f"'{p}': tensor '{name}' uses unsupported dtype '{dtype_name}'.")

        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if not (0 <= begin <= end <= len(data)) or end - begin != count * itemsize:
            raise ValueError(f"'{p}': tensor '{name}' has inconsistent data offsets or shape.")

        chunk = data[begin:end]
        if dtype_name == _BF16:
            tensors[name] = bfloat16_bits_to_float32(chunk.view("<u2")).reshape(shape)
        else:
            tensors[name] = chunk.view(_DTYPES[dtype_name]).reshape(shape).copy()
    return tensors


def write_safetensors(
    path: str | Path,
    tensors: Mapping[str, Any],
    *,
    bf16: Iterable[str] = (),
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write tensors to a ``.safetensors`` file.

    Tensors named in ``bf16`` are stored as ``BF16`` (rounded from float32); everything else
    keeps its NumPy dtype.
    """
    bf16_names = set(bf16)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = np.asarray(tensors[name])
        if name in bf16_names:
            dtype_name = _BF16
            blob = float32_to_bfloat16_bits(arr.astype(np.float32)).astype("<u2").tobytes()
        else:
            key = np.dtype(arr.dtype).newbyteorder("<")
            if key not in _SAFETENSORS_NAMES:
                raise ValueError(f"Tensor '{name}' has unsupported dtype {arr.dtype}.")
            dtype_name = _SAFETENSORS_NAMES[key]
            blob = np.ascontiguousarray(arr.astype(key, copy=False)).tobytes()
        header[name] = {
            "dtype": dtype_name,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)  # keep tensor data 8-byte aligned
    with Path(path).open("wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def _read_sharded(index_path: Path) -> dict[str, np.ndarray[Any, Any]]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as err:
        raise ValueError(f"'{index_path}' is not a valid safetensors shard index: {err}") from err
    if not isinstance(weight_map, dict):
        raise ValueError(f"'{index_path}': 'weight_map' must be an object.")

    shards: dict[str, list[str]] = {}
    for tensor_name, shard in weight_map.items():
        if Path(str(shard)).name != str(shard):
            raise ValueError(f"'{index_path}': shard '{shard}' must be a bare file name.")
        shards.setdefault(str(shard), []).append(str(tensor_name))

    result: dict[str, np.ndarray[Any, Any]] = {}
    for shard, names in sorted(shards.items()):
        shard_path = index_path.parent / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard listed in index is missing: {shard_path}")
        loaded = read_safetensors(shard_path)
        for tensor_name in names:
            if tensor_name not in loaded:
                raise ValueError(f"Tensor '{tensor_name}' not found in shard '{shard}'.")
            result[tensor_name] = loaded[tensor_name]
    return result


def read_checkpoint(path: str | Path) -> dict[str, np.ndarray[Any, Any]]:
    """Read a safetensors file, a shard-index JSON, or a checkpoint directory.

    A directory is resolved in order: ``model.safetensors.index.json``, ``model.safetensors``,
    then every ``*.safetensors`` file it contains (duplicate tensor names are an error).
    """
    p = Path(path)
    if p.is_dir():
        index = p / INDEX_FILENAME
        if index.is_file():
            return _read_sharded(index)
        single = p / "model.safetensors"
        if single.is_file():
            return read_safetensors(single)
        files = sorted(p.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors checkpoint found in directory: {p}")
        merged: dict[str, np.ndarray[Any, Any]] = {}
        for file in files:
            for name, tensor in read_safetensors(file).items():
                if name in merged:
                    raise ValueError(f"Duplicate tensor '{name}' across shards in {p}.")
                merged[name] = tensor
        return merged
    if p.suffix.lower() == ".json":
        return _read_sharded(p)
    return read_safetensors(p)
