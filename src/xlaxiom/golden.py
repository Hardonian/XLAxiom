"""Golden-file (snapshot) testing for numerical outputs.

Record a trusted result once, then compare every future run against it. Mismatches raise
``NumericalAssertionError`` with the full XLAxiom diagnostic; a *missing* golden file fails
closed (it never silently passes), and updating is always explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .assertions import NumericalAssertionError, NumericalDiff, _to_numpy, diff
from .ulp import max_ulp_error

UPDATE_ENV_VAR = "XLAXIOM_UPDATE_GOLDEN"
_META_KEY = "__xlaxiom_meta__"
_SINGLE_KEY = "data"


def _normalize(actual: Any) -> dict[str, np.ndarray[Any, Any]]:
    if isinstance(actual, Mapping):
        if not actual:
            raise ValueError("Cannot snapshot an empty mapping of tensors.")
        return {str(k): _to_numpy(v) for k, v in actual.items()}
    return {_SINGLE_KEY: _to_numpy(actual)}


def _digest(tensors: Mapping[str, np.ndarray[Any, Any]]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def save_golden(path: str | Path, actual: Any) -> Path:
    """Write ``actual`` (a tensor or ``{name: tensor}`` mapping) as a golden ``.npz`` file."""
    tensors = _normalize(actual)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "xlaxiom_version": __version__,
        "sha256": _digest(tensors),
        "tensors": {
            k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in tensors.items()
        },
    }
    np.savez(out, **tensors, **{_META_KEY: np.array(json.dumps(meta, sort_keys=True))})
    return out


def load_golden(path: str | Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load a golden file and verify its recorded checksum (detects corruption/tampering)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Golden file not found: {p}. Record it explicitly with "
            f"{UPDATE_ENV_VAR}=1 or `xlaxiom snapshot record`."
        )
    with np.load(p, allow_pickle=False) as data:
        tensors = {k: np.asarray(data[k]) for k in data.files if k != _META_KEY}
        meta_raw = str(data[_META_KEY]) if _META_KEY in data.files else None
    if meta_raw is not None:
        expected = json.loads(meta_raw).get("sha256")
        if expected and expected != _digest(tensors):
            raise ValueError(f"Golden file '{p}' failed its checksum; it is corrupt or edited.")
    return tensors


def check_golden(
    path: str | Path,
    actual: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_ulp: float | None = None,
    update: bool | None = None,
) -> dict[str, NumericalDiff]:
    """Compare ``actual`` against the golden file at ``path``.

    Parameters
    ----------
    update : bool, optional
        Rewrite the golden file instead of comparing. Defaults to ``True`` only when the
        ``XLAXIOM_UPDATE_GOLDEN`` environment variable is ``1``.
    max_ulp : float, optional
        Additionally require every element within this many ULPs.

    Returns
    -------
    dict[str, NumericalDiff]
        Per-tensor diagnostics (the single-tensor key is ``"data"``).

    Raises
    ------
    NumericalAssertionError
        On any tolerance, ULP, key, or shape violation.
    FileNotFoundError
        If the golden file does not exist and ``update`` is not set.
    """
    if update is None:
        update = os.environ.get(UPDATE_ENV_VAR) == "1"
    current = _normalize(actual)
    if update:
        save_golden(path, actual)
        return {}

    golden = load_golden(path)
    missing = sorted(set(golden) - set(current))
    extra = sorted(set(current) - set(golden))
    if missing or extra:
        raise NumericalAssertionError(
            f"Golden key mismatch for '{path}': missing={missing}, unexpected={extra}."
        )

    diffs: dict[str, NumericalDiff] = {}
    failures: list[str] = []
    for name in sorted(golden):
        d = diff(current[name], golden[name], rtol=rtol, atol=atol)
        diffs[name] = d
        if not d.is_close:
            failures.append(f"[{name}] {d.diagnosis}")
        elif max_ulp is not None and current[name].shape == golden[name].shape:
            worst = max_ulp_error(current[name], golden[name])
            if worst > max_ulp:
                failures.append(f"[{name}] max ULP error {worst:g} exceeds {max_ulp:g}")
    if failures:
        first_bad = next((d for d in diffs.values() if not d.is_close), None)
        raise NumericalAssertionError(
            f"Golden comparison failed against '{path}':\n" + "\n".join(failures), diff=first_bad
        )
    return diffs
