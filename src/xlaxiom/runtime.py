"""Reviewed, subprocess-only CPU runtime profiles for reproducibility studies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    identifier: str
    environment: dict[str, str]
    description: str


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "default": RuntimeProfile("default", {}, "inherit the operator's runtime thread settings"),
    "single-thread": RuntimeProfile(
        "single-thread",
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "one CPU worker thread for common numerical runtime libraries",
    ),
    "four-thread": RuntimeProfile(
        "four-thread",
        {
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        },
        "four CPU worker threads for common numerical runtime libraries",
    ),
}


def runtime_profile(identifier: str) -> RuntimeProfile:
    try:
        return RUNTIME_PROFILES[identifier]
    except KeyError as exc:
        raise ValueError(f"unknown runtime profile {identifier!r}") from exc
