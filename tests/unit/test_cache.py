from xlaxiom.performance import compare_compilation_cache


def test_compilation_cache_observation_is_identity_bound_and_conservative() -> None:
    cold = {"backend": "jax-cpu-jit", "workload_sha256": "same", "compilation_ms": 10.0}
    warm = {"backend": "jax-cpu-jit", "workload_sha256": "same", "compilation_ms": 5.0}

    assert compare_compilation_cache(cold, warm)["classification"] == "pass"
    assert compare_compilation_cache(cold, {**warm, "workload_sha256": "other"})[
        "classification"
    ] == "inconclusive"
