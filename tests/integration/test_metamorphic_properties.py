from xlaxiom.metamorphic import run_curated_properties


def test_curated_properties_pass_in_jax_jit_mode() -> None:
    evidence = run_curated_properties(seed=12, backend="jax-jit")

    assert evidence["counts"]["pass"] == 8
    assert all(
        record["comparison"]["classification"] == "pass" for record in evidence["properties"]
    )
