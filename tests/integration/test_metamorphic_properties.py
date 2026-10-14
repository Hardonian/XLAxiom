from xlaxiom.metamorphic import curated_properties, run_curated_properties


def test_curated_properties_pass_in_jax_jit_mode() -> None:
    identifiers = tuple(
        property_definition.identifier
        for property_definition in curated_properties()
        if not property_definition.requires_x64
    )
    evidence = run_curated_properties(seed=12, backend="jax-jit", identifiers=identifiers)

    assert evidence["counts"]["pass"] == len(identifiers)
    assert all(
        record["comparison"]["classification"] == "pass" for record in evidence["properties"]
    )
