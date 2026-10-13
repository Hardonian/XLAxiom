import numpy as np
import pytest

from xlaxiom.metamorphic import (
    MetamorphicPropertyError,
    build_property,
    curated_properties,
    run_curated_properties,
    run_property,
)
from xlaxiom.model import materialize_inputs


def test_curated_properties_exactly_match_the_reviewed_registry() -> None:
    assert [property_definition.identifier for property_definition in curated_properties()] == [
        "add-commutativity-v1",
        "transpose-involution-v1",
        "reshape-roundtrip-v1",
        "matmul-transpose-v1",
        "negation-involution-v1",
        "subtraction-as-addition-v1",
        "absolute-idempotence-v1",
        "integer-add-commutativity-v1",
    ]


@pytest.mark.parametrize(
    "identifier", [property_definition.identifier for property_definition in curated_properties()]
)
def test_property_pairs_share_deterministic_inputs_and_are_valid(identifier: str) -> None:
    pair = build_property(identifier, seed=83)

    assert pair.left.metadata["side"] == "left"
    assert pair.right.metadata["side"] == "right"
    left_inputs = materialize_inputs(pair.left)
    right_inputs = materialize_inputs(pair.right)
    assert left_inputs.keys() == right_inputs.keys()
    for name in left_inputs:
        assert np.array_equal(left_inputs[name], right_inputs[name])


def test_numpy_property_runner_records_passing_corpus_evidence() -> None:
    evidence = run_curated_properties(seed=12, backend="numpy")

    assert evidence["kind"] == "metamorphic-corpus-evidence"
    assert evidence["counts"]["pass"] == 8
    assert all(
        record["comparison"]["classification"] == "pass" for record in evidence["properties"]
    )


def test_unknown_property_and_invalid_seed_are_rejected() -> None:
    with pytest.raises(MetamorphicPropertyError, match="unknown curated property"):
        run_property("unreviewed", backend="numpy")
    with pytest.raises(MetamorphicPropertyError, match="seed must"):
        build_property("add-commutativity-v1", seed=-1)


def test_property_campaign_and_coverage_are_bounded_and_explicit() -> None:
    from xlaxiom.metamorphic import property_coverage, run_property_campaign

    campaign = run_property_campaign(cases=2, seed=20, backend="numpy")
    coverage = property_coverage(seed=20)

    assert campaign["counts"]["pass"] == 16
    assert campaign["candidate_observations"] == []
    assert coverage["property_count"] == 8
    assert {"add", "sub", "neg", "abs", "matmul", "reshape", "transpose"} <= set(
        coverage["operations"]
    )
    assert coverage["dtypes"] == ["float32", "int32"]
