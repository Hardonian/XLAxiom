import copy

import numpy as np
import pytest

from xlaxiom.metamorphic import (
    MetamorphicEvidenceError,
    MetamorphicPropertyError,
    build_property,
    compare_property_evidence,
    curated_properties,
    run_curated_properties,
    run_property,
    validate_property_evidence,
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
        "float64-add-commutativity-v1",
        "mixed-dtype-add-commutativity-v1",
        "signed-zero-negation-involution-v1",
        "nonfinite-absolute-idempotence-v1",
        "extreme-finite-absolute-idempotence-v1",
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
        assert np.array_equal(left_inputs[name], right_inputs[name], equal_nan=True)


def test_numpy_property_runner_records_passing_corpus_evidence() -> None:
    evidence = run_curated_properties(seed=12, backend="numpy")

    assert evidence["kind"] == "metamorphic-corpus-evidence"
    assert evidence["counts"]["pass"] == 13
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

    assert campaign["counts"]["pass"] == 26
    assert campaign["candidate_observations"] == []
    assert coverage["property_count"] == 13
    assert {"add", "sub", "neg", "abs", "matmul", "reshape", "transpose"} <= set(
        coverage["operations"]
    )
    assert coverage["dtypes"] == ["float32", "float64", "int32"]


def test_cross_backend_comparison_requires_identical_workload_identity() -> None:
    baseline = run_curated_properties(seed=12, backend="numpy")
    comparison = copy.deepcopy(baseline)
    comparison["backend"] = "jax-jit"
    for record in comparison["properties"]:
        record["backend"] = "jax-jit"

    result = compare_property_evidence(baseline, comparison)

    assert result["baseline"]["backend"] == "numpy"
    assert result["comparison"]["backend"] == "jax-jit"
    assert result["summary"]["property_count"] == 13
    assert result["summary"]["equivalent"] is True

    comparison["properties"][0]["left_workload_sha256"] = "0" * 64
    with pytest.raises(MetamorphicEvidenceError, match="workload identity mismatch"):
        compare_property_evidence(baseline, comparison)


def test_property_evidence_validation_rejects_drifted_counts() -> None:
    evidence = run_curated_properties(seed=4, backend="numpy")
    evidence["counts"]["pass"] -= 1

    with pytest.raises(MetamorphicEvidenceError, match="counts do not match"):
        validate_property_evidence(evidence)


def test_property_evidence_validation_rebuilds_curated_workload_identity() -> None:
    evidence = run_curated_properties(seed=4, backend="numpy")
    evidence["properties"][0]["left_workload_sha256"] = "a" * 64

    with pytest.raises(MetamorphicEvidenceError, match="workload identity mismatch"):
        validate_property_evidence(evidence)


def test_property_evidence_validation_rejects_invented_property() -> None:
    evidence = run_curated_properties(seed=4, backend="numpy")
    evidence["requested_properties"][0] = "invented-property-v1"
    evidence["properties"][0]["property"]["identifier"] = "invented-property-v1"

    with pytest.raises(MetamorphicEvidenceError, match="unknown curated propert"):
        validate_property_evidence(evidence)


def test_tpu_property_backend_never_falls_back_to_cpu() -> None:
    record = run_property("add-commutativity-v1", seed=3, backend="jax-tpu-jit")

    for side in ("left", "right"):
        observation = record["comparison"][side]
        if observation["status"] == "ok":
            assert observation["device"].startswith("tpu")
        else:
            assert observation["status"] == "unsupported_backend"
            assert "CPU fallback was not performed" in observation["error"]
