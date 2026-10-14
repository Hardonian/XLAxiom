"""Tests for the curated corpus admission bar and metamorphic properties.

The admission bar is the point: a finding is admitted only after independent
reproduction, and a checksum mismatch or a missing reproduction count must fail
validation rather than being waved through.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from xlaxiom.comparison import Classification
from xlaxiom.corpus import (
    CATEGORIES,
    REQUIRED_REPRODUCTIONS,
    CorpusEntry,
    build_entry,
    check_metamorphic_property,
    digest,
    load_corpus,
    minimize_finding_for_admission,
    reproduce_isolated,
    run_metamorphic_suite,
    save_corpus,
    validate_entry,
)
from xlaxiom.generation import GenerationConfig, generate


def make_entry(**overrides: Any) -> dict[str, Any]:
    workload = generate(GenerationConfig(seed=17))
    payload: dict[str, Any] = {
        "identifier": "finding-test",
        "category": "reproduced-finding",
        "title": "Reproduced numerical_discrepancy",
        "summary": "outputs differ under the configured policy",
        "classification": Classification.NUMERICAL_DISCREPANCY.value,
        "workload": workload.canonical_dict(),
        "workload_sha256": workload.digest(),
        "reproductions": REQUIRED_REPRODUCTIONS,
    }
    payload.update(overrides)
    return build_entry(CorpusEntry(**payload))


class TestAdmissionBar:
    """A finding must carry the evidence that justifies admitting it."""

    def test_entry_with_reproduction_is_admitted(self) -> None:
        entry = make_entry()
        assert entry["sha256"]
        validate_entry(entry)

    def test_entry_without_reproduction_is_rejected(self) -> None:
        entry = make_entry(reproductions=1)
        with pytest.raises(ValueError, match="independent reproduction"):
            validate_entry(entry)

    def test_entry_without_workload_is_rejected(self) -> None:
        entry = make_entry(workload=None)
        with pytest.raises(ValueError, match="reduced workload"):
            validate_entry(entry)

    def test_passing_entry_is_not_a_finding(self) -> None:
        entry = make_entry(classification=Classification.PASS.value)
        with pytest.raises(ValueError, match="non-pass classification"):
            validate_entry(entry)

    def test_tampered_entry_fails_its_checksum(self) -> None:
        entry = make_entry()
        entry["summary"] = "quietly changed after admission"
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_entry(entry)

    def test_malformed_reduced_workload_is_rejected_even_with_a_recomputed_entry_checksum(
        self,
    ) -> None:
        entry = make_entry()
        entry["workload"] = {"not": "a workload"}
        entry["sha256"] = digest(entry)

        with pytest.raises(ValueError, match="invalid reduced workload"):
            validate_entry(entry)

    def test_reduced_workload_digest_is_rebuilt_not_trusted(self) -> None:
        entry = make_entry(workload_sha256="0" * 64)
        entry["sha256"] = digest(entry)

        with pytest.raises(ValueError, match="workload checksum mismatch"):
            validate_entry(entry)

    def test_unknown_category_is_rejected(self) -> None:
        entry = make_entry(category="invented")
        with pytest.raises(ValueError, match="unsupported corpus category"):
            validate_entry(entry)

    def test_metamorphic_property_needs_its_check_record(self) -> None:
        entry = make_entry(category="metamorphic-property", property_check=None)
        with pytest.raises(ValueError, match="check record"):
            validate_entry(entry)

    def test_known_historical_is_a_first_class_category(self) -> None:
        assert "known-historical" in CATEGORIES


class TestCorpusRoundTrip:
    """A corpus document survives a write and a validated read."""

    def test_round_trip_preserves_entries(self, tmp_path) -> None:
        entries = [make_entry(identifier="finding-a"), make_entry(identifier="finding-b")]
        path = save_corpus(tmp_path / "corpus.json", entries)
        document = load_corpus(path)
        assert document["entry_count"] == 2
        assert document["categories"]["reproduced-finding"] == 2

    def test_tampered_file_fails_validation(self, tmp_path) -> None:
        entries = [make_entry()]
        path = save_corpus(tmp_path / "corpus.json", entries)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["entries"][0]["summary"] = "edited in place"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_corpus(path)

    def test_document_count_and_categories_are_validated(self, tmp_path) -> None:
        path = save_corpus(tmp_path / "corpus.json", [make_entry()])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["entry_count"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="entry_count does not match"):
            load_corpus(path)

        payload["entry_count"] = 1
        payload["categories"]["reproduced-finding"] = 0
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="category counts do not match"):
            load_corpus(path)

    def test_digest_ignores_its_own_field(self) -> None:
        entry = make_entry()
        assert digest(entry) == digest({**entry, "sha256": "anything"})


class TestMetamorphicSuite:
    """Properties are checked three ways and never assumed to hold."""

    def test_every_property_runs_to_a_conclusion(self) -> None:
        suite = run_metamorphic_suite()
        assert suite["property_count"] >= 4
        for result in suite["results"]:
            assert result["state"] in {"holds", "violated", "INCONCLUSIVE", "BLOCKED"}

    def test_properties_are_checked_against_an_independent_oracle(self) -> None:
        suite = run_metamorphic_suite()
        checked = [item for item in suite["results"] if item["state"] in {"holds", "violated"}]
        assert checked, "no property produced a comparable result"
        for item in checked:
            assert "oracle" in item
            assert "eager" in item
            assert "jit" in item

    def test_unknown_operation_is_reported_not_guessed(self) -> None:
        from xlaxiom.corpus import METAMORPHIC_PROPERTIES, MetamorphicProperty

        base = METAMORPHIC_PROPERTIES[0]
        bogus = MetamorphicProperty(
            identifier="bogus",
            description="an operation that does not exist",
            operation="no-such-operation",
            lhs_seed=1,
            rhs_seed=1,
        )
        record = check_metamorphic_property(bogus)
        assert record["state"] == "INCONCLUSIVE"
        assert "no-such-operation" in record["error"]
        assert base.identifier

    def test_violation_is_classified_not_swallowed(self) -> None:
        suite = run_metamorphic_suite()
        for item in suite["results"]:
            if item["state"] == "violated":
                assert item["classification"] == "metamorphic_violation"


class TestIsolatedReproduction:
    """Reproduction claims are only as strong as the processes behind them."""

    def test_passing_workload_reproduces_consistently(self) -> None:
        workload = generate(GenerationConfig(seed=2))
        record = reproduce_isolated(workload, attempts=2, timeout_s=90.0)
        assert record["attempts"] == 2
        assert len(record["observations"]) == 2
        if record["reproduced"]:
            assert record["classification"] is not None
            assert len({item["classification"] for item in record["observations"]}) == 1

    def test_every_attempt_runs_in_a_fresh_process(self) -> None:
        workload = generate(GenerationConfig(seed=2))
        record = reproduce_isolated(workload, attempts=2, timeout_s=90.0)
        assert "fresh interpreter" in record["note"]


class TestFindingMinimization:
    """Corpus admission minimizes the same classification before reproduction."""

    def test_non_pass_observation_is_minimized_with_recorded_stats(self) -> None:
        workload = generate(GenerationConfig(seed=8, max_nodes=6, max_dim=6))
        result = minimize_finding_for_admission(
            workload,
            Classification.NUMERICAL_DISCREPANCY,
            evaluator=lambda _: Classification.NUMERICAL_DISCREPANCY,
        )

        assert result.stats.preserved_classification == "numerical_discrepancy"
        assert result.stats.reduced_nodes <= result.stats.original_nodes
        assert result.stats.reduced_elements <= result.stats.original_elements

    def test_passing_observation_cannot_be_admitted_as_a_finding(self) -> None:
        workload = generate(GenerationConfig(seed=1))
        with pytest.raises(ValueError, match="passing observation"):
            minimize_finding_for_admission(workload, Classification.PASS)
