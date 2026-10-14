import json
import subprocess
import sys
from importlib.metadata import version

from xlaxiom import __version__


def test_package_and_cli_versions_match_distribution_metadata() -> None:
    assert __version__ == version("xlaxiom")

    for invocation in (
        [sys.executable, "-m", "xlaxiom", "--version"],
        [sys.executable, "-m", "xlaxiom.cli", "--version"],
    ):
        result = subprocess.run(invocation, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"xlaxiom {__version__}"


def test_module_invocation_forms_are_equivalent(tmp_path) -> None:
    """``python -m xlaxiom.cli`` must not silently succeed without doing anything.

    Without a ``__main__`` guard the module imports and exits 0 with no output, which
    looks like success and produces no evidence.
    """
    package_form = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "doctor"],
        capture_output=True,
        text=True,
        check=False,
    )
    cli_form = subprocess.run(
        [sys.executable, "-m", "xlaxiom.cli", "doctor"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert package_form.returncode == 0, package_form.stderr
    assert cli_form.returncode == 0, cli_form.stderr
    assert package_form.stdout.strip(), "python -m xlaxiom produced no output"
    assert cli_form.stdout.strip(), "python -m xlaxiom.cli produced no output"
    assert cli_form.stdout == package_form.stdout


def test_fresh_process_reproduction(tmp_path) -> None:
    original = tmp_path / "original.json"
    reduced = tmp_path / "reduced.json"
    reproduction_evidence = tmp_path / "reproduction.json"
    environment_evidence = tmp_path / "environment.json"
    generate = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "generate",
            "--synthetic",
            "--seed",
            "4",
            "--output",
            str(original),
        ],
        capture_output=True,
        text=True,
    )
    assert generate.returncode == 0, generate.stderr
    minimized = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "minimize",
            "--case",
            str(original),
            "--output",
            str(reduced),
        ],
        capture_output=True,
        text=True,
    )
    assert minimized.returncode == 0, minimized.stderr
    captured_environment = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "environment",
            "capture",
            "--label",
            "cli-test-host",
            "--output",
            str(environment_evidence),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert captured_environment.returncode == 0, captured_environment.stderr
    reproduced = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "reproduce",
            "--case",
            str(reduced),
            "--output",
            str(reproduction_evidence),
            "--environment-evidence",
            str(environment_evidence),
        ],
        capture_output=True,
        text=True,
    )
    assert reproduced.returncode == 0, reproduced.stdout
    record = json.loads(reproduction_evidence.read_text(encoding="utf-8"))
    environment = json.loads(environment_evidence.read_text(encoding="utf-8"))
    assert record["reproduction_evidence_version"] == "1.0"
    assert record["verdict"] == "reproduced"
    assert record["environment_sha256"] == environment["sha256"]


def test_strict_readiness_fails_closed_for_incomplete_evidence(tmp_path) -> None:
    case = tmp_path / "synthetic.json"
    generated = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "generate",
            "--synthetic",
            "--output",
            str(case),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr

    readiness = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "readiness", "--case", str(case), "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert readiness.returncode == 1
    assert json.loads(readiness.stdout)["ready"] is False


def test_invalid_bundle_has_actionable_cli_error(tmp_path) -> None:
    missing_bundle = tmp_path / "missing.zip"
    result = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "bundle", "validate", "--bundle", str(missing_bundle)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid bundle: cannot read bundle" in result.stderr


def test_property_corpus_cli_writes_replayable_evidence(tmp_path) -> None:
    output = tmp_path / "properties.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "property",
            "run",
            "--seed",
            "12",
            "--backend",
            "numpy",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["kind"] == "metamorphic-corpus-evidence"
    assert evidence["counts"]["pass"] == 19


def test_corpus_build_loads_packaged_historical_evidence_outside_the_checkout(tmp_path) -> None:
    """The installed CLI must not depend on a source-checkout-relative corpus path."""
    output = tmp_path / "findings.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "corpus",
            "build",
            "--backend",
            "numpy",
            "--cases",
            "0",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    historical = next(
        entry for entry in document["entries"] if entry["category"] == "known-historical"
    )
    assert historical["environment"]["recorded_from"] == (
        "package:xlaxiom/corpus/historical/jax-40634.json"
    )


def test_property_campaign_and_coverage_cli(tmp_path) -> None:
    campaign = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "property",
            "campaign",
            "--cases",
            "2",
            "--backend",
            "numpy",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    coverage_path = tmp_path / "coverage.json"
    coverage = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "property",
            "coverage",
            "--output",
            str(coverage_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert campaign.returncode == 0, campaign.stderr
    assert json.loads(campaign.stdout)["counts"]["pass"] == 38
    assert coverage.returncode == 0, coverage.stderr
    assert json.loads(coverage_path.read_text(encoding="utf-8"))["property_count"] == 19


def test_property_evidence_compare_cli(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    comparison_path = tmp_path / "comparison.json"
    output_path = tmp_path / "comparison-result.json"
    for path in (baseline_path, comparison_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "xlaxiom",
                "property",
                "run",
                "--backend",
                "numpy",
                "--seed",
                "9",
                "--output",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    compared = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "property",
            "compare",
            "--baseline",
            str(baseline_path),
            "--comparison",
            str(comparison_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert compared.returncode == 0, compared.stderr
    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence["summary"]["property_count"] == 19
    assert evidence["summary"]["equivalent"] is True


def test_relational_artifact_cli_round_trip(tmp_path) -> None:
    relation = tmp_path / "relation.json"
    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "relation",
            "create",
            "--property",
            "add-commutativity-v1",
            "--backend",
            "numpy",
            "--output",
            str(relation),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    executed = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "relation", "run", "--case", str(relation)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert created.returncode == 0, created.stderr
    assert executed.returncode == 0, executed.stderr
    assert json.loads(executed.stdout)["classification"] == "pass"


def test_cli_benchmark_and_cache_workflow(tmp_path) -> None:
    case_path = tmp_path / "case.json"
    bench1_path = tmp_path / "bench1.json"
    bench2_path = tmp_path / "bench2.json"

    gen = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "generate", "--seed", "1", "--output", str(case_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert gen.returncode == 0, gen.stderr

    b1 = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "benchmark",
            "--case",
            str(case_path),
            "--measurements",
            "3",
            "--output",
            str(bench1_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert b1.returncode == 0, b1.stderr
    data1 = json.loads(bench1_path.read_text(encoding="utf-8"))
    assert "median_ms" in data1
    assert "workload_sha256" in data1

    b2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "benchmark",
            "--case",
            str(case_path),
            "--measurements",
            "3",
            "--output",
            str(bench2_path),
            "--baseline",
            str(bench1_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert b2.returncode == 0, b2.stderr
    out2 = json.loads(b2.stdout)
    assert "baseline_assessment" in out2

    cache_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "cache",
            "--cold",
            str(bench1_path),
            "--warm",
            str(bench2_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cache_res.returncode == 0, cache_res.stderr
    cache_data = json.loads(cache_res.stdout)
    assert cache_data["comparable"] is True
