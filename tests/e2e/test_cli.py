import json
import subprocess
import sys


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
    reproduced = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "reproduce", "--case", str(reduced)],
        capture_output=True,
        text=True,
    )
    assert reproduced.returncode == 0, reproduced.stdout


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
    assert evidence["counts"]["pass"] == 4
