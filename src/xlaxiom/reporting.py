"""Evidence-first JSON and Markdown reporting, including a non-filing issue draft."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import ComparisonResult
from .storage import CaseArtifact, observation_record, save_json


def build_report(artifact: CaseArtifact, comparison: ComparisonResult) -> dict[str, Any]:
    record = observation_record(artifact, comparison)
    return {
        "report_version": "1.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "observed_result": record,
        "interpretation": (
            (
                "This is a synthetic control; it verifies the XLAxiom pipeline and is not evidence "
                "of a compiler defect."
            )
            if artifact.fault
            else (
                "The observation is differential evidence only; it requires "
                "independent reproduction before root-cause claims."
            )
        ),
        "confidence": (
            "single local execution; repeat on an independently configured "
            "environment before escalation"
        ),
        "limitations": [
            "NumPy is not an independent oracle for every JAX semantic or precision mode.",
            "No accelerator conclusion is implied by a CPU result.",
        ],
        "upstream_candidate": bool(
            comparison.classification.value == "numerical_discrepancy" and not artifact.fault
        ),
        "issue_draft": issue_draft(artifact, comparison),
    }


def issue_draft(artifact: CaseArtifact, comparison: ComparisonResult) -> dict[str, Any]:
    return {
        "title": (
            f"Differential observation: {comparison.classification.value} "
            f"for {artifact.workload.identifier}"
        ),
        "expected_behavior": "The compared executions agree under the recorded policy.",
        "actual_behavior": comparison.explanation,
        "reproduction": "xlaxiom reproduce --case <case.json>",
        "limitations": "Not suitable for filing while this case is synthetic."
        if artifact.fault
        else "Requires fresh-process and independent-environment reproduction.",
        "do_not_file_automatically": True,
    }


def write_report(
    directory: str | Path, artifact: CaseArtifact, comparison: ComparisonResult
) -> tuple[Path, Path]:
    destination = Path(directory)
    report = build_report(artifact, comparison)
    json_path = save_json(destination / "report.json", report)
    observed = report["observed_result"]["comparison"]
    markdown = "\n".join(
        [
            "# XLAxiom evidence report",
            "",
            "## Observed result",
            "",
            f"- Case: `{artifact.workload.identifier}`",
            f"- Classification: `{observed['classification']}`",
            f"- Policy: `{observed['policy']['identifier']}`",
            f"- Provenance: `{artifact.provenance}`",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Confidence and limitations",
            "",
            f"{report['confidence']}",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Reproduction",
            "",
            "```console",
            "xlaxiom reproduce --case case.json",
            "```",
            "",
            "No upstream issue is filed by XLAxiom.",
            "",
        ]
    )
    markdown_path = destination / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path
