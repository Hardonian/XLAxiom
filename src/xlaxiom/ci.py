"""Continuous Integration regression gate, PR commenting bot, and summary generator.

Formats GitHub Actions step summaries, sticky PR comments, and CI/CD quality gate
verdicts with actionable diagnostics, SNR tables, and failure prevention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .assertions import NumericalDiff


@dataclass(frozen=True)
class CIRegressionGateResult:
    """Verdict of a CI numerical regression gate."""

    passed: bool
    summary_markdown: str
    pr_comment_markdown: str
    worst_snr_db: float
    max_absolute_error: float
    max_relative_error: float
    regressed_tensors: int
    total_tensors: int


def generate_pr_comment(
    diff_or_diffs: NumericalDiff | list[tuple[str, NumericalDiff]],
    *,
    commit_sha: str | None = None,
    baseline_ref: str = "main",
    pr_number: int | None = None,
) -> str:
    """Format a rich GitHub Flavored Markdown comment for PR automation bots.

    Parameters
    ----------
    diff_or_diffs : NumericalDiff or list of (name, NumericalDiff)
        Comparison results to include in the PR summary.
    commit_sha : str, optional
        Git commit hash under test.
    baseline_ref : str, optional
        Reference baseline branch or tag (default: "main").
    pr_number : int, optional
        GitHub Pull Request number.

    Returns
    -------
    str
        Formatted Markdown comment suitable for posting via GitHub REST API.
    """
    if isinstance(diff_or_diffs, NumericalDiff):
        items = [("output", diff_or_diffs)]
    else:
        items = diff_or_diffs

    total_tensors = len(items)
    passed_items = [item for item in items if item[1].is_close]
    failed_items = [item for item in items if not item[1].is_close]
    all_passed = len(failed_items) == 0

    badge_status = "PASSED" if all_passed else "REGRESSION DETECTED"
    badge_color = "brightgreen" if all_passed else "red"
    badge_md = (
        f"![XLAxiom Parity](https://img.shields.io/badge/Numerical%20Parity-"
        f"{badge_status}-{badge_color}?style=flat-square)"
    )

    commit_str = f" for commit `{commit_sha[:7]}`" if commit_sha else ""
    pr_str = f" on PR #{pr_number}" if pr_number else ""

    lines = [
        f"## {badge_md}",
        "",
        f"**XLAxiom Numerical Regression Gate Report**{commit_str}{pr_str}",
        f"Comparing candidate against baseline (`{baseline_ref}`).",
        "",
        "| Metric | Status |",
        "| :--- | :--- |",
        f"| **Overall Gate Verdict** | **{'[PASS]' if all_passed else '[REGRESSION]'}** |",
        f"| **Tensors Evaluated** | `{total_tensors}` |",
        f"| **Passed Tensors** | `{len(passed_items)} / {total_tensors}` |",
    ]

    if failed_items:
        lines.append(f"| **Regressed Tensors** | `[! {len(failed_items)}]` |")
        worst_diff = min(
            (d for _, d in failed_items),
            key=lambda d: d.snr_db if not math.isinf(d.snr_db) else 1e9,
        )
        snr_display = f"{worst_diff.snr_db:.1f} dB" if not math.isinf(worst_diff.snr_db) else "inf"
        lines.append(f"| **Lowest SNR** | `{snr_display}` |")
        lines.append(f"| **Worst-case Absolute Drift** | `{worst_diff.max_absolute_error:.3e}` |")
        lines.append(f"| **Worst-case Relative Drift** | `{worst_diff.max_relative_error:.3e}` |")

    lines.append("")
    lines.append("### Detailed Tensor Parity Breakdown")
    lines.append("")
    lines.append(
        "| Tensor / Layer | Verdict | Max Abs | Max Rel | SNR (dB) | Cosine Sim | Diagnosis |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for name, d in items:
        status_icon = "[OK]" if d.is_close else "[FAIL]"
        snr_str = f"{d.snr_db:.1f}" if not math.isinf(d.snr_db) else "inf"
        lines.append(
            f"| `{name}` | {status_icon} | `{d.max_absolute_error:.2e}` | "
            f"`{d.max_relative_error:.2e}` | `{snr_str}` | `{d.cosine_similarity:.5f}` | "
            f"{d.diagnosis} |"
        )

    if failed_items:
        lines.append("")
        lines.append(
            "<details><summary><strong>Reproduction Code for Regressions</strong></summary>"
        )
        lines.append("")
        for name, d in failed_items[:3]:
            lines.append(f"#### Regressed Tensor: `{name}`")
            lines.append("```python")
            lines.append(d.repro_snippet.strip())
            lines.append("```")
        lines.append("</details>")

    lines.append("")
    lines.append("---")
    lines.append(
        "*Automated numerical verification powered by [XLAxiom](https://github.com/Hardonian/XLAxiom).*"
    )
    return "\n".join(lines)


def evaluate_ci_gate(
    diffs: list[tuple[str, NumericalDiff]],
    *,
    min_snr_db: float = 20.0,
    commit_sha: str | None = None,
    baseline_ref: str = "main",
) -> CIRegressionGateResult:
    """Evaluate a complete CI regression gate against tolerance thresholds."""
    all_passed = True
    worst_snr = float("inf")
    max_abs = 0.0
    max_rel = 0.0
    regressed_count = 0

    for _, d in diffs:
        if not d.is_close or (not math.isinf(d.snr_db) and d.snr_db < min_snr_db):
            all_passed = False
            regressed_count += 1
        if d.snr_db < worst_snr:
            worst_snr = d.snr_db
        if d.max_absolute_error > max_abs:
            max_abs = d.max_absolute_error
        if d.max_relative_error > max_rel:
            max_rel = d.max_relative_error

    comment = generate_pr_comment(diffs, commit_sha=commit_sha, baseline_ref=baseline_ref)

    return CIRegressionGateResult(
        passed=all_passed,
        summary_markdown=comment,
        pr_comment_markdown=comment,
        worst_snr_db=worst_snr,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        regressed_tensors=regressed_count,
        total_tensors=len(diffs),
    )
