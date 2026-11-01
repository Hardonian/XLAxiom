"""Continuous Integration regression gate, PR commenting bot, and summary generator.

Formats GitHub Actions step summaries, sticky PR comments, and CI/CD quality gate
verdicts with actionable diagnostics, SNR tables, and failure prevention.

All markers are plain ASCII so the output can be printed on legacy Windows consoles
(cp1252) without raising ``UnicodeEncodeError``.
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


def _is_regression(d: NumericalDiff, min_snr_db: float | None) -> bool:
    """Return True when a tensor comparison must fail the gate.

    A tensor regresses when it is outside tolerance, or when a finite SNR falls below the
    optional ``min_snr_db`` floor. Using one predicate for both the verdict and the report
    guarantees the PR comment can never disagree with the exit status.
    """
    if not d.is_close:
        return True
    if min_snr_db is None or math.isinf(d.snr_db) or math.isnan(d.snr_db):
        return False
    return d.snr_db < min_snr_db


def _md_cell(text: str) -> str:
    """Escape text for safe use inside a single-line Markdown table cell."""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'").replace("\n", " ")


def _format_snr(snr_db: float) -> str:
    return "inf" if math.isinf(snr_db) else f"{snr_db:.1f}"


def generate_pr_comment(
    diff_or_diffs: NumericalDiff | list[tuple[str, NumericalDiff]],
    *,
    commit_sha: str | None = None,
    baseline_ref: str = "main",
    pr_number: int | None = None,
    min_snr_db: float | None = None,
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
    min_snr_db : float, optional
        Minimum acceptable finite SNR. Tensors below it are reported as regressions even if
        they are inside ``rtol``/``atol``. ``None`` (default) applies tolerance checks only.

    Returns
    -------
    str
        Formatted Markdown comment suitable for posting via GitHub REST API.
    """
    if isinstance(diff_or_diffs, NumericalDiff):
        items = [("output", diff_or_diffs)]
    else:
        items = list(diff_or_diffs)

    total_tensors = len(items)
    failed_items = [item for item in items if _is_regression(item[1], min_snr_db)]
    passed_count = total_tensors - len(failed_items)
    all_passed = not failed_items

    status_text = "PASSED" if all_passed else "REGRESSION DETECTED"
    badge_status = status_text.replace(" ", "%20")
    badge_color = "brightgreen" if all_passed else "red"
    badge_md = (
        f"![XLAxiom Parity](https://img.shields.io/badge/Numerical%20Parity-"
        f"{badge_status}-{badge_color}?style=flat-square)"
    )

    commit_str = f" for commit `{commit_sha[:7]}`" if commit_sha else ""
    pr_str = f" on PR #{pr_number}" if pr_number is not None else ""

    lines = [
        f"## {badge_md}",
        "",
        f"**XLAxiom Numerical Regression Gate Report**{commit_str}{pr_str}",
        f"Status: **{status_text}**",
        f"Comparing candidate against baseline (`{_md_cell(baseline_ref)}`).",
        "",
        "| Metric | Status |",
        "| :--- | :--- |",
        f"| **Overall Gate Verdict** | **{'[PASS]' if all_passed else '[REGRESSION]'}** |",
        f"| **Tensors Evaluated** | `{total_tensors}` |",
        f"| **Passed Tensors** | `{passed_count} / {total_tensors}` |",
    ]

    if failed_items:
        failed_diffs = [d for _, d in failed_items]
        finite_snrs = [d.snr_db for d in failed_diffs if not math.isnan(d.snr_db)]
        lowest_snr = min(finite_snrs) if finite_snrs else float("nan")
        worst_abs = max(d.max_absolute_error for d in failed_diffs)
        worst_rel = max(d.max_relative_error for d in failed_diffs)
        snr_display = "n/a" if math.isnan(lowest_snr) else f"{_format_snr(lowest_snr)} dB"
        lines.append(f"| **Regressed Tensors** | `[! {len(failed_items)}]` |")
        lines.append(f"| **Lowest SNR** | `{snr_display}` |")
        lines.append(f"| **Worst-case Absolute Drift** | `{worst_abs:.3e}` |")
        lines.append(f"| **Worst-case Relative Drift** | `{worst_rel:.3e}` |")

    lines.append("")
    lines.append("### Detailed Tensor Parity Breakdown")
    lines.append("")
    lines.append(
        "| Tensor / Layer | Verdict | Max Abs | Max Rel | SNR (dB) | Cosine Sim | Diagnosis |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for name, d in items:
        status_icon = "[FAIL]" if _is_regression(d, min_snr_db) else "[OK]"
        lines.append(
            f"| `{_md_cell(name)}` | {status_icon} | `{d.max_absolute_error:.2e}` | "
            f"`{d.max_relative_error:.2e}` | `{_format_snr(d.snr_db)}` | "
            f"`{d.cosine_similarity:.5f}` | {_md_cell(d.diagnosis)} |"
        )

    if failed_items:
        lines.append("")
        lines.append(
            "<details><summary><strong>Reproduction Code for Regressions</strong></summary>"
        )
        lines.append("")
        for name, d in failed_items[:3]:
            lines.append(f"#### Regressed Tensor: `{_md_cell(name)}`")
            lines.append("```python")
            lines.append(d.repro_snippet.strip())
            lines.append("```")
        if len(failed_items) > 3:
            lines.append(f"_{len(failed_items) - 3} additional regressed tensor(s) not shown._")
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
    pr_number: int | None = None,
) -> CIRegressionGateResult:
    """Evaluate a complete CI regression gate against tolerance thresholds.

    Raises
    ------
    ValueError
        If ``diffs`` is empty. A gate that checked nothing must not report success.
    """
    if not diffs:
        raise ValueError("evaluate_ci_gate requires at least one (name, NumericalDiff) pair.")

    worst_snr = float("inf")
    max_abs = 0.0
    max_rel = 0.0
    regressed_count = 0

    for _, d in diffs:
        if _is_regression(d, min_snr_db):
            regressed_count += 1
        if d.snr_db < worst_snr:
            worst_snr = d.snr_db
        max_abs = max(max_abs, d.max_absolute_error)
        max_rel = max(max_rel, d.max_relative_error)

    comment = generate_pr_comment(
        diffs,
        commit_sha=commit_sha,
        baseline_ref=baseline_ref,
        pr_number=pr_number,
        min_snr_db=min_snr_db,
    )

    return CIRegressionGateResult(
        passed=regressed_count == 0,
        summary_markdown=comment,
        pr_comment_markdown=comment,
        worst_snr_db=worst_snr,
        max_absolute_error=max_abs,
        max_relative_error=max_rel,
        regressed_tensors=regressed_count,
        total_tensors=len(diffs),
    )
