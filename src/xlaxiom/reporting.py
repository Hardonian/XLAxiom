"""Evidence-first JSON and Markdown reporting, including a non-filing issue draft."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .comparison import ComparisonResult
from .storage import CaseArtifact, atomic_write_text, observation_record, save_json


def build_report(
    artifact: CaseArtifact,
    comparison: ComparisonResult,
    case_path: str | Path | None = None,
) -> dict[str, Any]:
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
        "issue_draft": issue_draft(artifact, comparison, case_path),
    }


def issue_draft(
    artifact: CaseArtifact,
    comparison: ComparisonResult,
    case_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a non-filing handoff draft tied to the actual case when known."""
    case_reference = str(case_path) if case_path is not None else "<case.json>"
    return {
        "title": (
            f"Differential observation: {comparison.classification.value} "
            f"for {artifact.workload.identifier}"
        ),
        "expected_behavior": "The compared executions agree under the recorded policy.",
        "actual_behavior": comparison.explanation,
        "case_sha256": artifact.to_dict()["sha256"],
        "workload_sha256": artifact.workload.digest(),
        "reproduction": f"xlaxiom reproduce --case {case_reference}",
        "limitations": "Not suitable for filing while this case is synthetic."
        if artifact.fault
        else "Requires fresh-process and independent-environment reproduction.",
        "do_not_file_automatically": True,
    }


def write_report(
    directory: str | Path,
    artifact: CaseArtifact,
    comparison: ComparisonResult,
    case_path: str | Path | None = None,
) -> tuple[Path, Path]:
    destination = Path(directory)
    report = build_report(artifact, comparison, case_path)
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
            report["issue_draft"]["reproduction"],
            "```",
            "",
            "No upstream issue is filed by XLAxiom.",
            "",
        ]
    )
    markdown_path = destination / "report.md"
    atomic_write_text(markdown_path, markdown)
    html_path = destination / "report.html"
    atomic_write_text(html_path, build_html_report(report))
    return json_path, markdown_path


def build_html_report(report: dict[str, Any]) -> str:
    """Generate a self-contained, interactive single-file HTML report with SVG diagnostics."""
    import html

    observed = report.get("observed_result", {})
    comp = observed.get("comparison", {})
    classification = str(comp.get("classification", "unknown"))
    raw_metrics = comp.get("metrics", [])
    first_metric: dict[str, Any] = (
        raw_metrics[0]
        if isinstance(raw_metrics, (list, tuple)) and len(raw_metrics) > 0 and isinstance(raw_metrics[0], dict)
        else (raw_metrics if isinstance(raw_metrics, dict) else {})
    )
    policy = comp.get("policy", {})
    env = report.get("environment", {})
    issue = report.get("issue_draft", {})
    repro = issue.get("reproduction", "")
    limitations = report.get("limitations", [])

    is_pass = classification in {"pass", "equivalent"}
    is_warn = classification == "numerical_discrepancy"
    badge_color = "#10b981" if is_pass else ("#f59e0b" if is_warn else "#ef4444")
    badge_bg = "#ecfdf5" if is_pass else ("#fffbeb" if is_warn else "#fef2f2")
    badge_border = "#a7f3d0" if is_pass else ("#fde68a" if is_warn else "#fecaca")
    badge_label = "PASSED / EQUIVALENT" if is_pass else classification.upper().replace("_", " ")

    max_diff = first_metric.get("max_absolute_difference", 0.0)
    mae = first_metric.get("mean_absolute_error", 0.0)
    rmse = first_metric.get("rms_error", 0.0)
    atol = policy.get("atol", 1e-5)
    rtol = policy.get("rtol", 1e-4)

    diff_val = float(max_diff) if max_diff is not None else 0.0
    tol_val = float(atol) if atol is not None else 1e-5
    ratio = min(max(diff_val / (tol_val * 2.0) if tol_val > 0 else 0.0, 0.0), 1.0)
    bar_width = int(ratio * 360)
    bar_color = "#10b981" if diff_val <= tol_val else "#ef4444"

    lim_items = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>XLAxiom Evidence Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }}
    .container {{
      max-width: 860px;
      margin: 0 auto;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .title-group h1 {{
      margin: 0;
      font-size: 1.75rem;
      font-weight: 700;
      letter-spacing: -0.025em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.875rem;
      margin-top: 0.25rem;
    }}
    .badge {{
      display: inline-block;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      border: 1px solid {badge_border};
      background: {badge_bg};
      color: {badge_color};
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1.1rem;
      font-weight: 600;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }}
    .metric-item {{
      background: rgba(15, 23, 42, 0.6);
      padding: 0.75rem 1rem;
      border-radius: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .metric-label {{
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 1.25rem;
      font-weight: 600;
      margin-top: 0.25rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    .gauge-container {{
      margin-top: 1rem;
      text-align: center;
    }}
    .gauge-svg {{
      width: 100%;
      max-width: 440px;
      height: auto;
    }}
    pre.code-box {{
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      padding: 1rem;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.875rem;
      color: #38bdf8;
    }}
    .disclaimer {{
      border-left: 4px solid #f59e0b;
      background: rgba(245, 158, 11, 0.1);
      padding: 0.75rem 1rem;
      border-radius: 0 0.5rem 0.5rem 0;
      font-size: 0.85rem;
      color: #fbbf24;
    }}
    ul.limitations {{
      margin: 0;
      padding-left: 1.25rem;
      color: var(--text-muted);
      font-size: 0.875rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title-group">
        <h1>XLAxiom Evidence Report</h1>
        <div class="subtitle">Generated {html.escape(report.get("timestamp_utc", ""))} &bull; Engine v{html.escape(report.get("report_version", "1.0"))}</div>
      </div>
      <div class="badge">{html.escape(badge_label)}</div>
    </div>

    <div class="card">
      <h2>Numerical Metrics</h2>
      <div class="grid-2">
        <div class="metric-item">
          <div class="metric-label">Max Absolute Difference</div>
          <div class="metric-value">{diff_val:.4e}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Mean Absolute Error (MAE)</div>
          <div class="metric-value">{float(mae) if mae is not None else 0.0:.4e}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">RMS Error</div>
          <div class="metric-value">{float(rmse) if rmse is not None else 0.0:.4e}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Policy (atol / rtol)</div>
          <div class="metric-value">{tol_val:.1e} / {float(rtol):.1e}</div>
        </div>
      </div>
      <div class="gauge-container">
        <svg viewBox="0 0 400 65" class="gauge-svg" xmlns="http://www.w3.org/2000/svg">
          <rect x="20" y="20" width="360" height="16" rx="8" fill="#334155" />
          <rect x="20" y="20" width="{bar_width}" height="16" rx="8" fill="{bar_color}" />
          <line x1="200" y1="12" x2="200" y2="44" stroke="#f8fafc" stroke-width="2" stroke-dasharray="3 3" />
          <text x="20" y="54" font-size="10" fill="#94a3b8">0.0</text>
          <text x="200" y="54" font-size="10" fill="#f8fafc" text-anchor="middle">atol ({tol_val:.1e})</text>
          <text x="380" y="54" font-size="10" fill="#94a3b8" text-anchor="end">2x atol</text>
        </svg>
      </div>
    </div>

    <div class="card">
      <h2>Reproduction Command</h2>
      <pre class="code-box"><code>{html.escape(repro)}</code></pre>
    </div>

    <div class="card">
      <h2>Environment &amp; Provenance</h2>
      <div class="grid-2">
        <div class="metric-item">
          <div class="metric-label">Platform</div>
          <div class="metric-value" style="font-size: 0.95rem;">{html.escape(str(env.get("platform", "unknown")))}</div>
        </div>
        <div class="metric-item">
          <div class="metric-label">Python &amp; NumPy</div>
          <div class="metric-value" style="font-size: 0.95rem;">Py {html.escape(str(env.get("python", "")))}, NP {html.escape(str(env.get("numpy", "")))}</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Governance &amp; Interpretation</h2>
      <p style="margin-top: 0;">{html.escape(str(report.get("interpretation", "")))}</p>
      <div class="disclaimer">
        <strong>Strict non-filing notice:</strong> No upstream issue is filed automatically. {html.escape(str(report.get("confidence", "")))}
      </div>
      <h3 style="font-size: 0.95rem; margin-top: 1rem; color: var(--text-muted);">Limitations</h3>
      <ul class="limitations">
        {lim_items}
      </ul>
    </div>
  </div>
</body>
</html>
"""


def write_html_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    content = build_html_report(report)
    return atomic_write_text(destination, content)
