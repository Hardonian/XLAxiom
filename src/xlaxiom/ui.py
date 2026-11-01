"""Interactive tensor diff visualizer and offline HTML dashboard generator.

Generates standalone, responsive single-file dashboards with an SVG error-percentile
chart, summary statistic cards, and a copyable forensic reproduction script. The output
has no external assets or scripts, so it renders identically offline.
"""

from __future__ import annotations

import html
import math

from .assertions import NumericalDiff


def _bar_height(value: float, max_val: float) -> int:
    """Scale an error value to a 4-80px bar height, tolerating NaN/inf inputs."""
    if math.isnan(value) or math.isnan(max_val):
        return 4
    if math.isinf(value):
        return 80
    if math.isinf(max_val) or max_val <= 0.0:
        return 4
    return max(4, int((value / max_val) * 80))


def generate_visualizer_html(
    diff_obj: NumericalDiff,
    *,
    title: str = "XLAxiom Tensor Diff Visualizer",
) -> str:
    """Generate a self-contained, responsive HTML dashboard for visualizing numerical discrepancies.

    Parameters
    ----------
    diff_obj : NumericalDiff
        The comparison diff object to visualize.
    title : str, optional
        Custom title for the dashboard header.

    Returns
    -------
    str
        Complete HTML string with embedded SVG visualizations and CSS.
    """
    status_text = "PASSED" if diff_obj.is_close else "DISCREPANCY DETECTED"
    badge_color = "#10b981" if diff_obj.is_close else "#ef4444"
    snr_str = f"{diff_obj.snr_db:.1f} dB" if not math.isinf(diff_obj.snr_db) else "inf (exact)"

    # Build an SVG error-percentile chart
    svg_bars = []
    percentiles = [
        ("p50", diff_obj.p50_absolute_error),
        ("p90", diff_obj.p90_absolute_error),
        ("p99", diff_obj.p99_absolute_error),
        ("max", diff_obj.max_absolute_error),
    ]
    max_val = diff_obj.max_absolute_error

    for i, (label, val) in enumerate(percentiles):
        height = _bar_height(val, max_val)
        y_pos = 90 - height
        x_pos = 40 + i * 80
        svg_bars.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="40" height="{height}" rx="3" fill="#3b82f6" />'
            f'<text x="{x_pos + 20}" y="105" font-size="10" '
            f'text-anchor="middle" fill="#64748b">{label}</text>'
            f'<text x="{x_pos + 20}" y="{y_pos - 4}" font-size="9" '
            f'text-anchor="middle" fill="#0f172a">{val:.1e}</text>'
        )

    histogram_svg = f"""
    <svg width="100%" style="max-width: 360px; background: #f8fafc; border-radius: 6px;
         padding: 10px;" viewBox="0 0 360 120" role="img"
         aria-label="Absolute error percentiles (p50, p90, p99, max)">
      <line x1="20" y1="90" x2="340" y2="90" stroke="#cbd5e1" stroke-width="1" />
      {"".join(svg_bars)}
    </svg>
    """

    escaped_repro = html.escape(diff_obj.repro_snippet)
    escaped_title = html.escape(title)
    shape_str = html.escape(str(diff_obj.shape_a))
    dtype_str = html.escape(str(diff_obj.dtype_a))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 24px; background: #f1f5f9; color: #0f172a;
    }}
    .container {{
      max-width: 900px; margin: 0 auto; background: #ffffff; border-radius: 12px;
      padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .header {{
      display: flex; justify-content: space-between; align-items: center;
      border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 20px;
    }}
    .badge {{
      background: {badge_color}; color: #ffffff; padding: 6px 14px; border-radius: 9999px;
      font-weight: 700; font-size: 13px;
    }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }}
    .card {{
      background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px;
    }}
    .card-title {{ font-size: 12px; color: #64748b; margin-bottom: 4px; }}
    .card-value {{ font-size: 18px; font-weight: 700; color: #0f172a; }}
    pre {{
      background: #0f172a; color: #f8fafc; padding: 14px; border-radius: 8px; overflow-x: auto;
      font-size: 13px; line-height: 1.4;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 style="margin: 0; font-size: 20px;">{html.escape(title)}</h1>
        <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
          Shape: {shape_str} | Dtype: {dtype_str}
        </div>
      </div>
      <span class="badge">{status_text}</span>
    </div>

    <div class="card" style="margin-bottom: 20px;">
      <div class="card-title">Root-Cause Diagnosis</div>
      <div style="font-size: 14px; color: #1e293b; font-weight: 600;">
        {html.escape(diff_obj.diagnosis)}
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Max Absolute Error</div>
        <div class="card-value">{diff_obj.max_absolute_error:.3e}</div>
      </div>
      <div class="card">
        <div class="card-title">Max Relative Error</div>
        <div class="card-value">{diff_obj.max_relative_error:.3e}</div>
      </div>
      <div class="card">
        <div class="card-title">Signal-to-Noise Ratio (SNR)</div>
        <div class="card-value">{snr_str}</div>
      </div>
      <div class="card">
        <div class="card-title">Cosine Similarity</div>
        <div class="card-value">{diff_obj.cosine_similarity:.6f}</div>
      </div>
      <div class="card">
        <div class="card-title">Mismatched Elements</div>
        <div class="card-value">
          {diff_obj.mismatched_elements} / {diff_obj.total_elements}
          ({diff_obj.mismatch_percentage:.1f}%)
        </div>
      </div>
      <div class="card">
        <div class="card-title">RMS Error</div>
        <div class="card-value">{diff_obj.rms_error:.3e}</div>
      </div>
    </div>

    <h3>Absolute Error Percentiles</h3>
    {histogram_svg}

    <h3 style="margin-top: 24px;">Standalone Minimal Reproduction Script</h3>
    <pre><code>{escaped_repro}</code></pre>
  </div>
</body>
</html>
"""
