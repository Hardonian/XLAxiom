"""Interactive tensor diff visualizer and offline HTML dashboard generator.

Generates standalone, responsive single-file dashboards with an SVG error-percentile
chart, summary statistic cards, and a copyable forensic reproduction script. The output
has no external assets or scripts, so it renders identically offline.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

from .assertions import NumericalDiff

if TYPE_CHECKING:
    from .hf import CheckpointParityReport


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


def generate_checkpoint_html(
    report: CheckpointParityReport,
    *,
    title: str = "XLAxiom Checkpoint Parity Dashboard",
) -> str:
    """Generate a self-contained, responsive HTML dashboard for checkpoint weight comparison.

    Parameters
    ----------
    report : CheckpointParityReport
        The checkpoint parity audit report.
    title : str, optional
        Custom header title for the dashboard.

    Returns
    -------
    str
        Complete HTML string with embedded search, filtering, and CSS.
    """
    total = report.total_weights
    matched = report.matched_weights
    regressed = report.regressed_weights
    match_pct = (matched / total * 100.0) if total > 0 else 100.0
    status_text = "PASSED" if report.is_identical else "WEIGHT DIVERGENCE DETECTED"
    badge_bg = "#10b981" if report.is_identical else "#ef4444"

    min_snr = float("inf")
    worst_cos = 1.0
    for _, d in report.weight_diffs:
        if not math.isnan(d.snr_db) and d.snr_db < min_snr:
            min_snr = d.snr_db
        if not math.isnan(d.cosine_similarity) and d.cosine_similarity < worst_cos:
            worst_cos = d.cosine_similarity

    min_snr_str = f"{min_snr:.1f} dB" if not math.isinf(min_snr) else "∞"
    escaped_title = html.escape(title)

    rows = []
    for name, d in report.weight_diffs:
        status_tag = "MATCH" if d.is_close else "DIVERGED"
        tag_color = "#10b981" if d.is_close else "#ef4444"
        snr_disp = f"{d.snr_db:.1f} dB" if not math.isinf(d.snr_db) else "∞"
        escaped_name = html.escape(name)
        escaped_diag = html.escape(d.diagnosis)
        status_attr = status_tag.lower()
        name_attr = escaped_name.lower()
        row_class = f'class="layer-row" data-status="{status_attr}" data-name="{name_attr}"'
        name_cell = (
            f'<td style="font-family: monospace; font-size: 12px; font-weight: 600;">'
            f"{escaped_name}</td>"
        )
        badge_cell = (
            f'<td><span style="background: {tag_color}; color: #fff; padding: 2px 6px; '
            f'border-radius: 4px; font-size: 11px; font-weight: 700;">{status_tag}</span></td>'
        )
        rows.append(
            f"<tr {row_class}>\n"
            f"  {name_cell}\n"
            f"  {badge_cell}\n"
            f"  <td>{d.shape_a}</td>\n"
            f"  <td>{d.dtype_a}</td>\n"
            f"  <td>{d.max_absolute_error:.2e}</td>\n"
            f"  <td>{d.max_relative_error:.2e}</td>\n"
            f"  <td>{snr_disp}</td>\n"
            f"  <td>{d.cosine_similarity:.5f}</td>\n"
            f'  <td style="font-size: 12px; color: #475569;">{escaped_diag}</td>\n'
            f"</tr>"
        )
    rows_html = "\n".join(rows)

    missing_html = ""
    if report.missing_keys:
        missing_count = len(report.missing_keys)
        missing_list = "".join(f"<li>{html.escape(k)}</li>" for k in report.missing_keys[:10])
        missing_html = f"""
        <div class="card" style="margin-top: 16px; border-left: 4px solid #f59e0b;">
          <div class="card-title" style="color: #b45309;">
            Missing Weights in Candidate ({missing_count})
          </div>
          <ul style="font-size: 12px; font-family: monospace; margin: 8px 0 0 20px;">
            {missing_list}
          </ul>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 24px; background: #f8fafc; color: #0f172a;
    }}
    .container {{
      max-width: 1200px; margin: 0 auto; background: #ffffff; border-radius: 12px;
      padding: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }}
    .header {{
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #e2e8f0;
    }}
    .badge {{
      background: {badge_bg}; color: white; padding: 6px 14px;
      border-radius: 9999px; font-weight: 700; font-size: 13px;
    }}
    .grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }}
    .card {{
      background: #f1f5f9; border-radius: 8px; padding: 16px; border: 1px solid #e2e8f0;
    }}
    .card-title {{ font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; }}
    .card-value {{ font-size: 22px; font-weight: 700; margin-top: 6px; color: #0f172a; }}
    .controls {{
      display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap;
    }}
    .search-box {{
      flex: 1; min-width: 260px; padding: 8px 14px; font-size: 14px; border: 1px solid #cbd5e1;
      border-radius: 6px; outline: none;
    }}
    .filter-btn {{
      padding: 8px 16px; font-size: 13px; font-weight: 600; border-radius: 6px;
      border: 1px solid #cbd5e1; background: #fff; cursor: pointer;
    }}
    .filter-btn.active {{
      background: #3b82f6; color: #fff; border-color: #3b82f6;
    }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;
    }}
    th {{
      background: #f8fafc; border-bottom: 2px solid #cbd5e1; padding: 10px 8px;
      font-weight: 600; color: #475569; position: sticky; top: 0;
    }}
    td {{
      border-bottom: 1px solid #f1f5f9; padding: 8px; color: #1e293b;
    }}
    tr:hover td {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 style="margin: 0; font-size: 22px;">{escaped_title}</h1>
        <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
          Offline Forensic Parity Inspection
        </div>
      </div>
      <span class="badge">{status_text}</span>
    </div>

    <div class="grid">
      <div class="card">
        <div class="card-title">Total Weights</div>
        <div class="card-value">{total}</div>
      </div>
      <div class="card">
        <div class="card-title">Parity Rate</div>
        <div class="card-value">{match_pct:.1f}%</div>
      </div>
      <div class="card">
        <div class="card-title">Regressed Weights</div>
        <div class="card-value" style="color: {"#ef4444" if regressed > 0 else "#10b981"};">{regressed}</div>
      </div>
      <div class="card">
        <div class="card-title">Minimum SNR</div>
        <div class="card-value">{min_snr_str}</div>
      </div>
      <div class="card">
        <div class="card-title">Worst Cosine Sim</div>
        <div class="card-value">{worst_cos:.5f}</div>
      </div>
    </div>

    {missing_html}

    <div class="controls" style="margin-top: 24px;">
      <input type="text" id="searchBox" class="search-box" placeholder="Filter layers by name (e.g. attn, mlp, down_proj)..." oninput="applyFilters()" />
      <button class="filter-btn active" id="btnAll" onclick="setFilter('all')">All Layers ({total})</button>
      <button class="filter-btn" id="btnDiverged" onclick="setFilter('diverged')">Diverged Only ({regressed})</button>
      <button class="filter-btn" id="btnMatched" onclick="setFilter('matched')">Matched ({matched})</button>
    </div>

    <div style="overflow-x: auto; max-height: 600px; border: 1px solid #e2e8f0; border-radius: 8px;">
      <table>
        <thead>
          <tr>
            <th>Layer Name</th>
            <th>Status</th>
            <th>Shape</th>
            <th>Dtype</th>
            <th>Max Abs Diff</th>
            <th>Max Rel Diff</th>
            <th>SNR</th>
            <th>Cosine Sim</th>
            <th>Diagnosis</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          {rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <script>
    let currentFilter = 'all';

    function setFilter(filter) {{
      currentFilter = filter;
      document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
      if (filter === 'all') document.getElementById('btnAll').classList.add('active');
      if (filter === 'diverged') document.getElementById('btnDiverged').classList.add('active');
      if (filter === 'matched') document.getElementById('btnMatched').classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      const query = document.getElementById('searchBox').value.toLowerCase();
      const rows = document.querySelectorAll('.layer-row');
      rows.forEach(row => {{
        const name = row.getAttribute('data-name');
        const status = row.getAttribute('data-status');
        const matchesQuery = name.includes(query);
        const matchesFilter = (currentFilter === 'all') || (status === currentFilter);
        if (matchesQuery && matchesFilter) {{
          row.style.display = '';
        }} else {{
          row.style.display = 'none';
        }}
      }});
    }}
  </script>
</body>
</html>
"""
