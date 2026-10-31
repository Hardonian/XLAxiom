"""Neural network layer-by-layer parity auditing and divergence localization.

Traces intermediate layer activations across PyTorch modules, Flax Linen/NNX models,
or compiled pipelines to pinpoint the exact layer where numerical drift originates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .assertions import _to_numpy, diff


@dataclass(frozen=True)
class LayerMetric:
    """Parity metrics for a single intermediate neural network layer."""

    layer_name: str
    shape: tuple[int, ...]
    dtype: str
    is_close: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    snr_db: float
    cosine_similarity: float
    diagnosis: str


@dataclass(frozen=True)
class LayerParityReport:
    """Full layer-by-layer parity report across two neural network architectures."""

    all_passed: bool
    divergence_origin: str | None
    layers: tuple[LayerMetric, ...]

    def format_table(self) -> str:
        """Format an ASCII / console comparison table."""
        status = "PASSED" if self.all_passed else "DIVERGENCE DETECTED"
        header = (
            f"=== Layer-by-Layer Parity Audit: {status} ===\n"
            f"{'Layer Name':<30} {'Status':<8} {'Max Abs':<11} {'Max Rel':<11} "
            f"{'SNR (dB)':<10} {'Cosine Sim':<10}"
        )
        separator = "-" * 85
        lines = [header, separator]

        for m in self.layers:
            st = "PASS" if m.is_close else "FAIL"
            snr_val = f"{m.snr_db:.1f}" if not math.isinf(m.snr_db) else "inf"
            flag = " <-- Divergence Origin" if m.layer_name == self.divergence_origin else ""
            line = (
                f"{m.layer_name:<30} {st:<8} {m.max_absolute_error:<11.2e} "
                f"{m.max_relative_error:<11.2e} {snr_val:<10} {m.cosine_similarity:<10.5f}{flag}"
            )
            lines.append(line)

        if self.divergence_origin:
            lines.append(separator)
            lines.append(f"Root divergence origin: '{self.divergence_origin}'")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Render interactive HTML table for Google Colab and Jupyter Notebooks."""
        badge_color = "#10b981" if self.all_passed else "#ef4444"
        verdict = "PASSED" if self.all_passed else "DRIFT DETECTED"

        rows = []
        for m in self.layers:
            pass_badge = (
                '<span style="background: #10b981; color: white; padding: 2px 8px; '
                'border-radius: 9999px; font-size: 11px;">PASS</span>'
            )
            fail_badge = (
                '<span style="background: #ef4444; color: white; padding: 2px 8px; '
                'border-radius: 9999px; font-size: 11px;">FAIL</span>'
            )
            status_badge = pass_badge if m.is_close else fail_badge
            is_origin = m.layer_name == self.divergence_origin
            row_bg = "background: #fef2f2;" if is_origin else ""
            snr_str = f"{m.snr_db:.1f} dB" if not math.isinf(m.snr_db) else "inf"
            origin_tag = ' <strong style="color: #b91c1c;">(Origin)</strong>' if is_origin else ""

            rows.append(
                f'<tr style="{row_bg}">'
                f'<td style="padding: 6px 8px; font-family: monospace; font-size: 12px;">'
                f"{m.layer_name}{origin_tag}</td>"
                f'<td style="padding: 6px 8px;">{status_badge}</td>'
                f'<td style="padding: 6px 8px;">{m.max_absolute_error:.2e}</td>'
                f'<td style="padding: 6px 8px;">{m.max_relative_error:.2e}</td>'
                f'<td style="padding: 6px 8px;">{snr_str}</td>'
                f'<td style="padding: 6px 8px;">{m.cosine_similarity:.5f}</td>'
                f"</tr>"
            )

        card_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 10px 0; "
            "background: #ffffff;"
        )
        badge_style = (
            f"background: {badge_color}; color: #ffffff; padding: 4px 10px; "
            "border-radius: 9999px; font-weight: 700; font-size: 12px;"
        )

        html = f"""
        <div style="{card_style}">
          <div style="display: flex; align-items: center; justify-content: space-between;
                      margin-bottom: 12px;">
            <div style="font-weight: 700; color: #0f172a; font-size: 16px;">
              XLAxiom Layer Parity Audit
            </div>
            <span style="{badge_style}">{verdict}</span>
          </div>
          <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
            <thead>
              <tr style="border-bottom: 2px solid #cbd5e1; color: #475569;">
                <th style="padding: 6px 8px;">Layer</th>
                <th style="padding: 6px 8px;">Status</th>
                <th style="padding: 6px 8px;">Max Abs Error</th>
                <th style="padding: 6px 8px;">Max Rel Error</th>
                <th style="padding: 6px 8px;">SNR</th>
                <th style="padding: 6px 8px;">Cosine Sim</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """
        return html


def _hook_pytorch_module(
    module: Any,
) -> tuple[dict[str, Any], list[Any]]:
    """Register forward hooks on a PyTorch module to capture intermediate layer outputs."""
    activations: dict[str, Any] = {}
    handles = []

    def make_hook(name: str) -> Any:
        def hook(mod: Any, inp: Any, out: Any) -> None:
            # Handle tuple output (e.g. attention returning (output, weights))
            val = out[0] if isinstance(out, tuple) else out
            activations[name] = _to_numpy(val)

        return hook

    for name, child in module.named_modules():
        if name:  # Skip the top-level container itself
            h = child.register_forward_hook(make_hook(name))
            handles.append(h)

    return activations, handles


def audit_layer_parity(
    model_a: Any,
    model_b: Any,
    *sample_inputs: Any,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> LayerParityReport:
    """Audit layer-by-layer activation parity between two models on sample inputs.

    Parameters
    ----------
    model_a : Callable or PyTorch Module or dict of activations
        First model under test (reference).
    model_b : Callable or PyTorch Module or dict of activations
        Second model under test (candidate / compiled / ported).
    sample_inputs : Any
        Inputs passed to both models.
    rtol : float, optional
        Relative tolerance (default: 1e-4).
    atol : float, optional
        Absolute tolerance (default: 1e-5).

    Returns
    -------
    LayerParityReport
        Comprehensive audit detailing metrics per layer and locating the origin of divergence.
    """
    acts_a: dict[str, Any] = {}
    acts_b: dict[str, Any] = {}
    handles_a: list[Any] = []
    handles_b: list[Any] = []

    try:
        # Case 1: Inputs are already dictionaries of activations
        if isinstance(model_a, dict) and isinstance(model_b, dict):
            acts_a = model_a
            acts_b = model_b

        # Case 2: PyTorch nn.Module with named_modules
        elif hasattr(model_a, "named_modules") and hasattr(model_b, "named_modules"):
            acts_a, handles_a = _hook_pytorch_module(model_a)
            acts_b, handles_b = _hook_pytorch_module(model_b)
            # Run models to populate activations
            model_a(*sample_inputs)
            model_b(*sample_inputs)

        # Case 3: Callables returning dictionaries
        elif callable(model_a) and callable(model_b):
            out_a = model_a(*sample_inputs)
            out_b = model_b(*sample_inputs)
            if isinstance(out_a, dict) and isinstance(out_b, dict):
                acts_a = out_a
                acts_b = out_b
            else:
                # Top-level single output
                acts_a = {"output": _to_numpy(out_a)}
                acts_b = {"output": _to_numpy(out_b)}

    finally:
        # Clean up PyTorch forward hooks
        for h in handles_a:
            h.remove()
        for h in handles_b:
            h.remove()

    layer_metrics: list[LayerMetric] = []
    divergence_origin: str | None = None

    for name in acts_a:
        if name not in acts_b:
            continue

        arr_a = _to_numpy(acts_a[name])
        arr_b = _to_numpy(acts_b[name])

        d = diff(arr_a, arr_b, rtol=rtol, atol=atol)

        metric = LayerMetric(
            layer_name=name,
            shape=d.shape_a,
            dtype=d.dtype_a,
            is_close=d.is_close,
            max_absolute_error=d.max_absolute_error,
            max_relative_error=d.max_relative_error,
            mean_absolute_error=d.mean_absolute_error,
            snr_db=d.snr_db,
            cosine_similarity=d.cosine_similarity,
            diagnosis=d.diagnosis,
        )
        layer_metrics.append(metric)

        if not d.is_close and divergence_origin is None:
            divergence_origin = name

    all_passed = all(m.is_close for m in layer_metrics)

    return LayerParityReport(
        all_passed=all_passed,
        divergence_origin=divergence_origin,
        layers=tuple(layer_metrics),
    )
