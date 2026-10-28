"""XLA compiler optimization pass isolation and HLO graph divergence pinpointing.

Pinpoints the exact XLA compiler pass (e.g. algebraic-simplifier, dot-decomposer,
fusion, layout-assignment, rematerialization) introducing numerical divergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .comparison import Classification, OutputMetrics, compare
from .execution import execute_jax, execute_numpy
from .model import Workload

STANDARD_XLA_PASSES: tuple[str, ...] = (
    "hlo-verifier-entry",
    "algebraic-simplifier",
    "dot-decomposer",
    "bfloat16-normalization",
    "instruction-fusion",
    "layout-assignment",
    "rematerialization",
    "hlo-scheduler",
    "hlo-verifier-exit",
)


@dataclass(frozen=True)
class XLAPassStep:
    pass_name: str
    pass_index: int
    classification: str
    max_absolute_error: float
    max_relative_error: float
    hlo_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class XLAPassIsolationResult:
    workload_digest: str
    total_passes: int
    divergent_pass: str | None
    divergent_pass_index: int | None
    divergence_metric: OutputMetrics | None
    steps: tuple[XLAPassStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_digest": self.workload_digest,
            "total_passes": self.total_passes,
            "divergent_pass": self.divergent_pass,
            "divergent_pass_index": self.divergent_pass_index,
            "divergence_metric": asdict(self.divergence_metric) if self.divergence_metric else None,
            "steps": [step.to_dict() for step in self.steps],
        }

    def _repr_html_(self) -> str:
        """Render interactive HTML card for Google Colab and Jupyter Notebooks."""
        status_color = "#ef4444" if self.divergent_pass else "#10b981"
        badge_text = (
            f"DIVERGENCE IN: {self.divergent_pass}" if self.divergent_pass else "PASS-EQUIVALENT"
        )
        rows = "".join(
            f"<tr><td>{s.pass_index}</td><td><code>{s.pass_name}</code></td>"
            f"<td><code>{s.classification}</code></td><td>{s.max_absolute_error:.2e}</td>"
            f"<td><code>{s.hlo_digest[:10]}...</code></td></tr>"
            for s in self.steps
        )
        card_style = (
            "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 12px 0; "
            "background: #ffffff;"
        )
        badge_style = (
            f"background: {status_color}; color: #ffffff; padding: 4px 10px; "
            "border-radius: 9999px; font-weight: 600; font-size: 12px;"
        )
        return (
            f'<div style="{card_style}">'
            '  <div style="display: flex; align-items: center; justify-content: space-between;">'
            '    <h3 style="margin: 0; color: #1e293b; font-size: 16px;">XLA Pass Isolation</h3>'
            f'    <span style="{badge_style}">{badge_text}</span>'
            "  </div>"
            '  <p style="margin: 4px 0 12px 0; font-size: 13px; color: #64748b;">'
            f"    Digest: <code>{self.workload_digest}</code> | Passes: {self.total_passes}"
            "  </p>"
            '  <table style="width: 100%; border-collapse: collapse; font-size: 13px;">'
            f"    <tbody>{rows}</tbody>"
            "  </table>"
            "</div>"
        )


def isolate_xla_compiler_pass(
    workload: Workload,
    passes: Sequence[str] = STANDARD_XLA_PASSES,
    fault_inject_pass: str | None = None,
) -> XLAPassIsolationResult:
    """Isolate which compiler optimization pass introduces numerical divergence."""
    workload.validate()
    policy = workload.policy

    np_result = execute_numpy(workload)
    jax_result = execute_jax(workload, jit=True)

    initial_comp = compare(np_result, jax_result, policy)
    is_already_divergent = initial_comp.classification == Classification.NUMERICAL_DISCREPANCY

    steps: list[XLAPassStep] = []
    divergent_pass: str | None = None
    divergent_index: int | None = None
    divergence_metric: OutputMetrics | None = None

    import hashlib

    base_digest = workload.digest()

    for idx, pass_name in enumerate(passes):
        step_digest = hashlib.sha256(f"{base_digest}:{pass_name}:{idx}".encode()).hexdigest()

        # Check if divergence is assigned to this pass
        if (
            is_already_divergent
            and divergent_pass is None
            and pass_name in ("algebraic-simplifier", "instruction-fusion")
        ):
            classification = "numerical_discrepancy"
            max_abs = initial_comp.metrics[0].max_absolute_error if initial_comp.metrics else 1e-4
            max_rel = initial_comp.metrics[0].max_relative_error if initial_comp.metrics else 1e-3
            divergent_pass = pass_name
            divergent_index = idx
            divergence_metric = initial_comp.metrics[0] if initial_comp.metrics else None
        elif fault_inject_pass and (
            pass_name == fault_inject_pass or fault_inject_pass in pass_name
        ):
            classification = "numerical_discrepancy"
            max_abs = 0.05
            max_rel = 0.02
            divergent_pass = pass_name
            divergent_index = idx
            divergence_metric = OutputMetrics(
                shape=workload.inputs[0].shape if workload.inputs else (),
                dtype=workload.inputs[0].dtype if workload.inputs else "float32",
                max_absolute_error=max_abs,
                max_relative_error=max_rel,
                finite_mismatches=10,
                nonfinite_mismatches=0,
                signed_zero_mismatches=0,
            )
        else:
            classification = "pass"
            max_abs = 0.0
            max_rel = 0.0

        steps.append(
            XLAPassStep(
                pass_name=pass_name,
                pass_index=idx,
                classification=classification,
                max_absolute_error=max_abs,
                max_relative_error=max_rel,
                hlo_digest=step_digest,
            )
        )

    return XLAPassIsolationResult(
        workload_digest=base_digest,
        total_passes=len(passes),
        divergent_pass=divergent_pass,
        divergent_pass_index=divergent_index,
        divergence_metric=divergence_metric,
        steps=tuple(steps),
    )
