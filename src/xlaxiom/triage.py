"""Automated upstream issue formatter and evidence packager.

Generates pre-formatted, audit-ready GitHub Issue markdown packages tailored
for Google JAX, PyTorch, and OpenXLA maintainer review templates.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from .pipeline import run_comparison
from .storage import load_case


def _generate_standalone_script(case_data: dict[str, Any]) -> str:
    """Generate a zero-dependency standalone Python repro script from a case dictionary."""
    workload = case_data.get("workload", {})
    seed = workload.get("seed", 42)
    inputs = workload.get("inputs", [])
    nodes = workload.get("nodes", [])
    outputs = workload.get("outputs", [])

    lines = [
        "# --- Standalone Reproduction Script (No XLAxiom required) ---",
        "import numpy as np",
        "import jax",
        "import jax.numpy as jnp",
        "",
        f"rng = np.random.default_rng({seed})",
    ]

    for inp in inputs:
        name = inp["name"]
        shape = inp["shape"]
        dtype = inp.get("dtype", "float32")
        lines.append(f"{name}_np = rng.normal(size={shape}).astype(np.{dtype})")
        lines.append(f"{name}_jax = jnp.asarray({name}_np)")

    lines.append("")
    lines.append("@jax.jit")
    lines.append("def compiled_fn(*args):")

    input_names = [f"{inp['name']}_jax" for inp in inputs]
    lines.append(f"    {', '.join(inp['name'] for inp in inputs)} = args")

    for node in nodes:
        op = node["op"]
        out = node["output"]
        args = ", ".join(node["inputs"])
        if op == "add":
            lines.append(f"    {out} = jnp.add({args})")
        elif op == "sub":
            lines.append(f"    {out} = jnp.subtract({args})")
        elif op == "mul":
            lines.append(f"    {out} = jnp.multiply({args})")
        elif op == "div":
            lines.append(f"    {out} = jnp.divide({args})")
        elif op == "neg":
            lines.append(f"    {out} = jnp.negative({args})")
        elif op in {"sin", "cos", "exp", "log", "sqrt", "square", "abs", "tanh"}:
            lines.append(f"    {out} = getattr(jnp, '{op}')({args})")
        elif op in {"maximum", "minimum"}:
            lines.append(f"    {out} = getattr(jnp, '{op}')({args})")
        elif op == "matmul":
            lines.append(f"    {out} = jnp.matmul({args})")
        elif op in {"sum", "mean", "prod", "max", "min"}:
            lines.append(f"    {out} = getattr(jnp, '{op}')({args})")
        else:
            lines.append(f"    {out} = {args}")

    lines.append(f"    return {', '.join(outputs)}")
    lines.append("")
    lines.append(f"res_jax = compiled_fn({', '.join(input_names)})")
    lines.append("print('Execution succeeded; evaluate numerical parity against reference.')")

    return "\n".join(lines)


def generate_minimal_reproduction_script(case_or_workload: Any) -> str:
    """Generate a zero-dependency standalone Python repro script from a case or workload."""
    if isinstance(case_or_workload, dict):
        data = case_or_workload
    elif hasattr(case_or_workload, "payload"):
        data = case_or_workload.payload()
    elif hasattr(case_or_workload, "canonical_dict"):
        data = {"workload": case_or_workload.canonical_dict()}
    else:
        data = {"workload": dict(case_or_workload)}
    return _generate_standalone_script(data)


def format_triage_issue(
    case_path: str | Path,
    target_tracker: str = "jax",
) -> str:
    """Format a complete GitHub Issue markdown report for a target compiler repository."""
    path = Path(case_path)
    if not path.is_file():
        raise FileNotFoundError(f"case file does not exist: {path}")

    case = load_case(path)
    case_dict = json.loads(path.read_text(encoding="utf-8"))

    workload = case.workload
    comp_result = run_comparison(case.workload, case.comparison_mode, case.fault)
    classification = comp_result.classification.value
    first_m = comp_result.metrics[0] if comp_result.metrics else None
    max_abs = first_m.max_absolute_error if first_m else "N/A"
    max_rel = first_m.max_relative_error if first_m else "N/A"
    mean_abs = getattr(first_m, "mean_absolute_error", "N/A") if first_m else "N/A"
    rms = getattr(first_m, "rms_error", "N/A") if first_m else "N/A"
    provenance = case_dict.get("provenance", {})

    target_lower = target_tracker.lower()
    if target_lower in ("tpu", "openxla-tpu"):
        target_name = "Google Cloud TPU / OpenXLA"
    elif target_lower == "jax":
        target_name = "Google JAX"
    else:
        target_name = target_tracker.upper()

    unique_ops = sorted({node.op for node in workload.nodes})
    ops_summary = ", ".join(unique_ops) or "workload"

    standalone_code = _generate_standalone_script(case_dict)
    if target_lower in ("tpu", "openxla-tpu"):
        # Prepend TPU platform hint
        standalone_code = (
            "# Request TPU backend explicitly:\n"
            "# export JAX_PLATFORMS=tpu\n"
            + standalone_code.replace("@jax.jit", "@jax.jit(backend='tpu')")
        )

    jax_version = "N/A"
    try:
        import jax

        jax_version = jax.__version__
    except Exception:
        pass
    if isinstance(provenance, dict):
        jax_version = str(provenance.get("jax_version", jax_version))
        device_str = str(provenance.get("device", "tpu" if "tpu" in target_lower else "cpu"))
    else:
        device_str = "tpu" if "tpu" in target_lower else "cpu"

    from .stablehlo import export_stablehlo

    stablehlo_text = export_stablehlo(workload)

    lines = [
        f"### Description: Numerical discrepancy observed in `{ops_summary}` under {target_name}",
        "",
        "#### Reproduction Verdict & Classification",
        f"- **Classification**: `{classification}`",
        f"- **Workload Digest**: `{workload.digest()}`",
        f"- **Node Count**: `{len(workload.nodes)}` (failure-preservingly minimized)",
        f"- **Input Count**: `{len(workload.inputs)}`",
        "",
        "#### Numerical Error Metrics",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Maximum Absolute Error | `{max_abs}` |",
        f"| Maximum Relative Error | `{max_rel}` |",
        f"| Mean Absolute Error | `{mean_abs}` |",
        f"| Root Mean Square (RMS) Error | `{rms}` |",
        f"| Relative Tolerance Threshold | `{workload.policy.rtol}` |",
        f"| Absolute Tolerance Threshold | `{workload.policy.atol}` |",
        "",
        "#### Environment Provenance",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| Operating System | `{platform.system()} {platform.release()}` |",
        f"| Architecture | `{platform.machine()}` |",
        f"| Python Version | `{platform.python_version()}` |",
        f"| JAX Version | `{jax_version}` |",
        f"| Backend / Device | `{device_str}` |",
        "",
        "#### Standalone Reproduction Script",
        "```python",
        standalone_code,
        "```",
        "",
        "#### StableHLO MLIR Dialect Representation",
        "```mlir",
        stablehlo_text,
        "```",
        "",
        "---",
        "*Automated evidence package generated by XLAxiom forensic compiler suite.*",
    ]

    return "\n".join(lines)

