"""HuggingFace checkpoint parity, Safetensors auditing, and serving engine verification.

Audits weight drift, quantized representation fidelity, and serving engine logit parity
(vLLM, SGLang, TensorRT-LLM vs HuggingFace reference models).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assertions import NumericalDiff, _to_numpy, diff


@dataclass(frozen=True)
class CheckpointParityReport:
    """Parity report comparing two model checkpoints or weight dictionaries."""

    is_identical: bool
    total_weights: int
    matched_weights: int
    regressed_weights: int
    missing_keys: tuple[str, ...]
    extra_keys: tuple[str, ...]
    weight_diffs: tuple[tuple[str, NumericalDiff], ...]

    def format_summary(self) -> str:
        """Format terminal diagnostic report."""
        status = "PASSED" if self.is_identical else "WEIGHT DIVERGENCE DETECTED"
        lines = [
            f"=== Checkpoint Weight Parity: {status} ===",
            f"Total weights evaluated: {self.total_weights}",
            f"Matched weights: {self.matched_weights} / {self.total_weights}",
        ]
        if self.missing_keys:
            lines.append(
                f"Missing keys in candidate ({len(self.missing_keys)}): {self.missing_keys[:5]}"
            )
        if self.extra_keys:
            lines.append(f"Extra keys in candidate ({len(self.extra_keys)}): {self.extra_keys[:5]}")
        if self.regressed_weights > 0:
            lines.append(f"Regressed weights exceeding tolerance: {self.regressed_weights}")
            lines.append("\nTop 5 Divergent Weights:")
            for name, d in self.weight_diffs[:5]:
                if not d.is_close:
                    lines.append(
                        f"  - {name}: max_abs={d.max_absolute_error:.2e}, "
                        f"SNR={d.snr_db:.1f} dB, Cosine={d.cosine_similarity:.5f}"
                    )
        return "\n".join(lines)


def _load_weights(source: dict[str, Any] | str | Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load weight tensors from dictionary, .npz, or .safetensors file."""
    if isinstance(source, dict):
        return {k: _to_numpy(v) for k, v in source.items()}

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    if p.suffix == ".npz":
        data = np.load(p)
        return {k: data[k] for k in data.files}

    if p.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file

            return load_file(str(p))
        except ImportError as err:
            raise ImportError(
                "safetensors package required to read .safetensors files. "
                "Run `pip install safetensors`."
            ) from err

    raise ValueError(f"Unsupported checkpoint format for file '{p}'. Supported: .npz, .safetensors")


def audit_checkpoint_parity(
    weights_a: dict[str, Any] | str | Path,
    weights_b: dict[str, Any] | str | Path,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> CheckpointParityReport:
    """Audit weight tensor parity between two checkpoints.

    Parameters
    ----------
    weights_a : dict, str, or Path
        Reference baseline weights (e.g. unquantized fp32/bf16 safetensors or state dict).
    weights_b : dict, str, or Path
        Candidate weights (e.g. quantized or framework-ported weights).
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    CheckpointParityReport
        Comprehensive weight parity report.
    """
    dict_a = _load_weights(weights_a)
    dict_b = _load_weights(weights_b)

    keys_a = set(dict_a.keys())
    keys_b = set(dict_b.keys())

    common_keys = sorted(keys_a & keys_b)
    missing_keys = tuple(sorted(keys_a - keys_b))
    extra_keys = tuple(sorted(keys_b - keys_a))

    weight_diffs: list[tuple[str, NumericalDiff]] = []
    regressed_count = 0

    for k in common_keys:
        d = diff(dict_b[k], dict_a[k], rtol=rtol, atol=atol)
        weight_diffs.append((k, d))
        if not d.is_close:
            regressed_count += 1

    is_identical = regressed_count == 0 and len(missing_keys) == 0 and len(extra_keys) == 0

    return CheckpointParityReport(
        is_identical=is_identical,
        total_weights=len(common_keys),
        matched_weights=len(common_keys) - regressed_count,
        regressed_weights=regressed_count,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
        weight_diffs=tuple(weight_diffs),
    )


@dataclass(frozen=True)
class ServingParityReport:
    """Parity report comparing serving engine output logits against reference models."""

    is_aligned: bool
    top1_agreement_pct: float
    top5_overlap_pct: float
    max_logit_difference: float
    mean_kl_divergence: float
    prompts_evaluated: int

    def format_summary(self) -> str:
        """Format terminal diagnostic report."""
        status = "PASSED" if self.is_aligned else "LOGIT DIVERGENCE DETECTED"
        return (
            f"=== Serving Engine Parity: {status} ===\n"
            f"Prompts evaluated: {self.prompts_evaluated}\n"
            f"Top-1 Token Agreement: {self.top1_agreement_pct:.2f}%\n"
            f"Top-5 Token Set Overlap: {self.top5_overlap_pct:.2f}%\n"
            f"Max Logit Difference: {self.max_logit_difference:.4e}\n"
            f"Mean KL Divergence: {self.mean_kl_divergence:.6e}"
        )


def _softmax(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def audit_serving_parity(
    engine_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    sample_inputs: Sequence[Any],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    kl_tolerance: float = 1e-4,
) -> ServingParityReport:
    """Audit output logit fidelity between an optimized serving engine and a reference model.

    Parameters
    ----------
    engine_fn : Callable
        Inference engine function (e.g. vLLM, SGLang, or TensorRT-LLM generating logits).
    reference_fn : Callable
        Reference model function (e.g. HuggingFace PyTorch eager).
    sample_inputs : Sequence[Any]
        List of input prompt tensors or token IDs.
    rtol : float, optional
        Relative tolerance on logits.
    atol : float, optional
        Absolute tolerance on logits.
    kl_tolerance : float, optional
        Maximum allowable KL divergence before flagging divergence.

    Returns
    -------
    ServingParityReport
        Report detailing Top-1/Top-5 token agreement and distribution KL divergence.
    """
    total_prompts = len(sample_inputs)
    top1_matches = 0
    top5_overlaps = 0.0
    max_logit_diff = 0.0
    kl_divergences: list[float] = []

    for inp in sample_inputs:
        logits_engine = _to_numpy(engine_fn(inp)).astype(np.float64)
        logits_ref = _to_numpy(reference_fn(inp)).astype(np.float64)

        # Check max difference
        diff_val = float(np.max(np.abs(logits_engine - logits_ref)))
        if diff_val > max_logit_diff:
            max_logit_diff = diff_val

        # Top-1 agreement
        top1_eng = np.argmax(logits_engine, axis=-1)
        top1_ref = np.argmax(logits_ref, axis=-1)
        top1_matches += int(np.sum(top1_eng == top1_ref))

        # Top-5 agreement
        if logits_engine.shape[-1] >= 5:
            top5_eng = np.argpartition(logits_engine, -5, axis=-1)[..., -5:]
            top5_ref = np.argpartition(logits_ref, -5, axis=-1)[..., -5:]
            # Compute set intersection per position
            eng_flat = top5_eng.reshape(-1, 5)
            ref_flat = top5_ref.reshape(-1, 5)
            overlaps = sum(len(set(a) & set(b)) for a, b in zip(eng_flat, ref_flat, strict=False))
            top5_overlaps += overlaps / (len(eng_flat) * 5.0)
        else:
            top5_overlaps += 1.0

        # KL Divergence: D_KL(P_ref || P_eng)
        p_ref = _softmax(logits_ref)
        p_eng = _softmax(logits_engine)
        p_eng_clipped = np.clip(p_eng, 1e-12, 1.0)
        kl = float(np.mean(np.sum(p_ref * np.log(p_ref / p_eng_clipped + 1e-12), axis=-1)))
        kl_divergences.append(max(0.0, kl))

    top1_pct = (top1_matches / max(1, total_prompts)) * 100.0
    top5_pct = (top5_overlaps / max(1, total_prompts)) * 100.0
    mean_kl = float(np.mean(kl_divergences)) if kl_divergences else 0.0

    is_aligned = mean_kl <= kl_tolerance and top1_pct >= 99.0

    return ServingParityReport(
        is_aligned=is_aligned,
        top1_agreement_pct=top1_pct,
        top5_overlap_pct=top5_pct,
        max_logit_difference=max_logit_diff,
        mean_kl_divergence=mean_kl,
        prompts_evaluated=total_prompts,
    )
