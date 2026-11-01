"""HuggingFace checkpoint parity, Safetensors auditing, and serving engine verification.

Audits weight drift, quantized representation fidelity, and serving engine logit parity
(vLLM, SGLang, TensorRT-LLM vs HuggingFace reference models).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assertions import NumericalDiff, _to_numpy, diff

_TOP_K = 5


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
            lines.append("\nTop 5 Divergent Weights (lowest SNR first):")
            divergent = sorted(
                ((name, d) for name, d in self.weight_diffs if not d.is_close),
                key=lambda item: -math.inf if math.isnan(item[1].snr_db) else item[1].snr_db,
            )
            for name, d in divergent[:5]:
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

    suffix = p.suffix.lower()
    if suffix == ".npz":
        # Context manager releases the file handle (open handles lock files on Windows).
        with np.load(p, allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in data.files}

    if suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError as err:
            raise ImportError(
                "safetensors package required to read .safetensors files. "
                "Run `pip install safetensors`."
            ) from err
        return load_file(str(p))

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

    keys_a = set(dict_a)
    keys_b = set(dict_b)

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

    is_identical = regressed_count == 0 and not missing_keys and not extra_keys

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
    """Parity report comparing serving engine output logits against reference models.

    ``top1_agreement_pct`` and ``top5_overlap_pct`` are averaged over every evaluated
    token position (a ``(seq_len, vocab)`` logit tensor contributes ``seq_len`` positions).
    """

    is_aligned: bool
    top1_agreement_pct: float
    top5_overlap_pct: float
    max_logit_difference: float
    mean_kl_divergence: float
    prompts_evaluated: int
    non_finite_prompts: int = 0
    logits_within_tolerance: bool | None = None

    def format_summary(self) -> str:
        """Format terminal diagnostic report."""
        status = "PASSED" if self.is_aligned else "LOGIT DIVERGENCE DETECTED"
        lines = [
            f"=== Serving Engine Parity: {status} ===",
            f"Prompts evaluated: {self.prompts_evaluated}",
            f"Top-1 Token Agreement: {self.top1_agreement_pct:.2f}%",
            f"Top-5 Token Set Overlap: {self.top5_overlap_pct:.2f}%",
            f"Max Logit Difference: {self.max_logit_difference:.4e}",
            f"Mean KL Divergence: {self.mean_kl_divergence:.6e}",
        ]
        if self.non_finite_prompts:
            lines.append(f"Prompts with NaN/+inf logits: {self.non_finite_prompts}")
        if self.logits_within_tolerance is not None:
            lines.append(f"Logits within rtol/atol: {self.logits_within_tolerance}")
        return "\n".join(lines)


def _log_softmax(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Numerically stable log-softmax over the last axis (supports -inf masked logits)."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    result: np.ndarray[Any, Any] = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return result


def _has_invalid_logits(x: np.ndarray[Any, Any]) -> bool:
    """NaN and +inf are always corruption; -inf is a legitimate masked-token logit."""
    return bool(np.isnan(x).any() or np.isposinf(x).any())


def audit_serving_parity(
    engine_fn: Callable[..., Any],
    reference_fn: Callable[..., Any],
    sample_inputs: Sequence[Any],
    *,
    rtol: float | None = None,
    atol: float | None = None,
    kl_tolerance: float = 1e-4,
    min_top1_agreement_pct: float = 99.0,
) -> ServingParityReport:
    """Audit output logit fidelity between an optimized serving engine and a reference model.

    Parameters
    ----------
    engine_fn : Callable
        Inference engine function (e.g. vLLM, SGLang, or TensorRT-LLM generating logits).
    reference_fn : Callable
        Reference model function (e.g. HuggingFace PyTorch eager).
    sample_inputs : Sequence[Any]
        List of input prompt tensors or token IDs. Must be non-empty.
    rtol : float, optional
        Relative tolerance on logits. When ``rtol`` or ``atol`` is given, every logit must
        also satisfy ``|engine - ref| <= atol + rtol * |ref|`` for the report to be aligned.
        Left unset (default), only KL divergence and top-1 agreement decide alignment.
    atol : float, optional
        Absolute tolerance on logits (see ``rtol``).
    kl_tolerance : float, optional
        Maximum allowable mean KL divergence before flagging divergence.
    min_top1_agreement_pct : float, optional
        Minimum top-1 token agreement, in percent (default: 99.0).

    Returns
    -------
    ServingParityReport
        Report detailing Top-1/Top-5 token agreement and distribution KL divergence.

    Raises
    ------
    ValueError
        If ``sample_inputs`` is empty or the engine and reference logits differ in shape.
    """
    if len(sample_inputs) == 0:
        raise ValueError("audit_serving_parity requires at least one sample input.")

    check_tolerance = rtol is not None or atol is not None
    rtol_eff = 0.0 if rtol is None else rtol
    atol_eff = 0.0 if atol is None else atol

    total_positions = 0
    top1_matches = 0
    top5_overlap_sum = 0.0
    kl_sum = 0.0
    max_logit_diff = 0.0
    non_finite_prompts = 0
    within_tolerance = True

    for index, inp in enumerate(sample_inputs):
        logits_engine = _to_numpy(engine_fn(inp)).astype(np.float64)
        logits_ref = _to_numpy(reference_fn(inp)).astype(np.float64)

        if logits_engine.shape != logits_ref.shape:
            raise ValueError(
                f"Logit shape mismatch for sample {index}: engine {logits_engine.shape} "
                f"vs reference {logits_ref.shape}."
            )
        if logits_engine.ndim == 0:
            raise ValueError(f"Logits for sample {index} must have a vocabulary axis.")

        if _has_invalid_logits(logits_engine) or _has_invalid_logits(logits_ref):
            # Never let NaN silently look like agreement; excluded from the position stats.
            non_finite_prompts += 1
            continue

        with np.errstate(invalid="ignore"):
            abs_diff = np.where(
                logits_engine == logits_ref, 0.0, np.abs(logits_engine - logits_ref)
            )
        max_logit_diff = max(max_logit_diff, float(np.max(abs_diff)))
        if check_tolerance:
            allowed = atol_eff + rtol_eff * np.abs(logits_ref)
            within_tolerance = within_tolerance and bool(np.all(abs_diff <= allowed))

        vocab = logits_ref.shape[-1]
        positions = int(np.prod(logits_ref.shape[:-1], dtype=np.int64)) or 1
        total_positions += positions

        top1_matches += int(
            np.sum(np.argmax(logits_engine, axis=-1) == np.argmax(logits_ref, axis=-1))
        )

        k = min(_TOP_K, vocab)
        top_eng = np.argpartition(logits_engine, -k, axis=-1)[..., -k:].reshape(-1, k)
        top_ref = np.argpartition(logits_ref, -k, axis=-1)[..., -k:].reshape(-1, k)
        shared = (top_eng[:, :, None] == top_ref[:, None, :]).any(axis=-1).sum(axis=-1)
        top5_overlap_sum += float(np.sum(shared / k))

        # D_KL(P_ref || P_eng) computed in log space so tiny probabilities are not clipped.
        lp_ref = _log_softmax(logits_ref)
        lp_eng = _log_softmax(logits_engine)
        p_ref = np.exp(lp_ref)
        with np.errstate(invalid="ignore"):
            terms = np.where(p_ref > 0.0, p_ref * (lp_ref - lp_eng), 0.0)
        kl_sum += float(np.sum(np.maximum(0.0, np.sum(terms, axis=-1))))

    n_prompts = len(sample_inputs)
    if total_positions > 0:
        top1_pct = top1_matches / total_positions * 100.0
        top5_pct = top5_overlap_sum / total_positions * 100.0
        mean_kl = kl_sum / total_positions
    else:
        top1_pct = top5_pct = 0.0
        mean_kl = 0.0

    is_aligned = (
        non_finite_prompts == 0
        and total_positions > 0
        and mean_kl <= kl_tolerance
        and top1_pct >= min_top1_agreement_pct
        and (within_tolerance or not check_tolerance)
    )

    return ServingParityReport(
        is_aligned=is_aligned,
        top1_agreement_pct=top1_pct,
        top5_overlap_pct=top5_pct,
        max_logit_difference=max_logit_diff,
        mean_kl_divergence=mean_kl,
        prompts_evaluated=n_prompts,
        non_finite_prompts=non_finite_prompts,
        logits_within_tolerance=within_tolerance if check_tolerance else None,
    )
