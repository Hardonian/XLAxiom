"""Top-k logprob parity for live serving engines (vLLM, SGLang, any OpenAI-compatible server).

OpenAI-compatible ``/v1/completions`` endpoints expose only the top-k log-probabilities of each
generated token, not full-vocabulary logits. This module audits agreement on exactly that
surface: it compares a serving engine against a reference callable that returns the same
``{token: logprob}`` mapping.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TopKFn = Callable[[str], Mapping[str, float]]


def _default_normalize(token: str) -> str:
    return token.strip()


@dataclass(frozen=True)
class TopKParityReport:
    """Agreement between two engines' top-k next-token distributions."""

    is_aligned: bool
    prompts_evaluated: int
    k: int
    top1_agreement_pct: float
    topk_overlap_pct: float
    mean_abs_logprob_diff: float
    max_abs_logprob_diff: float

    def format_summary(self) -> str:
        status = "PASSED" if self.is_aligned else "TOP-K DIVERGENCE DETECTED"
        return (
            f"=== Serving Top-{self.k} Parity: {status} ===\n"
            f"Prompts evaluated: {self.prompts_evaluated}\n"
            f"Top-1 Token Agreement: {self.top1_agreement_pct:.2f}%\n"
            f"Top-{self.k} Set Overlap: {self.topk_overlap_pct:.2f}%\n"
            f"Mean |logprob diff| (shared tokens): {self.mean_abs_logprob_diff:.4e}\n"
            f"Max  |logprob diff| (shared tokens): {self.max_abs_logprob_diff:.4e}"
        )


def _top_k(
    mapping: Mapping[str, float], k: int, normalize: Callable[[str], str]
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for token, logprob in mapping.items():
        key = normalize(token)
        value = float(logprob)
        if key not in merged or value > merged[key]:
            merged[key] = value
    ranked = sorted(merged.items(), key=lambda item: -item[1])[:k]
    return dict(ranked)


def audit_topk_parity(
    engine_fn: TopKFn,
    reference_fn: TopKFn,
    prompts: Sequence[str],
    *,
    k: int = 5,
    min_top1_agreement_pct: float = 99.0,
    min_overlap_pct: float = 80.0,
    max_mean_logprob_diff: float = 0.05,
    normalize: Callable[[str], str] = _default_normalize,
) -> TopKParityReport:
    """Compare the top-``k`` next-token logprobs of an engine and a reference.

    Parameters
    ----------
    engine_fn, reference_fn : Callable[[str], Mapping[str, float]]
        Map a prompt to ``{token_text: logprob}`` for the next token.
    normalize : Callable[[str], str], optional
        Token canonicalization applied before matching (default: ``str.strip``). Tokenizers
        render whitespace differently, so adjust this if your engines disagree on spelling.

    Raises
    ------
    ValueError
        On empty ``prompts``, ``k < 1``, or an empty/non-finite distribution.
    """
    if len(prompts) == 0:
        raise ValueError("audit_topk_parity requires at least one prompt.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    top1_hits = 0
    overlap_sum = 0.0
    diffs: list[float] = []
    for prompt in prompts:
        eng = _top_k(engine_fn(prompt), k, normalize)
        ref = _top_k(reference_fn(prompt), k, normalize)
        if not eng or not ref:
            raise ValueError(f"Empty top-k distribution returned for prompt {prompt!r}.")
        values = list(eng.values()) + list(ref.values())
        if not all(np.isfinite(v) or v == -np.inf for v in values) or any(
            v == np.inf for v in values
        ):
            raise ValueError(f"Non-finite logprob returned for prompt {prompt!r}.")

        top1_hits += int(next(iter(eng)) == next(iter(ref)))
        shared = set(eng) & set(ref)
        overlap_sum += len(shared) / max(1, min(k, len(ref)))
        diffs.extend(abs(eng[t] - ref[t]) for t in shared if np.isfinite(eng[t] - ref[t]))

    n = len(prompts)
    top1_pct = top1_hits / n * 100.0
    overlap_pct = overlap_sum / n * 100.0
    mean_diff = float(np.mean(diffs)) if diffs else float("inf")
    max_diff = float(np.max(diffs)) if diffs else float("inf")
    aligned = (
        top1_pct >= min_top1_agreement_pct
        and overlap_pct >= min_overlap_pct
        and mean_diff <= max_mean_logprob_diff
    )
    return TopKParityReport(
        is_aligned=aligned,
        prompts_evaluated=n,
        k=k,
        top1_agreement_pct=top1_pct,
        topk_overlap_pct=overlap_pct,
        mean_abs_logprob_diff=mean_diff,
        max_abs_logprob_diff=max_diff,
    )


def openai_compatible_topk(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    top_logprobs: int = 20,
    timeout: float = 60.0,
) -> TopKFn:
    """Build a ``prompt -> {token: logprob}`` callable for a vLLM/SGLang/OpenAI-style server.

    Sends a greedy, one-token ``POST {base_url}/v1/completions`` request with ``logprobs``.
    ``api_key`` is only sent when passed explicitly (no ambient credentials are read).
    Only ``http``/``https`` URLs are accepted.
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}.")
    endpoint = base_url.rstrip("/") + "/v1/completions"
    # Defense in depth: re-assert the final endpoint's scheme/netloc before any request
    # is built, so file:/custom schemes can never reach urlopen even if URL assembly
    # above changes.
    parsed_endpoint = urllib.parse.urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ValueError(f"serving endpoint must be an http(s) URL, got {endpoint!r}.")

    def _query(prompt: str) -> dict[str, float]:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": top_logprobs,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            # nosec B310 - endpoint is validated as an absolute http(s) URL (scheme +
            # netloc) above before this closure exists; bandit's blacklist check has no
            # dataflow analysis, so a genuinely dynamic (user-configured) endpoint can
            # only be cleared by justified nosec after the root-cause guard is in place.
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310  # nosec B310
                payload: Any = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"Serving endpoint returned HTTP {err.code}: {err.reason}") from err
        except urllib.error.URLError as err:
            raise RuntimeError(
                f"Could not reach serving endpoint {endpoint}: {err.reason}"
            ) from err
        try:
            top = payload["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as err:
            raise RuntimeError(
                "Response did not contain choices[0].logprobs.top_logprobs[0]."
            ) from err
        return {str(token): float(logprob) for token, logprob in top.items()}

    return _query


def topk_from_logits(
    logits_fn: Callable[[str], Any],
    decode: Callable[[int], str],
    *,
    k: int = 20,
) -> TopKFn:
    """Adapt a reference ``prompt -> next-token logits`` function to the top-k interface.

    ``logits_fn`` returns a 1-D vector over the vocabulary; ``decode`` maps a token id to text.
    """

    def _topk(prompt: str) -> dict[str, float]:
        logits = np.asarray(logits_fn(prompt), dtype=np.float64).reshape(-1)
        shifted = logits - np.max(logits)
        logprobs = shifted - np.log(np.sum(np.exp(shifted)))
        idx = np.argsort(-logprobs)[:k]
        return {decode(int(i)): float(logprobs[i]) for i in idx}

    return _topk


def huggingface_topk(
    model_id: str,
    *,
    k: int = 20,
    device: str = "cpu",
    torch_dtype: str | None = None,
    revision: str | None = None,
) -> TopKFn:
    """Reference ``prompt -> top-k logprobs`` from a HuggingFace causal LM.

    Requires ``transformers`` and ``torch`` (imported lazily).

    ``revision`` pins the model weights and tokenizer to an exact commit SHA or tag.
    For reproducible parity audits, callers should pin a revision; leaving it ``None``
    tracks the mutable ``main`` branch of the Hub repo.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:
        raise ImportError(
            "huggingface_topk requires `transformers` and `torch` (pip install transformers torch)."
        ) from err

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
    # ``revision`` is passed explicitly (never via **kwargs) so static auditors — and
    # readers — can see the download is pinnable (bandit B615 / CWE-494).
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **model_kwargs)
        .to(device)
        .eval()
    )

    def _logits(prompt: str) -> Any:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).logits[0, -1].float().cpu().numpy()
        return out

    return topk_from_logits(_logits, lambda i: tokenizer.decode([i]), k=k)
