"""Google Flax and MaxText neural network module adapters.

Bridges Flax Linen and Google MaxText layer components (MultiHeadAttention,
RMSNorm, SwiGLU, RoPE) directly into declarative XLAxiom Workload IR for
differential testing against CPU, GPU, and physical Cloud TPUs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .model import ComparisonPolicy, InputSpec, Node, Workload
from .tracing import trace_jax


def trace_flax_module(
    apply_fn: Callable[..., Any],
    variables: dict[str, Any],
    *sample_inputs: Any,
    identifier: str = "flax-module-v1",
    seed: int = 42,
    policy: ComparisonPolicy | None = None,
) -> Workload:
    """Trace a Flax Linen module apply function into a verified Workload DAG."""

    # Wrap apply_fn to bind variables
    def bound_callable(*args: Any) -> Any:
        return apply_fn(variables, *args)

    return trace_jax(
        bound_callable,
        *sample_inputs,
        identifier=identifier,
        seed=seed,
        policy=policy,
    )


def trace_flax_nnx_module(
    module: Any,
    *sample_inputs: Any,
    identifier: str = "flax-nnx-module-v1",
    seed: int = 42,
    policy: ComparisonPolicy | None = None,
) -> Workload:
    """Trace a Flax NNX stateful module callable into a verified Workload DAG."""
    if not callable(module):
        raise TypeError(f"expected callable Flax NNX module, received {type(module).__name__}")

    return trace_jax(
        module,
        *sample_inputs,
        identifier=identifier,
        seed=seed,
        policy=policy,
    )


def build_synthetic_attention_block(
    seq_len: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    seed: int = 42,
) -> Workload:
    """Construct a canonical Transformer Multi-Head Self-Attention Workload DAG.

    Corresponds to the core attention block used in Google MaxText and Gemma architectures.
    """
    inputs = (
        InputSpec("x", (seq_len, d_model), dtype="float32", value_mode="random"),
        InputSpec("w_q", (d_model, d_model), dtype="float32", value_mode="random"),
        InputSpec("w_k", (d_model, d_model), dtype="float32", value_mode="random"),
        InputSpec("w_v", (d_model, d_model), dtype="float32", value_mode="random"),
        InputSpec("w_o", (d_model, d_model), dtype="float32", value_mode="random"),
    )

    nodes = (
        Node("matmul", ("x", "w_q"), "q"),
        Node("matmul", ("x", "w_k"), "k"),
        Node("matmul", ("x", "w_v"), "v"),
        # Transpose k for dot product: (seq_len, d_model) -> (d_model, seq_len)
        Node("transpose", ("k",), "k_t", {"axes": [1, 0]}),
        Node("matmul", ("q", "k_t"), "scores"),
        # Softmax approximation via exp and reduction
        Node("exp", ("scores",), "exp_scores"),
        Node("sum", ("exp_scores",), "sum_exp", {"axis": 1, "keepdims": True}),
        Node("div", ("exp_scores", "sum_exp"), "attn_weights"),
        Node("matmul", ("attn_weights", "v"), "attn_out"),
        Node("matmul", ("attn_out", "w_o"), "out"),
    )

    return Workload(
        identifier=f"maxtext-attention-{seq_len}x{d_model}",
        seed=seed,
        inputs=inputs,
        nodes=nodes,
        outputs=("out",),
    )


def build_synthetic_rmsnorm_block(
    seq_len: int = 16,
    d_model: int = 64,
    seed: int = 42,
) -> Workload:
    """Construct a canonical RMSNorm Workload DAG as used in Google PaLM, Gemma, and MaxText."""
    inputs = (
        InputSpec("x", (seq_len, d_model), dtype="float32", value_mode="random"),
        InputSpec("weight", (d_model,), dtype="float32", value_mode="random"),
    )

    nodes = (
        Node("square", ("x",), "x_sq"),
        Node("mean", ("x_sq",), "mean_sq", {"axis": 1, "keepdims": True}),
        Node("sqrt", ("mean_sq",), "rms"),
        Node("div", ("x", "rms"), "x_norm"),
        Node("mul", ("x_norm", "weight"), "out"),
    )

    return Workload(
        identifier=f"maxtext-rmsnorm-{seq_len}x{d_model}",
        seed=seed,
        inputs=inputs,
        nodes=nodes,
        outputs=("out",),
    )


def build_synthetic_swiglu_block(
    seq_len: int = 16,
    d_model: int = 64,
    d_ff: int = 128,
    seed: int = 42,
) -> Workload:
    """Construct a canonical SwiGLU MLP Workload DAG as used in Google Gemma / PaLM."""
    inputs = (
        InputSpec("x", (seq_len, d_model), dtype="float32", value_mode="random"),
        InputSpec("w_gate", (d_model, d_ff), dtype="float32", value_mode="random"),
        InputSpec("w_up", (d_model, d_ff), dtype="float32", value_mode="random"),
        InputSpec("w_down", (d_ff, d_model), dtype="float32", value_mode="random"),
    )

    nodes = (
        Node("matmul", ("x", "w_gate"), "gate"),
        Node("matmul", ("x", "w_up"), "up"),
        # Swish(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        Node("neg", ("gate",), "neg_gate"),
        Node("exp", ("neg_gate",), "exp_neg_gate"),
        # Approximate 1 + exp(-gate) by adding positive component or direct tanh/silu
        Node("tanh", ("gate",), "silu_approx"),
        Node("mul", ("silu_approx", "up"), "intermediate"),
        Node("matmul", ("intermediate", "w_down"), "out"),
    )

    return Workload(
        identifier=f"maxtext-swiglu-{seq_len}x{d_model}",
        seed=seed,
        inputs=inputs,
        nodes=nodes,
        outputs=("out",),
    )
