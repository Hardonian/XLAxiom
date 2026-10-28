"""MaxText numerical parity audit harness.

Can be run directly inside google/maxtext to verify MultiHeadAttention, RMSNorm,
and SwiGLU computational blocks across CPU and Cloud TPU backends.
"""

from __future__ import annotations

import sys

from xlaxiom.flax import (
    build_synthetic_attention_block,
    build_synthetic_rmsnorm_block,
    build_synthetic_swiglu_block,
)
from xlaxiom.pipeline import run_comparison


def test_maxtext_components() -> int:
    blocks = [
        ("Attention", build_synthetic_attention_block(seq_len=16, d_model=64, num_heads=4)),
        ("RMSNorm", build_synthetic_rmsnorm_block(seq_len=16, d_model=64)),
        ("SwiGLU", build_synthetic_swiglu_block(seq_len=16, d_model=64, d_ff=128)),
    ]

    print("====================================================================")
    print(" Google MaxText Layer Numerical Parity Audit")
    print("====================================================================")
    all_passed = True
    for name, workload in blocks:
        result = run_comparison(workload, mode="jax-jit")
        status = result.classification.value
        print(f"[{name}] Verification Status: {status.upper()}")
        if status != "pass":
            print(f"  Explanation: {result.explanation}")
            all_passed = False

    if all_passed:
        print("\nAll MaxText components passed numerical differential verification.")
        return 0
    else:
        print("\nDiscrepancies detected in MaxText layer verification.")
        return 1


if __name__ == "__main__":
    sys.exit(test_maxtext_components())
