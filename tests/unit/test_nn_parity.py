"""Unit tests for neural network layer-by-layer parity auditing."""

from __future__ import annotations

import numpy as np

from xlaxiom.nn_parity import audit_layer_parity


def test_audit_layer_parity_dict_inputs() -> None:
    # Model A: Reference activations
    acts_a = {
        "embed": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer_0.attn": np.array([[0.5, 0.5], [0.8, 0.2]]),
        "layer_0.ffn": np.array([[1.2, 2.4], [3.6, 4.8]]),
    }

    # Model B: Layer 0 attention diverges!
    acts_b = {
        "embed": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "layer_0.attn": np.array([[0.9, 0.1], [0.8, 0.2]]),  # Divergence
        "layer_0.ffn": np.array([[2.0, 3.0], [4.0, 5.0]]),  # Downstream cascade
    }

    report = audit_layer_parity(acts_a, acts_b, rtol=1e-3, atol=1e-3)
    assert report.all_passed is False
    assert report.divergence_origin == "layer_0.attn"
    assert len(report.layers) == 3

    table_output = report.format_table()
    assert "Divergence Origin" in table_output
    assert "layer_0.attn" in table_output

    html_output = report._repr_html_()
    assert "XLAxiom Layer Parity Audit" in html_output
    assert "(Origin)" in html_output


def test_audit_layer_parity_callable_models() -> None:
    def model_a(x: np.ndarray) -> dict[str, np.ndarray]:
        h1 = x * 2.0
        h2 = h1 + 1.0
        return {"layer1": h1, "layer2": h2}

    def model_b(x: np.ndarray) -> dict[str, np.ndarray]:
        h1 = x * 2.0
        h2 = h1 + 1.0
        return {"layer1": h1, "layer2": h2}

    x = np.ones((2, 4))
    report = audit_layer_parity(model_a, model_b, x)
    assert report.all_passed is True
    assert report.divergence_origin is None
