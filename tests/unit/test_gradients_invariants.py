"""Unit tests for higher-order autodiff mathematical invariants."""

from __future__ import annotations

from xlaxiom.gradients import GradientCase, audit_higher_order_invariants


def test_audit_higher_order_invariants_tanh() -> None:
    case = GradientCase("test-tanh-0", "tanh", 0.5, dtype="float32")
    audit = audit_higher_order_invariants(case, u=1.0, v=0.5)

    assert audit["all_invariants_satisfied"] is True
    assert audit["classification"] == "pass"
    assert audit["clairaut_symmetry"]["satisfied"] is True
    assert audit["hessian_linearity"]["satisfied"] is True
    assert audit["taylor_consistency"]["satisfied"] is True


def test_audit_higher_order_invariants_exp() -> None:
    case = GradientCase("test-exp-0", "exp", 1.0, dtype="float32")
    audit = audit_higher_order_invariants(case, u=0.8, v=0.4)

    assert audit["all_invariants_satisfied"] is True
    assert audit["clairaut_symmetry"]["satisfied"] is True
    assert audit["hessian_linearity"]["satisfied"] is True
    assert audit["taylor_consistency"]["satisfied"] is True
