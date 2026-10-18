"""Unit tests for offline HTML report 2.0 and embedded SVG DAG visualizer (Priority 68)."""

from xlaxiom.comparison import Classification, ComparisonResult
from xlaxiom.execution import ExecutionResult, ExecutionStatus
from xlaxiom.generation import synthetic_control
from xlaxiom.model import ComparisonPolicy
from xlaxiom.reporting import build_html_report, build_report, render_svg_dag
from xlaxiom.storage import CaseArtifact


def test_render_svg_dag_structure() -> None:
    workload = synthetic_control(seed=42)
    svg = render_svg_dag(workload.canonical_dict())

    assert "<svg" in svg
    assert 'class="dag-svg"' in svg
    assert 'marker id="dag-arrow"' in svg
    assert "<rect" in svg
    assert "<path" in svg
    # Check that node operations and inputs appear
    for node in workload.nodes:
        assert node.op in svg


def test_build_html_report_includes_svg_dag() -> None:
    artifact = CaseArtifact(synthetic_control(seed=42), fault=None)
    policy = ComparisonPolicy(atol=1e-5, rtol=1e-5)
    import numpy as np

    left = ExecutionResult(
        "left", "eager", ExecutionStatus.OK, (np.array([1.0], dtype=np.float32),)
    )
    right = ExecutionResult(
        "right", "eager", ExecutionStatus.OK, (np.array([2.0], dtype=np.float32),)
    )
    comp = ComparisonResult(
        classification=Classification.NUMERICAL_DISCREPANCY,
        explanation="Values differed beyond tolerance.",
        policy=policy,
        left=left.summary(),
        right=right.summary(),
    )

    report = build_report(artifact, comp)
    html = build_html_report(report)

    assert "Computation Graph DAG" in html
    assert "dag-svg" in html
    assert 'marker id="dag-arrow"' in html
