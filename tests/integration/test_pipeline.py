from xlaxiom.comparison import Classification
from xlaxiom.generation import synthetic_control
from xlaxiom.minimization import minimize
from xlaxiom.pipeline import FaultInjection, run_comparison
from xlaxiom.storage import CaseArtifact, load_case, save_case


def test_jax_cpu_synthetic_control_minimizes_and_round_trips(tmp_path) -> None:
    workload = synthetic_control(7)
    fault = FaultInjection(0.125)
    assert (
        run_comparison(workload, fault=fault).classification == Classification.NUMERICAL_DISCREPANCY
    )
    result = minimize(
        workload,
        lambda item: run_comparison(item, fault=fault).classification,
        Classification.NUMERICAL_DISCREPANCY,
    )
    assert result.stats.reduced_nodes < result.stats.original_nodes
    path = save_case(
        tmp_path / "case.json",
        CaseArtifact(
            result.workload, "numerical_discrepancy", fault=fault, provenance="synthetic-control"
        ),
    )
    assert load_case(path).workload.digest() == result.workload.digest()
