"""The stable command-line interface for the CPU-first evidence pipeline."""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter
from typing import Any

from .bundle import create_bundle, run_bundle, validate_bundle
from .campaign import CampaignConfig, run_cpu_campaign
from .comparison import Classification
from .cuda import cuda_capability
from .execution import benchmark_jax, doctor
from .generation import GenerationConfig, generate, synthetic_control
from .historical import hyperbolic_gradient_study, replay_jax_40634
from .historical_matrix import run_hyperbolic_matrix
from .matrix import run_matrix
from .minimization import minimize
from .pipeline import FaultInjection, run_backend, run_comparison
from .reporting import write_report
from .storage import ArtifactError, CaseArtifact, load_case, save_case, save_json


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _load(path: str) -> CaseArtifact:
    try:
        return load_case(path)
    except ArtifactError as exc:
        raise SystemExit(f"invalid case artifact: {exc}") from exc


def command_doctor(_: argparse.Namespace) -> int:
    _emit(doctor())
    return 0


def command_generate(args: argparse.Namespace) -> int:
    workload = (
        synthetic_control(args.seed)
        if args.synthetic
        else generate(GenerationConfig(seed=args.seed, max_nodes=args.max_nodes))
    )
    artifact = CaseArtifact(
        workload=workload,
        expected_classification=Classification.NUMERICAL_DISCREPANCY.value
        if args.synthetic
        else None,
        fault=FaultInjection(args.offset) if args.synthetic else None,
        provenance="synthetic-control" if args.synthetic else "generated",
    )
    output = save_case(args.output, artifact)
    _emit({"case": str(output), "workload_sha256": workload.digest(), "synthetic": args.synthetic})
    return 0


def command_run(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    result = run_backend(artifact.workload, args.backend)
    _emit(result.summary())
    return 0 if result.status.value == "ok" else 2


def command_compare(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    comparison = run_comparison(
        artifact.workload,
        args.mode or artifact.comparison_mode,
        artifact.fault if args.inject else None,
    )
    record = comparison.to_dict()
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if comparison.classification == Classification.PASS else 1


def command_minimize(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    baseline = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
    expected_text = artifact.expected_classification or baseline.classification.value
    try:
        expected = Classification(expected_text)
    except ValueError as exc:
        raise SystemExit(f"case has invalid expected classification {expected_text!r}") from exc
    if baseline.classification != expected:
        _emit(
            {
                "baseline": baseline.to_dict(),
                "error": "failure predicate is not currently reproducible",
            }
        )
        return 2

    outcome = minimize(
        artifact.workload,
        lambda candidate: run_comparison(
            candidate, artifact.comparison_mode, artifact.fault
        ).failure_signature(),
        baseline.failure_signature(),
        args.max_evaluations,
    )
    minimized = CaseArtifact(
        workload=outcome.workload,
        expected_classification=expected.value,
        comparison_mode=artifact.comparison_mode,
        fault=artifact.fault,
        provenance=artifact.provenance,
        minimization={**outcome.stats.__dict__, "original_sha256": artifact.workload.digest()},
    )
    output = save_case(args.output, minimized)
    _emit({"case": str(output), "stats": minimized.minimization})
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    comparison = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
    expected = artifact.expected_classification
    verdict = expected is not None and comparison.classification.value == expected
    _emit(
        {
            "verdict": "reproduced" if verdict else "not_reproduced",
            "expected": expected,
            "observed": comparison.to_dict(),
        }
    )
    return 0 if verdict else 1


def command_report(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    comparison = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
    json_path, markdown_path = write_report(args.output, artifact, comparison)
    _emit(
        {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "classification": comparison.classification.value,
        }
    )
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    if args.backend == "jax-jit":
        _emit(benchmark_jax(artifact.workload, args.measurements))
        return 0
    warmup = run_backend(artifact.workload, args.backend)
    if warmup.status.value != "ok":
        _emit(warmup.summary())
        return 2
    timings: list[float] = []
    for _ in range(args.measurements):
        started = perf_counter()
        result = run_backend(artifact.workload, args.backend)
        if result.status.value != "ok":
            _emit(result.summary())
            return 2
        timings.append((perf_counter() - started) * 1000)
    _emit(
        {
            "backend": args.backend,
            "warmup_ms": warmup.duration_ms,
            "measurements": args.measurements,
            "median_ms": median(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
        }
    )
    return 0


def command_historical(_: argparse.Namespace) -> int:
    observation = replay_jax_40634()
    _emit(observation.to_dict())
    return (
        0
        if observation.classification
        in {"resolved_historical_regression", "known_historical_regression_reproduced"}
        else 1
    )


def command_historical_study(args: argparse.Namespace) -> int:
    study = hyperbolic_gradient_study()
    if args.output:
        save_json(args.output, study)
    _emit(study)
    return 0


def command_historical_matrix(args: argparse.Namespace) -> int:
    _emit(run_hyperbolic_matrix(args.version, args.output))
    return 0


def command_campaign(args: argparse.Namespace) -> int:
    try:
        record = run_cpu_campaign(CampaignConfig(args.cases, args.seed, args.max_duration_s))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if not record["candidate_anomalies"] else 1


def command_matrix_run(args: argparse.Namespace) -> int:
    _emit(run_matrix(args.config, args.case, args.output))
    return 0


def command_matrix_doctor(_: argparse.Namespace) -> int:
    _emit({"cuda": cuda_capability(), "format": "JSON matrix config only in v0.3"})
    return 0


def command_bundle_create(args: argparse.Namespace) -> int:
    _emit(create_bundle(args.case, args.output))
    return 0


def command_bundle_validate(args: argparse.Namespace) -> int:
    _emit(validate_bundle(args.bundle))
    return 0


def command_bundle_run(args: argparse.Namespace) -> int:
    _emit(run_bundle(args.bundle, args.output))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="xlaxiom", description="Evidence-first numerical differential testing for JAX/OpenXLA."
    )
    commands = root.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="inspect JAX/backend capabilities")
    doctor_parser.set_defaults(func=command_doctor)
    generate_parser = commands.add_parser(
        "generate", help="create a deterministic declarative case"
    )
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-nodes", type=int, default=8)
    generate_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="create a visibly labelled synthetic discrepancy control",
    )
    generate_parser.add_argument(
        "--offset", type=float, default=0.125, help="test-only injected output offset"
    )
    generate_parser.add_argument("--output", default="case.json")
    generate_parser.set_defaults(func=command_generate)
    for name, handler, help_text in (("run", command_run, "execute one backend"),):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("--case", required=True)
        subparser.add_argument(
            "--backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-jit"
        )
        subparser.set_defaults(func=handler)
    compare_parser = commands.add_parser("compare", help="compare NumPy and JAX executions")
    compare_parser.add_argument("--case", required=True)
    compare_parser.add_argument("--mode", choices=("jax-eager", "jax-jit", "jax-eager-vs-jit"))
    compare_parser.add_argument(
        "--inject", action="store_true", help="enable only the case's explicit synthetic control"
    )
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(func=command_compare)
    minimize_parser = commands.add_parser("minimize", help="failure-preserving workload reduction")
    minimize_parser.add_argument("--case", required=True)
    minimize_parser.add_argument("--output", required=True)
    minimize_parser.add_argument("--max-evaluations", type=int, default=64)
    minimize_parser.set_defaults(func=command_minimize)
    reproduce_parser = commands.add_parser(
        "reproduce", help="reexecute and evaluate the stored predicate"
    )
    reproduce_parser.add_argument("--case", required=True)
    reproduce_parser.set_defaults(func=command_reproduce)
    report_parser = commands.add_parser("report", help="write JSON and Markdown evidence reports")
    report_parser.add_argument("--case", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=command_report)
    benchmark_parser = commands.add_parser("benchmark", help="bounded timing measurement")
    benchmark_parser.add_argument("--case", required=True)
    benchmark_parser.add_argument(
        "--backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-jit"
    )
    benchmark_parser.add_argument("--measurements", type=int, default=5, choices=range(1, 51))
    benchmark_parser.set_defaults(func=command_benchmark)
    historical_parser = commands.add_parser(
        "historical", help="replay trusted adapters for documented historical regressions"
    )
    historical_parser.set_defaults(func=command_historical)
    historical_study = commands.add_parser(
        "historical-study", help="run per-operation #40634 study"
    )
    historical_study.add_argument("--output")
    historical_study.set_defaults(func=command_historical_study)
    historical_matrix = commands.add_parser(
        "historical-matrix", help="run #40634 study in isolated versions"
    )
    historical_matrix.add_argument("--version", action="append", required=True)
    historical_matrix.add_argument("--output", required=True)
    historical_matrix.set_defaults(func=command_historical_matrix)
    campaign_parser = commands.add_parser(
        "campaign", help="run a bounded trusted CPU NumPy/JAX differential campaign"
    )
    campaign_parser.add_argument("--cases", type=int, default=20)
    campaign_parser.add_argument("--seed", type=int, default=0)
    campaign_parser.add_argument("--max-duration-s", type=float, default=180.0)
    campaign_parser.add_argument("--output")
    campaign_parser.set_defaults(func=command_campaign)
    matrix_parser = commands.add_parser("matrix", help="run isolated JAX version matrices")
    matrix_commands = matrix_parser.add_subparsers(dest="matrix_command", required=True)
    matrix_run = matrix_commands.add_parser("run")
    matrix_run.add_argument("--config", required=True)
    matrix_run.add_argument("--case", required=True)
    matrix_run.add_argument("--output", required=True)
    matrix_run.set_defaults(func=command_matrix_run)
    matrix_doctor = matrix_commands.add_parser("doctor")
    matrix_doctor.set_defaults(func=command_matrix_doctor)
    bundle_parser = commands.add_parser("bundle", help="create and validate portable case bundles")
    bundle_commands = bundle_parser.add_subparsers(dest="bundle_command", required=True)
    bundle_create = bundle_commands.add_parser("create")
    bundle_create.add_argument("--case", action="append", required=True)
    bundle_create.add_argument("--output", required=True)
    bundle_create.set_defaults(func=command_bundle_create)
    bundle_validate = bundle_commands.add_parser("validate")
    bundle_validate.add_argument("--bundle", required=True)
    bundle_validate.set_defaults(func=command_bundle_validate)
    bundle_run = bundle_commands.add_parser("run")
    bundle_run.add_argument("--bundle", required=True)
    bundle_run.add_argument("--output", required=True)
    bundle_run.set_defaults(func=command_bundle_run)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))
