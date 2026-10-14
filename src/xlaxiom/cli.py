"""The stable command-line interface for the CPU-first evidence pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from . import __version__
from .attestation import (
    load_attestation,
    save_attestation,
    sign_attestation,
    verify_subject_attestation,
)
from .bundle import create_bundle, run_bundle, validate_bundle
from .campaign import CAMPAIGN_PROFILES, configure_campaign, run_cpu_campaign
from .comparison import Classification
from .cuda import cuda_capability, driver_capability, recommended_variant
from .environment import (
    capture_environment,
    compare_environments,
    load_environment,
    save_environment,
)
from .execution import benchmark_jax, doctor
from .generation import GenerationConfig, generate, synthetic_control
from .gpu_campaign import (
    CampaignBudget,
    replay_jax_40634_on_gpu,
    run_gpu_campaign,
)
from .gpu_validation import JAX_GPU_PREALLOCATE, DeviceState, validate_all, variant_python
from .gradient_artifacts import (
    GradientArtifactError,
    compare_gradient_campaigns,
    load_gradient_campaign,
)
from .gradients import generate_gradient_case, run_gradient_campaign, run_gradient_case
from .hardware import (
    enumerate_physical_gpus,
    library_path_risk,
    ordinal_ordering_risk,
    select_variant,
)
from .historical import hyperbolic_gradient_study, replay_jax_40634
from .historical_matrix import run_hyperbolic_matrix
from .ledger import completed_shard, load_ledger, save_shard
from .matrix import run_matrix, run_relation_matrix
from .metamorphic import (
    PROPERTY_BACKENDS,
    MetamorphicEvidenceError,
    MetamorphicPropertyError,
    build_property_artifact,
    compare_property_evidence,
    curated_properties,
    load_property_evidence,
    property_coverage,
    run_curated_properties,
    run_property_campaign,
)
from .minimization import minimize
from .model import ComparisonPolicy
from .mutation import evaluate_property_sharpness
from .outcomes import compare_outcomes
from .performance import PerformanceArtifactError, compare_benchmarks, compare_compilation_cache
from .pipeline import FaultInjection, run_backend, run_comparison
from .readiness import (
    ReadinessEvidenceError,
    assess_upstream_readiness,
    build_reproduction_evidence,
    load_reproduction_evidence,
)
from .relational import (
    minimize_relational,
    reproduce_relational_isolated,
    run_relational_comparison,
    write_relational_report,
)
from .reporting import write_report
from .runtime import RUNTIME_PROFILES, runtime_profile
from .stability import check_stability
from .storage import (
    ArtifactError,
    CaseArtifact,
    RelationalArtifact,
    load_case,
    load_relational_case,
    save_case,
    save_json,
    save_relational_case,
)


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _load(path: str) -> CaseArtifact:
    try:
        return load_case(path)
    except ArtifactError as exc:
        raise SystemExit(f"invalid case artifact: {exc}") from exc


def _load_relation(path: str) -> RelationalArtifact:
    try:
        return load_relational_case(path)
    except ArtifactError as exc:
        raise SystemExit(f"invalid relational artifact: {exc}") from exc


def command_doctor(_: argparse.Namespace) -> int:
    _emit(doctor())
    return 0


def command_generate(args: argparse.Namespace) -> int:
    workload = (
        synthetic_control(args.seed)
        if args.synthetic
        else generate(
            GenerationConfig(
                seed=args.seed,
                max_rank=args.max_rank,
                max_nodes=args.max_nodes,
                dtype=args.dtype,
                input_mode=args.input_mode,
            )
        )
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


def command_outcomes(args: argparse.Namespace) -> int:
    """Compare execution terminal states without conflating them with output equality."""
    artifact = _load(args.case)
    left = run_backend(artifact.workload, args.left_backend)
    right = run_backend(artifact.workload, args.right_backend)
    result = compare_outcomes(left, right, artifact.workload.policy)
    record = result.to_dict()
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if result.classification == Classification.PASS else 1


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
    environment = None
    if args.environment_evidence:
        try:
            environment = load_environment(args.environment_evidence)
        except ArtifactError as exc:
            raise SystemExit(f"invalid environment evidence: {exc}") from exc
    comparison = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
    try:
        record = build_reproduction_evidence(artifact, comparison, environment)
    except ReadinessEvidenceError as exc:
        raise SystemExit(f"cannot create reproduction evidence: {exc}") from exc
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if record["verdict"] == "reproduced" else 1


def command_report(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    comparison = run_comparison(artifact.workload, artifact.comparison_mode, artifact.fault)
    json_path, markdown_path = write_report(args.output, artifact, comparison, args.case)
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
        record: dict[str, Any] = benchmark_jax(artifact.workload, args.measurements)
    else:
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
        record = {
            "backend": args.backend,
            "warmup_ms": warmup.duration_ms,
            "measurements": args.measurements,
            "median_ms": median(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
        }
    record["performance_version"] = "1.0"
    record["workload_sha256"] = artifact.workload.digest()
    if args.output:
        save_json(args.output, record)
    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            if not isinstance(baseline, dict):
                raise PerformanceArtifactError("baseline must be a JSON object")
            assessment = compare_benchmarks(baseline, record, args.max_relative_regression)
        except (OSError, json.JSONDecodeError, PerformanceArtifactError) as exc:
            raise SystemExit(f"invalid performance baseline: {exc}") from exc
        _emit({"benchmark": record, "baseline_assessment": assessment})
        return (
            1 if assessment["classification"] == Classification.PERFORMANCE_REGRESSION.value else 0
        )
    _emit(record)
    return 0


def command_cache(args: argparse.Namespace) -> int:
    """Compare bounded cold/warm compilation records without overclaiming cache internals."""
    try:
        cold = json.loads(Path(args.cold).read_text(encoding="utf-8"))
        warm = json.loads(Path(args.warm).read_text(encoding="utf-8"))
        if not isinstance(cold, dict) or not isinstance(warm, dict):
            raise PerformanceArtifactError("cache records must be JSON objects")
        result = compare_compilation_cache(cold, warm, args.max_relative_warm_compile)
    except (OSError, json.JSONDecodeError, PerformanceArtifactError) as exc:
        raise SystemExit(f"invalid cache evidence: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0 if result["classification"] == Classification.PASS.value else 1


def command_stability(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    try:
        record = check_stability(artifact.workload, args.backend, args.runs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if record["classification"] == Classification.PASS.value else 1


def command_readiness(args: argparse.Namespace) -> int:
    artifact = _load(args.case)
    try:
        reproductions = tuple(
            load_reproduction_evidence(path) for path in args.reproduction_evidence or ()
        )
        environments = tuple(load_environment(path) for path in args.environment_evidence or ())
        assessment = assess_upstream_readiness(artifact, reproductions, environments)
    except (ArtifactError, ReadinessEvidenceError) as exc:
        raise SystemExit(f"invalid readiness evidence: {exc}") from exc
    _emit(assessment)
    return 0 if assessment["ready"] or not args.strict else 1


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


def command_gradient(args: argparse.Namespace) -> int:
    result = run_gradient_case(generate_gradient_case(args.seed))
    _emit(result)
    return 0 if result["classification"] == "pass" else 1


def command_gradient_campaign(args: argparse.Namespace) -> int:
    _ensure_gpu_runtime()
    try:
        result = run_gradient_campaign(args.cases, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0 if not result["candidate_anomalies"] else 1


def command_gradient_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_gradient_campaigns(
            load_gradient_campaign(args.baseline), load_gradient_campaign(args.comparison)
        )
    except GradientArtifactError as exc:
        raise SystemExit(f"invalid gradient campaign artifact: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0


def command_campaign(args: argparse.Namespace) -> int:
    try:
        record = run_cpu_campaign(
            configure_campaign(
                args.profile,
                cases=args.cases,
                seed=args.seed,
                max_duration_s=args.max_duration_s,
                isolate=args.isolate,
                case_timeout_s=args.case_timeout_s,
                case_memory_mb=args.case_memory_mb,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if not record["candidate_anomalies"] else 1


def command_property_list(_: argparse.Namespace) -> int:
    _emit(
        {
            "corpus_version": "1.0",
            "kind": "metamorphic-property-catalog",
            "properties": [
                {
                    "identifier": property_definition.identifier,
                    "title": property_definition.title,
                    "description": property_definition.description,
                    "relation": property_definition.relation,
                }
                for property_definition in curated_properties()
            ],
        }
    )
    return 0


def _property_policy(args: argparse.Namespace) -> ComparisonPolicy | None:
    if args.rtol is None and args.atol is None:
        return None
    default = ComparisonPolicy()
    policy = ComparisonPolicy(
        identifier="property-cli-override-v1",
        rtol=default.rtol if args.rtol is None else args.rtol,
        atol=default.atol if args.atol is None else args.atol,
    )
    policy.validate()
    return policy


def command_property_run(args: argparse.Namespace) -> int:
    if args.backend.startswith("jax-gpu"):
        _ensure_gpu_runtime()
    try:
        result = run_curated_properties(
            seed=args.seed,
            backend=args.backend,
            identifiers=tuple(args.property) if args.property else None,
            policy=_property_policy(args),
        )
    except (MetamorphicPropertyError, ValueError) as exc:
        raise SystemExit(f"invalid curated property corpus: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0 if result["counts"][Classification.PASS.value] == len(result["properties"]) else 1


def command_property_campaign(args: argparse.Namespace) -> int:
    if args.backend.startswith("jax-gpu"):
        _ensure_gpu_runtime()
    try:
        result = run_property_campaign(
            cases=args.cases,
            seed=args.seed,
            backend=args.backend,
            identifiers=tuple(args.property) if args.property else None,
            policy=_property_policy(args),
        )
    except (MetamorphicPropertyError, ValueError) as exc:
        raise SystemExit(f"invalid property campaign: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0 if not result["candidate_observations"] else 1


def command_property_compare(args: argparse.Namespace) -> int:
    """Compare identical curated-property workloads across two backends."""
    try:
        result = compare_property_evidence(
            load_property_evidence(args.baseline), load_property_evidence(args.comparison)
        )
    except MetamorphicEvidenceError as exc:
        raise SystemExit(f"invalid property evidence comparison: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0


def command_property_coverage(args: argparse.Namespace) -> int:
    try:
        result = property_coverage(args.seed)
    except MetamorphicPropertyError as exc:
        raise SystemExit(f"invalid curated property corpus: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0


def command_property_sharpness(args: argparse.Namespace) -> int:
    try:
        report = evaluate_property_sharpness(args.property, args.seed, args.backend)
    except (ValueError, MetamorphicPropertyError) as exc:
        raise SystemExit(f"cannot evaluate property sharpness: {exc}") from exc
    record = report.to_dict()
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0


def command_attest_sign(args: argparse.Namespace) -> int:
    try:
        record = sign_attestation(args.subject, args.private_key)
        path = save_attestation(args.output, record)
    except ArtifactError as exc:
        raise SystemExit(f"cannot sign attestation: {exc}") from exc
    _emit({"attestation": str(path), "subject_sha256": record["subject_sha256"]})
    return 0


def command_attest_verify(args: argparse.Namespace) -> int:
    try:
        valid = verify_subject_attestation(
            load_attestation(args.attestation), args.subject, args.public_key
        )
    except ArtifactError as exc:
        raise SystemExit(f"cannot verify attestation: {exc}") from exc
    _emit({"attestation": args.attestation, "verified": valid})
    return 0 if valid else 1


def command_environment_capture(args: argparse.Namespace) -> int:
    evidence = capture_environment(args.label)
    path = save_environment(args.output, evidence)
    _emit({"environment": str(path), "sha256": evidence["sha256"]})
    return 0


def command_environment_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_environments(
            load_environment(args.primary), load_environment(args.secondary)
        )
    except ArtifactError as exc:
        raise SystemExit(f"invalid environment evidence: {exc}") from exc
    if args.output:
        save_json(args.output, result)
    _emit(result)
    return 0 if result["independent"] else 1


def command_runtime_list(_: argparse.Namespace) -> int:
    _emit(
        {
            "profiles": [
                {
                    "identifier": profile.identifier,
                    "environment": profile.environment,
                    "description": profile.description,
                }
                for profile in RUNTIME_PROFILES.values()
            ],
            "note": (
                "Profiles are applied only to a fresh child process and never modify this shell."
            ),
        }
    )
    return 0


def command_runtime_compare(args: argparse.Namespace) -> int:
    """Run a comparison in a clean process under an explicit, non-persistent profile."""
    try:
        profile = runtime_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    environment = os.environ.copy()
    environment.update(profile.environment)
    command = [sys.executable, "-m", "xlaxiom", "compare", "--case", args.case]
    if args.mode:
        command.extend(("--mode", args.mode))
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    try:
        child = json.loads(completed.stdout)
    except json.JSONDecodeError:
        child = {"raw_stdout": completed.stdout[-2000:], "raw_stderr": completed.stderr[-2000:]}
    record = {
        "runtime_profile_version": "1.0",
        "profile": profile.identifier,
        "environment": profile.environment,
        "child_exit_code": completed.returncode,
        "child": child,
    }
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return completed.returncode


def _reexec_x64_if_requested(args: argparse.Namespace, argv: list[str] | None) -> int | None:
    """Start a clean interpreter with JAX x64 enabled before it imports JAX."""
    if not getattr(args, "x64", False) or os.environ.get("XLAXIOM_X64_REEXEC") == "1":
        return None
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["XLAXIOM_X64_REEXEC"] = "1"
    command_args = sys.argv[1:] if argv is None else argv
    completed = subprocess.run(
        [sys.executable, "-m", "xlaxiom", *command_args], env=environment, check=False
    )
    return completed.returncode


def command_relation_create(args: argparse.Namespace) -> int:
    try:
        artifact = build_property_artifact(args.property, args.seed, args.backend)
    except MetamorphicPropertyError as exc:
        raise SystemExit(f"invalid curated property: {exc}") from exc
    path = save_relational_case(args.output, artifact)
    _emit(
        {
            "case": str(path),
            "sha256": artifact.to_dict()["sha256"],
            "identifier": artifact.identifier,
        }
    )
    return 0


def command_relation_run(args: argparse.Namespace) -> int:
    artifact = _load_relation(args.case)
    comparison = run_relational_comparison(artifact)
    record = comparison.to_dict()
    if args.output:
        save_json(args.output, record)
    _emit(record)
    return 0 if comparison.classification == Classification.PASS else 1


def command_relation_minimize(args: argparse.Namespace) -> int:
    artifact = _load_relation(args.case)
    try:
        outcome = minimize_relational(artifact, args.max_evaluations)
    except ArtifactError as exc:
        raise SystemExit(f"cannot minimize relational artifact: {exc}") from exc
    path = save_relational_case(args.output, outcome.artifact)
    _emit(
        {
            "case": str(path),
            "left": asdict(outcome.left),
            "right": asdict(outcome.right),
            "duration_ms": outcome.duration_ms,
        }
    )
    return 0


def command_relation_reproduce(args: argparse.Namespace) -> int:
    artifact = _load_relation(args.case)
    try:
        result = reproduce_relational_isolated(artifact, args.attempts, args.timeout_s)
    except ArtifactError as exc:
        raise SystemExit(f"cannot reproduce relational artifact: {exc}") from exc
    _emit(result)
    expected = artifact.expected_classification
    return 0 if result["reproduced"] and result["classification"] == expected else 1


def command_relation_report(args: argparse.Namespace) -> int:
    artifact = _load_relation(args.case)
    comparison = run_relational_comparison(artifact)
    json_path, markdown_path = write_relational_report(args.output, artifact, comparison)
    _emit(
        {
            "json": str(json_path),
            "markdown": str(markdown_path),
            "classification": comparison.classification.value,
        }
    )
    return 0


def command_matrix_run(args: argparse.Namespace) -> int:
    _emit(run_matrix(args.config, args.case, args.output))
    return 0


def command_matrix_relation(args: argparse.Namespace) -> int:
    _emit(run_relation_matrix(args.config, args.case, args.output))
    return 0


def command_matrix_doctor(_: argparse.Namespace) -> int:
    _emit({"cuda": cuda_capability(), "format": "JSON matrix config only in v0.3"})
    return 0


def command_gpu_doctor(_: argparse.Namespace) -> int:
    """Report real hardware capability without executing anything on a GPU."""
    devices = enumerate_physical_gpus()
    _emit(
        {
            "driver": driver_capability(),
            "runtime": cuda_capability(),
            "recommended_variant": recommended_variant(),
            "environment_risks": {
                "ld_library_path": library_path_risk(),
                "ordinal_ordering": ordinal_ordering_risk(devices),
            },
        }
    )
    return 0


def command_gpu_validate(args: argparse.Namespace) -> int:
    """Validate every visible GPU: capability gate, idleness gate, then bounded probe."""
    devices = enumerate_physical_gpus()
    report = validate_all(devices, allow_busy=args.allow_busy)
    if args.output:
        save_json(args.output, report)
    _emit(report)
    if args.report:
        path = write_gpu_report(args.report, report)
        _emit({"markdown": str(path)})
    return 0 if report["summary"]["unsupported"] == 0 else 2


def _gpu_interpreter() -> tuple[str, int] | None:
    """Find the isolated interpreter that owns a CUDA-enabled jaxlib for a visible GPU.

    Returns the interpreter path and the CUDA ordinal to expose to it, or ``None`` when
    no usable isolated environment exists.
    """
    try:
        devices = enumerate_physical_gpus()
    except Exception:
        return None
    if not devices:
        return None
    for device in devices:
        variant, _ = select_variant(device)
        if variant is None:
            continue
        interpreter = variant_python(variant)
        if not os.path.isfile(interpreter):
            continue
        if _interpreter_has_gpu(interpreter):
            return interpreter, device.ordinal
    return None


def _interpreter_has_gpu(interpreter: str) -> bool:
    """True when that interpreter can really initialize a JAX GPU device."""
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment[JAX_GPU_PREALLOCATE] = "false"
    try:
        completed = subprocess.run(
            [
                interpreter,
                "-c",
                "import sys,jax;sys.exit(0 if jax.devices('gpu') else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _active_jax_gpu_runtime() -> bool:
    """True only when this interpreter can initialize a JAX GPU device."""
    # This command may be invoked directly from the isolated interpreter. Set the
    # bounded-memory policy before its first JAX import, not only for re-exec children.
    os.environ[JAX_GPU_PREALLOCATE] = "false"
    try:
        import jax

        return bool(jax.devices("gpu"))
    except Exception:
        return False


def _ensure_gpu_runtime() -> None:
    """Re-execute a GPU command inside the isolated interpreter owning the CUDA jaxlib.

    The repository's primary environment is deliberately CPU-only, so a GPU campaign or
    historical study would otherwise honestly report ``unsupported_backend``. When an
    isolated CUDA environment exists, the command re-executes itself there with the
    shadowing host library path removed for that process only. The host driver, system
    CUDA, and every other service are never touched.
    """
    if _active_jax_gpu_runtime():
        return
    target = _gpu_interpreter()
    if target is None:
        return
    interpreter, ordinal = target
    interpreter_path = os.path.abspath(interpreter)
    if os.environ.get("XLAXIOM_GPU_REEXEC") == interpreter_path:
        raise SystemExit(
            "the isolated CUDA interpreter did not initialize a GPU device; "
            "stopping to avoid a re-execution loop"
        )
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment[JAX_GPU_PREALLOCATE] = "false"
    # The marker is an exact interpreter path, not a generic boolean. A manually
    # inherited legacy value such as ``1`` must never suppress a required re-exec.
    environment["XLAXIOM_GPU_REEXEC"] = interpreter_path
    environment["CUDA_VISIBLE_DEVICES"] = str(ordinal)
    result = subprocess.run(
        [interpreter, "-m", "xlaxiom", *sys.argv[1:]],
        env=environment,
        check=False,
    )
    raise SystemExit(result.returncode)


def command_gpu_campaign(args: argparse.Namespace) -> int:
    """Run the bounded real-GPU campaign only for devices that passed every probe."""
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard index must be non-negative and less than shard count")
    if args.resume and not args.ledger:
        raise SystemExit("--resume requires --ledger")
    _ensure_gpu_runtime()
    devices = enumerate_physical_gpus()
    report = validate_all(devices, allow_busy=args.allow_busy)
    passed = [
        (position, device)
        for position, (device, result) in enumerate(zip(devices, report["results"], strict=True))
        if result["state"] == DeviceState.PASS.value
    ]
    if not passed:
        _emit(
            {
                "campaign": "skipped",
                "reason": "no GPU passed every bounded probe; no campaign was run",
                "validation_summary": report["summary"],
            }
        )
        return 2
    if args.device is not None and not 0 <= args.device < len(passed):
        _emit(
            {
                "campaign": "skipped",
                "reason": (
                    f"device index {args.device} is out of range; {len(passed)} device(s) "
                    "passed every bounded probe in this process"
                ),
                "available": [
                    {
                        "index": index,
                        "name": device.name,
                        "ordinal": device.ordinal,
                        "nvidia_smi_index": device.smi_index,
                    }
                    for index, (_, device) in enumerate(passed)
                ],
                "hint": (
                    "A GPU re-exec pins CUDA_VISIBLE_DEVICES to one ordinal, so only that "
                    "device is visible. Re-run with CUDA_VISIBLE_DEVICES set to the "
                    "ordinal you want, or omit --device to use the visible device."
                ),
            }
        )
        return 2
    position, device = passed[args.device] if args.device is not None else passed[0]
    variant = report["results"][position]["variant"]
    # Record which device this run actually targeted. Under CUDA_VISIBLE_DEVICES the
    # runtime remaps the selected GPU to ordinal 0, so the campaign must state the
    # device it was asked to validate rather than inferring it from the visible set.
    shard_seed = args.seed + args.shard_index * args.cases
    shard = {
        "index": args.shard_index,
        "count": args.shard_count,
        "seed": shard_seed,
        "cases": args.cases,
    }
    run_identity = {
        "device_uuid_redacted": device.uuid_redacted,
        "variant": variant,
        "measurements": args.measurements,
    }
    campaign: dict[str, Any] | None = None
    if args.resume and args.ledger:
        try:
            campaign = completed_shard(load_ledger(args.ledger), run_identity, shard)
        except ArtifactError as exc:
            raise SystemExit(f"invalid campaign ledger: {exc}") from exc
    if campaign is None:
        campaign = run_gpu_campaign(
            device, variant or "cuda12", CampaignBudget(args.cases, shard_seed, args.measurements)
        )
        if args.ledger:
            try:
                save_shard(args.ledger, run_identity, shard, campaign)
            except ArtifactError as exc:
                raise SystemExit(f"cannot update campaign ledger: {exc}") from exc
        campaign["resumed"] = False
    else:
        campaign = {**campaign, "resumed": True}
    campaign["shard"] = shard
    campaign["requested_device"] = {
        "name": device.name,
        "ordinal": device.ordinal,
        "nvidia_smi_index": device.smi_index,
        "compute_capability": device.compute_capability,
        "uuid_redacted": device.uuid_redacted,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.output:
        save_json(args.output, campaign)
    _emit(campaign)
    return 0


def command_gpu_historical(args: argparse.Namespace) -> int:
    """Replay the documented JAX #40634 large-float32 gradient study on the GPU."""
    _ensure_gpu_runtime()
    report = validate_all(enumerate_physical_gpus(), allow_busy=args.allow_busy)
    if report["summary"]["pass"] == 0:
        _emit(
            {
                "study": "skipped",
                "reason": "no GPU passed every bounded probe; the study was not run",
                "validation_summary": report["summary"],
            }
        )
        return 2
    study = replay_jax_40634_on_gpu()
    if args.output:
        save_json(args.output, study)
    _emit(study)
    return 0


def write_gpu_report(path: str, report: dict[str, Any]) -> Path:
    """Write the portable Markdown hardware validation report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# XLAxiom CUDA hardware validation report",
        "",
        f"- Platform: `{report['host']['platform']}`",
        f"- Python: `{report['host']['python']}`",
        f"- Driver: `{report['environment'].get('nvidia_driver')}`",
        f"- CUDA: `{report['environment'].get('cuda_version')}`",
        "",
        "## Summary",
        "",
        f"- PASS: {report['summary']['pass']}",
        f"- UNSUPPORTED: {report['summary']['unsupported']}",
        f"- BLOCKED: {report['summary']['blocked']}",
        f"- INCONCLUSIVE: {report['summary']['inconclusive']}",
        "",
        "## Devices",
        "",
        "| CUDA ordinal | nvidia-smi index | Device | SM | Variant | State | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in report["results"]:
        device = result["device"]
        lines.append(
            f"| {device['ordinal']} | {device['smi_index']} | {device['name']} | "
            f"{device['compute_capability']} | {result['variant']} | {result['state']} | "
            f"{result['reason']} |"
        )
    lines.extend(["", "## Reproduction (Linux)", "", "```console", "uv sync --extra dev"])
    for result in report["results"]:
        if result["state"] != DeviceState.PASS.value:
            continue
        environment = result.get("environment", {})
        lines.append(
            f"env -u LD_LIBRARY_PATH "
            f"CUDA_VISIBLE_DEVICES={environment.get('ordinal')} "
            f"uv run xlaxiom gpu validate"
        )
    lines.extend(
        [
            "```",
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in report["notes"]],
            "",
            "No secrets, absolute personal paths, SSH keys, or raw GPU UUIDs are recorded.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _platform_string() -> str:
    """The host platform string, recorded for portable corpus evidence."""
    import platform as platform_module

    return platform_module.platform()


def command_corpus_build(args: argparse.Namespace) -> int:
    """Build the curated corpus from reproduced findings and metamorphic properties."""
    from .corpus import (
        CorpusEntry,
        build_entry,
        minimize_finding_for_admission,
        reproduce_isolated,
        save_corpus,
    )
    from .generation import GenerationConfig, generate

    entries: list[dict[str, Any]] = []

    # Known historical behavior is admitted as known-historical, never as a new finding.
    known = _load_known_historical()
    if known is not None:
        entries.append(known)

    if args.backend.startswith("jax-gpu"):
        _ensure_gpu_runtime()

    # Use the same complete reviewed registry as ``property run``. Execution or
    # environment failures remain evidence but are not admitted as property results.
    suite = run_curated_properties(seed=args.seed, backend=args.backend)
    for result in suite["properties"]:
        if result["triage"] == "execution_or_environment_failure":
            continue
        definition = result["property"]
        entries.append(
            build_entry(
                CorpusEntry(
                    identifier=f"metamorphic-{definition['identifier']}",
                    category="metamorphic-property",
                    title=definition["title"],
                    summary=definition["description"],
                    classification=result["comparison"]["classification"],
                    reproductions=1,
                    property_check=result,
                    environment={"platform": _platform_string(), "backend": args.backend},
                    caveats=(
                        "A violation is an observation about the relation, not a "
                        "root-cause claim about any single backend.",
                    ),
                )
            )
        )

    # Reproduced findings require independent reproduction in fresh processes. Without
    # that evidence nothing is admitted, so a short run may yield no findings at all.
    for seed in range(args.cases):
        workload = generate(GenerationConfig(seed=args.seed + seed, max_dim=6))
        from .pipeline import run_comparison

        observation = run_comparison(workload, "jax-jit")
        if observation.classification == Classification.PASS:
            continue
        minimization = minimize_finding_for_admission(
            workload,
            observation.classification,
            max_evaluations=args.minimize_evaluations,
        )
        reduced = minimization.workload
        reproduction = reproduce_isolated(reduced, attempts=args.reproductions)
        if (
            not reproduction["reproduced"]
            or reproduction["classification"] != observation.classification.value
        ):
            continue
        entries.append(
            build_entry(
                CorpusEntry(
                    identifier=f"finding-{reduced.identifier}",
                    category="reproduced-finding",
                    title=f"Reproduced {observation.classification.value}",
                    summary=observation.explanation,
                    workload=reduced.canonical_dict(),
                    workload_sha256=reduced.digest(),
                    classification=observation.classification.value,
                    policy=reduced.policy.identifier,
                    reproductions=reproduction["attempts"],
                    minimization={
                        "original_workload_sha256": workload.digest(),
                        "reduced_workload_sha256": reduced.digest(),
                        "stats": asdict(minimization.stats),
                        "reproduction": reproduction,
                    },
                    environment={"platform": _platform_string()},
                    caveats=(
                        "Differential evidence only; requires independent "
                        "reproduction before any root-cause claim.",
                    ),
                )
            )
        )

    path = save_corpus(args.output, entries)
    _emit({"corpus": str(path), "entry_count": len(entries)})
    return 0


def command_corpus_validate(args: argparse.Namespace) -> int:
    """Validate a corpus document and report its composition."""
    from .corpus import load_corpus

    document = load_corpus(args.corpus)
    _emit(
        {
            "corpus": str(args.corpus),
            "entry_count": document["entry_count"],
            "categories": document["categories"],
        }
    )
    return 0


def command_corpus_metamorphic(args: argparse.Namespace) -> int:
    """Check every reviewed declarative property on the selected backend."""
    if args.backend.startswith("jax-gpu"):
        _ensure_gpu_runtime()
    suite = run_curated_properties(seed=args.seed, backend=args.backend)
    if args.output:
        save_json(args.output, suite)
    _emit(suite)
    return 0 if suite["counts"][Classification.PASS.value] == len(suite["properties"]) else 1


def _load_known_historical() -> dict[str, Any] | None:
    """Admit a documented historical regression as known-historical."""
    from .corpus import CorpusEntry, build_entry

    source = "package:xlaxiom/corpus/historical/jax-40634.json"
    try:
        payload = json.loads(
            files("xlaxiom")
            .joinpath("corpus", "historical", "jax-40634.json")
            .read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return build_entry(
        CorpusEntry(
            identifier=str(payload.get("identifier", "jax-40634")),
            category="known-historical",
            title="JAX issue #40634 large-float32 gradient",
            summary=(
                "Documented upstream behavior. Reproduced locally and explicitly not "
                "reported as a new XLAxiom finding."
            ),
            references=(str(payload.get("issue", "")),),
            reproductions=0,
            environment={"recorded_from": source},
            caveats=("Known upstream behavior; never represented as a new defect.",),
        )
    )


def command_bundle_create(args: argparse.Namespace) -> int:
    try:
        _emit(create_bundle(args.case, args.output))
    except ArtifactError as exc:
        raise SystemExit(f"cannot create bundle: {exc}") from exc
    return 0


def command_bundle_validate(args: argparse.Namespace) -> int:
    try:
        _emit(validate_bundle(args.bundle))
    except ArtifactError as exc:
        raise SystemExit(f"invalid bundle: {exc}") from exc
    return 0


def command_bundle_run(args: argparse.Namespace) -> int:
    try:
        _emit(run_bundle(args.bundle, args.output))
    except ArtifactError as exc:
        raise SystemExit(f"invalid bundle: {exc}") from exc
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="xlaxiom", description="Evidence-first numerical differential testing for JAX/OpenXLA."
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="inspect JAX/backend capabilities")
    doctor_parser.set_defaults(func=command_doctor)
    attest_parser = commands.add_parser(
        "attest", help="create detached signatures with a maintainer-supplied private key"
    )
    attest_commands = attest_parser.add_subparsers(dest="attest_command", required=True)
    attest_sign = attest_commands.add_parser("sign", help="sign a release evidence file")
    attest_sign.add_argument("--subject", required=True)
    attest_sign.add_argument("--private-key", required=True)
    attest_sign.add_argument("--output", required=True)
    attest_sign.set_defaults(func=command_attest_sign)
    attest_verify = attest_commands.add_parser("verify", help="verify a detached release signature")
    attest_verify.add_argument("--attestation", required=True)
    attest_verify.add_argument("--subject", required=True)
    attest_verify.add_argument("--public-key", required=True)
    attest_verify.set_defaults(func=command_attest_verify)
    environment_parser = commands.add_parser(
        "environment", help="capture or compare redacted independent-host evidence"
    )
    environment_commands = environment_parser.add_subparsers(
        dest="environment_command", required=True
    )
    environment_capture = environment_commands.add_parser("capture", help="capture this host")
    environment_capture.add_argument(
        "--label", required=True, help="operator-supplied non-secret label"
    )
    environment_capture.add_argument("--output", required=True)
    environment_capture.set_defaults(func=command_environment_capture)
    environment_compare = environment_commands.add_parser(
        "compare", help="assess whether two records establish independent hosts"
    )
    environment_compare.add_argument("--primary", required=True)
    environment_compare.add_argument("--secondary", required=True)
    environment_compare.add_argument("--output")
    environment_compare.set_defaults(func=command_environment_compare)
    runtime_parser = commands.add_parser(
        "runtime", help="run isolated CPU thread/runtime profile comparisons"
    )
    runtime_commands = runtime_parser.add_subparsers(dest="runtime_command", required=True)
    runtime_list = runtime_commands.add_parser("list", help="list reviewed runtime profiles")
    runtime_list.set_defaults(func=command_runtime_list)
    runtime_compare = runtime_commands.add_parser(
        "compare", help="run a fresh child comparison under one profile"
    )
    runtime_compare.add_argument("--case", required=True)
    runtime_compare.add_argument("--profile", choices=tuple(RUNTIME_PROFILES), required=True)
    runtime_compare.add_argument("--mode", choices=("jax-eager", "jax-jit", "jax-eager-vs-jit"))
    runtime_compare.add_argument("--output")
    runtime_compare.set_defaults(func=command_runtime_compare)
    generate_parser = commands.add_parser(
        "generate", help="create a deterministic declarative case"
    )
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--max-rank", type=int, choices=range(1, 7), default=2)
    generate_parser.add_argument("--max-nodes", type=int, default=8)
    generate_parser.add_argument(
        "--dtype", choices=("float32", "float64", "int32", "int64"), default="float32"
    )
    generate_parser.add_argument(
        "--input-mode",
        choices=(
            "random",
            "zeros",
            "ones",
            "finite_edges",
            "extreme_finite",
            "signed_zero",
            "nonfinite",
        ),
        default="finite_edges",
        help="deterministic input-value strategy for generated workloads",
    )
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
    outcomes_parser = commands.add_parser(
        "outcomes", help="compare terminal execution outcomes without comparing values"
    )
    outcomes_parser.add_argument("--case", required=True)
    outcomes_parser.add_argument(
        "--left-backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-eager"
    )
    outcomes_parser.add_argument(
        "--right-backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-jit"
    )
    outcomes_parser.add_argument("--output")
    outcomes_parser.set_defaults(func=command_outcomes)
    minimize_parser = commands.add_parser("minimize", help="failure-preserving workload reduction")
    minimize_parser.add_argument("--case", required=True)
    minimize_parser.add_argument("--output", required=True)
    minimize_parser.add_argument("--max-evaluations", type=int, default=64)
    minimize_parser.set_defaults(func=command_minimize)
    reproduce_parser = commands.add_parser(
        "reproduce", help="reexecute and evaluate the stored predicate"
    )
    reproduce_parser.add_argument("--case", required=True)
    reproduce_parser.add_argument(
        "--environment-evidence",
        help="validated environment JSON to bind to this reproduction record",
    )
    reproduce_parser.add_argument(
        "--output", help="write checksummed reproduction evidence JSON for readiness review"
    )
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
    benchmark_parser.add_argument("--output", help="write a portable benchmark evidence record")
    benchmark_parser.add_argument(
        "--baseline", help="compare against a saved benchmark evidence record"
    )
    benchmark_parser.add_argument(
        "--max-relative-regression",
        type=float,
        default=0.2,
        help="maximum median slowdown fraction before reporting a regression",
    )
    benchmark_parser.set_defaults(func=command_benchmark)
    cache_parser = commands.add_parser(
        "cache", help="assess bounded cold/warm compilation evidence conservatively"
    )
    cache_parser.add_argument("--cold", required=True, help="first compilation JSON record")
    cache_parser.add_argument("--warm", required=True, help="second compilation JSON record")
    cache_parser.add_argument("--max-relative-warm-compile", type=float, default=0.1)
    cache_parser.add_argument("--output")
    cache_parser.set_defaults(func=command_cache)

    stability_parser = commands.add_parser(
        "stability", help="repeat one backend and detect output nondeterminism"
    )
    stability_parser.add_argument("--case", required=True)
    stability_parser.add_argument(
        "--backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-jit"
    )
    stability_parser.add_argument("--runs", type=int, default=3)
    stability_parser.add_argument("--output")
    stability_parser.set_defaults(func=command_stability)
    readiness_parser = commands.add_parser(
        "readiness", help="assess evidence needed before a human files an upstream issue"
    )
    readiness_parser.add_argument("--case", required=True)
    readiness_parser.add_argument(
        "--reproduction-evidence",
        action="append",
        help="checksummed JSON from `xlaxiom reproduce --output`; repeat for each host",
    )
    readiness_parser.add_argument(
        "--environment-evidence",
        action="append",
        help="validated host-environment JSON bound by a reproduction record; repeat as needed",
    )
    readiness_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero until every readiness check passes",
    )
    readiness_parser.set_defaults(func=command_readiness)
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
    gradient_parser = commands.add_parser(
        "gradient", help="run a bounded declarative analytic-gradient case"
    )
    gradient_parser.add_argument("--seed", type=int, default=0)
    gradient_parser.set_defaults(func=command_gradient)
    gradient_campaign = commands.add_parser(
        "gradient-campaign", help="run bounded analytic-gradient research cases"
    )
    gradient_campaign.add_argument("--cases", type=int, default=20)
    gradient_campaign.add_argument("--seed", type=int, default=0)
    gradient_campaign.add_argument("--output")
    gradient_campaign.set_defaults(func=command_gradient_campaign)
    gradient_compare = commands.add_parser(
        "gradient-compare", help="compare two gradient-campaign evidence artifacts"
    )
    gradient_compare.add_argument("--baseline", required=True)
    gradient_compare.add_argument("--comparison", required=True)
    gradient_compare.add_argument("--output")
    gradient_compare.set_defaults(func=command_gradient_compare)
    campaign_parser = commands.add_parser(
        "campaign", help="run a bounded trusted CPU NumPy/JAX differential campaign"
    )
    campaign_parser.add_argument("--profile", choices=tuple(CAMPAIGN_PROFILES), default="standard")
    campaign_parser.add_argument("--cases", type=int)
    campaign_parser.add_argument("--seed", type=int)
    campaign_parser.add_argument("--max-duration-s", type=float)
    campaign_parser.add_argument("--output")
    campaign_parser.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "run each case in a spawned child under a hard timeout, so a hung kernel or "
            "an out-of-memory failure contains one case instead of the whole campaign"
        ),
    )
    campaign_parser.add_argument(
        "--case-timeout-s", type=float, help="per-case timeout with --isolate"
    )
    campaign_parser.add_argument(
        "--case-memory-mb",
        type=int,
        help="per-case resident-memory limit with --isolate (16–65536 MiB)",
    )
    campaign_parser.set_defaults(func=command_campaign)
    property_parser = commands.add_parser(
        "property", help="inspect or run reviewed declarative metamorphic properties"
    )
    property_commands = property_parser.add_subparsers(dest="property_command", required=True)
    property_list = property_commands.add_parser("list", help="list the curated property corpus")
    property_list.set_defaults(func=command_property_list)
    property_run = property_commands.add_parser(
        "run", help="run all or selected curated metamorphic properties"
    )
    property_run.add_argument(
        "--property",
        action="append",
        choices=tuple(
            property_definition.identifier for property_definition in curated_properties()
        ),
        help="property identifier; repeat to select more than one (default: all)",
    )
    property_run.add_argument("--seed", type=int, default=0)
    property_run.add_argument("--backend", choices=PROPERTY_BACKENDS, default="jax-jit")
    property_run.add_argument("--rtol", type=float, help="explicit property relative tolerance")
    property_run.add_argument("--atol", type=float, help="explicit property absolute tolerance")
    property_run.add_argument("--x64", action="store_true", help="run in a clean JAX x64 process")
    property_run.add_argument("--output", help="JSON evidence path")
    property_run.set_defaults(func=command_property_run)
    property_campaign = property_commands.add_parser(
        "campaign", help="run deterministic property shards across a bounded seed range"
    )
    property_campaign.add_argument("--cases", type=int, default=20)
    property_campaign.add_argument("--seed", type=int, default=0)
    property_campaign.add_argument(
        "--property",
        action="append",
        choices=tuple(
            property_definition.identifier for property_definition in curated_properties()
        ),
        help="property identifier; repeat to select more than one (default: all)",
    )
    property_campaign.add_argument(
        "--x64", action="store_true", help="run in a clean JAX x64 process"
    )
    property_campaign.add_argument("--backend", choices=PROPERTY_BACKENDS, default="jax-jit")
    property_campaign.add_argument(
        "--rtol", type=float, help="explicit property relative tolerance"
    )
    property_campaign.add_argument(
        "--atol", type=float, help="explicit property absolute tolerance"
    )
    property_campaign.add_argument("--output", help="JSON evidence path")
    property_campaign.set_defaults(func=command_property_campaign)
    property_compare = property_commands.add_parser(
        "compare", help="compare identical property workloads across two backends"
    )
    property_compare.add_argument("--baseline", required=True)
    property_compare.add_argument("--comparison", required=True)
    property_compare.add_argument("--output", help="JSON comparison evidence path")
    property_compare.set_defaults(func=command_property_compare)
    property_coverage_parser = property_commands.add_parser(
        "coverage", help="report reviewed operations, dtypes, ranks, and edge-value coverage"
    )
    property_coverage_parser.add_argument("--seed", type=int, default=0)
    property_coverage_parser.add_argument("--output", help="JSON evidence path")
    property_coverage_parser.set_defaults(func=command_property_coverage)
    property_sharpness_parser = property_commands.add_parser(
        "sharpness", help="evaluate mutant kill rate and fault-detection power of a property"
    )
    property_sharpness_parser.add_argument(
        "--property",
        required=True,
        choices=tuple(
            property_definition.identifier for property_definition in curated_properties()
        ),
    )
    property_sharpness_parser.add_argument("--seed", type=int, default=42)
    property_sharpness_parser.add_argument("--backend", default="numpy")
    property_sharpness_parser.add_argument("--output", help="JSON evidence path")
    property_sharpness_parser.set_defaults(func=command_property_sharpness)
    relation_parser = commands.add_parser(
        "relation", help="create, minimize, reproduce, and report relational case artifacts"
    )
    relation_commands = relation_parser.add_subparsers(dest="relation_command", required=True)
    relation_create = relation_commands.add_parser(
        "create", help="materialize a curated property as a portable relational artifact"
    )
    relation_create.add_argument(
        "--property",
        required=True,
        choices=tuple(
            property_definition.identifier for property_definition in curated_properties()
        ),
    )
    relation_create.add_argument("--seed", type=int, default=0)
    relation_create.add_argument(
        "--backend", choices=("numpy", "jax-eager", "jax-jit"), default="jax-jit"
    )
    relation_create.add_argument("--output", required=True)
    relation_create.set_defaults(func=command_relation_create)
    relation_run = relation_commands.add_parser("run", help="execute a relational artifact")
    relation_run.add_argument("--case", required=True)
    relation_run.add_argument("--output")
    relation_run.set_defaults(func=command_relation_run)
    relation_minimize = relation_commands.add_parser(
        "minimize", help="failure-preserving minimization of both relation sides"
    )
    relation_minimize.add_argument("--case", required=True)
    relation_minimize.add_argument("--output", required=True)
    relation_minimize.add_argument("--max-evaluations", type=int, default=64)
    relation_minimize.set_defaults(func=command_relation_minimize)
    relation_reproduce = relation_commands.add_parser(
        "reproduce", help="reexecute a relational artifact in fresh processes"
    )
    relation_reproduce.add_argument("--case", required=True)
    relation_reproduce.add_argument("--attempts", type=int, default=2)
    relation_reproduce.add_argument("--timeout-s", type=float, default=60.0)
    relation_reproduce.set_defaults(func=command_relation_reproduce)
    relation_report = relation_commands.add_parser(
        "report", help="write JSON and Markdown evidence for a relational artifact"
    )
    relation_report.add_argument("--case", required=True)
    relation_report.add_argument("--output", required=True)
    relation_report.set_defaults(func=command_relation_report)
    matrix_parser = commands.add_parser("matrix", help="run isolated JAX version matrices")
    matrix_commands = matrix_parser.add_subparsers(dest="matrix_command", required=True)
    matrix_run = matrix_commands.add_parser("run")
    matrix_run.add_argument("--config", required=True)
    matrix_run.add_argument("--case", required=True)
    matrix_run.add_argument("--output", required=True)
    matrix_run.set_defaults(func=command_matrix_run)

    matrix_relation = matrix_commands.add_parser("relation")
    matrix_relation.add_argument("--config", required=True)
    matrix_relation.add_argument("--case", required=True)
    matrix_relation.add_argument("--output", required=True)
    matrix_relation.set_defaults(func=command_matrix_relation)
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
    corpus_parser = commands.add_parser(
        "corpus", help="build and validate the curated reproducible corpus"
    )
    corpus_commands = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_commands.add_parser("build", help="build the curated corpus")
    corpus_build.add_argument("--cases", type=int, default=8)
    corpus_build.add_argument("--seed", type=int, default=0)
    corpus_build.add_argument("--reproductions", type=int, default=2)
    corpus_build.add_argument("--minimize-evaluations", type=int, default=64)
    corpus_build.add_argument("--backend", choices=PROPERTY_BACKENDS, default="jax-jit")
    corpus_build.add_argument("--x64", action="store_true", help="enable JAX x64 before execution")
    corpus_build.add_argument("--output", default="corpus/findings.json")
    corpus_build.set_defaults(func=command_corpus_build)
    corpus_validate = corpus_commands.add_parser("validate", help="validate a corpus document")
    corpus_validate.add_argument("--corpus", required=True)
    corpus_validate.set_defaults(func=command_corpus_validate)
    corpus_metamorphic = corpus_commands.add_parser(
        "metamorphic", help="check every registered metamorphic property"
    )
    corpus_metamorphic.add_argument("--seed", type=int, default=0)
    corpus_metamorphic.add_argument("--backend", choices=PROPERTY_BACKENDS, default="jax-jit")
    corpus_metamorphic.add_argument(
        "--x64", action="store_true", help="enable JAX x64 before execution"
    )
    corpus_metamorphic.add_argument("--output")
    corpus_metamorphic.set_defaults(func=command_corpus_metamorphic)
    gpu_parser = commands.add_parser(
        "gpu", help="real CUDA hardware validation and bounded GPU campaigns"
    )
    gpu_commands = gpu_parser.add_subparsers(dest="gpu_command", required=True)
    gpu_doctor_parser = gpu_commands.add_parser(
        "doctor", help="report driver and runtime CUDA capability"
    )
    gpu_doctor_parser.set_defaults(func=command_gpu_doctor)
    gpu_validate_parser = gpu_commands.add_parser(
        "validate", help="run bounded probes on every visible GPU"
    )
    gpu_validate_parser.add_argument("--output", help="JSON evidence path")
    gpu_validate_parser.add_argument("--report", help="Markdown report path")
    gpu_validate_parser.add_argument(
        "--allow-busy",
        action="store_true",
        help=(
            "explicit operator override: probe a device that holds an external "
            "workload, only when ample free VRAM and idle utilization are present"
        ),
    )
    gpu_validate_parser.set_defaults(func=command_gpu_validate)
    gpu_campaign_parser = gpu_commands.add_parser(
        "campaign", help="run a small bounded CPU-versus-GPU campaign"
    )
    gpu_campaign_parser.add_argument("--cases", type=int, default=4)
    gpu_campaign_parser.add_argument("--seed", type=int, default=0)
    gpu_campaign_parser.add_argument("--measurements", type=int, default=5)
    gpu_campaign_parser.add_argument(
        "--device", type=int, default=None, help="index into the devices that passed"
    )
    gpu_campaign_parser.add_argument("--output", help="JSON evidence path")
    gpu_campaign_parser.add_argument("--allow-busy", action="store_true")
    gpu_campaign_parser.add_argument("--shard-index", type=int, default=0)
    gpu_campaign_parser.add_argument("--shard-count", type=int, default=1)
    gpu_campaign_parser.add_argument("--ledger", help="checksummed resumable GPU shard ledger")
    gpu_campaign_parser.add_argument(
        "--resume", action="store_true", help="reuse an exact completed ledger shard"
    )
    gpu_campaign_parser.set_defaults(func=command_gpu_campaign)
    gpu_historical_parser = gpu_commands.add_parser(
        "historical", help="replay JAX #40634 large-float32 gradients on the GPU"
    )
    gpu_historical_parser.add_argument("--output", help="JSON evidence path")
    gpu_historical_parser.add_argument("--allow-busy", action="store_true")
    gpu_historical_parser.set_defaults(func=command_gpu_historical)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    x64_exit_code = _reexec_x64_if_requested(args, argv)
    if x64_exit_code is not None:
        return x64_exit_code
    return int(args.func(args))


if __name__ == "__main__":
    # ``python -m xlaxiom.cli`` must behave exactly like ``python -m xlaxiom``. Without
    # this guard the module imports and exits 0 without doing anything, which looks like
    # success and silently produces no evidence.
    raise SystemExit(main())
