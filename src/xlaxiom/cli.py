"""The stable command-line interface for the CPU-first evidence pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from .bundle import create_bundle, run_bundle, validate_bundle
from .campaign import CampaignConfig, run_cpu_campaign
from .comparison import Classification
from .cuda import cuda_capability, driver_capability, recommended_variant
from .execution import benchmark_jax, doctor
from .generation import GenerationConfig, generate, synthetic_control
from .gpu_campaign import (
    CampaignBudget,
    replay_jax_40634_on_gpu,
    run_gpu_campaign,
)
from .gpu_validation import VARIANT_PYTHON, DeviceState, validate_all
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
            CampaignConfig(
                cases=args.cases,
                seed=args.seed,
                max_duration_s=args.max_duration_s,
                isolate=args.isolate,
                case_timeout_s=args.case_timeout_s,
            )
        )
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
        interpreter = VARIANT_PYTHON[variant]
        if not os.path.isfile(interpreter):
            continue
        if _interpreter_has_gpu(interpreter):
            return interpreter, device.ordinal
    return None


def _interpreter_has_gpu(interpreter: str) -> bool:
    """True when that interpreter can really initialize a JAX GPU device."""
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
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


def _ensure_gpu_runtime() -> None:
    """Re-execute a GPU command inside the isolated interpreter owning the CUDA jaxlib.

    The repository's primary environment is deliberately CPU-only, so a GPU campaign or
    historical study would otherwise honestly report ``unsupported_backend``. When an
    isolated CUDA environment exists, the command re-executes itself there with the
    shadowing host library path removed for that process only. The host driver, system
    CUDA, and every other service are never touched.
    """
    if os.environ.get("XLAXIOM_GPU_REEXEC") == "1":
        return
    try:
        import jax

        if jax.devices("gpu"):
            return
    except Exception:
        pass
    target = _gpu_interpreter()
    if target is None:
        return
    interpreter, ordinal = target
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment["XLAXIOM_GPU_REEXEC"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = str(ordinal)
    result = subprocess.run(
        [interpreter, "-m", "xlaxiom", *sys.argv[1:]],
        env=environment,
        check=False,
    )
    raise SystemExit(result.returncode)


def command_gpu_campaign(args: argparse.Namespace) -> int:
    """Run the bounded real-GPU campaign only for devices that passed every probe."""
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
    if args.device is not None and args.device >= len(passed):
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
    campaign = run_gpu_campaign(
        device, variant or "cuda12", CampaignBudget(args.cases, args.seed, args.measurements)
    )
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
        reproduce_isolated,
        run_metamorphic_suite,
        save_corpus,
    )
    from .generation import GenerationConfig, generate

    entries: list[dict[str, Any]] = []

    # Known historical behavior is admitted as known-historical, never as a new finding.
    known = _load_known_historical()
    if known is not None:
        entries.append(known)

    # Metamorphic properties are checked, and the check record is admitted only when it
    # actually ran to a conclusion.
    suite = run_metamorphic_suite()
    for result in suite["results"]:
        if result["state"] == "INCONCLUSIVE":
            continue
        entries.append(
            build_entry(
                CorpusEntry(
                    identifier=f"metamorphic-{result['identifier']}",
                    category="metamorphic-property",
                    title=result["identifier"],
                    summary=result["description"],
                    classification=result["classification"],
                    reproductions=1,
                    property_check=result,
                    environment={"platform": suite["platform"]},
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
        reproduction = reproduce_isolated(workload, attempts=args.reproductions)
        if not reproduction["reproduced"]:
            continue
        entries.append(
            build_entry(
                CorpusEntry(
                    identifier=f"finding-{workload.identifier}",
                    category="reproduced-finding",
                    title=f"Reproduced {observation.classification.value}",
                    summary=observation.explanation,
                    workload=workload.canonical_dict(),
                    workload_sha256=workload.digest(),
                    classification=observation.classification.value,
                    policy=workload.policy.identifier,
                    reproductions=reproduction["attempts"],
                    minimization={"reproduction": reproduction},
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
    """Check every registered metamorphic property against an independent oracle."""
    from .corpus import run_metamorphic_suite

    suite = run_metamorphic_suite()
    if args.output:
        save_json(args.output, suite)
    _emit(suite)
    return 0 if suite["summary"]["violated"] == 0 else 1


def _load_known_historical() -> dict[str, Any] | None:
    """Admit a documented historical regression as known-historical."""
    from .corpus import CorpusEntry, build_entry

    source = Path("corpus/historical/jax-40634.json")
    if not source.is_file():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
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
            environment={"recorded_from": source.as_posix()},
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
    campaign_parser.add_argument("--cases", type=int, default=20)
    campaign_parser.add_argument("--seed", type=int, default=0)
    campaign_parser.add_argument("--max-duration-s", type=float, default=180.0)
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
        "--case-timeout-s", type=float, default=60.0, help="per-case timeout with --isolate"
    )
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
    corpus_parser = commands.add_parser(
        "corpus", help="build and validate the curated reproducible corpus"
    )
    corpus_commands = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_build = corpus_commands.add_parser("build", help="build the curated corpus")
    corpus_build.add_argument("--cases", type=int, default=8)
    corpus_build.add_argument("--seed", type=int, default=0)
    corpus_build.add_argument("--reproductions", type=int, default=2)
    corpus_build.add_argument("--output", default="corpus/findings.json")
    corpus_build.set_defaults(func=command_corpus_build)
    corpus_validate = corpus_commands.add_parser("validate", help="validate a corpus document")
    corpus_validate.add_argument("--corpus", required=True)
    corpus_validate.set_defaults(func=command_corpus_validate)
    corpus_metamorphic = corpus_commands.add_parser(
        "metamorphic", help="check every registered metamorphic property"
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
    return int(args.func(args))


if __name__ == "__main__":
    # ``python -m xlaxiom.cli`` must behave exactly like ``python -m xlaxiom``. Without
    # this guard the module imports and exits 0 without doing anything, which looks like
    # success and silently produces no evidence.
    raise SystemExit(main())
