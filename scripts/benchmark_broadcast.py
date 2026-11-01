#!/usr/bin/env python3
"""Micro-benchmarks for XLAxiom hot paths.

Measures throughput and per-operation latency across:
1. Matching shapes (fast-path equality)
2. Equal-rank broadcasting with singleton expansions
3. Trailing-dimension differing-rank broadcasting
4. Multi-tensor chain broadcasting
5. Workload validation throughput
6. BoundExecutionPlan evaluation throughput
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from xlaxiom.execution import compile_execution_plan
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.model import _broadcast_shape


def benchmark_broadcasting(iterations: int = 100_000) -> dict[str, Any]:
    """Benchmark shape broadcasting across canonical tensor shapes."""
    cases: list[tuple[str, tuple[tuple[int, ...], ...]]] = [
        ("identical_rank_3", ((64, 128, 256), (64, 128, 256))),
        ("equal_rank_broadcast", ((64, 1, 256), (1, 128, 256))),
        ("differing_rank_broadcast", ((64, 128, 256), (256,))),
        ("multi_operand_chain", ((64, 1, 256), (1, 128, 1), (1, 1, 256))),
        ("scalar_broadcast", ((), (64, 128))),
    ]

    results: dict[str, Any] = {}
    for name, shapes in cases:
        # Warmup
        for _ in range(1_000):
            _broadcast_shape(*shapes)

        start = time.perf_counter_ns()
        for _ in range(iterations):
            _broadcast_shape(*shapes)
        elapsed_ns = time.perf_counter_ns() - start

        ns_per_op = elapsed_ns / iterations
        ops_per_sec = iterations / (elapsed_ns / 1e9)
        results[name] = {
            "shapes": [list(s) for s in shapes],
            "result": list(_broadcast_shape(*shapes)),
            "iterations": iterations,
            "total_ms": round(elapsed_ns / 1e6, 3),
            "ns_per_op": round(ns_per_op, 2),
            "ops_per_sec": round(ops_per_sec, 0),
        }
    return results


def benchmark_workload_validation(iterations: int = 10_000) -> dict[str, Any]:
    """Benchmark Workload.validate() throughput across synthetic workloads."""
    workloads = [generate(GenerationConfig(seed=i, max_nodes=10, max_rank=3)) for i in range(1, 6)]

    # Warmup
    for wl in workloads:
        wl.validate()

    start = time.perf_counter_ns()
    total_validations = iterations * len(workloads)
    for _ in range(iterations):
        for wl in workloads:
            wl.validate()
    elapsed_ns = time.perf_counter_ns() - start

    ns_per_op = elapsed_ns / total_validations
    ops_per_sec = total_validations / (elapsed_ns / 1e9)
    return {
        "workload_count": len(workloads),
        "total_validations": total_validations,
        "total_ms": round(elapsed_ns / 1e6, 3),
        "ns_per_op": round(ns_per_op, 2),
        "ops_per_sec": round(ops_per_sec, 0),
    }


def benchmark_plan_evaluation(iterations: int = 25_000) -> dict[str, Any]:
    """Benchmark BoundExecutionPlan.evaluate() throughput."""
    workload = generate(GenerationConfig(seed=42, max_nodes=8, max_rank=2))
    plan = compile_execution_plan(workload)
    bound = plan.bind(np)
    inputs = [np.ones(spec.shape, dtype=np.dtype(spec.dtype)) for spec in workload.inputs]

    # Warmup
    for _ in range(100):
        bound.evaluate(inputs)

    start = time.perf_counter_ns()
    for _ in range(iterations):
        bound.evaluate(inputs)
    elapsed_ns = time.perf_counter_ns() - start

    ns_per_op = elapsed_ns / iterations
    ops_per_sec = iterations / (elapsed_ns / 1e9)
    return {
        "nodes": len(workload.nodes),
        "iterations": iterations,
        "total_ms": round(elapsed_ns / 1e6, 3),
        "ns_per_op": round(ns_per_op, 2),
        "ops_per_sec": round(ops_per_sec, 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="XLAxiom Hot Path Micro-Benchmarks")
    parser.add_argument(
        "--iterations", type=int, default=100_000, help="Iterations for broadcast benchmarks"
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    broadcast_results = benchmark_broadcasting(args.iterations)
    validation_results = benchmark_workload_validation(args.iterations // 10)
    evaluation_results = benchmark_plan_evaluation(args.iterations // 4)

    full_results = {
        "broadcasting": broadcast_results,
        "workload_validation": validation_results,
        "plan_evaluation": evaluation_results,
    }

    if args.json:
        print(json.dumps(full_results, indent=2))
        return

    print("=" * 70)
    print("XLAxiom Hot Path Micro-Benchmarks")
    print("=" * 70)
    print("\n1. Shape Broadcasting Throughput:")
    print(f"   {'Case':<28} {'Latency':<12} {'Throughput':<18}")
    print("   " + "-" * 58)
    for name, data in broadcast_results.items():
        print(
            f"   {name:<28} {data['ns_per_op']:>7.1f} ns/op   {data['ops_per_sec']:>10,.0f} ops/sec"
        )

    print("\n2. Workload Validation Throughput (O(N) optimized):")
    val_count = validation_results["total_validations"]
    val_ms = validation_results["total_ms"]
    print(f"   {val_count:,} validations in {val_ms} ms")
    print(f"   Latency:    {validation_results['ns_per_op']:,.1f} ns/validation")
    print(f"   Throughput: {validation_results['ops_per_sec']:,.0f} validations/sec")

    print("\n3. Bound Execution Plan Evaluation Throughput:")
    eval_iter = evaluation_results["iterations"]
    eval_nodes = evaluation_results["nodes"]
    eval_ms = evaluation_results["total_ms"]
    print(f"   {eval_iter:,} evaluations ({eval_nodes} nodes) in {eval_ms} ms")
    print(f"   Latency:    {evaluation_results['ns_per_op']:,.1f} ns/eval")
    print(f"   Throughput: {evaluation_results['ops_per_sec']:,.0f} evals/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()
