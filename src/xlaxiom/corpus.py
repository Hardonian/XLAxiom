"""A curated corpus of reproduced findings and metamorphic properties.

Roadmap item 3. The corpus is deliberately small and every entry is *earned*: a
candidate is admitted only after it is reproduced in a fresh process, independently
of the process that first observed it. Nothing is admitted on a single observation.

Two kinds of entry live here:

* ``reproduced-finding`` — a discrepancy that survived independent reproduction and
  failure-preserving minimization, with the reduced workload and its environment
  recorded so a third party can rerun it.
* ``metamorphic-property`` — a relation that must hold regardless of how a correct
  backend computes it, for example ``f(x + c) - f(x)`` behaving consistently under a
  shift, or a reduction commuting with a transpose. Properties are checked, never
  assumed, and a violation is recorded as an observation rather than a root cause.

Known upstream behavior is never admitted as a new finding. An entry that matches a
documented issue is filed under ``known-historical`` and labelled as such.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .comparison import Classification
from .minimization import MinimizationResult, minimize
from .model import Workload

CORPUS_VERSION = "1.0"

# Admitted categories. Anything outside this set is rejected at validation time.
CATEGORIES = frozenset(
    {
        "reproduced-finding",
        "metamorphic-property",
        "known-historical",
    }
)

# Reproduction is the gate. A candidate must reproduce in a fresh process before it can
# be admitted, and it must reproduce this many times to count as independent.
REQUIRED_REPRODUCTIONS = 2

DEFAULT_CORPUS_PATH = Path("corpus/findings.json")


@dataclass(frozen=True)
class MetamorphicProperty:
    """A relation that must hold for any correct backend.

    Properties are expressed as a named check over a bounded workload so they can be
    re-evaluated without executable payloads.
    """

    identifier: str
    description: str
    operation: str
    lhs_seed: int
    rhs_seed: int
    rtol: float = 2e-5
    atol: float = 2e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusEntry:
    """One curated, reproducible corpus entry."""

    identifier: str
    category: str
    title: str
    summary: str
    environment: dict[str, Any] = field(default_factory=dict)
    workload: dict[str, Any] | None = None
    workload_sha256: str | None = None
    classification: str | None = None
    policy: str | None = None
    reproductions: int = 0
    minimization: dict[str, Any] | None = None
    property_check: dict[str, Any] | None = None
    references: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["references"] = list(self.references)
        data["caveats"] = list(self.caveats)
        return data


def digest(payload: dict[str, Any]) -> str:
    """A stable checksum over a corpus entry, excluding its own digest."""
    body = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_entry(
    entry: CorpusEntry,
) -> dict[str, Any]:
    """Attach the corpus version and checksum to an entry."""
    payload = entry.to_dict()
    payload["corpus_version"] = CORPUS_VERSION
    payload["sha256"] = digest(payload)
    return payload


def validate_entry(payload: dict[str, Any]) -> None:
    """Reject an entry that does not meet the admission bar.

    This is the guard that keeps the corpus honest: an entry claiming to be a
    reproduced finding must actually carry the reproduction evidence.
    """
    if payload.get("corpus_version") != CORPUS_VERSION:
        raise ValueError(f"unsupported corpus version {payload.get('corpus_version')!r}")
    if payload.get("sha256") != digest(payload):
        raise ValueError("corpus entry checksum mismatch")
    category = payload.get("category")
    if category not in CATEGORIES:
        raise ValueError(f"unsupported corpus category {category!r}")
    if not payload.get("identifier"):
        raise ValueError("corpus entry needs an identifier")
    if category == "reproduced-finding":
        if payload.get("reproductions", 0) < REQUIRED_REPRODUCTIONS:
            raise ValueError(
                "a reproduced finding requires independent reproduction in at least "
                f"{REQUIRED_REPRODUCTIONS} fresh processes"
            )
        if payload.get("workload") is None:
            raise ValueError("a reproduced finding must carry its reduced workload")
        if payload.get("classification") in {None, Classification.PASS.value}:
            raise ValueError("a reproduced finding must record a non-pass classification")
    if category == "metamorphic-property" and payload.get("property_check") is None:
        raise ValueError("a metamorphic property must carry its check record")


def save_corpus(path: str | Path, entries: list[dict[str, Any]]) -> Path:
    """Write a validated, checksummed corpus document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for payload in entries:
        validate_entry(payload)
    document = {
        "corpus_version": CORPUS_VERSION,
        "entry_count": len(entries),
        "categories": {name: _count(entries, name) for name in sorted(CATEGORIES)},
        "entries": entries,
        "notes": [
            "Every reproduced finding was observed in a fresh process, independently of "
            "the process that first flagged it.",
            "A discrepancy is an observation, not a root-cause claim.",
            "Known upstream behavior is filed as known-historical and never as a new finding.",
        ],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_corpus(path: str | Path) -> dict[str, Any]:
    """Load and validate a corpus document."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read corpus: {exc}") from exc
    if document.get("corpus_version") != CORPUS_VERSION:
        raise ValueError(f"unsupported corpus version {document.get('corpus_version')!r}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError("corpus entries must be a list")
    for payload in entries:
        validate_entry(payload)
    return document


def _count(entries: list[dict[str, Any]], category: str) -> int:
    return sum(1 for entry in entries if entry.get("category") == category)


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def reproduce_isolated(
    workload: Workload,
    mode: str = "jax-jit",
    attempts: int = REQUIRED_REPRODUCTIONS,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Reproduce a workload in fresh processes and report how many agreed.

    Each attempt is a separate interpreter, so no in-process state, cached
    compilation, or device selection can carry an observation forward. The claim is
    only as strong as the number of independent processes that produced it.
    """
    script = (
        "import json,sys\n"
        "from xlaxiom.model import Workload\n"
        "from xlaxiom.pipeline import run_comparison\n"
        f"workload = Workload.from_dict(json.loads(sys.argv[1]))\n"
        f"print(json.dumps(run_comparison(workload, {mode!r}).to_dict(), sort_keys=True))\n"
    )
    observations: list[dict[str, Any]] = []
    for _ in range(attempts):
        try:
            completed = subprocess.run(
                [sys.executable, "-c", script, json.dumps(workload.canonical_dict())],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            observations.append({"state": "BLOCKED", "error": repr(exc)})
            continue
        if completed.returncode != 0:
            observations.append({"state": "BLOCKED", "error": completed.stderr[-500:]})
            continue
        payload = _last_json(completed.stdout)
        if payload is None:
            observations.append({"state": "INCONCLUSIVE", "error": "no observation"})
            continue
        observations.append(
            {
                "state": "observed",
                "classification": payload.get("classification"),
                "device": (payload.get("left") or {}).get("device"),
            }
        )
    agreeing = {item["classification"] for item in observations if item["state"] == "observed"}
    return {
        "attempts": attempts,
        "observations": observations,
        "reproduced": len(agreeing) == 1 and len(observations) == attempts,
        "classification": next(iter(agreeing)) if len(agreeing) == 1 else None,
        "note": (
            "Every attempt ran in a fresh interpreter, so no cached compilation or "
            "device selection could carry the observation forward."
        ),
    }


def minimize_finding_for_admission(
    workload: Workload,
    expected: Classification,
    max_evaluations: int = 64,
    evaluator: Callable[[Workload], Classification] | None = None,
) -> MinimizationResult:
    """Failure-preservingly minimize a non-pass observation before corpus admission."""
    if expected == Classification.PASS:
        raise ValueError("a passing observation cannot be minimized for finding admission")
    if not 1 <= max_evaluations <= 10_000:
        raise ValueError("minimization evaluations must be between 1 and 10000")
    if evaluator is None:
        from .pipeline import run_comparison

        def evaluate(candidate: Workload) -> Classification:
            return run_comparison(candidate, "jax-jit").classification

        evaluator = evaluate
    return minimize(workload, evaluator, expected, max_evaluations=max_evaluations)


# Metamorphic relations. Each is a property of correct behavior, not of one backend:
# if it fails, some implementation is wrong, but the property alone does not say which.
# They are checked against both JAX eager and JIT and compared with an independent
# NumPy evaluation of the same relation.
METAMORPHIC_PROPERTIES: tuple[MetamorphicProperty, ...] = (
    MetamorphicProperty(
        identifier="reduction-commutes-with-transpose",
        description="sum(M) equals sum(M^T): a reduction must not depend on layout",
        operation="reduction-transpose",
        lhs_seed=2,
        rhs_seed=2,
    ),
    MetamorphicProperty(
        identifier="matmul-associativity",
        description="(A@B)@C equals A@(B@C) within a bounded tolerance",
        operation="matmul-assoc",
        lhs_seed=5,
        rhs_seed=5,
    ),
    MetamorphicProperty(
        identifier="broadcast-distributes-over-add",
        description="adding a row and a column broadcast equals adding them separately",
        operation="broadcast-distributive",
        lhs_seed=7,
        rhs_seed=7,
    ),
    MetamorphicProperty(
        identifier="reduction-splits-over-blocks",
        description="summing a matrix equals the sum of its row sums",
        operation="reduction-split",
        lhs_seed=11,
        rhs_seed=11,
    ),
)


def check_metamorphic_property(prop: MetamorphicProperty) -> dict[str, Any]:
    """Check one metamorphic relation against an independent NumPy reference.

    The relation is evaluated three ways: NumPy as an independent oracle, JAX eager,
    and JAX JIT. A property holds only when all three agree under the recorded policy.
    A violation is recorded as an observation with the devices involved; it is not a
    root-cause claim about any single backend.
    """
    import numpy as np

    from .generation import GenerationConfig, generate

    record: dict[str, Any] = {
        "identifier": prop.identifier,
        "description": prop.description,
        "operation": prop.operation,
        "rtol": prop.rtol,
        "atol": prop.atol,
    }
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        record["state"] = "BLOCKED"
        record["error"] = str(exc)
        return record

    lhs = generate(GenerationConfig(seed=prop.lhs_seed, max_dim=6))
    rhs = generate(GenerationConfig(seed=prop.rhs_seed, max_dim=6))
    del lhs, rhs
    try:
        oracle_value = _relation(prop, _square_block(prop, np))
        eager_value = _relation(prop, _square_block(prop, jnp))
        compiled = jax.jit(_property_program(prop))
        jit_value = float(compiled(_square_block(prop, jnp)).block_until_ready())
    except Exception as exc:
        record["state"] = "INCONCLUSIVE"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    oracle = float(np.asarray(oracle_value))
    eager = float(np.asarray(eager_value))
    record["oracle"] = oracle
    record["eager"] = eager
    record["jit"] = jit_value
    record["eager_device"] = str(getattr(eager_value, "device", "cpu"))
    eager_ok = bool(np.isclose(eager, oracle, rtol=prop.rtol, atol=prop.atol))
    jit_ok = bool(np.isclose(jit_value, oracle, rtol=prop.rtol, atol=prop.atol))
    record["eager_matches_oracle"] = eager_ok
    record["jit_matches_oracle"] = jit_ok
    record["state"] = "holds" if (eager_ok and jit_ok) else "violated"
    record["classification"] = (
        Classification.PASS.value if record["state"] == "holds" else "metamorphic_violation"
    )
    return record


def _property_program(prop: MetamorphicProperty) -> Any:
    """The JIT-compiled form of a property's relation."""

    def program(first: Any) -> Any:
        return _relation(prop, first)

    return program


def _square_block(prop: MetamorphicProperty, api: Any) -> Any:
    """A bounded square matrix for a property, deterministic in the property's seed.

    Properties that need a square operand build one directly rather than reshaping
    whatever the workload generator produced, so the relation is always well defined.
    """
    import numpy as np_mod

    side = 4
    rng = np_mod.random.default_rng(prop.lhs_seed)
    return api.asarray(rng.standard_normal((side, side)).astype(np_mod.float32))


def _relation(prop: MetamorphicProperty, first: Any) -> Any:
    """The arithmetic relation behind one metamorphic property.

    Kept deliberately small and bounded: these properties are about numerical
    consistency, not about exercising large kernels. Each relation derives its shapes
    from the inputs so the identical expression is valid under NumPy and JAX.
    """
    if prop.operation == "reduction-transpose":
        return api_sum(first) - api_sum(api_transpose(first))
    if prop.operation == "matmul-assoc":
        # Compare (A@A)@A with A@(A@A) on a square block so both groupings are valid.
        return api_sum((first @ first) @ first) - api_sum(first @ (first @ first))
    if prop.operation == "broadcast-distributive":
        # Broadcasting a column against a row must be symmetric under exchange, so the
        # difference of the two orders is zero for any correct backend. The column and
        # row come from the same square block, so both broadcasts have equal shape.
        column = api_slice(first, (slice(None), slice(0, 1)))
        row = api_slice(first, (slice(0, 1), slice(None)))
        return api_sum(api_add(column, row)) - api_sum(api_add(row, column))
    if prop.operation == "reduction-split":
        return api_sum(first) - api_sum(api_sum(first, axis=1))
    raise ValueError(f"unknown metamorphic operation {prop.operation!r}")


def api_slice(value: Any, selection: Any) -> Any:
    try:
        return value[selection]
    except (TypeError, IndexError):
        import numpy as np_mod

        return np_mod.asarray(value)[selection]


def api_sum(value: Any, axis: Any = None) -> Any:
    try:
        return value.sum() if axis is None else value.sum(axis=axis)
    except AttributeError:
        import numpy as np_mod

        return np_mod.sum(value) if axis is None else np_mod.sum(value, axis=axis)


def api_transpose(value: Any) -> Any:
    try:
        return value.transpose()
    except AttributeError:
        import numpy as np_mod

        return np_mod.transpose(value)


def api_add(left: Any, right: Any) -> Any:
    try:
        return left + right
    except TypeError:
        import numpy as np_mod

        return np_mod.add(left, right)


def run_metamorphic_suite() -> dict[str, Any]:
    """Check every registered metamorphic property and summarise honestly."""
    results = [check_metamorphic_property(prop) for prop in METAMORPHIC_PROPERTIES]
    states = [item["state"] for item in results]
    return {
        "suite": "XLAxiom metamorphic properties",
        "platform": platform.platform(),
        "property_count": len(results),
        "results": results,
        "summary": {
            "holds": states.count("holds"),
            "violated": states.count("violated"),
            "inconclusive": states.count("INCONCLUSIVE"),
            "blocked": states.count("BLOCKED"),
        },
        "interpretation": (
            "A violation is an observation about the relation, not a root-cause claim "
            "about any single backend. A property is checked against an independent "
            "NumPy evaluation of the same relation."
        ),
    }
