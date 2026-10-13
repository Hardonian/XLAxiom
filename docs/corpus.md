# Curated corpus

XLAxiom's curated corpus is a small, reviewed set of declarative numerical relations
and, in future, evidence-admitted findings. It is not a collection of arbitrary Python
examples: the packaged JSON manifest can select only fixed, audited builders in
`xlaxiom.metamorphic`. The manifest is validated against that closed registry before a
property is run.

## Run the current property corpus

```console
uv run xlaxiom property list
uv run xlaxiom property run --seed 42 --backend jax-jit --x64 --output results/properties.json
uv run xlaxiom property campaign --cases 20 --backend jax-jit --x64
uv run xlaxiom property coverage
uv run xlaxiom property run --property matmul-transpose-v1 --backend numpy
uv run xlaxiom relation create --property matmul-transpose-v1 --output relation.json
uv run xlaxiom relation reproduce --case relation.json
```

Each result records both generated workload checksums, the execution backend, numerical
policy, and the ordinary XLAxiom comparison classification. A passing relation only
confirms the recorded bounded input and backend. A discrepancy is an observation to
review, minimize, and reproduce; it does not diagnose a compiler defect.

## v1 metamorphic properties

| Identifier | Relation |
| --- | --- |
| `add-commutativity-v1` | `x + y` equals `y + x`. |
| `transpose-involution-v1` | Transposing a rank-two tensor twice restores it. |
| `reshape-roundtrip-v1` | Flattening and restoring a rank-two tensor preserves it. |
| `matmul-transpose-v1` | `(x @ y).T` equals `y.T @ x.T`. |
| `negation-involution-v1` | Negating twice preserves values and signed zero. |
| `subtraction-as-addition-v1` | `x - y` equals `x + (-y)`. |
| `absolute-idempotence-v1` | `abs(x)` equals `abs(abs(x))`. |
| `integer-add-commutativity-v1` | `int32 x + y` equals `y + x`. |
| `float64-add-commutativity-v1` | `float64 x + y` equals `y + x` in an explicit x64 process. |
| `mixed-dtype-add-commutativity-v1` | `int32 x + float32 y` equals the reversed operands. |
| `signed-zero-negation-involution-v1` | Double negation preserves both zero signs. |
| `nonfinite-absolute-idempotence-v1` | `abs(x)` equals `abs(abs(x))` for NaN and infinities. |
| `extreme-finite-absolute-idempotence-v1` | `abs(x)` equals `abs(abs(x))` at normal/subnormal finite extremes. |

The corpus manifest is [`src/xlaxiom/corpus/curated-v1.json`](../src/xlaxiom/corpus/curated-v1.json).
It carries no executable payload and has no general `builder` or import field.

## Finding admission

No newly discovered compiler finding is admitted in corpus v1. The historical JAX
#40634 adapter remains a labelled historical study rather than a new discovery.

A proposed finding must include all of the following before admission:

1. A non-synthetic case artifact with the recorded numerical policy.
2. Failure-preserving minimization and its original workload digest.
3. A fresh-process reproduction verdict.
4. Reproduction in an independently configured environment, with versions and device
   evidence captured.

Use the normal evidence workflow—`minimize`, `reproduce`, `report`, and optionally a
validated `bundle`—before proposing an entry. A failed property can be materialized as
a relational artifact, minimized on both sides, reproduced in fresh processes, reported,
and bundled without executable payloads. This preserves the distinction between an
observed discrepancy and a supported root-cause claim.
