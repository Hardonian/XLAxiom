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
uv run xlaxiom property campaign --cases 20 --backend jax-jit --x64 --output results/campaign.json
uv run xlaxiom property coverage
uv run xlaxiom property run --property matmul-transpose-v1 --seed 42 --backend jax-jit --x64 --output results/cpu.json
# On a separately validated GPU host, use the same property and seed:
xlaxiom property run --property matmul-transpose-v1 --seed 42 --backend jax-gpu-jit --x64 --output results/gpu.json
# Bring both evidence files together before comparing them:
uv run xlaxiom property compare --baseline results/cpu.json --comparison results/gpu.json --output results/comparison.json
uv run xlaxiom relation create --property matmul-transpose-v1 --output relation.json
uv run xlaxiom relation reproduce --case relation.json
```

`property compare` compares two existing single-property or corpus-evidence artifacts;
it does not create `cpu.json` or `gpu.json`, and campaign evidence is intentionally not
accepted as a comparison input. The two artifacts must use the same requested property
ordering and seed so their workload identities can be checked before classifications are
compared.

Each result records both generated workload checksums, the execution backend, numerical
policy, and the ordinary XLAxiom comparison classification. A passing relation only
confirms the recorded bounded input and backend. A discrepancy is an observation to
review, minimize, and reproduce; it does not diagnose a compiler defect.

Property execution supports NumPy, JAX CPU eager/JIT, JAX GPU eager/JIT, and JAX TPU
eager/JIT backends. Accelerator requests are mandatory placement requests: an absent
GPU or TPU is `unsupported_backend`, never a CPU fallback. Cross-backend comparison
requires matching seeds, property sets, ordering, and both workload SHA-256 identities.

## v1 metamorphic properties

| Identifier | Relation |
| --- | --- |
| `add-commutativity-v1` | `x + y` and `y + x` produce equivalent elementwise outputs. |
| `transpose-involution-v1` | Transposing a rank-two tensor twice restores the original tensor. |
| `reshape-roundtrip-v1` | Flattening and restoring a rank-two tensor preserves every element. |
| `matmul-transpose-v1` | `(x @ y).T` and `y.T @ x.T` produce equivalent outputs. |
| `negation-involution-v1` | Negating a tensor twice preserves values and signed zero. |
| `subtraction-as-addition-v1` | `x - y` and `x + (-y)` produce equivalent elementwise outputs. |
| `absolute-idempotence-v1` | `abs(x)` and `abs(abs(x))` produce equivalent outputs. |
| `integer-add-commutativity-v1` | `int32 x + y` and `y + x` produce equivalent elementwise outputs. |
| `float64-add-commutativity-v1` | `float64 x + y` and `y + x` produce equivalent outputs in an explicit JAX x64 process. |
| `mixed-dtype-add-commutativity-v1` | `int32 x + float32 y` and `float32 y + int32 x` use same-operation promotion semantics consistently. |
| `signed-zero-negation-involution-v1` | Negating signed-zero inputs twice preserves both values and zero signs. |
| `nonfinite-absolute-idempotence-v1` | `abs(x)` and `abs(abs(x))` agree for NaN, infinities, and signed zero under the explicit policy. |
| `extreme-finite-absolute-idempotence-v1` | `abs(x)` and `abs(abs(x))` agree for finite values spanning normal and subnormal boundaries. |
| `maximum-commutativity-v1` | `maximum(x, y)` and `maximum(y, x)` produce equivalent elementwise outputs. |
| `minimum-commutativity-v1` | `minimum(x, y)` and `minimum(y, x)` produce equivalent elementwise outputs. |
| `min-max-duality-v1` | `minimum(x, y)` equals `-maximum(-x, -y)` elementwise. |
| `square-as-multiplication-v1` | `square(x)` and `mul(x, x)` produce equivalent elementwise outputs. |
| `square-negation-involution-v1` | `square(x)` and `square(-x)` produce equivalent elementwise outputs. |
| `min-max-reduction-duality-v1` | `min(x)` and `-max(-x)` produce equivalent reduction outputs. |
| `mul-commutativity-v1` | `x * y` and `y * x` produce equivalent elementwise outputs. |
| `min-max-addition-identity-v1` | `minimum(x, y) + maximum(x, y)` equals `x + y` elementwise. |
| `square-absolute-identity-v1` | `square(abs(x))` and `square(x)` produce equivalent elementwise outputs. |
| `negation-multiplication-duality-v1` | `(-x) * y` equals `-(x * y)` elementwise. |
| `sum-negation-duality-v1` | `sum(-x)` equals `-sum(x)` for global reduction. |
| `abs-negation-involution-v1` | `abs(-x)` and `abs(x)` produce equivalent elementwise outputs preserving signed zero. |
| `maximum-idempotence-v1` | `maximum(x, x)` and `minimum(x, x)` produce equivalent elementwise outputs equal to `x`. |
| `lattice-absorption-v1` | `maximum(x, minimum(x, y))` and `minimum(x, maximum(x, y))` produce equivalent elementwise outputs equal to `x`. |
| `abs-multiplication-v1` | `abs(x * y)` and `abs(x) * abs(y)` produce equivalent elementwise outputs. |
| `sqrt-square-identity-v1` | `sqrt(square(x))` and `abs(x)` produce equivalent elementwise outputs. |
| `mean-negation-duality-v1` | `mean(-x)` and `-mean(x)` produce equivalent reduction outputs. |
| `distributivity-mul-add-v1` | `x * (y + z)` and `(x * y) + (x * z)` produce equivalent elementwise outputs. |
| `add-associativity-v1` | `(x + y) + z` and `x + (y + z)` produce equivalent elementwise outputs. |
| `mul-associativity-v1` | `(x * y) * z` and `x * (y * z)` produce equivalent elementwise outputs. |
| `square-abs-multiplication-identity-v1` | `abs(x * x)` and `square(abs(x))` produce equivalent elementwise outputs. |
| `cos-even-duality-v1` | `cos(x)` and `cos(-x)` produce equivalent elementwise outputs. |

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

`xlaxiom corpus build` applies failure-preserving minimization before its independent
fresh-process reproduction gate. The checked-in corpus currently contains all thirty-five
reviewed properties, one labelled historical record, and no newly discovered finding.


The retained 100-seed HX370 JAX x64 campaign executed 1,300 property comparisons with
1,300 passes and no candidates. Separate seed-0 artifacts retain identical results on
the HX370 CPU, Tesla V100, and Tesla P40, plus identity-checked CPU↔GPU and V100↔P40
comparisons in [`experiments/`](experiments/).
