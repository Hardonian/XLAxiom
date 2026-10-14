# Architecture

`model` defines a typed, versioned JSON-compatible workload and validates graph references and shapes with a closed operation registry spanning unary elementwise, binary elementwise, matrix multiplication, reshape, transpose, and multi-axis reductions (including `maximum`, `minimum`, `sqrt`, `square`, `log`, `prod`, `max`, and `min`). `generation` supplies deterministic bounded cases with operation targeting. `execution` interprets that model with NumPy or actual JAX CPU/eager/JIT execution. `comparison` turns observations into typed classifications and records complete error distributions (maximum absolute/relative error, mean absolute error, and root-mean-square error). `minimization` proposes valid simplifications and accepts only candidates that reexecute with the same expected classification. `storage` adds a canonical SHA-256 checksum to cases; `reporting` emits structured and readable evidence. `metamorphic` executes the checked-in curated corpus of safe relations; its JSON manifest can select only a closed, audited builder registry and never executable code.

The abstraction intentionally distinguishes an execution observation from a compiler root-cause assertion. Synthetic controls are carried in `fault` and `provenance`, preserving that distinction in the case and report.

The backend boundary is deliberately narrow. CPU, CUDA, and TPU requests must verify a
genuine initialized device and real numerical operation before returning support; they
otherwise return an unsupported-backend result without CPU fallback. Physical TPU
execution remains unverified even though the fail-closed backend path is implemented.

For integrations that expose a NumPy-compatible namespace, `execute_array_api` binds
the same validated execution plan to that namespace and records a caller-supplied
backend label. Adapters can provide explicit input/output conversion functions without
adding a framework dependency to XLAxiom. JAX accelerator work should continue to use
`execute_jax`, which additionally verifies the requested device placement.

```python
from xlaxiom import execute_array_api

result = execute_array_api(workload, compatible_namespace, backend="my-array-backend")
```
