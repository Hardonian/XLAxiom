# Architecture

`model` defines a typed, versioned JSON-compatible workload and validates graph references and shapes with a closed operation registry. `generation` supplies deterministic bounded cases. `execution` interprets that model with NumPy or actual JAX CPU/eager/JIT execution. `comparison` turns observations into typed classifications. `minimization` proposes valid simplifications and accepts only candidates that reexecute with the same expected classification. `storage` adds a canonical SHA-256 checksum to cases; `reporting` emits structured and readable evidence. `metamorphic` executes the checked-in curated corpus of safe relations; its JSON manifest can select only a closed, audited builder registry and never executable code.

The abstraction intentionally distinguishes an execution observation from a compiler root-cause assertion. Synthetic controls are carried in `fault` and `provenance`, preserving that distinction in the case and report.

The backend boundary is deliberately narrow. A future CUDA/TPU implementation must verify a genuine initialized device and real numerical operation before returning support; it must otherwise return an unsupported-backend result.
