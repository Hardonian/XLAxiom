# Support and verification matrix

| Surface | Status | Evidence rule |
| --- | --- | --- |
| NumPy CPU | Supported | Unit, integration, and property tests run locally and in CI. |
| JAX CPU eager/JIT | Supported | Device placement and numerical policy are recorded. |
| CUDA 12/13 | Capability-gated | A real bounded probe on a compatible, idle device is required. |
| TPU | Capability-gated | No local claim; absent TPU is reported unsupported. |
| JAX x64 | Configurable | State must be recorded; no x64 result may be inferred from x32. |
| Windows/macOS/Linux | Portable commands | Each platform needs its own retained evidence result. |

See [`roadmap.md`](roadmap.md) for the distinction between implemented code and
hardware/external verification that remains unavailable to the current host.
