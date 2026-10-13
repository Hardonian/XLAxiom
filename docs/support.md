# Support and verification matrix

| Surface | Status | Evidence rule |
| --- | --- | --- |
| NumPy CPU | Supported | Unit, integration, and property tests run locally and in CI. |
| JAX CPU eager/JIT | Supported | Device placement and numerical policy are recorded. |
| CUDA 12/13 | Capability-gated | A real bounded probe on a compatible, idle device is required. |
| TPU | Capability-gated | No local claim; absent TPU is reported unsupported. |
| JAX x64 | Supported on configured hosts | `property run --x64` re-execs before JAX import and records x64 state. |
| Windows/macOS/Linux | Portable commands | Each platform needs its own retained environment evidence result. |

Use `xlaxiom environment capture` on each host and `xlaxiom environment compare` to
attach redacted, checksummed independent-host metadata. Use `xlaxiom runtime list` and
`xlaxiom runtime compare` for fresh-process CPU thread-profile studies.

See [`roadmap.md`](roadmap.md) for the distinction between implemented code and
hardware/external verification that remains unavailable to the current host.
