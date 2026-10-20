# Support and verification matrix

| Surface | Status | Evidence rule |
| --- | --- | --- |
| NumPy CPU | Supported | Unit, integration, and property tests run locally and in CI. |
| JAX CPU eager/JIT | Supported | Device placement and numerical policy are recorded. |
| PyTorch Eager/Inductor | Supported | Supports eager and `torch.compile` (Inductor) differential execution with recorded device placement. |
| CUDA 12/13 | Verified, capability-gated | V100, P40, and RTX 3060 evidence is retained; every new host still requires its own bounded probe. |
| TPU | Verified on Google Cloud TPU v5e | Retained physical evidence on 4-chip v5litepod-4 (`TPU_0`..`TPU_3`) confirms 40/40 eager, 40/40 JIT, 2,000/2,000 campaign evaluations, CPU/TPU classification equivalence, x64 execution, and higher-order autodiff without CPU fallback. |
| TPU simulation | Supported, non-physical | The offline qualification runs the full x64 corpus on CPU JIT with explicit simulation provenance; it cannot close physical TPU claims. |
| Apple Silicon MPS | Implemented, fail-closed | Non-destructive hardware discovery via system profiler; commands execute without synthetic CPU fallback. |
| AMD ROCm / HIP | Implemented, fail-closed | GPU discovery via `rocm-smi` and sysfs; fails closed on missing driver or hardware. |
| JAX x64 | Supported on configured hosts | `property run --x64` re-execs before JAX import and records x64 state. |
| Windows/Linux | Verified | Independent HX370 Windows and EPYC Linux records are checksummed and compared. |
| macOS | Portable, unverified | Commands are portable; a macOS host must supply retained evidence before support is claimed. |

Use `xlaxiom environment capture` on each host and `xlaxiom environment compare` to
attach redacted, checksummed independent-host metadata. Use `xlaxiom runtime list` and
`xlaxiom runtime compare` for fresh-process CPU thread-profile studies.

Use `xlaxiom property compare` to compare the same checksummed property pairs across
CPU, GPU, or TPU evidence artifacts. GPU property artifacts retain redacted physical
device identity and both host and runtime ordinals.

See [`roadmap.md`](roadmap.md) for the distinction between implemented code and
hardware/external verification that remains unavailable to the current host.
