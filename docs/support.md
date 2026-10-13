# Support and verification matrix

| Surface | Status | Evidence rule |
| --- | --- | --- |
| NumPy CPU | Supported | Unit, integration, and property tests run locally and in CI. |
| JAX CPU eager/JIT | Supported | Device placement and numerical policy are recorded. |
| CUDA 12/13 | Verified, capability-gated | V100, P40, and RTX 3060 evidence is retained; every new host still requires its own bounded probe. |
| TPU | Implemented, hardware-unverified | Explicit TPU eager/JIT requests fail closed without CPU fallback; no physical TPU claim is made. |
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
