# Reproducibility and numerical methodology

Workload identity is the SHA-256 hash of canonical JSON. Inputs are regenerated with NumPy `SeedSequence` using the saved workload seed. Case artifacts separately checksum their complete canonical payload; a checksum mismatch stops loading.

The default policy uses `abs(a-b) <= atol + rtol * abs(reference)` for finite values, selected explicitly for float32-oriented CPU/JAX checks. It separately tracks finite mismatches, NaN placement, infinity sign, and signed-zero differences. Users must record and justify policy changes. JAX x64 mode and matmul precision can alter valid results and must be treated as configuration variables, not silently normalized away.

Every minimizer candidate is validated, reexecuted, and compared to the original classification. A newly introduced exception, invalid workload, or different classification is rejected.
