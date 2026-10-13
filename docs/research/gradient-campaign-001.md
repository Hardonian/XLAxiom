# Gradient Campaign 001

**Status:** local CPU observation; not CUDA evidence  
**Oracle:** analytic derivatives evaluated at the float32 value supplied to JAX  
**Command:** `uv run xlaxiom gradient-campaign --cases 40 --seed 0 --output docs/experiments/gradient-campaign-local.json`

The deterministic forty-case campaign covers `tanh`, `exp`, `arcsinh`, and
`acosh` ten times each.  Its machine-readable artifact records 34 passes, two
previously documented large-magnitude `arcsinh` observations, and four
candidate observations.

Every campaign artifact records the Python/JAX/jaxlib versions, platform, and
enumerated JAX devices. This makes an EPYC/CUDA run directly comparable to the
local CPU record without inferring its backend from the filename.

The candidates are all `float32 tanh(10)`: eager and JIT reverse-mode return
zero while the analytic derivative is approximately `8.2446e-09`.  They are
retained as candidates rather than silently tolerated because the current
relative-only comparison has no absolute-error budget at this scale.  Re-run
the same campaign on EPYC/CUDA and compare the artifact before assigning a
backend-specific classification.

When the EPYC artifact is available, compare it without altering either source
record:

```console
uv run xlaxiom gradient-compare \
  --baseline docs/experiments/gradient-campaign-local.json \
  --comparison results/gradient-campaign-epyc-cuda.json \
  --output results/gradient-campaign-hx370-vs-epyc.json
```

The comparison requires the same campaign configuration before treating the
case signatures as directly comparable, and retains both environment records.

The `acosh` inputs are deliberately inside (not on) the singular boundary at
one.  Analytic references use the float32-rounded input, which prevents
mistaking normal representation rounding near that boundary for a compiler
regression.
