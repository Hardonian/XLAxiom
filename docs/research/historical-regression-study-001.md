# XLAxiom Historical Regression Study 001 — JAX Hyperbolic Gradient Stability

## Question

Does the behavior reported in JAX issue #40634 change between released JAX 0.11.1 and 0.11.2, when each operation is checked against an independent analytic derivative?

## Method

Each environment was resolved in a separate `uv --isolated` process. The study evaluated eager and JIT gradients for float32 inputs. The oracle for `arcsinh(x)` is `1 / hypot(1, abs(x))`; for real-domain `acosh(x)` it is `1 / (sqrt(x - 1) * sqrt(x + 1))`. Both are evaluated with host float64 arithmetic. A relative tolerance of `2e-5` was used. JAX x64 remained disabled so the JAX input and gradient stayed float32.

## Actual observations

| JAX | Observation | 0.11.1 | 0.11.2 | Oracle | Transition |
| --- | --- | ---: | ---: | ---: | --- |
| 0.11.x | `arcsinh(+1e22)` gradient | 0.0 | 0.0 | `1e-22` | stable historical discrepancy |
| 0.11.x | `arcsinh(-1e22)` gradient | 0.0 | 0.0 | `1e-22` | stable historical discrepancy |
| 0.11.x | `acosh(+1e22)` gradient | 0.0 | `9.99999968e-23` | `1e-22` | **FIXED** |

Eager and JIT agreed within each observation. The exact structured evidence is [hyperbolic-matrix-0.11.json](../experiments/hyperbolic-matrix-0.11.json).

## Lineage and limitations

The [upstream issue](https://github.com/jax-ml/jax/issues/40634) identifies intermediate squaring overflow in the derivative rules. [Commit 86d3fe7](https://github.com/jax-ml/jax/commit/86d3fe7) changes the arcsinh JVP and adds a regression test. Its earliest official released wheel is **pending verification**; therefore this study makes no package-level claim that the commit is released.

This is a reproduction of an existing upstream report, not a new XLAxiom discovery. It was executed only on Windows CPU. CUDA/EPYC validation is blocked by unavailable hardware access.

## Reproduction

```console
uv run xlaxiom historical-matrix --version 0.11.1 --version 0.11.2 --output results/hyperbolic.json
```
