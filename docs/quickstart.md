# Reproducible quickstart

## CPU

```console
uv sync --extra dev
uv run xlaxiom doctor
uv run xlaxiom property coverage
uv run xlaxiom property campaign --cases 20 --backend jax-jit --x64
uv run xlaxiom property compare --baseline cpu.json --comparison accelerator.json
uv run xlaxiom campaign --profile smoke --output results/campaign.json
```

For a relational handoff, materialize a property, run it, then report or bundle it:

```console
uv run xlaxiom relation create --property matmul-transpose-v1 --output relation.json
uv run xlaxiom relation run --case relation.json
uv run xlaxiom relation reproduce --case relation.json
uv run xlaxiom relation report --case relation.json --output results/relation-report
```

## CUDA or TPU

Do not install drivers, change system CUDA, or assume a CPU result represents an
accelerator. On a compatible target host, begin with `xlaxiom gpu doctor` and
`xlaxiom gpu validate`; retain its artifact before running a GPU campaign. Curated
relations use `--backend jax-gpu-jit` or `--backend jax-tpu-jit`. TPU follows the same
capability-first rule: an absent device is unsupported, not a fallback target.
