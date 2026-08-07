# Reproducing the STalign comparison

One container reproduces every STalign port comparison from **pinned, public sources** —
no cluster, no sibling checkout, no environment tricks. It bundles this repository, the
pinned squidpy JAX port (`selmanozleyen/squidpy@6a63ff8`), the pinned upstream STalign
submodule (`b2068ed`), and the example datasets, so a run needs only a GPU and this file.

## Build

From the repository root (the `%files` paths are resolved relative to it):

```bash
apptainer build container/stalign.sif container/stalign.def
```

Add `--fakeroot` if you build unprivileged. The build needs network (PyPI, GitHub) and
produces a multi-GB image (CUDA-enabled Torch + JAX, plus ~265 MB of upstream data).

## Run

`--nv` exposes the host GPU; `--bind` chooses where results land.

```bash
# whole suite (all 17 upstream notebooks)
apptainer run --nv --bind ./results:/output container/stalign.sif

# one notebook
apptainer run --nv --bind ./results:/output container/stalign.sif xenium-xenium-alignment.ipynb
```

Each notebook leaves `<name>-metrics.json` (per-quantity relative L2 between upstream
PyTorch and the JAX port), `<name>-manifest.json` (host, versions, timings, pins), and a
comparison panel in `./results`.

## What is pinned

| Component | Pin | Source |
|---|---|---|
| Upstream STalign | `b2068edc98974efa54537eca194736e177bbe11d` | submodule, baked in |
| squidpy JAX port | `selmanozleyen/squidpy@6a63ff8` (`feat/experimental-fit-core`) | installed from git at build |
| squidpy-ports | this checkout | `%files` at build |

## CPU-only

For a machine without a GPU, swap `jax[cuda12]` → `jax` in `stalign.def` and drop `--nv`.
It reproduces the same numbers, far more slowly (the upstream fits are the bottleneck).

> **Status:** draft — not yet build-tested (authored on a machine without a GPU or
> Apptainer). Validate on the first real build; the pins and entrypoint are correct, the
> base-image tag and any missing build deps may need a nudge.
