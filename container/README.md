# Reproducing a STalign notebook

The executed notebooks are produced by **one container**, from **pinned, public sources** —
no sibling checkout, no environment tricks. It bundles this repository, the
pinned squidpy JAX port (`selmanozleyen/squidpy@6c4b5ce`), the pinned upstream STalign
submodule (`b2068ed`) with its datasets, and the notebooks themselves. Anyone with a GPU box
and Apptainer can rebuild any result.

## Build

From the repository root (the `%files` paths resolve relative to it):

```bash
apptainer build container/stalign.sif container/stalign.def
```

Add `--fakeroot` if you build unprivileged. The build needs network (PyPI, GitHub) and
produces a multi-GB image (CUDA-enabled Torch + JAX + the vendored data).

## Run

Pass one notebook; the container fits it through squidpy's public API and writes the executed
notebook, figures and all, into the bound output directory.

```bash
apptainer run --nv --writable-tmpfs --bind ./out:/output \
    container/stalign.sif merfish-merfish.ipynb
```

- `--nv` exposes the host GPU; `--writable-tmpfs` gives the run a scratch overlay; `--bind`
  chooses where results land.
- Notebooks: any file in [`docs/notebooks/squidpy-api/`][api]. The two `*-allen3Datlas` ones
  additionally download the Allen atlas at run time; the rest read only vendored data.

[api]: https://github.com/theislab/squidpy-ports/tree/main/docs/notebooks/squidpy-api

Each run leaves in `./out`:

| File | Contents |
|---|---|
| `<name>.ipynb` | the notebook as executed, with its figures and printed numbers embedded |

The notebook is the whole record: it names the fit's parameters, prints what the fit reached, and
carries the figures it produced. The squidpy commit is the one pinned in this container.

## What is pinned

| Component | Pin | Source |
|---|---|---|
| Upstream STalign | `b2068edc98974efa54537eca194736e177bbe11d` | submodule, baked in |
| squidpy JAX port | `selmanozleyen/squidpy@6c4b5ce` (`feat/experimental-fit-core`) | installed from git at build |
| squidpy-ports | this checkout | `%files` at build |

## CPU-only

For a machine without a GPU, swap `jax[cuda12]` → `jax` in `stalign.def` and drop `--nv`. It
reproduces the same numbers, far more slowly (the upstream fits dominate).
