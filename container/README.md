# Reproducing a STalign comparison

The comparison panels are produced by **one container**, from **pinned, public sources** —
no sibling checkout, no environment tricks. It bundles this repository, the
pinned squidpy JAX port (`selmanozleyen/squidpy@6a63ff8`), the pinned upstream STalign
submodule (`b2068ed`) with its datasets, and the comparison notebooks. Anyone with a GPU box
and Apptainer can rebuild any result.

## Build

From the repository root (the `%files` paths resolve relative to it):

```bash
apptainer build container/stalign.sif container/stalign.def
```

Add `--fakeroot` if you build unprivileged. The build needs network (PyPI, GitHub) and
produces a multi-GB image (CUDA-enabled Torch + JAX + the vendored data).

## Run

Pass one comparison notebook; the container fits **both** upstream STalign (PyTorch) and the
squidpy JAX port and writes the executed notebook, its panel, and a provenance manifest.

```bash
apptainer run --nv --writable-tmpfs --bind ./out:/output \
    container/stalign.sif stalign-xenium-comparison.ipynb
```

- `--nv` exposes the host GPU; `--writable-tmpfs` gives the run a scratch overlay; `--bind`
  chooses where results land.
- Notebooks: `stalign-xenium-comparison.ipynb`, `stalign-merfish-comparison.ipynb`,
  `stalign-visium-affine-comparison.ipynb`.

Each run leaves in `./out`:

| File | Contents |
|---|---|
| `<stem>-executed.ipynb` | the notebook with the density plot, the upstream-vs-port panel, and the metric table embedded |
| `<stem>-panel.png` | the comparison panel on its own |
| `<stem>-manifest.json` | package versions **and the resolved squidpy fork commit** — the results are never orphaned from the code |

The manifest's `squidpy_commit` is read from pip's install record, so a container built from
the pin above stamps `6a63ff8…` into every run automatically — that is the answer to "how was
this made".

## What is pinned

| Component | Pin | Source |
|---|---|---|
| Upstream STalign | `b2068edc98974efa54537eca194736e177bbe11d` | submodule, baked in |
| squidpy JAX port | `selmanozleyen/squidpy@6a63ff8` (`feat/experimental-fit-core`) | installed from git at build |
| squidpy-ports | this checkout | `%files` at build |

## CPU-only

For a machine without a GPU, swap `jax[cuda12]` → `jax` in `stalign.def` and drop `--nv`. It
reproduces the same numbers, far more slowly (the upstream fits dominate).
