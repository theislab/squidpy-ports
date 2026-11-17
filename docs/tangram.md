# Tangram

squidpy's `experimental.tl.impute_tangram` reimplements [Tangram](https://github.com/broadinstitute/Tangram)'s
default `mode="cells"` mapping in JAX.

- **[Tangram tutorial](notebooks/squidpy-api/tangram-tutorial.ipynb)** — the cells-mode
  part of upstream's tutorial, which maps mouse cortex scRNA-seq onto a Visium section, written against the
  public API. Each implementation starts from its own random mapping; on the held-out genes both reach a median
  score of 0.455, and their per-gene scores correlate at 0.999.

The numerics are gated by
[`test_tangram_reference.py`](https://github.com/selmanozleyen/squidpy/blob/feat/impute-tangram/tests/experimental/test_tangram_reference.py)
in squidpy, which starts the port from upstream's own initial mapping and compares the trajectory against values
upstream Tangram computed (`tangram_reference.zarr.zip`).

Upstream Tangram is pinned to
[`4c68995`](https://github.com/broadinstitute/Tangram/tree/4c68995a418f41dc8caef567598c4d9b47781a13), and the
squidpy fork to [`f4c0c9a`](https://github.com/selmanozleyen/squidpy/tree/f4c0c9adf7f6976ffc70265fd84c93351375aacc).

```{toctree}
:hidden: true
:maxdepth: 2

Tangram tutorial <notebooks/squidpy-api/tangram-tutorial>
```
