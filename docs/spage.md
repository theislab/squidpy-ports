# SpaGE

squidpy's `experimental.tl.impute_spage` reimplements [SpaGE](https://github.com/tabdelaal/SpaGE)'s gene
imputation.

- **[SpaGE tutorial](notebooks/squidpy-api/spage-tutorial.ipynb)** — upstream's tutorial, which predicts osmFISH
  genes from AllenSSp scRNA-seq, written against the public API. On the same inputs the port and upstream
  `SpaGE()` agree to 4e-15.

The numerics are gated by
[`test_spage_reference.py`](https://github.com/selmanozleyen/squidpy/blob/feat/impute-spage/tests/experimental/test_spage_reference.py)
in squidpy, against values upstream SpaGE computed on a seeded fixture (`spage_reference.zarr.zip`).

Upstream SpaGE is pinned to
[`edd0fb7`](https://github.com/tabdelaal/SpaGE/tree/edd0fb7086eca791c6589020893eb4f034b195c7), and the squidpy
fork to [`f4c0c9a`](https://github.com/selmanozleyen/squidpy/tree/f4c0c9adf7f6976ffc70265fd84c93351375aacc).

```{toctree}
:hidden: true
:maxdepth: 2

SpaGE tutorial <notebooks/squidpy-api/spage-tutorial>
```
