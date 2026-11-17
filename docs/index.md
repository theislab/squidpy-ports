# squidpy-ports

Reference implementations and evidence for squidpy ports: an upstream method reimplemented inside
squidpy, and what establishes that the reimplementation is the same algorithm.

One section per port.

- **[STalign](stalign.md)** — affine + LDDMM alignment, reimplemented in JAX as
  `squidpy.experimental.tl`. The public API is covered by sixteen of upstream's own analyses
  rewritten against it and executed on a GPU.
- **[SpaGE](spage.md)** — gene imputation, reimplemented as `squidpy.experimental.tl.impute_spage`. Upstream's
  tutorial rewritten against it reproduces upstream to 4e-15.
- **[Tangram](tangram.md)** — cell-to-spot mapping and gene imputation, reimplemented in JAX as
  `squidpy.experimental.tl.impute_tangram`. Upstream's tutorial rewritten against it reaches the same gene
  scores as upstream.

Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 2

STalign <stalign.md>
SpaGE <spage.md>
Tangram <tangram.md>
```
