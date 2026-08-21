# squidpy-ports

Reference implementations and evidence for squidpy ports: an upstream method reimplemented inside
squidpy, and what establishes that the reimplementation is the same algorithm.

One section per port.

- **[STalign](stalign.md)** — affine + LDDMM alignment, reimplemented in JAX as
  `squidpy.experimental.tl.align`. The public API is covered by sixteen of upstream's own analyses
  rewritten against it and executed, and the numerics are gated by a seeded reference suite.

Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 2

STalign <stalign.md>
```
