# squidpy-ports

Reference implementations and evidence for squidpy ports: an upstream method reimplemented inside
squidpy, and what establishes that the reimplementation is the same algorithm.

One section per port.

- **[STalign](stalign.md)** — affine + LDDMM alignment, reimplemented in JAX as
  `squidpy.experimental.tl.align`. Sixteen of upstream's own analyses rewritten against the public
  API and executed, plus the seeded reference suite that gates the numerics.

Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 2

STalign <stalign.md>
```
