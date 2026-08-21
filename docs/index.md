# squidpy-ports

squidpy's `experimental.tl.align` reimplements [STalign](https://github.com/JEFworks-Lab/STalign)'s
affine + LDDMM alignment in JAX. This site is the evidence that the port is the same algorithm.

- **[STalign: squidpy API equivalence](squidpy-api.md)** — sixteen of STalign's seventeen analyses
  rewritten against squidpy's public API, executed on a GPU, one page per upstream notebook.
- **[Numerical tests](correctness.md)** — the seeded reference suite that gates the port: what
  each test asserts, and its result.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`6c4b5ce`](https://github.com/selmanozleyen/squidpy/tree/6c4b5ce93b100f43fe873949aef6446461a276c0).
Every page names the commit behind its own numbers.
Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 1

STalign: squidpy API equivalence <squidpy-api.md>
Numerical tests <correctness.md>
Divergence ledger <STALIGN_DIVERGENCES.md>
```
