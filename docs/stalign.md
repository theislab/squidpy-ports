# STalign

squidpy's `experimental.tl.align` reimplements
[STalign](https://github.com/JEFworks-Lab/STalign)'s affine + LDDMM alignment in JAX. This section
is the evidence that the port is the same algorithm, in two parts that answer different questions.

- **[squidpy API equivalence](squidpy-api.md)** — sixteen of STalign's seventeen analyses rewritten
  against squidpy's public API and executed on a GPU, one page per upstream notebook. These show
  the results are reasonable on real data, through the public surface only.
- **[Numerical tests](correctness.md)** — the seeded reference suite that gates the port. A figure
  can look right while the numerics underneath have drifted; this is what rules that out, by
  asserting every stage against values upstream STalign itself computed.
- **[Divergence ledger](STALIGN_DIVERGENCES.md)** — every place the port and the original disagree,
  measured, with the verdict.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`6c4b5ce`](https://github.com/selmanozleyen/squidpy/tree/6c4b5ce93b100f43fe873949aef6446461a276c0).
Every page names the commit behind its own numbers.

```{toctree}
:hidden: true
:maxdepth: 2

squidpy API equivalence <squidpy-api.md>
Numerical tests <correctness.md>
Divergence ledger <STALIGN_DIVERGENCES.md>
```
