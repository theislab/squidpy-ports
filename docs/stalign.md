# STalign

squidpy's `experimental.tl.align` reimplements
[STalign](https://github.com/JEFworks-Lab/STalign)'s affine + LDDMM alignment in JAX. This section
is the evidence that the port is the same algorithm, in two parts that answer different questions.

- **[squidpy API equivalence](squidpy-api.md)** — the public API,
  `squidpy.experimental.tl.align`, is covered by sixteen of STalign's seventeen analyses rewritten
  against it and executed on a GPU, one page per upstream notebook. These show the results are
  reasonable on real data, through the public surface only.
- **[Numerical tests](correctness.md)** — the seeded reference suite that gates the port. A figure
  can look right while the numerics underneath have drifted; this is what rules that out, by
  asserting every stage against values upstream STalign itself computed.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`5c67ea7`](https://github.com/selmanozleyen/squidpy/tree/5c67ea76647227f2d758704a2709c9162d318e1c).
Every page names the commit behind its own numbers.

```{toctree}
:hidden: true
:maxdepth: 2

squidpy API equivalence <squidpy-api.md>
Numerical tests <correctness.md>
```
