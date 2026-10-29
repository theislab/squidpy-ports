# STalign

squidpy's `experimental.tl` reimplements
[STalign](https://github.com/JEFworks-Lab/STalign)'s affine + LDDMM alignment in JAX. This section
is the evidence that the port is the same algorithm, in two parts that answer different questions.

- **[squidpy API equivalence](squidpy-api.md)** — the public API,
  `squidpy.experimental.tl`, is covered by sixteen of STalign's seventeen analyses rewritten
  against it and executed on a GPU, one page per upstream notebook. These show the results are
  reasonable on real data, through the public surface only.
- **[Intentional divergences](intentional-divergences.md)** — the one place the port
  deliberately departs from upstream's algorithm, and the reading behind it.

The numerics are gated separately, by a test suite rather than by a page:
[`test_stalign_reference.py`](https://github.com/scverse/squidpy/blob/main/tests/experimental/tl/test_stalign_reference.py)
in squidpy asserts every stage of the port against values upstream STalign itself computed, since a figure can
look right while the algorithm underneath has drifted. The **[reference bundle](reference-bundle.md)**
page says what those values are and why they are stored rather than recomputed.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`aa2b414`](https://github.com/selmanozleyen/squidpy/tree/aa2b4142ae18903ab599676ebacc83e696b7dc17).
Every page names the commit behind its own numbers.

```{toctree}
:hidden: true
:maxdepth: 2

squidpy API equivalence <squidpy-api.md>
Intentional divergences <intentional-divergences.md>
Reference bundle <reference-bundle.md>
```
