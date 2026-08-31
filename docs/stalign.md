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

The numerics are gated separately, by the test suites in this repository rather than by a page:
[`tests/test_reference.py`](https://github.com/theislab/squidpy-ports/blob/main/tests/test_reference.py)
asserts every stage of the port against values upstream STalign itself computed, since a figure can
look right while the algorithm underneath has drifted.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`69dae69`](https://github.com/selmanozleyen/squidpy/tree/69dae69ad18ed3d3cade77e68cae0d35c285d873).
Every page names the commit behind its own numbers.

```{toctree}
:hidden: true
:maxdepth: 2

squidpy API equivalence <squidpy-api.md>
Intentional divergences <intentional-divergences.md>
```
