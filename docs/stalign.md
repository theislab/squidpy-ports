# STalign

squidpy's `experimental.tl` reimplements
[STalign](https://github.com/JEFworks-Lab/STalign)'s affine + LDDMM alignment in JAX. This section
is the evidence that the port is the same algorithm, in two parts that answer different questions.

- **[squidpy API equivalence](squidpy-api.md)** — the public API,
  `squidpy.experimental.tl`, is covered by sixteen of STalign's seventeen analyses rewritten
  against it and executed on a GPU, one page per upstream notebook. These show the results are
  reasonable on real data, through the public surface only.

The numerics are gated separately, by the test suites in this repository rather than by a page:
[`tests/test_reference.py`](https://github.com/theislab/squidpy-ports/blob/main/tests/test_reference.py)
asserts every stage of the port against values upstream STalign itself computed, since a figure can
look right while the algorithm underneath has drifted.

## Rank-3 divergence

At rank 2 the port matches upstream to ~1e-15 on the velocity field. At rank 3 it does not, and
the cause reads as a slip rather than a modelling choice: the rank-3 regularisation *energy*
transforms two of three spatial axes, `fftn(v, dim=(1,2))`
([`STalign.py:1504`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1504)) -- byte-identical to the rank-2 line at
[`:1193`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1193), where two axes is all of them -- while the gradient it descends smooths
all three, `dim=(1,2,3)` ([`:1527`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1527)). A rank-2 line reused at rank 3 without
extending it to the new axis. Transforming all three moves the fitted velocity field by 31x,
which is why `sigmaR` is retuned from upstream's `1e8` to `1e6`.

Not checked against the paper ([Clifton et al., *Nat Commun* 14, 8123,
2023](https://doi.org/10.1038/s41467-023-43915-7)) -- the text was not reachable. The
divergence is pinned by a strict xfail in `tests/test_reference.py`, not asserted away.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
and the squidpy fork to
[`69dae69`](https://github.com/selmanozleyen/squidpy/tree/69dae69ad18ed3d3cade77e68cae0d35c285d873).
Every page names the commit behind its own numbers.

```{toctree}
:hidden: true
:maxdepth: 2

squidpy API equivalence <squidpy-api.md>
```
