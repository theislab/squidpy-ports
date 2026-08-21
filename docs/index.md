# squidpy-ports

squidpy's `experimental.tl.align` reimplements [STalign](https://github.com/JEFworks-Lab/STalign)'s
affine + LDDMM alignment in JAX. This site is the evidence that the port is the same algorithm:
upstream's own published results beside ours on real data, the per-variable agreement across every
upstream notebook, and the seeded test suite that actually gates it.

- **[Visual comparison](stalign-comparison.md)** — three datasets, upstream's figure beside ours.
- **[Notebook values comparison](parity.md)** — every variable in all 17 upstream notebooks, replayed both ways.
- **[Numerical tests](correctness.md)** — the seeded reference suite, what each test asserts, and its result.
- **[The squidpy API](squidpy-api.md)** — all seventeen analyses rewritten against the public API, executed.
- **[Divergence ledger](STALIGN_DIVERGENCES.md)** — every place the port and the original disagree, measured, with the verdict.

Upstream STalign is pinned throughout to
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d).
The squidpy fork moves: the notebook values page is a run at
[`9735ba3c`](https://github.com/selmanozleyen/squidpy/tree/9735ba3c8198ba279f3f688df92ba5a6876e28c4),
the figures and the test results still come from
[`6a63ff8`](https://github.com/selmanozleyen/squidpy/tree/6a63ff832e6e13ddf02c6add7baa28ff74c6b392).
Every page names the commit behind its own numbers.
Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 1

Visual comparison <stalign-comparison.md>
Notebook values comparison <parity.md>
The squidpy API <squidpy-api.md>
Numerical tests <correctness.md>
```
