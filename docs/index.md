# squidpy-ports

squidpy's `experimental.tl.align` reimplements [STalign](https://github.com/JEFworks-Lab/STalign)'s
affine + LDDMM alignment in JAX. This site is the evidence that the port is the same algorithm:
upstream's own published results beside ours on real data, the per-variable agreement across every
upstream notebook, and the seeded test suite that actually gates it.

- **[Upstream vs the port](stalign-comparison.md)** — three datasets, side by side, with metrics.
- **[Parity: all 17 upstream notebooks](parity.md)** — per-variable relative L2, replay by replay.
- **[How correctness is enforced](correctness.md)** — the reference bundle and the tests that assert it.

Pinned throughout: upstream STalign
[`b2068ed`](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d),
squidpy fork
[`6a63ff8`](https://github.com/selmanozleyen/squidpy/tree/feat/experimental-fit-core).
Source: [theislab/squidpy-ports](https://github.com/theislab/squidpy-ports).

```{toctree}
:hidden: true
:maxdepth: 1

Upstream vs the port <stalign-comparison.md>
Parity: all 17 notebooks <parity.md>
How correctness is enforced <correctness.md>
```
