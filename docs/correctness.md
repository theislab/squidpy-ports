# How correctness is enforced

The [side-by-side panels](stalign-comparison.md) are real-data evidence, but they are **not** the
gating check — a figure can look right while the algorithm underneath has drifted. The gate is a
layered, seeded reference suite: every stage of the algorithm is asserted against values that
upstream STalign itself computed.

## The three layers

1. **This repo (`squidpy-ports`)** runs upstream STalign — vendored and pinned to `b2068ed`, never
   edited — on small **synthetic, seeded** inputs and writes a reference bundle
   ([`generate.py`](https://github.com/theislab/squidpy-ports/blob/main/src/squidpy_ports/stalign/generate.py)
   → `.npz` files). Each carries a provenance blob pinned to the upstream commit, so the fixtures
   stay falsifiable rather than becoming magic numbers.
   [`tests/test_stalign.py`](https://github.com/theislab/squidpy-ports/blob/main/tests/test_stalign.py)
   guards the generator itself: that the vendored checkout is pinned, that the fixtures are
   deterministic, and that the port's captured **energy and gradient** agree with upstream's LDDMM
   loop step for step.

2. **squidpy** commits that bundle under `tests/_data/stalign_reference/` and asserts the JAX port
   reproduces it at every layer —
   [`test_stalign_reference.py`](https://github.com/selmanozleyen/squidpy/blob/6a63ff8/tests/experimental/methods/test_stalign_reference.py)
   checks `primitives` (rasterisation) → `energy` → `gradients` → `trajectory` at 1, 5, 50 iterations
   → `converged` at 500 → image warping, and that every fixture's provenance names upstream
   `b2068ed`. On the internals the port matches upstream to **near machine precision** (the LDDMM
   velocity field to ~1e-15); the ~1e-3 end-to-end figures on the comparison page come from
   rasterisation and interpolation at the boundaries, not from the fit.

3. **The public API** (`sq.experimental.tl.align`) is covered by
   [`test_align.py`](https://github.com/selmanozleyen/squidpy/blob/6a63ff8/tests/experimental/tl/test_align.py)
   — `AnnData`/`SpatialData` in-place vs copy, the path grammar, landmark handling, and recovering a
   known synthetic shift.

So "the alignment is correct" means the port is asserted against upstream at the primitive, energy,
gradient, trajectory, converged and image levels — with the real-data panels as corroboration. Every
file above is linked: none of it has to be taken on faith from a figure.

## Regenerating the reference bundle

```bash
git submodule update --init
hatch run generate:run --out ../squidpy/tests/_data/stalign_reference
```

Then commit the bundle in squidpy. Every `.npz` carries a `__provenance__` JSON blob recording the
upstream commit, this repo's commit, a checksum of `fixtures.py`, and the resolved
torch/numpy/platform — squidpy asserts these on load.

### Determinism

Upstream is bit-reproducible on CPU in float64 (no RNG, no dropout, no atomics), *given* a fixed
reduction order. The `generate` hatch environment pins `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`,
and the script refuses to run otherwise.

**Reduction order still differs across platforms** (macOS Accelerate vs Linux OpenBLAS). squidpy's
scheduled job runs on ubuntu, so the committed bundle should be generated on ubuntu too: trigger the
`generate` workflow and download its artifact rather than committing a bundle built on a laptop.

## Why not just reimplement upstream here?

Nothing in this repository reimplements upstream. Every value is produced by calling a public
upstream function, or by observing the unmodified `LDDMM` loop through autograd hooks — a
reimplementation would make *this* repository the thing under test. The single documented exception
(the four statements computing `LL`/`K`/`DV`, which live inside the loop with no function boundary)
is quoted verbatim with line references.
