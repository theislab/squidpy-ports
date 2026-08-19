# Numerical tests

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

## Results

The tables below are a real run, not a claim about one, and nothing in them is written by hand.
Statuses and expected-failure reasons come from pytest's JUnit report; each description is the
test's **own docstring**, read from the source with `ast`. A wrong description is therefore a wrong
docstring, and both land in the same review.

```{include} _static/tests/results.md
```

Run against upstream STalign `b2068ed` and squidpy fork `6a63ff8` — the same pinned pair behind the
figures on the [visual comparison](stalign-comparison.md) page, so the pictures and the assertions
describe one environment. Raw pytest output is served beside this page:
[`squidpy-ports-tests.log`](_static/tests/squidpy-ports-tests.log) and
[`squidpy-fork-tests.log`](_static/tests/squidpy-fork-tests.log).
[Running the tests yourself](https://github.com/theislab/squidpy-ports#running-the-tests) is four
commands, and regenerating this table is a fifth.

:::{warning}
The fork's reference suite is marked `pytest.mark.reference`, and squidpy's `addopts` carry
`-m "not reference"`. A plain `pytest` **silently deselects all 62 of them** and reports the
remaining 32 as a clean run — so it must be overridden explicitly:

```bash
pytest -m "reference or not reference" \
    tests/experimental/methods/test_stalign_reference.py tests/experimental/tl/test_align.py
```
:::

The two expected failures are deliberate divergences where **squidpy is the more correct of the
two**, pinned as strict `xfail` so that silently adopting upstream's behaviour breaks the build.
Each cites a row in the [divergence ledger](STALIGN_DIVERGENCES.md), and
`test_divergences_doc_covers_all_xfails` asserts every xfail cites a row that exists.

Nothing is skipped in the run above because both sides are installed. Running this repo's suite
alone leaves one test skipped — it needs JAX and squidpy, which this repo deliberately does not
depend on — and it says so rather than passing quietly.

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
