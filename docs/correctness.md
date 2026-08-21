# Tests: numeric and notebook comparison

Two different jobs, and it is worth keeping them apart.

The [executed notebooks](squidpy-api.md) already establish that the results are reasonable — real
data, real figures, through the public API only. What they cannot establish is that the algorithm
underneath is upstream's: a figure can look right while the numerics have drifted. That is what
the seeded reference suite below is for, asserting every stage against values upstream STalign
itself computed.

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

Run against upstream STalign `b2068ed` and squidpy fork `6a63ff8` — the same pinned pair behind the
figures in the [executed notebooks](squidpy-api.md), so the pictures and the assertions
describe one environment. Raw pytest output is served beside this page:
[`squidpy-ports-tests.log`](_static/tests/squidpy-ports-tests.log) and
[`squidpy-fork-tests.log`](_static/tests/squidpy-fork-tests.log).
[Running the tests yourself](https://github.com/theislab/squidpy-ports#running-the-tests) is four
commands.

:::{warning}
The fork's reference suite is marked `pytest.mark.reference`, and squidpy's `addopts` carry
`-m "not reference"`. A plain `pytest` **silently deselects the whole reference suite** and reports
the remainder as a clean run — so it must be overridden explicitly:

```bash
pytest -m "reference or not reference" \
    tests/experimental/methods/test_stalign_reference.py tests/experimental/tl/test_align.py
```
:::

The two expected failures are deliberate divergences where **squidpy is the more correct of the
two**, pinned as strict `xfail` so that silently adopting upstream's behaviour breaks the build.
Each cites a row in the [divergence ledger](STALIGN_DIVERGENCES.md), and
`test_divergences_doc_covers_all_xfails` asserts every xfail cites a row that exists.

Nothing is skipped when both sides are installed. Running this repo's suite
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

## Notebook comparison

A second, **development-only** check, and the one that established the foundation in the first
place: replay each of upstream's own notebooks and compare its every intermediate variable with
squidpy's, per notebook, per variable. It is what demonstrated the port was the same algorithm
rather than merely a plausible one — agreement to ~1e-15 on the LDDMM velocity field for the
rank-2 notebooks, and it is how each row of the [divergence ledger](STALIGN_DIVERGENCES.md) was
measured rather than argued.

It is not part of the gate, and it is not expected to keep working:

- **It reaches into internal API.** To compare intermediates at all it has to call the same
  private entry points upstream's notebooks call and read squidpy's internals beside them. Those
  are free to move; nothing promises otherwise.
- **It is pinned to a fixed branch.** The comparison was run against one squidpy revision and one
  upstream revision. Squidpy has moved since, so the harness may well have diverged from what it
  was written against — a failure there today is as likely to mean the internals were renamed as
  that anything is wrong.
- **Its evidence has been retired from this site.** The per-variable tables and the replayed
  upstream notebooks were 87 MB of pages restating what the ledger already records in a page.

The code is kept because the reference suite borrows from it: `test_stalign_reference.py` issues
its fits through `notebook_suite`'s own helpers deliberately, so the reference numbers cannot
drift from the way the fit is actually invoked. Run it with
`.claude/run_stalign_comparisons.sbatch`; expect to fix internal-API drift first.
