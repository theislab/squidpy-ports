# squidpy-ports

[![Tests][badge-tests]][tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/squidpy-ports/test.yaml?branch=main

Reference outputs for squidpy's ported algorithms.

squidpy reimplements some methods that originate elsewhere. A reimplementation that
merely produces plausible, correctly-shaped output can drift from the algorithm it
claims to be without anyone noticing. This repository runs the *original* implementation
— vendored and pinned — and emits the numbers squidpy's tests compare against, so the
heavyweight reference dependencies never enter squidpy itself.

Currently covers: **STalign** (see [scverse/squidpy#1243]), and **SpaGE** ([below](#spage)).

[scverse/squidpy#1243]: https://github.com/scverse/squidpy/issues/1243

## Results & reproduction

| | |
| --- | --- |
| **squidpy API equivalence** | [Sixteen of STalign's seventeen analyses][api], rewritten against `squidpy.experimental.tl` and executed on a GPU — one page per upstream notebook, each linking the original it mirrors. |
| **Numerical tests** | [`tests/test_stalign_generator.py`][ports-tests] guards this repo's own inputs — the vendored pin, the fixture definitions, the provenance stamp. The port itself is asserted against upstream at rank 2 and rank 3 in squidpy, by [`test_stalign_reference.py`][ref-tests], which downloads the bundle this repo publishes. |
| **Reference bundle** | [What it holds and why it is stored][bundle]; [publishing a new one](#publishing-the-reference-bundle). |
| **Reproduce on any GPU box** | [`container/README.md`][container] — one pinned container, any GPU box. |

The executed notebooks are corroboration, not the gate. The gate is
[`test_stalign_reference.py`][ref-tests] in squidpy, which compares the port at every layer — rasterisation, energy,
gradients, trajectory, converged fit, image warp, and the section-into-volume fit. At rank 2 it
matches upstream to ~1e-15 on the velocity field. At rank 3 it matches the solver on the seeded
fixture, but the two implementations descend on different objectives once the velocity moves:
upstream regularises over two spatial axes while smoothing that energy's gradient over three. That
divergence is pinned by a strict xfail rather than asserted away, and the xfail's own docstring
says what it costs.

**Which repository holds what.** squidpy holds every test of squidpy, the reference comparison
included, so a change that breaks the port fails in the PR that makes it. This repository holds
what produces the references: the vendored upstream, the generator, and the published bundle.

## Licensing

This repository is BSD-3-Clause. The vendored upstreams are included as git submodules, not
redistributed as part of any built artifact: `vendor/STalign` is **GPL-3.0** and `vendor/SpaGE` is
**MIT**, both recorded in [`LICENSE.vendor`][license-vendor]. Nothing is published to PyPI.

The reference bundle holds numerical outputs computed from synthetic, seeded
inputs defined in this repository — no upstream data is copied or redistributed.

## Layout

| Path | What it is |
| --- | --- |
| `vendor/STalign` | Upstream, pinned to one commit. Never edited, never linted, never pip-installed. |
| `src/squidpy_ports/stalign/fixtures.py` | Input definitions, shared by the generator and the tests; depends on numpy only. |
| `src/squidpy_ports/stalign/upstream.py` | Loads upstream by path, asserts the pin, and observes the LDDMM loop. |
| `src/squidpy_ports/stalign/generate.py` | Writes the reference bundle, `stalign_reference.zarr.zip`. |
| `vendor/SpaGE`, `src/squidpy_ports/spage/` | The same for SpaGE; writes `spage_reference.zarr.zip`. |
| `src/squidpy_ports/_common.py` | What every method shares: the pin check and by-path import, the bundle format and its provenance, the generators' CLI, and the bundle comparison the workflow runs. |

Every upstream is imported by path rather than installed. STalign's `setup.py` reads
`requirements.txt` verbatim and pins `numpy==1.23.4` / `torch==2.0.0` / `Pillow==9.5.0`,
none of which resolve on a current Python. Vendoring makes those pins irrelevant;
`uv.lock` supplies the reproducibility instead.

Nothing here reimplements upstream — every value comes from calling a public upstream function or
observing the unmodified `LDDMM` loop.

## Environments and tests

Each method has its own extra in `pyproject.toml` (`stalign`, `spage`) holding only what
its upstream imports, so each runs in its own environment. CI tests and generates every method in
its own job, from `uv.lock`:

```bash
uv sync --frozen --extra test --extra spage && uv run --no-sync pytest tests/test_common.py tests/test_spage_generator.py
```

To keep the environments side by side locally, give each its own directory with
`UV_PROJECT_ENVIRONMENT=.venv-spage`. The hatch environments of the same names are a shortcut,
`hatch run spage:test` and `hatch run spage:generate --out spage_reference.zarr.zip`, which also set
the variables the generators require (single-threaded BLAS), but
hatch resolves fresh versions rather than reading `uv.lock`. `tests/test_common.py` guards what the
methods share and needs only the base install and the `test` extra; `tests/test_<method>_generator.py` guards each
generator and needs the `vendor/<upstream>` submodule.

## Publishing the reference bundle

The `Generate reference bundle` workflow writes one bundle per method, each in its own job. For
STalign it writes `stalign_reference.zarr.zip`, an AnnData written to a zipped zarr store (each
section a dict of arrays in `uns`, with its provenance; the fixture parameters in
`uns["fixtures"]`), checks
that a second run gives identical arrays, and uploads it as a run artifact. Publish it by hand:

```bash
gh workflow run generate.yaml -R theislab/squidpy-ports
gh run download <run-id> -R theislab/squidpy-ports -n stalign-reference-bundle
gh release create stalign-reference-vN stalign_reference.zarr.zip -R theislab/squidpy-ports \
  --notes "Generated by run <run-id>."
```

Then point squidpy's `test_stalign_reference.py` at the new URL and the sha256 the workflow
prints: it pins that one build (provenance timestamps differ between builds, the arrays do not). It runs on ubuntu because BLAS/FFT reduction order
differs across platforms. Locally, the workflow's commands write the same store from the same lockfile.

## SpaGE

squidpy ports SpaGE gene imputation as `squidpy.experimental.tl.impute_spage`. Upstream
([tabdelaal/SpaGE]) is vendored at `vendor/SpaGE`, pinned to `edd0fb7`, and imported by path: it
has no `setup.py`, and `main.py` imports its siblings as `SpaGE.*`.

`SpaGE()` hard-codes a cosine threshold of 0.3 and 50 neighbors. `spage.upstream.run` swaps both
for the length of one call by replacing `PVComputation` and `NearestNeighbors` in `SpaGE.main`'s
namespace, and observes the number of principal vectors used, the cosine similarities, and the
query cells whose `n_neighbors`-th neighbor ties with the next one. At the hard-coded settings it
returns exactly what `SpaGE()` does, which `tests/test_spage_generator.py` asserts.

`spage_reference.zarr.zip` is the same format as the STalign bundle:

| `uns` key | What it holds |
| --- | --- |
| `fixtures` | `SEED`, `N_PV`, `COSINE_THRESHOLDS` and `N_NEIGHBORS` (one entry per run), the gene names (`genes`, `measured`, `unmeasured`), and the checksum of `spage/fixtures.py`. |
| `inputs` | `ref` (300 x 12), `query` (150 x 8, the first 8 genes), and `truth`, the query's true values of the 4 unmeasured genes. The same arrays as `_data()` in squidpy's `test_impute_spage.py`. |
| `cosine{30,99}_k{20,50,300}` | One run of upstream at `n_pv=3`: `imputed` (150 x 4), `n_pv_used`, `cosine`, `tied_rows`, `cosine_threshold`, `n_neighbors`, and the provenance. |

The fixture has two reference cells that are zero on every measured gene, so they are at the same
distance from every query cell. Where they straddle the `n_neighbors`-th neighbor, which one is kept
is the search backend's choice, so a correct port may differ from upstream on exactly the rows in
`tied_rows`, and must match it everywhere else.

The workflow publishes it as the `spage-reference-bundle` artifact; release it as for STalign, with a
`spage-reference-vN` tag.

[tabdelaal/SpaGE]: https://github.com/tabdelaal/SpaGE

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[api]: https://theislab.github.io/squidpy-ports/squidpy-api.html
[ports-tests]: https://github.com/theislab/squidpy-ports/blob/main/tests/test_stalign_generator.py
[bundle]: https://theislab.github.io/squidpy-ports/reference-bundle.html
[ref-tests]: https://github.com/scverse/squidpy/blob/main/tests/experimental/tl/test_stalign_reference.py
[container]: https://github.com/theislab/squidpy-ports/blob/main/container/README.md
[license-vendor]: https://github.com/theislab/squidpy-ports/blob/main/LICENSE.vendor
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/squidpy-ports/issues
[tests]: https://github.com/theislab/squidpy-ports/actions/workflows/test.yaml
