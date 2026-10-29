# squidpy-ports

[![Tests][badge-tests]][tests]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/squidpy-ports/test.yaml?branch=main

Reference outputs for squidpy's ported algorithms.

squidpy reimplements some methods that originate elsewhere. A reimplementation that
merely produces plausible, correctly-shaped output can drift from the algorithm it
claims to be without anyone noticing. This repository runs the *original* implementation
— vendored and pinned — and emits the numbers squidpy's tests compare against, so the
heavyweight reference dependencies never enter squidpy itself.

Currently covers: **STalign** (see [scverse/squidpy#1243]).

[scverse/squidpy#1243]: https://github.com/scverse/squidpy/issues/1243

## Results & reproduction

| | |
| --- | --- |
| **squidpy API equivalence** | [Sixteen of STalign's seventeen analyses][api], rewritten against `squidpy.experimental.tl` and executed on a GPU — one page per upstream notebook, each linking the original it mirrors. |
| **Numerical tests** | [`tests/test_generator.py`][ports-tests] guards this repo's own inputs — the vendored pin, the fixture definitions, the provenance stamp. The port itself is asserted against upstream at rank 2 and rank 3 in squidpy, by [`test_stalign_reference.py`][ref-tests], which downloads the bundle this repo publishes. |
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

This repository is BSD-3-Clause. The vendored upstream at `vendor/STalign` is
**GPL-3.0** ([`LICENSE.vendor`][license-vendor]) and is included as a git submodule, not
redistributed as part of any built artifact. Nothing is published to PyPI.

The generated `.npz` bundle holds numerical outputs computed from synthetic, seeded
inputs defined in this repository — no upstream data is copied or redistributed.

## Layout

| Path | What it is |
| --- | --- |
| `vendor/STalign` | Upstream, pinned to one commit. Never edited, never linted, never pip-installed. |
| `src/squidpy_ports/stalign/fixtures.py` | Input definitions, shared by the generator and the tests; depends on numpy only. |
| `src/squidpy_ports/stalign/upstream.py` | Loads upstream by path, asserts the pin, and observes the LDDMM loop. |
| `src/squidpy_ports/stalign/generate.py` | Writes the `.npz` bundle. |

Upstream is imported by path rather than installed because its `setup.py` reads
`requirements.txt` verbatim and pins `numpy==1.23.4` / `torch==2.0.0` / `Pillow==9.5.0`,
none of which resolve on a current Python. Vendoring makes those pins irrelevant;
`uv.lock` supplies the reproducibility instead.

Nothing here reimplements upstream — every value comes from calling a public upstream function or
observing the unmodified `LDDMM` loop.

## Running the tests

```bash
hatch test
```

`tests/test_generator.py` guards the generator and needs the `vendor/STalign` submodule.

## Publishing the reference bundle

Run the `Generate reference bundle` workflow with a release tag. It writes
`stalign_reference.zarr.zip` (one zarr group per section, its provenance in the group's
attributes, the fixture parameters in the root's), checks that a second run is bit-identical,
and publishes it as a release asset. Then point squidpy's `test_stalign_reference.py` at the new
URL and the sha256 the workflow prints. It runs on ubuntu because BLAS/FFT reduction order
differs across platforms; `hatch run generate:run --out stalign_reference.zarr.zip` writes the
same store locally.

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[api]: https://theislab.github.io/squidpy-ports/squidpy-api.html
[ports-tests]: https://github.com/theislab/squidpy-ports/blob/main/tests/test_generator.py
[bundle]: https://theislab.github.io/squidpy-ports/reference-bundle.html
[ref-tests]: https://github.com/scverse/squidpy/blob/main/tests/experimental/tl/test_stalign_reference.py
[container]: https://github.com/theislab/squidpy-ports/blob/main/container/README.md
[license-vendor]: https://github.com/theislab/squidpy-ports/blob/main/LICENSE.vendor
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/squidpy-ports/issues
[tests]: https://github.com/theislab/squidpy-ports/actions/workflows/test.yaml
