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
| **Visual comparison** | [Upstream's figure beside ours][comparison] — three real datasets: upstream's published figure beside ours, with the numerical agreement. |
| **Notebook values comparison** | [Every variable in all 17 upstream notebooks][parity]; the per-variable metrics and manifests behind it are in [`docs/parity/`][parity-data]. |
| **Reproduce any panel on any GPU box** | [`container/README.md`][container] — one pinned container, any GPU box. |
| **Regenerate the reference bundle** | [Numerical tests][tests-section] — the command, the provenance blob, and the platform caveat that matters before you commit one. |
| **Numerical tests** | [The test suites][tests-section], layer by layer: [`tests/test_stalign.py`][ports-tests] here guards the generator; the fork's [`test_stalign_reference.py`][ref-tests] and [`test_align.py`][api-tests] assert the port against it. |

The panels are corroboration, not the gate. The gate is the `.npz` bundle this repo emits:
squidpy replays it at every layer — rasterisation, energy, gradients, trajectory, converged
fit, image warp — and matches upstream to ~1e-15 on the velocity field.

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
| `src/squidpy_ports/stalign/fixtures.py` | Input definitions. **Vendored byte-identically into squidpy**; depends on numpy only. |
| `src/squidpy_ports/stalign/upstream.py` | Loads upstream by path, asserts the pin, and observes the LDDMM loop. |
| `src/squidpy_ports/stalign/generate.py` | Writes the `.npz` bundle. |

Upstream is imported by path rather than installed because its `setup.py` reads
`requirements.txt` verbatim and pins `numpy==1.23.4` / `torch==2.0.0` / `Pillow==9.5.0`,
none of which resolve on a current Python. Vendoring makes those pins irrelevant;
`uv.lock` supplies the reproducibility instead.

Nothing here reimplements upstream — every value comes from calling a public upstream function or
observing the unmodified `LDDMM` loop. [Numerical tests][tests-section] has the reasoning, the one
documented exception, and how to regenerate the bundle.

## Running the tests

This repo's own suite guards the generator and the replay harness. It needs nothing but a clone:

```bash
pytest
```

One test skips without squidpy and JAX installed — deliberately, since this repo does not depend on
either. Install both and it runs.

The suites that assert the *port* against upstream live in the squidpy fork, and need its reference
bundle. No GPU: they are float64 on CPU and land identically on an H100 (50 s) and on a laptop
(25 s).

```bash
git clone https://github.com/selmanozleyen/squidpy .squidpy-fork
git -C .squidpy-fork checkout 6a63ff8
uv pip install -e ".[test]" "./.squidpy-fork[jax]" pytest
cd .squidpy-fork && JAX_ENABLE_X64=1 pytest -m "reference or not reference" \
    tests/experimental/methods/test_stalign_reference.py \
    tests/experimental/tl/test_align.py
```

> **`-m "reference or not reference"` is not optional.** The reference suite is marked
> `pytest.mark.reference` and the fork's `addopts` carry `-m "not reference"`, so a plain `pytest`
> **silently deselects all 62 of them** — the entire layer that asserts the port against upstream —
> and still reports a green run.

### How the results table is generated

The table on [Numerical tests][tests-section] is not written by hand. Run the suites above with
`--junitxml`, then:

```bash
python -m squidpy_ports.stalign.test_report \
    --suite "this repo:tests/test_stalign.py:ports.xml" \
    --suite "reference suite:.squidpy-fork/tests/experimental/methods/test_stalign_reference.py:fork.xml" \
    --suite "public API:.squidpy-fork/tests/experimental/tl/test_align.py:fork.xml" \
    --output docs/_static/tests/results.md
```

Statuses and expected-failure reasons come from the JUnit XML; each test's description is its own
docstring, read from the source. Nothing paraphrases a test, so a wrong description is a wrong
docstring. The raw pytest logs are committed beside the generated table in
[`docs/_static/tests/`][test-reports].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[comparison]: https://squidpy-ports.readthedocs.io/en/latest/stalign-comparison.html
[tests-section]: https://squidpy-ports.readthedocs.io/en/latest/correctness.html
[test-reports]: https://github.com/theislab/squidpy-ports/tree/main/docs/_static/tests
[ports-tests]: https://github.com/theislab/squidpy-ports/blob/main/tests/test_stalign.py
[ref-tests]: https://github.com/selmanozleyen/squidpy/blob/6a63ff8/tests/experimental/methods/test_stalign_reference.py
[api-tests]: https://github.com/selmanozleyen/squidpy/blob/6a63ff8/tests/experimental/tl/test_align.py
[parity]: https://squidpy-ports.readthedocs.io/en/latest/parity.html
[parity-data]: https://github.com/theislab/squidpy-ports/tree/main/docs/parity
[container]: https://github.com/theislab/squidpy-ports/blob/main/container/README.md
[license-vendor]: https://github.com/theislab/squidpy-ports/blob/main/LICENSE.vendor
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/squidpy-ports/issues
[tests]: https://github.com/theislab/squidpy-ports/actions/workflows/test.yaml
