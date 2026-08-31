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
| **Numerical tests** | Two suites, split by what they are allowed to touch: [`tests/test_generator.py`][ports-tests] guards this repo's own inputs — the vendored pin, the fixture definitions, the provenance stamp; [`tests/test_reference.py`][ref-tests] asserts the port against upstream at rank 2 and rank 3. |
| **Regenerate the reference bundle** | [Running the tests](#running-the-tests) — the command, the provenance blob, and the platform caveat that matters before you commit one. |
| **Reproduce on any GPU box** | [`container/README.md`][container] — one pinned container, any GPU box. |

The executed notebooks are corroboration, not the gate. The gate is
`tests/test_reference.py`, which compares the port at every layer — rasterisation, energy,
gradients, trajectory, converged fit, image warp, and the section-into-volume fit. At rank 2 it
matches upstream to ~1e-15 on the velocity field. At rank 3 it matches the solver on the seeded
fixture, but the two implementations descend on different objectives once the velocity moves:
upstream regularises over two spatial axes while smoothing that energy's gradient over three. That
divergence is pinned by a strict xfail rather than asserted away, and the xfail's own docstring
says what it costs.

**Which suite a test belongs to.** squidpy owns the public path: `align_stalign_*`,
`align_landmarks`, `rasterize_points`, `sample_volume`, asserted through their public entry points
only. This repository owns everything that imports a private name or compares against a value
upstream computed — which is the whole of `tests/test_reference.py`, and why it lives here rather
than in squidpy's CI.

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
observing the unmodified `LDDMM` loop.

## Running the tests

This repo's own suite guards the generator. It needs nothing but a clone:

```bash
pytest
```

One test skips without squidpy and JAX installed — deliberately, since this repo does not depend on
either. Install both and it runs.

`tests/test_reference.py` is the layer that asserts the *port*. It needs squidpy and JAX,
so it skips without them; install squidpy and it runs. Every upstream value it compares against is
computed **in this process** from the vendored checkout, by the same generator that writes the
shareable bundle — so no committed binary can go stale, here or in squidpy. No GPU: float64 on CPU,
about 80 s on a laptop.

```bash
uv pip install -e ".[test]" -e /path/to/squidpy"[jax]" pytest
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=1 pytest tests/test_reference.py
```

> **`JAX_ENABLE_X64=1` is not optional.** Upstream is double throughout; without it the port runs in
> single precision and every tolerance in the suite is meaningless. The suite skips rather than
> lying if the flag is absent, and it has to come from the environment — `jax.config.update` is
> process-global and would flip the float32 tests in squidpy's own suite in the same worker.

The `.npz` bundle is still what a *shareable* reference looks like, and
`python -m squidpy_ports.stalign.generate --out DIR` writes it. Nothing in this repo's tests reads
it from disk; it exists for anyone who wants the numbers without a torch install.

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[api]: https://theislab.github.io/squidpy-ports/squidpy-api.html
[ports-tests]: https://github.com/theislab/squidpy-ports/blob/main/tests/test_generator.py
[ref-tests]: https://github.com/theislab/squidpy-ports/blob/main/tests/test_reference.py
[container]: https://github.com/theislab/squidpy-ports/blob/main/container/README.md
[license-vendor]: https://github.com/theislab/squidpy-ports/blob/main/LICENSE.vendor
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/squidpy-ports/issues
[tests]: https://github.com/theislab/squidpy-ports/actions/workflows/test.yaml
