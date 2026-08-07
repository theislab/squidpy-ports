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

Nothing here reimplements upstream. Every value is produced by calling a public upstream
function or by observing the unmodified `LDDMM` loop through autograd hooks — a
reimplementation would make *this* repository the thing under test. The single
documented exception (the four statements computing `LL`/`K`/`DV`, which live inside the
loop with no function boundary) is quoted verbatim with line references.

## Regenerating

```bash
git submodule update --init
hatch run generate:run --out ../squidpy/tests/_data/stalign_reference
```

Then commit the bundle in squidpy. Every `.npz` carries a `__provenance__` JSON blob
recording the upstream commit, this repo's commit, a checksum of `fixtures.py`, and the
resolved torch/numpy/platform — squidpy asserts these on load, so the fixtures stay
falsifiable rather than becoming magic numbers.

### Determinism

Upstream is bit-reproducible on CPU in float64 (no RNG, no dropout, no atomics), *given*
a fixed reduction order. The `generate` hatch environment pins `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, and the script refuses to run otherwise.

**Reduction order still differs across platforms** (macOS Accelerate vs Linux OpenBLAS).
squidpy's scheduled job runs on ubuntu, so the committed bundle should be generated on
ubuntu too: trigger the `generate` workflow and download its artifact rather than
committing a bundle built on a developer laptop.

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[license-vendor]: https://github.com/theislab/squidpy-ports/blob/main/LICENSE.vendor
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/squidpy-ports/issues
[tests]: https://github.com/theislab/squidpy-ports/actions/workflows/test.yaml
