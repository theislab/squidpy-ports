# STalign upstream notebook comparison status

The comparison harness covers all 17 notebooks from upstream STalign at
`b2068edc98974efa54537eca194736e177bbe11d`. It runs the original preprocessing,
captures the upstream fit inputs, and compares the PyTorch result with Squidpy's
JAX implementation on H100 GPUs -- one Slurm array task per notebook.

## Complete suite, job array 38957316

Working notes, kept out of the published docs. The numbers below are from job array
38957316, whose result directory has since been superseded by
`docs/notebooks/stalign-upstream/results-38960447/` (same notebooks, later fork) — see
`HANDOFF.md` §2 for the current table. Full-resolution PNG
panels and GPU monitoring stay on cluster storage under
`stalign-full-suite/source-20260731-019fb8c6/squidpy-ports/cluster-results/38957316`; the
notebooks embed a web-resolution copy of the same panel.

| notebook | status | warped density | aligned points | matching weights |
| --- | --- | --- | --- | --- |
| `visium-visium-alignment-affine-only` | compared | 1.71e-07 | 3.61e-08 | 0 |
| `merfish-merfish-alignment` | compared | 2.07e-06 | 2.38e-07 | 3.32e-19 |
| `merfish-xenium-alignment` | compared | 9.49e-06 | 4.51e-07 | 1.09e-18 |
| `merfish-merfish-alignment-using-L-T` | compared | 1.13e-05 | 3.84e-07 | 2.88e-19 |
| `heart-alignment-varying-thickness` | compared | 1.48e-05 | 8.33e-07 | 6.79e-16 |
| `heart-alignment` | compared | 2.03e-05 | 1.13e-06 | 1.94e-16 |
| `merfish-merfish-alignment-simulation` | compared | 3.71e-05 | 1.57e-06 | 1.89e-19 |
| `merfish-merfish-alignment-affine-only` | compared | 4.39e-05 | 2.83e-06 | 3.33e-19 |
| `xenium-starmap-alignment` | compared | 7.52e-05 | 3.44e-06 | 7.81e-16 |
| `xenium-xenium-alignment` | compared | 1.29e-04 | 7.55e-06 | 6.80e-16 |
| `merfish-visium-alignment` | compared | 3.83e-04 | 1.38e-05 | 1.99e-06 |
| `merfish-visium-alignment-with-point-annotator` | compared | 5.80e-04 | 2.70e-05 | 1.28e-05 |
| `xenium-heimage-alignment` | compared | **1.88e-01** | 7.80e-03 | 6.13e-02 |
| `merfish-merfish-alignment-affine-only-with-points` | compared-affine | see below | | |
| `merfish-visium-alignment-with-curve-annotator` | unreplayable-upstream | — | — | — |
| `merfish-allen3Datlas-alignment` | unsupported-3d | — | — | — |
| `starmap-allen3Datlas-alignment` | unsupported-3d | — | — | — |

All figures are relative L2 against the upstream PyTorch result. The port is 7-34x faster
per fit (for example 707.5s to 8.9s on `merfish-merfish-alignment`).

**Run-to-run noise floor: ~1e-12 relative.** Rerunning `heart-alignment-varying-thickness`
unchanged on a different H100 reproduced its metrics to 12 significant digits, two of them
bit-identical. Divergences at 1e-5 are therefore real but tiny; the `xenium-heimage` result
is not arithmetic noise.

The affine landmark comparison recorded a linear relative L2 difference of `0.0034826207`,
translation relative L2 of `0.0037151724`, and a median aligned-landmark delta of `7.0977`
coordinate units -- reproducing the earlier partial job 38938878 to every recorded digit
(that partial result directory has been removed, superseded by this complete run).

## In flight at the last laptop switch (2026-08-20)

Written down because the last switch lost a day's work. Everything below is pushed; the only
thing not in git is the cluster output, which is on Lustre.

- **Canonical sweep `39759108`** -- all 17 notebooks, clean squidpy `2bff194f` from
  `stalign-3d/squidpy-clone` (a real git checkout, so manifests resolve provenance
  themselves), harness fix `9f1d087`, determinism pinned, convergence panels on. This is the
  run meant to replace `docs/parity/` and the executed notebooks on the site. **The docs pass
  had not been done yet**: sync the 17 metrics/manifests/statuses plus
  `notebooks/*.ipynb` into `docs/`, regenerate the two tables in `docs/parity.md` (largest
  `<var> relative L2`, excluding `*published*`), rebuild under `-W`, commit, push.
- **D11 flag runs `39758694-97`** -- `merfish-allen3Datlas` only, two reps each of flag off /
  on, serialized on `supergpu21`. Numbers and what they do and do not support are in the TODO
  above.
- **`SQUIDPY_STALIGN_UPSTREAM_REG_ENERGY_AXES`** lives on squidpy commit `b10e35a1`
  (`feat/experimental-fit-core`), **committed locally and never pushed**. Comparison-only. If
  it is not pushed, the D11 measurement is not reproducible by anyone else -- decide either
  way, but do not leave it dangling.
- **Cluster layout**: `stalign-3d/squidpy-clone` (git, canonical), `squidpy-regaxes` (rsynced,
  carries the flag), `squidpy-2bff194f` (older rsync), `squidpy-ports` (rsynced working copy).
  Prefer the clone and `git fetch` from now on -- the rsync route once shipped 2.1GB of
  `.pytest_cache`/`.tox`/`.mypy_cache`, and deleting them mid-run killed three tasks by
  racing the staging tar.
- **Two things to tell the STalign authors**, both measured and independent of each other:
  `LDDMM_3D_to_slice` is non-reproducible on CUDA (245um, 29.5% of regions, same node,
  determinism pinned, no RNG in their source, CPU bit-identical); and its rank-3
  regularisation energy transforms two of three spatial axes at `STalign.py:1504` -- the line
  is byte-identical to the rank-2 one at `:1193` where two axes is all of them -- while the
  gradient it descends smooths all three at `:1527`. Correcting that moves the fitted
  velocity field by 31x. The paper could not be checked: Nature is behind auth, PMC behind a
  captcha, bioRxiv 403s.

## TODO

- [ ] **Rewrite the remaining sixteen notebooks against the public API.** The template is
  `docs/notebooks/squidpy-api/starmap-allen3Datlas.ipynb` (pushed, executed on an H100). It
  shows the shape: no upstream call, no replay harness, `rasterize_points` ->
  `align_stalign_volume` -> `Stalign3DResult.transform` -> `sample_volume`. What is left:

    - `merfish-allen3Datlas` -- the other rank-3 one, same shape, `dx=10` so no D14 exposure.
    - the fourteen rank-2 notebooks -- these map onto `align_stalign_image` /
      `align_stalign_obs`, a different and simpler shape than the volume case. Not attempted.
    - `merfish-merfish-alignment-affine-only-with-points` -- landmark affine only,
      `align_landmarks(fit="affine")`.

  **Three things to settle before writing sixteen of them, because each one shapes all of them:**

    1. *What is the reference now.* Upstream is no longer called in the notebooks, so
       "upstream pass vs squidpy pass" is not the unit any more. The comparison either lives
       only in `tests/test_stalign_reference.py`, or against committed reference values, or the
       notebooks stop being a comparison. Not decided.
    2. *`docs/parity.md` and `docs/stalign-comparison.md` still assume the replay harness* --
       they are organised around per-cell figure pairing and per-variable metrics keyed off
       upstream's own variable names. If the notebooks are public-API, both pages collapse into
       one and most of `notebook_suite.py` (`_namespace_metrics`, `_paired_frames`,
       `_require_same_cells_ran`, `_compose_pair`) has no caller.
    3. *Two things have no squidpy equivalent* and would have to be written as notebook code:
       `plot_brain_regions` (the legend figure), and the `coord0/1/2` naming every current
       parity metric keys off. `sample_volume` returns integer structure ids; the acronym
       lookup is a `pandas` map against the Allen ontology CSV.

  Two traps the template already documents, worth carrying into every notebook: `initial_scale`
  is uniform so it cannot express anisotropy (use `initial_affine`), and an element's `.coords`
  are pixel indices rather than microns -- reading them as microns turned 425um of depth into
  140 and the fit silently never recovered.

  Runtime, for planning: the volume notebook is **~11 min on a laptop** and **under 90s** on an
  H100 through the held-node workflow (`d11run.sh`), with squidpy installed straight from a git
  rev -- `uv pip install "squidpy @ git+.../squidpy.git@<rev>"`, no rsync.

- [ ] **Run the D13 flag sweep.** `sbatch .claude/run_stalign_d13_flag.sbatch` -- written, never
  submitted. Ledger row D13: both `allen3Datlas` notebooks pass a length-3 `muA`/`muB` against a
  single-channel `J`, upstream sums over the broadcast axis (`STalign.py:1554-1555`) so its
  artifact and background widths are effectively `sigma/sqrt(3)` while `WM`'s is not, and the
  replay collapses the mean for the port because squidpy validates the length. A candidate cause
  of the rank-3 divergence *independent of D11*, with no number yet. The job collapses upstream's
  mean instead of expanding the port's, so it needs no squidpy change and is reproducible from a
  clean checkout -- unlike the D11 flag below, which is still on an unpushed commit. Runs on
  `cpu_p`, no GPU: that is route 2 of the item below, taken for this experiment first because a
  standing reference is cheaper than averaging a moving one. Read the result with
  `python3 src/squidpy_ports/stalign/flag_report.py <output>`; if the two replicates of a
  condition disagree at all, the CPU-determinism premise is wrong and that has to be chased
  before any number from the sweep is quoted.

- [ ] **Attribute the rank-3 per-cell region disagreement, or stop quoting it.** The D11
  flag experiment (jobs 39758694-97, `merfish-allen3Datlas`, two reps per condition on one
  pinned node) settled the velocity field: `v` relative L2 goes **1.81 -> 0.055** when
  squidpy reproduces upstream's two-axis energy, an effect **159x** the run-to-run spread,
  and squidpy's own reported objective flips from **+3.66%** total drop to **-5.15%**,
  bit-identical across reps. Both stand.

  What does *not* stand is the region-label claim. With one rep it looked like
  49.8% -> 18.1%; the second rep came back at 42.2%, so the effect (0.194) is smaller than
  the spread (0.241). Same for `coord2`: 7.9um then 110.6um. The reason is that every metric
  measured *against* upstream inherits upstream's CUDA non-reproducibility, which on these
  notebooks moves cells up to 245um and reassigns 15-22% of regions between two runs of
  upstream alone (39748939 vs 39756626, same node, determinism pinned). The reference is the
  thing moving, so a 30-point effect cannot be read from a handful of samples.

  Two routes, and the second is much cheaper than it sounds:
    1. Many more reps per condition, enough to average upstream's wander down.
    2. **A deterministic upstream reference.** Upstream's `LDDMM_3D_to_slice` is
       bit-identical across processes on **CPU** (`A` and `v` to the byte, checked twice) and
       only wanders on CUDA. Running the upstream pass on CPU makes the reference stand
       still, at the cost of wall-clock (~104s -> minutes on these notebooks). That turns
       every rank-3 port-vs-upstream metric from indicative into measured, and it is the
       prerequisite for any claim about region assignment.

  Until one of those happens, quote `v` and the energy sign flip for D11, and treat the
  label and depth numbers as unattributed.

- [x] Preserve Python scalar types in `notebook_suite._jax_kwargs`. Every captured
  argument is now cast to the type the port declares for that parameter, so `niter=100`
  and `diffeo_start=100` stay Python `int`s that `jax.jit` can hash as static arguments,
  while `muA`/`muB`, the landmarks, and a warm-start velocity stay arrays. Reading the
  cast from the port's own signature keeps the two from drifting apart silently.
- [x] Add a regression test for captured scalar, array, affine, point, fixed
  `muA`/`muB`, and warm-start velocity arguments before another cluster run. The port's
  signature is mirrored in `tests/test_stalign.py` so the conversion is testable where
  squidpy and JAX are absent, and `test_port_signature_matches_the_mirror` checks the
  mirror against the real signature wherever they are installed.
- [x] Emit a failure notebook and full traceback for every failed comparison so
  a partial cluster run remains self-explanatory. Each failure now writes
  `<notebook>-traceback.txt`, a manifest with `status: failed`, and a notebook carrying
  the traceback as an `error` output; `suite-status.json` is rewritten after every
  notebook, so a killed allocation still records what ran, what failed, and what was
  never reached.
- [x] Rerun all 14 two-dimensional LDDMM notebooks in the `gpu_normal` H100 job and commit
  their executed notebooks, metrics, and manifests. Run as a 17-task array (one notebook
  per task) rather than one serial job: the sweep is ~51,000 upstream iterations, and one
  hung notebook would otherwise spend the whole allocation. Reruns of a subset are
  `sbatch --array=<indices>`.
- [ ] Review numerical divergence notebook by notebook, with particular
  attention to the partially overlapping Xenium/Xenium sections and matching
  weights rather than plot orientation alone. See "Divergence review" below --
  `xenium-heimage-alignment` is the one open question.
- [x] Replace the generated wrapper notebooks with the final executed versions
  after the complete suite succeeds. The panel embedded in each notebook is
  web-resolution (`EMBED_DPI`, 256 colours) while the archival PNG beside it stays at
  `ARCHIVE_DPI`; emitting the suite at archival resolution added ~32 MB per rerun.

## Divergence review

Ten of the thirteen compared notebooks agree with upstream to 1e-5 or better on warped
density and 1e-6 on aligned points, with matching weights at machine precision
(1e-16..1e-19). Against a measured ~1e-12 noise floor these are real but negligible.

`xenium-xenium-alignment` was called out for review because its two sections overlap only
partially. It lands at 1.29e-04 density with matching weights at 6.80e-16 -- machine
precision. Since the weights are what identify the supported overlap, the partial overlap
is being handled identically by both implementations, and the density figure reflects
unmatched cells rather than a disagreement.

`merfish-visium-alignment` (3.83e-04) and `merfish-visium-alignment-with-point-annotator`
(5.80e-04) are the next largest. Both set all three mixture sigmas to ~0.18-0.2, where the
E-step exponent carries a factor `1/(2*sigma^2)` of ~15 and the three-component posterior
is close to degenerate, so the weights are ill-conditioned by construction.

`xenium-heimage-alignment` is the outlier and remains open: 1.88e-01 density, 7.80e-03
points, and 6.13e-02 matching weights, with a maximum weight difference of 0.78 -- at some
pixels the two implementations assign opposite mixture classes. The pointwise panel shows a
smooth spatial gradient of 20-80 coordinate units over a ~8000-unit domain, the signature of
a slightly different transform rather than noise. What has been ruled out:

- **Not the mixture E step.** The affine divergence is already 3.9e-04 at iteration 49,
  before the E step first runs at iteration 50 (`MIXTURE_E_STEP_START`).
- **Not the landmark term.** Refitting with `pointsI`/`pointsJ` removed leaves it unchanged
  (6.89e-04 against 3.93e-04 at iteration 49). An apparent correlation across notebooks
  between "passes landmarks" and "diverges more" is therefore confounding, not causal.
- **Not a mean-estimation or cadence difference.** This is the only notebook fixing both
  `muA` and `muB`, but `estimate_muA`/`estimate_muB` gate only the mean M step in the port,
  not the E step, and the every-fifth-iteration cadence and `(2*pi*sigma^2)^(C/2)`
  normaliser both match upstream.

What is established is that the divergence accumulates from early iterations rather than
appearing at one step: machine precision at iteration 1, ~4e-04 by iteration 49. The
velocity field is the remaining suspect and needs a comparison that accounts for upstream
returning `A` one iteration behind `v` and `WM`.

## Not planned

`merfish-allen3Datlas-alignment.ipynb` and `starmap-allen3Datlas-alignment.ipynb` call
upstream's `LDDMM_3D_to_slice`, a separate volume-to-slice solver. Squidpy's port
implements two-dimensional LDDMM only, and porting the 3D estimator is out of scope for
this comparison. Both notebooks therefore stay at status `unsupported-3d` permanently:
the suite emits an explicit status panel for each rather than a number that would imply
a comparison it did not make.
