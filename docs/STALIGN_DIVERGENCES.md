# STalign port — divergence ledger

Where squidpy's JAX port (`src/squidpy/experimental/tl/_align/_stalign_impl/`) differs from
the PyTorch original it was ported from, why, and what was done about it.

Every number below is **measured**, not estimated, by
[`tests/test_stalign_reference.py`](../tests/test_stalign_reference.py). That suite computes
each upstream value **in this process** from the vendored checkout in `vendor/STalign`
(pinned at `b2068edc98974efa54537eca194736e177bbe11d`), by the same generator that writes
the shareable bundle — `squidpy_ports.stalign.generate` — so no committed binary can go
stale here or in squidpy. Upstream line references are into that commit's
`STalign/STalign.py`.

Each row has an id. Every `xfail(strict=True)` in the test module cites its row id, and
`test_divergences_doc_covers_all_xfails` asserts the citation resolves — so this file
cannot silently rot.

Comparisons go through the public API wherever one exists: point transforms via
`Stalign2DResult.transform`, the image warp via `Stalign2DResult.warp_image`, the landmark
initialisation via `align_stalign_obs(..., niter=0)`, and at rank 3 the reference sampling
via `squidpy.experimental.im.sample_volume`, and the landmark affine via
`align_landmarks(fit="affine")`. What stays white-box is what has no
public route — the objective and its gradients are not observable from outside, and the
regulariser and grids are preconditions rather than results.

## Where it stands

The port now reproduces the original **to machine precision** on everything that defines
the optimisation:

| | relative error |
| --- | --- |
| objective `E` | **0.0** (bit-for-bit, `131.057468`) |
| `dE/dL`, `dE/dT`, `dE/dv` | **≤ 6e-15** |
| `_interp` | 4e-16 |
| `_transform_grid_backward` | 5e-18 |
| forward point transform | 5e-17 |
| regulariser `LL` / `K` / `DV` | ≤ 1e-15 |
| raster and velocity grids | identical |
| full loop, 1 / 5 / 50 steps | ≤ 1e-10 / 1e-9 / 1e-6 on `A`, `v`, `WM`, `WA`, `WB` |
| converged (500 steps) | final `E` within 1 %; 95th-percentile point disagreement < 0.1·`dx` |

That table is **rank 2** — a section fitted to a section, upstream's `LDDMM`. Rank 3 — a
section fitted into a volume, upstream's `LDDMM_3D_to_slice` — reaches the same precision on
everything the two implementations share: ≤ 1e-12 on the velocity grids, the regulariser
build, the interpolation, the backward grid integration and both public transforms, and
≤ 1e-10 on the 12-step path and the converged affine. It diverges on one thing, by
construction and on purpose: **D11**.

The 50-step case matters on its own: the mixture-weight E step is gated on `it >= 50`
(`STalign.py:1233`), so a shorter run leaves that branch entirely untested.

Three findings were real bugs and are **fixed** (D2, D3, R1). One is a deliberate
approximation, now measured and budgeted (D1). The rest are places where squidpy is
deliberately *not* bug-compatible with upstream (D4, D5, D6, D7, D9) — those stay, pinned.

## Ledger

| id | What | Upstream | squidpy | Measured | Status |
| --- | --- | --- | --- | --- | --- |
| **D1** | Rasterisation algorithm | exact sub-pixel Gaussian splat per point, truncated to ±`ceil(4·max(blur))` px and renormalised over that window — `:178-201` | bilinear deposit onto the grid, then one `scipy.ndimage.gaussian_filter` per scale, mass-corrected — `_stalign_impl/_helpers.py` | relL2 **4.08 % / 0.81 % / 2.87 %** at blur 2.0/1.0/0.5; mass exact | **Accepted, budgeted.** Improved from 6.20/1.98/5.96 % — see R2 |
| **D2** | Grid construction off-by-one | `np.arange(lo, hi, dx)` — `:137-138`, `:1069` | was `np.arange(lo, hi + dx, dx)` | was +1 sample per axis, and *length varied with rounding* — `n+1` in 7 of 9 sweep cases, `n+2` in 2 | **Fixed.** Count derived from the interval first; grids now identical to upstream |
| **D3** | Gradient of the contrast transform | ridge coefficients solved under `torch.no_grad()` — `:1184-1188` | was differentiating through `jnp.linalg.solve` | was `dE/dL` 1.25e-3, `dE/dT` 9.98e-4, `dE/dv` 1.22e-3; now **≤ 6e-15** | **Fixed** with `jax.lax.stop_gradient`. The ridge fit is an EM **M step**; differentiating through it turned alternating minimisation into joint optimisation |
| **D4** | Returned affine lags a step | `A = to_A(L,T)` built at the *top* of the loop and returned — `:1155`, `:1308` | built after the loop | `LDDMM(n)["A"]` reflects `n-1` updates | **squidpy is right** — upstream discards its last step. Comparisons use `squidpy(n).A ↔ upstream(n+1).A` |
| **D5** | Padding when sampling outside the domain | `grid_sample` default `padding_mode='zeros'` for velocity *and* point warps (`:1163`, `:1167`); only the image warp uses `'border'` (`:1171`) | `map_coordinates(mode="nearest")` (≈`border`) everywhere | outside the domain: **25 %** from upstream-`zeros`, **2.8e-16** from upstream-`border` | **squidpy is right.** Upstream's zeros make a point that drifts off the velocity grid snap to *no* displacement — a discontinuity. Pinned `xfail` |
| **D6** | Backward point transform, time order | integrates `-v[t]` for `t` in `range(nt)` — forward order — `:1828-1843` | `reversed(range(nt))` | outputs differ by **1.03e-6**; forward∘backward round-trip error **5.79e-7 (squidpy)** vs **9.01e-7 (upstream)** | **squidpy is right** — it is the correct explicit-Euler inverse, and upstream contradicts its own image warp (`:1163`, which *does* reverse). Pinned `xfail` |
| **D7** | Landmark affine solve | normal equations for the plain least-squares fit, with explicit `np.linalg.inv` on the Gram matrix — `:897-910` | `skimage.transform.estimate_transform("affine")`, a Hartley-normalised homogeneous solve by SVD | `L` differs **6.4e-4**, `T` **7.2e-3**. Fit residual on clean landmarks: **21.7026** vs **21.6984** (1.9e-4 apart). On near-collinear landmarks: **7.4e-13** vs **5.6e+2** | **Different estimators, not the same one twice.** skimage minimises algebraic error, upstream geometric; upstream is a hair better on clean input and collapses when ill-conditioned. Keep squidpy's — see R7 |
| **D8** | `lddmm()` cannot take a precomputed `xv`/`v` | `LDDMM` accepts both — `:1060-1064` | no such parameters | — | **No change needed.** The generator forces *upstream* onto squidpy's grid instead |
| **D9** | Division guards | none | `jnp.maximum(…, 1e-12)` in `_update_mixture_weights` | inert for short runs | **squidpy is right.** Keep |
| **D10** | On-grid interpolation kink | normalises `(c-x0)/(x[-1]-x0)`, then `grid_sample(align_corners=True)` scales by `(n-1)` | `(c-x0)/(x[1]-x0)` | equal to ~1 ulp, but a sample landing exactly on a grid line can `floor()` to different neighbours. **Measured cost: 1e-12 → 1e-3** on the velocity field (see below) | Not a defect in either. Every fixture is built off-grid on purpose and asserts it |
| **D12** | ~~Velocity grid on `xenium-heimage-alignment`~~ — **not a port divergence: three bugs in this harness** | **17×23** cells, source spanning 4100×5516 expanded units at `dv=250` | **48×66** — the cell count of a grid over the *density's* 6000×8250 µm span, because the replay handed squidpy the density as its moving image | with all three corrected, the replay's own call reproduces upstream's affine to **1.7e-12** (**3.3e-12** with landmarks), against **3.65** as it stood. Pinned by `test_replay_image_call_reproduces_upstream`. Re-swept: `xenium-heimage` **1.8e+00 → 2.8e-01** and its fit **217s → 7s**, and *sixteen of seventeen* notebooks came back unchanged to the precision the table shows | **Fixed here, not in squidpy.** `_compare_lddmm` passed upstream's moving `I` as `ref` (the *fixed* side at rank 2), its row-col `pointsI`/`pointsJ` into `(x, y)` landmark parameters, and its row-col starting affine into `initial_affine`, which reverses the axes itself. Three mirrored errors that nearly cancel wherever `I` and `J` are similar rasters on centred pixel axes — fourteen of seventeen notebooks — and cancel not at all for an H&E in pixels against a density in microns. `70069fc9` in squidpy exposed them by reading each element's real placement instead of building centred pixel axes, which had collapsed the swapped grid to a harmless near-rigid **2×3**. Rank 3 was already correct and is unchanged |
| **D11** | Rank-3 regulariser axes | the regularisation *energy* transforms two of the three spatial axes (`dim=(1,2)`, `:1504`) while the smoothing applied to that same energy's gradient spans all three (`dim=(1,2,3)`, `:1527`) | all three spatial axes in both places | upstream's regularisation energy reads **~5.4×** the three-axis value on the reference velocity field; squidpy matches the three-axis value to **< 1e-12**. On a whole notebook: making squidpy reproduce upstream's two-axis energy moves the fitted velocity field by **31×**, `v` relL2 **1.81 → 0.055**, which is **159×** the run-to-run spread (`merfish-allen3Datlas`, two reps per condition, one pinned node) — and squidpy's own reported objective flips from **+3.66%** total drop to **−5.15%**, bit-identical across reps, which is the mismatch made visible: it descends the three-axis gradient while reporting the two-axis energy, so the number it prints climbs while it "improves" | **squidpy is right.** Kept, measured and pinned. Rank 3 only — at rank 2 the two readings coincide. **What this does *not* explain:** the per-cell region disagreement and the depth offset are *within* run-to-run noise here — every metric measured against upstream inherits upstream's CUDA non-reproducibility (245 µm, 15–22 % of regions between two runs of upstream alone), so the reference is the thing moving. Attributing those needs a deterministic upstream reference; see the TODO in `.claude/stalign-divergence-notes.md` |

## Review — beyond the divergences

Port-quality findings, not upstream comparisons.

### R1. `lddmm(niter=0)` raised `UnboundLocalError` — fixed

`energy` and `transformed_points` were bound only inside the loop, so the `return`
read unbound locals. `niter=0` is a reasonable request (evaluate the initial affine and
stop); both are now initialised before the loop, and
`test_lddmm_accepts_zero_iterations` covers it.

### R2. Rasteriser accuracy — fixed, ~2× better

The gap was two separable problems, dominating at opposite ends of the blur range.
Measured, all four combinations:

| variant | blur 2.0 | blur 1.0 | blur 0.5 | mass (of 800) |
| --- | --- | --- | --- | --- |
| nearest-cell + leaky border (was) | 6.20 % | 1.98 % | 5.96 % | 770.3 / 795.1 / 799.5 |
| bilinear deposit only | 6.34 % | 1.25 % | 2.88 % | leaks |
| mass conservation only | 3.92 % | 1.72 % | 5.97 % | exact |
| **both (now)** | **4.08 %** | **0.81 %** | **2.87 %** | **exact** |

1. **Sub-pixel quantisation.** `np.rint` snapped every point to a cell centre before any
   blurring — up to half a cell of positional error, comparable to the features being
   registered. Now deposited bilinearly across the four neighbouring cells, still fully
   vectorised (`np.bincount`, no Python loop over points).
2. **Border mass loss.** `mode="constant"` let kernels near the edge spill off-grid, so
   the density was biased low around the whole rim — 3 % of total mass at `blur=2.0`. The
   mass a point at cell `c` retains is `sum_p K(p-c)`, which by symmetry of `K` equals
   `gaussian_filter(ones)[c]`; dividing by that before blurring conserves mass exactly,
   for one extra filter pass.

Note the two are not additive: bilinear deposit alone made `blur=2.0` slightly *worse*,
because the error there was dominated by mass loss.

### R3. `blur` docstring — fixed

It said blur was "the kernel width in units of `2 * dx`", which reads as if `blur` were
scaled by `2·dx`. The code was always correct and matches upstream exactly; the docstring
now says σ = `2·blur` pixels = `2·blur·dx` physical.

### R4. The default dtype is float32; upstream is float64 throughout — documented

`jax_dtype()` returns `float32` unless `jax_enable_x64` is set, and nothing in squidpy
sets it — so the shipped default runs a 5000-iteration gradient descent with
`sigmaR=5e5` in single precision, while every published STalign result is double.
the `align_stalign_*` docstrings now say so and shows how to enable x64. This suite requires
`JAX_ENABLE_X64=1` and skips without it.

### D10 in practice: how the image path was nearly mis-assessed

`align_stalign_image` had no reference comparison at all — the last gap in the port's public
surface. Adding one produced a **5.3e-2** disagreement on the affine and **2.6e-1** on the
velocity field, which looked like a real port bug.

It was not. The bisect is worth recording, because the same trap is waiting for anyone who
adds a fixture without the off-grid guard:

| step | finding |
| --- | --- |
| energy at iteration 0 | matched to **4.6e-16** — the objective is identical |
| energy at iteration 1 | already 3e-3 apart — the *first gradient step* diverges |
| conditioning check | a 1e-12 nudge in `epV` moved the answer by 1e-13, linearly — not chaos |
| grids, axes, images, every parameter | verified identical on both sides |
| first hypothesis: padding | 11 % of the target grid samples the source out of domain, because each raster is centred on its own centre. **Wrong** — padding agrees on values *and* gradients |
| actual cause | centred pixel axes are **integers**, so an identity starting affine put every interpolation sample exactly on a grid line: D10, at full strength |

Starting the fixture from a deliberately off-grid affine (a 0.0371449 rad rotation and a
non-integer shift, mirroring `fixtures.THETA`/`SHIFT`) drops the disagreement to **3.95e-12**
on the affine and **2.43e-12** on the velocity — with the 11 % padded samples still present.

`image_trajectory_matched` keeps the degenerate case on purpose: both rasters cropped to a
common extent, so their axes are the *same* integers and grid coincidence persists even
off-grid. It still lands at ~1e-3, and `test_on_grid_sampling_costs_six_orders_of_magnitude`
pins that gap so the cost of D10 stays a measured number rather than a warning.

### D11 in practice: rank 3 descends on a different objective

Rank 3 is the first place the port and the original do not converge to the same answer, and
it is the original that disagrees with itself. Upstream builds the regularisation energy over
two of the three spatial axes, then smooths that same energy's gradient over all three:

| | axes | line |
| --- | --- | --- |
| the regularisation **energy** `ER` | `dim=(1,2)` | `STalign.py:1504` |
| the smoothing applied to **its gradient** | `dim=(1,2,3)` | `STalign.py:1527` |

So the quantity it reports is not the quantity it descends on — autograd carries the
two-axis energy into the loss while three-axis smoothing shapes the step, and the mismatch
lands in the search direction. squidpy uses every spatial axis in both places.

On the reference velocity field upstream's regularisation energy reads **~5.4×** the
three-axis value. `test_slice_regularizer_axes_diverge_from_upstream` pins both sides:
squidpy matches the three-axis regulariser to `< 1e-12`, *and* is asserted to stay more than
10 % away from upstream's two-axis reading — so if the port ever quietly adopts upstream's
convention, the test fails and says this row has to be rewritten.

What it costs end to end, on the two Allen-CCF notebooks (relative L2, `shim2-2229` run):

| | `A` | `v` |
| --- | --- | --- |
| `merfish-allen3Datlas-alignment` | 0.0214 | 1.90 |
| `starmap-allen3Datlas-alignment` | 0.0109 | 0.879 |

The affine stays close, the velocity field does not — which is what descending on a
different objective looks like when the initialisation is shared. At rank 2 the two readings
coincide, so this has no 2D counterpart and every rank-2 row above still holds to machine
precision.

### R8. The image path's defaults are not upstream's, on purpose — documented

Not a divergence in the algorithm: a divergence in the *defaults*, and a trap for anything
that compares the two implementations.

| | upstream `LDDMM` / `_SOLVER_DEFAULTS` | `_IMAGE_DEFAULTS` |
| --- | --- | --- |
| `a` (regulariser kernel width) | 500.0 | **20.0** |
| `niter` | 5000 | **200** |
| `diffeo_start` | 0 | **100** |
| `epV` | 2e3 | **1.0** |

squidpy's reasons are good ones, and both are improvements. A kernel width of 500 is in the
velocity grid's physical units, so it is reasonable for cells measured in microns and exceeds
the whole picture for an image measured in pixels. And `diffeo_start` halfway through the run
lets the affine settle before the velocity field switches on, instead of letting it absorb
what is really a global translation — which is what upstream's `diffeo_start=0` does, and is
worse-conditioned for it. R5 already records the other half of that policy: upstream's
`niter=5000` ran blind.

The trap is that `align_stalign_image` resolves these, **fourteen of the seventeen notebooks
pass none of the four**, and the notebooks are not the image modality anyway — they rasterise
cells in microns, and squidpy agrees, because `_OBS_DEFAULTS` leaves all four at upstream's
values. A replay that lets the image entry point fill them runs the port on a different fit
from upstream and reports the gap as a port defect. So `notebook_suite` puts every solver
keyword on the call explicitly, from `_SOLVER_DEFAULTS`;
`test_replay_fills_omitted_keywords_from_upstream_not_the_image_path` pins it, and
`.claude/smoke_stalign_suite.sbatch` catches it in one iteration rather than after a
forty-five-minute sweep.

### R7. `affine_from_points` silently changes the landmark estimator — open

Following on from D7: `estimate_transform("affine")` is not a drop-in for upstream's
normal-equations fit. It minimises **algebraic** error on Hartley-normalised coordinates;
upstream minimises **geometric** residual. Every landmark-initialised alignment therefore
starts from a measurably different affine (`L` by 6.4e-4, `T` by 7.2e-3), which for a
5000-iteration descent is a different starting point, not a rounding detail.

skimage's choice is the safer one, so this is not a request to revert — it is a request to
*say so* in the docstring. If exact least-squares is wanted, `np.linalg.lstsq` on the
padded design matrix gives upstream's answer without upstream's conditioning problem.

### R5. `niter=5000` ran blind — fixed

`lddmm` now returns the per-iteration `energies` trace and the `n_iter` actually run, so
a converged run is distinguishable from a diverged one without running it again.
(Upstream accumulates the same values in a local `Esave` and throws them away.)

It also takes an optional `tol` / `patience` early stop: it halts once the objective's
relative improvement over the last `patience` iterations falls below `tol`. Off by
default, so the shipped behaviour is unchanged.

**The window must clear iteration 50.** The mixture-weight E step switches on at
`MIXTURE_E_STEP_START` (`STalign.py:1233`), which changes what the objective *is* — and
its value jumps **upward** there, on the reference fixture from `3.259` to `5.489`. Any
rule comparing across that boundary reads the jump as "no longer improving" and quits
immediately: with a naive guard every tolerance from `1e-3` to `1e-5` stopped at
iteration 76 regardless. The comparison window now has to sit entirely after the jump.
`test_early_stopping_never_fires_before_the_weights_switch_on` pins this.

### R6. The Python loop forgave most of JAX's advantage — fixed

The descent was a Python `for` loop around a jitted `value_and_grad`, paying dispatch
per iteration and blocking XLA from fusing across steps. It is now a single
`lax.while_loop` inside one `jax.jit`, with `lax.cond` for the every-5th-iteration
weight update and `jnp.where` for the `diffeo_start` gates.

Measured on the reference fixture, warm:

| | ms/iteration | `niter=5000` |
| --- | --- | --- |
| Python loop (was) | 2.20 | ~11.0 s |
| `lax.while_loop`, not jitted | 0.91 | ~4.6 s |
| **`lax.while_loop` inside `jit`** | **0.46** | **~2.4 s** |

The middle row is worth keeping in mind: `lax.while_loop` outside a `jit` re-traces its
body on *every call*, and tracing `value_and_grad` through the interpolation and FFTs
costs roughly a second — about as much as a thousand iterations of running it. The loop
has to be inside the `jit` to get the win.

Reference parity is unchanged at 1 / 5 / 50 / 500 iterations, which is what makes this
rewrite safe to make at all.
