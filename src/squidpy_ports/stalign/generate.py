"""Emit the reference bundle consumed by squidpy's STalign tests.

Run with::

    hatch run generate:run --out ../squidpy/tests/_data/stalign_reference

Everything here either calls a public upstream function or observes the unmodified
``LDDMM`` loop (see :mod:`.upstream`). The one exception is noted inline: ``LL`` / ``K``
/ ``DV`` are four statements inside ``LDDMM`` with no function boundary, so they are
quoted verbatim with line references.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import fixtures as F
from . import upstream

__all__ = ["main"]

#: Iteration counts for the trajectory snapshots. 50 crosses the `it >= 50` gate at
#: STalign.py:1233 where the mixture-weight E-step switches on; 1 and 5 sit below it.
TRAJECTORY_ITERS = (1, 5, 50)
#: "Enough iterations to actually converge", per scverse/squidpy#1243.
CONVERGED_ITERS = 500


def _pin_determinism() -> None:
    """Fix reduction order so the emitted numbers are reproducible."""
    import matplotlib
    import torch

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(var) != "1":
            raise RuntimeError(
                f"{var} must be 1 before torch is imported, otherwise threaded BLAS "
                f"reduction order varies run to run. Use `hatch run generate:run`."
            )
    torch.set_num_threads(1)
    matplotlib.use("Agg")
    warnings.simplefilter("ignore")


def _provenance(**extra: Any) -> np.ndarray:
    """A JSON blob stamped into every ``.npz`` so the fixtures stay falsifiable."""
    import torch

    root = Path(__file__).resolve().parents[3]
    try:
        ports_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:  # pragma: no cover - not a checkout
        ports_commit = "unknown"

    payload = {
        "upstream_sha": upstream.UPSTREAM_SHA,
        "upstream_url": upstream.UPSTREAM_URL,
        "ports_commit": ports_commit,
        "fixtures_checksum": F.checksum(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # Deliberately no argv: it holds the --out path, which would make the committed
        # fixtures differ depending on where they happened to be generated.
        "generated_utc": datetime.now(UTC).isoformat(),
        **extra,
    }
    return np.array(json.dumps(payload, indent=2, sort_keys=True))


def _regularizer(xv, *, a: float, p: float):
    """``LL``, ``K`` and ``DV`` exactly as ``LDDMM`` computes them.

    Verbatim from STalign.py:1076-1090. These four statements live inside the loop
    body with no function boundary, so there is nothing public to call; re-expressing
    them is the single documented exception to "only observe upstream".
    """
    import torch

    XV = torch.stack(torch.meshgrid(*xv, indexing="ij"), -1)
    dv = torch.as_tensor([x[1] - x[0] for x in xv], dtype=XV.dtype)
    fv = [torch.arange(n, dtype=XV.dtype) / n / d for n, d in zip(XV.shape, dv, strict=False)]
    FV = torch.stack(torch.meshgrid(*fv, indexing="ij"), -1)
    LL = (1.0 + 2.0 * a**2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1)) ** (p * 2.0)
    K = 1.0 / LL
    DV = torch.prod(dv)
    return LL.numpy(), K.numpy(), float(DV)


def _rasters(st, clouds: F.Clouds):
    """Upstream rasterisations, returned as ``(source, target)`` = ``(query, ref)``.

    LDDMM's ``I``/``xI`` is the *moving* image and ``J``/``xJ`` the fixed one, matching
    ``pointsI``/``pointsJ``. The query is what moves, so the query raster is the source --
    feeding the ref raster as ``I`` while passing query landmarks as ``pointsI`` sets the
    image term and the landmark term against each other and the fit goes nowhere.
    """
    xq, yq, image_query = st.rasterize(clouds.query[:, 0], clouds.query[:, 1], draw=0, **F.RASTER_PARAMS)
    xr, yr, image_ref = st.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], draw=0, **F.RASTER_PARAMS)
    return (xq, yq, image_query), (xr, yr, image_ref)


def _upstream_velocity_grid(st, *, rasters) -> list[np.ndarray]:
    """The grid ``LDDMM`` builds for itself, read off a one-iteration probe run.

    Taken from the return value rather than recomputed, so the `torch.arange` at
    STalign.py:1069 stays the only definition of it.
    """
    (xs, ys, source), (xt, yt, target) = rasters
    probe = st.LDDMM(
        [ys, xs], source, [yt, xt], target,
        L=np.eye(2), T=np.zeros(2), niter=1, **F.LDDMM_PARAMS,
    )  # fmt: skip
    return [x.numpy() for x in probe["xv"]]


def _smooth_velocity(xv, nt: int) -> np.ndarray:
    """A plausible non-zero velocity field on ``xv``.

    Double-cumsum of a normal draw, i.e. low frequency. A raw normal draw is nothing like
    a real velocity field and would exercise only the interpolator's noise response.

    This matters for more than the interpolation tests: at ``v = 0`` the regularisation
    term and its gradient are both identically zero, so an objective evaluated only at
    the origin cannot see that term at all. Every energy and gradient fixture is therefore
    emitted twice -- once at ``v = 0`` (LDDMM's true initial state) and once here.
    """
    rng = np.random.default_rng(F.SEED + 2)
    field = rng.normal(scale=12.0, size=(nt, xv[0].size, xv[1].size, 2))
    return np.cumsum(np.cumsum(field, axis=1), axis=2) / (xv[0].size * xv[1].size)


def _write_primitives(st, clouds: F.Clouds, out: Path) -> None:
    import torch

    # Source is the query (it moves); target is the ref. See `_rasters`.
    (xs, ys, source_image), (xt, yt, target_image) = _rasters(st, clouds)

    # --- interp, both padding modes -------------------------------------------------
    # A 20x25 patch of off-grid sample points well inside the image, plus a second set
    # deliberately outside it so the zeros-vs-border difference is measurable.
    rng = np.random.default_rng(F.SEED + 1)
    rows = np.linspace(ys[2], ys[-3], 20)
    cols = np.linspace(xs[2], xs[-3], 25)
    grid = np.stack(np.meshgrid(rows, cols, indexing="ij"))
    grid = grid + rng.uniform(-0.37, 0.37, size=grid.shape) * (ys[1] - ys[0])

    span_r, span_c = ys[-1] - ys[0], xs[-1] - xs[0]
    outside = np.stack(
        np.meshgrid(
            np.linspace(ys[0] - span_r, ys[-1] + span_r, 20),
            np.linspace(xs[0] - span_c, xs[-1] + span_c, 25),
            indexing="ij",
        )
    )

    interp_border = st.interp([ys, xs], source_image, torch.as_tensor(grid), padding_mode="border").numpy()
    interp_zeros_outside = st.interp([ys, xs], source_image, torch.as_tensor(outside)).numpy()
    interp_border_outside = st.interp([ys, xs], source_image, torch.as_tensor(outside), padding_mode="border").numpy()

    # --- to_A -----------------------------------------------------------------------
    lin = np.array([[0.987, -0.153], [0.171, 1.014]])
    trans = np.array([12.5, -7.25])
    affine = st.to_A(torch.as_tensor(lin), torch.as_tensor(trans)).numpy()

    # --- velocity field and the point transforms -------------------------------------
    xv = _upstream_velocity_grid(st, rasters=((xs, ys, source_image), (xt, yt, target_image)))
    smooth = _smooth_velocity(xv, F.LDDMM_PARAMS["nt"])
    velocity = torch.as_tensor(smooth)

    # `build_transform(direction='b')` is the public form of exactly what squidpy's
    # `_transform_grid_backward` does: invert the affine, then integrate -v backwards in
    # time. (`v_to_phii` is not the analogue -- it takes v as (nt, 2, v0, v1) and skips
    # the affine.) Note the axis order differs: upstream returns (H, W, 2), squidpy
    # returns (2, H, W).
    grid_backward = st.build_transform(
        [torch.as_tensor(x) for x in xv],
        velocity,
        torch.as_tensor(affine),
        direction="b",
        XJ=[torch.as_tensor(yt), torch.as_tensor(xt)],
    ).numpy()

    pts = clouds.query_rc[:200]
    fwd = st.transform_points_source_to_target(
        [torch.as_tensor(x) for x in xv], velocity, torch.as_tensor(affine), pts
    ).numpy()
    bwd = st.transform_points_target_to_source(
        [torch.as_tensor(x) for x in xv], velocity, torch.as_tensor(affine), pts
    ).numpy()
    # Round trip, to show which side's time ordering actually inverts (divergence 6).
    roundtrip = st.transform_points_target_to_source(
        [torch.as_tensor(x) for x in xv], velocity, torch.as_tensor(affine), fwd
    ).numpy()

    # --- L_T_from_points, well and ill conditioned -----------------------------------
    lt_well = st.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)
    ill_src = np.stack([np.linspace(0.0, 500.0, 6), np.linspace(0.0, 500.0, 6) + 1e-4], axis=1)
    ill_dst = ill_src @ np.array([[1.02, 0.01], [-0.01, 0.99]]).T + np.array([3.0, -2.0])
    lt_ill = st.L_T_from_points(ill_src, ill_dst)

    ll, kernel, dv_prod = _regularizer([torch.as_tensor(x) for x in xv], a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])

    np.savez_compressed(
        out / "primitives.npz",
        __provenance__=_provenance(section="primitives"),
        ref=clouds.ref,
        query=clouds.query,
        landmarks_ref=clouds.landmarks_ref,
        landmarks_query=clouds.landmarks_query,
        # The source raster is the query's, the target raster is the ref's. Keys are
        # named after the cloud rather than the role so the two cannot be mixed up.
        raster_query_x=xs,
        raster_query_y=ys,
        raster_query=source_image,
        raster_ref_x=xt,
        raster_ref_y=yt,
        raster_ref=target_image,
        interp_coords=grid,
        interp_border=interp_border,
        interp_coords_outside=outside,
        interp_zeros_outside=interp_zeros_outside,
        interp_border_outside=interp_border_outside,
        to_A_linear=lin,
        to_A_translation=trans,
        to_A=affine,
        xv_upstream_0=xv[0],
        xv_upstream_1=xv[1],
        velocity=smooth,
        grid_backward=grid_backward,
        points=pts,
        points_forward=fwd,
        points_backward=bwd,
        points_roundtrip=roundtrip,
        lt_well_L=lt_well[0],
        lt_well_T=lt_well[1],
        ill_src=ill_src,
        ill_dst=ill_dst,
        lt_ill_L=lt_ill[0],
        lt_ill_T=lt_ill[1],
        regularizer_LL=ll,
        regularizer_K=kernel,
        regularizer_DV=np.array(dv_prod),
    )


def _lddmm_kwargs(clouds: F.Clouds, rasters, *, with_points: bool, nt: int | None = None):
    (xs, ys, source), (xt, yt, target) = rasters
    params = dict(F.LDDMM_PARAMS)
    if nt is not None:
        params["nt"] = nt
    kwargs: dict[str, Any] = {
        "xI": [ys, xs],
        "I": source,
        "xJ": [yt, xt],
        "J": target,
        **params,
    }
    if with_points:
        kwargs["pointsI"] = clouds.landmarks_query_rc
        kwargs["pointsJ"] = clouds.landmarks_ref_rc
    return kwargs


def _write_energy(st, clouds: F.Clouds, out: Path) -> None:
    rasters = _rasters(st, clouds)
    lin, trans = st.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)

    xv = _upstream_velocity_grid(st, rasters=rasters)
    payload: dict[str, Any] = {
        "__provenance__": _provenance(section="energy"),
        "L": lin,
        "T": trans,
        # Upstream's own velocity grid. `_lddmm_loss` takes `xv` explicitly, so the
        # squidpy side can evaluate the objective in exactly this configuration.
        "xv_0": xv[0],
        "xv_1": xv[1],
    }
    for nt in (1, 3):
        for with_points in (False, True):
            kwargs = _lddmm_kwargs(clouds, rasters, with_points=with_points, nt=nt)
            key = f"E_nt{nt}_{'points' if with_points else 'nopoints'}"

            # v = 0, LDDMM's true starting state.
            _, captured = upstream.lddmm_with_grads(st, niter=1, L=lin, T=trans, **kwargs)
            payload[key] = np.array(captured["E"][0])

            # v != 0, so the regularisation term is actually exercised. Without this the
            # whole ER term is invisible: at v = 0 it contributes exactly nothing, and a
            # sign error in it would compare equal.
            warm = _smooth_velocity(xv, nt)
            payload[f"{key}_warmv"] = np.array(
                upstream.lddmm_with_grads(st, niter=1, L=lin, T=trans, xv=list(xv), v=warm, **kwargs)[1]["E"][0]
            )
    payload["warm_velocity_nt1"] = _smooth_velocity(xv, 1)
    payload["warm_velocity_nt3"] = _smooth_velocity(xv, 3)
    np.savez_compressed(out / "energy.npz", **payload)


def _write_gradients(st, clouds: F.Clouds, out: Path) -> None:
    rasters = _rasters(st, clouds)
    lin, trans = st.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)
    kwargs = _lddmm_kwargs(clouds, rasters, with_points=True)

    one, captured = upstream.lddmm_with_grads(st, niter=1, L=lin, T=trans, **kwargs)
    grad_l_hook = captured["L"][0].numpy()
    grad_t_hook = captured["T"][0].numpy()

    # The K-smoothed velocity gradient, recovered from the step itself. v starts at
    # exactly zero (STalign.py:1071) and diffeo_start=0, so v_after = -epV * g.
    grad_v = -one["v"].numpy() / F.LDDMM_PARAMS["epV"]

    # Second, independent extraction of dE/dL and dE/dT. Divergence 4 means
    # LDDMM(niter=1)['A'] is the *initial* affine and carries no gradient information,
    # so the update has to be read out of a two-iteration run. The /10 is the
    # (it >= diffeo_start) * 9 scaling at STalign.py:1205-1206, active from iteration 0
    # at the default diffeo_start=0.
    two = st.LDDMM(niter=2, L=lin, T=trans, **kwargs)
    affine_after = two["A"].numpy()
    scale = 1.0 + 9.0 * (0 >= F.LDDMM_PARAMS["diffeo_start"])
    grad_l_delta = (lin - affine_after[:2, :2]) / (F.LDDMM_PARAMS["epL"] / scale)
    grad_t_delta = (trans - affine_after[:2, -1]) / (F.LDDMM_PARAMS["epT"] / scale)

    for name, hook, delta, tol in (
        ("dE/dL", grad_l_hook, grad_l_delta, 1e-6),
        ("dE/dT", grad_t_hook, grad_t_delta, 1e-6),
    ):
        rel = np.linalg.norm(hook - delta) / max(np.linalg.norm(hook), 1e-300)
        if rel > tol:
            raise RuntimeError(
                f"the two independent extractions of {name} disagree by {rel:.3e}; "
                f"one of the spies no longer matches the vendored loop"
            )

    # The same three gradients at a non-zero velocity. At v = 0 the regularisation term
    # and its gradient both vanish identically, so gradients taken only at the origin
    # cannot detect any error in that term.
    xv = [x.numpy() for x in one["xv"]]
    warm = _smooth_velocity(xv, F.LDDMM_PARAMS["nt"])
    warm_run, warm_captured = upstream.lddmm_with_grads(st, niter=1, L=lin, T=trans, xv=list(xv), v=warm, **kwargs)
    grad_v_warm = (warm - warm_run["v"].numpy()) / F.LDDMM_PARAMS["epV"]

    np.savez_compressed(
        out / "gradients.npz",
        __provenance__=_provenance(section="gradients"),
        L=lin,
        T=trans,
        # Upstream's own grid; `_lddmm_loss` accepts `xv`, so squidpy differentiates the
        # objective in exactly this configuration.
        xv_0=xv[0],
        xv_1=xv[1],
        grad_L=grad_l_hook,
        grad_T=grad_t_hook,
        grad_L_from_delta=grad_l_delta,
        grad_T_from_delta=grad_t_delta,
        grad_v_smoothed=grad_v,
        warm_velocity=warm,
        grad_L_warmv=warm_captured["L"][0].numpy(),
        grad_T_warmv=warm_captured["T"][0].numpy(),
        grad_v_smoothed_warmv=grad_v_warm,
    )


def _write_trajectory(st, clouds: F.Clouds, out: Path) -> None:
    rasters = _rasters(st, clouds)
    lin, trans = st.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)

    # Upstream's own grid, which squidpy's `_build_velocity_grid` now reproduces exactly.
    # Passed back in explicitly (STalign.py:1060-1064 accepts `xv`/`v`) so the trajectory
    # comparison stays valid even if either side's grid construction drifts again.
    xv = _upstream_velocity_grid(st, rasters=rasters)
    v0 = np.zeros((F.LDDMM_PARAMS["nt"], xv[0].size, xv[1].size, 2))
    kwargs = _lddmm_kwargs(clouds, rasters, with_points=True)

    for niter in (*TRAJECTORY_ITERS, CONVERGED_ITERS):
        run = st.LDDMM(niter=niter, L=lin, T=trans, xv=list(xv), v=v0, **kwargs)
        # Divergence 4: upstream builds A at the top of each iteration and returns it,
        # so LDDMM(n)['A'] reflects n-1 updates. squidpy(n).A is the post-update affine,
        # which is what LDDMM(n+1)['A'] holds.
        nxt = st.LDDMM(niter=niter + 1, L=lin, T=trans, xv=list(xv), v=v0, **kwargs)

        payload: dict[str, Any] = {
            "__provenance__": _provenance(section=f"trajectory_n{niter}", niter=niter),
            "xv_0": xv[0],
            "xv_1": xv[1],
            "L": lin,
            "T": trans,
            "A_stale": run["A"].numpy(),
            "A": nxt["A"].numpy(),
            "v": run["v"].numpy(),
            "WM": run["WM"].numpy(),
            "WA": run["WA"].numpy(),
            "WB": run["WB"].numpy(),
        }
        if niter == CONVERGED_ITERS:
            aligned = st.transform_points_source_to_target(run["xv"], run["v"], nxt["A"], clouds.query_rc).numpy()
            moved_landmarks = st.transform_points_source_to_target(
                run["xv"], run["v"], nxt["A"], clouds.landmarks_query_rc
            ).numpy()
            tre_mean, tre_std = st.calculate_tre(moved_landmarks, clouds.landmarks_ref_rc)
            _, captured = upstream.lddmm_with_grads(st, niter=niter, L=lin, T=trans, xv=list(xv), v=v0, **kwargs)
            payload |= {
                "aligned_points_rc": aligned,
                "aligned_landmarks_rc": moved_landmarks,
                "tre_mean": np.array(float(tre_mean)),
                "tre_std": np.array(float(tre_std)),
                "E_first": np.array(captured["E"][0]),
                "E_last": np.array(captured["E"][-1]),
            }
            name = f"converged_n{niter}.npz"
        else:
            name = f"trajectory_n{niter}.npz"
        np.savez_compressed(out / name, **payload)


#: Solver settings for the image path. `fit_stalign_image` works in pixel units, so the
#: kernel width and velocity step are far smaller than the micron-scale point-cloud
#: defaults; these are the values the squidpy side passes too.
IMAGE_PARAMS = {"a": 8.0, "p": 2.0, "expand": 2.0, "nt": 2, "diffeo_start": 4, "epV": 1.0}
IMAGE_ITERS = 12

#: Initial affine for the image comparisons. Deliberately *not* identity, and not a round
#: angle or an integer shift. Centred pixel axes are integers, so an identity start makes
#: every interpolation sample land exactly on a grid line -- the degenerate case where
#: upstream and squidpy can floor() to different neighbours (ledger row D10) and the
#: comparison measures that rather than the port. The point-cloud fixture avoids this the
#: same way, via `fixtures.THETA` / `fixtures.SHIFT`.
IMAGE_THETA = 0.0371449
IMAGE_SHIFT = (0.41372, -0.28913)


def _image_start() -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(IMAGE_THETA), np.sin(IMAGE_THETA)
    return np.array([[c, -s], [s, c]]), np.array(IMAGE_SHIFT)


def _centred_axes(shape: tuple[int, int]) -> list[np.ndarray]:
    """The axes `fit_stalign_image` builds for an image of this shape.

    Reproduced verbatim from `_stalign.py` rather than imported: this package must not
    depend on squidpy, or the reference would be defined in terms of the thing it checks.
    Pixel units, centred so the affine starts near identity.
    """
    return [(np.arange(n, dtype=float) - (n - 1) / 2.0) for n in shape]


#: Step of the target axes in the mixed-unit fixture, against the source's step of 1.
#: 30 is not arbitrary: `xenium-heimage-alignment` pairs an H&E at 1 unit per pixel with a
#: Xenium density rasterised at `dx=30`, and that pairing is what this fixture is for.
MIXED_TARGET_STEP = 30.0


def _write_image_mixed_units(st, clouds: F.Clouds, out: Path) -> None:
    """Upstream run on a source and target whose axes are in *different* units.

    Every other image fixture places both sides on centred pixel axes, so each carries one
    unit and any confusion between them is invisible. Real pairs do not: an H&E in pixels
    against a density rasterised in microns is the ordinary case, and it is the case
    `xenium-heimage-alignment` is. Nothing measured the velocity grid there until
    `stalign_align_image` began reading each element's real placement, at which point the
    grid moved by two orders of magnitude -- see ledger row D12.

    `a` is deliberately *not* rescaled. Upstream builds the velocity grid from the source's
    axes, which stay in pixels, so the notebook's own shape is a kernel width tuned for the
    source's units against a target measured in something else -- `a=500` against a 2050px
    H&E, paired with a density in microns. Scaling `a` by the target step instead collapses
    the source's span to a single velocity sample and upstream raises.
    """
    (_, _, query), (_, _, ref) = _rasters(st, clouds)
    x_source = _centred_axes(query.shape[1:])
    # The target lives in a coarser unit, spanning a correspondingly larger physical domain.
    x_target = [axis * MIXED_TARGET_STEP for axis in _centred_axes(ref.shape[1:])]

    lin, trans = _image_start()
    params = dict(IMAGE_PARAMS)
    kwargs = dict(xI=x_source, I=query, xJ=x_target, J=ref, L=lin, T=trans, **params)
    run = st.LDDMM(niter=IMAGE_ITERS, **kwargs)
    nxt = st.LDDMM(niter=IMAGE_ITERS + 1, **kwargs)

    np.savez_compressed(
        out / "image_mixed_units.npz",
        __provenance__=_provenance(section="image_mixed_units", niter=IMAGE_ITERS),
        start_L=lin,
        start_T=trans,
        source_axis_0=x_source[0],
        source_axis_1=x_source[1],
        target_axis_0=x_target[0],
        target_axis_1=x_target[1],
        query=query,
        ref=ref,
        a=np.asarray(params["a"]),
        A=nxt["A"].numpy(),
        v=run["v"].numpy(),
        WM=run["WM"].numpy(),
        xv_0=run["xv"][0].numpy(),
        xv_1=run["xv"][1].numpy(),
    )


def _write_image_trajectory_matched(st, clouds: F.Clouds, out: Path) -> None:
    """The same comparison with both images cropped to a common extent.

    The control for `image_trajectory`. There the two rasters have different shapes, and
    because `fit_stalign_image` centres each on its own centre they end up spanning
    different domains -- so ~11% of the target grid samples the source through padding,
    where upstream's `grid_sample` and squidpy's `map_coordinates` agree on values but
    not on derivatives. Cropping to a shared extent removes the padding entirely, so if
    this run agrees and the other does not, padding is the cause rather than a suspicion.
    """
    (xq, yq, query), (xr, yr, ref) = _rasters(st, clouds)
    rows = min(query.shape[1], ref.shape[1])
    cols = min(query.shape[2], ref.shape[2])
    query, ref = query[:, :rows, :cols], ref[:, :rows, :cols]
    x_source = _centred_axes((rows, cols))

    lin, trans = _image_start()
    kwargs = dict(xI=x_source, I=query, xJ=list(x_source), J=ref, L=lin, T=trans, **IMAGE_PARAMS)
    run = st.LDDMM(niter=IMAGE_ITERS, **kwargs)
    nxt = st.LDDMM(niter=IMAGE_ITERS + 1, **kwargs)
    _, captured = upstream.lddmm_with_grads(st, niter=IMAGE_ITERS, **kwargs)

    np.savez_compressed(
        out / "image_trajectory_matched.npz",
        __provenance__=_provenance(section="image_trajectory_matched", niter=IMAGE_ITERS),
        start_L=lin,
        start_T=trans,
        axis_0=x_source[0],
        axis_1=x_source[1],
        query=query,
        ref=ref,
        A=nxt["A"].numpy(),
        v=run["v"].numpy(),
        energies=np.asarray(captured["E"], dtype=float),
    )


def _write_image_trajectory(st, clouds: F.Clouds, out: Path) -> None:
    """Upstream run on the image path's own axes.

    `fit_stalign_image` is a public entry point with a different coordinate convention
    from the point-cloud one -- centred pixels rather than physical microns -- so the
    solver agreeing on rasters says nothing about it until the axes are checked too.
    The rasters from the point-cloud fixture are reused as the image pair, so nothing in
    `fixtures.py` changes and its checksum stays stable.
    """
    (xq, yq, query), (xr, yr, ref) = _rasters(st, clouds)
    x_source = _centred_axes(query.shape[1:])
    x_target = _centred_axes(ref.shape[1:])

    kwargs = dict(
        xI=x_source,
        I=query,
        xJ=x_target,
        J=ref,
        L=_image_start()[0],
        T=_image_start()[1],
        **IMAGE_PARAMS,
    )
    run = st.LDDMM(niter=IMAGE_ITERS, **kwargs)
    nxt = st.LDDMM(niter=IMAGE_ITERS + 1, **kwargs)
    # Per-iteration objective, so a disagreement can be bisected to the step it starts at
    # rather than only observed at the end.
    _, captured = upstream.lddmm_with_grads(st, niter=IMAGE_ITERS, **kwargs)

    # What `align(by="images", out="images/...")` materialises. Upstream's own composition
    # of build_transform + interp, rather than reassembling it here.
    warped = st.transform_image_source_to_target(run["xv"], run["v"], nxt["A"], x_source, query, x_target)

    np.savez_compressed(
        out / "image_trajectory.npz",
        __provenance__=_provenance(section="image_trajectory", niter=IMAGE_ITERS),
        start_L=_image_start()[0],
        start_T=_image_start()[1],
        source_axis_0=x_source[0],
        source_axis_1=x_source[1],
        target_axis_0=x_target[0],
        target_axis_1=x_target[1],
        query=query,
        ref=ref,
        A=nxt["A"].numpy(),
        A_stale=run["A"].numpy(),
        v=run["v"].numpy(),
        WM=run["WM"].numpy(),
        WA=run["WA"].numpy(),
        WB=run["WB"].numpy(),
        warped=warped.numpy(),
        energies=np.asarray(captured["E"], dtype=float),
        xv_0=run["xv"][0].numpy(),
        xv_1=run["xv"][1].numpy(),
    )


#: Solver settings for the volume-to-section path. Upstream's `LDDMM_3D_to_slice`
#: defaults (`a=500`, `expand=1.25`) assume a real 50um atlas; on a fixture-sized volume
#: `a=500` collapses the velocity grid to a single sample, so the kernel width is scaled
#: down the same way `IMAGE_PARAMS` does. `sigmaR` is upstream's 3D default.
#:
#: `epL`/`epT` are three orders below upstream's 3e-D defaults for a measured reason: at
#: `epL=1e-6, epT=1e1` the very first step throws the section clean off the volume, where
#: border padding makes the objective constant and its gradient exactly zero -- the run
#: then sits at E=481.85 forever. A comparison in that regime says nothing about either
#: implementation, so the steps are the largest of the probed pairs that actually descend
#: (E 48.126 -> 47.957 over 8 iterations).
SLICE_PARAMS = {
    "a": 8.0,
    "p": 2.0,
    "expand": 1.25,
    "nt": 2,
    "epL": 1e-9,
    "epT": 1e-2,
    "epV": 1.0,
    "sigmaR": 1e8,
}
SLICE_ITERS = 12

#: Off-grid start for the 3D comparison, for the reason spelled out at `IMAGE_THETA`:
#: centred voxel axes are integers, and an identity start would put every interpolation
#: sample exactly on a grid line, where the comparison measures ledger row D10 rather
#: than the port. The z shift is deliberately not a whole number of slices.
SLICE_THETA = 0.0271833
SLICE_SCALE = 0.93
SLICE_SHIFT = (0.6137, 0.41372, -0.28913)


def _slice_inputs(st, clouds: F.Clouds):
    """A reference volume and a single section cut from it, plus their axes.

    Built here rather than in ``fixtures.py`` so that file's checksum -- which pins the
    whole 2D bundle -- stays untouched. The volume is the existing reference raster swept
    along a third axis with a per-slice roll, so its structure genuinely moves with ``z``:
    a separable product would leave the out-of-plane gradient near zero and the fit would
    have nothing to find.

    The reference carries **two** channels and the section **one**, which is upstream's
    own recipe (``merfish-allen3Datlas-alignment`` cell [24] concatenates the normalised
    Nissl volume with its centred square) and the case a same-channel-count check would
    reject.
    """
    (_, _, image_query), (_, _, image_ref) = _rasters(st, clouds)
    base = np.asarray(image_ref[1], dtype=float)
    n_z = 9
    volume = np.stack([np.roll(base, shift=index - n_z // 2, axis=1) for index in range(n_z)])
    volume = volume * np.exp(-(((np.arange(n_z) - n_z / 2.0) / 3.0) ** 2))[:, None, None]

    normalised = volume[None] / np.mean(np.abs(volume))
    reference = np.concatenate((normalised, (normalised - np.mean(normalised)) ** 2))
    section = np.asarray(image_query[1:2], dtype=float)
    section = section / np.mean(np.abs(section))

    x_reference = _centred_axes(reference.shape[1:])
    x_section = _centred_axes(section.shape[1:])
    return reference, x_reference, section, x_section


def _slice_start() -> tuple[np.ndarray, np.ndarray]:
    """The initial affine, in the ``(z, y, x)`` array order both sides work in.

    An in-plane rotation about the out-of-plane axis, then a uniform scale -- the same
    construction ``fit_stalign_volume``'s ``initial_rotation`` / ``initial_scale`` build,
    and the notebook's cell [30] with ``scale_x == scale_y == scale_z``.
    """
    cos, sin = np.cos(SLICE_THETA), np.sin(SLICE_THETA)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    return SLICE_SCALE * rotation, np.array(SLICE_SHIFT)


def _write_slice(st, clouds: F.Clouds, out: Path) -> None:
    """Upstream ``LDDMM_3D_to_slice``, in the two regimes that can be compared.

    The velocity is frozen for the trajectory comparison (``diffeo_start`` past
    ``niter``, so the ``it >= diffeo_start`` gate at STalign.py:1519 never opens and ``v``
    stays at zero). That is not a way of avoiding a hard case -- it is the only regime in
    which upstream's 3D objective and a correct one *agree*, so it is the only one where
    the trajectory is a statement about the port rather than about the discrepancy below.

    ``ER`` is emitted twice at a non-zero velocity to measure that discrepancy. Upstream
    transforms two of the three spatial axes in the energy (``dim=(1,2)`` at
    STalign.py:1504) while smoothing its gradient over all three (``dim=(1,2,3)`` at
    :1527), so the regulariser it descends on is not the one it evaluates. ``ER_2axis``
    quotes upstream's expression verbatim; ``ER_3axis`` is the same expression with the
    energy's axes corrected to match the smoothing. At rank 2 the two coincide, which is
    why this only appears here. Both are computed from upstream's own ``v`` and ``LL``.
    """
    import torch

    reference, x_reference, section, x_section = _slice_inputs(st, clouds)
    linear, translation = _slice_start()

    kwargs = dict(
        xI=x_reference,
        I=reference,
        xJ=x_section,
        J=section,
        L=linear,
        T=translation,
        **SLICE_PARAMS,
    )
    frozen = dict(kwargs, diffeo_start=SLICE_ITERS + 1)
    run = st.LDDMM_3D_to_slice(niter=SLICE_ITERS, **frozen)
    # Divergence D4 again: upstream builds A at the top of each iteration and returns it.
    nxt = st.LDDMM_3D_to_slice(niter=SLICE_ITERS + 1, **frozen)
    _, captured = upstream.lddmm_with_grads(st, niter=SLICE_ITERS, entry="LDDMM_3D_to_slice", to_A="to_A_3D", **frozen)
    if not np.allclose(run["v"].numpy(), 0.0):
        raise RuntimeError("the velocity moved despite diffeo_start > niter; STalign.py:1519 has changed")

    xv = [axis.numpy() for axis in run["xv"]]
    velocity = _smooth_velocity_3d(xv, SLICE_PARAMS["nt"])

    # LL/K/DV: four statements inside the loop with no function boundary, so quoted
    # verbatim from STalign.py:1384-1397 -- the one documented exception in this module.
    dv = torch.as_tensor([x[1] - x[0] for x in xv], dtype=torch.float64)
    shape = tuple(len(x) for x in xv)
    fv = [torch.arange(n, dtype=torch.float64) / n / d for n, d in zip(shape, dv, strict=True)]
    FV = torch.stack(torch.meshgrid(*fv, indexing="ij"), -1)
    LL = (1.0 + 2.0 * SLICE_PARAMS["a"] ** 2 * torch.sum((1.0 - torch.cos(2.0 * np.pi * FV * dv)) / dv**2, -1)) ** (
        SLICE_PARAMS["p"] * 2.0
    )
    K = 1.0 / LL
    DV = torch.prod(dv)

    v_t = torch.as_tensor(velocity, dtype=torch.float64)
    er_2axis = (
        torch.sum(torch.sum(torch.abs(torch.fft.fftn(v_t, dim=(1, 2))) ** 2, dim=(0, -1)) * LL)
        * DV / 2.0 / v_t.shape[1] / v_t.shape[2] / SLICE_PARAMS["sigmaR"] ** 2
    )  # fmt: skip
    er_3axis = (
        torch.sum(torch.sum(torch.abs(torch.fft.fftn(v_t, dim=(1, 2, 3))) ** 2, dim=(0, -1)) * LL)
        * DV / 2.0 / v_t.shape[1] / v_t.shape[2] / v_t.shape[3] / SLICE_PARAMS["sigmaR"] ** 2
    )  # fmt: skip

    # `Xs` is the map the objective samples the volume through, and the same quantity
    # `analyze3Dalign` turns into `coord0`/`coord1`/`coord2` (STalign.py:2001-2003).
    section_grid = np.stack(np.meshgrid(np.zeros(1), x_section[0], x_section[1], indexing="ij"), -1)
    grid_backward = st.build_transform3D(
        xv, velocity, nxt["A"].numpy(), direction="b", XJ=torch.as_tensor(section_grid)
    )
    # interp3D on the same points, which is what `sample_reference` has to reproduce.
    sampled = st.interp3D(
        [torch.as_tensor(x) for x in x_reference],
        torch.as_tensor(reference),
        grid_backward.permute(-1, 0, 1, 2),
        padding_mode="border",
    )
    sampled_nearest = st.interp3D(
        [torch.as_tensor(x) for x in x_reference],
        torch.as_tensor(reference),
        grid_backward.permute(-1, 0, 1, 2),
        mode="nearest",
        padding_mode="border",
    )

    np.savez_compressed(
        out / "slice_trajectory.npz",
        __provenance__=_provenance(section="slice_trajectory", niter=SLICE_ITERS),
        start_L=linear,
        start_T=translation,
        ref_axis_0=x_reference[0],
        ref_axis_1=x_reference[1],
        ref_axis_2=x_reference[2],
        query_axis_0=x_section[0],
        query_axis_1=x_section[1],
        ref=reference,
        query=section,
        A=nxt["A"].numpy(),
        A_stale=run["A"].numpy(),
        to_A=st.to_A_3D(torch.as_tensor(linear), torch.as_tensor(translation)).detach().numpy(),
        WM=run["WM"].numpy(),
        WA=run["WA"].numpy(),
        WB=run["WB"].numpy(),
        Xs=run["Xs"].numpy(),
        energies=np.asarray(captured["E"], dtype=float),
        grad_L=torch.stack(captured["L"]).numpy(),
        grad_T=torch.stack(captured["T"]).numpy(),
        xv_0=xv[0],
        xv_1=xv[1],
        xv_2=xv[2],
        velocity=velocity,
        regularizer_LL=LL.numpy(),
        regularizer_K=K.numpy(),
        regularizer_DV=DV.numpy(),
        ER_2axis=np.array(float(er_2axis)),
        ER_3axis=np.array(float(er_3axis)),
        grid_backward=grid_backward.numpy(),
        interp_border=sampled.numpy(),
        interp_nearest=sampled_nearest.numpy(),
    )


def _smooth_velocity_3d(xv, nt: int) -> np.ndarray:
    """:func:`_smooth_velocity` at rank 3 -- low frequency, one cumsum per spatial axis."""
    rng = np.random.default_rng(F.SEED + 3)
    field = rng.normal(scale=12.0, size=(nt, xv[0].size, xv[1].size, xv[2].size, 3))
    for axis in (1, 2, 3):
        field = np.cumsum(field, axis=axis)
    return field / (xv[0].size * xv[1].size * xv[2].size)


def main(argv: list[str] | None = None) -> int:
    """Generate the whole bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="directory to write the .npz bundle into")
    args = parser.parse_args(argv)

    _pin_determinism()
    st = upstream.load()
    clouds = F.make_clouds()

    args.out.mkdir(parents=True, exist_ok=True)
    for step, fn in (
        ("primitives", _write_primitives),
        ("energy", _write_energy),
        ("gradients", _write_gradients),
        ("trajectory", _write_trajectory),
        ("image_trajectory", _write_image_trajectory),
        ("image_trajectory_matched", _write_image_trajectory_matched),
        ("slice_trajectory", _write_slice),
    ):
        print(f"generating {step}...", flush=True)
        fn(st, clouds, args.out)

    import matplotlib.pyplot as plt

    plt.close("all")  # upstream leaks 4 figures per LDDMM call
    print(f"wrote {len(list(args.out.glob('*.npz')))} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
