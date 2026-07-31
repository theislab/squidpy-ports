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


def _squidpy_velocity_grid(x_source: tuple[np.ndarray, np.ndarray], *, a: float, expand: float):
    """Reproduce the grid squidpy's ``_build_velocity_grid`` builds.

    Upstream's ``LDDMM`` accepts ``xv=``/``v=`` (STalign.py:1060-1064), so forcing it
    onto *squidpy's* grid is how the trajectory comparison isolates the solver from the
    grid off-by-one (divergence 2) without needing any change to squidpy's source.

    Squidpy's grid is one point longer on each axis than upstream's, because it stops at
    ``hi + step`` where upstream stops at ``hi``. The squidpy side asserts this array
    equals what ``_build_velocity_grid`` actually returns, so the two cannot drift.
    """
    minimum = np.array([x_source[0][0], x_source[1][0]])
    maximum = np.array([x_source[0][-1], x_source[1][-1]])
    center = (minimum + maximum) / 2.0
    half_width = (maximum - minimum) * expand / 2.0
    step = a * 0.5
    return (
        np.arange(center[0] - half_width[0], center[0] + half_width[0] + step, step),
        np.arange(center[1] - half_width[1], center[1] + half_width[1] + step, step),
    )


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
    """Upstream rasterisations of both clouds, in the ``[Y, X]`` axis order LDDMM wants."""
    xi, yi, image_ref = st.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], draw=0, **F.RASTER_PARAMS)
    xj, yj, image_query = st.rasterize(clouds.query[:, 0], clouds.query[:, 1], draw=0, **F.RASTER_PARAMS)
    return (xi, yi, image_ref), (xj, yj, image_query)


def _upstream_velocity_grid(st, *, rasters) -> list[np.ndarray]:
    """The grid ``LDDMM`` builds for itself, read off a one-iteration probe run.

    Taken from the return value rather than recomputed, so the `torch.arange` at
    STalign.py:1069 stays the only definition of it.
    """
    (xi, yi, image_ref), (xj, yj, image_query) = rasters
    probe = st.LDDMM(
        [yi, xi], image_ref, [yj, xj], image_query,
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

    (xi, yi, image_ref), (xj, yj, image_query) = _rasters(st, clouds)

    # --- interp, both padding modes -------------------------------------------------
    # A 20x25 patch of off-grid sample points well inside the image, plus a second set
    # deliberately outside it so the zeros-vs-border difference is measurable.
    rng = np.random.default_rng(F.SEED + 1)
    rows = np.linspace(yi[2], yi[-3], 20)
    cols = np.linspace(xi[2], xi[-3], 25)
    grid = np.stack(np.meshgrid(rows, cols, indexing="ij"))
    grid = grid + rng.uniform(-0.37, 0.37, size=grid.shape) * (yi[1] - yi[0])

    span_r, span_c = yi[-1] - yi[0], xi[-1] - xi[0]
    outside = np.stack(
        np.meshgrid(
            np.linspace(yi[0] - span_r, yi[-1] + span_r, 20),
            np.linspace(xi[0] - span_c, xi[-1] + span_c, 25),
            indexing="ij",
        )
    )

    interp_border = st.interp([yi, xi], image_ref, torch.as_tensor(grid), padding_mode="border").numpy()
    interp_zeros_outside = st.interp([yi, xi], image_ref, torch.as_tensor(outside)).numpy()
    interp_border_outside = st.interp([yi, xi], image_ref, torch.as_tensor(outside), padding_mode="border").numpy()

    # --- to_A -----------------------------------------------------------------------
    lin = np.array([[0.987, -0.153], [0.171, 1.014]])
    trans = np.array([12.5, -7.25])
    affine = st.to_A(torch.as_tensor(lin), torch.as_tensor(trans)).numpy()

    # --- velocity field and the point transforms -------------------------------------
    xv = _upstream_velocity_grid(st, rasters=((xi, yi, image_ref), (xj, yj, image_query)))
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
        XJ=[torch.as_tensor(yj), torch.as_tensor(xj)],
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
        raster_ref_x=xi,
        raster_ref_y=yi,
        raster_ref=image_ref,
        raster_query_x=xj,
        raster_query_y=yj,
        raster_query=image_query,
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
    (xi, yi, image_ref), (xj, yj, image_query) = rasters
    params = dict(F.LDDMM_PARAMS)
    if nt is not None:
        params["nt"] = nt
    kwargs: dict[str, Any] = {
        "xI": [yi, xi],
        "I": image_ref,
        "xJ": [yj, xj],
        "J": image_query,
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
    (xi, yi, _), _ = rasters
    lin, trans = st.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)

    xv = _squidpy_velocity_grid((yi, xi), a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
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
            "xv_squidpy_0": xv[0],
            "xv_squidpy_1": xv[1],
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
    ):
        print(f"generating {step}...", flush=True)
        fn(st, clouds, args.out)

    import matplotlib.pyplot as plt

    plt.close("all")  # upstream leaks 4 figures per LDDMM call
    print(f"wrote {len(list(args.out.glob('*.npz')))} files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
