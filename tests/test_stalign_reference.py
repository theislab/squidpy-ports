"""Numerical comparison of squidpy's STalign port against the pinned original.

Implements scverse/squidpy#1243. Every upstream value here is computed **in this process**
from the vendored PyTorch STalign, by the same generator that writes the shareable bundle
(:mod:`squidpy_ports.stalign.generate`) -- so there is no committed binary to go stale, in
this repo or in squidpy, and the comparison cannot silently drift from the pin.

This is the layer that asserts the *port*, so it needs squidpy and JAX; both are optional
here and the module skips without them. Each xfail's own docstring says what
comparison found and why the known-divergent ones are pinned rather than fixed.

Run them with::

    JAX_ENABLE_X64=1 pytest tests/test_stalign_reference.py

Rank 2 is section-to-section (upstream ``LDDMM``); rank 3 is section-into-volume
(``LDDMM_3D_to_slice``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("squidpy")

import anndata as ad  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from squidpy.experimental.im import sample_volume  # noqa: E402
from squidpy.experimental.tl import (  # noqa: E402
    Stalign2DResult,
    Stalign3DResult,
    align_stalign_image,
    align_stalign_obs,
    align_stalign_volume,
)

# The estimators stay private on purpose: this module's job is to pin the *kernel* against
# upstream at the same axes, and the public `align_stalign_*` entry points take an element's
# placement rather than its axes -- squidpy rebuilds them from a `Scale`/`Translation`, which
# is not bit-exact for every grid (see `tests._replay.axis_placement`). A comparison that
# asserts agreement at ~1e-15 cannot afford the harness perturbing its own inputs.
#
# The obs case is the exception and goes through the public `align_stalign_obs`: it starts
# from points, which `obsm` stores verbatim, so there is no placement to reconstruct.
from squidpy.experimental.tl._align._stalign import (  # noqa: E402
    fit_stalign_image,
    fit_stalign_volume,
)

# The objective, its gradients and the grid primitives. No public route exists or should:
# they are not observable from outside a fit, which is what makes them the white-box set.
from squidpy.experimental.tl._align._stalign_impl import _core, _helpers  # noqa: E402

from squidpy_ports.stalign import fixtures as F  # noqa: E402
from squidpy_ports.stalign import generate as G  # noqa: E402
from squidpy_ports.stalign import upstream  # noqa: E402


@pytest.fixture(scope="session")
def bundle(tmp_path_factory) -> Path:
    """Every upstream value, generated once per session from the vendored checkout.

    Session-scoped because the generator runs the real upstream loops; per-test would pay
    that repeatedly for identical numbers.
    """
    try:
        st = upstream.load()
    except FileNotFoundError as exc:  # pragma: no cover - submodule not checked out
        pytest.skip(str(exc))

    G._pin_determinism()
    out = tmp_path_factory.mktemp("stalign_reference")
    clouds = F.make_clouds()
    for write in (
        G._write_primitives,
        G._write_energy,
        G._write_gradients,
        G._write_trajectory,
        G._write_image_trajectory,
        G._write_image_trajectory_matched,
        G._write_image_mixed_units,
        G._write_slice,
    ):
        write(st, clouds, out)

    import matplotlib.pyplot as plt

    plt.close("all")  # upstream leaks 4 figures per LDDMM call
    return out


# Tolerances are calibrated from the measured gaps between the port and upstream,
# not guessed. Everything the port reproduces faithfully lands at 1e-15 or better, so a
# 1e-12 budget leaves three orders of headroom while still being three orders tighter
# than any divergence we are pinning.
EXACT = 1e-12

#: ``lddmm`` carries no defaults of its own -- they live in ``_stalign._SOLVER_DEFAULTS``
#: so they exist in one place, and the ``fit_stalign_*`` wrappers resolve them. The
#: reference bundle was generated with these values, and ``_stalign_fixtures.py`` cannot
#: carry them because it is checksummed against the bundle.
_SOLVER_TAIL = {"muA": None, "muB": None, "tol": None, "patience": 25}


def _skip_without_x64() -> None:
    """float64 is a precondition, and it cannot be turned on from inside a test.

    ``jax.config.update`` is process-global; under ``-n auto`` every worker imports every
    test module, so setting it here would silently flip the float32 tests in
    ``test_stalign.py`` to float64 in the same worker. It has to come from the
    environment.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip(
            "STalign reference comparison needs float64 (upstream is double throughout). "
            "Set JAX_ENABLE_X64=1 in the environment -- not via jax.config.update, which "
            "would corrupt the float32 tests in this directory."
        )


def _load(bundle: Path, name: str) -> np.lib.npyio.NpzFile:
    return np.load(bundle / f"{name}.npz")


def rel(actual, expected) -> float:
    """Relative L2 error, the single measure used throughout."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / max(denominator, np.finfo(float).tiny))


@pytest.fixture(scope="session", autouse=True)
def _x64():
    _skip_without_x64()


@pytest.fixture(scope="session")
def primitives(bundle):
    return _load(bundle, "primitives")


@pytest.fixture(scope="session")
def clouds():
    return F.make_clouds()


@pytest.fixture(scope="session")
def source_grid(primitives):
    """The *moving* raster and its axes, as ``((y, x), image)``.

    The query moves onto the reference, so the query raster is the source -- the same
    role LDDMM's ``I``/``xI`` and ``pointsI`` play.
    """
    return (jnp.asarray(primitives["raster_query_y"]), jnp.asarray(primitives["raster_query_x"])), jnp.asarray(
        primitives["raster_query"]
    )


@pytest.fixture(scope="session")
def target_grid(primitives):
    """The *fixed* raster and its axes: the reference."""
    return (jnp.asarray(primitives["raster_ref_y"]), jnp.asarray(primitives["raster_ref_x"])), jnp.asarray(
        primitives["raster_ref"]
    )


@pytest.fixture(scope="session")
def velocity_grid(primitives):
    """Upstream's own velocity grid, which ``_lddmm_loss`` accepts directly."""
    return jnp.asarray(primitives["xv_upstream_0"]), jnp.asarray(primitives["xv_upstream_1"])


@pytest.fixture(scope="session")
def fitted(primitives, velocity_grid):
    """A :class:`Stalign2DResult` carrying upstream's own fitted deformation.

    Point transforms are compared through the public result object rather than the
    internal row-col helper, so what is pinned is what callers actually reach.
    """
    return Stalign2DResult(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
        aligned_points=jnp.zeros((0, 2)),
    )


def _transform_rc(result: Stalign2DResult, points_rc, *, direction: str = "forward") -> np.ndarray:
    """``result.transform`` on row-col points, for comparison with upstream.

    The public API speaks ``(x, y)`` and the reference speaks row-col, so the flip that
    ``transform`` performs internally is undone on both sides here. Anything the flip
    itself got wrong would still show up -- it is applied, not bypassed.
    """
    got = result.transform(np.asarray(points_rc)[:, ::-1], direction=direction)
    return np.asarray(got)[:, ::-1]


# --------------------------------------------------------------------------------------
# Provenance -- the fixtures have to stay falsifiable
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "primitives",
        "energy",
        "gradients",
        "trajectory_n1",
        "trajectory_n5",
        "trajectory_n50",
        "converged_n500",
        "image_trajectory",
        "image_trajectory_matched",
        "slice_trajectory",
    ],
)
def test_every_fixture_carries_provenance(bundle, name):
    """Without this the .npz files are unfalsifiable magic numbers within a year."""
    payload = json.loads(str(_load(bundle, name)["__provenance__"]))
    assert payload["upstream_sha"] == "b2068edc98974efa54537eca194736e177bbe11d"
    for key in ("ports_commit", "torch", "numpy", "python", "platform", "generated_utc"):
        assert payload[key], f"{name}: empty provenance field {key!r}"


def test_fixture_definitions_have_not_drifted(bundle):
    """The generator and these tests must build inputs from the same file, byte for byte.

    ``_stalign_fixtures.py`` is vendored from squidpy-ports. If someone edits one copy
    without regenerating, every comparison below silently compares different inputs.
    """
    recorded = json.loads(str(_load(bundle, "primitives")["__provenance__"]))["fixtures_checksum"]
    assert F.checksum() == recorded, (
        "_stalign_fixtures.py differs from the copy the reference bundle was generated "
        "with. Re-sync it from squidpy-ports and regenerate the bundle."
    )


# --------------------------------------------------------------------------------------
# Fixture validation -- runs before anything that depends on it
# --------------------------------------------------------------------------------------


def test_fixture_samples_are_off_grid(primitives):
    """No interpolation sample may sit on a grid line. See ledger row D10.

    Upstream and squidpy compute the fractional index by formulas that agree to ~1 ulp;
    exactly on a grid line they can floor() to different neighbours, giving an O(1)
    difference that says nothing about the port.
    """
    coords = np.asarray(primitives["interp_coords"])
    axes = (np.asarray(primitives["raster_ref_y"]), np.asarray(primitives["raster_ref_x"]))
    for axis, values in zip(axes, coords, strict=True):
        fractional = np.abs(np.modf((values - axis[0]) / (axis[1] - axis[0]))[0])
        assert np.minimum(fractional, 1.0 - fractional).min() > 1e-6


def test_fixture_stays_inside_velocity_grid(primitives, velocity_grid):
    """Keeps the padding divergence (D5) out of every test that is not about it."""
    points = np.asarray(primitives["points"])
    for axis, column in zip(velocity_grid, points.T, strict=True):
        assert column.min() > float(axis[0]) and column.max() < float(axis[-1])


# --------------------------------------------------------------------------------------
# Primitives that the port reproduces faithfully
# --------------------------------------------------------------------------------------


def test_interp_matches_upstream(primitives, source_grid, record_property):
    """``_interp`` vs ``STalign.interp(padding_mode='border')``."""
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords"]))
    error = rel(got, primitives["interp_border"])
    record_property("rel_error", error)
    assert error < EXACT


def test_regularizer_matches_upstream(primitives, velocity_grid, record_property):
    """``LL``/``K``/``DV`` vs STalign.py:1078-1090.

    This is a precondition, not a nicety: the regulariser sets the scale of the whole
    velocity term, so if it disagreed every later comparison would be meaningless.
    """
    kernel, ll, dv_prod = _core._build_regularizer(velocity_grid, a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])
    for name, got, expected in (
        ("LL", ll, primitives["regularizer_LL"]),
        ("K", kernel, primitives["regularizer_K"]),
        ("DV", dv_prod, primitives["regularizer_DV"]),
    ):
        error = rel(got, expected)
        record_property(f"rel_error_{name}", error)
        assert error < EXACT, name


def test_transform_grid_backward_matches_upstream(primitives, source_grid, target_grid, velocity_grid, record_property):
    """``Stalign2DResult.deformation_grid(direction='backward')`` vs ``STalign.build_transform('b')``.

    This is the inner loop of the objective: invert the affine, then integrate ``-v``
    backwards in time. Upstream returns ``(H, W, 2)``; squidpy returns ``(2, H, W)``.

    Through the public method rather than ``_core.transform_grid_row_col``: it delegates to
    exactly that call on the same fitted state, and its docstring pins that it is "not an
    approximation for plotting".
    """
    axes, _ = target_grid
    query_axes, _ = source_grid
    result = Stalign2DResult(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
    )
    # Backward evaluates the *reference* grid in the query frame, so the grid under test
    # goes in as `ref_axes`; the query side is only there to satisfy the guard.
    got = result.deformation_grid(direction="backward", query_axes=query_axes, ref_axes=axes)
    error = rel(np.moveaxis(np.asarray(got), 0, -1), primitives["grid_backward"])
    record_property("rel_error", error)
    assert error < EXACT


def test_warp_image_uses_the_upstream_grid(primitives, source_grid, target_grid, velocity_grid, record_property):
    """``Stalign2DResult.warp_image`` vs sampling on upstream's own backward grid.

    Each half was already checked against upstream separately; feeding upstream's
    ``grid_backward`` through the same interpolation isolates the composition.
    """
    source_axes, source_image = source_grid
    target_axes, _ = target_grid
    result = Stalign2DResult(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
        aligned_points=jnp.zeros((0, 2)),
        query_axes=source_axes,
        ref_axes=target_axes,
    )
    upstream_grid = jnp.asarray(np.moveaxis(np.asarray(primitives["grid_backward"]), -1, 0))
    expected = _core.interp(source_axes, source_image, upstream_grid)

    error = rel(result.warp_image(source_image), expected)
    record_property("rel_error", error)
    assert error < EXACT


def test_transform_points_forward_matches_upstream(fitted, primitives, record_property):
    """``Stalign2DResult.transform`` vs ``STalign.transform_points_source_to_target``."""
    got = _transform_rc(fitted, primitives["points"])
    error = rel(got, primitives["points_forward"])
    record_property("rel_error", error)
    assert error < EXACT


def test_to_affine_matches_upstream(primitives):
    """``_to_affine`` vs ``STalign.to_A``."""
    got = _core._to_affine(jnp.asarray(primitives["to_A_linear"]), jnp.asarray(primitives["to_A_translation"]))
    np.testing.assert_allclose(np.asarray(got), primitives["to_A"], rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# The objective itself
# --------------------------------------------------------------------------------------


def _loss_arguments(primitives, source_grid, target_grid, velocity_grid, *, nt, with_points):
    source_axes, source_image = source_grid
    target_axes, target_image = target_grid
    kernel, ll, dv_prod = _core._build_regularizer(velocity_grid, a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])
    empty = jnp.zeros((0, 2))
    landmarks_source = jnp.asarray(primitives["landmarks_query"])[:, ::-1]
    landmarks_target = jnp.asarray(primitives["landmarks_ref"])[:, ::-1]
    return (
        kernel,
        {
            "x_source": source_axes,
            "source_image": source_image,
            "x_target": target_axes,
            "target_image": target_image,
            "xv": velocity_grid,
            # LDDMM's initial state: uniform 0.5 matching weights (STalign.py:1102).
            "match_weights": jnp.full(target_image.shape[1:], 0.5),
            "ll": ll,
            "dv_prod": dv_prod,
            "points_source": landmarks_source if with_points else empty,
            "points_target": landmarks_target if with_points else empty,
            "sigmaM": F.LDDMM_PARAMS["sigmaM"],
            "sigmaR": F.LDDMM_PARAMS["sigmaR"],
            "sigmaP": F.LDDMM_PARAMS["sigmaP"],
        },
        jnp.zeros((nt, velocity_grid[0].shape[0], velocity_grid[1].shape[0], 2)),
    )


@pytest.mark.parametrize("nt", [1, 3])
@pytest.mark.parametrize("with_points", [False, True])
@pytest.mark.parametrize("warm", [False, True], ids=["v_zero", "v_nonzero"])
def test_energy_matches_upstream(
    bundle, primitives, source_grid, target_grid, velocity_grid, nt, with_points, warm, record_property
):
    """``_lddmm_loss`` vs upstream's ``E`` at iteration 0 -- the two optimise the same function.

    The ``v_nonzero`` half is not redundant: at ``v = 0`` the regularisation term contributes
    nothing, so evaluating only there would pass even with ``ER`` negated.
    """
    energy = _load(bundle, "energy")
    _, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=nt, with_points=with_points
    )
    key = f"E_nt{nt}_{'points' if with_points else 'nopoints'}"
    if warm:
        velocity = jnp.asarray(energy[f"warm_velocity_nt{nt}"])
        key += "_warmv"

    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    error = rel(total, energy[key])
    record_property("rel_error", error)
    assert error < EXACT, f"squidpy {float(total)!r} vs upstream {float(energy[key])!r}"


def test_energy_budget_is_not_vacuous(bundle, primitives, source_grid, target_grid, velocity_grid):
    """A tolerance that a perturbed run also passes is decoration, not a test.

    Nudge one physically meaningful knob and the same assertion must fail.
    """
    energy = _load(bundle, "energy")
    _, kwargs, velocity = _loss_arguments(primitives, source_grid, target_grid, velocity_grid, nt=3, with_points=True)
    kwargs["sigmaM"] *= 1.0001
    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    assert rel(total, energy["E_nt3_points"]) > EXACT


# --------------------------------------------------------------------------------------
# Gradients -- ledger row D3
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session", params=[False, True], ids=["v_zero", "v_nonzero"])
def measured_gradients(request, bundle, primitives, source_grid, target_grid, velocity_grid):
    """Gradients of ``_lddmm_loss``, at ``v = 0`` and at a non-zero velocity.

    Both are needed for the same reason as the energy test: ``dER/dv`` vanishes at the
    origin, so gradients taken only there cannot see the regularisation term.
    """
    gradients = _load(bundle, "gradients")
    kernel, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=F.LDDMM_PARAMS["nt"], with_points=True
    )
    suffix = ""
    if request.param:
        velocity = jnp.asarray(gradients["warm_velocity"])
        suffix = "_warmv"

    (_, _), (grad_l, grad_t, grad_v) = jax.value_and_grad(_core._lddmm_loss, argnums=(0, 1, 2), has_aux=True)(
        jnp.asarray(gradients["L"]), jnp.asarray(gradients["T"]), velocity, **kwargs
    )
    # Upstream stores the Sobolev-smoothed velocity gradient, since that is what actually
    # drives the step (STalign.py:1215).
    smoothed = jnp.fft.ifftn(jnp.fft.fftn(grad_v, axes=(1, 2)) * kernel[None, ..., None], axes=(1, 2)).real
    return gradients, {"L": grad_l, "T": grad_t, "v": smoothed}, suffix


@pytest.mark.parametrize(("component", "key"), [("L", "grad_L"), ("T", "grad_T"), ("v", "grad_v_smoothed")])
def test_gradients_match_upstream(measured_gradients, component, key, record_property):
    """``dE/dL``, ``dE/dT`` and the smoothed ``dE/dv`` vs upstream's, at iteration 0.

    Together with the energy test this pins both halves of the optimisation: the same
    objective *and* the same search direction. See ledger row D3 -- these disagreed by
    ~1.2e-3 until ``_contrast_transform`` stopped differentiating through the ridge solve.
    """
    gradients, computed, suffix = measured_gradients
    error = rel(computed[component], gradients[key + suffix])
    record_property("rel_error", error)
    assert error < EXACT


def test_contrast_transform_freezes_the_ridge_coefficients(record_property):
    """Regression guard for ledger row D3, stated without reference to upstream.

    The ridge fit is an EM M-step solved exactly at the current estimate, so its coefficients
    must be constant with respect to the optimisation. Dropping ``stop_gradient`` changes only
    the gradient, not the value -- invisible to any test that checks outputs.
    """
    rng = np.random.default_rng(F.SEED)
    warped = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    target = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    weights = jnp.asarray(rng.uniform(0.2, 0.8, size=(12, 15)))

    def live(x):
        return jnp.sum(_core._contrast_transform(x, target, weights) ** 2)

    def leaky(x):
        # _contrast_transform verbatim, minus the stop_gradient: the bug as it was.
        flat_source = x.reshape(x.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        design = jnp.concatenate((jnp.ones((1, flat_source.shape[1]), dtype=x.dtype), flat_source), axis=0)
        weighted = design * weights.reshape(-1)[None, :]
        coefficients = jnp.linalg.solve(
            weighted @ design.T + 0.1 * jnp.eye(design.shape[0], dtype=x.dtype),
            weighted @ flat_target.T,
        )
        return jnp.sum((coefficients.T @ design).reshape(target.shape) ** 2)

    # Identical value, different gradient -- which is exactly why this needs its own test.
    np.testing.assert_allclose(float(live(warped)), float(leaky(warped)), rtol=1e-12)
    error = rel(jax.grad(live)(warped), jax.grad(leaky)(warped))
    record_property("rel_error_vs_leaky", error)
    assert error > 1e-6, "the ridge coefficients are being differentiated through again"


# --------------------------------------------------------------------------------------
# Pinned divergences
# --------------------------------------------------------------------------------------


def test_rasterize_grid_matches_upstream(primitives, clouds):
    """Ledger row D2, fixed: the raster axes are identical to upstream's."""
    grid_x, grid_y, _ = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    np.testing.assert_allclose(grid_x, primitives["raster_ref_x"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(grid_y, primitives["raster_ref_y"], rtol=0, atol=1e-9)


def test_velocity_grid_matches_upstream(primitives, source_grid):
    """Ledger row D2, fixed: same off-by-one, same fix, in ``_build_velocity_grid``."""
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
    np.testing.assert_allclose(np.asarray(built[0]), primitives["xv_upstream_0"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(built[1]), primitives["xv_upstream_1"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("step", [30.0, 7.3, 0.017])
def test_grid_length_is_stable_against_float_rounding(step):
    """The other half of D2: ``np.arange(lo, hi + step, step)`` had an unstable length.

    Deriving the count from the interval is exact; the old form emitted one or two extra points
    depending on how ``hi`` rounded. Reached through ``squidpy.experimental.im``, which shares
    the builder with the velocity grid so the two cannot drift apart.
    """
    from squidpy.experimental.im._rasterize_points import axis

    assert axis is _core._axis
    for start, n in ((-400.4123, 33), (0.1, 41), (-1234.567, 77)):
        assert axis(start, start + n * step, step).size == n


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D6: squidpy integrates the backward flow in reversed time order, "
        "upstream does not (STalign.py:1828-1843). squidpy is correct -- see "
        "test_backward_transform_inverts_better, which is the assertion that matters. "
        "This xfail exists to pin the literal difference, and should be deleted only if "
        "squidpy ever deliberately adopts upstream's ordering."
    ),
)
def test_transform_points_backward_matches_upstream(fitted, primitives):
    got = _transform_rc(fitted, primitives["points"], direction="backward")
    assert rel(got, primitives["points_backward"]) < EXACT


def test_backward_transform_inverts_better(fitted, primitives, record_property):
    """D6, stated usefully: squidpy's backward map is the better inverse of the forward one."""
    points = np.asarray(primitives["points"])
    roundtrip = _transform_rc(fitted, primitives["points_forward"], direction="backward")
    ours = rel(roundtrip, points)
    theirs = rel(primitives["points_roundtrip"], points)
    record_property("roundtrip_squidpy", ours)
    record_property("roundtrip_upstream", theirs)
    assert ours <= theirs


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D5: upstream samples the velocity field with grid_sample's default "
        "padding_mode='zeros' (STalign.py:1163, :1167), squidpy uses "
        "map_coordinates(mode='nearest'). squidpy is correct -- zeros make a point that "
        "drifts off the velocity grid snap to no displacement at all."
    ),
)
def test_interp_outside_domain_matches_upstream(primitives, source_grid):
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    assert rel(got, primitives["interp_zeros_outside"]) < EXACT


def test_interp_outside_domain_is_border_padding(primitives, source_grid, record_property):
    """The positive half of D5: squidpy's behaviour is exactly upstream's 'border' mode."""
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    error = rel(got, primitives["interp_border_outside"])
    record_property("rel_error", error)
    assert error < EXACT
    assert rel(got, primitives["interp_zeros_outside"]) > 0.1


# --------------------------------------------------------------------------------------
# Budgeted divergences
# --------------------------------------------------------------------------------------

#: Per-blur relative-L2 budget for the rasteriser (ledger row D1). squidpy bins onto a
#: grid and convolves once; upstream splats an exact sub-pixel Gaussian per point and
#: renormalises it over a truncated window. Measured 4.08 % / 0.81 % / 2.87 % at blur
#: 2.0 / 1.0 / 0.5 -- 6 % leaves headroom without going vacuous.
RASTER_BUDGET = 0.06


def test_rasterize_stays_within_budget(primitives, clouds, record_property):
    """D1 is a deliberate speedup, so it gets a measured budget rather than equality."""
    _, _, got = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    expected = primitives["raster_ref"]
    got = np.asarray(got)
    assert got.shape == expected.shape

    for index, blur in enumerate(F.RASTER_PARAMS["blur"]):
        error = rel(got[index], expected[index])
        correlation = float(np.corrcoef(got[index].ravel(), expected[index].ravel())[0, 1])
        record_property(f"rel_error_blur{blur}", error)
        assert error < RASTER_BUDGET, f"blur={blur}: relL2 {error:.4%} exceeds {RASTER_BUDGET:.0%}"
        assert correlation > 0.99, f"blur={blur}: correlation {correlation:.5f}"


def test_rasterize_conserves_mass(clouds, record_property):
    """Every point contributes exactly one unit, wherever it sits.

    Upstream renormalises each point's kernel over its (possibly clipped) window, so a
    point near the border still carries unit mass. A plain ``mode="constant"`` blur does
    not, and used to lose 3 % of the total at the coarsest scale -- a density biased low
    around the whole rim.
    """
    n_points = clouds.ref.shape[0]
    _, _, got = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)

    for index, blur in enumerate(F.RASTER_PARAMS["blur"]):
        mass = float(np.asarray(got)[index].sum())
        record_property(f"mass_blur{blur}", mass)
        assert mass == pytest.approx(n_points, rel=1e-9), f"blur={blur}: {mass:.2f} of {n_points}"


def _residual(linear, translation, source, target) -> float:
    source, target = np.asarray(source), np.asarray(target)
    return float(np.linalg.norm(source @ np.asarray(linear).T + np.asarray(translation) - target))


def test_affine_from_points_is_equivalent_when_well_conditioned(primitives, record_property):
    """D7: the two are different estimators, not the same one implemented twice.

    Upstream solves the normal equations for a plain least-squares fit; skimage solves a
    Hartley-normalised homogeneous system by SVD, minimising algebraic rather than geometric
    error. Their coefficients differ by ~1e-3 even on clean landmarks, so what must hold is
    that neither is meaningfully worse at the job.
    """
    source = np.asarray(primitives["landmarks_query"])[:, ::-1]
    target = np.asarray(primitives["landmarks_ref"])[:, ::-1]

    def wrap(points):
        """The points as the public API takes them: an ``obsm`` entry, stored verbatim."""
        adata = ad.AnnData(np.empty((np.asarray(points).shape[0], 0), dtype=float))
        adata.obsm["spatial"] = np.asarray(points, dtype=float)
        return adata

    # Through the *public* obs entry point. This case starts from points, which is the obs
    # modality, so nothing is reconstructed on the way in: `obsm` holds the coordinates
    # verbatim and squidpy derives the raster axes from `dx`/`raster_expand` exactly as the
    # array-level estimator does. `niter=0` fits nothing, so the returned affine *is* the
    # landmark initialisation -- which also pins the wiring, that `landmarks_*` really do
    # reach the solver as its starting affine.
    fit = align_stalign_obs(
        wrap(primitives["ref"]),
        wrap(primitives["query"]),
        landmarks_query=primitives["landmarks_query"],
        landmarks_ref=primitives["landmarks_ref"],
        niter=0,
        dx=F.RASTER_PARAMS["dx"],
        blur=F.RASTER_PARAMS["blur"],
        raster_expand=F.RASTER_PARAMS["expand"],
    )
    affine = np.asarray(fit.affine)
    linear, translation = affine[:2, :2], affine[:2, 2]

    ours = _residual(linear, translation, source, target)
    theirs = _residual(primitives["lt_well_L"], primitives["lt_well_T"], source, target)
    record_property("residual_squidpy", ours)
    record_property("residual_upstream", theirs)
    record_property("rel_error_L", rel(linear, primitives["lt_well_L"]))
    assert abs(ours - theirs) / theirs < 1e-2, f"squidpy {ours:.6f} vs upstream {theirs:.6f}"


def test_affine_from_points_survives_ill_conditioning(primitives, record_property):
    """D7, the half that matters: upstream's ``inv(XᵀX)`` collapses, skimage does not."""
    source, target = primitives["ill_src"], primitives["ill_dst"]
    linear, translation = _helpers.affine_from_points(jnp.asarray(source), jnp.asarray(target))

    ours = _residual(linear, translation, source, target)
    theirs = _residual(primitives["lt_ill_L"], primitives["lt_ill_T"], source, target)
    record_property("residual_squidpy", ours)
    record_property("residual_upstream", theirs)
    assert ours < theirs * 1e-6, (
        f"expected upstream to lose badly on near-collinear landmarks, got squidpy {ours:.3e} vs upstream {theirs:.3e}"
    )


# --------------------------------------------------------------------------------------
# The whole iteration loop
# --------------------------------------------------------------------------------------


def _run_lddmm(primitives, snapshot, niter):
    source_axes = (jnp.asarray(primitives["raster_query_y"]), jnp.asarray(primitives["raster_query_x"]))
    target_axes = (jnp.asarray(primitives["raster_ref_y"]), jnp.asarray(primitives["raster_ref_x"]))
    return _core.lddmm(
        source_axes,
        jnp.asarray(primitives["raster_query"]),
        target_axes,
        jnp.asarray(primitives["raster_ref"]),
        L=jnp.asarray(snapshot["L"]),
        T=jnp.asarray(snapshot["T"]),
        points_source=jnp.asarray(primitives["landmarks_query"])[:, ::-1],
        points_target=jnp.asarray(primitives["landmarks_ref"])[:, ::-1],
        niter=niter,
        **F.LDDMM_PARAMS,
        **_SOLVER_TAIL,
    )


#: Relative-error budget after ``n`` gradient steps. Single steps agree to ~1e-14; the
#: allowance grows because bilinear-resampling VJPs and FFTs accumulate in different
#: orders on the two backends, and 50 steps of gradient descent amplify that.
_TRAJECTORY_BUDGET = {1: 1e-10, 5: 1e-9, 50: 1e-6}


@pytest.mark.parametrize("niter", [1, 5, 50])
def test_trajectory_matches_upstream(bundle, primitives, niter, record_property):
    """Run the real loop for ``n`` steps and compare every state it carries.

    ``niter=50`` is deliberate: the mixture-weight E-step is gated on ``it >= 50``
    (STalign.py:1233), so below it the weights stay frozen and that branch goes untested.
    Upstream builds ``A`` at the top of each iteration, so ``LDDMM(n)["A"]`` reflects ``n-1``
    updates; the fixture stores the un-lagged affine as ``A``. See ledger row D4.
    """
    snapshot = _load(bundle, f"trajectory_n{niter}")
    result = _run_lddmm(primitives, snapshot, niter)

    budget = _TRAJECTORY_BUDGET[niter]
    for name, got, expected in (
        ("A", result["A"], snapshot["A"]),
        ("v", result["v"], snapshot["v"]),
        ("WM", result["WM"], snapshot["WM"]),
        ("WA", result["WA"], snapshot["WA"]),
        ("WB", result["WB"], snapshot["WB"]),
    ):
        error = rel(got, expected)
        record_property(f"rel_error_{name}", error)
        assert error < budget, f"n={niter} {name}: {error:.3e} exceeds {budget:.0e}"


#: Matches ``IMAGE_PARAMS`` / ``IMAGE_ITERS`` in the generator. The image path works in
#: pixel units, so the kernel width and velocity step are far below the micron-scale
#: point-cloud defaults. The trailing five are upstream's own defaults, which the
#: generator left unset -- spelled out here because ``lddmm`` no longer carries any.
_IMAGE_PARAMS = {
    "a": 8.0,
    "p": 2.0,
    "expand": 2.0,
    "nt": 2,
    "diffeo_start": 4,
    "epV": 1.0,
    "epL": 2e-8,
    "epT": 2e-1,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 5e5,
    "sigmaP": 2e1,
}
_IMAGE_ITERS = 12


@pytest.fixture(scope="session")
def image_reference(bundle):
    return _load(bundle, "image_trajectory")


def _harness_shaped_fit(fixture, *, swap_roles=False):
    """The image fit exactly as the retired harness issued it.

    Through the harness's own `as_sdata` / `_initial_affine_xy` rather than a second copy of
    them, so this fails if either drifts.
    """
    from tests._replay import (
        _IMAGE_KEY,
        _channels_first,
        _initial_affine_xy,
        as_sdata,
    )

    ax_moving = (fixture["source_axis_0"], fixture["source_axis_1"])
    ax_fixed = (fixture["target_axis_0"], fixture["target_axis_1"])
    moving = as_sdata(_channels_first(fixture["query"], ndim=2), ax_moving)
    fixed = as_sdata(_channels_first(fixture["ref"], ndim=2), ax_fixed)
    affine = _initial_affine_xy({"L": np.asarray(fixture["start_L"]), "T": np.asarray(fixture["start_T"])})
    # `query` is the moving side, so upstream's `I` goes in second. Swapped on request, which
    # is what this guards against.
    first, second = (moving, fixed) if swap_roles else (fixed, moving)
    return align_stalign_image(
        first, second, image_key=_IMAGE_KEY, initial_affine=affine, niter=_IMAGE_ITERS, **_IMAGE_PARAMS
    )


def test_replay_image_call_reproduces_upstream(image_reference, record_property):
    """The call the notebook replay actually issues, against upstream's own result.

    The gap this closes: the reference suite pinned ``_core.lddmm`` and ``fit_stalign_image``
    against upstream, and the harness pinned its keyword conversion, but nothing checked the
    *composition* -- that the replay hands the public entry point the arguments it means to.
    It did not. Upstream's moving image went in as ``ref`` (the fixed side), its row-col
    landmarks went in as ``(x, y)``, and its row-col starting affine went into a parameter
    that reverses the axes itself. Three mirrored errors, largely self-cancelling on the
    fourteen notebooks whose two images are alike, and not at all on the one where they are
    not (ledger row D12).
    """
    fit = _harness_shaped_fit(image_reference)
    error = rel(fit.affine, image_reference["A"])
    record_property("rel_error", error)
    assert error < 1e-10, f"the replay's own call disagrees with upstream by {error:.3e}"


def test_replay_image_call_is_role_sensitive(image_reference):
    """Swapping ref and query must be *loud*, not a 1e-5 wobble.

    The original inversion survived because its effect was small wherever the two images were
    similar. Asserting the swap disagrees grossly is what keeps that from being reintroduced
    and absorbed into a tolerance.
    """
    swapped = _harness_shaped_fit(image_reference, swap_roles=True)
    assert rel(swapped.affine, image_reference["A"]) > 1e-3


@pytest.fixture(scope="session")
def image_mixed_units(bundle):
    """Source in one unit, target in another -- see :func:`generate._write_image_mixed_units`."""
    return _load(bundle, "image_mixed_units")


def _fit_mixed_units(fixture, niter):
    """The image path on the mixed-unit axes, from the fixture's own ``a``.

    No landmarks and no starting affine: the velocity grid is built from the axes, ``a`` and
    ``expand`` alone, so neither can change what this measures, and leaving them out keeps
    the comparison to the one quantity under test.
    """
    params = {**_IMAGE_PARAMS, "a": float(fixture["a"])}
    return fit_stalign_image(
        fixture["ref"],
        fixture["query"],
        ref_axes=(fixture["target_axis_0"], fixture["target_axis_1"]),
        query_axes=(fixture["source_axis_0"], fixture["source_axis_1"]),
        niter=niter,
        **params,
    )


def test_mixed_unit_velocity_grid_matches_upstream(image_mixed_units, record_property):
    """The velocity grid upstream builds when the two sides disagree about units.

    The grid is the deformation's entire degrees of freedom, so a difference here is not a
    tolerance question: on `xenium-heimage-alignment` the port builds 48x66 where upstream
    builds 17x23, and that fit bends the section into a dome (ledger row D12).

    This passes, and that is the useful part: mixed units *alone* do not reproduce D12, so
    the trigger is something more specific about that notebook's inputs. Kept as the control
    that rules the general case out -- see D12 for what remains open.
    """
    fit = _fit_mixed_units(image_mixed_units, _IMAGE_ITERS)
    got = tuple(int(np.asarray(axis).size) for axis in fit.velocity_grid)
    expected = (int(image_mixed_units["xv_0"].size), int(image_mixed_units["xv_1"].size))
    record_property("velocity_grid_squidpy", got)
    record_property("velocity_grid_upstream", expected)
    assert got == expected, f"velocity grid {got} against upstream's {expected}"


def test_mixed_unit_axes_are_actually_mixed(image_mixed_units):
    """The fixture is only meaningful while the two sides really carry different units.

    A single normalisation of the generator would silently turn this into a duplicate of
    `image_trajectory` -- passing, and testing nothing.
    """
    source_step = float(image_mixed_units["source_axis_0"][1] - image_mixed_units["source_axis_0"][0])
    target_step = float(image_mixed_units["target_axis_0"][1] - image_mixed_units["target_axis_0"][0])
    assert source_step == pytest.approx(1.0)
    assert target_step == pytest.approx(G.MIXED_TARGET_STEP)


def test_image_path_axes_match_the_reference(image_reference):
    """The image entry point's coordinate convention, checked before its results.

    ``fit_stalign_image`` centres pixel coordinates rather than using the point path's physical
    microns. The solver agreeing on rasters says nothing about that, so it is pinned separately.
    """
    fit = fit_stalign_image(image_reference["ref"], image_reference["query"], niter=0, **_IMAGE_PARAMS)

    for got, expected in (
        (fit.query_axes[0], image_reference["source_axis_0"]),
        (fit.query_axes[1], image_reference["source_axis_1"]),
        (fit.ref_axes[0], image_reference["target_axis_0"]),
        (fit.ref_axes[1], image_reference["target_axis_1"]),
    ):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-12)


def _run_image(fixture, axes_ref: bool = True):
    """Drive the solver on the image fixture, from the reference's own starting affine."""
    source = (jnp.asarray(fixture["source_axis_0"]), jnp.asarray(fixture["source_axis_1"]))
    target = (jnp.asarray(fixture["target_axis_0"]), jnp.asarray(fixture["target_axis_1"]))
    return _core.lddmm(
        source,
        jnp.asarray(fixture["query"]),
        target,
        jnp.asarray(fixture["ref"]),
        L=fixture["start_L"],
        T=fixture["start_T"],
        niter=_IMAGE_ITERS,
        **_IMAGE_PARAMS,
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "v", "WM", "WA", "WB"])
def test_image_path_matches_upstream(image_reference, name, record_property):
    """The image entry point vs upstream's LDDMM on the same images, axes and start.

    ~11% of the target grid samples the source through padding, because the two rasters have
    different shapes and each is centred on its own centre. That agrees to 1e-12 as well, on
    values and on gradients.
    """
    result = _run_image(image_reference)
    error = rel(result[name], image_reference[name])
    record_property("rel_error", error)
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_image_energy_trace_matches_upstream(image_reference, record_property):
    """Every iteration of the objective, not just the endpoint.

    A trajectory can agree at the end while disagreeing throughout; comparing the whole
    trace is what localises a disagreement to the step it starts at.
    """
    result = _run_image(image_reference)
    got = np.asarray(result["energies"])
    expected = np.asarray(image_reference["energies"])
    worst = max(abs(got[i] - expected[i]) / abs(expected[i]) for i in range(expected.size))
    record_property("worst_iteration_rel_error", worst)
    assert worst < 1e-10


def test_image_warp_matches_upstream(image_reference, record_property):
    """``warp_image`` vs ``STalign.transform_image_source_to_target``.

    Exactly what ``align_stalign_image(key_added=...)`` writes, against upstream's own
    image-warping composition rather than a reassembled one.
    """
    result = _run_image(image_reference)
    fit = Stalign2DResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=jnp.zeros((0, 2)),
        query_axes=(jnp.asarray(image_reference["source_axis_0"]), jnp.asarray(image_reference["source_axis_1"])),
        ref_axes=(jnp.asarray(image_reference["target_axis_0"]), jnp.asarray(image_reference["target_axis_1"])),
    )
    error = rel(fit.warp_image(jnp.asarray(image_reference["query"])), image_reference["warped"])
    record_property("rel_error", error)
    assert error < 1e-9


def test_on_grid_sampling_costs_six_orders_of_magnitude(bundle, record_property):
    """Ledger row D10, measured rather than asserted.

    The control fixture crops both rasters to a common extent, so their axes are the same
    integers and samples keep landing exactly on grid lines -- where upstream and squidpy can
    ``floor()`` to different neighbours. Same solver, same inputs otherwise, yet accuracy drops
    from 1e-12 to ~1e-3 on the velocity field. Hence both fixtures start from a deliberately
    off-grid affine.
    """
    matched = _load(bundle, "image_trajectory_matched")
    axes = (jnp.asarray(matched["axis_0"]), jnp.asarray(matched["axis_1"]))
    result = _core.lddmm(
        axes,
        jnp.asarray(matched["query"]),
        axes,
        jnp.asarray(matched["ref"]),
        L=matched["start_L"],
        T=matched["start_T"],
        niter=_IMAGE_ITERS,
        **_IMAGE_PARAMS,
        **_SOLVER_TAIL,
    )
    degraded = rel(result["v"], matched["v"])
    record_property("rel_error_on_grid", degraded)
    assert 1e-6 < degraded < 1e-1, (
        f"on-grid sampling now costs {degraded:.3e}; if this has become exact, D10 is gone "
        f"and the off-grid fixture design can be simplified"
    )


def test_velocity_grid_is_the_one_the_reference_used(bundle, primitives, source_grid):
    """The trajectory comparison is only meaningful on a shared velocity grid.

    Upstream was driven onto this grid explicitly via its ``xv=``/``v=`` parameters, so
    if squidpy's construction ever diverges again the trajectory numbers would be
    comparing two different problems rather than two implementations.
    """
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
    snapshot = _load(bundle, "trajectory_n1")
    np.testing.assert_allclose(np.asarray(built[0]), snapshot["xv_0"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(built[1]), snapshot["xv_1"], rtol=0, atol=1e-9)


def test_converged_solution_matches_upstream(bundle, primitives, record_property):
    """500 iterations -- "enough to actually converge", per #1243.

    Elementwise equality is the wrong instrument this far in: 500 steps of descent
    amplify last-ulp backend differences without either answer being wrong. What must
    hold is that both converge to the same registration.
    """
    snapshot = _load(bundle, "converged_n500")
    result = _run_lddmm(primitives, snapshot, 500)

    # `lddmm` returns the whole per-iteration trace now; the last completed step is the
    # one upstream's `E_last` records.
    energy = rel(result["energies"][result["n_iter"] - 1], snapshot["E_last"])
    record_property("rel_error_E", energy)
    assert energy < 0.01, f"final energy differs by {energy:.3%}"

    # Convergence is a statement about the registration, not the objective: the affine
    # initialised from landmarks is already near-optimal, so `E` barely moves over the
    # run even though the fit is good. Target registration error is the honest measure.
    clouds = F.make_clouds()
    landmarks_ref = clouds.landmarks_ref_rc
    before = float(np.mean(np.linalg.norm(clouds.landmarks_query_rc - landmarks_ref, axis=1)))
    after = float(snapshot["tre_mean"])
    record_property("tre_before", before)
    record_property("tre_after", after)
    assert after < 0.4 * F.RASTER_PARAMS["dx"], f"reference did not converge: TRE {after:.2f}"
    assert after < before / 5.0, f"reference barely moved: TRE {before:.2f} -> {after:.2f}"

    converged = Stalign2DResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=jnp.zeros((0, 2)),
    )
    aligned = _transform_rc(converged, np.asarray(primitives["query"])[:, ::-1])
    displacement = np.linalg.norm(aligned - np.asarray(snapshot["aligned_points_rc"]), axis=1)
    percentile = float(np.percentile(displacement, 95))
    record_property("p95_displacement", percentile)
    assert percentile < 0.1 * F.RASTER_PARAMS["dx"], (
        f"95th-percentile point disagreement {percentile:.4f} exceeds a tenth of a grid cell"
    )


# --------------------------------------------------------------------------------------
# Rank 3: a section fitted into a reference volume
# --------------------------------------------------------------------------------------

#: Matches ``SLICE_PARAMS`` / ``SLICE_ITERS`` in the generator. ``epL``/``epT`` are far
#: below upstream's 3D defaults deliberately -- see the generator's note: at the published
#: values the first step throws the section off the volume into border padding, where the
#: objective is constant and its gradient exactly zero.
_SLICE_PARAMS = {
    "a": 8.0,
    "p": 2.0,
    "expand": 1.25,
    "nt": 2,
    "epL": 1e-9,
    "epT": 1e-2,
    "epV": 1.0,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 1e8,
}
_SLICE_ITERS = 12


@pytest.fixture(scope="session")
def slice_reference(bundle):
    return _load(bundle, "slice_trajectory")


@pytest.fixture(scope="session")
def slice_axes(slice_reference):
    """``(reference (z, y, x) axes, section (y, x) axes)``, both as JAX arrays."""
    reference = tuple(jnp.asarray(slice_reference[f"ref_axis_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(slice_reference[f"query_axis_{axis}"]) for axis in range(2))
    return reference, section


@pytest.fixture(scope="session")
def slice_target_axes(slice_reference, slice_axes):
    """The section's axes as the rank-3 solver sees them: a single-sample z prepended.

    This is upstream's ``if len(xJ) == 2: xJ = [[0.0], xJ[0], xJ[1]]`` (STalign.py:1416),
    and the whole of its 3D-to-slice special case.
    """
    _, section = slice_axes
    return (jnp.zeros(1, dtype=section[0].dtype), *section)


def test_slice_to_affine_matches_upstream(slice_reference):
    """``_to_affine`` at rank 3 vs ``STalign.to_A_3D``."""
    got = _core._to_affine(jnp.asarray(slice_reference["start_L"]), jnp.asarray(slice_reference["start_T"]))
    np.testing.assert_allclose(np.asarray(got), slice_reference["to_A"], rtol=0, atol=0)


def test_slice_velocity_grid_matches_upstream(slice_reference, slice_axes, record_property):
    """``_build_velocity_grid`` at rank 3 vs the grid upstream builds for itself.

    The regulariser and every velocity comparison below are defined on this grid, so a
    disagreement here would make all of them compare two different problems.
    """
    reference, _ = slice_axes
    built = _core._build_velocity_grid(reference, a=_SLICE_PARAMS["a"], expand=_SLICE_PARAMS["expand"])
    assert len(built) == 3
    for axis, got in enumerate(built):
        np.testing.assert_allclose(np.asarray(got), slice_reference[f"xv_{axis}"], rtol=0, atol=1e-9)


def test_slice_regularizer_matches_upstream(slice_reference, record_property):
    """``LL``/``K``/``DV`` at rank 3 vs STalign.py:1384-1397."""
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=_SLICE_PARAMS["a"], p=_SLICE_PARAMS["p"])
    for name, got, expected in (
        ("LL", ll, slice_reference["regularizer_LL"]),
        ("K", kernel, slice_reference["regularizer_K"]),
        ("DV", dv_prod, slice_reference["regularizer_DV"]),
    ):
        error = rel(got, expected)
        record_property(f"rel_error_{name}", error)
        assert error < EXACT, name


@pytest.mark.parametrize(("order", "key"), [(1, "interp_border"), (0, "interp_nearest")])
def test_slice_interp_matches_upstream(slice_reference, slice_axes, order, key, record_property):
    """``_interp`` at rank 3 vs ``STalign.interp3D``, linear and nearest.

    ``order=0`` is not decoration: reading integer structure ids off an annotation volume
    is the entire point of the recipe, and interpolating them linearly would average two
    ids into a third, unrelated one.
    """
    reference, _ = slice_axes
    grid = jnp.asarray(np.moveaxis(np.asarray(slice_reference["grid_backward"]), -1, 0))
    got = _core.interp(reference, jnp.asarray(slice_reference["ref"]), grid, order=order)
    error = rel(got, slice_reference[key])
    record_property("rel_error", error)
    assert error < EXACT


def test_slice_grid_backward_matches_upstream(slice_reference, slice_axes, record_property):
    """``Stalign3DResult.deformation_grid(direction='backward')`` at rank 3 vs ``build_transform3D``.

    Upstream's 3D backward integration runs in reversed time order (STalign.py:1474), the
    same order squidpy uses -- so ledger row D6, which pins a real disagreement on the 2D
    path, does not apply here and this is an equality rather than an xfail.

    The section's *two* axes go in: the rank-3 method applies upstream's single-sample z lift
    itself (STalign.py:1416) and rejects a grid that already carries one.
    """
    reference, section = slice_axes
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    result = Stalign3DResult(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=xv,
        ref_axes=reference,
        query_axes=section,
    )
    got = result.deformation_grid(direction="backward")
    error = rel(np.moveaxis(np.asarray(got), 0, -1), slice_reference["grid_backward"])
    record_property("rel_error", error)
    assert error < EXACT


def _run_slice(fixture, niter=_SLICE_ITERS):
    """Drive the rank-3 solver from the reference's own axes, images and starting affine.

    The velocity is frozen (``diffeo_start`` past ``niter``), matching the generator. That
    is the only regime in which upstream's 3D objective and a correct one agree -- see
    ``test_slice_regularizer_axes_diverge_from_upstream`` -- so it is the only one in which
    a trajectory comparison measures the port rather than the discrepancy.
    """
    reference = tuple(jnp.asarray(fixture[f"ref_axis_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(fixture[f"query_axis_{axis}"]) for axis in range(2))
    target = (jnp.zeros(1, dtype=section[0].dtype), *section)
    return _core.lddmm(
        reference,
        jnp.asarray(fixture["ref"]),
        target,
        jnp.asarray(fixture["query"])[:, None],
        L=jnp.asarray(fixture["start_L"]),
        T=jnp.asarray(fixture["start_T"]),
        niter=niter,
        diffeo_start=niter + 1,
        sigmaP=2e1,
        **_SLICE_PARAMS,
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "WM", "WA", "WB"])
def test_slice_path_matches_upstream(slice_reference, name, record_property):
    """The rank-3 solver vs upstream's ``LDDMM_3D_to_slice`` on the same inputs.

    The rank-3 counterpart of ``test_image_path_matches_upstream``: what says the solver core
    was generalised rather than rewritten.
    """
    result = _run_slice(slice_reference)
    error = rel(result[name], slice_reference[name])
    record_property("rel_error", error)
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_slice_energy_trace_matches_upstream(slice_reference, record_property):
    """Every iteration of the rank-3 objective, not just the endpoint."""
    result = _run_slice(slice_reference)
    got = np.asarray(result["energies"])
    expected = np.asarray(slice_reference["energies"])
    worst = max(abs(got[i] - expected[i]) / abs(expected[i]) for i in range(expected.size))
    record_property("worst_iteration_rel_error", worst)
    assert worst < 1e-10


def test_slice_fixture_actually_descends(slice_reference):
    """A trajectory that sits still would agree with anything.

    At upstream's published 3D step sizes the first step throws the section off the volume into
    border padding, where the gradient is exactly zero -- the run then holds one value and every
    comparison above passes vacuously.
    """
    energies = np.asarray(slice_reference["energies"])
    assert energies[-1] < energies[0], f"reference did not descend: {energies[0]} -> {energies[-1]}"
    assert len(np.unique(energies)) == energies.size, "reference objective plateaued; retune epL/epT"


def test_slice_transform_matches_upstream_coords(slice_reference, slice_axes, record_property):
    """``Stalign3DResult.transform`` vs upstream's ``coord0``/``coord1``/``coord2``.

    ``analyze3Dalign`` builds those by indexing ``build_transform3D``'s output at each cell's
    nearest raster cell (STalign.py:2001-2003); evaluating on the section's own grid points
    makes them directly comparable. What is pinned is the composition -- the lift onto
    ``z = 0``, the backward integration, and the ``(z, y, x)`` -> ``(x, y, z)`` reversal.
    """
    reference, section = slice_axes
    result = Stalign3DResult(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3)),
        ref_axes=reference,
        query_axes=section,
    )
    rows, cols = np.meshgrid(np.asarray(section[0]), np.asarray(section[1]), indexing="ij")
    points_xy = np.column_stack((cols.reshape(-1), rows.reshape(-1)))

    got = np.asarray(result.transform(points_xy))
    # `build_transform3D` is what `analyze3Dalign` calls, and it stores `(z, y, x)`;
    # the public method returns `(x, y, z)`.
    expected = np.asarray(slice_reference["grid_backward"])[0].reshape(-1, 3)[:, ::-1]
    error = rel(got, expected)
    record_property("rel_error", error)
    assert error < EXACT


def test_slice_transform_matches_the_solver_own_sampling_grid(slice_reference, slice_axes, record_property):
    """``transform`` vs ``Xs``, the grid the fitted loop itself sampled the volume through.

    ``Xs`` comes from the frozen-velocity run, so this is the affine-only map: it says the
    public method reproduces what the solver actually did, not what a separate upstream helper
    would do. Pairs with ``A_stale`` rather than ``A``, per ledger row D4.
    """
    reference, section = slice_axes
    result = Stalign3DResult(
        affine=jnp.asarray(slice_reference["A_stale"]),
        velocity=jnp.zeros_like(jnp.asarray(slice_reference["velocity"])),
        velocity_grid=tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3)),
        ref_axes=reference,
        query_axes=section,
    )
    rows, cols = np.meshgrid(np.asarray(section[0]), np.asarray(section[1]), indexing="ij")
    points_xy = np.column_stack((cols.reshape(-1), rows.reshape(-1)))

    got = np.asarray(result.transform(points_xy))
    expected = np.asarray(slice_reference["Xs"])[0].reshape(-1, 3)[:, ::-1]
    error = rel(got, expected)
    record_property("rel_error", error)
    assert error < EXACT


@pytest.mark.parametrize(("order", "key"), [(1, "interp_border"), (0, "interp_nearest")])
def test_slice_sample_volume_matches_upstream(slice_reference, slice_axes, order, key, record_property):
    """``im.sample_volume`` vs ``interp3D`` on upstream's own backward grid.

    The second half of the composition callers actually reach -- ``transform`` puts a cell in
    the reference frame, this reads the volume there, and that pair is how a cell gets an
    atlas value. Upstream's own grid goes in rather than ``transform``'s output, so a failure
    here is the interpolator and not the map.

    ``order=0`` is the case that matters for an annotation volume: interpolating integer
    structure ids would invent regions that do not exist. It is also what makes the label
    disagreement in ledger row D11 a step function of the fit.
    """
    reference, _ = slice_axes
    # `sample_volume` takes points in `(x, y, z)`, the reverse of `axes` in `(z, y, x)`.
    grid = np.asarray(slice_reference["grid_backward"])[0].reshape(-1, 3)[:, ::-1]
    got = sample_volume(jnp.asarray(slice_reference["ref"]), reference, grid, order=order)
    expected = np.asarray(slice_reference[key]).reshape(2, -1)
    error = rel(got, expected)
    record_property("rel_error", error)
    assert error < EXACT


def test_slice_regularizer_axes_diverge_from_upstream(slice_reference, record_property):
    """Ledger row D11, measured: squidpy's rank-3 regulariser is the 3-axis one.

    Upstream transforms two spatial axes in the regularisation *energy* (``dim=(1,2)``,
    STalign.py:1504) while smoothing that energy's gradient over all three (``dim=(1,2,3)``,
    :1527), so autograd propagates the mismatch into the search direction. On this velocity
    field upstream's value is ~5.4x the correct one. At rank 2 the readings coincide, which is
    why this has no 2D counterpart.
    """
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(slice_reference[f"query_axis_{axis}"]) for axis in range(2))
    reference = tuple(jnp.asarray(slice_reference[f"ref_axis_{axis}"]) for axis in range(3))
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=_SLICE_PARAMS["a"], p=_SLICE_PARAMS["p"])
    target_image = jnp.asarray(slice_reference["query"])[:, None]

    _, aux = _core._lddmm_loss(
        jnp.asarray(slice_reference["start_L"]),
        jnp.asarray(slice_reference["start_T"]),
        jnp.asarray(slice_reference["velocity"]),
        x_source=reference,
        source_image=jnp.asarray(slice_reference["ref"]),
        x_target=(jnp.zeros(1, dtype=section[0].dtype), *section),
        target_image=target_image,
        xv=xv,
        match_weights=jnp.full(target_image.shape[1:], 0.5),
        ll=ll,
        dv_prod=dv_prod,
        points_source=jnp.zeros((0, 3)),
        points_target=jnp.zeros((0, 3)),
        sigmaM=_SLICE_PARAMS["sigmaM"],
        sigmaR=_SLICE_PARAMS["sigmaR"],
        sigmaP=2e1,
    )
    reg_energy = aux[3]

    correct = rel(reg_energy, slice_reference["ER_3axis"])
    against_upstream = rel(reg_energy, slice_reference["ER_2axis"])
    record_property("rel_error_vs_3axis", correct)
    record_property("rel_error_vs_2axis", against_upstream)
    record_property("upstream_over_correct", float(slice_reference["ER_2axis"] / slice_reference["ER_3axis"]))
    assert correct < EXACT, f"squidpy no longer matches the 3-axis regulariser: {correct:.3e}"
    assert against_upstream > 0.1, (
        "squidpy now agrees with upstream's 2-axis regularisation energy; if that is "
        "deliberate, ledger row D11 has to be rewritten"
    )


def test_slice_fit_reaches_the_same_place_as_the_solver(slice_reference, record_property):
    """``fit_stalign_volume`` vs the solver it wraps, driven to the same starting affine.

    The entry point owns the section's z padding, the axis construction and the ``initial_*``
    -> ``L``/``T`` translation -- exactly what a solver-level comparison cannot see.
    """
    # `initial_affine` is public `(x, y, z)`; the fixture's is the solver's `(z, y, x)`.
    # Conjugating by the axis reversal converts it -- reversing the *spatial* axes only,
    # which is why this is a permutation of the first three rows and columns rather than
    # a plain `[::-1, ::-1]` (that would move the homogeneous row too).
    swap = np.eye(4)[[2, 1, 0, 3]]
    fit = fit_stalign_volume(
        slice_reference["ref"],
        slice_reference["query"],
        ref_axes=[slice_reference[f"ref_axis_{axis}"] for axis in range(3)],
        query_axes=[slice_reference[f"query_axis_{axis}"] for axis in range(2)],
        initial_affine=swap @ np.asarray(slice_reference["to_A"]) @ swap,
        niter=_SLICE_ITERS,
        diffeo_start=_SLICE_ITERS + 1,
        **_SLICE_PARAMS,
        **{key: value for key, value in _SOLVER_TAIL.items() if key != "sigmaP"},
    )
    error = rel(fit.affine, slice_reference["A"])
    record_property("rel_error_A", error)
    assert error < 1e-10


# --------------------------------------------------------------------------------------
# The ledger has to stay in sync
# --------------------------------------------------------------------------------------


def _harness_shaped_volume_fit(fixture, niter=_SLICE_ITERS):
    """The rank-3 fit exactly as the retired harness issued it.

    Through the replay's own `as_sdata` / `_initial_affine_xyz` / `solver_keys`, so this fails
    if any of them drifts. The velocity is frozen the way the generator freezes it
    (``diffeo_start`` past ``niter``); see :func:`_run_slice` for why that is the only regime
    where a rank-3 trajectory comparison measures the port rather than ledger row D11.
    """
    from tests._replay import (
        _IMAGE_KEY,
        _channels_first,
        _initial_affine_xyz,
        as_sdata,
        solver_keys,
    )

    reference_axes = [np.asarray(fixture[f"ref_axis_{axis}"]) for axis in range(3)]
    section_axes = [np.asarray(fixture[f"query_axis_{axis}"]) for axis in range(2)]
    forwarded = {"niter": niter, "diffeo_start": niter + 1, "sigmaP": 2e1, **_SLICE_PARAMS}
    return align_stalign_volume(
        as_sdata(_channels_first(fixture["ref"], ndim=3), reference_axes),
        as_sdata(_channels_first(fixture["query"], ndim=2), section_axes),
        image_key=_IMAGE_KEY,
        initial_affine=_initial_affine_xyz({"L": np.asarray(fixture["start_L"]), "T": np.asarray(fixture["start_T"])}),
        **{key: value for key, value in forwarded.items() if key in solver_keys()},
    )


def test_replay_volume_call_resolves_the_axes_it_was_given(slice_reference):
    """The rank-3 half of the question the rank-2 seam answered: are the axes bit-faithful?

    The public entry points take an element's *placement* and rebuild its physical axes from
    the scale and translation it carries, so a container round-trip sits between the replay
    and the solver. At rank 2 that round-trip is exact, which is what makes the rank-2 numbers
    trustworthy. Rank 3 adds a third axis and a section lifted onto ``z = 0``, and the atlas
    axes carry offsets far larger than their step -- exactly the case
    :func:`tests._replay.axis_placement` warns cannot always be recovered by differencing.
    """
    fit = _harness_shaped_volume_fit(slice_reference)

    for axis in range(3):
        expected = np.asarray(slice_reference[f"ref_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.ref_axes[axis]), expected, rtol=0, atol=1e-9)
    for axis in range(2):
        expected = np.asarray(slice_reference[f"query_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.query_axes[axis]), expected, rtol=0, atol=1e-9)


def test_replay_volume_call_reproduces_upstream(slice_reference, record_property):
    """The rank-3 counterpart of ``test_replay_image_call_reproduces_upstream``.

    Its absence is why this took a day to find at rank 2: every other rank-3 test drives the
    array-level ``fit_stalign_volume``, so the composition the replay actually performs --
    containers, placement round-trip, ``(x, y, z)`` affine order, keyword filtering -- was
    pinned nowhere. Three inverted conventions hid in that gap at rank 2 and cancelled well
    enough to look like agreement.

    Rank 3 cannot carry rank 2's role swap: the volume is rank 3 and the section rank 2, so
    the shapes refuse the exchange rather than quietly fitting the inverse. What it *can*
    carry is a mis-ordered affine or a lossy placement, which is what this measures.
    """
    fit = _harness_shaped_volume_fit(slice_reference)
    error = rel(fit.affine, slice_reference["A"])
    record_property("rel_error", error)
    assert error < 1e-10, f"the replay's own rank-3 call disagrees with upstream by {error:.3e}"
