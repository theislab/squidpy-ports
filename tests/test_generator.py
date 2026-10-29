"""Guards on the reference itself.

These do not check STalign's maths — they check that the thing we call "the reference"
is still the pinned upstream, and that the spies in :mod:`squidpy_ports.stalign.upstream`
still line up with the loop they observe. If either drifts, every tolerance in squidpy's
test suite quietly becomes meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidpy_ports.stalign import fixtures as F
from squidpy_ports.stalign import upstream


def test_vendored_upstream_is_pinned(stalign):
    assert hasattr(stalign, "LDDMM")
    assert hasattr(stalign, "rasterize")


def test_clouds_are_deterministic():
    first, second = F.make_clouds(), F.make_clouds()
    np.testing.assert_array_equal(first.ref, second.ref)
    np.testing.assert_array_equal(first.query, second.query)


def test_query_is_a_thinned_transform_of_ref():
    clouds = F.make_clouds()
    assert clouds.ref.shape == (F.N_POINTS, 2)
    assert clouds.query.shape[0] < F.N_POINTS  # dropout applied
    assert clouds.landmarks_ref.shape == (F.N_LANDMARKS, 2)
    # Row-col views must be materialised: torch.tensor rejects negative strides.
    assert clouds.ref_rc.flags["C_CONTIGUOUS"]


def test_energy_and_gradient_spies_agree_with_the_loop(stalign):
    """The two independent gradient extractions must agree.

    One reads autograd hooks on the live leaf tensors; the other backs the gradient out
    of the parameter delta across two iterations. They share no machinery, so agreement
    is real evidence that both still match the vendored loop.
    """
    clouds = F.make_clouds()
    xi, yi, image_ref = stalign.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], draw=0, **F.RASTER_PARAMS)
    xj, yj, image_query = stalign.rasterize(clouds.query[:, 0], clouds.query[:, 1], draw=0, **F.RASTER_PARAMS)
    lin, trans = stalign.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)

    kwargs = {
        "xI": [yi, xi],
        "I": image_ref,
        "xJ": [yj, xj],
        "J": image_query,
        "pointsI": clouds.landmarks_query_rc,
        "pointsJ": clouds.landmarks_ref_rc,
        **F.LDDMM_PARAMS,
    }

    _, captured = upstream.lddmm_with_grads(stalign, niter=1, L=lin, T=trans, **kwargs)
    two = stalign.LDDMM(niter=2, L=lin, T=trans, **kwargs)

    scale = 1.0 + 9.0 * (0 >= F.LDDMM_PARAMS["diffeo_start"])
    grad_l_delta = (lin - two["A"].numpy()[:2, :2]) / (F.LDDMM_PARAMS["epL"] / scale)
    np.testing.assert_allclose(captured["L"][0].numpy(), grad_l_delta, rtol=1e-6)

    assert np.isfinite(captured["E"][0])


def test_returned_affine_lags_one_step(stalign):
    """Upstream builds A at the top of the loop, so LDDMM(n)['A'] reflects n-1 updates.

    squidpy builds it after the loop. Every trajectory comparison depends on this offset,
    so it is asserted rather than assumed.
    """
    clouds = F.make_clouds()
    xi, yi, image_ref = stalign.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], draw=0, **F.RASTER_PARAMS)
    xj, yj, image_query = stalign.rasterize(clouds.query[:, 0], clouds.query[:, 1], draw=0, **F.RASTER_PARAMS)
    lin, trans = stalign.L_T_from_points(clouds.landmarks_query_rc, clouds.landmarks_ref_rc)
    kwargs = {"xI": [yi, xi], "I": image_ref, "xJ": [yj, xj], "J": image_query, **F.LDDMM_PARAMS}

    one = stalign.LDDMM(niter=1, L=lin, T=trans, **kwargs)
    np.testing.assert_array_equal(one["A"].numpy()[:2, :2], lin)

    two = stalign.LDDMM(niter=2, L=lin, T=trans, **kwargs)
    assert not np.array_equal(two["A"].numpy()[:2, :2], lin)


@pytest.mark.parametrize("name", ["THETA", "SHIFT"])
def test_fixture_transform_is_not_round(name):
    """A round angle or integer shift puts samples exactly on grid lines.

    There, upstream's and squidpy's index formulas — equal to ~1 ulp — can floor() to
    different neighbours, producing an O(1) disagreement that says nothing about the port.
    """
    value = np.atleast_1d(getattr(F, name)).astype(float)
    assert np.all(np.abs(value - np.round(value, 2)) > 1e-6)
