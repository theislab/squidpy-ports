"""Guards on the reference itself.

These do not check STalign's maths — they check that the thing we call "the reference"
is still the pinned upstream, and that the spies in :mod:`squidpy_ports.stalign.upstream`
still line up with the loop they observe. If either drifts, every tolerance in squidpy's
test suite quietly becomes meaningless.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from squidpy_ports.stalign import fixtures as F
from squidpy_ports.stalign import upstream

#: The port hands these to `jax.jit` as static arguments, so each one must arrive as a
#: hashable Python scalar (`_core.py` `static_argnames`).
STATIC_ARGUMENTS = (
    "niter",
    "diffeo_start",
    "epL",
    "epT",
    "epV",
    "sigmaM",
    "sigmaA",
    "sigmaB",
    "sigmaR",
    "sigmaP",
    "tol",
    "patience",
)


def test_vendored_upstream_is_pinned(stalign):
    assert hasattr(stalign, "LDDMM")
    assert hasattr(stalign, "rasterize")


def test_staged_checkout_uses_preverified_sha(monkeypatch, tmp_path):
    """Scratch staging may omit Git metadata but must carry a verified source SHA."""
    failed = upstream.subprocess.CompletedProcess([], returncode=128, stdout="", stderr="not a repository")
    monkeypatch.setattr(upstream.subprocess, "run", lambda *args, **kwargs: failed)
    monkeypatch.setenv("SQUIDPY_PORTS_STALIGN_SHA", upstream.UPSTREAM_SHA)

    assert upstream._checkout_sha(tmp_path) == upstream.UPSTREAM_SHA


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


# --------------------------------------------------------------------------------------
# The rank-3 (volume-to-section) comparison's conversion helpers
# --------------------------------------------------------------------------------------


def test_reference_figures_are_verbatim_upstream_output():
    """Every published figure on the comparison page is upstream's own, unrecomputed.

    ``docs/_static/reference/SOURCES.md`` claims each PNG was copied verbatim from a named
    cell of a pinned upstream notebook. That claim is the whole basis for calling those
    panels "upstream's result" rather than "our replay of it", so it is checked by bytes
    rather than trusted -- a regenerated or edited figure, or a wrong cell reference, fails
    here. It caught one: `upstream-merfish-merfish.png` was documented as cell 44 of a
    37-cell notebook.
    """
    import base64

    from squidpy_ports.stalign import upstream

    reference = Path(__file__).resolve().parents[1] / "docs" / "_static" / "reference"
    rows = re.findall(
        r"^\|\s*`([^`]+\.png)`\s*\|[^|]*\|\s*\[([^\]]+)\]\([^)]+\)\s*\|\s*(\d+)\s*\|",
        (reference / "SOURCES.md").read_text(),
        re.M,
    )
    assert len(rows) >= 11, f"SOURCES.md table did not parse; got {len(rows)} rows"

    notebooks = upstream.vendor_root() / "docs" / "notebooks"
    for name, notebook, cell in rows:
        committed = reference / name
        assert committed.exists(), f"{name} is listed in SOURCES.md but not present"
        cells = json.loads((notebooks / notebook).read_text())["cells"]
        assert int(cell) < len(cells), f"{name}: SOURCES.md cites cell {cell}, {notebook} has {len(cells)}"
        published = [
            base64.b64decode(output["data"]["image/png"])
            for output in cells[int(cell)].get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        assert committed.read_bytes() in published, (
            f"{name} is not byte-identical to any image upstream committed at "
            f"{notebook} cell {cell} -- it was recomputed, edited, or the cell is wrong"
        )


def test_rank_three_defaults_are_actually_different_from_rank_two():
    """The five knobs where upstream's 3D entry point departs from its 2D one.

    If `_VOLUME_DEFAULTS` were ever collapsed into `_SOLVER_DEFAULTS`, the test above would
    still pass on every shared key while the rank-3 fit silently ran with 2D step sizes.
    """
    pytest.importorskip("squidpy")
    # Private by necessity: the port's default *values* have no public accessor, and this
    # is the check that a notebook passing nothing puts identical numbers on both sides.
    from squidpy.experimental.tl._align import _stalign

    differing = {
        name
        for name, value in _stalign._VOLUME_DEFAULTS.items()
        if name in _stalign._SOLVER_DEFAULTS and _stalign._SOLVER_DEFAULTS[name] != value
    }
    assert differing == {"expand", "epL", "epT", "epV", "sigmaR"}, sorted(differing)


def test_squidpy_commit_prefers_the_install_record_and_never_guesses(monkeypatch):
    """A manifest may record the squidpy commit only when it can prove it.

    A VCS install carries the resolved sha in pip's metadata; an editable install of a
    working tree carries only a directory, which is why the staged cluster job resolves the
    commit itself and passes `SQUIDPY_COMMIT`. With neither, the answer is `None` -- a
    manifest admitting it does not know beats one asserting a commit it cannot substantiate.
    """
    from squidpy_ports.stalign import upstream

    monkeypatch.delenv("SQUIDPY_COMMIT", raising=False)
    # no install record, no env var -> no claim
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: (_ for _ in ()).throw(FileNotFoundError))
    assert upstream.squidpy_commit() is None

    # the staged job's value is used when pip cannot answer
    monkeypatch.setenv("SQUIDPY_COMMIT", "0123456789abcdef")
    assert upstream.squidpy_commit() == "0123456789abcdef"

    # an empty env var is not a commit
    monkeypatch.setenv("SQUIDPY_COMMIT", "")
    assert upstream.squidpy_commit() is None
