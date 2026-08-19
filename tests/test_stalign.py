"""Guards on the reference itself.

These do not check STalign's maths — they check that the thing we call "the reference"
is still the pinned upstream, and that the spies in :mod:`squidpy_ports.stalign.upstream`
still line up with the loop they observe. If either drifts, every tolerance in squidpy's
test suite quietly becomes meaningless.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from squidpy_ports.stalign import fixtures as F
from squidpy_ports.stalign import upstream
from squidpy_ports.stalign.notebook_suite import (
    NOTEBOOKS,
    PORT_DEFAULTS,
    PORT_PARAMETERS,
    THREE_D_NOTEBOOKS,
    UNREPLAYABLE_NOTEBOOKS,
    _convert_kwargs,
    _initial_affine_xyz,
    notebook_for_index,
    solver_keys,
    write_failure,
)

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


def convert(**kwargs: Any) -> dict[str, Any]:
    return _convert_kwargs(kwargs, PORT_PARAMETERS)


def test_vendored_upstream_is_pinned(stalign):
    assert hasattr(stalign, "LDDMM")
    assert hasattr(stalign, "rasterize")


def test_staged_checkout_uses_preverified_sha(monkeypatch, tmp_path):
    """Scratch staging may omit Git metadata but must carry a verified source SHA."""
    failed = upstream.subprocess.CompletedProcess([], returncode=128, stdout="", stderr="not a repository")
    monkeypatch.setattr(upstream.subprocess, "run", lambda *args, **kwargs: failed)
    monkeypatch.setenv("SQUIDPY_PORTS_STALIGN_SHA", upstream.UPSTREAM_SHA)

    assert upstream._checkout_sha(tmp_path) == upstream.UPSTREAM_SHA


def test_notebook_suite_maps_every_upstream_notebook():
    found = {path.name for path in (upstream.vendor_root() / "docs" / "notebooks").glob("*.ipynb")}
    assert set(NOTEBOOKS) == found
    assert len(THREE_D_NOTEBOOKS) == 2


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


def test_port_signature_matches_the_mirror():
    """Where squidpy is installed, `PORT_DEFAULTS` must still describe the real port.

    The conversion casts each captured argument to the type the mirror declares, so a
    renamed keyword would change what the suite sends. squidpy is not a dependency here (it
    is the thing under comparison), so the mirror stands in -- and is checked against the
    real thing whenever it can be.

    Checked against squidpy's *public* solver TypedDicts, not `lddmm`'s signature: the
    TypedDicts are exported and so carry a stability contract, and the four renames on
    2026-08-19 alone are the argument for depending on the surface that has one. The port's
    default *values* are deliberately not asserted -- nothing here supplies them any more,
    since the `align_stalign_*` entry points resolve their own.
    """
    pytest.importorskip("jax")
    pytest.importorskip("squidpy")

    declared = solver_keys()
    unmirrored = declared - set(PORT_DEFAULTS)
    assert not unmirrored, f"squidpy declares solver keywords the mirror does not: {sorted(unmirrored)}"


@pytest.mark.parametrize("wrap", [lambda v: v, np.asarray, torch.as_tensor], ids=["python", "numpy", "torch"])
def test_captured_scalars_stay_hashable_python_scalars(wrap):
    """Static JIT arguments must survive capture as Python scalars, whatever carried them.

    A 0-D NumPy array is unhashable, so `jax.jit` rejects it as a static argument: the
    first cluster run died on `diffeo_start` for exactly this reason, after upstream had
    already finished. The values may arrive as Python numbers, NumPy scalars, or torch
    tensors depending on the notebook.
    """
    converted = convert(
        niter=wrap(1000),
        diffeo_start=wrap(100),
        a=wrap(500),  # upstream notebooks pass this as an `int`
        epV=wrap(1000),
        sigmaB=wrap(0.1),
        tol=wrap(1e-6),
    )

    assert type(converted["niter"]) is int and converted["niter"] == 1000
    assert type(converted["diffeo_start"]) is int and converted["diffeo_start"] == 100
    assert type(converted["a"]) is float and converted["a"] == 500.0
    assert type(converted["epV"]) is float and converted["epV"] == 1000.0
    assert converted["sigmaB"] == pytest.approx(0.1)
    assert type(converted["tol"]) is float
    for name in STATIC_ARGUMENTS:
        if name in converted:
            hash(converted[name])  # what jax.jit does with them


def test_captured_arrays_stay_arrays():
    """Per-channel mixture means and landmarks are traced values, not static ones."""
    converted = convert(
        muA=torch.tensor([1, 1, 1]),
        muB=torch.tensor([0, 0, 0]),
        pointsI=np.arange(8.0).reshape(4, 2),
        pointsJ=np.arange(8.0).reshape(4, 2) + 3.0,
    )

    assert isinstance(converted["muA"], np.ndarray) and converted["muA"].shape == (3,)
    np.testing.assert_array_equal(converted["muB"], np.zeros(3))
    # The port renames the landmark arguments; passing the upstream names would silently
    # drop the point term from the objective.
    assert "pointsI" not in converted and "pointsJ" not in converted
    np.testing.assert_array_equal(converted["points_source"], np.arange(8.0).reshape(4, 2))
    np.testing.assert_array_equal(converted["points_target"], np.arange(8.0).reshape(4, 2) + 3.0)


def test_mixture_means_left_to_be_estimated_are_dropped():
    """`muA=None` means "estimate it"; forwarding `None` explicitly must keep that meaning."""
    converted = convert(niter=10, muA=None, muB=torch.tensor([0.0, 0.0, 0.0]))
    assert "muA" not in converted
    assert "muB" in converted


def test_captured_affine_is_split_into_linear_and_translation():
    """Upstream's warm start may arrive as one 3x3 `A`; the port takes `L` and `T`."""
    affine = np.array([[0.9, -0.2, 12.0], [0.3, 1.1, -4.0], [0.0, 0.0, 1.0]])
    converted = convert(A=torch.as_tensor(affine), L=np.eye(2), T=np.ones(2))

    # `A` wins over `L`/`T`: upstream ignores them when `A` is given (STalign.py:914-919).
    np.testing.assert_allclose(converted["L"], affine[:2, :2])
    np.testing.assert_allclose(converted["T"], affine[:2, 2])
    assert "A" not in converted


def test_captured_linear_and_translation_pass_through():
    linear, translation = np.array([[0.5, 0.1], [-0.1, 0.7]]), np.array([3.0, -2.0])
    converted = convert(L=torch.as_tensor(linear), T=torch.as_tensor(translation))
    np.testing.assert_allclose(converted["L"], linear)
    np.testing.assert_allclose(converted["T"], translation)


def test_identity_is_used_when_no_affine_was_captured():
    converted = convert(niter=5)
    np.testing.assert_array_equal(converted["L"], np.eye(2))
    np.testing.assert_array_equal(converted["T"], np.zeros(2))


def test_captured_warm_start_velocity_is_renamed_with_its_grid():
    """A second upstream fit continues from `v` on `xv`; both have to travel together."""
    velocity = np.zeros((3, 5, 6, 2))
    grid = (np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 6))
    converted = convert(v=torch.as_tensor(velocity), xv=[torch.as_tensor(axis) for axis in grid])

    np.testing.assert_array_equal(converted["initial_velocity"], velocity)
    assert isinstance(converted["velocity_grid"], tuple)
    assert all(isinstance(axis, np.ndarray) for axis in converted["velocity_grid"])
    np.testing.assert_allclose(converted["velocity_grid"][1], grid[1])
    assert "v" not in converted and "xv" not in converted


def test_upstream_only_keywords_are_dropped():
    """`device`/`dtype` are upstream's; the port picks its backend from the JAX config."""
    converted = convert(niter=7, device="cuda:0", dtype=torch.float64)
    assert set(converted) == {"niter", "L", "T"}


def test_an_array_for_a_scalar_argument_is_refused():
    """Better a named error here than a shape error thousands of iterations later."""
    with pytest.raises(TypeError, match="niter"):
        convert(niter=np.array([100, 200]))


def test_the_unreplayable_notebook_is_still_unreplayable_upstream():
    """The claim in `UNREPLAYABLE_NOTEBOOKS` is checked against the vendored data.

    `merfish-visium-alignment-with-curve-annotator.ipynb` is excluded because upstream's own
    two saved curve files disagree on vertex count, so `L_T_from_points` raises. If upstream
    ever ships consistent curves, this fails and the notebook goes back into the comparison
    rather than sitting excluded on a stale excuse.
    """
    (notebook,) = UNREPLAYABLE_NOTEBOOKS
    assert notebook in NOTEBOOKS
    curves = upstream.vendor_root() / "docs" / "visium_data"
    source = np.load(curves / "Merfish_S2_R3_curves.npy", allow_pickle=True).tolist()
    target = np.load(curves / "tissue_hires_image_curves.npy", allow_pickle=True).tolist()
    counts = (sum(len(c) for c in source.values()), sum(len(c) for c in target.values()))
    assert counts == (10, 15), f"upstream curve vertex counts changed to {counts}"
    assert f"({counts[0]})" in UNREPLAYABLE_NOTEBOOKS[notebook]

    payload = json.loads((upstream.vendor_root() / "docs" / "notebooks" / notebook).read_text())
    # Upstream committed the notebook with this exception in its own output.
    recorded = [
        output
        for cell in payload["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert any("is not equal to number of pointsJ" in output["evalue"] for output in recorded)


def test_replay_keeps_a_cells_own_imports(stalign):
    """Dropping upstream's import must not drop the cell's other imports with it.

    Several notebooks put `import pandas as pd` in the same cell as `import STalign`. A
    cell-level skip left the replay without pandas and failed with a bare `NameError` many
    cells later; only upstream's own import line may be removed.
    """
    from squidpy_ports.stalign.notebook_suite import _clean_cell

    cell = "import numpy as np\nimport pandas as pd\nfrom STalign import STalign\nimport STalign\nx = 1\n"
    cleaned = _clean_cell(cell)
    assert "import pandas as pd" in cleaned
    assert "import numpy as np" in cleaned
    assert "STalign" not in cleaned

    # `merfish-merfish-alignment-using-L-T.ipynb` cell 38. A reload that worked would restore
    # the real `LDDMM` mid-notebook, leaving the Squidpy pass comparing upstream to itself.
    reloaded = _clean_cell("import importlib\nimportlib.reload(STalign)\nkeep = 1\n")
    assert "reload" not in reloaded
    assert "keep = 1" in reloaded

    # The replay namespace pre-seeds the patched module, so the dropped import is not needed.
    namespace: dict[str, Any] = {"STalign": stalign}
    exec(compile(cleaned, "<cell>", "exec"), namespace)
    assert namespace["pd"].__name__ == "pandas"
    assert namespace["STalign"] is stalign


def test_replay_directory_resolves_both_upstream_path_depths():
    """Upstream disagrees with itself about how far up its data lives; both must work.

    Fifteen notebooks read `../<name>_data/...`; `merfish-xenium-alignment.ipynb` reads
    `../../merfish_data/...` for files in the same place. The scratch replay directory
    carries the data at both depths so neither convention has to be edited in the vendored
    checkout.
    """
    from squidpy_ports.stalign.notebook_suite import _replay_directory

    notebook_path = upstream.vendor_root() / "docs" / "notebooks" / "merfish-xenium-alignment.ipynb"
    if not notebook_path.is_file():
        pytest.skip("vendored upstream not checked out")
    wanted = "merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz"

    outside = os.getcwd()
    with _replay_directory(notebook_path):
        assert os.path.isfile(f"../{wanted}"), "the convention 15 notebooks use"
        assert os.path.isfile(f"../../{wanted}"), "the convention merfish-xenium-alignment uses"
        # Writes land in scratch, not in the vendored GPL checkout.
        assert upstream.vendor_root().resolve() not in Path(os.getcwd()).resolve().parents
    assert os.getcwd() == outside


def test_fit_result_supports_every_api_the_notebooks_use():
    """All three ways the pinned notebooks read a fit result have to work.

    13 notebooks use `out['A']`, `merfish-merfish-alignment-simulation` uses `out[0]`, and
    `merfish-merfish-alignment-using-L-T` unpacks `A, v, xv = LDDMM(...)`. Upstream returns a
    plain dict, so the last two raise against the pinned commit and the replay would stop
    before the notebook's own plots -- which are the whole point of the comparison.
    """
    from squidpy_ports.stalign.notebook_suite import _FitResult

    result = _FitResult(A=np.eye(3), v=np.zeros((3, 2, 2, 2)), xv=(np.arange(2.0), np.arange(2.0)), WM=np.ones((2, 2)))

    linear, velocity, grid = result  # `A, v, xv = LDDMM(...)`
    np.testing.assert_array_equal(linear, np.eye(3))
    assert velocity.shape == (3, 2, 2, 2)
    assert len(grid) == 2

    np.testing.assert_array_equal(result[0], np.eye(3))  # `out[0]`
    assert result[1].shape == (3, 2, 2, 2)
    assert len(result[2]) == 2

    np.testing.assert_array_equal(result["A"], np.eye(3))  # `out['A']`
    assert set(result.keys()) == {"A", "v", "xv", "WM"}
    assert dict(result)["WM"].shape == (2, 2)


def test_renamed_helpers_resolve_and_are_still_renames(stalign):
    """The old `atlas` spellings must reach the new functions, and only because they are renames.

    `merfish-merfish-alignment-simulation.ipynb` calls `transform_points_atlas_to_target`,
    which upstream renamed in `5837b03` ("change atlas to source"). Aliasing is faithful only
    while the new name is genuinely the same function, so both halves are asserted.
    """
    from squidpy_ports.stalign.notebook_suite import _RENAMED_HELPERS, _UpstreamProxy

    proxy = _UpstreamProxy(stalign)
    for old, new in _RENAMED_HELPERS.items():
        assert not hasattr(stalign, old), f"upstream grew {old} back; the alias now hides it"
        assert hasattr(stalign, new), f"upstream no longer has {new}"
        assert getattr(proxy, old) is getattr(stalign, new)

    # Anything not renamed still resolves straight through, including the patched fit.
    assert proxy.LDDMM is stalign.LDDMM
    assert proxy.rasterize is stalign.rasterize


def test_replay_uses_the_notebooks_own_variables_for_metrics():
    """Metrics are named after what the notebook computed, not after quantities invented here."""
    from squidpy_ports.stalign.notebook_suite import _namespace_metrics

    upstream_ns = {
        "phiI": np.ones((4, 4)),
        "tpointsI": np.arange(8.0).reshape(4, 2),
        "I": np.zeros((4, 4)),  # an input, shared by both passes
        "_hidden": np.ones(9),
        "note": "not an array",
        "small": np.ones(2),
    }
    squidpy_ns = {
        "phiI": np.ones((4, 4)) * 1.01,
        "tpointsI": np.arange(8.0).reshape(4, 2),
        "I": np.ones((4, 4)),
        "_hidden": np.zeros(9),
        "note": "also not an array",
        "small": np.zeros(2),
    }

    metrics = _namespace_metrics(upstream_ns, squidpy_ns)
    assert set(metrics) == {"phiI relative L2", "tpointsI relative L2"}
    assert metrics["phiI relative L2"] == pytest.approx(0.01, rel=1e-6)
    assert metrics["tpointsI relative L2"] == 0.0


def test_replay_fills_omitted_keywords_from_upstream_not_the_image_path():
    """A keyword a notebook omits must reach the port as *upstream's* default.

    The public `align_stalign_image` resolves `_IMAGE_DEFAULTS`, which are squidpy's own
    choices for images and deliberately not upstream's: `a` 20 against 500, `niter` 200
    against 5000, `diffeo_start` 100 against 0, `epV` 1.0 against 2e3. Fourteen of the
    seventeen notebooks pass none of those four, so letting the entry point fill them would
    run the port on a different fit from upstream and publish the gap as a port divergence.
    Every solver keyword therefore goes on the call explicitly.
    """
    pytest.importorskip("squidpy")
    from squidpy.experimental.tl._align import _stalign

    from squidpy_ports.stalign.notebook_suite import _completed_kwargs, solver_keys

    accepted = solver_keys()
    # A notebook that passes only `sigmaM`, as most of them effectively do for these knobs.
    completed = _completed_kwargs({"sigmaM": 3.0}, accepted)

    assert completed["sigmaM"] == 3.0, "an explicitly passed keyword must survive"
    for knob in ("a", "niter", "diffeo_start", "epV"):
        assert completed[knob] == _stalign._SOLVER_DEFAULTS[knob], f"{knob} is not upstream's default"
        if _stalign._IMAGE_DEFAULTS[knob] != _stalign._SOLVER_DEFAULTS[knob]:
            assert completed[knob] != _stalign._IMAGE_DEFAULTS[knob], (
                f"{knob} came from the image path's defaults, not upstream's"
            )


def test_smoke_gate_caps_both_passes_and_keeps_the_velocity_in_play(monkeypatch):
    """The parameter gate must cap `niter` without quietly comparing affines only.

    `diffeo_start` gates the diffeomorphic half: upstream's own 3D defaults start it at 0, but
    the image path's start it at 100. Capping `niter` to 1 while leaving `diffeo_start` at 100
    would run one affine-only step, so the gate would pass on exactly the parameters whose
    mismatch it exists to catch.
    """
    from squidpy_ports.stalign.notebook_suite import _capped, smoke_niter

    monkeypatch.delenv("STALIGN_SMOKE_NITER", raising=False)
    assert smoke_niter() is None
    untouched = {"niter": 5000, "diffeo_start": 100}
    assert _capped(untouched) is untouched, "the gate must be inert when unset"

    monkeypatch.setenv("STALIGN_SMOKE_NITER", "1")
    assert smoke_niter() == 1
    capped = _capped({"niter": 5000, "diffeo_start": 100, "a": 500.0})
    assert capped["niter"] == 1
    assert capped["diffeo_start"] == 0, "a diffeo_start past the cap would test the affine only"
    assert capped["a"] == 500.0, "the gate caps iterations, it does not retune"
    # A diffeo_start already inside the cap is upstream's own and must survive.
    assert _capped({"niter": 5000, "diffeo_start": 0})["diffeo_start"] == 0


def test_axis_placement_reproduces_upstreams_grids():
    """The placement handed to the public API must rebuild upstream's axes, or say it did not.

    `align_stalign_*` reads an element's axes off its scale and translation, so a silent lossy
    round-trip would mean the two passes no longer see identical inputs -- a divergence this
    harness invented. Upstream builds axes both centred (the atlas) and from a corner
    (`rasterize`), so both forms are covered.

    Most grids rebuild bit-for-bit. A minority cannot: `arange(n) * step + offset` has already
    absorbed rounding, and no `(step, shift)` recovers it. Those must be *recorded*, and the
    error must stay far below the ~1e-12 the comparison asserts at.
    """
    from squidpy_ports.stalign.notebook_suite import _AXIS_RESIDUAL, axis_placement

    exact = inexact = 0
    for n in (17, 33, 41, 57, 100, 101, 256, 501):
        for step in (50.0, 30.0, 25.0, 12.5, 1.0, 0.65):
            for axis in (
                np.arange(n) * step - (n - 1) * step / 2.0,  # centred: the atlas grids
                np.arange(n) * step,  # from the origin
            ):
                _AXIS_RESIDUAL.clear()
                got_step, got_shift = axis_placement(axis)
                rebuilt = np.arange(n, dtype=float) * got_step + got_shift
                if np.array_equal(rebuilt, axis):
                    exact += 1
                    assert not _AXIS_RESIDUAL, f"exact rebuild still recorded a residual: {_AXIS_RESIDUAL}"
                else:
                    inexact += 1
                    error = float(np.max(np.abs(rebuilt - axis)))
                    assert _AXIS_RESIDUAL["max"] == pytest.approx(error), "an inexact rebuild went unrecorded"
                    assert error < 1e-12, f"n={n} step={step}: {error:.3e} is too large to absorb"

    # A regression that stops recovering the step at all would leave this near zero.
    assert exact / (exact + inexact) > 0.9, f"only {exact}/{exact + inexact} grids rebuild exactly"


def test_axis_placement_records_what_it_cannot_reproduce():
    """Where the step is unrecoverable the residual is reported, not swallowed.

    `arange(n) * step + offset` loses the low bits of `step` once `offset` dominates it, and
    no `(step, shift)` pair rebuilds the stored values exactly. That is a real limit of the
    public API's placement contract, so it has to surface as a number.
    """
    from squidpy_ports.stalign.notebook_suite import _AXIS_RESIDUAL, axis_placement

    _AXIS_RESIDUAL.clear()
    axis = np.arange(17) * 0.65 - 1234.5
    step, shift = axis_placement(axis)
    rebuilt = np.arange(17, dtype=float) * step + shift

    assert not np.array_equal(rebuilt, axis)  # the premise: this one cannot be exact
    assert _AXIS_RESIDUAL["max"] == pytest.approx(np.max(np.abs(rebuilt - axis)))
    assert _AXIS_RESIDUAL["max"] < 1e-12  # small, but reported rather than absorbed


def test_replay_scores_categorical_columns_relative_l2_cannot_see():
    """A region assignment is a string, so only a label metric can quantify the two legends.

    Mirrors the volume-to-section notebooks: ``df`` holds numeric coordinates beside the
    ``acronym`` a boundary cell can land either side of.
    """
    import pandas as pd

    from squidpy_ports.stalign.notebook_suite import _namespace_metrics

    def frame(acronyms, shift=0.0):
        return pd.DataFrame(
            {
                "coord0": np.arange(4.0) + shift,  # the warped atlas coordinate, microns
                "x": np.arange(4.0),  # a shared input: identical on both sides
                "struct_id": np.arange(4),  # the acronym as an id, not a distance
                "acronym": acronyms,
            }
        )

    upstream_ns = {"df": frame(["VISp4", "VISp5", "CA1", None])}
    squidpy_ns = {"df": frame(["VISp4", "VISp5", "DG-mo", None], shift=0.5)}

    metrics = _namespace_metrics(upstream_ns, squidpy_ns)

    # One of four rows reassigned; the pair that is null on both sides is not a disagreement.
    assert metrics["df[acronym] label disagreement"] == pytest.approx(0.25)
    # The *set* difference distinguishes boundary jitter from bulk displacement: here CA1
    # lost its only cell and DG-mo gained one, so one label leaves and one arrives.
    assert metrics["df[acronym] labels only upstream"] == 1.0  # CA1
    assert metrics["df[acronym] labels only squidpy"] == 1.0  # DG-mo
    assert metrics["df[acronym] label set jaccard"] == pytest.approx(2 / 4)  # {VISp4,VISp5} of 4
    # The displacement that explains it, in the column's own units.
    assert metrics["df[coord0] median abs delta"] == pytest.approx(0.5)
    # A column both passes share is the control: it has to read exactly zero.
    assert metrics["df[x] median abs delta"] == 0.0
    # An id column is neither a distance nor a second copy of the acronym.
    assert not [key for key in metrics if "struct_id" in key]
    # A mixed frame is still not scored whole: as an array it is object dtype, as before.
    assert "df relative L2" not in metrics
    # An all-numeric frame keeps its whole-frame relative L2 and gains the per-column delta,
    # which is the absolute-scale companion to a norm that is scale-free by construction.
    numeric = {"df1": pd.DataFrame({"a": np.arange(4.0)})}
    assert set(_namespace_metrics(numeric, numeric)) == {
        "df1 relative L2",
        "df1[a] median abs delta",
    }


def test_index_maps_array_task_ids_onto_notebooks():
    """A Slurm array task knows only its id, so this mapping is what selects the work."""
    assert notebook_for_index(0) == NOTEBOOKS[0]
    assert notebook_for_index(len(NOTEBOOKS) - 1) == NOTEBOOKS[-1]
    for bad in (-1, len(NOTEBOOKS)):
        with pytest.raises(IndexError):
            notebook_for_index(bad)


def test_cli_index_runs_one_notebook_and_writes_its_status(tmp_path):
    """The array's entry point, end to end.

    A broken `--index` would waste a whole array submission, and the batch script has no
    other way in. Index 8 is the unreplayable notebook: it emits a status panel without
    fitting anything, so this stays cheap while still going through argparse,
    `compare_notebook`, `write_result`, and the per-task status file.

    It used to use index 2, back when the 3D notebooks were also status panels. They now run
    a real comparison -- which downloads the Allen atlas and fits for 2000 iterations, so
    pointing this at one turns a one-second test into an hour-long one.
    """
    index = 8
    notebook = notebook_for_index(index)
    assert notebook in UNREPLAYABLE_NOTEBOOKS  # keeps this test cheap; fails loudly if reordered

    environment = {**os.environ, "MPLBACKEND": "agg", "SQUIDPY_PORTS_STALIGN_SHA": upstream.UPSTREAM_SHA}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "squidpy_ports.stalign.notebook_suite",
            "--index",
            str(index),
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    stem = notebook.removesuffix(".ipynb")
    status = json.loads((tmp_path / f"{stem}-status.json").read_text())
    # Named after the notebook, not after the suite: array tasks share one output dir.
    assert status["statuses"] == {notebook: "unreplayable-upstream"}
    assert status["failures"] == {}
    assert (tmp_path / f"{stem}-comparison.png").exists()
    assert (tmp_path / "notebooks" / notebook).exists()


def test_cli_rejects_selecting_nothing():
    completed = subprocess.run(
        [sys.executable, "-m", "squidpy_ports.stalign.notebook_suite"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "one of the arguments" in completed.stderr


def test_written_evidence_keeps_the_notebook_lighter_than_the_archive(tmp_path):
    """The committed notebook carries a web-resolution panel; the PNG beside it stays archival.

    These notebooks are committed and rendered by the docs, so the embedded copy is what
    drives repository size. At `ARCHIVE_DPI` a full suite added ~32 MB per rerun.
    """
    import base64
    import io

    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from squidpy_ports.stalign.notebook_suite import ARCHIVE_DPI, EMBED_DPI, ComparisonResult, write_result

    figure, ax = plt.subplots(figsize=(20, 9.5))
    rng = np.random.default_rng(0)
    ax.imshow(rng.random((400, 900)), cmap="magma")  # noisy, so PNG cannot trivially compress it
    result = ComparisonResult("heart-alignment.ipynb", "compared", {"phiI relative L2": 1.5e-05}, [figure])

    write_result(result, tmp_path)
    plt.close(figure)

    archive = (tmp_path / "heart-alignment-comparison.png").stat().st_size
    payload = json.loads((tmp_path / "notebooks" / "heart-alignment.ipynb").read_text())
    embedded = base64.b64decode(payload["cells"][2]["outputs"][0]["data"]["image/png"])
    assert len(embedded) < archive, f"embedded {len(embedded)} should undercut archival {archive}"

    # Both are cropped by `bbox_inches="tight"`, so compare their ratio, not absolute widths.
    panel = Image.open(io.BytesIO(embedded))
    archival = Image.open(tmp_path / "heart-alignment-comparison.png")
    assert panel.width == pytest.approx(archival.width * EMBED_DPI / ARCHIVE_DPI, rel=0.05)
    assert json.loads((tmp_path / "heart-alignment-metrics.json").read_text()) == result.metrics


def test_port_figures_are_written_out_byte_for_byte(tmp_path):
    """The docs pair the port's plot with upstream's *published* figure, not with our replay.

    So the port half has to land on disk as its own PNG. Cropping it back out of the composed
    pair is a manual step that rots the moment a plot's aspect changes.
    """
    from squidpy_ports.stalign.notebook_suite import ComparisonResult, write_result

    port_pngs = [b"\x89PNG\r\n\x1a\n-first", b"\x89PNG\r\n\x1a\n-second"]
    result = ComparisonResult("heart-alignment.ipynb", "compared", {}, [], port_figures=port_pngs)
    write_result(result, tmp_path)

    assert (tmp_path / "heart-alignment-port.png").read_bytes() == port_pngs[0]
    assert (tmp_path / "heart-alignment-port-1.png").read_bytes() == port_pngs[1]


def test_failed_comparison_writes_a_traceback_and_notebook(tmp_path):
    """A partial cluster run has to explain its failures without the suite log."""
    notebook = NOTEBOOKS[0]
    try:
        raise ValueError("Non-hashable static arguments are not supported.")
    except ValueError as error:
        write_failure(notebook, error, tmp_path)

    stem = notebook.removesuffix(".ipynb")
    trace = (tmp_path / f"{stem}-traceback.txt").read_text()
    assert "ValueError: Non-hashable static arguments" in trace
    assert "test_failed_comparison_writes_a_traceback_and_notebook" in trace  # a real frame, not the message

    manifest = json.loads((tmp_path / f"{stem}-manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["upstream_sha"] == upstream.UPSTREAM_SHA

    payload = json.loads((tmp_path / "notebooks" / notebook).read_text())
    (output,) = payload["cells"][1]["outputs"]
    assert output["output_type"] == "error"
    assert output["ename"] == "ValueError"
    assert output["traceback"], "an error output with no traceback explains nothing"
    assert payload["metadata"]["stalign_comparison"]["status"] == "failed"


@pytest.mark.parametrize("name", ["THETA", "SHIFT"])
def test_fixture_transform_is_not_round(name):
    """A round angle or integer shift puts samples exactly on grid lines.

    There, upstream's and squidpy's index formulas — equal to ~1 ulp — can floor() to
    different neighbours, producing an O(1) disagreement that says nothing about the port.
    """
    value = np.atleast_1d(getattr(F, name)).astype(float)
    assert np.all(np.abs(value - np.round(value, 2)) > 1e-6)


def test_report_folds_parametrised_cases_and_separates_xfail_from_skip(tmp_path):
    """A generated page must not report `passed` for a function that also xfailed.

    JUnit writes an xfail as `<skipped type="pytest.xfail">`, so the two are one tag apart, and
    conflating them would turn a pinned deliberate divergence into "this did not run".
    """
    from squidpy_ports.stalign.test_report import collect, render

    source = tmp_path / "test_thing.py"
    source.write_text(
        'def test_many():\n    """Runs three ways."""\n\n\ndef test_mixed():\n    """Passes once, xfails once."""\n'
        '\n\ndef test_gone():\n    """Never ran."""\n'
    )
    (tmp_path / "r.xml").write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="t" name="test_many[a]"/>'
        '<testcase classname="t" name="test_many[b]"/>'
        '<testcase classname="t" name="test_many[c]"/>'
        '<testcase classname="t" name="test_mixed[x]"/>'
        '<testcase classname="t" name="test_mixed[y]">'
        '<skipped type="pytest.xfail" message="ledger row D6"/></testcase>'
        "</testsuite></testsuites>"
    )

    functions = {f.name: f for f in collect(tmp_path / "r.xml", source)}
    assert "test_gone" not in functions, "a function with no reported cases must not appear"
    assert functions["test_many"].cases == 3, "parametrised cases fold into one row"
    assert functions["test_many"].status == "passed"
    # The whole point: one xfail out of two cases must not read as a pass.
    assert functions["test_mixed"].status == "xfailed"
    assert functions["test_mixed"].reasons == ["ledger row D6"]

    page = render([("demo", source, list(functions.values()))])
    assert "ledger row D6" in page, "the reason has to survive into the page"
    assert "Runs three ways." in page, "descriptions come from the docstring, not a paraphrase"


# --------------------------------------------------------------------------------------
# The rank-3 (volume-to-section) comparison's conversion helpers
# --------------------------------------------------------------------------------------


def test_initial_affine_is_reversed_into_xy_z_order():
    """Upstream's `A` is `(z, y, x)`; `fit_stalign_volume` takes `(x, y, z)`.

    Reversing the *spatial* axes only is a permutation of the first three rows and columns.
    A plain `[::-1, ::-1]` would move the homogeneous row too and silently produce a
    different transform -- so this checks the translation lands where it belongs rather
    than just that the shape is (4, 4).
    """
    affine_zyx = np.array(
        [
            [1.0, 0.0, 0.0, 7.0],
            [0.0, 2.0, 0.0, 8.0],
            [0.0, 0.0, 3.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    got = _initial_affine_xyz({"A": torch.as_tensor(affine_zyx)})

    np.testing.assert_allclose(got[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(got[:3, 3], [9.0, 8.0, 7.0])  # (z, y, x) -> (x, y, z)
    np.testing.assert_allclose(np.diag(got)[:3], [3.0, 2.0, 1.0])
    # Its own inverse: reversing twice is the identity.
    swap = np.eye(4)[[2, 1, 0, 3]]
    np.testing.assert_allclose(swap @ got @ swap, affine_zyx)


def test_initial_affine_accepts_the_l_t_pair_the_notebooks_use():
    """`merfish-allen3Datlas-alignment` cell [30] builds `L` and `T`, never a whole `A`."""
    linear = np.diag([1.0, 2.0, 3.0])
    translation = np.array([7.0, 8.0, 9.0])
    from_pair = _initial_affine_xyz({"L": linear, "T": translation})
    from_affine = _initial_affine_xyz({"A": np.block([[linear, translation[:, None]], [np.zeros((1, 3)), 1.0]])})

    np.testing.assert_allclose(from_pair, from_affine)


def test_initial_affine_is_none_when_the_notebook_specifies_nothing():
    """Upstream defaults both to `None`, meaning identity -- which the port also defaults to."""
    assert _initial_affine_xyz({"niter": 10}) is None


def test_initial_affine_fills_in_the_half_the_notebook_omits():
    """`L` without `T` (or the reverse) is legal upstream; the missing half is the identity."""
    only_linear = _initial_affine_xyz({"L": np.diag([1.0, 2.0, 3.0])})
    np.testing.assert_allclose(only_linear[:3, 3], 0.0)

    only_translation = _initial_affine_xyz({"T": np.array([7.0, 8.0, 9.0])})
    np.testing.assert_allclose(only_translation[:3, :3], np.eye(3))
    np.testing.assert_allclose(only_translation[:3, 3], [9.0, 8.0, 7.0])


def test_slice_fit_result_exposes_what_the_3d_notebooks_read():
    """Cell [33] unpacks `A`, `v`, `xv` and `Xs`; the last is not stored on the result.

    `Xs` is recomputed from the fitted deformation. Checked against the section grid it is
    supposed to describe: with an identity affine and no velocity, the backward map is the
    lifted section grid itself, so every sampled `z` is 0 and the `(y, x)` corners are the
    axes' own endpoints.
    """
    pytest.importorskip("jax")
    pytest.importorskip("squidpy")
    import jax.numpy as jnp
    from squidpy.experimental.tl import Stalign3DResult

    from squidpy_ports.stalign.notebook_suite import _torch_slice_fit

    section_axes = [np.linspace(-4.0, 4.0, 5), np.linspace(-6.0, 6.0, 7)]
    grid = [np.linspace(-8.0, 8.0, 3)] * 3
    result = Stalign3DResult(
        affine=jnp.eye(4),
        velocity=jnp.zeros((1, 3, 3, 3, 3)),
        velocity_grid=tuple(jnp.asarray(axis) for axis in grid),
        ref_axes=tuple(jnp.asarray(axis) for axis in grid),
        query_axes=tuple(jnp.asarray(axis) for axis in section_axes),
        match_weights=jnp.zeros((1, 5, 7)),
        artifact_weights=jnp.zeros((1, 5, 7)),
        background_weights=jnp.zeros((1, 5, 7)),
    )
    fit = _torch_slice_fit(result, section_axes)

    # `.keys()`, not `set(fit)`: `_FitResult.__iter__` yields the positional tuple form.
    assert set(fit.keys()) >= {"A", "v", "xv", "WM", "WA", "WB", "Xs"}
    A, v, xv = fit  # the tuple-unpacking form `merfish-merfish-alignment-using-L-T` uses
    assert A.shape == (4, 4) and v.shape == (1, 3, 3, 3, 3) and len(xv) == 3

    # Upstream stores `Xs` component-*last*; squidpy's grid is component-first.
    Xs = fit["Xs"].numpy()
    assert Xs.shape == (1, 5, 7, 3)
    np.testing.assert_allclose(Xs[..., 0], 0.0, atol=1e-12)  # the z=0 plane
    np.testing.assert_allclose(Xs[0, :, 0, 1], section_axes[0], atol=1e-12)
    np.testing.assert_allclose(Xs[0, 0, :, 2], section_axes[1], atol=1e-12)


def test_atlas_cache_downloads_once_and_survives_a_fresh_cwd(tmp_path, monkeypatch):
    """The atlas is fetched once, then served from the cache for every later pass.

    This is the plumbing the 3D notebooks depend on: upstream re-GETs unconditionally and
    each replay pass gets its own working directory, so without the cache one comparison
    pulls the atlas four times. Two passes are simulated here by calling through twice.
    """
    from squidpy_ports.stalign.notebook_suite import ATLAS_CACHE_ENV, _AtlasCache, _patch_atlas_downloads

    url = "http://download.alleninstitute.org/informatics-archive/x/ara_nissl_50.nrrd"
    payload = b"nrrd-bytes" * 512
    calls = []

    class _Response:
        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int = 1024):
            yield payload

    class _Requests:
        def get(self, target: str, *args: Any, **kwargs: Any) -> Any:
            calls.append(target)
            assert kwargs["stream"] is True  # streamed, not slurped into memory
            return _Response()

        marker = "passthrough"

    cache = _AtlasCache(_Requests(), tmp_path)
    first, second = cache.get(url), cache.get(url)

    assert calls == [url], "the second pass re-downloaded instead of using the cache"
    assert first.content == second.content == payload
    assert b"".join(second.iter_content(64)) == payload
    assert cache.marker == "passthrough"  # anything but `get` still reaches requests
    assert not list(tmp_path.glob("*.partial")), "a staging file was left behind"

    # Unset means untouched: a plain local run keeps upstream's own download behaviour.
    module = type(sys)("stalign-stub")
    module.requests = sentinel = object()
    monkeypatch.delenv(ATLAS_CACHE_ENV, raising=False)
    _patch_atlas_downloads(module)
    assert module.requests is sentinel

    monkeypatch.setenv(ATLAS_CACHE_ENV, str(tmp_path / "made-on-demand"))
    _patch_atlas_downloads(module)
    assert isinstance(module.requests, _AtlasCache)
    _patch_atlas_downloads(module)  # idempotent: no cache wrapping a cache
    assert module.requests._requests is sentinel


def test_only_absolute_author_paths_are_rewritten():
    """`starmap-allen3Datlas` reads its input from the author's home directory.

    No symlink can rescue `/home/manju`, so the replay rewrites it -- but the relative
    conventions the other notebooks use must survive untouched, including
    `merfish-xenium`'s two-level one.
    """
    from squidpy_ports.stalign.notebook_suite import _clean_cell

    assert (
        _clean_cell('df = pd.read_csv(r"/home/manju/Documents/STalign_build/docs/starmap_data/well11.csv.gz")')
        == 'df = pd.read_csv(r"../starmap_data/well11.csv.gz")'
    )
    for untouched in (
        "df = pd.read_csv('../merfish_data/s1r1_metadata.csv.gz')",
        "df = pd.read_csv('../../merfish_data/s1r1_metadata.csv.gz')",
    ):
        assert _clean_cell(untouched) == untouched

    # The rewrite is exactly one notebook's problem: assert nothing else in the pinned set
    # carries an absolute path, so this stays a targeted fix rather than a growing shim.
    import json

    from squidpy_ports.stalign import upstream

    culprits = {
        path.name
        for path in (upstream.vendor_root() / "docs" / "notebooks").glob("*.ipynb")
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
        if re.search(r"""['"]r?/(?!\*)""", line)
    }
    assert culprits == {"starmap-allen3Datlas-alignment.ipynb"}, culprits


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
    import json
    import re

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


def test_region_colours_are_keyed_to_the_region_not_its_position():
    """A region keeps its colour when the other pass assigns a different region set.

    Upstream colours by index into `np.unique(df['acronym'])`, so one extra region recolours
    every alphabetically later one and the pair looks wholly different for a reason that has
    nothing to do with the fit. The first pass still reproduces upstream's own assignment, so
    the panel stays comparable with upstream's published figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from squidpy_ports.stalign.notebook_suite import _REGION_COLOURS, _patch_region_colours

    def frame(regions):
        return pd.DataFrame({"acronym": regions, "x": range(len(regions)), "y": range(len(regions))})

    module = type(sys)("stalign-stub")
    _REGION_COLOURS.clear()
    _patch_region_colours(module)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # pass 1 (upstream): reproduces upstream's own position-indexed assignment
    module.plot_brain_regions(frame(["CA1", "CA3", "DG-sg", "VISp4"]))
    assert [_REGION_COLOURS[r] for r in ("CA1", "CA3", "DG-sg", "VISp4")] == cycle[:4]

    # pass 2 (squidpy): "CA2" lands between CA1 and CA3, which under upstream's scheme would
    # have shifted CA3, DG-sg and VISp4 each one colour along
    module.plot_brain_regions(frame(["CA1", "CA2", "CA3", "DG-sg", "VISp4"]))
    assert [_REGION_COLOURS[r] for r in ("CA1", "CA3", "DG-sg", "VISp4")] == cycle[:4]
    assert _REGION_COLOURS["CA2"] == cycle[4], "a squidpy-only region must get an unused colour"
    plt.close("all")


def test_published_results_are_matched_only_where_upstream_shipped_this_commit_output():
    """The published-result anchor resolves for exactly the notebooks it can be trusted for.

    Upstream ships the output of its own runs, which is a stronger reference than either
    replay pass. But three shipped files are headerless `%.18e` dumps with no
    `aligned_x`/`aligned_y` -- they predate the `to_csv` call the pinned notebook makes, so
    they are not this commit's output and must not be compared against.
    """
    import pandas as pd

    from squidpy_ports.stalign.notebook_suite import NOTEBOOKS, _published_metrics, _published_result

    resolved = {nb: _published_result(nb) for nb in NOTEBOOKS}
    comparable = {
        nb
        for nb, path in resolved.items()
        if path is not None and {"aligned_x", "aligned_y"} <= set(pd.read_csv(path, nrows=1).columns)
    }
    excluded = {nb for nb, path in resolved.items() if path is not None} - comparable

    assert len(comparable) == 8, sorted(comparable)
    assert excluded == {
        "xenium-heimage-alignment.ipynb",
        "xenium-starmap-alignment.ipynb",
        "xenium-xenium-alignment.ipynb",
    }, sorted(excluded)
    # `heart-alignment` writes a result file upstream never committed.
    assert resolved["heart-alignment.ipynb"] is None

    # the comparison itself: exact agreement reads zero, and a row-count mismatch reports
    # nothing rather than a number computed against different cells
    notebook = "visium-visium-alignment-affine-only.ipynb"
    published = pd.read_csv(_published_result(notebook))
    exact = published[["aligned_x", "aligned_y"]].copy()
    shifted = exact.copy()
    shifted["aligned_x"] += 1.0

    got = _published_metrics(notebook, {"results": exact}, {"results": shifted})
    assert got["upstream vs upstream published relative L2"] == pytest.approx(0.0, abs=1e-12)
    assert got["squidpy vs upstream published relative L2"] > 0

    truncated = {"results": exact.iloc[:10]}
    assert "upstream vs upstream published relative L2" not in _published_metrics(notebook, truncated, {})


@pytest.mark.parametrize(
    ("defaults", "entry"),
    [("_SOLVER_DEFAULTS", "LDDMM"), ("_VOLUME_DEFAULTS", "LDDMM_3D_to_slice")],
    ids=["rank-2", "rank-3"],
)
def test_solver_defaults_match_upstream(defaults, entry):
    """squidpy's own solver defaults are upstream's, so an omitted keyword is not a divergence.

    `align_stalign_volume` fills every keyword a notebook did not pass, because
    the port's solver is a bare kernel declaring none. Upstream fills the same omissions from
    its signature. Both ranks are checked: the two allen3d notebooks pass none of the five
    knobs where rank 3 differs from rank 2, so a drift there would reach the published `v` as
    a fake D11 divergence rather than as an error.
    """
    import inspect as _inspect

    pytest.importorskip("squidpy")
    # Private by necessity: the port's default *values* have no public accessor, and this
    # is the check that a notebook passing nothing puts identical numbers on both sides.
    from squidpy.experimental.tl._align import _stalign

    from squidpy_ports.stalign import upstream

    port = getattr(_stalign, defaults)
    signature = _inspect.signature(getattr(upstream.load(), entry)).parameters
    theirs = {n: p.default for n, p in signature.items() if p.default is not _inspect.Parameter.empty}
    shared = {name: value for name, value in port.items() if name in theirs}
    assert shared, f"{defaults} shares no keyword with {entry} -- nothing is being checked"
    drifted = {n: (v, theirs[n]) for n, v in shared.items() if v != theirs[n]}
    assert not drifted, f"{defaults} drifted from upstream {entry}: {drifted}"


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


def test_results_table_header_matches_its_rows():
    """The published table's column titles have to describe the cells under them.

    The header and the row are built by two separate f-strings, so they can disagree
    silently -- and did: `Test functions` sat above the prose label and `Cases` above the
    function count, in a table `docs/correctness.md` includes verbatim.
    """
    from squidpy_ports.stalign.test_report import TestFunction, render

    functions = [
        TestFunction(name="test_one", doc="first", outcomes=Counter({"passed": 2})),
        TestFunction(name="test_two", doc="second", outcomes=Counter({"passed": 1})),
    ]
    table = render([("what it covers", Path("test_demo.py"), functions)])
    # the first three are the per-suite summary; `render` also emits a per-test detail table
    header, _, row = [line for line in table.splitlines() if line.startswith("|")][:3]
    columns = [c.strip() for c in header.strip("|").split("|")]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert len(columns) == len(cells)

    by_column = dict(zip(columns, cells, strict=True))
    assert by_column["Suite"] == "`test_demo.py`"
    assert by_column["What it covers"] == "what it covers"
    assert by_column["Test functions"] == "2", "the function count must sit under its own title"
    assert "3 cases" in by_column["Result"], "the case total belongs in Result, not in a count column"


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


def test_numpy_shim_only_touches_device_tensors_and_restores_itself():
    """Upstream's `.numpy()` calls must survive the replay's device override, and only that.

    The override is what makes those calls fail; the shim undoes the artifact rather than
    changing upstream's behaviour. A CPU tensor must go through the untouched path, and
    `torch.Tensor.numpy` must be exactly what it was once the replay ends.
    """
    import torch

    from squidpy_ports.stalign.notebook_suite import _tensors_convertible_to_numpy

    before = torch.Tensor.numpy
    with _tensors_convertible_to_numpy():
        assert torch.Tensor.numpy is not before, "the shim did not install"
        np.testing.assert_allclose(torch.tensor([1.0, 2.0]).numpy(), [1.0, 2.0])
        # a tensor needing grad still refuses without an explicit detach on CPU, exactly as
        # upstream would see it -- the shim must not paper over unrelated errors
        with pytest.raises(RuntimeError):
            torch.tensor([1.0], requires_grad=True).numpy()
    assert torch.Tensor.numpy is before, "the shim outlived the replay"
