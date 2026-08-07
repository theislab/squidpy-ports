"""Guards on the reference itself.

These do not check STalign's maths — they check that the thing we call "the reference"
is still the pinned upstream, and that the spies in :mod:`squidpy_ports.stalign.upstream`
still line up with the loop they observe. If either drifts, every tolerance in squidpy's
test suite quietly becomes meaningless.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from squidpy_ports.stalign import fixtures as F
from squidpy_ports.stalign import upstream
from squidpy_ports.stalign.notebook_suite import (
    NOTEBOOKS,
    THREE_D_NOTEBOOKS,
    UNREPLAYABLE_NOTEBOOKS,
    _convert_kwargs,
    jax_parameters,
    notebook_for_index,
    write_failure,
    write_notebook_wrappers,
)

#: Names and default *types* of squidpy's ``lddmm``, mirrored so the argument conversion
#: is testable in an environment without squidpy and JAX (neither is installed here or in
#: CI; both only exist inside the GPU comparison job). `test_port_signature_matches_the_mirror`
#: fails wherever they *are* installed if the port's signature drifts from this.
PORT_DEFAULTS: dict[str, Any] = {
    "xI": inspect.Parameter.empty,
    "I": inspect.Parameter.empty,
    "xJ": inspect.Parameter.empty,
    "J": inspect.Parameter.empty,
    "L": inspect.Parameter.empty,
    "T": inspect.Parameter.empty,
    "initial_velocity": None,
    "velocity_grid": None,
    "points_source": None,
    "points_target": None,
    "a": 500.0,
    "p": 2.0,
    "expand": 2.0,
    "nt": 3,
    "niter": 5000,
    "diffeo_start": 0,
    "epL": 2e-8,
    "epT": 2e-1,
    "epV": 2e3,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 5e5,
    "sigmaP": 2e1,
    "muA": None,
    "muB": None,
    "tol": None,
    "patience": 25,
}
PORT_PARAMETERS = {
    name: inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
    for name, default in PORT_DEFAULTS.items()
}

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


def test_notebook_wrappers_are_one_to_one(tmp_path):
    write_notebook_wrappers(tmp_path)
    assert {path.name for path in tmp_path.glob("*.ipynb")} == set(NOTEBOOKS)
    for path in tmp_path.glob("*.ipynb"):
        payload = json.loads(path.read_text())
        # nbformat 4.5 requires cell ids; without them the docs build warns, and `docs:build`
        # runs sphinx with `-W`.
        assert all(cell.get("id") for cell in payload["cells"]), path.name


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
    """Where squidpy is installed, `PORT_DEFAULTS` must still describe its real signature.

    The conversion casts each captured argument using the port's declared default, so a
    renamed parameter or an `int`-turned-`float` would change what the suite sends. This
    repo cannot install squidpy (it is the thing under comparison, pulled in only inside
    the GPU job), so the mirror above stands in -- and is checked here whenever it can be.
    """
    pytest.importorskip("jax")
    pytest.importorskip("squidpy")

    real = jax_parameters()
    assert set(real) == set(PORT_DEFAULTS)
    assert {name: type(p.default) for name, p in real.items()} == {
        name: type(default) for name, default in PORT_DEFAULTS.items()
    }


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
    other way in. Index 2 is a 3D notebook: it emits a status panel without fitting
    anything, so this stays cheap while still going through argparse, `compare_notebook`,
    `write_result`, and the per-task status file.
    """
    notebook = notebook_for_index(2)
    assert notebook in THREE_D_NOTEBOOKS  # keeps this test cheap; fails loudly if reordered

    environment = {**os.environ, "MPLBACKEND": "agg", "SQUIDPY_PORTS_STALIGN_SHA": upstream.UPSTREAM_SHA}
    completed = subprocess.run(
        [sys.executable, "-m", "squidpy_ports.stalign.notebook_suite", "--index", "2", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    stem = notebook.removesuffix(".ipynb")
    status = json.loads((tmp_path / f"{stem}-status.json").read_text())
    # Named after the notebook, not after the suite: array tasks share one output dir.
    assert status["statuses"] == {notebook: "unsupported-3d"}
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


#: Parity replays regenerated at squidpy fork `6a63ff8` whose `-metrics.json` was left behind at
#: the earlier job-38960447 run, so notebook and JSON describe different fork commits and disagree
#: from about the 10th significant digit. Listed rather than tolerated: a loosened tolerance would
#: also swallow real drift. Re-running the parity replay for these two clears the list.
REGENERATED_WITHOUT_METRICS = frozenset({"merfish-merfish-alignment", "xenium-xenium-alignment"})


def test_committed_notebooks_report_their_recorded_metrics():
    """Every committed comparison notebook must still carry the numbers its run produced.

    These notebooks are evidence, and JSON formatters re-emit float literals from their own
    parse: biome's pre-commit hook silently moved `1.1311371582589875e-06` to `...77e-06`
    (one ULP) while reformatting them, which is precisely the kind of quiet drift this
    repository exists to make impossible. Notebooks are excluded from that hook now; this
    fails if anything reintroduces it.
    """
    root = Path(__file__).resolve().parents[1] / "docs"
    docs = root / "notebooks" / "stalign-upstream"
    results = root / "parity"
    # Not `pytest.skip`: this pointed at `results-38957316` for months -- a directory that was
    # never committed -- so the guard silently passed as a skip and checked nothing at all. If the
    # metrics move again, that must fail loudly rather than quietly stop covering anything.
    assert results.is_dir(), f"committed parity metrics missing at {results}"

    checked = 0
    stale: list[str] = []
    for metrics_path in sorted(results.glob("*-metrics.json")):
        recorded = json.loads(metrics_path.read_text())
        if not recorded:
            continue  # status-panel notebooks carry no metrics
        if metrics_path.name.removesuffix("-metrics.json") in REGENERATED_WITHOUT_METRICS:
            stale.append(metrics_path.name)
            continue
        notebook = docs / (metrics_path.name.removesuffix("-metrics.json") + ".ipynb")
        payload = json.loads(notebook.read_text())
        embedded = [
            output["data"]["application/json"]
            for cell in payload["cells"]
            for output in cell.get("outputs", [])
            if "application/json" in output.get("data", {})
        ]
        assert embedded, f"{notebook.name} carries no metrics output"
        assert embedded[0] == recorded, f"{notebook.name} disagrees with {metrics_path.name}"
        checked += 1
    assert checked >= 12, f"expected the compared notebooks to be checked, only saw {checked}"


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
