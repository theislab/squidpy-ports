"""Replay every upstream STalign notebook and compare its fit with Squidpy."""

from __future__ import annotations

import argparse
import base64
import inspect
import io
import json
import os
import platform
import re
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from . import upstream

NOTEBOOKS = (
    "heart-alignment-varying-thickness.ipynb",
    "heart-alignment.ipynb",
    "merfish-allen3Datlas-alignment.ipynb",
    "merfish-merfish-alignment-affine-only-with-points.ipynb",
    "merfish-merfish-alignment-affine-only.ipynb",
    "merfish-merfish-alignment-simulation.ipynb",
    "merfish-merfish-alignment-using-L-T.ipynb",
    "merfish-merfish-alignment.ipynb",
    "merfish-visium-alignment-with-curve-annotator.ipynb",
    "merfish-visium-alignment-with-point-annotator.ipynb",
    "merfish-visium-alignment.ipynb",
    "merfish-xenium-alignment.ipynb",
    "starmap-allen3Datlas-alignment.ipynb",
    "visium-visium-alignment-affine-only.ipynb",
    "xenium-heimage-alignment.ipynb",
    "xenium-starmap-alignment.ipynb",
    "xenium-xenium-alignment.ipynb",
)

THREE_D_NOTEBOOKS = {
    "merfish-allen3Datlas-alignment.ipynb",
    "starmap-allen3Datlas-alignment.ipynb",
}
AFFINE_NOTEBOOK = "merfish-merfish-alignment-affine-only-with-points.ipynb"

#: Resolution of the standalone PNG panel: archival, kept with the run's evidence.
ARCHIVE_DPI = 180
#: Resolution of the copy embedded in the executed notebook. Lower on purpose -- these
#: notebooks are committed and rendered by the documentation, and a 20x9.5in panel at
#: `ARCHIVE_DPI` base64s to ~2 MB, so a full suite would add ~32 MB to the repository on
#: every rerun. The numbers live in the metrics JSON; the embedded panel only has to be
#: legible on a docs page.
EMBED_DPI = 100

#: Notebooks that do not run at the pinned upstream commit, with the evidence that this is
#: upstream's state rather than something this replay breaks. Reproducing an upstream
#: failure is the honest outcome; inventing a comparison around it would not be.
UNREPLAYABLE_NOTEBOOKS = {
    "merfish-visium-alignment-with-curve-annotator.ipynb": (
        "Not compared: this notebook does not run at the pinned upstream commit. Its two "
        "saved curve files hold 10 and 15 vertices, so the `L_T_from_points` call raises "
        "`Number of pointsI (10) is not equal to number of pointsJ (15)` -- and upstream's "
        "own committed output for that cell records the same exception. The replay "
        "reproduces an upstream defect; there is no fit to compare."
    ),
}
#: The divergence both volume-to-section notebooks carry; see row D11.
_THREE_D_NOTE = (
    "This notebook fits a 3D volume to a 2D section. The two sides are *expected* to differ "
    "numerically here, unlike every 2D notebook above: upstream's 3D regularisation energy "
    "transforms two of the three spatial axes (`dim=(1,2)`, STalign.py:1504) while smoothing "
    "that same energy's gradient over all three (`dim=(1,2,3)`, :1527), so it descends on a "
    "different objective than the one it reports. Squidpy uses every spatial axis in both "
    "places. The divergence below measures that deliberate choice -- see "
    "`docs/STALIGN_DIVERGENCES.md` row D11 -- and is not a port defect."
)

UPSTREAM_NOTES = {
    "merfish-merfish-alignment-using-L-T.ipynb": (
        "The notebook's second fit uses the stale `A, v, xv = LDDMM(...)` tuple API, which "
        "upstream no longer returns. The replay supplies a result that unpacks both ways, so "
        "both of its fits are compared."
    ),
    "xenium-xenium-alignment.ipynb": (
        "The two sections overlap only partially. Unmatched cells are expected; the "
        "matching-weight panels identify the supported overlap."
    ),
    "merfish-allen3Datlas-alignment.ipynb": _THREE_D_NOTE,
    "starmap-allen3Datlas-alignment.ipynb": (
        _THREE_D_NOTE + " Its cell [6] also reads the STARmap table through an absolute path "
        "on the notebook author's own machine (`/home/manju/Documents/...`), the only such "
        "path in the pinned notebook set; the replay rewrites the leading directories to the "
        "`../starmap_data/` convention the other notebooks use. Same file, same bytes."
    ),
}


@dataclass(slots=True)
class ComparisonResult:
    """Evidence emitted for one upstream notebook."""

    notebook: str
    status: str
    metrics: dict[str, float]
    figures: list[Any]
    upstream_seconds: float = 0.0
    squidpy_seconds: float = 0.0
    note: str | None = None
    #: Upstream cell source that produced each figure, parallel to ``figures`` -- lets the
    #: written notebook read as upstream's own cells (code + output) rather than opaque
    #: ``result.figures[N]`` dumps. Empty for synthetic status panels.
    figure_sources: list[str] = field(default_factory=list)
    #: Per-cell frames both passes carry, merged side by side -- see :func:`_paired_frames`.
    #: Written beside the metrics so a label disagreement can be located in space instead of
    #: only counted.
    paired_frames: dict[str, Any] = field(default_factory=dict)
    #: The squidpy half of each pair, as its own PNG, parallel to ``figures``. The docs pages
    #: show the port's plot beside upstream's *published* figure rather than beside our replay
    #: of it, so they need the port panel alone -- cropping it back out of the composed pair
    #: is a manual step that silently rots the moment a plot's aspect changes.
    port_figures: list[bytes] = field(default_factory=list)


class _FitResult(dict):
    """Upstream's fit output, readable through every API its own notebooks still use.

    The 17 pinned notebooks read the result three different ways: ``out['A']`` (13 of them),
    ``out[0]`` (`merfish-merfish-alignment-simulation`), and ``A, v, xv = LDDMM(...)``
    (`merfish-merfish-alignment-using-L-T`). Upstream returns a plain dict, so the latter two
    raise `KeyError: 0` and a tuple-unpacking error against the pinned commit -- and the
    replay would stop before reaching the plots that are the point of the comparison.
    Accepting all three costs only ``dict`` iteration over keys, which no notebook does.
    """

    #: What positions 0, 1, 2 meant in the tuple upstream used to return.
    _POSITIONAL = ("A", "v", "xv")

    def __iter__(self):
        return iter(tuple(self[name] for name in self._POSITIONAL))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return super().__getitem__(self._POSITIONAL[key])
        return super().__getitem__(key)


@dataclass(slots=True)
class _ReplayPass:
    """One end-to-end execution of a notebook, with some fit standing in for ``LDDMM``."""

    figures: list[tuple[int, bytes]]
    namespace: dict[str, Any]
    fits: list[tuple[tuple[Any, ...], dict[str, Any]]]
    seconds: float
    skipped: list[tuple[int, str]]


@contextmanager
def _tensors_convertible_to_numpy():
    """Let upstream's own cells call ``.numpy()`` on a fitted tensor, as they were written to.

    The notebooks ask for ``device='cpu'`` and the replay overrides that to the fit device so
    both sides are timed and computed alike. On a GPU the override makes upstream's *own*
    post-fit cells raise ``can't convert cuda:0 device type tensor to numpy`` -- seven of the
    seventeen notebooks skip cells for exactly this reason, and one consequence is that the
    ``results`` frame upstream publishes never gets built, so the published-coordinates
    comparison has nothing to compare.

    The error is an artifact of the override, not of upstream's code: on the device the
    notebook asked for, `.numpy()` works. Restoring that costs one hop to host memory and is
    applied identically to both passes, so it cannot advantage either. Off GPU it is inert.
    """
    import torch

    original = torch.Tensor.numpy

    def numpy(self, *args: Any, **kwargs: Any) -> Any:
        return original(self.detach().cpu(), *args, **kwargs) if self.is_cuda else original(self, *args, **kwargs)

    torch.Tensor.numpy = numpy
    try:
        yield
    finally:
        torch.Tensor.numpy = original


@contextmanager
def _replay_directory(notebook_path: Path):
    """A scratch working directory in which upstream's relative data paths resolve.

    Upstream notebooks read their data as ``../<name>_data/...``, relative to
    ``docs/notebooks``. ``merfish-xenium-alignment.ipynb`` is the one exception: it reaches
    a level further up (``../../merfish_data/...``) for files that live at the same place as
    everyone else's, so from its own directory those paths do not exist.

    Rather than edit the vendored checkout -- which has to stay byte-identical to the pinned
    commit -- the replay runs two levels below a scratch root that carries a symlink to
    every data directory at *both* depths, so either convention reaches the real data. This
    also keeps anything a notebook writes out of the vendored tree.
    """
    data_root = notebook_path.parent.parent
    data_dirs = sorted(path for path in data_root.iterdir() if path.is_dir() and path.name.endswith("_data"))
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="stalign-replay-") as scratch:
        root = Path(scratch)
        inner = root / data_root.name
        work = inner / notebook_path.parent.name
        work.mkdir(parents=True)
        for level in (root, inner):
            for data in data_dirs:
                (level / data.name).symlink_to(data)
        os.chdir(work)
        try:
            yield work
        finally:
            os.chdir(previous)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _fit_device() -> str:
    """Where both fits run and where both hand their results back."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _await_fit_device() -> None:
    """Block until queued GPU work has finished, so a timing is not stopped mid-flight.

    The port's side of every comparison ends in ``jax.block_until_ready``. Without the torch
    equivalent the two passes are timed by different rules: CUDA kernels are asynchronous, so
    ``time.monotonic()`` can stop while upstream still has work queued, and upstream's seconds
    come out flattering. Its loop synchronises implicitly in many places -- it prints energies
    and draws convergence figures, both of which pull values back to the host -- so the bias is
    small, but "small" is not a basis for publishing a speed comparison.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _on_fit_device(value: Any) -> Any:
    """Put a fit result on :func:`_fit_device`, preserving its container shape.

    Both sides land on the same device so neither is advantaged and no asymmetry can creep
    into the comparison. Notebooks whose own tensors sit elsewhere will raise a device
    mismatch from upstream's helpers; that is reported as a finding, not worked around.
    """
    import torch

    if isinstance(value, list | tuple):
        return type(value)(_on_fit_device(item) for item in value)
    return value.detach().to(torch.device(_fit_device())) if hasattr(value, "detach") else value


#: Upstream is imported by path and pre-seeded into the replay namespace already patched, so
#: the notebook's own ways of (re)binding it must not run. A plain `import STalign` would find
#: the package on `sys.path` and replace the patched module; `importlib.reload(STalign)` --
#: which `merfish-merfish-alignment-using-L-T.ipynb` does at cell 38 -- is worse, because a
#: reload that succeeded would restore the real `LDDMM` half way through the notebook and
#: leave the Squidpy pass silently comparing upstream against itself. It fails loudly here
#: only because the module is loaded by path and never enters `sys.modules`.
_STALIGN_REBIND = re.compile(
    r"^\s*(import\s+STalign\b|from\s+STalign\s+import\b|importlib\.reload\s*\(\s*STalign\s*\))"
)


#: One notebook -- `starmap-allen3Datlas-alignment`, cell [6] -- reads its input through an
#: absolute path on the author's own machine
#: (`/home/manju/Documents/STalign-master/STalign_build/docs/starmap_data/...`), the only
#: absolute path in the whole vendored notebook set. :func:`_replay_directory` can place data
#: at either *relative* depth the notebooks use, but no symlink can rescue `/home/manju`, so
#: the leading directories are rewritten to the `../<name>_data/` convention every other
#: notebook already uses. Deliberately anchored on the opening quote followed by `/`: paths
#: that are already relative (`../merfish_data/`, and `merfish-xenium`'s `../../merfish_data/`)
#: do not start with a slash and are left exactly as upstream wrote them. This moves where the
#: file is read from and nothing else -- same bytes, same fit.
_AUTHOR_ABSOLUTE_DATA_PATH = re.compile(r"""(['"])/[^'"]*?([A-Za-z0-9]+_data)/""")


def _clean_cell(source: str) -> str:
    """Strip what cannot be replayed from one notebook cell, and nothing else.

    Only the shell/magic lines and upstream's own (re)binding of itself are dropped. Dropping
    the *cell*
    instead would take the rest of it with them: several notebooks put `import pandas as
    pd` (and numpy, torch, matplotlib) in the same cell as `import STalign`, so a
    cell-level skip left the whole notebook without pandas and failed several cells later
    with a bare `NameError`.
    """
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")) or _STALIGN_REBIND.match(line):
            continue
        lines.append(_AUTHOR_ABSOLUTE_DATA_PATH.sub(r"\1../\2/", line))
    return "\n".join(lines)


#: Helpers upstream renamed after these notebooks were written. `git show 5837b03` in the
#: vendored checkout ("change atlas to source") changes nothing but these names and one
#: docstring line, so resolving the old spelling to the new function is faithful rather than
#: a guess. Only `merfish-merfish-alignment-simulation` still needs it, but the whole set is
#: mapped so the next stale notebook does not cost another cluster run to find.
_RENAMED_HELPERS = {
    "transform_image_atlas_with_A": "transform_image_source_with_A",
    "transform_image_atlas_to_target": "transform_image_source_to_target",
    "transform_image_target_to_atlas": "transform_image_target_to_source",
    "transform_points_atlas_to_target": "transform_points_source_to_target",
    "transform_points_target_to_atlas": "transform_points_target_to_source",
}


class _UpstreamProxy:
    """The vendored module, answering to the helper names its own notebooks still use.

    Resolution goes through the live module every time, so the fit stand-in patched onto it
    is what the notebook calls.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, _RENAMED_HELPERS.get(name, name))


def _notebook_cells(notebook: str) -> list[tuple[int, str]]:
    """The runnable code cells of one upstream notebook, in order."""
    notebook_path = upstream.vendor_root() / "docs" / "notebooks" / notebook
    payload = json.loads(notebook_path.read_text())
    cells = []
    for cell_number, cell in enumerate(payload["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Writing results back into the vendored checkout is not part of the comparison.
        if "np.savez(" in source or ".to_csv(" in source:
            continue
        cleaned = _clean_cell(source)
        if cleaned.strip():
            cells.append((cell_number, cleaned))
    return cells


def _replay_notebook(notebook: str, fit: Any, *, function: str = "LDDMM") -> _ReplayPass:
    """Run a whole notebook with ``fit`` standing in for ``STalign.<function>``.

    ``fit`` is called as ``fit(args, kwargs, original)``. The unpatched function is handed
    over explicitly because reaching for ``STalign.LDDMM`` inside the callback finds the
    stand-in and recurses until the stack gives out.

    The point is to get *past* the fit. Everything a notebook shows -- the contour grids
    from ``build_transform``, the warped images, the aligned-point scatters -- is produced by
    cells after it, and those cells are upstream's own code reading ``A``, ``v`` and ``xv``
    off the fit result. Swapping only the fit and letting the rest run verbatim is what makes
    the two passes comparable: identical plotting code, identical interpolation, and the
    fitted arrays as the single difference.

    Figures are collected per cell rather than at the end because notebooks reuse
    ``plt.subplots`` freely and upstream's ``LDDMM`` draws unconditionally.
    """
    import matplotlib.pyplot as plt
    import torch

    st = upstream.load()
    notebook_path = upstream.vendor_root() / "docs" / "notebooks" / notebook
    namespace: dict[str, Any] = {
        "__file__": str(notebook_path),
        "__name__": "__main__",
        "STalign": _UpstreamProxy(st),
    }
    figures: list[tuple[int, bytes]] = []
    fits: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    skipped: list[tuple[int, str]] = []
    elapsed = 0.0
    original = getattr(st, function)

    def standin(*args: Any, **kwargs: Any) -> Any:
        nonlocal elapsed
        fits.append((args, kwargs))
        started = time.monotonic()
        # Upstream's LDDMM draws its own convergence figures unconditionally
        # (STalign.py:1094, :1140, :1142) and the port draws nothing. Discarding whatever the
        # fit itself opened keeps both passes emitting exactly the notebook's own figures, in
        # the same order -- otherwise every pair after the fit cell is off by one.
        before = set(plt.get_fignums())
        result = fit(args, kwargs, original)
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        elapsed += time.monotonic() - started
        return result

    setattr(st, function, standin)
    # Both fits return on `_fit_device`, so the notebook has to build its own tensors there
    # too or upstream's helpers refuse to mix them. The notebooks disagree among themselves:
    # `heart-alignment` sets this, `merfish-merfish-alignment-simulation` never does, and
    # without it the latter lost every cell after `build_transform`. Setting it once for the
    # whole replay is a device-only change, applied identically to both passes.
    previous_device = torch.get_default_device()
    torch.set_default_device(_fit_device())
    try:
        with _replay_directory(notebook_path), _tensors_convertible_to_numpy():
            for cell_number, source in _notebook_cells(notebook):
                try:
                    exec(compile(source, f"{notebook}:cell-{cell_number}", "exec"), namespace)
                    for number in plt.get_fignums():
                        buffer = io.BytesIO()
                        plt.figure(number).savefig(buffer, format="png", dpi=EMBED_DPI, bbox_inches="tight")
                        figures.append((cell_number, buffer.getvalue()))
                except Exception as error:  # noqa: BLE001 - see below; both passes must agree
                    # A cell can fail for reasons that have nothing to do with the fit: these
                    # notebooks predate their dependencies, and e.g.
                    # `mean_squared_error(..., squared=False)` was removed in scikit-learn 1.6.
                    # Aborting the comparison over an evaluation cell would throw away every
                    # figure already drawn, so the replay records it and carries on. Nothing is
                    # hidden: the caller reports these, and requires both passes to fail the
                    # same cells before trusting the comparison.
                    skipped.append((cell_number, f"{type(error).__name__}: {error}"))
                finally:
                    plt.close("all")
    finally:
        setattr(st, function, original)
        torch.set_default_device(previous_device)

    if not fits:
        raise RuntimeError(f"{notebook}: never called STalign.{function}; {skipped}")
    return _ReplayPass(figures=figures, namespace=namespace, fits=fits, seconds=elapsed, skipped=skipped)


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


#: Namespace entries that say nothing about the fit: inputs both passes share, and the
#: coordinate axes the notebooks build before calling LDDMM.
_INPUT_NAMES = frozenset({"I", "J", "XI", "XJ", "YI", "YJ", "xI", "xJ", "yI", "yJ", "Ifoo", "Jfoo"})


def _frame_metrics(name: str, expected: Any, actual: Any) -> dict[str, float]:
    """Per-column scores for the frames a notebook carries beside its arrays.

    Relative L2 needs numbers, so the region assignment in the volume-to-section notebooks --
    ``df['acronym']``, looked up per cell from the aligned atlas coordinate -- was plotted by
    ``plot_brain_regions`` and scored nowhere. The two passes' legends visibly disagree there
    (see ``docs/STALIGN_DIVERGENCES.md`` row D11), and without this a reader cannot tell six
    reassigned boundary cells from six hundred.

    Two kinds of column, two kinds of score:

    *Text* columns get the fraction of rows whose label differs. Rows null on *both* sides
    count as agreeing: a cell outside the atlas has no region, and ``!=`` would otherwise
    score every one of them as a disagreement.

    *Float* columns get the median absolute difference, in the column's own units -- microns,
    for the atlas coordinates. That is what turns the label fraction from a number into an
    explanation: the acronym is a nearest-voxel lookup into the 50 micron annotation volume
    (``analyze3Dalign`` warps it with ``mode='nearest'`` and rounds the query points), so a
    label fraction alone cannot say whether the cells moved by a fifth of a voxel or by ten.
    A relative L2 cannot say either, being scale-free. Columns the two passes share as inputs
    land at exactly 0.0 here, which is the control: it shows the difference is the fit and not
    the harness.

    Integer columns are skipped. In these notebooks they are ids -- ``df['struct_id']`` is the
    acronym in numeric clothing -- so a distance would be meaningless and a disagreement would
    duplicate the text column beside it.
    """
    metrics: dict[str, float] = {}
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        return metrics
    for column in expected.columns:
        left, right = expected[column], actual[column]
        if len(left) < 4 or left.dtype.kind in "iu":
            continue
        if left.dtype.kind == "f":
            delta = np.abs(left.to_numpy() - right.to_numpy())
            metrics[f"{name}[{column}] median abs delta"] = float(np.nanmedian(delta))
            continue
        differs = left.to_numpy() != right.to_numpy()
        both_null = left.isna().to_numpy() & right.isna().to_numpy()
        metrics[f"{name}[{column}] label disagreement"] = float((differs & ~both_null).mean())

        # The per-row fraction alone cannot tell boundary jitter from bulk displacement.
        # Jitter flips cells between *adjacent* regions, so both sides still contain both and
        # the label *sets* barely move; a region only vanishes when it loses every cell it
        # had. So report how many labels appear on exactly one side, which is what makes the
        # two legends visibly different lists rather than the same list reordered.
        upstream_labels = set(left.dropna().unique())
        squidpy_labels = set(right.dropna().unique())
        shared = upstream_labels & squidpy_labels
        union = upstream_labels | squidpy_labels
        metrics[f"{name}[{column}] labels only upstream"] = float(len(upstream_labels - squidpy_labels))
        metrics[f"{name}[{column}] labels only squidpy"] = float(len(squidpy_labels - upstream_labels))
        metrics[f"{name}[{column}] label set jaccard"] = float(len(shared) / len(union)) if union else 1.0
    return metrics


def _corrected_cell_coords(namespace: dict[str, Any]) -> np.ndarray | None:
    """Per-cell atlas coordinates read at the indices ``analyze3Dalign`` uses for its *labels*.

    Ledger row **D14**. Upstream builds its coordinate grid at 10 microns --
    ``res = np.array(dx)`` is overwritten by ``res = 10.0`` three lines later
    (``STalign.py:1975``, ``:1978``) -- and then indexes that grid two different ways for the
    same cells: ``qi = round((q - origin)/res)`` for the region labels (``:1994``), and
    ``col/row = (x - X_[0])/dx`` for ``coord0``/``coord1``/``coord2`` (``:1999-2000``). With
    the notebooks' ``dx=50`` the coordinate readout therefore addresses one fifth of each axis,
    so ``df[coord*]`` describes the top-left **4 % of the section by area** stretched over all
    of it. Measured on a real fit, upstream's two readings of its own ``tform`` sit **4267 um**
    apart at the median, and the reported depth spread is ~10x too small.

    Recomputed here at the label indices, through upstream's own ``build_transform3D`` and
    upstream's own index formula -- so this is a self-consistent reading of upstream's grid,
    not a reimplementation of the transform. Applied identically to both passes, so the pair
    stays comparable; what changes is that the metric describes the whole section.

    Cell positions come from ``df['x']``/``df['y']``, never the namespace's ``x``/``y``: cell
    [41] of both volume-to-section notebooks rebinds those to ``df['coord0']`` and
    ``-df['coord1']``, so by the end of the replay the namespace names no longer hold cell
    positions at all.

    ``None`` when the namespace is not a volume-to-section replay, or is missing a piece.
    """
    import torch

    frame, xv, velocity, affine, xJ = (namespace.get(name) for name in ("df", "xv", "v", "A", "xJ"))
    if frame is None or xv is None or velocity is None or affine is None or xJ is None:
        return None
    if not all(column in getattr(frame, "columns", ()) for column in ("x", "y")):
        return None
    st = upstream.load()
    res = 10.0  # upstream's own grid resolution, the one `tform` is actually built on
    axis_row, axis_col = np.asarray(_numpy(xJ[0]), dtype=float), np.asarray(_numpy(xJ[1]), dtype=float)
    grid = np.stack(
        np.meshgrid(
            np.zeros(1),
            np.arange(axis_row[0], axis_row[-1], res),
            np.arange(axis_col[0], axis_col[-1], res),
            indexing="ij",
        ),
        -1,
    )
    # Everything goes to CPU, including the fit's own tensors. `build_transform3D` keeps `v`
    # and `A` wherever it found them and `interp3D` then subtracts `xv[i][0]` from a grid
    # derived from `XJ` (STalign.py:697), so *any* of the four arriving on a different device
    # raises "Expected all tensors to be on the same device". Matching the fit's device instead
    # means guessing which of the four carries the authoritative one -- two attempts got that
    # wrong. This is a one-off geometric evaluation over the section's cells, not part of the
    # fit, so paying for a host copy buys the ambiguity away.
    def cpu(value: Any) -> Any:
        return torch.as_tensor(_numpy(value), dtype=torch.float64)

    # The context manager is the load-bearing part, not the coercion above it.
    # `build_transform3D` re-wraps `A`, `v` and `XJ` in `torch.tensor(...)` itself, and those
    # factory calls land on whatever default device is ambient -- CUDA, under the replay. The
    # `xv` list is the one argument it does *not* re-wrap, so it stayed on CPU and `interp3D`
    # hit "Expected all tensors to be on the same device" at `phii[i] -= x[i][0]`
    # (STalign.py:697). Coercing the arguments alone did not fix that, twice. Pinning the
    # default device makes upstream's own internal calls agree with them.
    with torch.device("cpu"):
        tform = st.build_transform3D(
            [cpu(axis) for axis in xv],
            cpu(velocity),
            cpu(affine),
            direction="b",
            XJ=torch.as_tensor(grid, dtype=torch.float64),
        )
    query = np.stack((frame["y"].to_numpy(dtype=float), frame["x"].to_numpy(dtype=float)))
    index = np.round((query - np.stack([axis_row[0], axis_col[0]])[..., None]) / res).astype(int)
    # Upstream's own comment concedes a different resolution can put points out of bounds; clip
    # rather than raise, so one stray cell cannot cost the whole metric.
    index[0] = np.clip(index[0], 0, tform.shape[1] - 1)
    index[1] = np.clip(index[1], 0, tform.shape[2] - 1)
    return np.asarray(_numpy(tform[0, index[0], index[1], :]), dtype=float)


def _corrected_coord_metrics(upstream_ns: dict[str, Any], squidpy_ns: dict[str, Any]) -> dict[str, float]:
    """``df[coord*]`` scored again at self-consistent indices -- see :func:`_corrected_cell_coords`.

    Emitted *beside* the uncorrected columns rather than replacing them: the published figures
    quote the uncorrected ones, so removing them would silently rewrite history, and the pair
    together is what shows how much the indexing bug was costing.
    """
    left, right = _corrected_cell_coords(upstream_ns), _corrected_cell_coords(squidpy_ns)
    if left is None or right is None or left.shape != right.shape:
        return {}
    metrics: dict[str, float] = {}
    for axis in range(left.shape[1]):
        delta = np.abs(left[:, axis] - right[:, axis])
        metrics[f"df[coord{axis}] corrected median abs delta"] = float(np.nanmedian(delta))
    return metrics


def _paired_frames(upstream_ns: dict[str, Any], squidpy_ns: dict[str, Any]) -> dict[str, Any]:
    """The frames both passes computed, merged column-wise with an ``upstream_``/``squidpy_`` prefix.

    ``_frame_metrics`` reduces these to one number per column, which answers *how many* cells
    changed region but never *which* ones. That distinction is the whole question for the
    volume-to-section notebooks: the acronym is a nearest-voxel lookup into a 50 micron
    annotation volume, so cells displaced by one or two voxels are expected to flip label at a
    region boundary and nowhere else. Only the per-cell rows can show whether the disagreement
    sits on boundaries or is spread through region interiors -- the second would mean something
    other than displacement is wrong.

    Emitted for every notebook that carries a frame; the 3D ones are simply the only ones where
    it currently says anything.
    """
    paired: dict[str, Any] = {}
    for name, upstream_value in sorted(upstream_ns.items()):
        if name.startswith("_") or name in _INPUT_NAMES or name not in squidpy_ns:
            continue
        actual = squidpy_ns[name]
        if not (hasattr(upstream_value, "columns") and hasattr(actual, "columns")):
            continue
        if list(upstream_value.columns) != list(actual.columns) or len(upstream_value) != len(actual):
            continue
        import pandas as pd

        paired[name] = pd.concat(
            [upstream_value.add_prefix("upstream_"), actual.add_prefix("squidpy_").set_axis(upstream_value.index)],
            axis=1,
        )
    return paired


def _namespace_metrics(upstream_ns: dict[str, Any], squidpy_ns: dict[str, Any]) -> dict[str, float]:
    """Relative L2 on every array the notebook itself computed, under its own variable name.

    The notebooks are the specification here: whatever they assign and plot -- ``phii``,
    ``phiI``, ``tpointsI``, ``xI_LDDMM`` -- is what a reader of the upstream tutorial would
    compare. Diffing the two namespaces reports exactly those, instead of quantities this
    harness invented.
    """
    metrics: dict[str, float] = {}
    for name, upstream_value in sorted(upstream_ns.items()):
        if name.startswith("_") or name in _INPUT_NAMES or name not in squidpy_ns:
            continue
        # Frames carry both kinds: the numeric path below still scores an all-numeric one whole,
        # and this adds the columns relative L2 cannot see.
        if hasattr(upstream_value, "columns") and hasattr(squidpy_ns[name], "columns"):
            try:
                metrics.update(_frame_metrics(name, upstream_value, squidpy_ns[name]))
            except Exception:  # noqa: BLE001 - a frame this harness did not build may hold anything
                pass
        try:
            expected, actual = _numpy(upstream_value), _numpy(squidpy_ns[name])
        except Exception:  # noqa: BLE001 - namespaces hold modules, dataframes, closures
            continue
        if expected.dtype.kind not in "fiu" or expected.shape != actual.shape or expected.size < 4:
            continue
        metrics[f"{name} relative L2"] = _relative_l2(actual, expected)
    return metrics


def _rank(value: Any) -> int:
    """Number of dimensions, without forcing a device transfer on a CUDA tensor."""
    ndim = getattr(value, "ndim", None)
    return int(ndim) if ndim is not None else int(np.ndim(value))


def _cast_like(value: Any, default: Any, *, name: str) -> Any:
    """Convert one captured upstream argument to the type the port declares for it.

    The cast is read from the port's own default rather than from a list kept here, so
    the two cannot drift apart silently. It matters because ``lddmm`` passes its scalar
    arguments (``niter``, ``diffeo_start``, the step sizes, the sigmas) to ``jax.jit`` as
    *static* arguments: JAX hashes those, and a 0-D NumPy array is unhashable, so a
    blanket ``np.asarray`` turns every scalar upstream passes into a trace-time
    ``ValueError``. Array arguments (``muA``/``muB``, landmarks, warm-start velocities)
    must stay arrays.
    """
    for python_type in (bool, int, float):
        if isinstance(default, python_type):
            if _rank(value) != 0:
                raise TypeError(f"upstream passed a rank-{_rank(value)} value for scalar argument {name!r}")
            return python_type(_numpy(value).item())
    # `tol`-style parameters declare `None` but are still hashed when supplied.
    if default is None and not isinstance(value, str) and _rank(value) == 0:
        return float(_numpy(value).item())
    return _numpy(value)


#: The port's keyword names, with one value each carrying the *type* a captured upstream
#: argument has to be cast to. Nothing reads these values as defaults any more -- the
#: ``align_stalign_*`` entry points resolve the port's own -- so they exist only to say
#: whether a keyword is an ``int``, a ``float`` or an array. Mirrored here rather than read
#: from squidpy so the conversion stays testable without squidpy and JAX installed, neither
#: of which is a dependency of this repo. ``test_port_signature_matches_the_mirror`` fails
#: wherever squidpy *is* installed if squidpy declares a solver keyword this omits.
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
    # squidpy's `_CONSUMED_KEYS`: declared by the public solver TypedDicts but eaten by the
    # `align_stalign_*` entry points rather than forwarded to the solver. Mirrored for the
    # types alone -- upstream's `LDDMM` has no such keywords (it is handed images that were
    # rasterized outside the fit), so nothing captured ever carries them, and
    # `_completed_kwargs` cannot fill them either: they are absent from `_SOLVER_DEFAULTS`.
    # `initial_affine` is the exception the call site already passes by hand.
    "dx": 30.0,
    "blur": (2.0, 1.0, 0.5),
    "raster_expand": 1.1,
    "initial_affine": None,
}
PORT_PARAMETERS: Mapping[str, inspect.Parameter] = {
    name: inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
    for name, default in PORT_DEFAULTS.items()
}


def _convert_kwargs(kwargs: dict[str, Any], parameters: Mapping[str, inspect.Parameter]) -> dict[str, Any]:
    """Map captured upstream ``LDDMM`` keywords onto the port's ``lddmm`` signature.

    Keywords upstream accepts but the port does not express (``device``, ``dtype``) are
    dropped; the ones it renames (``A``/``L``/``T``, ``pointsI``/``pointsJ``, ``v``/``xv``)
    are translated explicitly. Everything else is cast by :func:`_cast_like`.
    """
    renamed = {"A", "L", "T", "pointsI", "pointsJ", "v", "xv"}
    converted = {
        key: _cast_like(value, parameters[key].default, name=key)
        for key, value in kwargs.items()
        if key in parameters and key not in renamed and value is not None
    }
    if kwargs.get("A") is not None:
        affine = _numpy(kwargs["A"])
        converted["L"], converted["T"] = affine[:2, :2], affine[:2, 2]
    else:
        # Upstream defaults both to `None`, meaning identity.
        linear, translation = kwargs.get("L"), kwargs.get("T")
        converted["L"] = np.eye(2) if linear is None else _numpy(linear)
        converted["T"] = np.zeros(2) if translation is None else _numpy(translation)
    if kwargs.get("pointsI") is not None:
        converted["points_source"] = _numpy(kwargs["pointsI"])
        converted["points_target"] = _numpy(kwargs["pointsJ"])
    if kwargs.get("v") is not None:
        converted["initial_velocity"] = _numpy(kwargs["v"])
        converted["velocity_grid"] = tuple(_numpy(axis) for axis in kwargs["xv"])
    return converted


#: Axis reconstruction residual, in the axes' own units, per notebook run. Filled by
#: :func:`axis_placement` and reported beside the comparison metrics: the public
#: ``align_stalign_*`` API takes an element's placement rather than its axes, so the
#: harness has to hand it a ``Scale``/``Translation`` and squidpy rebuilds the axes from
#: that. Where the rebuild is not bit-exact, the two passes no longer see byte-identical
#: inputs, and that has to be a measured number rather than a silent one.
_AXIS_RESIDUAL: dict[str, float] = {}


def axis_placement(axis: np.ndarray) -> tuple[float, float]:
    """The ``(step, shift)`` whose ``arange(n) * step + shift`` best reproduces ``axis``.

    squidpy's ``_element_axes`` rebuilds an image's physical axes as
    ``np.arange(size) * step + shift`` from the scale and translation the element carries.
    Upstream builds its own the same way -- ``np.arange(n) * dx - (n - 1) * dx / 2`` -- so
    passing the original ``dx`` and offset would reproduce it bit-for-bit. The replay never
    sees them: it intercepts a ``LDDMM`` call that is already holding axis *arrays*, and
    ``arange(n) * dx + offset`` has by then absorbed rounding into the values it stores.
    Recovering the step by differencing is exact only when the offset is small next to the
    step -- ``arange(17) * 0.65 - 1234.5`` gives ``x[1] - x[0] == 0.650000000000091``.

    So try the constructions upstream actually uses and take the first that is exact; where
    none is, take the one with the smallest maximum error and record it in
    :data:`_AXIS_RESIDUAL`. Measured over the notebooks' axis shapes this is exact in the
    large majority of cases and ~1e-14 in the rest -- small, but the comparison asserts
    agreement at ~1e-15, so it is reported rather than absorbed.
    """
    n = axis.size
    if n < 2:
        return 1.0, float(axis[0]) if n else 0.0
    span = float(axis[-1] - axis[0])
    candidates = (
        (span / (n - 1), float(axis[0])),  # whole-span slope: best conditioned
        (float(axis[1] - axis[0]), float(axis[0])),  # adjacent difference
        (-2.0 * float(axis[0]) / (n - 1), float(axis[0])),  # upstream's centred grid
        (float(np.median(np.diff(axis))), float(axis[0])),  # robust to one bad sample
    )
    best: tuple[float, tuple[float, float]] | None = None
    for step, shift in candidates:
        error = float(np.max(np.abs(np.arange(n, dtype=float) * step + shift - axis)))
        if error == 0.0:
            return step, shift
        if best is None or error < best[0]:
            best = (error, (step, shift))
    assert best is not None
    _AXIS_RESIDUAL["max"] = max(_AXIS_RESIDUAL.get("max", 0.0), best[0])
    return best[1]


#: The image key both sides of a pair are parsed under. One name, so `image_key` is a
#: plain string rather than a `(ref, query)` tuple.
_IMAGE_KEY = "replay"


def as_sdata(array: np.ndarray, axes: Sequence[np.ndarray]) -> Any:
    """Wrap a channels-first array and its axes as a one-image ``SpatialData``.

    The public ``align_stalign_*`` entry points read an element's physical axes off the
    scale and translation it carries into a coordinate system, so the axes upstream passed
    are expressed as exactly that -- see :func:`axis_placement` for the part of this that
    cannot be made exact.
    """
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Image3DModel
    from spatialdata.transformations import Scale, Translation
    from spatialdata.transformations import Sequence as TransformSequence

    spatial = ("z", "y", "x")[-len(axes) :]
    placement = [axis_placement(np.asarray(axis, dtype=float)) for axis in axes]
    model = Image3DModel if len(axes) == 3 else Image2DModel
    element = model.parse(
        array,
        dims=("c", *spatial),
        transformations={
            "global": TransformSequence(
                [
                    Scale([step for step, _ in placement], axes=spatial),
                    Translation([shift for _, shift in placement], axes=spatial),
                ]
            )
        },
    )
    return SpatialData(images={_IMAGE_KEY: element})


def _channels_first(image: Any, *, ndim: int) -> np.ndarray:
    """Upstream's image as squidpy reads it: channels first, one channel if it has none."""
    array = np.asarray(_numpy(image), dtype=float)
    return array if array.ndim == ndim + 1 else array[None]


def solver_keys() -> frozenset[str]:
    """Every tuning keyword the public API declares, read off its public TypedDicts.

    These are what the ``align_stalign_*`` entry points accept, so they are what a captured
    upstream keyword may be forwarded as. Read from squidpy rather than listed here, and
    from its *public* surface rather than from ``lddmm``'s signature: the TypedDicts are
    exported, so unlike the kernel they carry a stability contract.
    """
    from squidpy.experimental.tl import (
        StalignImageSolverKwargs,
        StalignObsSolverKwargs,
        StalignVolumeSolverKwargs,
    )

    keys: set[str] = set()
    for declaration in (StalignImageSolverKwargs, StalignObsSolverKwargs, StalignVolumeSolverKwargs):
        keys |= set(declaration.__required_keys__) | set(declaration.__optional_keys__)
    return frozenset(keys)


def upstream_solver_defaults() -> Mapping[str, Any]:
    """The kernel's defaults, which are upstream's ``LDDMM`` defaults.

    Private by necessity, and load-bearing: ``align_stalign_image`` resolves
    ``_IMAGE_DEFAULTS`` instead -- squidpy's own choices for images, deliberately *not*
    upstream's (``a`` 20 against 500, ``niter`` 200 against 5000, ``diffeo_start`` 100
    against 0, ``epV`` 1.0 against 2e3). Fourteen of the seventeen notebooks pass none of
    those four, so letting the entry point fill them would run the port on a different fit
    from upstream and publish the gap as a port divergence.

    ``test_solver_defaults_match_upstream`` pins that these really are upstream's, so the
    two passes start from identical parameters wherever a notebook is silent.
    """
    from squidpy.experimental.tl._align._stalign import _SOLVER_DEFAULTS

    return _SOLVER_DEFAULTS


def smoke_niter() -> int | None:
    """Iteration cap for the parameter gate, from ``STALIGN_SMOKE_NITER``.

    A full sweep costs ~45 minutes of GPU and only reports a divergence once every fit has
    converged, which is the worst moment to discover that the two sides were handed different
    parameters. One iteration is enough to catch that: after a single step ``A`` and ``v``
    still agree to ~1e-10 when the inputs match, and disagree grossly when they do not. The
    `_IMAGE_DEFAULTS` mix-up -- ``a`` 20 against 500, ``diffeo_start`` 100 against 0 -- would
    have shown on the first step of the first notebook.

    Applied to *both* passes, so it changes what is compared, never which side gets what.
    """
    capped = os.environ.get("STALIGN_SMOKE_NITER")
    return int(capped) if capped else None


def niter_scale() -> float | None:
    """Iteration multiplier from ``STALIGN_NITER_SCALE``, or ``None`` for the notebook's own.

    Some upstream notebooks stop while their objective is still dropping --
    `starmap-allen3Datlas-alignment` asks for 800 iterations and its aligned atlas slice sits
    visibly loose against the section. Comparing two implementations at an iteration cap
    compares where each ran out of budget as much as where its optimum is, and the port is not
    line-for-line upstream anyway (see the ledger), so pinning the published cap buys less than
    it costs.

    Applied to *both* passes, like :func:`smoke_niter`: it changes what is compared, never
    which side gets what. ``diffeo_start`` scales with it, because it is a phase boundary
    rather than an absolute step -- stretching the run without moving it would silently give
    the diffeomorphic phase a different share of the schedule than the notebook intended.
    """
    scale = os.environ.get("STALIGN_NITER_SCALE")
    return float(scale) if scale else None


def collapse_upstream_mixture_means() -> bool:
    """Whether upstream's fixed mixture means are collapsed too, from ``STALIGN_D13_COLLAPSE_MEANS``.

    **Comparison-only, and off by default.** With it set the upstream pass no longer replays
    the notebook verbatim, so a published sweep must never carry it. Ledger row D13 is what
    this measures.

    Both ``allen3Datlas`` notebooks pass a length-3 ``muA``/``muB`` against a single-channel
    ``J``. Upstream broadcasts the mean and sums over the *broadcast* axis
    (``STalign.py:1554-1555``), so its artifact and background exponents carry a factor 3
    while ``WM``'s does not -- effective widths ``sigmaA/sqrt(3)`` and ``sigmaB/sqrt(3)``.
    squidpy validates the length, so the replay collapses the mean for the port and the two
    sides fit different mixtures. Setting this collapses upstream's the same way, which
    isolates what that costs: the difference between a run with it and a run without is D13's
    contribution and nothing else.

    Collapsing *upstream* rather than expanding the port, deliberately. Upstream accepts a
    length-1 mean and then handles it identically to squidpy, so this needs no squidpy change
    and is reproducible from a clean checkout -- unlike D11's flag, which lives on an unpushed
    commit. Handing the port ``sigmaA/sqrt(3)`` instead would match the exponent but leave its
    ``(2*pi*sigma^2)^(C/2)`` prefactor ``sqrt(3)`` too large on ``WA`` and ``WB`` and unchanged
    on ``WM``, and that does not cancel in the normalised posterior.

    Rank 3 only, like the divergence: no rank-2 notebook passes a fixed mean whose length
    disagrees with its section, so there is nothing there for this to change.
    """
    return os.environ.get("STALIGN_D13_COLLAPSE_MEANS", "") not in {"", "0"}


def _section_channels(section: Any) -> int:
    """Channel count of the fixed section, which is what a fixed mixture mean has to match."""
    array = np.asarray(_numpy(section))
    return array.shape[0] if array.ndim == 3 else 1


def _collapsed_mean(mean: Any, n_channels: int) -> np.ndarray:
    """One fixed mixture-mean entry per section channel.

    The notebooks' length-3 means are three copies of one value, so resizing loses no
    information -- what it drops is upstream's factor-of-3 sum over the broadcast axis. Shared
    by both passes so :func:`collapse_upstream_mixture_means` cannot collapse one side by a
    different rule than the other.
    """
    return np.resize(np.atleast_1d(_numpy(mean)).astype(float), n_channels)


def _capped(kwargs: dict[str, Any]) -> dict[str, Any]:
    """``kwargs`` with ``niter`` rewritten by the smoke gate or the iteration scale.

    The gate wins: it exists to answer one question in a minute, and a scaled-up run is the
    opposite of that.
    """
    niter = smoke_niter()
    if niter is not None:
        capped = dict(kwargs)
        capped["niter"] = niter
        # A diffeomorphic start beyond the cap would leave the velocity field untouched, so the
        # gate would pass while comparing affines only -- exactly the half that was already fine.
        if int(capped.get("diffeo_start") or 0) >= niter:
            capped["diffeo_start"] = 0
        return capped

    scale = niter_scale()
    if scale is None or kwargs.get("niter") is None:
        return kwargs
    scaled = dict(kwargs)
    scaled["niter"] = max(1, int(round(int(kwargs["niter"]) * scale)))
    if kwargs.get("diffeo_start"):
        scaled["diffeo_start"] = max(0, int(round(int(kwargs["diffeo_start"]) * scale)))
    return scaled


def _completed_kwargs(converted: dict[str, Any], accepted: frozenset[str]) -> dict[str, Any]:
    """``converted``, plus upstream's default for every solver keyword it omits.

    Every keyword goes on the call explicitly, so the entry point's own defaults never apply.
    That is the whole point: see :func:`upstream_solver_defaults`.
    """
    filled = {name: value for name, value in upstream_solver_defaults().items() if name in accepted}
    filled.update({key: value for key, value in converted.items() if key in accepted})
    return filled


def _result_dict(fit: Any) -> dict[str, Any]:
    """A ``Stalign2DResult``/``Stalign3DResult`` under upstream's own ``LDDMM`` key names.

    The notebook's remaining cells read ``A``, ``v``, ``xv`` and the mixture weights off
    whatever the fit returned, so the port's result is renamed to those rather than the
    cells being rewritten. ``affine`` is already in the solver's array order, which is the
    order upstream's ``A`` is in -- ``affine_xyz`` would silently transpose the meaning.
    """
    return {
        "A": fit.affine,
        "v": fit.velocity,
        "xv": fit.velocity_grid,
        "WM": fit.match_weights,
        "WA": fit.artifact_weights,
        "WB": fit.background_weights,
    }


def landmark_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""The landmark affine taking ``source`` onto ``target``, through the public API.

    ``align_landmarks`` is the public route, and it takes containers rather than arrays, so
    the two point sets are wrapped as one-column ``AnnData``\\ s with the landmarks in
    ``obsm``. Points are stored verbatim, so unlike an image's axes nothing is reconstructed
    and nothing is lost.

    Two details decide whether this measures what ledger row D7 says it does:

    *Direction.* ``align_landmarks`` solves ``estimate_transform("affine", src=query,
    dst=ref)``, so its matrix maps *query* onto *ref*. To get ``source -> target`` the
    ``source`` points go in as the query and ``target`` as the reference. Passing them the
    other way round would return the inverse map and quietly halve D7's residual.

    *Degenerate input.* ``align_landmarks`` raises below three pairs, where the estimator the
    STalign path uses internally falls back to a pure centroid shift. The fallback is kept
    here, so replacing the estimator does not also change the behaviour under three
    landmarks -- upstream's ``L_T_from_points`` has no such floor either.
    """
    import anndata as ad
    from squidpy.experimental.tl import align_landmarks

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f"Expected matching landmark shapes, found {source.shape} and {target.shape}.")
    if source.shape[0] < 3:
        return np.eye(2, dtype=float), np.mean(target, axis=0) - np.mean(source, axis=0)

    key = "landmarks"

    def wrap(points: np.ndarray) -> Any:
        adata = ad.AnnData(np.empty((points.shape[0], 0), dtype=float))
        adata.obsm[key] = points
        return adata

    matrix = np.asarray(align_landmarks(wrap(target), wrap(source), landmark_key=key, fit="affine"), dtype=float)
    return matrix[:2, :2], matrix[:2, 2]


def _reversed_points(points: Any) -> np.ndarray | None:
    """Upstream's row-col landmarks as the ``(x, y)`` the public API takes.

    Same boundary as :func:`_initial_affine_xy` and the same omission: upstream's ``pointsI``
    and ``pointsJ`` are row-col -- the reference generator passes ``landmarks_query_rc`` --
    while ``landmarks_ref``/``landmarks_query`` are public coordinates and so are ``(x, y)``,
    which ``_initial_affine_and_landmarks`` reverses on the way in. Measured on the image
    fixture with five landmarks: reversed here agrees with upstream to **3.3e-12**, left
    row-col it is **0.37** away.
    """
    return None if points is None else np.asarray(points, dtype=float)[:, ::-1]


def _initial_affine_xy(converted: dict[str, Any]) -> np.ndarray:
    """The notebook's starting affine as one homogeneous matrix, in ``(x, y)`` order.

    ``_convert_kwargs`` splits upstream's ``A`` (or its ``L``/``T`` pair) into a linear part
    and a translation because the kernel took them separately. The public API takes one
    ``initial_affine``, so they are recombined here.

    The order matters and used to be wrong. Upstream works in row-col ``(y, x)``, while
    ``initial_affine`` is a *public* coordinate and so is ``(x, y)`` --
    ``_initial_affine_and_landmarks`` calls ``affine_xy_to_rc`` on it itself. Handing it a
    row-col matrix converted it a second time; measured on the image fixture, that alone put
    the fitted affine **0.53** relative away from upstream, where the corrected order agrees
    to **1.7e-12**. Reversing the *spatial* axes is a permutation of the first two rows and
    columns, not a plain ``[::-1, ::-1]``, which would move the homogeneous row too -- the
    same reasoning as :func:`_initial_affine_xyz`, which had it right at rank 3.
    """
    linear = np.asarray(converted.pop("L"), dtype=float)
    translation = np.asarray(converted.pop("T"), dtype=float)
    affine = np.eye(linear.shape[0] + 1, dtype=float)
    affine[: linear.shape[0], : linear.shape[1]] = linear
    affine[: translation.size, -1] = translation
    swap = np.eye(affine.shape[0])[[1, 0, 2]]
    return swap @ affine @ swap


def _require_same_cells_ran(notebook: str, upstream_pass: _ReplayPass, squidpy_pass: _ReplayPass) -> str | None:
    """Refuse to compare two passes that did not execute the same notebook.

    Cells skipped for dependency drift are fine as long as *both* passes skip them -- the
    comparison is still like for like. A cell that runs under one fit and not the other means
    the two notebooks diverged, and pairing their figures would be meaningless.
    """
    upstream_skipped = [cell for cell, _ in upstream_pass.skipped]
    squidpy_skipped = [cell for cell, _ in squidpy_pass.skipped]
    if upstream_skipped != squidpy_skipped:
        raise RuntimeError(
            f"{notebook}: the passes skipped different cells "
            f"(upstream {upstream_pass.skipped}, Squidpy {squidpy_pass.skipped}); "
            "they did not run the same notebook and cannot be compared"
        )
    if not upstream_skipped:
        return None
    reasons = "; ".join(f"cell {cell}: {reason}" for cell, reason in upstream_pass.skipped)
    return f"Both passes skipped {len(upstream_skipped)} cell(s) unrelated to the fit -- {reasons}"


def _compose_pair(upstream_png: bytes, squidpy_png: bytes, title: str):
    """Put the same notebook figure from both passes side by side, labelled."""
    import matplotlib.pyplot as plt
    from PIL import Image

    left, right = Image.open(io.BytesIO(upstream_png)), Image.open(io.BytesIO(squidpy_png))
    width = (left.width + right.width) / EMBED_DPI
    height = max(left.height, right.height) / EMBED_DPI
    figure, axes = plt.subplots(1, 2, figsize=(width, height * 1.08), constrained_layout=True)
    for ax, image, label in zip(axes, (left, right), ("upstream PyTorch", "Squidpy JAX"), strict=True):
        ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    figure.suptitle(title, fontsize=10)
    return figure


def _torch_fit(result: dict[str, Any]) -> _FitResult:
    """Squidpy's fit in the shape -- and on the device -- the notebook's remaining cells expect."""
    import torch

    def tensor(value: Any) -> Any:
        # Same device as upstream's results get; see `_on_default_device`.
        return torch.as_tensor(_numpy(value), dtype=torch.float64, device=torch.device(_fit_device()))

    return _FitResult(
        A=tensor(result["A"]),
        v=tensor(result["v"]),
        xv=[tensor(axis) for axis in result["xv"]],
        WM=tensor(result["WM"]),
        WB=tensor(result["WB"]),
        WA=tensor(result["WA"]),
    )


def _convergence_metrics(fit: Any) -> dict[str, float]:
    """How much descending the fit had left to do when its iteration cap stopped it.

    A divergence only means something once both sides have converged; a fit still dropping
    its objective at the last iteration is reporting where it ran out of iterations, not
    where the optimum is. `starmap-allen3Datlas-alignment` asks for `niter=800` and its
    aligned atlas slice looks visibly loose against the section, which is the question this
    answers with a number instead of an eye.

    ``energies``/``n_iter`` are fields of the public result, so nothing here reaches into
    squidpy's internals. Upstream keeps its own trace local -- ``Esave`` is plotted and never
    returned -- so there is no upstream column to compare against; the ledger's rank-3 row
    pins the port's objective to the original's at 4e-16 across the whole trace, which is
    what makes the port's trace usable as a stand-in.
    """
    if getattr(fit, "energies", None) is None or getattr(fit, "n_iter", None) is None:
        return {}
    energies = np.asarray(fit.energies, dtype=float)[: int(fit.n_iter)]
    energies = energies[np.isfinite(energies)]
    if energies.size < 20:
        return {}
    tail = max(1, energies.size // 10)
    scale = max(abs(float(energies[0])), np.finfo(float).tiny)
    return {
        "energy first": float(energies[0]),
        "energy last": float(energies[-1]),
        # Total progress, and how much of it was still arriving in the final tenth. A tail
        # share near zero is converged; comparable to the total means the cap bound the fit.
        "energy total drop fraction": float((energies[0] - energies[-1]) / scale),
        "energy last-tenth drop fraction": float((energies[-tail] - energies[-1]) / scale),
        "energy iterations": float(energies.size),
    }


def _convergence_figure(notebook: str, energies: Any, metrics: Mapping[str, float]) -> Any:
    """The port's objective per iteration, as a panel the notebook carries with its results.

    Reading a divergence without this is guesswork: a curve still dropping at the right-hand
    edge says the iteration cap stopped the fit, and any disagreement downstream is partly a
    disagreement about where two runs happened to stop. `starmap-allen3Datlas-alignment`
    prompted it -- `niter=800`, and an aligned atlas slice that sits visibly loose against the
    section.

    One curve, not two. Upstream computes the same trace but keeps it local (``Esave``, which
    it plots and does not return), so there is no second series to draw; the ledger's rank-3
    row pins the two objectives to 4e-16 across the whole trace, which is what lets one curve
    stand for both. Log-scaled because the drop is orders of magnitude, and annotated with the
    share of the total descent that arrived in the final tenth.
    """
    import matplotlib.pyplot as plt

    trace = np.asarray(energies, dtype=float)
    trace = trace[np.isfinite(trace)]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(np.arange(1, trace.size + 1), trace, lw=1.2)
    ax.set_yscale("log" if np.all(trace > 0) else "linear")
    ax.set_xlabel("iteration")
    ax.set_ylabel("objective (squidpy)")
    tail = metrics.get("energy last-tenth drop fraction")
    total = metrics.get("energy total drop fraction")
    caption = f"{notebook} — {trace.size} iterations"
    if tail is not None and total is not None:
        caption += f", total drop {total:.1%} of the initial objective, final tenth {tail:.2%}"
    ax.set_title(caption, fontsize=10)
    return fig


def _append_convergence_panel(result: ComparisonResult, metrics: Mapping[str, float], traces: list[Any]) -> None:
    """Add the convergence panel to a completed comparison, if the fit reported a trace.

    Appended rather than paired: `figures` normally holds upstream-beside-port composites, and
    this is one curve because only one side returns its trace. `figure_sources` gets an empty
    entry so the written notebook keeps its cells and figures in step.
    """
    trace = next((t for t in traces if t is not None), None)
    if trace is None or not metrics:
        return
    result.figures.append(_convergence_figure(result.notebook, trace, metrics))
    result.figure_sources.append("")


def _pin_fit_determinism() -> None:
    """Ask both backends for deterministic kernels before either fit runs.

    Rank 2 reproduces to ~1e-12 across runs, so this was never needed there. Rank 3 does not:
    rerunning the two ``allen3Datlas`` notebooks with identical code moved cells by up to
    **109 um** and reassigned **15-22 %** of Allen regions *on the upstream side alone* --
    measured between jobs 39746531 and 39747874, same pinned upstream, one of them on the same
    host. A 50 um annotation volume cannot be read reproducibly through a fit that wanders one
    to two voxels, which makes every rank-3 number indicative rather than measured.

    ``warn_only=True`` on purpose. Raising would abort the sweep on the first op lacking a
    deterministic kernel and tell us nothing; warning names the op in the job log while still
    taking the deterministic path everywhere one exists. The environment half --
    ``CUBLAS_WORKSPACE_CONFIG`` for cuBLAS and ``--xla_gpu_deterministic_ops`` for XLA -- has
    to be set before the libraries load, so it lives in the launcher rather than here.
    """
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False  # no autotuning, so the algorithm cannot vary
    torch.manual_seed(0)


def _capture_fit_arguments(notebook: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Dump one notebook's real fit arguments when ``STALIGN_CAPTURE_DIR`` is set.

    A divergence that appears only on a notebook's own inputs cannot be chased from the
    synthetic fixtures. Three attempts at rebuilding `xenium-heimage`'s regime by hand each
    diverged for reasons of their own -- mis-scaled sigmas against a density raster, then an
    effectively unregularised fit -- and agreed at every commit, which measures the
    reconstruction rather than the port. The replay already holds the true arguments at the
    substitution site, so capturing them costs one run and ends the guessing.

    Raw ``kwargs``, deliberately before :func:`_capped`, so a fast ``STALIGN_SMOKE_NITER=1``
    capture still records the notebook's real ``niter``. ``None`` is dropped rather than
    stored: ``np.savez`` would need pickling for it, and an absent key reads back as ``None``.
    """
    directory = os.environ.get("STALIGN_CAPTURE_DIR")
    if not directory:
        return
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for index, value in enumerate(args):
        if isinstance(value, (list, tuple)):  # the axis pairs, `xI` and `xJ`
            for axis, item in enumerate(value):
                payload[f"arg{index}_{axis}"] = _numpy(item)
        else:
            payload[f"arg{index}"] = _numpy(value)
    for key, value in kwargs.items():
        if value is None or key == "device":
            continue
        payload[f"kw_{key}"] = _numpy(value) if hasattr(value, "shape") else np.asarray(value)
    np.savez_compressed(out / f"{Path(notebook).stem}-fit-args.npz", **payload)


def _compare_lddmm(notebook: str) -> ComparisonResult:
    """Run the notebook twice -- upstream's fit, then Squidpy's -- and pair up its figures.

    Only the fit differs. Every figure below it is drawn by the notebook's own cells calling
    upstream's `build_transform` / `transform_image_*` / `transform_points_*`, so a visible
    difference is attributable to `A`, `v` and `xv` alone.
    """
    import torch
    from squidpy.experimental.tl import align_stalign_image

    device = _fit_device()
    torch.set_default_dtype(torch.float64)
    _pin_fit_determinism()
    convergence: dict[str, float] = {}
    traces: list[Any] = []

    def upstream_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        _capture_fit_arguments(notebook, args, kwargs)
        call = _capped(kwargs)
        call["device"] = device
        for key in ("muA", "muB"):
            if call.get(key) is not None:
                call[key] = torch.as_tensor(call[key], device=device, dtype=torch.float64)
        result = original(*args, **call)
        _await_fit_device()  # same rule as the port's `block_until_ready`; see `_await_fit_device`
        return _FitResult({key: _on_fit_device(value) for key, value in result.items()})

    def squidpy_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        import jax

        source_axes = tuple(_numpy(axis) for axis in args[0])  # upstream's `xI`: the moving image
        target_axes = tuple(_numpy(axis) for axis in args[2])  # upstream's `xJ`: the fixed image
        converted = _convert_kwargs(_capped(kwargs), PORT_PARAMETERS)
        # `query` is the moving side at rank 2 -- `_stalign.py` reads `source_image =
        # as_chw(query)`, and `_initial_affine_and_landmarks` sends `landmarks_query` to the
        # source. So upstream's `I`/`pointsI` go in as the *query* and `J`/`pointsJ` as the
        # reference. This was inverted, and rank 3 is the opposite convention (there the
        # volume is `ref` and does move), which is presumably how it survived: the two
        # mirrored mistakes nearly cancel whenever `I` and `J` are similar-sized rasters on
        # centred pixel axes, which is fourteen of the seventeen notebooks. They do not
        # cancel for `xenium-heimage-alignment`, whose H&E is in pixels against a density in
        # microns -- there the velocity grid came out 48x66 against upstream's 17x23.
        landmarks_query = _reversed_points(converted.pop("points_source", None))
        landmarks_ref = _reversed_points(converted.pop("points_target", None))
        initial_affine = _initial_affine_xy(converted)
        accepted = solver_keys()
        fit = align_stalign_image(
            as_sdata(_channels_first(args[3], ndim=2), target_axes),
            as_sdata(_channels_first(args[1], ndim=2), source_axes),
            image_key=_IMAGE_KEY,
            landmarks_ref=landmarks_ref,
            landmarks_query=landmarks_query,
            initial_affine=initial_affine,
            **_completed_kwargs(converted, accepted - {"initial_affine"}),
        )
        jax.block_until_ready(fit.affine)
        convergence.update(_convergence_metrics(fit))
        traces.append(getattr(fit, "energies", None))
        return _torch_fit(_result_dict(fit))

    result = _pair_passes(notebook, upstream_fit, squidpy_fit)
    result.metrics.update(convergence)
    _append_convergence_panel(result, convergence, traces)
    return result


#: Where a notebook writes its aligned coordinates, from its own ``to_csv`` call.
_RESULT_TARGET = re.compile(r"""(\w+)\.to_csv\(\s*['"]([^'"]+)['"]""")


def _published_result(notebook: str) -> Path | None:
    """The aligned coordinates upstream committed for this notebook, when they are comparable.

    Upstream ships the output of its own runs beside the notebooks, which is a stronger
    reference than either replay pass: neither pass produced it, so it also says whether the
    replay reproduces upstream's *published* result and not merely upstream's code.

    Only the copies the pinned notebook would actually write are returned. Three of the
    shipped files -- ``starmap_STalign_to_xenium`` and both Xenium ones -- are headerless
    ``%.18e`` dumps with no ``aligned_x``/``aligned_y``, so they predate the ``to_csv`` call
    the notebook now makes and are not this commit's output. ``starmap_STalign_to_xenium.csv``
    and its ``.csv.gz`` disagree with each other, which is the same story from the other side.
    ``heart-alignment`` writes a file upstream never committed at all.
    """
    # Read the notebook directly: `_notebook_cells` drops the `to_csv` cell, since writing
    # results back into the vendored checkout is not part of the replay.
    path = upstream.vendor_root() / "docs" / "notebooks" / notebook
    for cell in json.loads(path.read_text())["cells"]:
        match = _RESULT_TARGET.search("".join(cell.get("source", [])))
        if not match:
            continue
        target = (path.parent / match.group(2)).resolve()
        return target if target.exists() else None
    return None


def _published_metrics(notebook: str, upstream_ns: dict[str, Any], squidpy_ns: dict[str, Any]) -> dict[str, float]:
    """Both passes' aligned coordinates against the copy upstream published."""
    target = _published_result(notebook)
    if target is None:
        return {}
    import pandas as pd

    published = pd.read_csv(target)
    columns = ["aligned_x", "aligned_y"]
    if not set(columns) <= set(published.columns):
        return {}
    expected = published[columns].to_numpy(dtype=float)

    metrics: dict[str, float] = {}
    for label, namespace in (("upstream", upstream_ns), ("squidpy", squidpy_ns)):
        frame = namespace.get("results")
        if frame is None or not set(columns) <= set(getattr(frame, "columns", ())):
            continue
        actual = frame[columns].to_numpy(dtype=float)
        # A row-count mismatch means the two are not the same cells; reporting a number for it
        # would be worse than reporting none.
        if actual.shape != expected.shape:
            continue
        metrics[f"{label} vs upstream published relative L2"] = _relative_l2(actual, expected)
    return metrics


def _pair_passes(notebook: str, upstream_fit: Any, squidpy_fit: Any, *, function: str = "LDDMM") -> ComparisonResult:
    """Replay ``notebook`` once per fit and pair up the figures and variables.

    Rank-agnostic: the only thing that differs between the 2D and 3D comparisons is which
    upstream function is swapped and what stands in for it.
    """
    import matplotlib.pyplot as plt

    upstream_pass = _replay_notebook(notebook, upstream_fit, function=function)
    plt.close("all")
    squidpy_pass = _replay_notebook(notebook, squidpy_fit, function=function)
    plt.close("all")

    skipped_note = _require_same_cells_ran(notebook, upstream_pass, squidpy_pass)
    metrics = _namespace_metrics(upstream_pass.namespace, squidpy_pass.namespace)
    metrics.update(_published_metrics(notebook, upstream_pass.namespace, squidpy_pass.namespace))
    # Beside `df[coord*]`, which upstream reads off the wrong fifth of each axis -- ledger D14.
    # A no-op for every notebook that has no `df`/`xv`/`v`/`A`, i.e. all fifteen rank-2 ones.
    metrics.update(_corrected_coord_metrics(upstream_pass.namespace, squidpy_pass.namespace))
    cell_source = dict(_notebook_cells(notebook))
    figures, figure_sources, port_figures = [], [], []
    for (cell_number, upstream_png), (_, squidpy_png) in zip(upstream_pass.figures, squidpy_pass.figures, strict=False):
        figures.append(_compose_pair(upstream_png, squidpy_png, f"{notebook} — cell {cell_number}"))
        figure_sources.append(cell_source.get(cell_number, ""))
        port_figures.append(squidpy_png)
    if len(upstream_pass.figures) != len(squidpy_pass.figures):
        raise RuntimeError(
            f"{notebook}: the two passes drew {len(upstream_pass.figures)} and "
            f"{len(squidpy_pass.figures)} figures; they are no longer comparable"
        )

    return ComparisonResult(
        notebook=notebook,
        status="compared",
        metrics=metrics,
        figures=figures,
        figure_sources=figure_sources,
        port_figures=port_figures,
        paired_frames=_paired_frames(upstream_pass.namespace, squidpy_pass.namespace),
        upstream_seconds=upstream_pass.seconds,
        squidpy_seconds=squidpy_pass.seconds,
        note=" ".join(filter(None, (UPSTREAM_NOTES.get(notebook), skipped_note))) or None,
    )


#: Upstream's 3D-to-slice defaults that squidpy's rank-3 estimator names differently or
#: does not express. `device`/`dtype` have no JAX equivalent; `sigmaP` is dropped because
#: upstream's 3D loop has no point term at all (its `EP` block is commented out at
#: STalign.py:1505-1509), so carrying it would advertise a knob that does nothing.
_SLICE_DROPPED = frozenset({"device", "dtype", "sigmaP", "pointsI", "pointsJ", "A", "L", "T", "v", "xv"})


#: Directory the Allen atlas is cached in, so repeated passes do not re-download it.
ATLAS_CACHE_ENV = "STALIGN_ATLAS_CACHE"


class _CachedResponse:
    """The parts of a ``requests`` response upstream's two download helpers actually touch."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def text(self) -> str:
        return self._path.read_text()

    @property
    def content(self) -> bytes:
        return self._path.read_bytes()

    def iter_content(self, chunk_size: int = 1024):
        with self._path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                yield chunk


class _AtlasCache:
    """Stands in for STalign's module-level ``requests``, serving atlas URLs off disk.

    Both atlas helpers re-fetch unconditionally -- ``download_aba_image_labels``
    (STalign.py:1885) and ``download_aba_ontology`` (:1846) each open their target ``'wb'``/
    ``'w'`` with no existence check -- and :func:`_replay_directory` hands every pass a fresh
    working directory. A 3D comparison is two passes per notebook, so the ~40 MB atlas would
    be pulled four times per notebook and eight times for the pair.

    Wrapping ``requests`` rather than either helper is what keeps upstream's own code intact:
    the ontology helper parses the CSV it downloads in the same breath, so short-circuiting
    the function would mean duplicating that parse here. Substituting the transport leaves
    every line of upstream's parsing untouched and unaware.

    Scoped deliberately to the STalign module's own attribute rather than the shared
    ``requests`` module, so nothing else in the process is affected.
    """

    def __init__(self, requests_module: Any, cache_dir: Path) -> None:
        self._requests = requests_module
        self._cache_dir = cache_dir

    def __getattr__(self, name: str) -> Any:
        return getattr(self._requests, name)

    def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
        target = self._cache_dir / (Path(urlparse(url).path).name or "download")
        if not (target.exists() and target.stat().st_size):
            kwargs["stream"] = True
            response = self._requests.get(url, *args, **kwargs)
            response.raise_for_status()
            # Staged then renamed: a job killed mid-download leaves no half-file behind that
            # a later run would mistake for a warm cache entry.
            staging = target.with_name(target.name + ".partial")
            with staging.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
            staging.replace(target)
        return _CachedResponse(target)


def _patch_atlas_downloads(st: Any) -> None:
    """Route upstream's atlas downloads through :envvar:`STALIGN_ATLAS_CACHE`, if set.

    A no-op when unset, so a plain local run keeps upstream's own download behaviour.
    """
    cache = os.environ.get(ATLAS_CACHE_ENV)
    if not cache:
        return
    cache_dir = Path(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(st.requests, _AtlasCache):
        st.requests = _AtlasCache(st.requests, cache_dir)


#: Region -> colour, fixed by whichever pass runs first so both sides draw a region alike.
_REGION_COLOURS: dict[str, Any] = {}


def _patch_region_colours(st: Any) -> None:
    """Give ``plot_brain_regions`` a region-keyed palette instead of a position-keyed one.

    Upstream calls ``ax.scatter`` once per region with no colour argument
    (STalign.py:2029-2033), so a region's colour is its index in ``np.unique(df['acronym'])``
    modulo the ten-entry property cycle. The two passes assign a handful of boundary cells to
    different regions, which changes that list's length -- and every alphabetically later
    region then shifts colour. The pair ends up looking wholly different for a reason that has
    nothing to do with the fit, which is the one thing it exists to isolate.

    The map is fixed by the pass that runs first, which is upstream's. Its own figure is
    therefore unchanged -- those colours already *were* its own indexing, so this also keeps
    the panel comparable with upstream's published one -- and squidpy's is drawn with the same
    region -> colour assignment. Regions only squidpy assigns continue the cycle past
    upstream's list, so a real difference still shows up as a new colour rather than silently
    borrowing an existing one.

    This is the one place the replay does not run upstream's plotting verbatim. It is applied
    identically to both passes, and it changes only the palette: same points, same axes, same
    legend order.
    """
    import matplotlib.pyplot as plt

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def plot_brain_regions(df: Any) -> None:
        regions = list(np.unique(df["acronym"]))
        for region in regions:
            if region not in _REGION_COLOURS:
                _REGION_COLOURS[region] = cycle[len(_REGION_COLOURS) % len(cycle)]
        _, ax = plt.subplots()
        for region in regions:
            subset = df[df["acronym"] == region]
            ax.scatter(subset["x"], subset["y"], label=region, s=0.1, color=_REGION_COLOURS[region])
            ax.legend()

    st.plot_brain_regions = plot_brain_regions


def _compare_lddmm_3d(notebook: str) -> ComparisonResult:
    """The volume-to-section notebooks, replayed around ``LDDMM_3D_to_slice``.

    Same shape as :func:`_compare_lddmm`: only the fit is swapped, and every figure below it
    is the notebook's own code reading `A`, `v`, `xv` and `Xs` off the result -- including
    `analyze3Dalign`, which is what turns the fit into per-cell atlas coordinates.

    Expect a *real* numerical difference here rather than agreement to 1e-15. Upstream's 3D
    regularisation energy transforms two of the three spatial axes while smoothing its own
    gradient over all three (STalign.py:1504 vs :1527), so the two implementations descend on
    different objectives once the velocity moves; squidpy uses every spatial axis in both
    places. See `docs/STALIGN_DIVERGENCES.md` row D11 -- the metrics below quantify that
    choice, they do not fail on it.
    """
    import torch
    from squidpy.experimental.tl import align_stalign_volume

    device = _fit_device()
    torch.set_default_dtype(torch.float64)
    _pin_fit_determinism()
    convergence: dict[str, float] = {}
    traces: list[Any] = []
    _REGION_COLOURS.clear()  # per notebook, so one notebook's regions cannot colour another's
    _patch_atlas_downloads(upstream.load())
    _patch_region_colours(upstream.load())

    def upstream_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        call = _capped(kwargs)
        call["device"] = device
        n_channels = _section_channels(args[3])
        for key in ("muA", "muB"):
            if call.get(key) is not None:
                mean = call[key]
                if collapse_upstream_mixture_means():
                    mean = _collapsed_mean(mean, n_channels)
                call[key] = torch.as_tensor(mean, device=device, dtype=torch.float64)
        result = original(*args, **call)
        _await_fit_device()  # same rule as the port's `block_until_ready`; see `_await_fit_device`
        return _FitResult({key: _on_fit_device(value) for key, value in result.items()})

    def squidpy_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        import jax

        reference_axes = [_numpy(axis) for axis in args[0]]
        section_axes = [_numpy(axis) for axis in args[2]]
        # Upstream's own rank-3 defaults underneath the notebook's kwargs -- see
        # `_slice_forwarded`. Without this the port silently uses squidpy's `_VOLUME_DEFAULTS`
        # for anything the notebook omits, and any retuning there contaminates the parity
        # numbers instead of showing up as a documented divergence.
        forwarded = _slice_forwarded(kwargs)
        # The notebooks pass `muA=[3,3,3]` against a single-channel section and rely on
        # upstream broadcasting three identical terms -- which quietly makes its artifact
        # scale `sigmaA/sqrt(3)`. squidpy validates the length, so the mean is collapsed to
        # one entry per section channel here. What that costs is ledger row D13, measured by
        # `STALIGN_D13_COLLAPSE_MEANS` -- see `collapse_upstream_mixture_means`.
        section = np.asarray(_numpy(args[3]))
        n_channels = _section_channels(section)
        for key in ("muA", "muB"):
            if kwargs.get(key) is not None:
                forwarded[key] = _collapsed_mean(kwargs[key], n_channels)

        fit = align_stalign_volume(
            as_sdata(_channels_first(args[1], ndim=3), reference_axes),
            as_sdata(_channels_first(section, ndim=2), section_axes),
            image_key=_IMAGE_KEY,
            initial_affine=_initial_affine_xyz(kwargs),
            **{key: value for key, value in forwarded.items() if key in solver_keys()},
        )
        jax.block_until_ready(fit.affine)
        convergence.update(_convergence_metrics(fit))
        traces.append(getattr(fit, "energies", None))
        return _torch_slice_fit(fit, section_axes)

    result = _pair_passes(notebook, upstream_fit, squidpy_fit, function="LDDMM_3D_to_slice")
    result.metrics.update(convergence)
    _append_convergence_panel(result, convergence, traces)
    return result


def upstream_slice_defaults() -> dict[str, Any]:
    """Upstream's own ``LDDMM_3D_to_slice`` defaults, read off its signature.

    The rank-2 path fills every keyword a notebook omits from
    :func:`upstream_solver_defaults`, so both passes start from identical parameters wherever
    the notebook is silent. Rank 3 had no equivalent: it forwarded only what the notebook
    passed and let squidpy's own ``_VOLUME_DEFAULTS`` supply the rest.

    That was invisible for exactly as long as those defaults matched upstream's, and it stops
    being invisible the moment squidpy retunes one of them for rank 3 -- the comparison would
    then run the port on a different fit and publish the gap as a port divergence. Which is
    the ``_IMAGE_DEFAULTS`` mistake that :func:`upstream_solver_defaults` exists to prevent,
    one rank up. Reading them from upstream keeps the pin no matter what squidpy chooses.

    Note these are *not* ``LDDMM``'s: upstream's 3D entry point declares its own
    ``expand=1.25``, ``epL=1e-6``, ``epT=1e1``, ``epV=1e3``, ``sigmaR=1e8``.
    """
    st = upstream.load()
    return {
        name: parameter.default
        for name, parameter in inspect.signature(st.LDDMM_3D_to_slice).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _slice_forwarded(captured: dict[str, Any]) -> dict[str, Any]:
    """What the port is handed for a rank-3 replay: upstream's defaults, then the notebook's own.

    Order matters and is the whole point -- a keyword the notebook passes must win, and a
    keyword it omits must come from upstream rather than from squidpy. Kept as a function
    rather than inlined so the precedence is testable without a GPU.
    """
    defaults = {
        key: value
        for key, value in upstream_slice_defaults().items()
        if key not in _SLICE_DROPPED and key in PORT_DEFAULTS and value is not None
    }
    forwarded = {
        key: value
        for key, value in {**defaults, **_capped(captured)}.items()
        if key not in _SLICE_DROPPED and key in PORT_DEFAULTS and value is not None
    }
    return {key: _cast_like(value, PORT_DEFAULTS[key], name=key) for key, value in forwarded.items()}


def _initial_affine_xyz(kwargs: dict[str, Any]) -> np.ndarray | None:
    """The notebook's starting affine, converted to the `(x, y, z)` order squidpy takes.

    Upstream accepts either a whole `A` or an `L`/`T` pair, both in `(z, y, x)`. Reversing
    the *spatial* axes only is a permutation of the first three rows and columns -- not a
    plain `[::-1, ::-1]`, which would move the homogeneous row as well.
    """
    if kwargs.get("A") is not None:
        affine = _numpy(kwargs["A"])
    else:
        linear, translation = kwargs.get("L"), kwargs.get("T")
        if linear is None and translation is None:
            return None
        affine = np.eye(4)
        affine[:3, :3] = np.eye(3) if linear is None else _numpy(linear)
        affine[:3, 3] = np.zeros(3) if translation is None else _numpy(translation)
    swap = np.eye(4)[[2, 1, 0, 3]]
    return swap @ np.asarray(affine, dtype=float) @ swap


def _torch_slice_fit(fit: Any, section_axes: list[np.ndarray]) -> _FitResult:
    """Squidpy's rank-3 fit in the shape the 3D notebooks' remaining cells expect.

    They read `A`, `v`, `xv` and `Xs` (cell [33]). `Xs` is not stored on the result -- it is
    the backward grid, recomputed here from the fitted deformation rather than kept around,
    since upstream only returns it as a by-product of its final iteration.
    """
    import torch

    def tensor(value: Any) -> Any:
        return torch.as_tensor(_numpy(value), dtype=torch.float64, device=torch.device(_fit_device()))

    # `deformation_grid` applies the single-sample z lift itself and rejects a pre-lifted
    # grid, so the section's own two axes go in. Its docstring pins that this is the same
    # call on the same fitted state the objective samples through, not a plotting estimate.
    grid = fit.deformation_grid(
        direction="backward",
        query_axes=tuple(np.asarray(axis, dtype=float) for axis in section_axes),
    )
    return _FitResult(
        A=tensor(fit.affine),
        v=tensor(fit.velocity),
        xv=[tensor(axis) for axis in fit.velocity_grid],
        WM=tensor(fit.match_weights),
        WB=tensor(fit.background_weights),
        WA=tensor(fit.artifact_weights),
        # Upstream stores `Xs` as `(..., component)`; squidpy returns component-first.
        Xs=tensor(np.moveaxis(_numpy(grid), 0, -1)),
    )


def _compare_affine(notebook: str) -> ComparisonResult:
    """The landmark-affine notebook, replayed the same way but around ``L_T_from_points``.

    This notebook never reaches ``LDDMM``; its plots come from the affine that
    ``L_T_from_points`` returns, so that is the call the two passes swap.
    """
    import matplotlib.pyplot as plt

    residuals: dict[str, float] = {}

    def record(label: str, args: tuple[Any, ...], linear: Any, translation: Any) -> None:
        source, target = _numpy(args[0]), _numpy(args[1])
        aligned = source @ np.asarray(linear).T + np.asarray(translation)
        residuals[f"{label} landmark residual"] = float(np.linalg.norm(aligned - target))

    def upstream_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        linear, translation = original(*args, **kwargs)
        record("upstream", args, linear, translation)
        return linear, translation

    def squidpy_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        linear, translation = landmark_affine(_numpy(args[0]), _numpy(args[1]))
        record("squidpy", args, linear, translation)
        return np.asarray(linear), np.asarray(translation)

    upstream_pass = _replay_notebook(notebook, upstream_fit, function="L_T_from_points")
    plt.close("all")
    squidpy_pass = _replay_notebook(notebook, squidpy_fit, function="L_T_from_points")
    plt.close("all")

    skipped_note = _require_same_cells_ran(notebook, upstream_pass, squidpy_pass)
    metrics = {**_namespace_metrics(upstream_pass.namespace, squidpy_pass.namespace), **residuals}
    cell_source = dict(_notebook_cells(notebook))
    figure_sources = [cell_source.get(cell_number, "") for cell_number, _ in upstream_pass.figures]
    figures = [
        _compose_pair(upstream_png, squidpy_png, f"{notebook} — cell {cell_number}")
        for (cell_number, upstream_png), (_, squidpy_png) in zip(
            upstream_pass.figures, squidpy_pass.figures, strict=True
        )
    ]
    note = " ".join(filter(None, (UPSTREAM_NOTES.get(notebook), skipped_note))) or None
    return ComparisonResult(
        notebook,
        "compared-affine",
        metrics,
        figures,
        note=note,
        figure_sources=figure_sources,
        port_figures=[png for _, png in squidpy_pass.figures],
    )


def _status_panel(notebook: str, status: str, note: str) -> ComparisonResult:
    """Emit a panel that says why a notebook carries no numbers, instead of no evidence."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, notebook, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.38, note, ha="center", va="center", fontsize=11, wrap=True)
    return ComparisonResult(notebook, status, {}, [fig], note=note)


def notebook_for_index(index: int) -> str:
    """Resolve a Slurm array task id to the notebook it is responsible for.

    Raises
    ------
    IndexError
        If the array was submitted with a wider range than there are notebooks -- better
        a named failure than a task that silently compares the wrong notebook.
    """
    if not 0 <= index < len(NOTEBOOKS):
        raise IndexError(f"notebook index {index} is outside 0-{len(NOTEBOOKS) - 1}")
    return NOTEBOOKS[index]


def compare_notebook(notebook: str) -> ComparisonResult:
    """Replay and compare one notebook from the pinned upstream checkout."""
    if notebook not in NOTEBOOKS:
        raise ValueError(f"Unknown upstream notebook {notebook!r}.")
    if notebook in UNREPLAYABLE_NOTEBOOKS:
        return _status_panel(notebook, "unreplayable-upstream", UNREPLAYABLE_NOTEBOOKS[notebook])
    if notebook in THREE_D_NOTEBOOKS:
        return _compare_lddmm_3d(notebook)
    if notebook == AFFINE_NOTEBOOK:
        return _compare_affine(notebook)
    return _compare_lddmm(notebook)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _manifest(
    notebook: str,
    status: str,
    note: str | None,
    upstream_seconds: float = 0.0,
    squidpy_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "notebook": notebook,
        "status": status,
        "note": note,
        "upstream_seconds": upstream_seconds,
        "squidpy_seconds": squidpy_seconds,
        "upstream_sha": upstream.UPSTREAM_SHA,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in ("squidpy", "squidpy-ports", "jax", "jaxlib", "torch")},
        "squidpy_commit": upstream.squidpy_commit(),
        # Recorded on every notebook, not only the two it can affect: a manifest without the
        # key would be indistinguishable from one written before the flag existed.
        "d13_collapse_means": collapse_upstream_mixture_means(),
        # Comparison-only flags that change *what was compared* rather than how it was run.
        # A result must not be separable from the condition it ran under, so both are recorded
        # on every notebook whether or not it can be affected.
        "d11_upstream_reg_energy_axes": bool(os.environ.get("SQUIDPY_STALIGN_UPSTREAM_REG_ENERGY_AXES")),
    }


def _header_cell(notebook: str, status: str, note: str | None) -> dict[str, Any]:
    upstream_url = f"{upstream.UPSTREAM_URL}/blob/{upstream.UPSTREAM_SHA}/docs/notebooks/{notebook}"
    return {
        "cell_type": "markdown",
        # nbformat 4.5 requires a cell id; a fixed one keeps re-runs diffable.
        "id": "header",
        "metadata": {},
        "source": [
            f"# STalign parity: `{notebook}`\n",
            "\n",
            f"Executed one-to-one replay of [the pinned upstream notebook]({upstream_url}).\n",
            "\n",
            f"Status: `{status}`. {note or ''}\n",
        ],
    }


def _compare_source(notebook: str, tail: str) -> list[str]:
    return [
        "from squidpy_ports.stalign.notebook_suite import compare_notebook\n",
        f"result = compare_notebook({notebook!r})\n",
        tail,
    ]


def _embedded_panel(figure: Any) -> str:
    """Base64 PNG of the panel, sized for a documentation page rather than for an archive.

    The executed notebooks are committed and rendered by the docs, so their embedded copy is
    what drives repository size: a 20x9.5in panel at :data:`ARCHIVE_DPI` base64s to ~2 MB, so
    a 17-notebook suite would add ~32 MB per rerun. Dropping to :data:`EMBED_DPI` and a
    256-colour palette cuts that ~6x with no visible difference on these plots, while the
    full-resolution PNG stays beside the notebook in the run's evidence.
    """
    from PIL import Image

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=EMBED_DPI, bbox_inches="tight")
    buffer.seek(0)
    panel = Image.open(buffer).convert("RGB")
    reduced = panel.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    encoded = io.BytesIO()
    reduced.save(encoded, format="PNG", optimize=True)
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def _write_notebook(payload: dict[str, Any], manifest: dict[str, Any], output_dir: Path) -> None:
    payload["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
        "stalign_comparison": manifest,
    }
    payload["nbformat"], payload["nbformat_minor"] = 4, 5
    notebook_dir = output_dir / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    (notebook_dir / manifest["notebook"]).write_text(json.dumps(payload, indent=1) + "\n")


def write_result(result: ComparisonResult, output_dir: Path) -> None:
    """Persist every paired figure, the metrics, and a provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.notebook).stem
    for index, figure in enumerate(result.figures):
        suffix = "" if index == 0 else f"-{index}"
        figure.savefig(output_dir / f"{stem}-comparison{suffix}.png", dpi=ARCHIVE_DPI, bbox_inches="tight")
    for index, png in enumerate(result.port_figures):
        # Written verbatim: this is the exact PNG the port's own cell drew, before it was
        # rescaled into the pair. Re-encoding it through matplotlib would only lose pixels.
        (output_dir / f"{stem}-port{'' if index == 0 else f'-{index}'}.png").write_bytes(png)
    (output_dir / f"{stem}-metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    for name, frame in result.paired_frames.items():
        # CSV rather than parquet: these are read by hand as often as by code, and the largest
        # is a few MB. Compressed because Lustre would rather have one small file than one big.
        frame.to_csv(output_dir / f"{stem}-cells-{name}.csv.gz", index=False)
    manifest = _manifest(result.notebook, result.status, result.note, result.upstream_seconds, result.squidpy_seconds)
    (output_dir / f"{stem}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    cells = [
        _header_cell(result.notebook, result.status, result.note),
        {
            "cell_type": "code",
            "execution_count": 1,
            "id": "metrics",
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "application/json": result.metrics,
                        "text/plain": [json.dumps(result.metrics, indent=2, sort_keys=True)],
                    },
                    "execution_count": 1,
                    "metadata": {},
                    "output_type": "execute_result",
                }
            ],
            "source": ["# per-variable relative L2: upstream STalign vs the squidpy JAX port\n"],
        },
    ]
    for index, figure in enumerate(result.figures):
        raw = result.figure_sources[index] if index < len(result.figure_sources) else ""
        source = raw.splitlines(keepends=True) if raw.strip() else [f"# upstream vs squidpy — figure {index + 1}\n"]
        cells.append(
            {
                "cell_type": "code",
                "execution_count": index + 2,
                "id": f"figure-{index}",
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "image/png": _embedded_panel(figure),
                            "text/plain": [f"<upstream (left) vs squidpy (right), figure {index + 1}>"],
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": source,
            }
        )
    _write_notebook({"cells": cells}, manifest, output_dir)


def write_failure(notebook: str, error: BaseException, output_dir: Path) -> None:
    """Persist the full traceback of a failed comparison, as text and as a notebook.

    A batch run that dies part way through still has to explain itself. Without this a
    partial result directory holds evidence only for the notebooks that succeeded, and
    the failures survive as the one-line exception message in the suite log -- which is
    lost entirely if the allocation is killed before the log is copied back.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(notebook).stem
    formatted = traceback.format_exception(type(error), error, error.__traceback__)
    (output_dir / f"{stem}-traceback.txt").write_text("".join(formatted))
    manifest = _manifest(notebook, "failed", f"{type(error).__name__}: {error}")
    (output_dir / f"{stem}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    payload = {
        "cells": [
            _header_cell(notebook, "failed", manifest["note"]),
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "traceback",
                "metadata": {},
                "outputs": [
                    {
                        "output_type": "error",
                        "ename": type(error).__name__,
                        "evalue": str(error),
                        "traceback": "".join(formatted).splitlines(),
                    }
                ],
                "source": _compare_source(notebook, "result.metrics\n"),
            },
        ]
    }
    _write_notebook(payload, manifest, output_dir)


def main() -> None:
    """Run one or every mapped upstream notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("notebook", nargs="?", choices=(*NOTEBOOKS, "all"))
    # One Slurm array task per notebook: the task id is the only thing the batch script
    # knows, and resolving it here keeps `NOTEBOOKS` the single source of that ordering.
    selection.add_argument("--index", type=int, help=f"select one notebook by position, 0-{len(NOTEBOOKS) - 1}")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("STALIGN_COMPARISON_OUTPUT", "comparison-results")),
    )
    args = parser.parse_args()

    if args.index is not None:
        selected = (notebook_for_index(args.index),)
    else:
        selected = NOTEBOOKS if args.notebook == "all" else (args.notebook,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Array tasks share one output directory, so the status file is named after what this
    # process is responsible for rather than after the suite as a whole.
    status_name = "suite-status.json" if len(selected) > 1 else f"{Path(selected[0]).stem}-status.json"
    status_path = args.output_dir / status_name
    statuses: dict[str, str] = dict.fromkeys(selected, "not-reached")
    failures: dict[str, str] = {}

    def record() -> None:
        # Rewritten after every notebook: a killed allocation must still leave behind
        # which notebooks ran, which failed, and which were never reached.
        status_path.write_text(
            json.dumps({"statuses": statuses, "failures": failures}, indent=2, sort_keys=True) + "\n"
        )

    record()
    for index, notebook in enumerate(selected, start=1):
        print(f"### notebook {index}/{len(selected)}: {notebook}", flush=True)
        statuses[notebook] = "running"
        record()
        try:
            result = compare_notebook(notebook)
            write_result(result, args.output_dir)
            statuses[notebook] = result.status
            print(
                f"### {notebook}: {result.status}; upstream={result.upstream_seconds:.1f}s "
                f"squidpy={result.squidpy_seconds:.1f}s",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001 - batch suite must preserve later evidence
            statuses[notebook] = "failed"
            failures[notebook] = f"{type(error).__name__}: {error}"
            print(f"### {notebook}: FAILED: {failures[notebook]}", flush=True)
            traceback.print_exception(type(error), error, error.__traceback__)
            try:
                write_failure(notebook, error, args.output_dir)
            except OSError as write_error:  # evidence for one notebook is not worth the rest
                print(f"### {notebook}: could not write failure evidence: {write_error}", flush=True)
        record()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
