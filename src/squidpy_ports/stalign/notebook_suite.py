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
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

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
        lines.append(line)
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
        with _replay_directory(notebook_path):
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


def jax_parameters() -> Mapping[str, inspect.Parameter]:
    """The port's ``lddmm`` signature, resolved at call time so imports stay optional."""
    from squidpy.experimental.methods.align_samples._stalign_impl._core import lddmm

    return inspect.signature(lddmm).parameters


def _jax_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _convert_kwargs(kwargs, jax_parameters())


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


def _compare_lddmm(notebook: str) -> ComparisonResult:
    """Run the notebook twice -- upstream's fit, then Squidpy's -- and pair up its figures.

    Only the fit differs. Every figure below it is drawn by the notebook's own cells calling
    upstream's `build_transform` / `transform_image_*` / `transform_points_*`, so a visible
    difference is attributable to `A`, `v` and `xv` alone.
    """
    import matplotlib.pyplot as plt
    import torch
    from squidpy.experimental.methods.align_samples._stalign_impl._core import lddmm

    device = _fit_device()
    torch.set_default_dtype(torch.float64)

    def upstream_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        call = dict(kwargs)
        call["device"] = device
        for key in ("muA", "muB"):
            if call.get(key) is not None:
                call[key] = torch.as_tensor(call[key], device=device, dtype=torch.float64)
        return _FitResult({key: _on_fit_device(value) for key, value in original(*args, **call).items()})

    def squidpy_fit(args: tuple[Any, ...], kwargs: dict[str, Any], original: Any) -> Any:
        import jax

        source_axes = tuple(_numpy(axis) for axis in args[0])
        target_axes = tuple(_numpy(axis) for axis in args[2])
        fit = lddmm(source_axes, _numpy(args[1]), target_axes, _numpy(args[3]), **_jax_kwargs(kwargs))
        jax.block_until_ready(fit["A"])
        return _torch_fit(fit)

    upstream_pass = _replay_notebook(notebook, upstream_fit)
    plt.close("all")
    squidpy_pass = _replay_notebook(notebook, squidpy_fit)
    plt.close("all")

    skipped_note = _require_same_cells_ran(notebook, upstream_pass, squidpy_pass)
    metrics = _namespace_metrics(upstream_pass.namespace, squidpy_pass.namespace)
    figures = []
    for (cell_number, upstream_png), (_, squidpy_png) in zip(upstream_pass.figures, squidpy_pass.figures, strict=False):
        figures.append(_compose_pair(upstream_png, squidpy_png, f"{notebook} — cell {cell_number}"))
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
        upstream_seconds=upstream_pass.seconds,
        squidpy_seconds=squidpy_pass.seconds,
        note=" ".join(filter(None, (UPSTREAM_NOTES.get(notebook), skipped_note))) or None,
    )


def _compare_affine(notebook: str) -> ComparisonResult:
    """The landmark-affine notebook, replayed the same way but around ``L_T_from_points``.

    This notebook never reaches ``LDDMM``; its plots come from the affine that
    ``L_T_from_points`` returns, so that is the call the two passes swap.
    """
    import matplotlib.pyplot as plt
    from squidpy.experimental.methods.align_samples._stalign_impl._helpers import affine_from_points

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
        linear, translation = affine_from_points(_numpy(args[0]), _numpy(args[1]))
        record("squidpy", args, linear, translation)
        return np.asarray(linear), np.asarray(translation)

    upstream_pass = _replay_notebook(notebook, upstream_fit, function="L_T_from_points")
    plt.close("all")
    squidpy_pass = _replay_notebook(notebook, squidpy_fit, function="L_T_from_points")
    plt.close("all")

    skipped_note = _require_same_cells_ran(notebook, upstream_pass, squidpy_pass)
    metrics = {**_namespace_metrics(upstream_pass.namespace, squidpy_pass.namespace), **residuals}
    figures = [
        _compose_pair(upstream_png, squidpy_png, f"{notebook} — cell {cell_number}")
        for (cell_number, upstream_png), (_, squidpy_png) in zip(
            upstream_pass.figures, squidpy_pass.figures, strict=True
        )
    ]
    note = " ".join(filter(None, (UPSTREAM_NOTES.get(notebook), skipped_note))) or None
    return ComparisonResult(notebook, "compared-affine", metrics, figures, note=note)


def _status_panel(notebook: str, status: str, note: str) -> ComparisonResult:
    """Emit a panel that says why a notebook carries no numbers, instead of no evidence."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, notebook, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.38, note, ha="center", va="center", fontsize=11, wrap=True)
    return ComparisonResult(notebook, status, {}, [fig], note=note)


def _three_d_status(notebook: str) -> ComparisonResult:
    return _status_panel(
        notebook,
        "unsupported-3d",
        "Not numerically compared: this upstream notebook calls LDDMM_3D_to_slice. "
        "Squidpy currently implements the 2D LDDMM solver only; a separate volume-to-image "
        "estimator is required for an honest comparison.",
    )


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
        return _three_d_status(notebook)
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
    (output_dir / f"{stem}-metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
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
            "source": _compare_source(result.notebook, "result.metrics\n"),
        },
    ]
    for index, figure in enumerate(result.figures):
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
                            "text/plain": [f"<upstream vs Squidpy, figure {index + 1}>"],
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": [f"result.figures[{index}]\n"],
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


def write_notebook_wrappers(output_dir: Path) -> None:
    """Generate the one-to-one notebook map used by the documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for notebook in NOTEBOOKS:
        upstream_url = f"{upstream.UPSTREAM_URL}/blob/{upstream.UPSTREAM_SHA}/docs/notebooks/{notebook}"
        payload = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "id": "header",
                    "metadata": {},
                    "source": [
                        f"# STalign parity: `{notebook}`\n",
                        "\n",
                        f"One-to-one replay of [the pinned upstream notebook]({upstream_url}). ",
                        "Its preprocessing and fit arguments are executed directly from the vendored notebook; ",
                        "the captured fit is then run with upstream PyTorch and Squidpy JAX.\n",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "metrics",
                    "metadata": {},
                    "outputs": [],
                    "source": _compare_source(notebook, "result.metrics\n"),
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "figure",
                    "metadata": {},
                    "outputs": [],
                    "source": ["result.figures[0]\n"],
                },
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.13"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (output_dir / notebook).write_text(json.dumps(payload, indent=1) + "\n")


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
