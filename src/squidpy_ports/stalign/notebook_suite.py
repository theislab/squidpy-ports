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
        "The notebook's second fit uses the stale `A, v, xv = LDDMM(...)` tuple API. "
        "The suite compares its first complete 10,000-iteration fit."
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
    figure: Any
    upstream_seconds: float = 0.0
    squidpy_seconds: float = 0.0
    note: str | None = None


class _CapturedCall(Exception):
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        super().__init__("captured upstream call")
        self.args_ = args
        self.kwargs_ = kwargs


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


#: Upstream is imported by path and pre-seeded into the replay namespace already patched,
#: so the notebook's own import of it must not run -- a plain `import STalign` would find
#: the package on `sys.path` and silently replace the patched module, losing the capture.
_STALIGN_IMPORT = re.compile(r"^\s*(import\s+STalign\b|from\s+STalign\s+import\b)")


def _clean_cell(source: str) -> str:
    """Strip what cannot be replayed from one notebook cell, and nothing else.

    Only the shell/magic lines and upstream's own import are dropped. Dropping the *cell*
    instead would take the rest of it with them: several notebooks put `import pandas as
    pd` (and numpy, torch, matplotlib) in the same cell as `import STalign`, so a
    cell-level skip left the whole notebook without pandas and failed several cells later
    with a bare `NameError`.
    """
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")) or _STALIGN_IMPORT.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _capture_notebook_call(notebook: str, function: str) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Execute upstream preprocessing unchanged and stop at the requested fit call."""
    import matplotlib.pyplot as plt

    st = upstream.load()
    notebook_path = upstream.vendor_root() / "docs" / "notebooks" / notebook
    payload = json.loads(notebook_path.read_text())
    namespace: dict[str, Any] = {
        "__file__": str(notebook_path),
        "__name__": "__main__",
        "STalign": st,
    }
    original = getattr(st, function)

    def capture(*args: Any, **kwargs: Any) -> None:
        raise _CapturedCall(args, kwargs)

    setattr(st, function, capture)
    try:
        with _replay_directory(notebook_path):
            for cell_number, cell in enumerate(payload["cells"], start=1):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                if "np.savez(" in source or ".to_csv(" in source:
                    continue
                cleaned = _clean_cell(source)
                if not cleaned.strip():
                    continue
                try:
                    exec(compile(cleaned, f"{notebook}:cell-{cell_number}", "exec"), namespace)
                finally:
                    plt.close("all")
    except _CapturedCall as captured:
        return captured.args_, captured.kwargs_, namespace
    finally:
        setattr(st, function, original)
    raise RuntimeError(f"{notebook}: did not call STalign.{function}")


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def _unit_peak(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    return array / max(float(np.max(np.abs(array))), np.finfo(float).tiny)


def _scalar_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    return np.mean(array, axis=0)


def _point_cloud(namespace: dict[str, Any], suffix: str) -> np.ndarray | None:
    x, y = namespace.get(f"x{suffix}"), namespace.get(f"y{suffix}")
    if x is None or y is None:
        return None
    x_array, y_array = _numpy(x), _numpy(y)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape or x_array.size < 20:
        return None
    return np.stack((x_array, y_array), axis=1)


def _sample_points(namespace: dict[str, Any], source_axes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    source = _point_cloud(namespace, "I")
    if source is not None:
        if len(source) <= 20_000:
            return source
        rng = np.random.default_rng(0)
        return source[np.sort(rng.choice(len(source), 20_000, replace=False))]
    rows = np.linspace(source_axes[0][0], source_axes[0][-1], 45)
    cols = np.linspace(source_axes[1][0], source_axes[1][-1], 45)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    return np.stack((xx.ravel(), yy.ravel()), axis=1)


def _extent(axes: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float, float]:
    row_step, col_step = axes[0][1] - axes[0][0], axes[1][1] - axes[1][0]
    return (
        float(axes[1][0] - col_step / 2),
        float(axes[1][-1] + col_step / 2),
        float(axes[0][-1] + row_step / 2),
        float(axes[0][0] - row_step / 2),
    )


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


def _plot_comparison(
    notebook: str,
    target: np.ndarray,
    target_axes: tuple[np.ndarray, np.ndarray],
    upstream_warped: np.ndarray,
    squidpy_warped: np.ndarray,
    upstream_points: np.ndarray,
    squidpy_points: np.ndarray,
    target_points: np.ndarray | None,
    upstream_weights: np.ndarray,
    squidpy_weights: np.ndarray,
):
    import matplotlib.pyplot as plt

    extent = _extent(target_axes)
    upstream_scalar = _unit_peak(_scalar_image(upstream_warped))
    squidpy_scalar = _unit_peak(_scalar_image(squidpy_warped))
    density_difference = np.abs(upstream_scalar - squidpy_scalar)
    point_delta = np.linalg.norm(upstream_points - squidpy_points, axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), constrained_layout=True)
    panels = (
        (upstream_scalar, "upstream warped source", "magma"),
        (squidpy_scalar, "Squidpy warped source", "magma"),
        (density_difference, "absolute density difference", "viridis"),
        (_unit_peak(_scalar_image(target)), "fixed target", "magma"),
    )
    for ax, (image, title, cmap) in zip(axes[0], panels, strict=True):
        artist = ax.imshow(image, extent=extent, cmap=cmap)
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(artist, ax=ax, shrink=0.72)

    for ax, points, title in zip(
        axes[1, :2],
        (upstream_points, squidpy_points),
        ("upstream aligned source", "Squidpy aligned source"),
        strict=True,
    ):
        if target_points is not None:
            fixed = target_points
            if len(fixed) > 20_000:
                fixed = fixed[np.linspace(0, len(fixed) - 1, 20_000, dtype=int)]
            ax.scatter(fixed[:, 0], fixed[:, 1], s=1, alpha=0.12, label="fixed")
        else:
            ax.imshow(_unit_peak(_scalar_image(target)), extent=extent, cmap="Greys", alpha=0.35)
        ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.22, label="aligned source")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    axes[1, 0].legend(markerscale=5, frameon=False)

    delta_artist = axes[1, 2].scatter(upstream_points[:, 0], upstream_points[:, 1], c=point_delta, s=3, cmap="viridis")
    axes[1, 2].set(title="pointwise |upstream − Squidpy|", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(delta_artist, ax=axes[1, 2], label="distance", shrink=0.72)

    weight_difference = np.abs(np.asarray(upstream_weights) - np.asarray(squidpy_weights))
    weight_artist = axes[1, 3].imshow(weight_difference, extent=extent, cmap="viridis")
    axes[1, 3].set(title="absolute matching-weight difference", xlabel="x", ylabel="y")
    fig.colorbar(weight_artist, ax=axes[1, 3], shrink=0.72)

    note = UPSTREAM_NOTES.get(notebook)
    title = f"STalign notebook parity — {notebook}"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=13)
    return fig, upstream_scalar, squidpy_scalar, point_delta, weight_difference


def _compare_lddmm(notebook: str) -> ComparisonResult:
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import torch
    from squidpy.experimental.methods.align_samples import StalignResult
    from squidpy.experimental.methods.align_samples._stalign_impl._core import lddmm

    args, kwargs, namespace = _capture_notebook_call(notebook, "LDDMM")
    source_axes = tuple(_numpy(axis) for axis in args[0])
    source_image = _numpy(args[1])
    target_axes = tuple(_numpy(axis) for axis in args[2])
    target_image = _numpy(args[3])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    upstream_kwargs = dict(kwargs)
    upstream_kwargs["device"] = device
    for key in ("muA", "muB"):
        if upstream_kwargs.get(key) is not None:
            upstream_kwargs[key] = torch.as_tensor(upstream_kwargs[key], device=device, dtype=torch.float64)
    torch.set_default_dtype(torch.float64)

    st = upstream.load()
    started = time.monotonic()
    upstream_fit = st.LDDMM(*args, **upstream_kwargs)
    upstream_seconds = time.monotonic() - started
    plt.close("all")

    upstream_warped = _numpy(
        st.transform_image_source_to_target(
            upstream_fit["xv"],
            upstream_fit["v"],
            upstream_fit["A"],
            [torch.as_tensor(axis, device=device, dtype=torch.float64) for axis in args[0]],
            torch.as_tensor(args[1], device=device, dtype=torch.float64),
            [torch.as_tensor(axis, device=device, dtype=torch.float64) for axis in args[2]],
        )
    )
    sample_xy = _sample_points(namespace, source_axes)
    sample_rc = np.ascontiguousarray(sample_xy[:, ::-1])
    upstream_points = _numpy(
        st.transform_points_source_to_target(
            upstream_fit["xv"],
            upstream_fit["v"],
            upstream_fit["A"],
            torch.as_tensor(sample_rc, device=device, dtype=torch.float64),
        )
    )[:, ::-1]
    upstream_weights = _numpy(upstream_fit["WM"])

    jax.block_until_ready(jnp.asarray(0.0))
    started = time.monotonic()
    jax_fit = lddmm(source_axes, source_image, target_axes, target_image, **_jax_kwargs(kwargs))
    jax.block_until_ready(jax_fit["A"])
    squidpy_seconds = time.monotonic() - started
    result = StalignResult(
        affine=jax_fit["A"],
        velocity=jax_fit["v"],
        velocity_grid=jax_fit["xv"],
        aligned_points=jnp.zeros((0, 2)),
        query_axes=source_axes,
        ref_axes=target_axes,
        match_weights=jax_fit["WM"],
        artifact_weights=jax_fit["WA"],
        background_weights=jax_fit["WB"],
        energies=jax_fit["energies"],
        n_iter=int(jax_fit["n_iter"]),
    )
    squidpy_warped = _numpy(result.warp_image(source_image))
    squidpy_points = _numpy(result.transform(sample_xy))
    squidpy_weights = _numpy(jax_fit["WM"])
    target_points = _point_cloud(namespace, "J")

    figure, upstream_scalar, squidpy_scalar, point_delta, weight_difference = _plot_comparison(
        notebook,
        target_image,
        target_axes,
        upstream_warped,
        squidpy_warped,
        upstream_points,
        squidpy_points,
        target_points,
        upstream_weights,
        squidpy_weights,
    )
    metrics = {
        "warped density relative L2": _relative_l2(squidpy_scalar, upstream_scalar),
        "aligned points relative L2": _relative_l2(squidpy_points, upstream_points),
        "aligned points median |delta|": float(np.median(point_delta)),
        "aligned points p95 |delta|": float(np.quantile(point_delta, 0.95)),
        "matching weights relative L2": _relative_l2(squidpy_weights, upstream_weights),
        "matching weights max |delta|": float(np.max(weight_difference)),
    }
    return ComparisonResult(
        notebook=notebook,
        status="compared",
        metrics=metrics,
        figure=figure,
        upstream_seconds=upstream_seconds,
        squidpy_seconds=squidpy_seconds,
        note=UPSTREAM_NOTES.get(notebook),
    )


def _compare_affine(notebook: str) -> ComparisonResult:
    import matplotlib.pyplot as plt
    from squidpy.experimental.methods.align_samples._stalign_impl._helpers import affine_from_points

    args, _, _ = _capture_notebook_call(notebook, "L_T_from_points")
    source, target = _numpy(args[0]), _numpy(args[1])
    st = upstream.load()
    upstream_linear, upstream_translation = st.L_T_from_points(source, target)
    squidpy_linear, squidpy_translation = affine_from_points(source, target)
    upstream_aligned = source @ upstream_linear.T + upstream_translation
    squidpy_aligned = source @ np.asarray(squidpy_linear).T + np.asarray(squidpy_translation)
    delta = np.linalg.norm(upstream_aligned - squidpy_aligned, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, aligned, title in zip(
        axes[:2],
        (upstream_aligned, squidpy_aligned),
        ("upstream affine landmarks", "Squidpy affine landmarks"),
        strict=True,
    ):
        ax.scatter(target[:, 1], target[:, 0], s=45, label="target")
        ax.scatter(aligned[:, 1], aligned[:, 0], s=30, label="aligned source")
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    axes[0].legend(frameon=False)
    artist = axes[2].scatter(upstream_aligned[:, 1], upstream_aligned[:, 0], c=delta, s=50, cmap="viridis")
    axes[2].set(title="landmark |upstream − Squidpy|", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(artist, ax=axes[2], label="distance")
    fig.suptitle(f"STalign notebook parity — {notebook}")
    metrics = {
        "affine linear relative L2": _relative_l2(np.asarray(squidpy_linear), upstream_linear),
        "affine translation relative L2": _relative_l2(np.asarray(squidpy_translation), upstream_translation),
        "aligned landmarks median |delta|": float(np.median(delta)),
        "upstream landmark residual": float(np.linalg.norm(upstream_aligned - target)),
        "squidpy landmark residual": float(np.linalg.norm(squidpy_aligned - target)),
    }
    return ComparisonResult(notebook, "compared-affine", metrics, fig)


def _status_panel(notebook: str, status: str, note: str) -> ComparisonResult:
    """Emit a panel that says why a notebook carries no numbers, instead of no evidence."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.62, notebook, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.38, note, ha="center", va="center", fontsize=11, wrap=True)
    return ComparisonResult(notebook, status, {}, fig, note=note)


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
    """Persist a comparison figure, metrics, and provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.notebook).stem
    image_path = output_dir / f"{stem}-comparison.png"
    result.figure.savefig(image_path, dpi=ARCHIVE_DPI, bbox_inches="tight")
    (output_dir / f"{stem}-metrics.json").write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n")
    manifest = _manifest(result.notebook, result.status, result.note, result.upstream_seconds, result.squidpy_seconds)
    (output_dir / f"{stem}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    image_data = _embedded_panel(result.figure)
    executed = {
        "cells": [
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
            {
                "cell_type": "code",
                "execution_count": 2,
                "id": "figure",
                "metadata": {},
                "outputs": [
                    {
                        "data": {"image/png": image_data, "text/plain": ["<STalign parity comparison figure>"]},
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": ["result.figure\n"],
            },
        ]
    }
    _write_notebook(executed, manifest, output_dir)


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
                    "source": ["result.figure\n"],
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
