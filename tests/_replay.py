"""Helpers the reference suite borrows, carved out of the retired notebook-comparison harness.

`test_reference.py` issues its fits the way the harness issued them, deliberately: the
reference numbers are only meaningful if they come from the same call the port actually makes.
When the harness itself was dropped -- it replayed upstream's notebooks through internal API and
its evidence has been retired from the docs -- these eight pieces were all that the numeric tests
still needed, 135 lines of its 2138.

Test-local on purpose. Nothing ships them, and nothing outside `tests/` should grow a dependency
on them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


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


_IMAGE_KEY = "replay"


def as_sdata(array: np.ndarray, axes: Sequence[np.ndarray]) -> Any:
    """Wrap a channels-first array and its axes as a one-image ``SpatialData``.

    The public ``stalign_align_*`` entry points read an element's physical axes off the
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

    These are what the ``stalign_align_*`` entry points accept, so they are what a captured
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
