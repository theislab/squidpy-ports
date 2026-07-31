"""Load the vendored upstream STalign module and observe it without modifying it.

Upstream is imported *by path* from ``vendor/STalign`` rather than pip-installed: its
``setup.py`` reads ``requirements.txt`` verbatim and pins ``numpy==1.23.4`` /
``torch==2.0.0``, neither of which resolves on a current Python. Loading by path also
keeps a GPL-3.0 project out of this package's dependency metadata.
"""

from __future__ import annotations

import importlib.util
import subprocess
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["UPSTREAM_SHA", "UPSTREAM_URL", "load", "lddmm_with_grads", "vendor_root"]

#: The single upstream commit this package is a reference for. Upstream has no tags and
#: no releases; without a pin, "correct" silently changes under us.
UPSTREAM_SHA = "b2068edc98974efa54537eca194736e177bbe11d"
UPSTREAM_URL = "https://github.com/JEFworks-Lab/STalign"


def vendor_root() -> Path:
    """Path to the vendored upstream checkout."""
    return Path(__file__).resolve().parents[3] / "vendor" / "STalign"


@cache
def load() -> ModuleType:
    """Import the vendored ``STalign.STalign`` module, asserting the pin.

    Raises
    ------
    FileNotFoundError
        If the submodule is not checked out.
    RuntimeError
        If the checkout is not at :data:`UPSTREAM_SHA`.
    """
    root = vendor_root()
    path = root / "STalign" / "STalign.py"
    if not path.is_file():
        raise FileNotFoundError(f"vendored STalign missing at {path}; run `git submodule update --init`")

    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != UPSTREAM_SHA:
        raise RuntimeError(
            f"vendored STalign is at {head}, expected {UPSTREAM_SHA}. Re-pin deliberately "
            f"in upstream.py and regenerate the whole bundle -- a moved reference "
            f"invalidates every tolerance downstream."
        )

    import matplotlib

    # LDDMM creates figures unconditionally (STalign.py:1094, :1140, :1142).
    matplotlib.use("Agg")

    spec = importlib.util.spec_from_file_location("_vendored_stalign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _assert_meshgrid_is_ij()
    return module


def _assert_meshgrid_is_ij() -> None:
    """Fail loudly if torch ever flips ``meshgrid``'s default indexing.

    ``STalign.py`` calls ``torch.meshgrid(xv)`` without ``indexing=`` at :756, :781,
    :1063 and :1070. Torch still defaults to ``'ij'`` but warns and has signalled intent
    to change. If that lands, every grid upstream builds silently transposes and the
    fixtures become quietly wrong rather than obviously broken.
    """
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows, _ = torch.meshgrid(torch.arange(2), torch.arange(3))
    if tuple(rows.shape) != (2, 3):
        raise RuntimeError(
            f"torch.meshgrid without `indexing=` now yields shape {tuple(rows.shape)}, not 'ij'. "
            f"Every grid in the vendored STalign is transposed; pin an older torch."
        )


def lddmm_with_grads(
    st: ModuleType,
    *,
    niter: int,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, list]]:
    """Run the real upstream ``LDDMM`` and capture what it never returns.

    Upstream keeps the objective in a local ``Esave`` list and zeroes ``.grad`` at
    STalign.py:1210-1211 before returning, so ``E``, ``dE/dL`` and ``dE/dT`` are
    unobservable from the outside. We *observe* them rather than reimplementing the
    loop -- a reimplementation would make this package the thing under test.

    - ``E`` comes from wrapping ``Tensor.backward``. ``E.backward()`` (:1204) is the
      only ``.backward()`` call in the loop, so the n-th capture is iteration n.
    - ``dE/dL`` / ``dE/dT`` come from autograd hooks on the live leaf tensors, reached
      by monkeypatching ``to_A``, which :1155 calls with them every iteration. Hooks
      fire during ``E.backward()``, hence before the zeroing.

    Returns
    -------
    The ``LDDMM`` output dict, and a dict with per-iteration ``E``, ``L`` and ``T``.
    """
    import torch

    captured: dict[str, list] = {"E": [], "L": [], "T": []}

    real_to_A = st.to_A
    real_backward = torch.Tensor.backward
    hooked = False

    def spy_to_A(L, T):
        nonlocal hooked
        if not hooked:  # the leaves are stable across iterations; hook once
            hooked = True
            L.register_hook(lambda g: captured["L"].append(g.detach().clone()))
            T.register_hook(lambda g: captured["T"].append(g.detach().clone()))
        return real_to_A(L, T)

    def spy_backward(self, *args, **kw):
        captured["E"].append(float(self.detach()))
        return real_backward(self, *args, **kw)

    st.to_A = spy_to_A
    torch.Tensor.backward = spy_backward
    try:
        out = st.LDDMM(niter=niter, **kwargs)
    finally:
        st.to_A = real_to_A
        torch.Tensor.backward = real_backward

    # If upstream ever reorders, fail here rather than silently comparing the wrong thing.
    if not (len(captured["E"]) == len(captured["L"]) == len(captured["T"]) == niter):
        raise RuntimeError(
            f"expected {niter} captures per quantity, got "
            f"E={len(captured['E'])} L={len(captured['L'])} T={len(captured['T'])}; "
            f"the vendored LDDMM loop no longer matches what these spies assume"
        )
    return out, captured
