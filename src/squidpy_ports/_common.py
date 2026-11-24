"""What every method shares: the pinned vendored upstream, the bundle format, and the generator's CLI.

A bundle is one AnnData whose ``uns`` holds everything, in a zipped zarr store: the fixture
parameters in ``uns["fixtures"]``, and one dict of arrays per section, each with its provenance.
Top-level imports are stdlib and numpy only, so the fixture modules can use this module too.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "THREADS",
    "Imputation",
    "checksum",
    "differing",
    "generator_main",
    "import_vendored",
    "provenance",
    "read_bundle",
    "verify_pin",
    "write_bundle",
]

ROOT = Path(__file__).resolve().parents[2]
#: Single-threaded BLAS, so reduction order, and with it every emitted number, is fixed.
THREADS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def checksum(path: str | Path) -> str:
    """SHA-256 of a file, so a bundle can prove which fixture definitions it came from."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pin(pkg: Path, sha256: dict[str, str], upstream_sha: str) -> None:
    """Check that the vendored files under ``pkg`` are the ones at ``upstream_sha``.

    The pin is checked on the code that runs, with or without git metadata.

    Raises
    ------
    FileNotFoundError
        If the submodule is not checked out.
    RuntimeError
        If a file is not the one at ``upstream_sha``.
    """
    if not all((pkg / name).is_file() for name in sha256):
        raise FileNotFoundError(f"vendored upstream missing at {pkg}; run `git submodule update --init`")
    for name, expected in sha256.items():
        if (digest := checksum(pkg / name)) != expected:
            raise RuntimeError(
                f"vendored {pkg.name}/{name} has sha256 {digest}, not the file at {upstream_sha}. Re-pin "
                f"deliberately in upstream.py and regenerate the bundle -- a moved reference invalidates "
                f"every tolerance downstream."
            )


def import_vendored(root: Path, name: str) -> ModuleType:
    """Import ``name`` from the vendored checkout ``root``, never from an installed copy."""
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.remove(str(root))
    if not Path(str(module.__file__)).resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"imported {name} from {module.__file__}, not from {root}")
    return module


class Imputation(NamedTuple):
    """Reference and query expression, and the true query values of the genes the query lacks."""

    ref: np.ndarray
    query: np.ndarray
    truth: np.ndarray
    genes: list[str]

    @property
    def measured(self) -> list[str]:
        """Genes the query measures, the first ones."""
        return self.genes[: self.query.shape[1]]

    @property
    def unmeasured(self) -> list[str]:
        """Genes the query lacks, which are imputed."""
        return self.genes[self.query.shape[1] :]

    def uns(self, fixtures: ModuleType, **params: Any) -> dict[str, Any]:
        """A bundle's ``fixtures`` and ``inputs``, so a test reads its inputs rather than rebuilding them."""
        return {
            "fixtures": {
                "checksum": fixtures.checksum(),
                "SEED": fixtures.SEED,
                "genes": np.asarray(self.genes),
                "measured": np.asarray(self.measured),
                "unmeasured": np.asarray(self.unmeasured),
                **params,
            },
            "inputs": {"ref": self.ref, "query": self.query, "truth": self.truth},
        }


def provenance(upstream: ModuleType, fixtures: ModuleType, packages: tuple[str, ...], **extra: Any) -> dict[str, Any]:
    """The provenance stamped into every section, so the outputs stay falsifiable."""
    try:
        ports_commit = subprocess.run(
            ["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover - not a checkout, or no git
        ports_commit = "unknown"
    return {
        "upstream_sha": upstream.UPSTREAM_SHA,
        "upstream_url": upstream.UPSTREAM_URL,
        "ports_commit": ports_commit,
        "fixtures_checksum": fixtures.checksum(),
        **{pkg: version(pkg) for pkg in packages},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # no argv: the --out path would make the bundle depend on where it was generated
        "generated_utc": datetime.now(UTC).isoformat(),
        **extra,
    }


def write_bundle(uns: dict[str, Any], out: Path) -> None:
    """Write ``uns`` into an otherwise empty AnnData, stored as a zipped zarr store at ``out``."""
    import anndata as ad
    from zarr.storage import MemoryStore

    # written in memory and zipped once: anndata rewrites zarr.json while writing, which a
    # ZipStore can only append (duplicate members), and a directory on a case-insensitive
    # filesystem merges keys such as `A` and `a`
    store: dict = {}
    ad.AnnData(uns=uns).write_zarr(MemoryStore(store_dict=store))
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key in sorted(store):
            zf.writestr(key, store[key].to_bytes())


def read_bundle(path: Path) -> dict[str, Any]:
    """The ``uns`` of a bundle written by :func:`write_bundle`."""
    import anndata as ad
    from zarr.storage import ZipStore

    with ZipStore(path, mode="r") as store:
        return ad.read_zarr(store).uns


def differing(a: Mapping[str, Any], b: Mapping[str, Any], prefix: str = "") -> list[str]:
    """The path of every value that differs between two bundles' ``uns``, or is in only one; provenance aside."""
    out = []
    for key in sorted(a.keys() | b.keys()):
        path = f"{prefix}{key}"
        if key == "provenance":
            continue
        if key not in a or key not in b:
            out.append(path)
        elif isinstance(a[key], Mapping) and isinstance(b[key], Mapping):
            out += differing(a[key], b[key], f"{path}/")
        elif not _equal(a[key], b[key]):
            out.append(path)
    return out


def _equal(a: Any, b: Any) -> bool:
    """Equal values, NaN equal to NaN, and numbers only at the same dtype."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "biufc" or b.dtype.kind in "biufc":
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    return np.array_equal(a, b)


def generator_main(build: Callable[[], dict[str, Any]], env: dict[str, str], argv: list[str] | None, doc: str) -> int:
    """``--out PATH``: check ``env``, which fixes what the numbers depend on, then write ``build()``'s bundle."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--out", type=Path, required=True, help="the .zarr.zip store to write")
    args = parser.parse_args(argv)
    for var, value in env.items():
        if os.environ.get(var) != value:
            raise RuntimeError(f"{var} must be {value} before Python starts; see the `generate` workflow.")
    write_bundle(build(), args.out)
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m squidpy_ports._common A B``: fail unless two bundles hold the same arrays."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("bundles", type=Path, nargs=2)
    a, b = (read_bundle(path) for path in parser.parse_args(argv).bundles)
    if bad := differing(a, b):
        sys.exit("not reproducible: " + ", ".join(bad))
    print("identical arrays across runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
