"""Execute the STalign comparison notebooks and persist their evidence."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

NOTEBOOKS = {
    "xenium": "stalign-xenium-comparison.ipynb",
    "merfish": "stalign-merfish-comparison.ipynb",
    "visium": "stalign-visium-affine-comparison.ipynb",
}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _execute_notebook(path: Path) -> dict[str, Any]:
    """Execute code cells from a comparison notebook in one shared namespace."""
    notebook = json.loads(path.read_text())
    namespace: dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells, start=1):
        print(f"### {path.stem}: cell {index}/{len(code_cells)}", flush=True)
        source = "".join(cell["source"])
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    return namespace


def _write_evidence(name: str, namespace: dict[str, Any], output_dir: Path, elapsed: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = namespace.get("metrics")
    if metrics is None:
        raise RuntimeError(f"{name}: notebook did not leave a `metrics` table in its namespace")
    metrics.to_csv(output_dir / f"{name}-metrics.csv")
    (output_dir / f"{name}-metrics.json").write_text(metrics["value"].to_json(indent=2) + "\n")

    figure = namespace.get("fig")
    if figure is None:
        raise RuntimeError(f"{name}: notebook did not leave a final `fig` in its namespace")
    figure.savefig(output_dir / f"{name}-comparison.png", dpi=180, bbox_inches="tight")

    jax = namespace.get("jax")
    torch = namespace.get("torch")
    manifest = {
        "comparison": name,
        "elapsed_seconds": elapsed,
        "host": platform.node(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "packages": {
            package: _package_version(package)
            for package in ("squidpy", "squidpy-ports", "jax", "jaxlib", "torch", "numpy", "scipy")
        },
        "jax_backend": jax.default_backend() if jax is not None else None,
        "jax_devices": [str(device) for device in jax.devices()] if jax is not None else [],
        "torch_cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "torch_device": str(namespace.get("device")),
    }
    (output_dir / f"{name}-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    """Run one named comparison and write its durable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", choices=NOTEBOOKS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("STALIGN_COMPARISON_OUTPUT", "comparison-results")),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    notebook = root / "docs" / "notebooks" / NOTEBOOKS[args.comparison]
    started = time.monotonic()
    namespace = _execute_notebook(notebook)
    elapsed = time.monotonic() - started
    _write_evidence(args.comparison, namespace, args.output_dir, elapsed)
    print(f"### {args.comparison}: completed in {elapsed:.1f}s; evidence={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
