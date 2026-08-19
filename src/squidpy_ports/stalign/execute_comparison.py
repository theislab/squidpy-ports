"""Execute one STalign comparison notebook and save it with its figures + provenance.

This is the reproducible unit behind the comparison panels. Run it -- ideally through the
container in ``container/`` on any GPU box -- and it re-executes a single
``stalign-*-comparison.ipynb`` against the pinned upstream STalign and the squidpy JAX port,
writing:

* ``<stem>-executed.ipynb`` -- the notebook with the density plot, the upstream-vs-port panel
  and the metric table embedded;
* ``<stem>-panel.png`` -- the comparison panel on its own, plus ``<stem>-figure-N.png`` for
  every figure the notebook drew, so an intermediate one (the port's standalone overlay) can
  be used as-is rather than cropped out of a composed pair;
* ``<stem>-manifest.json`` -- the exact package versions **and the squidpy fork commit**, so
  the numbers are never orphaned from the code that produced them.

The comparison notebooks call ``plt.show()`` but the inline backend does not take effect
under a headless execution kernel, so a throwaway first cell is injected to make
``plt.show()`` embed every open figure as a PNG -- backend-independent, and the committed
notebooks stay free of the ``%matplotlib`` magic that would break a plain ``exec``.
"""

from __future__ import annotations

import argparse
import base64
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from squidpy_ports.stalign import upstream

# Prepended before execution; dropped from the saved notebook afterwards.
_CAPTURE_SOURCE = (
    "import io as _io\n"
    "import matplotlib.pyplot as _plt\n"
    "from IPython.display import Image as _Image, display as _display\n"
    "def _capture_show(*args, **kwargs):\n"
    "    for _num in _plt.get_fignums():\n"
    "        _buffer = _io.BytesIO()\n"
    "        _plt.figure(_num).savefig(_buffer, format='png', dpi=130, bbox_inches='tight')\n"
    "        _display(_Image(data=_buffer.getvalue()))\n"
    "    _plt.close('all')\n"
    "_plt.show = _capture_show\n"
)

# Appended before execution and kept in the saved notebook: a visible version stamp that
# complements the machine-readable manifest. Uses the watermark API (not the magic) and is
# tolerant, so a provenance nicety can never fail the run.
_WATERMARK_SOURCE = (
    "try:\n"
    "    from watermark import watermark as _watermark\n"
    "    print(_watermark(packages='squidpy,squidpy_ports,jax,jaxlib,torch', python=True, machine=True))\n"
    "except Exception as _exc:  # noqa: BLE001 - a version stamp must never fail the run\n"
    "    print(f'watermark unavailable: {_exc}')\n"
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def execute(notebook: Path, output_dir: Path, *, timeout: int = 5400) -> Path:
    """Execute ``notebook`` and write the executed notebook, its panel, and a manifest."""
    import nbformat
    from nbclient import NotebookClient

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = notebook.stem

    nb = nbformat.read(notebook, as_version=4)
    shim = nbformat.v4.new_code_cell(_CAPTURE_SOURCE)
    nb.cells.insert(0, shim)
    stamp = nbformat.v4.new_code_cell(_WATERMARK_SOURCE)
    nb.cells.append(stamp)
    NotebookClient(nb, timeout=timeout, kernel_name="python3").execute()
    nb.cells = [cell for cell in nb.cells if cell is not shim]  # keep the watermark stamp

    # Tidy execution counts back to 1..N now that the shim is gone.
    count = 0
    for cell in nb.cells:
        if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
            count += 1
            cell["execution_count"] = count
            for output in cell.get("outputs", []):
                if output.get("execution_count") is not None:
                    output["execution_count"] = count

    executed = output_dir / f"{stem}-executed.ipynb"
    nbformat.write(nb, executed)

    panels = [
        output["data"]["image/png"]
        for cell in nb.cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    for index, encoded in enumerate(panels):
        (output_dir / f"{stem}-figure-{index}.png").write_bytes(base64.b64decode(encoded))
    if panels:
        # The comparison panel is the last figure; the density plot comes first. It is written
        # twice on purpose -- `-panel.png` is the name the docs and container README quote, and
        # the indexed dump is how the intermediate figures (the port's standalone overlay among
        # them) get out of the notebook without being cropped back out of a composed pair.
        (output_dir / f"{stem}-panel.png").write_bytes(base64.b64decode(panels[-1]))

    manifest = {
        "notebook": notebook.name,
        "packages": {name: _package_version(name) for name in ("squidpy", "squidpy-ports", "jax", "jaxlib", "torch")},
        "squidpy_commit": upstream.squidpy_commit(),
    }
    (output_dir / f"{stem}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return executed


def main() -> None:
    """Stalign"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="comparison notebook: a basename under --notebooks-dir, or a path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--notebooks-dir", type=Path, default=Path("docs/notebooks"))
    args = parser.parse_args()

    notebook = Path(args.notebook)
    if not notebook.exists():
        notebook = args.notebooks_dir / args.notebook
    print(f"executing {notebook} -> {args.output_dir}", flush=True)
    print(f"wrote {execute(notebook, args.output_dir)}")


if __name__ == "__main__":
    main()
