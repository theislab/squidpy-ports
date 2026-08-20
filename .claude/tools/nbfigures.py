"""Dump an executed notebook's figures to PNGs, so a run can be reviewed without opening it.

    python .claude/tools/nbfigures.py <notebook.ipynb> [out-dir]

Named by the cell that drew them, so a figure can be traced back to its code.
"""

import base64
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Write every ``image/png`` output of ``argv[0]`` into ``argv[1]``."""
    if not argv:
        print(__doc__)
        return 2
    source = Path(argv[0])
    out = Path(argv[1] if len(argv) > 1 else source.parent / "figures")
    out.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(source.read_text())

    written = []
    for index, cell in enumerate(notebook["cells"]):
        for k, output in enumerate(cell.get("outputs", [])):
            png = (output.get("data") or {}).get("image/png")
            if png is None:
                continue
            name = f"{source.stem}-cell{index:02d}" + (f"-{k}" if k else "") + ".png"
            (out / name).write_bytes(base64.b64decode(png))
            written.append(out / name)
    for path in written:
        print(f"{path}  ({path.stat().st_size // 1024} kB)")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
