# Upstream STalign reference results

Result figures copied **verbatim** (no recompute) from the pinned upstream
STalign submodule. These are the fixed reference the squidpy JAX port
comparisons are diffed against.

- Repo: [JEFworks-Lab/STalign](https://github.com/JEFworks-Lab/STalign)
- Pinned commit: `b2068edc98974efa54537eca194736e177bbe11d`
- `Cell` = 0-based cell index in the upstream `.ipynb`.

| Image | Dataset | Upstream notebook (@ pinned commit) | Cell | Figure |
|---|---|---|---|---|
| `upstream-merfish-merfish.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 32 | Left: initial affine vs target. Right: STaligned vs target. |
| `upstream-visium-visium-affine.png` | Visium ↔ Visium (affine-only) | [visium-visium-alignment-affine-only.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/visium-visium-alignment-affine-only.ipynb) | 31 | Left: source vs target. Right: STaligned vs target. |
| `upstream-xenium-xenium.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 44 | Moving rep1 aligned onto fixed rep2; tissue structures overlap in the shared band. |
