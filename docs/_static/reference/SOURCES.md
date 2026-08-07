# Upstream STalign reference results

Result figures copied **verbatim** (no recompute) from the pinned upstream
STalign submodule. These are the fixed reference the squidpy JAX port
comparisons are diffed against.

- Repo: [JEFworks-Lab/STalign](https://github.com/JEFworks-Lab/STalign)
- Pinned commit: `b2068edc98974efa54537eca194736e177bbe11d`
- `Cell` = 0-based cell index in the upstream `.ipynb`.

Each LDDMM dataset (merfish, xenium) has three figures: the final aligned
**overlay**, the **deformation grid** (the diffeomorphism made visible), and the
**convergence** curves (E / matching / regularization vs iteration). Visium is
**affine-only** — no diffeomorphic velocity field, so the grid degenerates to a
gridless density and the convergence flatlines immediately; it keeps the overlay
only.

| Image | Dataset | Upstream notebook (@ pinned commit) | Cell | Figure |
|---|---|---|---|---|
| `upstream-merfish-merfish.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 44 | Overlay: initial affine vs target · STaligned vs target. |
| `upstream-merfish-merfish-grid.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 25 | Deformation grid on warped source→target density. |
| `upstream-merfish-merfish-convergence.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 21 | Energy convergence: E / EM / ER vs iteration. |
| `upstream-visium-visium-affine.png` | Visium ↔ Visium (affine-only) | [visium-visium-alignment-affine-only.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/visium-visium-alignment-affine-only.ipynb) | 31 | Overlay: source vs target · STaligned vs target. |
| `upstream-xenium-xenium.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 44 | Overlay: moving rep1 aligned onto fixed rep2. |
| `upstream-xenium-xenium-grid.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 32 | Deformation grid on warped source→target density (landmarks overlaid). |
| `upstream-xenium-xenium-convergence.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 29 | Energy convergence: E / EM / ER / EP vs iteration. |
