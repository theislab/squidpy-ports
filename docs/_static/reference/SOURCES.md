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
| `upstream-merfish-merfish.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 32 | Overlay: initial affine vs target · STaligned vs target. |
| `upstream-merfish-merfish-grid.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 25 | Deformation grid on warped source→target density. |
| `upstream-merfish-merfish-convergence.png` | MERFISH ↔ MERFISH (LDDMM) | [merfish-merfish-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-merfish-alignment.ipynb) | 21 | Energy convergence: E / EM / ER vs iteration. |
| `upstream-visium-visium-affine.png` | Visium ↔ Visium (affine-only) | [visium-visium-alignment-affine-only.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/visium-visium-alignment-affine-only.ipynb) | 31 | Overlay: source vs target · STaligned vs target. |
| `upstream-xenium-xenium.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 44 | Overlay: moving rep1 aligned onto fixed rep2. |
| `upstream-xenium-xenium-grid.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 32 | Deformation grid on warped source→target density (landmarks overlaid). |
| `upstream-xenium-xenium-convergence.png` | Xenium ↔ Xenium (LDDMM) | [xenium-xenium-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/xenium-xenium-alignment.ipynb) | 29 | Energy convergence: E / EM / ER / EP vs iteration. |
| `upstream-merfish-allen3d-overlay.png` | MERFISH → Allen CCF (rank-3 LDDMM) | [merfish-allen3Datlas-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-allen3Datlas-alignment.ipynb) | 39 | MERFISH section · z=0 slice of the aligned 3D atlas · overlay. |
| `upstream-merfish-allen3d-surface.png` | MERFISH → Allen CCF (rank-3 LDDMM) | [merfish-allen3Datlas-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-allen3Datlas-alignment.ipynb) | 41 | Marching-cubes surface of the atlas with the fitted section placed in it. |
| `upstream-merfish-allen3d-regions.png` | MERFISH → Allen CCF (rank-3 LDDMM) | [merfish-allen3Datlas-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-allen3Datlas-alignment.ipynb) | 43 | Per-cell Allen region assignment from `analyze3Dalign`. |
| `upstream-merfish-allen3d-subset.png` | MERFISH → Allen CCF (rank-3 LDDMM) | [merfish-allen3Datlas-alignment.ipynb](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks/merfish-allen3Datlas-alignment.ipynb) | 45 | The same, restricted to VISp4 / VISp5. |

`starmap-allen3Datlas-alignment.ipynb` contributes nothing here: upstream committed **no** output
images for it at the pinned commit, so there is no published figure to copy. Its comparison rests on
the two replay passes alone.

The rank-3 notebook's own convergence plots (upstream cell 32) are also absent. The replay closes
whatever figures the fit itself opens, so that both passes emit exactly the notebook's figures in
the same order — see `_replay_notebook`. Comparing them would compare two solvers' private plotting,
not the alignment.
