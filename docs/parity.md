# Notebook values comparison

Each upstream notebook is replayed **twice**: once with upstream's PyTorch fit, once with
Squidpy's JAX fit substituted for it. Everything after the fit is upstream's own code, so
the figures below are the notebook's own plots and the only difference between the two
sides is the fitted `A`, `v` and `xv`. Metrics are relative L2 on the variables the
notebook itself computes.

The per-variable metrics, manifests and package versions behind this table are committed
under [`docs/parity/`](https://github.com/theislab/squidpy-ports/tree/main/docs/parity) —
one `-metrics.json`, `-manifest.json` and `-status.json` per notebook. Each manifest
records the host, package versions and the resolved `squidpy_commit`, so no result is
separated from the code that produced it.

| notebook | status | largest divergence | upstream | squidpy |
| --- | --- | --- | --- | --- |
| [`merfish-allen3Datlas-alignment`](notebooks/stalign-upstream/merfish-allen3Datlas-alignment) | compared | `v` = `1.9e+00` | 86s | 18s |
| [`starmap-allen3Datlas-alignment`](notebooks/stalign-upstream/starmap-allen3Datlas-alignment) | compared | `AI` = `9.5e-01` | 44s | 16s |
| [`xenium-heimage-alignment`](notebooks/stalign-upstream/xenium-heimage-alignment) | compared | `v` = `1.9e-01` | 163s | 7s |
| [`merfish-visium-alignment-with-point-annotator`](notebooks/stalign-upstream/merfish-visium-alignment-with-point-annotator) | compared | `testM` = `5.5e-03` | 236s | 9s |
| [`merfish-merfish-alignment-affine-only`](notebooks/stalign-upstream/merfish-merfish-alignment-affine-only) | compared | `A` = `6.3e-04` | 59s | 5s |
| [`xenium-starmap-alignment`](notebooks/stalign-upstream/xenium-starmap-alignment) | compared | `A` = `3.0e-04` | 165s | 4s |
| [`merfish-visium-alignment`](notebooks/stalign-upstream/merfish-visium-alignment) | compared | `phiI` = `2.9e-04` | 241s | 7s |
| [`merfish-merfish-alignment-simulation`](notebooks/stalign-upstream/merfish-merfish-alignment-simulation) | compared | `A` = `2.4e-04` | 65s | 3s |
| [`xenium-xenium-alignment`](notebooks/stalign-upstream/xenium-xenium-alignment) | compared | `phiiJ` = `1.3e-04` | 11s | 5s |
| [`heart-alignment`](notebooks/stalign-upstream/heart-alignment) | compared | `A` = `4.9e-05` | 47s | 6s |
| [`heart-alignment-varying-thickness`](notebooks/stalign-upstream/heart-alignment-varying-thickness) | compared | `A` = `2.4e-05` | 39s | 3s |
| [`merfish-xenium-alignment`](notebooks/stalign-upstream/merfish-xenium-alignment) | compared | `A` = `2.4e-05` | 590s | 14s |
| [`merfish-merfish-alignment-using-L-T`](notebooks/stalign-upstream/merfish-merfish-alignment-using-L-T) | compared | `phiiJ` = `1.3e-05` | 800s | 17s |
| [`merfish-merfish-alignment`](notebooks/stalign-upstream/merfish-merfish-alignment) | compared | `A` = `9.8e-06` | 404s | 7s |
| [`visium-visium-alignment-affine-only`](notebooks/stalign-upstream/visium-visium-alignment-affine-only) | compared | `A` = `1.0e-06` | 42s | 3s |
| [`merfish-merfish-alignment-affine-only-with-points`](notebooks/stalign-upstream/merfish-merfish-alignment-affine-only-with-points) | compared-affine | `squidpy landmark residual` = `6.5e+02` | — | — |
| [`merfish-visium-alignment-with-curve-annotator`](notebooks/stalign-upstream/merfish-visium-alignment-with-curve-annotator) | unreplayable-upstream | — | — | — |

## Against upstream's published output

Upstream ships the aligned coordinates from its own runs beside six of the notebooks, which
is a stronger reference than either replay pass — neither pass produced it. Where the two
columns agree, the port reproduces upstream's *published* result and not merely upstream's
code. `merfish-merfish-alignment` is the one case where both sides sit `6.4e-03` from the
shipped CSV: upstream's code no longer reproduces upstream's committed output, and the port
tracks the code. These are excluded from the divergence column above, which measures the two
passes against each other.

| notebook | upstream vs published | squidpy vs published |
| --- | --- | --- |
| [`merfish-merfish-alignment-affine-only`](notebooks/stalign-upstream/merfish-merfish-alignment-affine-only) | `2.1e-16` | `1.8e-06` |
| [`merfish-merfish-alignment`](notebooks/stalign-upstream/merfish-merfish-alignment) | `6.4e-03` | `6.4e-03` |
| [`merfish-merfish-alignment-using-L-T`](notebooks/stalign-upstream/merfish-merfish-alignment-using-L-T) | `2.4e-16` | `4.6e-07` |
| [`merfish-visium-alignment`](notebooks/stalign-upstream/merfish-visium-alignment) | `1.1e-06` | `1.4e-05` |
| [`merfish-visium-alignment-with-point-annotator`](notebooks/stalign-upstream/merfish-visium-alignment-with-point-annotator) | `1.7e-06` | `2.6e-05` |
| [`visium-visium-alignment-affine-only`](notebooks/stalign-upstream/visium-visium-alignment-affine-only) | `1.8e-16` | `3.6e-08` |

## Notes recorded during the run

- **`merfish-allen3Datlas-alignment`** — This notebook fits a 3D volume to a 2D section. The two sides are *expected* to differ numerically here, unlike every 2D notebook above: upstream's 3D regularisation energy transforms two of the three spatial axes (`dim=(1,2)`, STalign.py:1504) while smoothing that same energy's gradient over all three (`dim=(1,2,3)`, :1527), so it descends on a different objective than the one it reports. Squidpy uses every spatial axis in both places. The divergence below measures that deliberate choice -- see `docs/STALIGN_DIVERGENCES.md` row D11 -- and is not a port defect.
- **`starmap-allen3Datlas-alignment`** — This notebook fits a 3D volume to a 2D section. The two sides are *expected* to differ numerically here, unlike every 2D notebook above: upstream's 3D regularisation energy transforms two of the three spatial axes (`dim=(1,2)`, STalign.py:1504) while smoothing that same energy's gradient over all three (`dim=(1,2,3)`, :1527), so it descends on a different objective than the one it reports. Squidpy uses every spatial axis in both places. The divergence below measures that deliberate choice -- see `docs/STALIGN_DIVERGENCES.md` row D11 -- and is not a port defect. Its cell [6] also reads the STARmap table through an absolute path on the notebook author's own machine (`/home/manju/Documents/...`), the only such path in the pinned notebook set; the replay rewrites the leading directories to the `../starmap_data/` convention the other notebooks use. Same file, same bytes.
- **`xenium-heimage-alignment`** — Both passes skipped 1 cell(s) unrelated to the fit -- cell 40: NameError: name 'tpointsI' is not defined
- **`merfish-visium-alignment`** — Both passes skipped 6 cell(s) unrelated to the fit -- cell 30: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!; cell 31: NameError: name 'muA' is not defined; cell 35: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!; cell 36: NameError: name 'muA' is not defined; cell 37: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!; cell 38: NameError: name 'muA' is not defined
- **`merfish-merfish-alignment-simulation`** — Both passes skipped 1 cell(s) unrelated to the fit -- cell 33: TypeError: got an unexpected keyword argument 'squared'
- **`xenium-xenium-alignment`** — The two sections overlap only partially. Unmatched cells are expected; the matching-weight panels identify the supported overlap. Both passes skipped 2 cell(s) unrelated to the fit -- cell 50: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu); cell 51: RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
- **`merfish-merfish-alignment-using-L-T`** — The notebook's second fit uses the stale `A, v, xv = LDDMM(...)` tuple API, which upstream no longer returns. The replay supplies a result that unpacks both ways, so both of its fits are compared.
- **`merfish-visium-alignment-with-curve-annotator`** — Not compared: this notebook does not run at the pinned upstream commit. Its two saved curve files hold 10 and 15 vertices, so the `L_T_from_points` call raises `Number of pointsI (10) is not equal to number of pointsJ (15)` -- and upstream's own committed output for that cell records the same exception. The replay reproduces an upstream defect; there is no fit to compare.

```{toctree}
:hidden: true
:maxdepth: 1

heart-alignment <notebooks/stalign-upstream/heart-alignment>
heart-alignment-varying-thickness <notebooks/stalign-upstream/heart-alignment-varying-thickness>
merfish-merfish-alignment <notebooks/stalign-upstream/merfish-merfish-alignment>
merfish-merfish-alignment-affine-only <notebooks/stalign-upstream/merfish-merfish-alignment-affine-only>
merfish-merfish-alignment-affine-only-with-points <notebooks/stalign-upstream/merfish-merfish-alignment-affine-only-with-points>
merfish-merfish-alignment-simulation <notebooks/stalign-upstream/merfish-merfish-alignment-simulation>
merfish-merfish-alignment-using-L-T <notebooks/stalign-upstream/merfish-merfish-alignment-using-L-T>
merfish-visium-alignment <notebooks/stalign-upstream/merfish-visium-alignment>
merfish-visium-alignment-with-curve-annotator <notebooks/stalign-upstream/merfish-visium-alignment-with-curve-annotator>
merfish-visium-alignment-with-point-annotator <notebooks/stalign-upstream/merfish-visium-alignment-with-point-annotator>
merfish-xenium-alignment <notebooks/stalign-upstream/merfish-xenium-alignment>
visium-visium-alignment-affine-only <notebooks/stalign-upstream/visium-visium-alignment-affine-only>
xenium-heimage-alignment <notebooks/stalign-upstream/xenium-heimage-alignment>
xenium-starmap-alignment <notebooks/stalign-upstream/xenium-starmap-alignment>
xenium-xenium-alignment <notebooks/stalign-upstream/xenium-xenium-alignment>
merfish-allen3Datlas-alignment <notebooks/stalign-upstream/merfish-allen3Datlas-alignment>
starmap-allen3Datlas-alignment <notebooks/stalign-upstream/starmap-allen3Datlas-alignment>
```
