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
| [`xenium-heimage-alignment`](notebooks/stalign-upstream/xenium-heimage-alignment) | compared | `v` = `3.1e-01` | 141s | 6s |
| [`merfish-visium-alignment-with-point-annotator`](notebooks/stalign-upstream/merfish-visium-alignment-with-point-annotator) | compared | `testM` = `5.5e-03` | 240s | 7s |
| [`merfish-merfish-alignment-affine-only`](notebooks/stalign-upstream/merfish-merfish-alignment-affine-only) | compared | `A` = `6.3e-04` | 57s | 4s |
| [`xenium-starmap-alignment`](notebooks/stalign-upstream/xenium-starmap-alignment) | compared | `A` = `3.0e-04` | 148s | 4s |
| [`merfish-visium-alignment`](notebooks/stalign-upstream/merfish-visium-alignment) | compared | `phiI` = `3.0e-04` | 238s | 6s |
| [`merfish-merfish-alignment-simulation`](notebooks/stalign-upstream/merfish-merfish-alignment-simulation) | compared | `A` = `2.4e-04` | 55s | 3s |
| [`xenium-xenium-alignment`](notebooks/stalign-upstream/xenium-xenium-alignment) | compared | `phiiJ` = `1.3e-04` | 19s | 7s |
| [`heart-alignment`](notebooks/stalign-upstream/heart-alignment) | compared | `A` = `4.9e-05` | 41s | 3s |
| [`heart-alignment-varying-thickness`](notebooks/stalign-upstream/heart-alignment-varying-thickness) | compared | `A` = `2.4e-05` | 58s | 4s |
| [`merfish-xenium-alignment`](notebooks/stalign-upstream/merfish-xenium-alignment) | compared | `A` = `2.4e-05` | 566s | 12s |
| [`merfish-merfish-alignment-using-L-T`](notebooks/stalign-upstream/merfish-merfish-alignment-using-L-T) | compared | `phiiJ` = `1.3e-05` | 824s | 21s |
| [`merfish-merfish-alignment`](notebooks/stalign-upstream/merfish-merfish-alignment) | compared | `A` = `9.8e-06` | 384s | 9s |
| [`visium-visium-alignment-affine-only`](notebooks/stalign-upstream/visium-visium-alignment-affine-only) | compared | `A` = `1.0e-06` | 32s | 4s |
| [`merfish-merfish-alignment-affine-only-with-points`](notebooks/stalign-upstream/merfish-merfish-alignment-affine-only-with-points) | compared-affine | `squidpy landmark residual` = `6.5e+02` | — | — |
| [`merfish-visium-alignment-with-curve-annotator`](notebooks/stalign-upstream/merfish-visium-alignment-with-curve-annotator) | unreplayable-upstream | — | — | — |
| [`merfish-allen3Datlas-alignment`](notebooks/stalign-upstream/merfish-allen3Datlas-alignment) | unsupported-3d | — | — | — |
| [`starmap-allen3Datlas-alignment`](notebooks/stalign-upstream/starmap-allen3Datlas-alignment) | unsupported-3d | — | — | — |

## Notes recorded during the run

- **`xenium-heimage-alignment`** — Both passes skipped 1 cell(s) unrelated to the fit -- cell 40: NameError: name 'tpointsI' is not defined
- **`merfish-merfish-alignment-affine-only`** — Both passes skipped 5 cell(s) unrelated to the fit -- cell 25: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 27: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 31: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 32: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 34: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
- **`merfish-visium-alignment`** — Both passes skipped 10 cell(s) unrelated to the fit -- cell 26: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 30: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 31: NameError: name 'muA' is not defined; cell 35: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 36: NameError: name 'muA' is not defined; cell 37: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 38: NameError: name 'muA' is not defined; cell 42: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 46: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 48: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
- **`merfish-merfish-alignment-simulation`** — Both passes skipped 4 cell(s) unrelated to the fit -- cell 27: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 29: NameError: name 'xI_LDDMM' is not defined; cell 31: NameError: name 'xI_LDDMM' is not defined; cell 33: TypeError: got an unexpected keyword argument 'squared'
- **`xenium-xenium-alignment`** — The two sections overlap only partially. Unmatched cells are expected; the matching-weight panels identify the supported overlap.
- **`merfish-xenium-alignment`** — Both passes skipped 4 cell(s) unrelated to the fit -- cell 19: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 21: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 25: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 27: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
- **`merfish-merfish-alignment-using-L-T`** — The notebook's second fit uses the stale `A, v, xv = LDDMM(...)` tuple API, which upstream no longer returns. The replay supplies a result that unpacks both ways, so both of its fits are compared. Both passes skipped 7 cell(s) unrelated to the fit -- cell 25: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 27: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 31: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 32: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 34: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 47: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 51: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
- **`visium-visium-alignment-affine-only`** — Both passes skipped 5 cell(s) unrelated to the fit -- cell 25: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 27: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 31: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 32: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.; cell 34: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
- **`merfish-visium-alignment-with-curve-annotator`** — Not compared: this notebook does not run at the pinned upstream commit. Its two saved curve files hold 10 and 15 vertices, so the `L_T_from_points` call raises `Number of pointsI (10) is not equal to number of pointsJ (15)` -- and upstream's own committed output for that cell records the same exception. The replay reproduces an upstream defect; there is no fit to compare.
- **`merfish-allen3Datlas-alignment`** — Not numerically compared: this upstream notebook calls LDDMM_3D_to_slice. Squidpy currently implements the 2D LDDMM solver only; a separate volume-to-image estimator is required for an honest comparison.
- **`starmap-allen3Datlas-alignment`** — Not numerically compared: this upstream notebook calls LDDMM_3D_to_slice. Squidpy currently implements the 2D LDDMM solver only; a separate volume-to-image estimator is required for an honest comparison.

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
