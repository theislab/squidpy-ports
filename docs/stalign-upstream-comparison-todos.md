# STalign upstream notebook comparison status

The comparison harness covers all 17 notebooks from upstream STalign at
`b2068edc98974efa54537eca194736e177bbe11d`. It runs the original preprocessing,
captures the upstream fit inputs, and compares the PyTorch result with Squidpy's
JAX implementation on one H100 allocation.

## Results preserved from job 38938878

- [3D Allen atlas status notebook](notebooks/stalign-upstream/results-partial-38938878/notebooks/merfish-allen3Datlas-alignment.ipynb)
  and [status image](notebooks/stalign-upstream/results-partial-38938878/merfish-allen3Datlas-alignment-comparison.png).
- [Affine landmark comparison notebook](notebooks/stalign-upstream/results-partial-38938878/notebooks/merfish-merfish-alignment-affine-only-with-points.ipynb)
  and [comparison image](notebooks/stalign-upstream/results-partial-38938878/merfish-merfish-alignment-affine-only-with-points-comparison.png).
- The affine comparison recorded a linear relative L2 difference of
  `0.0034826`, translation relative L2 difference of `0.0037152`, and median
  aligned-landmark delta of `7.0977` coordinate units.
- The result directory also contains package versions, H100 details, GPU
  monitoring, manifests, metrics, and the partial suite log.

## TODO

- [ ] Preserve Python scalar types in `notebook_suite._jax_kwargs`. The current
  generic conversion turns static arguments such as `niter=100` into 0-D NumPy
  arrays, which JAX cannot hash as static JIT arguments.
- [ ] Add a regression test for captured scalar, array, affine, point, fixed
  `muA`/`muB`, and warm-start velocity arguments before another cluster run.
- [ ] Emit a failure notebook and full traceback for every failed comparison so
  a partial cluster run remains self-explanatory.
- [ ] Rerun all 14 two-dimensional LDDMM notebooks in the single `gpu_normal`
  H100 batch job and commit their executed notebooks, PNGs, metrics, and
  manifests.
- [ ] Review numerical divergence notebook by notebook, with particular
  attention to the partially overlapping Xenium/Xenium sections and matching
  weights rather than plot orientation alone.
- [ ] Expose and implement a separate 3D-volume-to-2D-slice estimator before
  numerically comparing `merfish-allen3Datlas-alignment.ipynb` and
  `starmap-allen3Datlas-alignment.ipynb`; the current Squidpy port only supports
  two-dimensional LDDMM.
- [ ] Replace the generated wrapper notebooks with the final executed versions
  after the complete suite succeeds.
