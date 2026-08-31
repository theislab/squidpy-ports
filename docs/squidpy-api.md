# squidpy API equivalence

The public API — `squidpy.experimental.tl` and the two `experimental.im` helpers it needs —
is covered by these notebooks. Upstream's own notebooks call `STalign.LDDMM` and friends directly
and build their rasters, coordinate axes and initial affines by hand along the way; these are
sixteen of those seventeen analyses written against the public surface instead, one page per
upstream notebook and named after it, so ours can be read beside
[STalign's own](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks).

Nothing here reaches into a private module. Four calls cover every case:

| call | when |
| --- | --- |
| `align_landmarks` | paired points, closed-form affine, no iteration |
| `stalign_align_obs` | two point clouds; it rasterizes both sides itself |
| `stalign_align_image` | one side is an image, so the other is rasterized onto its grid |
| `stalign_align_volume` | a 2D section into a 3D reference volume |

Each notebook links the upstream analysis it mirrors, and every function it calls links to its
source. Where a notebook departs from upstream — a different raster scale, a different starting
affine, curves resampled so they can be paired at all — it says so at that point in the notebook
rather than in a list here.

## Intentional divergences from STalign

One, and it is algorithmic rather than per-notebook. At rank 3 upstream's regularisation *energy*
transforms two of three spatial axes ([`STalign.py:1504`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1504)) -- byte-identical to the
rank-2 line at [`:1193`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1193), where two axes is all of them -- while the gradient it
descends smooths all three ([`:1527`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1527)): a rank-2 line reused without extending it to
the new axis. squidpy transforms all three, which moves the fitted velocity field by 31x and is
why `sigmaR` is retuned from upstream's `1e8` to `1e6`. Unverified against the paper
([Clifton et al. 2023](https://doi.org/10.1038/s41467-023-43915-7)), and pinned by a strict xfail
in `tests/test_reference.py` rather than asserted away.

## A section into a reference volume

```{toctree}
:maxdepth: 1

notebooks/squidpy-api/starmap-allen3Datlas
notebooks/squidpy-api/merfish-allen3Datlas
```

## Two point clouds

```{toctree}
:maxdepth: 1

notebooks/squidpy-api/merfish-merfish
notebooks/squidpy-api/merfish-merfish-initial-affine
notebooks/squidpy-api/merfish-merfish-affine-only
notebooks/squidpy-api/merfish-merfish-affine-only-with-points
notebooks/squidpy-api/merfish-xenium
notebooks/squidpy-api/xenium-xenium
notebooks/squidpy-api/xenium-starmap
notebooks/squidpy-api/visium-visium-affine-only
notebooks/squidpy-api/heart-alignment
notebooks/squidpy-api/heart-alignment-varying-thickness
```

## Points onto an image

```{toctree}
:maxdepth: 1

notebooks/squidpy-api/merfish-visium
notebooks/squidpy-api/merfish-visium-with-point-annotator
notebooks/squidpy-api/merfish-visium-with-curve-annotator
notebooks/squidpy-api/xenium-heimage
```

The one upstream notebook without an equivalent here is `merfish-merfish-alignment-simulation`,
which fits a section against a synthetically deformed copy of itself rather than against another
section.
