# squidpy API equivalence

The public API — `squidpy.experimental.tl.align` and the two `experimental.im` helpers it needs —
is covered by these notebooks. Upstream's own notebooks call `STalign.LDDMM` and friends directly
and build their rasters, coordinate axes and initial affines by hand along the way; these are
sixteen of those seventeen analyses written against the public surface instead, one page per
upstream notebook and named after it, so ours can be read beside
[STalign's own](https://github.com/JEFworks-Lab/STalign/tree/b2068edc98974efa54537eca194736e177bbe11d/docs/notebooks).

Nothing here reaches into a private module. Four calls cover every case:

| call | when |
| --- | --- |
| `align_landmarks` | paired points, closed-form affine, no iteration |
| `align_stalign_obs` | two point clouds; it rasterizes both sides itself |
| `align_stalign_image` | one side is an image, so the other is rasterized onto its grid |
| `align_stalign_volume` | a 2D section into a 3D reference volume |

Each notebook links the upstream analysis it mirrors, and every function it calls links to its
source. Where a notebook departs from upstream — a different raster scale, a different starting
affine, curves resampled so they can be paired at all — it says so at that point in the notebook
rather than in a list here.

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
