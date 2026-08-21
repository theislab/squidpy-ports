# The same alignments, through squidpy's public API

Upstream's notebooks call `STalign.LDDMM` and friends directly, and build their own rasters,
coordinate axes and initial affines along the way. These are the same seventeen analyses written
against squidpy's public API instead — one page per upstream notebook, named after it, so ours can
be read beside [the replayed originals](parity.md).

Nothing here reaches into a private module. Four calls cover every case:

| call | when |
| --- | --- |
| `align_landmarks` | paired points, closed-form affine, no iteration |
| `align_stalign_obs` | two point clouds; it rasterizes both sides itself |
| `align_stalign_image` | one side is an image, so the other is rasterized onto its grid |
| `align_stalign_volume` | a 2D section into a 3D reference volume |

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
notebooks/squidpy-api/merfish-merfish-using-L-T
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

## What differs from upstream, and why

These are re-expressions, not transcriptions. Where they depart from the upstream notebook, the
notebook says so in place:

- **Solver values are upstream's own** wherever upstream sets them. The exception is
  `heart-alignment`, where upstream's `niter=1000` leaves the two sections 1742 → 1587 um apart;
  the default 5000 closes it to 238. Its other overrides are kept.
- **`visium-visium-affine-only` reads its spot files with `header=None`.** They carry no header, so
  a plain `read_csv` promotes the first spot to column names and silently drops it.
- **`xenium-starmap` puts STARmap into the Xenium frame first** — axes swapped, divided by 5, the
  new y flipped, as upstream does. Read as-is the clouds sit ~16 mm apart and the affine's
  translation step is 0.2 units per iteration, so no iteration budget closes it.
- **`xenium-heimage` runs the pair the other way round.** Upstream warps the H&E onto the
  rasterized cells and inverts to place them; here the H&E is the reference, which puts the cells
  on the image without an inverse.
- **`merfish-visium-with-curve-annotator` resamples its curves.** Upstream's own run of this
  notebook raises — its two curve files hold 10 and 15 vertices, and paired landmarks are matched
  by row. Each curve is resampled to a common count along its own arc length, so vertex *k* on one
  side is the same fraction along the shape as vertex *k* on the other.

Every notebook here was executed on one H100, and the executed copy is what is committed. The
generators that produce them are in `.claude/notebooks/`.
