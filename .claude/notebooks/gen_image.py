"""Generate the three image-route notebooks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_both import code, md, write

HE_LOAD = """
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel
from squidpy.experimental.im import rasterize_points
from squidpy.experimental.tl import align_stalign_image

def as_image(rgb, key):
    return sd.SpatialData(images={key: Image2DModel.parse(
        np.moveaxis(rgb, -1, 0).astype(float), dims=('c', 'y', 'x'))})

def rasterized(xy, dx):
    sdata = sd.SpatialData(points={'cells': PointsModel.parse(xy)})
    rasterize_points(sdata, 'cells', dx=dx, blur=1.0, key_added='section')
    return sdata
"""

TRACE = """
MIXTURE_GATE = 50
energies = np.asarray(fit.energies)[: fit.n_iter]
descent = energies[MIXTURE_GATE + 1 :]
tail = descent[-max(len(descent) // 10, 1) :]
print(f'after the gate: {descent[0]:.0f} -> {descent[-1]:.0f}, minimum {descent.min():.0f} '
      f'at iteration {MIXTURE_GATE + 1 + int(descent.argmin())}')
print(f'last tenth: mean {tail.mean():.0f}, spread {np.ptp(tail):.0f} '
      f'({100 * np.ptp(tail) / tail.mean():.1f}% of its mean)')
plt.plot(energies, lw=0.8); plt.axvline(MIXTURE_GATE, color='0.6', ls='--', lw=0.8)
plt.xlabel('iteration'); plt.ylabel('objective'); plt.grid(alpha=0.3)
"""

# ------------------------------------------------------------------ xenium-heimage
write(
    "docs/notebooks/squidpy-api/xenium-heimage.ipynb",
    [
        md("""
# Placing Xenium cells on their H&E image

The same route as [merfish-visium](merfish-visium.ipynb): the cell cloud is rasterized and
`align_stalign_image` fits image to image, with paired landmarks carrying the correspondence
that intensity alone cannot.

Upstream's equivalent is `xenium-heimage-alignment`, and it runs the pair the other way round --
it warps the H&E onto the rasterized cells and then inverts to place the cells. Here the H&E is
the reference, which is the direction that puts the cells on the image without an inverse.
"""),
        md("## Inputs"),
        code(
            HE_LOAD
            + """
he = plt.imread('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png')[..., :3]
cells = pd.read_csv('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz')
xy = np.c_[cells['x_centroid'], cells['y_centroid']].astype(float)

visium = as_image(he, 'he')
xenium = rasterized(xy, 30.0)
print(f'{len(xy)} cells over {xy[:, 0].max():.0f} x {xy[:, 1].max():.0f} um, '
      f'rasterized to {tuple(np.asarray(xenium["section"]).shape)}; H&E is {he.shape}')
"""
        ),
        md("""
Four landmark pairs, hardcoded upstream. They are stored there as `(y, x)` -- squidpy's public
API takes `(x, y)`, so each is reversed once on the way in rather than the arrays being
transposed later.
"""),
        code("""
landmarks_he = np.array([[1050., 950.], [700., 2200.], [500., 1550.], [1550., 1840.]])[:, ::-1]
landmarks_cells = np.array([[3108., 2100.], [4480., 6440.], [5040., 4200.], [1260., 5320.]])[:, ::-1]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].imshow(he); ax[0].scatter(*landmarks_he.T, s=40, c='red')
ax[0].set_title('H&E, in pixels')
ax[1].scatter(*xy.T, s=0.12, alpha=0.3); ax[1].scatter(*landmarks_cells.T, s=40, c='red')
ax[1].set_title('Xenium cells, in microns'); ax[1].invert_yaxis(); ax[1].set_aspect('equal')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("## The fit\n\nUpstream's own solver values for this pair."),
        code("""
fit = align_stalign_image(
    visium, xenium, image_key=('he', 'section'),
    landmarks_ref=landmarks_he, landmarks_query=landmarks_cells,
    niter=2000, sigmaM=0.15, sigmaB=0.10, sigmaA=0.11, epV=10,
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
        md("## Every cell, placed on the image"),
        code("""
placed = np.asarray(fit.transform(xy))
residual = np.linalg.norm(np.asarray(fit.transform(landmarks_cells)) - landmarks_he, axis=1)
rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'landmark residual: median {np.median(residual):.1f} px, worst {residual.max():.1f} px')
print(f'{100 * inside.mean():.0f}% of cells land within the {columns} x {rows} image')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].set_title('H&E')
ax[1].imshow(he); ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].scatter(*landmarks_he.T, s=40, c='red', label='target landmarks')
ax[1].set_title('Xenium cells placed on it'); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("## The objective's trace"),
        code(TRACE),
    ],
)

# ----------------------------------------- merfish-visium-with-point-annotator
write(
    "docs/notebooks/squidpy-api/merfish-visium-with-point-annotator.ipynb",
    [
        md("""
# MERFISH onto Visium H&E, from annotated points

The same pair as [merfish-visium](merfish-visium.ipynb), with landmarks picked interactively
rather than hardcoded: five named regions, several points each, saved per side.

Upstream's equivalent is `merfish-visium-alignment-with-point-annotator`. Its own annotator is a
plotly click handler; what it saves is two dicts keyed by region name, and those are the input
here.
"""),
        md("## Inputs"),
        code(
            HE_LOAD
            + """
MERFISH = ('merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase'
           '_Slice2_Replicate3_cell_metadata_S2R3.csv.gz')
cells = pd.read_csv(MERFISH)
xy = np.c_[cells['center_x'], cells['center_y']].astype(float)

he = plt.imread('visium_data/tissue_hires_image.png')[..., :3]
visium = as_image(he, 'he')
merfish = rasterized(xy, 30.0)
print(f'{len(xy)} cells rasterized to {tuple(np.asarray(merfish["section"]).shape)}, '
      f'H&E is {he.shape}')
"""
        ),
        md("""
Row order is the correspondence, so both sides are flattened in the same key order. Each side's
points are in that side's own units -- microns against pixels -- and nothing restates them: the
elements carry their placement and the solver reads it off them.
"""),
        code("""
picked = {side: np.load(f'visium_data/{name}_points.npy', allow_pickle=True).item()
          for side, name in (('query', 'Merfish_S2_R3'), ('ref', 'tissue_hires_image'))}
regions = list(picked['ref'])
paired = {side: np.array([p for r in regions for p in picked[side][r]], dtype=float)
          for side in picked}
print(f'{len(paired["ref"])} pairs over {len(regions)} regions: {", ".join(regions)}')
"""),
        md("## The fit\n\nUpstream's own solver values, which differ from the hardcoded-landmark variant."),
        code("""
fit = align_stalign_image(
    visium, merfish, image_key=('he', 'section'),
    landmarks_ref=paired['ref'], landmarks_query=paired['query'],
    niter=200, sigmaM=0.18, sigmaB=0.18, sigmaA=0.18, sigmaP=2e-1,
    epL=5e-11, epT=5e-4, epV=5e1,
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
        md("""
## Every cell, placed on the image

The H&E covers one hemisphere while the MERFISH section is a whole coronal slice, so part of it
has no tissue to land on. All the landmarks sit on the covered half, which is why the residual
below is small whatever happens to the other one.
"""),
        code("""
placed = np.asarray(fit.transform(xy))
residual = np.linalg.norm(np.asarray(fit.transform(paired['query'])) - paired['ref'], axis=1)
rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'landmark residual: median {np.median(residual):.1f} px, worst {residual.max():.1f} px')
print(f'{100 * inside.mean():.0f}% of cells land within the {columns} x {rows} image')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=30, c='red')
ax[0].set_title('Visium H&E with its annotated points')
ax[1].imshow(he); ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].set_title('MERFISH cells placed on it')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("## The objective's trace"),
        code(TRACE),
    ],
)

# ----------------------------------------- merfish-visium-with-curve-annotator
write(
    "docs/notebooks/squidpy-api/merfish-visium-with-curve-annotator.ipynb",
    [
        md("""
# MERFISH onto Visium H&E, from annotated curves

Landmarks traced as curves rather than clicked as points. A curve is a correspondence between
two *shapes*, not between two lists of vertices, and that distinction is the whole content of
this notebook.

**Upstream's own version of this notebook does not run.** Its two saved curve files hold 10 and
15 vertices, so `L_T_from_points` raises `Number of pointsI (10) is not equal to number of
pointsJ (15)` -- and upstream's committed output records that same exception. Squidpy raises
too, and for the same good reason: paired landmarks are matched by row, so unequal counts have
no meaning. The fix belongs to the caller, and it is to resample.
"""),
        md("## Inputs"),
        code(
            HE_LOAD
            + """
MERFISH = ('merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase'
           '_Slice2_Replicate3_cell_metadata_S2R3.csv.gz')
cells = pd.read_csv(MERFISH)
xy = np.c_[cells['center_x'], cells['center_y']].astype(float)

he = plt.imread('visium_data/tissue_hires_image.png')[..., :3]
visium = as_image(he, 'he')
merfish = rasterized(xy, 30.0)

traced = {side: np.load(f'visium_data/{name}_curves.npy', allow_pickle=True).item()
          for side, name in (('query', 'Merfish_S2_R3'), ('ref', 'tissue_hires_image'))}
for name in traced['ref']:
    print(f'{name:<8} ref {len(traced["ref"][name]):>3} vertices, '
          f'query {len(traced["query"][name]):>3}')
"""
        ),
        md("""
## Resampling the curves

Each curve is resampled to the same number of points, evenly along its own arc length. That is
what makes the two sides comparable: vertex *k* of one curve then corresponds to vertex *k* of
the other because both are the same fraction along the shape, which is not true of whatever
vertex count the annotator happened to record.
"""),
        code("""
def resampled(curve, n):
    points = np.asarray(curve, dtype=float)
    walked = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    even = np.linspace(0.0, walked[-1], n)
    return np.c_[np.interp(even, walked, points[:, 0]), np.interp(even, walked, points[:, 1])]

PER_CURVE = 8
paired = {side: np.vstack([resampled(traced[side][name], PER_CURVE) for name in traced['ref']])
          for side in traced}
print(f'{len(paired["ref"])} pairs from {len(traced["ref"])} curves at {PER_CURVE} points each')

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=18, c='red')
ax[0].set_title('Visium H&E, curves resampled')
ax[1].scatter(*xy.T, s=0.12, alpha=0.3); ax[1].scatter(*paired['query'].T, s=18, c='red')
ax[1].set_title('MERFISH section, curves resampled')
ax[1].invert_yaxis(); ax[1].set_aspect('equal')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("## The fit"),
        code("""
fit = align_stalign_image(
    visium, merfish, image_key=('he', 'section'),
    landmarks_ref=paired['ref'], landmarks_query=paired['query'],
    niter=200, sigmaM=0.18, sigmaB=0.18, sigmaA=0.18, sigmaP=2e-1,
    epL=5e-11, epT=5e-4, epV=5e1,
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
        md("## Every cell, placed on the image"),
        code("""
placed = np.asarray(fit.transform(xy))
residual = np.linalg.norm(np.asarray(fit.transform(paired['query'])) - paired['ref'], axis=1)
rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'landmark residual: median {np.median(residual):.1f} px, worst {residual.max():.1f} px')
print(f'{100 * inside.mean():.0f}% of cells land within the {columns} x {rows} image')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=18, c='red')
ax[0].set_title('Visium H&E')
ax[1].imshow(he); ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].set_title('MERFISH cells placed on it')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("## The objective's trace"),
        code(TRACE),
    ],
)
