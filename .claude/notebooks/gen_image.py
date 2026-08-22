"""Generate the three image-route notebooks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_both import code, md, write

HE_LOAD = """
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations import get_transformation
from squidpy.experimental.im import rasterize_points
from squidpy.experimental.tl import align_stalign_image, stalign_transform_points

# `sigmaM`, `sigmaB` and `sigmaA` are in the images' own intensity units, and upstream states
# its values for both sides mapped onto [0, 1]. A raw density raster of 167k cells has a mean
# near 3 and a max well above 10, which leaves widths of ~0.1 about a hundred times too tight:
# the fit then converges on an objective that has stopped measuring the overlap.
def unit(a):
    a = np.asarray(a, dtype=float)
    return (a - a.min()) / np.ptp(a)

def as_image(rgb, key):
    return sd.SpatialData(images={key: Image2DModel.parse(
        unit(np.moveaxis(rgb, -1, 0)), dims=('c', 'y', 'x'))})

def rasterized(xy, dx):
    sdata = sd.SpatialData(points={'cells': PointsModel.parse(xy)})
    rasterize_points(sdata, 'cells', dx=dx, blur=1.0, key_added='section')
    element = sdata.images['section']
    sdata.images['section'] = Image2DModel.parse(
        unit(np.asarray(element)), dims=('c', 'y', 'x'),
        transformations={'global': get_transformation(element, 'global')})
    return sdata
"""

# ------------------------------------------------------------------ xenium-heimage
write(
    "docs/notebooks/squidpy-api/xenium-heimage.ipynb",
    [
        md("""
# Aligning Xenium cells with their H&E image

`align_stalign_image` fits image to image, so the cell cloud is rasterized first and the H&E is
matched against that raster. Upstream's equivalent is `xenium-heimage-alignment`.

**Which side is the reference matters here, and it is not a presentation choice.** The objective
is computed on the reference's grid. Make the H&E the reference and it is evaluated over 2051 x
2759 x 3 pixels against a 201 x 276 section -- roughly three hundred times more reference pixels
than there is section to match -- and the deformation gets driven by H&E texture with no
counterpart: measured, the landmarks start 16 px apart and the fit walks them out to 524. With
the raster as the reference the same fit starts at an objective of 14,352 instead of 4,417,134
and ends *better* than it started. So the rasterized cells are the reference, as upstream has it.
"""),
        md("## Inputs"),
        code(
            HE_LOAD
            + """
he = plt.imread('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.png')[..., :3]
cells = pd.read_csv('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz')
xy = np.c_[cells['x_centroid'], cells['y_centroid']].astype(float)

image = as_image(he, 'he')
section = rasterized(xy, 30.0)
print(f'{len(xy)} cells over {xy[:, 0].max():.0f} x {xy[:, 1].max():.0f} um, '
      f'rasterized to {tuple(np.asarray(section["section"]).shape)}; H&E is {he.shape}')
"""
        ),
        md("""
Four landmark pairs, hardcoded upstream. They are stored there as `(y, x)`; squidpy's public API
takes `(x, y)`, so each is reversed once on the way in. Only one reading is even possible --
read as `(x, y)`, the second H&E point's 2200 would exceed the image's 2051 height.
"""),
        code("""
landmarks_he = np.array([[1050., 950.], [700., 2200.], [500., 1550.], [1550., 1840.]])[:, ::-1]
landmarks_cells = np.array([[3108., 2100.], [4480., 6440.], [5040., 4200.], [1260., 5320.]])[:, ::-1]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].imshow(he); ax[0].scatter(*landmarks_he.T, s=12, c='red')
ax[0].set_title('H&E, in pixels')
ax[1].scatter(*xy.T, s=0.12, alpha=0.3); ax[1].scatter(*landmarks_cells.T, s=12, c='red')
ax[1].set_title('Xenium cells, in microns'); ax[1].invert_yaxis(); ax[1].set_aspect('equal')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("""
## The fit

Upstream's own solver values. `niter=0` is not used here: with the smaller grid as the reference
it raises `IndexError` inside the initialisation path, so the starting affine cannot be inspected
the way the volume notebooks inspect theirs.
"""),
        code("""
fit = align_stalign_image(
    section, image, image_key=('section', 'he'),
    landmarks_ref=landmarks_cells, landmarks_query=landmarks_he,
    niter=2000, sigmaM=0.15, sigmaB=0.10, sigmaA=0.11, epV=10,
)
print(f'{fit["n_iter"]} iterations, objective '
      f'{float(fit["energies"][0]):.0f} -> {float(fit["energies"][-1]):.0f}')

residual = np.linalg.norm(np.asarray(stalign_transform_points(fit, landmarks_he)) - landmarks_cells, axis=1)
print(f'landmark residual: median {np.median(residual):.1f} um, worst {residual.max():.1f} um')
"""),
        md("""
## The two together

`stalign_transform_points` maps the query into the reference frame, which here is H&E pixels into microns --
the opposite of what a picture of cells-on-tissue wants. `stalign_warp_image` supplies the other
direction: `backward` resamples a reference-frame image onto the query's grid, so the cell
density lands on the H&E's own pixels. That is the figure upstream publishes for this pair.

Going the other way for *points* -- cells into H&E pixels -- is not available: `stalign_transform_points` only
runs query to reference, and the public API exposes no inverse for a point set.
"""),
        code("""
def physical_axes(element):
    matrix = get_transformation(element, 'global').to_affine_matrix(
        input_axes=('y', 'x'), output_axes=('y', 'x'))
    return tuple((np.asarray(element.coords[a]) - 0.5) * matrix[k, k] + matrix[k, -1]
                 for k, a in enumerate(('y', 'x')))

# The fit runs on upstream's dx=30 raster -- 201 x 276 -- and resampling that onto 2051 x 2759
# is a tenfold upsample, so it arrives soft no matter how it is drawn. `stalign_warp_image` takes
# explicit axes for exactly this: the same fitted deformation can carry a finer raster, which
# has the detail to survive the trip. The fit is unchanged; only what is pushed through it is.
from squidpy.experimental.tl import stalign_warp_image

display = rasterized(xy, 8.0)
density = np.asarray(stalign_warp_image(fit,
    np.asarray(display['section']), direction='backward',
    ref_axes=physical_axes(display['section']))).squeeze()
print(f'fitted on {tuple(np.asarray(section["section"]).shape)}, '
      f'displayed through a {tuple(np.asarray(display["section"]).shape)} raster '
      f'-> {density.shape}')

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].imshow(he); ax[0].set_title('H&E')
ax[1].imshow(he)
# Clipped at a high percentile rather than the max: a few dense cores otherwise take the whole
# colour range and flatten everything else to nothing.
ax[1].imshow(density, cmap='Blues', alpha=0.55, vmin=0,
             vmax=np.percentile(density[density > 0], 99))
ax[1].scatter(*landmarks_he.T, s=12, c='red', label='landmarks')
ax[1].set_title('Xenium cell density warped onto it'); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
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
print(f'{fit["n_iter"]} iterations, objective '
      f'{float(fit["energies"][0]):.0f} -> {float(fit["energies"][-1]):.0f}')
"""),
        md("""
## Every cell, placed on the image

The H&E covers one hemisphere while the MERFISH section is a whole coronal slice, so part of it
has no tissue to land on. All the landmarks sit on the covered half, which is why the residual
below is small whatever happens to the other one.
"""),
        code("""
placed = np.asarray(stalign_transform_points(fit, xy))
residual = np.linalg.norm(np.asarray(stalign_transform_points(fit, paired['query'])) - paired['ref'], axis=1)
rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'landmark residual: median {np.median(residual):.1f} px, worst {residual.max():.1f} px')
print(f'{100 * inside.mean():.0f}% of cells land within the {columns} x {rows} image')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=12, c='red')
ax[0].set_title('Visium H&E with its annotated points')
ax[1].imshow(he); ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].set_title('MERFISH cells placed on it')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
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
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=12, c='red')
ax[0].set_title('Visium H&E, curves resampled')
ax[1].scatter(*xy.T, s=0.12, alpha=0.3); ax[1].scatter(*paired['query'].T, s=12, c='red')
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
print(f'{fit["n_iter"]} iterations, objective '
      f'{float(fit["energies"][0]):.0f} -> {float(fit["energies"][-1]):.0f}')
"""),
        md("## Every cell, placed on the image"),
        code("""
placed = np.asarray(stalign_transform_points(fit, xy))
residual = np.linalg.norm(np.asarray(stalign_transform_points(fit, paired['query'])) - paired['ref'], axis=1)
rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'landmark residual: median {np.median(residual):.1f} px, worst {residual.max():.1f} px')
print(f'{100 * inside.mean():.0f}% of cells land within the {columns} x {rows} image')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].scatter(*paired['ref'].T, s=12, c='red')
ax[0].set_title('Visium H&E')
ax[1].imshow(he); ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].set_title('MERFISH cells placed on it')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
    ],
)
