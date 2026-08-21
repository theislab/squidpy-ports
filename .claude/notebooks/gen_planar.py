import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_both import code, md, write

LOAD = """
import anndata as ad, numpy as np, pandas as pd

MERFISH = 'merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate'

def section(replicate):
    df = pd.read_csv(f'{MERFISH}{replicate}_cell_metadata_S2R{replicate}.csv.gz')
    xy = np.c_[df['center_x'], df['center_y']].astype(float)
    return ad.AnnData(X=np.zeros((len(xy), 1)), obsm={'spatial': xy})

# S2R2 is the reference; S2R3 is the section that moves.
ref, query = section(2), section(3)

# Thirteen landmark pairs picked by hand, stored as `(x, y)` -- which is what squidpy's public
# API takes, so unlike upstream's own notebook nothing here transposes them on the way in.
landmarks = {r: np.asarray(np.load(f'merfish_data/Merfish_S2_R{r}_points.npy',
                                   allow_pickle=True).item()['all'], dtype=float)
             for r in (2, 3)}

# They stay plain arrays: landmarks are correspondences *between* the two sections rather
# than observations *of* either, so they have no `obs` axis to hang off -- and every
# function here takes them as arrays.
print(f'{ref.n_obs} reference cells, {query.n_obs} query cells, '
      f'{len(landmarks[2])} landmark pairs')
"""

# ---------------------------------------------------------------- merfish-merfish
write(
    "docs/notebooks/squidpy-api/merfish-merfish.ipynb",
    [
        md("""
# Aligning two MERFISH sections

Two replicate sections of the same tissue brought into a common frame, entirely through
squidpy's public API: `align_stalign_obs` fits a diffeomorphism directly between two point
clouds, rasterizing both sides itself.

Upstream's equivalent is `merfish-merfish-alignment`, which rotates the source by hand before
rasterizing. Here the paired landmarks supply the starting affine instead.
"""),
        md("## Inputs"),
        code(LOAD),
        md("""
## The fit

Handing over the landmarks does two separate things: they derive the starting affine, and they
stay in the objective as a matching term weighted by `sigmaP` -- so the fit is pulled toward
them rather than merely started there.

`dx` and `blur` are the rasterization the solver does internally; upstream's notebook uses
30 um and 1.5, and `epV=50` is its one departure from the solver defaults.
"""),
        code("""
from squidpy.experimental.tl import align_stalign_obs

fit = align_stalign_obs(
    ref, query, spatial_key='spatial',
    landmarks_ref=landmarks[2], landmarks_query=landmarks[3],
    dx=30.0, blur=1.5, niter=10000, epV=50,
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
        md("""
## Where the cells land

`transform` evaluates the fitted map at each point, so a cell lands where it lands rather than
at the nearest raster cell. The middle panel is what the landmark affine alone achieves, for
comparison -- the difference between the two is what the diffeomorphism bought.
"""),
        code("""
import matplotlib.pyplot as plt
from squidpy.experimental.tl import align_landmarks

moved = np.asarray(fit.transform(query.obsm['spatial']))
affine = align_landmarks(landmarks[2], landmarks[3], fit='affine')
affine_only = query.obsm['spatial'] @ affine[:2, :2].T + affine[:2, 2]

fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
for a, (pts, title) in zip(ax, [
        (query.obsm['spatial'], 'before'),
        (affine_only, 'after the landmark affine'),
        (moved, 'after the diffeomorphism')], strict=True):
    a.scatter(*ref.obsm['spatial'].T, s=0.12, alpha=0.3, label='reference (S2R2)')
    a.scatter(*pts.T, s=0.12, alpha=0.3, label='query (S2R3)')
    a.set_title(title); a.set_aspect('equal'); a.invert_yaxis()
    a.set_xticks([]); a.set_yticks([])
ax[0].legend(markerscale=90, loc='lower left', fontsize=8)
"""),
        md("""
The objective's own trace. As in the volume notebooks, the mixture E step switches on at
iteration 50 and the energy changes definition there, so only the part after the dashed line
is one function.
"""),
        code("""
MIXTURE_GATE = 50
energies = np.asarray(fit.energies)[: fit.n_iter]
descent = energies[MIXTURE_GATE + 1 :]
tail = descent[-len(descent) // 10 :]
print(f'after the gate: {descent[0]:.0f} -> {descent[-1]:.0f}, minimum {descent.min():.0f} '
      f'at iteration {MIXTURE_GATE + 1 + int(descent.argmin())}')
print(f'last tenth: mean {tail.mean():.0f}, spread {np.ptp(tail):.0f} '
      f'({100 * np.ptp(tail) / tail.mean():.1f}% of its mean)')
plt.plot(energies, lw=0.8); plt.axvline(MIXTURE_GATE, color='0.6', ls='--', lw=0.8)
plt.xlabel('iteration'); plt.ylabel('objective'); plt.grid(alpha=0.3)
"""),
    ],
)

# ------------------------------------- merfish-merfish-affine-only-with-points
write(
    "docs/notebooks/squidpy-api/merfish-merfish-affine-only-with-points.ipynb",
    [
        md("""
# Aligning two MERFISH sections from landmarks alone

The cheapest of the alignments: `align_landmarks` solves for the affine in closed form from
paired points. No iteration, no rasterization, no images -- and no diffeomorphism, so it can
only express what an affine can.

Upstream's equivalent is `merfish-merfish-alignment-affine-only-with-points`.
"""),
        md("## Inputs"),
        code(LOAD),
        md("""
## The fit

`fit` chooses how much freedom the affine gets. `"similarity"` allows rotation, one uniform
scale and translation -- four degrees of freedom. `"affine"` adds non-uniform scale and shear,
for six. The constrained fit is the safer default precisely because it *cannot* shear a
section that should not be sheared; use it unless the extra two degrees are earned.
"""),
        code("""
from squidpy.experimental.tl import align_landmarks

fits = {name: align_landmarks(landmarks[2], landmarks[3], fit=name)
        for name in ('similarity', 'affine')}

for name, matrix in fits.items():
    moved = landmarks[3] @ matrix[:2, :2].T + matrix[:2, 2]
    residual = np.linalg.norm(moved - landmarks[2], axis=1)
    print(f'{name:<11} residual over the 13 landmarks: '
          f'median {np.median(residual):7.1f} um, worst {residual.max():7.1f} um')
print()
print('affine:'); print(fits['affine'].round(3))
"""),
        md("""
## What it does to the section

Six degrees of freedom is the whole model here, so the two point clouds agree where the tissue
moved rigidly and disagree wherever it deformed. That residual disagreement is exactly what
`align_stalign_obs` exists to absorb -- see `merfish-merfish`.
"""),
        code("""
import matplotlib.pyplot as plt

matrix = fits['affine']
moved = query.obsm['spatial'] @ matrix[:2, :2].T + matrix[:2, 2]

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for a, (pts, title) in zip(ax, [(query.obsm['spatial'], 'before'),
                                (moved, 'after the affine')], strict=True):
    a.scatter(*ref.obsm['spatial'].T, s=0.12, alpha=0.3, label='reference (S2R2)')
    a.scatter(*pts.T, s=0.12, alpha=0.3, label='query (S2R3)')
    a.set_title(title); a.set_aspect('equal'); a.invert_yaxis()
    a.set_xticks([]); a.set_yticks([])
ax[0].scatter(*landmarks[2].T, s=40, c='k', marker='x', label='landmarks')
ax[0].legend(markerscale=40, loc='lower left', fontsize=8)
"""),
    ],
)

# ---------------------------------------------------------------- merfish-visium
write(
    "docs/notebooks/squidpy-api/merfish-visium.ipynb",
    [
        md("""
# Placing MERFISH cells on a Visium H&E image

When one side is an image there is nothing to rasterize it into, so the *other* side is
rasterized instead and `align_stalign_image` fits image to image.

The two live in different units -- microns for the MERFISH section, pixels for the H&E -- and
neither is restated anywhere: each element carries its own placement and the solver reads the
units off it. Upstream's equivalent is `merfish-visium-alignment`.
"""),
        md("## Inputs"),
        code("""
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel
from squidpy.experimental.im import rasterize_points

MERFISH = ('merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase'
           '_Slice2_Replicate3_cell_metadata_S2R3.csv.gz')
cells = pd.read_csv(MERFISH)
xy = np.c_[cells['center_x'], cells['center_y']].astype(float)

he = plt.imread('visium_data/tissue_hires_image.png')[..., :3]
visium = sd.SpatialData(images={'he': Image2DModel.parse(
    np.moveaxis(he, -1, 0).astype(float), dims=('c', 'y', 'x'))})

merfish = sd.SpatialData(points={'cells': PointsModel.parse(xy)})
rasterize_points(merfish, 'cells', dx=30.0, blur=1.0, key_added='section')
print(f'{len(xy)} cells rasterized to {tuple(np.asarray(merfish["section"]).shape)}, '
      f'H&E is {he.shape}')
"""),
        md("""
Twelve landmark pairs, matched by row order, each side in its own units -- microns for the
section, pixels for the H&E. Neither is restated anywhere: the elements carry their placement
and the solver reads the units off them.
"""),
        code("""
# Upstream's own pairs for this notebook, twelve of them, stored as `(x, y)` -- which is what
# squidpy takes, so nothing is transposed on the way in. The region-keyed `.npy` files belong to
# the point-annotator variant rather than to this one.
data = np.load('visium_data/visium2_points.npz')
paired = {'query': np.asarray(data['pointsI'], dtype=float),
          'ref': np.asarray(data['pointsJ'], dtype=float)}
print(f'{len(paired["ref"])} landmark pairs')

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].scatter(*xy.T, s=0.12, alpha=0.3); ax[0].scatter(*paired['query'].T, s=30, c='red')
ax[0].set_title('MERFISH section, in microns'); ax[0].invert_yaxis(); ax[0].set_aspect('equal')
ax[1].imshow(he); ax[1].scatter(*paired['ref'].T, s=30, c='red')
ax[1].set_title('Visium H&E, in pixels')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
        md("""
## The fit

Upstream's own solver values for this pair. `sigmaP` weights the landmark matching term, which
matters more here than in the point-cloud case: the two modalities do not share an intensity
scale, so the landmarks carry much of the correspondence.
"""),
        code("""
from squidpy.experimental.tl import align_stalign_image

fit = align_stalign_image(
    visium, merfish, image_key=('he', 'section'),
    landmarks_ref=paired['ref'], landmarks_query=paired['query'],
    niter=200, sigmaM=0.2, sigmaB=0.19, sigmaA=0.3, sigmaP=2e-1,
    epL=5e-11, epT=5e-4, epV=5e1,
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
        md("""
## Every cell, placed on the image

**The H&E covers one hemisphere; the MERFISH section is a whole coronal slice.** So roughly
half the cells have no tissue to land on and end up beside the image rather than on it. That is
the data, not a failed fit -- upstream's own notebook pairs these same two files and its
published figure shows the same overhang.

It is also why the landmark residual below is not evidence of much: all twelve landmarks sit on
the covered hemisphere, so the fit can satisfy them exactly while the uncovered half is pulled
along by the deformation alone, with nothing to match against. The fraction of cells landing
inside the image is the number that actually describes the situation.
"""),
        code("""
placed = np.asarray(fit.transform(xy))
moved_landmarks = np.asarray(fit.transform(paired['query']))
residual = np.linalg.norm(moved_landmarks - paired['ref'], axis=1)
print(f'landmark residual after fitting: median {np.median(residual):.1f} px, '
      f'worst {residual.max():.1f} px')

rows, columns = he.shape[:2]
inside = ((placed[:, 0] >= 0) & (placed[:, 0] < columns)
          & (placed[:, 1] >= 0) & (placed[:, 1] < rows))
print(f'{inside.sum()} of {len(placed)} cells ({100 * inside.mean():.0f}%) land within the '
      f'{columns} x {rows} image; the rest are the hemisphere it does not cover')

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(he); ax[0].set_title('Visium H&E')
ax[1].imshow(he)
ax[1].scatter(*placed.T, s=0.12, alpha=0.3, c='tab:blue')
ax[1].scatter(*paired['ref'].T, s=30, c='red', label='target landmarks')
ax[1].set_title('MERFISH cells placed on it'); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
"""),
    ],
)
