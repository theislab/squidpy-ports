"""Generate the nine point-cloud notebooks. One route, nine datasets."""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_both import code, md, write

MERFISH = "merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate{r}_cell_metadata_S2R{r}.csv.gz"

LOADERS = {
    "merfish2": f"read_xy('{MERFISH.format(r=2)}', 'center_x', 'center_y')",
    "merfish3": f"read_xy('{MERFISH.format(r=3)}', 'center_x', 'center_y')",
    "xenium_mouse": "read_xy('xenium_data/Xenium_V1_FF_Mouse_Brain_MultiSection_1_cells.csv.gz',"
    " 'x_centroid', 'y_centroid')",
    "xenium_bc1": "read_xy('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz',"
    " 'x_centroid', 'y_centroid')",
    "xenium_bc2": "read_xy('xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep2_cells.csv.gz',"
    " 'x_centroid', 'y_centroid')",
    "heart_e1": "read_xy('heart_data/CN73_E1.csv.gz', 'x', 'y')",
    "heart_e2": "read_xy('heart_data/CN73_E2.csv.gz', 'x', 'y')",
    "heart_d2": "read_xy('heart_data/3_CN73_D2.csv.gz', 'x', 'y')",
    "heart_c2": "read_xy('heart_data/4_CN73_C2.csv.gz', 'x', 'y')",
    "visium1": "read_headerless('visium_data/slice1_coor.csv')",
    "visium2": "read_headerless('visium_data/slice2_coor.csv')",
    "starmap": "read_xy('starmap_data/well11_spatial.csv.gz', 'X', 'Y', skip_first=True)",
    "starmap_xenium_frame": "read_starmap_in_xenium_frame()",
}

IMPORTS = "import anndata as ad, numpy as np, pandas as pd\n"

READ_XY = """
def read_xy(path, x, y, *, skip_first=False):
    df = pd.read_csv(path)
    xy = np.c_[np.asarray(df[x])[1:] if skip_first else df[x],
               np.asarray(df[y])[1:] if skip_first else df[y]].astype(float)
    return ad.AnnData(X=np.zeros((len(xy), 1)), obsm={'spatial': xy})
"""

READ_HEADERLESS = """
# No header in these files, so a plain `read_csv` would promote the first spot to column
# names and silently drop it.
def read_headerless(path):
    xy = pd.read_csv(path, header=None).to_numpy(dtype=float)
    return ad.AnnData(X=np.zeros((len(xy), 1)), obsm={'spatial': xy})
"""

READ_STARMAP = """
# Axes swapped, divided by 5, new y flipped: upstream's own framing, and not cosmetic. Read
# as-is the two clouds sit ~16 mm apart, and the affine's translation step is 0.2 units per
# iteration, so no iteration budget closes it -- while the objective still falls ~18%.
def read_starmap_in_xenium_frame():
    df = pd.read_csv('starmap_data/well11_spatial.csv.gz')
    x = np.asarray(df['Y'])[1:].astype(float) / 5.0
    y = np.asarray(df['X'])[1:].astype(float) / 5.0
    xy = np.c_[x, y.max() - y]
    return ad.AnnData(X=np.zeros((len(xy), 1)), obsm={'spatial': xy})
"""

ROTATED = """
# Upstream's starting guess, applied to the coordinates rather than passed as `initial_affine`:
# the fit then starts from the identity, and the initialisation stays visible here instead of
# folded into a matrix convention.
def rotated_onto(query, ref, degrees):
    theta = np.deg2rad(-degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centred = query.obsm['spatial'] - query.obsm['spatial'].mean(0)
    out = query.copy()
    out.obsm['spatial'] = centred @ rotation.T + ref.obsm['spatial'].mean(0)
    return out
"""


# name -> (upstream, ref, query, rotate_degrees|None, solver, blurb)
SPECS = {
    "heart-alignment": (
        "heart-alignment",
        "heart_e2",
        "heart_e1",
        None,
        "dx=100.0, blur=1.0",
        "Two serial sections of the same heart. The tissue is far less regular than brain, so "
        "the matching term does most of the work and `sigmaB` is dropped to 0.1 to keep "
        "background from being read as tissue. Upstream caps this one at `niter=1000`, which is not enough: the two sections start 1742 um apart and at the default translation step that budget only closes it to 1587. Leaving `niter` alone -- squidpy's default is 5000, and so is upstream's `LDDMM`'s -- gets it to 238 um. Nothing else needed tuning; it was an iteration budget, not a parameter.",
    ),
    "heart-alignment-varying-thickness": (
        "heart-alignment-varying-thickness",
        "heart_c2",
        "heart_d2",
        None,
        "dx=100.0, blur=1.0",
        "The same heart pairing, on sections cut at different thicknesses -- so the two differ "
        "in density as well as in shape, and the intensity regression inside the solver has to "
        "absorb the difference before the deformation can be read. Upstream caps this one at `niter=1000`, which is not enough: the two sections start 1742 um apart and at the default translation step that budget only closes it to 1587. Leaving `niter` alone -- squidpy's default is 5000, and so is upstream's `LDDMM`'s -- gets it to 238 um. Nothing else needed tuning; it was an iteration budget, not a parameter.",
    ),
    "merfish-merfish-affine-only": (
        "merfish-merfish-alignment-affine-only",
        "merfish2",
        "merfish3",
        45,
        "dx=15.0, blur=1.5, diffeo_start=5001",
        "The same pair as `merfish-merfish`, held to an affine. `diffeo_start` is the iteration "
        "at which the velocity field is allowed to start moving; setting it past `niter` means "
        "it never does, so only the affine part is ever fitted.",
    ),
    "merfish-merfish-using-L-T": (
        "merfish-merfish-alignment-using-L-T",
        "merfish2",
        "merfish3",
        45,
        "dx=15.0, blur=1.5",
        "The same pair again, initialised by an explicit rotation and translation rather than by "
        "landmarks. Upstream's variant exists to show the `L`/`T` entry point; here the rotation "
        "is applied to the coordinates and the fit starts from the identity.",
    ),
    "merfish-xenium": (
        "merfish-xenium-alignment",
        "xenium_mouse",
        "merfish3",
        None,
        "dx=15.0, blur=1.5",
        "Across technologies: a MERFISH section onto a Xenium one. Nothing in the call changes "
        "for the modality -- both sides are just point clouds in microns.",
    ),
    "visium-visium-affine-only": (
        "visium-visium-alignment-affine-only",
        "visium2",
        "visium1",
        None,
        "dx=1.0, blur=1.0, a=5.0, diffeo_start=5001",
        "Two Visium sections, spots rather than cells: about 250 points each, and their "
        "coordinates are array indices spanning roughly 6 to 29 rather than microns. Every "
        "length in the solver is in those units, which is why `dx` is 1 and not 30, and why `a` "
        "is 5 and not 500 -- the defaults are tuned for micron-scale data and would put the "
        "whole section inside a single raster cell.",
    ),
    "xenium-starmap": (
        "xenium-starmap-alignment",
        "starmap_xenium_frame",
        "xenium_mouse",
        None,
        "dx=30.0, blur=1.0",
        "A STARmap section onto Xenium. The sigmas are upstream's and are wider than the "
        "brain-to-brain cases, because the two do not share a cell-density scale.",
    ),
    "xenium-xenium": (
        "xenium-xenium-alignment",
        "xenium_bc1",
        "xenium_bc2",
        None,
        "dx=30.0, blur=1.0",
        "Two Xenium replicates of the same breast-cancer block. They overlap only partially, so "
        "some cells have no counterpart at all -- the matching weights are what identify the "
        "supported overlap, and unmatched cells are expected rather than a failure.",
    ),
}

LANDMARKS = {
    "xenium-xenium": (
        "xenium_data/Xenium_Breast_Cancer_Rep1_points.npy",
        "xenium_data/Xenium_Breast_Cancer_Rep2_points.npy",
    ),
}

for name, (upstream, ref_key, query_key, degrees, solver, blurb) in SPECS.items():
    lm = LANDMARKS.get(name)
    calls = LOADERS[ref_key] + LOADERS[query_key]
    load = IMPORTS
    for helper, marker in (
        (READ_XY, "read_xy("),
        (READ_HEADERLESS, "read_headerless("),
        (READ_STARMAP, "read_starmap_in_xenium_frame("),
    ):
        if marker in calls:
            load += helper
    if degrees:
        load += ROTATED
    load += f"\nref = {LOADERS[ref_key]}\nquery = {LOADERS[query_key]}\n"
    if lm:
        load += f"""
# Landmark pairs picked by hand, one array per side, matched by row order.
def read_landmarks(path):
    picked = np.load(path, allow_pickle=True).item()
    return np.array([p for key in sorted(picked) for p in picked[key]], dtype=float)

landmarks_ref = read_landmarks('{lm[0]}')
landmarks_query = read_landmarks('{lm[1]}')
print(f'{{ref.n_obs}} reference cells, {{query.n_obs}} query cells, '
      f'{{len(landmarks_ref)}} landmark pairs')
"""
    else:
        load += "print(f'{ref.n_obs} reference cells, {query.n_obs} query cells')\n"
    if degrees:
        load += f"\n# Upstream starts this pair {degrees} degrees apart.\nquery = rotated_onto(query, ref, {degrees})\n"

    fit_args = "ref, query, spatial_key='spatial',\n    "
    if lm:
        fit_args += "landmarks_ref=landmarks_ref, landmarks_query=landmarks_query,\n    "
    fit_args += solver + ","

    write(
        f"docs/notebooks/squidpy-api/{name}.ipynb",
        [
            md(f"""
# {name.replace("-", " ").replace("_", " ").capitalize()}

{blurb}

Upstream's equivalent is `{upstream}`. One call does the alignment: `align_stalign_obs` fits a
diffeomorphism straight between two point clouds, rasterizing both sides itself.
"""),
            md("## Inputs"),
            code(load),
            md("## The fit\n\nUpstream's own solver values, and squidpy's defaults for everything else."),
            code(f"""
from squidpy.experimental.tl import align_stalign_obs

fit = align_stalign_obs(
    {fit_args}
)
print(f'{{fit.n_iter}} iterations, objective '
      f'{{float(fit.energies[0]):.0f}} -> {{float(fit.energies[-1]):.0f}}')
"""),
            md("""
## Where the cells land

`transform` evaluates the fitted map at each point, so a cell lands where it lands rather than
at the nearest raster cell.
"""),
            code("""
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

moved = np.asarray(fit.transform(query.obsm['spatial']))

# Whether the fit worked, in the units of the data rather than of the objective. The energy
# can halve while the sections stay as far apart as they started -- that is what an iteration
# budget that ran out looks like, and it is invisible in the energy alone.
tree = cKDTree(ref.obsm['spatial'])
before = tree.query(query.obsm['spatial'])[0]
after = tree.query(moved)[0]
print(f"centroid offset {np.linalg.norm(query.obsm['spatial'].mean(0) - ref.obsm['spatial'].mean(0)):.0f}"
      f" -> {np.linalg.norm(moved.mean(0) - ref.obsm['spatial'].mean(0)):.0f}")
print(f'distance to the nearest reference cell: median {np.median(before):.0f} -> '
      f'{np.median(after):.0f}, 90th percentile {np.percentile(before, 90):.0f} -> '
      f'{np.percentile(after, 90):.0f}')

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for a, (pts, title) in zip(ax, [(query.obsm['spatial'], 'before'),
                                (moved, 'after the fit')], strict=True):
    a.scatter(*ref.obsm['spatial'].T, s=0.12, alpha=0.3, label='reference')
    a.scatter(*pts.T, s=0.12, alpha=0.3, label='query')
    a.set_title(title); a.set_aspect('equal'); a.invert_yaxis()
    a.set_xticks([]); a.set_yticks([])
ax[0].legend(markerscale=90, loc='lower left', fontsize=8)
"""),
            md("""
## Which cells the fit could actually use

Two sections rarely cover the same tissue. The solver splits the target into three classes as
it goes -- matching, artifact and background -- and reweights them every iteration, so regions
with no counterpart stop pulling on the deformation. Reading those weights back is what turns
a partial overlap into something visible, rather than a fit that merely looks bad: the
unmatched parts are supposed to be unmatched.
"""),
            code("""
from scipy.ndimage import map_coordinates

# Which grid the weight raster lives on decides which points can index it, so it is chosen by
# matching shapes: the mixture is estimated over the target, yet upstream interpolates the same
# array on the source axes, and a wrong guess silently samples the wrong place.
def per_cell(weight_field, points):
    field = np.asarray(weight_field).squeeze()
    for axes in (fit.ref_axes, fit.query_axes):
        if axes is not None and field.shape == tuple(len(np.asarray(a)) for a in axes):
            y, x = (np.asarray(a) for a in axes)
            rows = (points[:, 1] - y[0]) / (y[1] - y[0])
            cols = (points[:, 0] - x[0]) / (x[1] - x[0])
            return map_coordinates(field, np.vstack([rows, cols]), order=1, mode='nearest')
    raise ValueError(f'no axes match the weight field shape {field.shape}')

for name in ('match_weights', 'artifact_weights', 'background_weights'):
    w = getattr(fit, name)
    print(f'{name}: {None if w is None else np.asarray(w).shape}')

matching = per_cell(fit.match_weights, moved)
print(f'{100 * (matching > 0.5).mean():.0f}% of the query cells sit where the fit had real '
      f'support (matching weight > 0.5)')

fig, ax = plt.subplots(figsize=(7, 6))
dots = ax.scatter(*moved.T, c=matching, s=0.12, vmin=0, vmax=1, cmap='viridis')
ax.set_title('query cells, coloured by matching weight')
ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(dots, ax=ax, fraction=0.046)
"""),
            md("""
The objective's trace. The mixture E step switches on at iteration 50 and the energy changes
definition there, so only the part after the dashed line is one function.
"""),
            code("""
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
"""),
        ],
    )
