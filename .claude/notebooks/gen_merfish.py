import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_both import GUESS_TAIL, NORM_CODE, NORM_MD, PREP, code, md, tail, write

cells = [
    md("""
# Placing a MERFISH section in the Allen CCF

The same volume-to-section fit as [the STARmap notebook](starmap-allen3Datlas.ipynb), on a
denser section and a finer raster. It is the simpler of the two to write: this section needs
no anisotropic initialisation, so `initial_slice`, `initial_rotation` and `initial_scale` say
the whole starting guess and `initial_affine` never appears.
"""),
    md("""
## Inputs

Cells as a points element, the atlas and its annotation volume as 3D images. The physical
placement lives on the elements, so nothing downstream builds coordinate axes by hand.
"""),
    code(
        """
import nrrd, numpy as np, pandas as pd, spatialdata as sd
from spatialdata.models import Image3DModel, PointsModel
from spatialdata.transformations import Scale, Sequence, Translation

cells = pd.read_csv('merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase'
                    '_Slice1_Replicate1_cell_metadata_S1R1.csv.gz')
xy = np.c_[cells['center_x'], cells['center_y']].astype(float)

atlas, hdr = nrrd.read('ara_nissl_50.nrrd')       # (z, y, x)
labels, _ = nrrd.read('annotation_50.nrrd')       # same frame, integer structure ids
voxel = tuple(np.diag(hdr['space directions']))
"""
        + PREP
        + """
sdata = sd.SpatialData(
    points={'cells': PointsModel.parse(xy)},
    images={'atlas': Image3DModel.parse(
        reference_channels(atlas.astype(float)), dims=('c', 'z', 'y', 'x'),
        transformations={'global': Sequence([
            Scale(list(voxel), axes=('z', 'y', 'x')),
            Translation(-(np.asarray(atlas.shape) - 1) * np.asarray(voxel) / 2,
                        axes=('z', 'y', 'x')),
        ])},
    )},
)
sdata
"""
    ),
    md("""
## The section

`rasterize_points` turns the centroids into a density image: each cell deposits unit mass
bilinearly, blurred once per scale, so total intensity is exactly the cell count.
"""),
    code("""
from squidpy.experimental.im import rasterize_points, sample_volume

rasterize_points(sdata, 'cells', dx=10.0, blur=1.0, key_added='section')
sdata['section']
"""),
    md(NORM_MD),
    code(
        NORM_CODE
        + """
mean_normalised(sdata, 'section')
section = np.asarray(sdata['section']).squeeze()
print(f'section now spans {section.min():.2f} to {section.max():.2f}, mean {section.mean():.2f}')
"""
    ),
    md("""
## The fit

The section is close to coronal and close to atlas scale, so the initialisation is three
numbers: which slice to centre on, how far to rotate in plane, and one uniform scale.

Upstream also nudges the starting translation by a single landmark pair, worth about 42 um in
`x` and 7 um in `y` -- four pixels and under one. That is an initialisation, not an answer, and
the next cell checks the initialisation directly, so it is left out here rather than
reintroducing a hand-built affine to carry it.
"""),
    code("""
from squidpy.experimental.tl import align_stalign_volume, stalign_deformation_grid, stalign_transform_points

slice_index, rotation, scale = 177, 0.0, 0.9

# Upstream's own values, now that the section is on the scale they were written for. One
# channel, not three: upstream passes `muA=[3, 3, 3]` against a single-channel target and sums
# over the broadcast axis, making its effective widths sigma/sqrt(3) -- divergence D13.
#
# `sigmaR` is deliberately absent. Upstream's `LDDMM_3D_to_slice` declares 1e8, which weights
# the regulariser so weakly that the velocity field grows unchecked; squidpy's volume default
# is the retuned 1e6, and letting the default supply it is the point.
SOLVER = dict(a=500.0, nt=4, sigmaM=2.0, sigmaA=2.0, sigmaB=2.0, muA=[3.0], muB=[0.0])
"""),
    md("""
### Is the initialisation in the right place?

`niter=0` returns the starting affine without fitting, so the initial guess can be looked at
through the same public route as the result. Worth doing: an initialisation that starts in the
wrong place produces a fit that never recovers, and the objective alone does not say so.
"""),
    code(
        """
guess = align_stalign_volume(
    sdata, image_key=('atlas', 'section'), niter=0,
    initial_slice=slice_index, initial_rotation=rotation, initial_scale=scale, **SOLVER,
)
"""
        + GUESS_TAIL
        + """
# A fit reports the reference axes it read off the element, so the depth of the chosen slice
# needs no second derivation from the NRRD header -- and cannot disagree with the one used.
z_axis = np.asarray(guess['ref_axes'][0])
initial_depth = np.asarray(stalign_transform_points(guess, xy))[:, 2]
print(f'initial guess places the section at z = {initial_depth.mean():.0f} um '
      f'(slice {slice_index} sits at {z_axis[slice_index]:.0f} um)')
"""
    ),
    code("""
fit = align_stalign_volume(
    sdata, image_key=('atlas', 'section'), niter=2000,
    initial_slice=slice_index, initial_rotation=rotation, initial_scale=scale, **SOLVER,
)
print(f'{fit["n_iter"]} iterations, objective '
      f'{float(fit["energies"][0]):.0f} -> {float(fit["energies"][-1]):.0f}')
"""),
] + tail("MERFISH section", "atlas at the initial guess")

write("docs/notebooks/squidpy-api/merfish-allen3Datlas.ipynb", cells)
