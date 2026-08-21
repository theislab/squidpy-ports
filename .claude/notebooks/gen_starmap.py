import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_both import GUESS_TAIL, NORM_CODE, PREP, code, md, tail, write

cells = [
    md("""
# Placing a STARmap section in the Allen CCF

A 2D section fitted into a 3D reference volume, entirely through squidpy's public API:
`rasterize_points` -> `align_stalign_volume` -> `Stalign3DResult.transform` -> `sample_volume`.
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

cells = pd.read_csv('starmap_data/well11_spatial.csv.gz')
xy = np.c_[np.array(cells['X'])[1:], np.array(cells['Y'])[1:]].astype(float)

atlas, hdr = nrrd.read('ara_nissl_50.nrrd')       # (z, y, x)
labels, _ = nrrd.read('annotation_50.nrrd')       # same frame, integer structure ids
voxel = tuple(np.diag(hdr['space directions']))

def unit_range(a):
    return (a - a.min()) / np.ptp(a)
"""
        + PREP
        + """
sdata = sd.SpatialData(
    points={'cells': PointsModel.parse(xy)},
    images={'atlas': Image3DModel.parse(
        unit_range(reference_channels(atlas.astype(float))), dims=('c', 'z', 'y', 'x'),
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

rasterize_points(sdata, 'cells', dx=50.0, blur=1.0, key_added='section')
sdata['section']
"""),
    md("""
### Putting both volumes on the scale the solver's parameters assume

`sigmaM`, `sigmaA`, `sigmaB`, `muA` and `muB` are in the **target's** intensity units, and
upstream states its values against a section that has been divided by its own mean and then
mapped onto `[0, 1]`. The reference gets the same treatment, which the solver's intensity
regression would absorb on its own -- but the second channel it is given alongside is not
something any regression can recover, so both steps are done here rather than assumed.
"""),
    code(
        NORM_CODE
        + """
mean_normalised(sdata, 'section')
element = sdata.images['section']
sdata.images['section'] = Image2DModel.parse(
    unit_range(np.asarray(element)), dims=('c', 'y', 'x'),
    transformations={'global': get_transformation(element, 'global')},
)
section = np.asarray(sdata['section']).squeeze()
print(f'section now spans {section.min():.2f} to {section.max():.2f}, mean {section.mean():.2f}')
"""
    ),
    md("""
## The fit

`initial_slice`, `initial_rotation` and `initial_scale` are an initialisation, not the answer --
the full 3D deformation is fitted, so the section need not be exactly coronal.
"""),
    code("""
from squidpy.experimental.tl import align_stalign_volume

def physical_axes(element, axes):
    matrix = get_transformation(element, 'global').to_affine_matrix(input_axes=axes, output_axes=axes)
    return [(np.asarray(element.coords[a]) - 0.5) * matrix[k, k] + matrix[k, -1]
            for k, a in enumerate(axes)]

# `initial_scale` is one uniform factor for all three axes, and this dataset is anisotropic:
# ~4x in plane, ~0.9x through the slice axis. `initial_affine` is the escape hatch.
#
# Built in the solver's (z, y, x) and reversed once at the end. Composing it directly in
# (x, y, z) means hand-transposing a rotation, two scales and three translations, and each is
# a chance to mirror an axis silently -- so the reversal gets exactly one line.
theta, scale_xy, scale_z, slice_index = np.pi / 2, 4.0, 0.9, 140
z_axis, _, _ = physical_axes(sdata['atlas'], ('z', 'y', 'x'))
y_axis, x_axis = physical_axes(sdata['section'], ('y', 'x'))

# The one landmark this analysis pins: atlas (0, 0) sits at (-3700, 0) in the section, in (-y, -x).
landmark_yx = np.array([-3700.0, 0.0])

rotation = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(theta), -np.sin(theta)],
                     [0.0, np.sin(theta), np.cos(theta)]])
affine_zyx = np.eye(4)
affine_zyx[:3, :3] = rotation @ np.diag([scale_z, scale_xy, scale_xy])
affine_zyx[:3, 3] = [-z_axis[slice_index],
                     y_axis.mean() - landmark_yx[0] * scale_xy,
                     x_axis.mean() - landmark_yx[1] * scale_xy]

reverse = np.eye(4)[[2, 1, 0, 3]]          # spatial axes only; the homogeneous row stays put
initial_affine = reverse @ affine_zyx @ reverse

# `sigmaR` is deliberately absent. Upstream's `LDDMM_3D_to_slice` declares 1e8, which weights
# the regulariser so weakly that the velocity field grows unchecked; squidpy's volume default
# is the retuned 1e6, and letting the default supply it is the point.
SOLVER = dict(a=250.0, nt=4, sigmaM=0.1, sigmaA=0.1, sigmaB=0.1, muA=[0.7], muB=[0.0])
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
    sdata, image_key=('atlas', 'section'), initial_affine=initial_affine, niter=0, **SOLVER
)
"""
        + GUESS_TAIL
        + """
initial_depth = np.asarray(guess.transform(xy))[:, 2]
print(f'initial guess places the section at z = {initial_depth.mean():.0f} um '
      f'(slice {slice_index} sits at {z_axis[slice_index]:.0f} um)')
"""
    ),
    code("""
fit = align_stalign_volume(
    sdata, image_key=('atlas', 'section'), initial_affine=initial_affine, niter=800, **SOLVER
)
print(f'{fit.n_iter} iterations, objective '
      f'{float(fit.energies[0]):.0f} -> {float(fit.energies[-1]):.0f}')
"""),
] + tail("STARmap section", "atlas at the initial guess")

write("docs/notebooks/squidpy-api/starmap-allen3Datlas.ipynb", cells)
