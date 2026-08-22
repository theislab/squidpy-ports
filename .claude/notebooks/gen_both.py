import json
import os
import re

#: STalign's pinned revision, and which of its notebooks each of ours re-expresses. Injected into
#: the first markdown cell by `write`, so every page opens with a link to the original it mirrors.
STALIGN_REV = "b2068edc98974efa54537eca194736e177bbe11d"
UPSTREAM = {
    "starmap-allen3Datlas": "starmap-allen3Datlas-alignment",
    "merfish-allen3Datlas": "merfish-allen3Datlas-alignment",
    "merfish-merfish": "merfish-merfish-alignment",
    "merfish-merfish-initial-affine": "merfish-merfish-alignment-using-L-T",
    "merfish-merfish-affine-only": "merfish-merfish-alignment-affine-only",
    "merfish-merfish-affine-only-with-points": "merfish-merfish-alignment-affine-only-with-points",
    "merfish-xenium": "merfish-xenium-alignment",
    "xenium-xenium": "xenium-xenium-alignment",
    "xenium-starmap": "xenium-starmap-alignment",
    "xenium-heimage": "xenium-heimage-alignment",
    "visium-visium-affine-only": "visium-visium-alignment-affine-only",
    "heart-alignment": "heart-alignment",
    "heart-alignment-varying-thickness": "heart-alignment-varying-thickness",
    "merfish-visium": "merfish-visium-alignment",
    "merfish-visium-with-point-annotator": "merfish-visium-alignment-with-point-annotator",
    "merfish-visium-with-curve-annotator": "merfish-visium-alignment-with-curve-annotator",
}

#: The squidpy revision these notebooks are executed against. The functions they call live on a
#: fork branch and are absent from squidpy's published API docs, so an intersphinx role would be
#: an unresolved reference and `nitpicky = True` plus `-W` would fail the build. Linking to the
#: source at a pinned revision is clickable, exact, and cannot rot into pointing at something else.
SQUIDPY_REV = "e9a94c4d125fc3ac7b791a8ce6c6ff58e1e885e4"
_SRC = f"https://github.com/selmanozleyen/squidpy/blob/{SQUIDPY_REV}/src/squidpy"
API = {
    "rasterize_points": "experimental/im/_rasterize_points.py",
    "sample_volume": "experimental/im/_rasterize_points.py",
    "align_stalign_obs": "experimental/tl/_align/_api.py",
    "align_stalign_image": "experimental/tl/_align/_api.py",
    "align_stalign_volume": "experimental/tl/_align/_api.py",
    "align_landmarks": "experimental/tl/_align/_api.py",
    "stalign_apply_transform": "experimental/tl/_align/_api.py",
    "stalign_apply_warp": "experimental/tl/_align/_api.py",
    "stalign_transform_points": "experimental/tl/_align/_stalign.py",
    "stalign_warp_image": "experimental/tl/_align/_stalign.py",
    "stalign_deformation_grid": "experimental/tl/_align/_stalign.py",
}
#: Classes that *are* published, so these resolve through intersphinx instead.
ROLES = {
    "Image2DModel": ":class:`~spatialdata.models.Image2DModel`",
    "PointsModel": ":class:`~spatialdata.models.PointsModel`",
}


def linkify(src):
    """Make every API mention in prose clickable, once, leaving code cells alone."""
    for name, module in API.items():
        src = re.sub(rf"(?<![\[`\w])`{name}`(?!\])", f"[`{name}`]({_SRC}/{module})", src)
    for name, role in ROLES.items():
        src = re.sub(rf"(?<![\[`\w])`{name}`(?!\])", role, src)
    return src


def lines(src):
    """Split a source block into nbformat's list-of-lines, terminators included."""
    ls = src.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]


def md(s):
    """A markdown cell, with API mentions linked."""
    s = linkify(s)
    return {"cell_type": "markdown", "metadata": {}, "source": lines(s)}


def code(s):
    """A code cell, unexecuted."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(s)}


PREP = """
# The second channel is not decoration: the solver regresses the deformed reference onto the
# section before measuring the match, so one channel fits `a + b*I` and two fit
# `a + b*I + c*(I - mean I)**2`. A regression absorbs a rescaling of its input; it cannot
# invent a channel it was never handed.
def reference_channels(volume):
    v = volume[None] / np.mean(np.abs(volume))
    return np.concatenate([v, (v - v.mean()) ** 2])
"""

NORM_MD = """
### Putting the section on the scale the solver's parameters assume

`sigmaM`, `sigmaA`, `sigmaB`, `muA` and `muB` are all in the **target's** intensity units, and
upstream states its values against a mean-normalised section. Handing over the raw raster
instead leaves every one of those widths wrong by the raster's own mean -- which for a `dx=10`
raster is a factor of nine, enough to flatten the artifact/background/matching split into
near-constants and leave the fit with almost no gradient to descend.
"""

NORM_CODE = """
from spatialdata.models import Image2DModel
from spatialdata.transformations import get_transformation

def mean_normalised(sdata, key):
    element = sdata.images[key]
    values = np.asarray(element)
    sdata.images[key] = Image2DModel.parse(
        values / np.abs(values).mean(), dims=('c', 'y', 'x'),
        transformations={'global': get_transformation(element, 'global')},
    )
"""


def tail(section_title):
    """The cells every volume notebook ends with: placement, ontology, panels."""
    return [
        md("""
## Where each cell lands

`stalign_transform_points` maps `(x, y)` section coordinates to `(x, y, z)` reference coordinates, evaluated at
each point rather than at the nearest raster cell. `sample_volume` then reads the annotation
volume there -- `order=0` because structure ids must not be interpolated.
"""),
        code("""
coords = np.asarray(stalign_transform_points(fit, xy))  # (N, 3), (x, y, z) in microns
structure_id = sample_volume(labels, fit['ref_axes'], coords, order=0).astype(int)

print(f'{len(coords)} cells placed, {np.unique(structure_id).size} distinct structures, '
      f'{100 * (structure_id == 0).mean():.1f}% outside any annotated structure')
print(f'depth (z) spans {coords[:, 2].min():.0f} to {coords[:, 2].max():.0f} um')
"""),
        md("Structure ids become acronyms through the Allen ontology."),
        code("""
ontology = pd.read_csv('allen_ontology.csv').set_index('id')['acronym']
acronym = pd.Series(structure_id).map(ontology).fillna('unassigned')
acronym.value_counts().head(12)
"""),
        md("## The aligned atlas over the section"),
        code(f"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def atlas_at(result):
    plane = np.moveaxis(np.asarray(stalign_deformation_grid(result, direction='backward')), 0, -1)[0]
    sampled = sample_volume(atlas, result['ref_axes'], plane.reshape(-1, 3)[:, ::-1])
    return sampled.reshape(plane.shape[:2])

section = np.asarray(sdata['section']).squeeze()
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(section, cmap=mpl.cm.Blues)
ax[1].imshow(atlas_at(fit), cmap=mpl.cm.Reds)
ax[2].imshow(section, cmap=mpl.cm.Blues, alpha=0.9)
ax[2].imshow(atlas_at(fit), cmap=mpl.cm.Reds, alpha=0.3)
for a, t in zip(ax, ('{section_title}', 'atlas after fitting', 'overlaid'), strict=True):
    a.set_title(t); a.set_xticks([]); a.set_yticks([])
"""),
        md("## Cells coloured by structure"),
        code("""
keep = acronym.value_counts()
keep = keep[keep >= 50].index                       # a legend of singletons reads as noise
fig, ax = plt.subplots(figsize=(7, 6))
for region in keep:
    m = (acronym == region).to_numpy()
    ax.scatter(xy[m, 0], xy[m, 1], s=0.08, label=region)
ax.invert_yaxis(); ax.set_aspect('equal')
ax.legend(handles=[Line2D([], [], marker='o', ls='', ms=4, color=h.get_facecolor()[0], label=r)
                   for r, h in zip(keep, ax.collections, strict=True)],
          fontsize=6, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
"""),
    ]


nb_meta = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def write(path, cells):
    """Write a notebook, refusing one whose code does not parse.

    A notebook whose sources are already what we would emit is left untouched, outputs and
    all. Rewriting it would blank the figures of something that did not change -- which is how
    a one-line edit to one notebook has repeatedly cost the executed outputs of every other
    notebook the same generator emits.
    """
    for c in cells:
        if c["cell_type"] == "code":
            import ast

            ast.parse("".join(c["source"]))
    # nbformat requires a cell id and warns loudly without one -- "will become a hard error in
    # future nbformat versions". Deterministic, so regenerating does not churn the diff.
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")

    origin = UPSTREAM.get(os.path.basename(path).removesuffix(".ipynb"))
    if origin is not None:
        url = f"https://github.com/JEFworks-Lab/STalign/blob/{STALIGN_REV}/docs/notebooks/{origin}.ipynb"
        note = f"\n\nSTalign's own version of this analysis: [`{origin}`]({url}).\n"
        first = cells[0]["source"]
        if "STalign's own version" not in "".join(first):
            first[-1] = first[-1].rstrip("\n") + note

    if os.path.exists(path):
        current = json.load(open(path))
        same = [c["source"] for c in current["cells"]] == [c["source"] for c in cells]
        if same and all("id" in c for c in current["cells"]):
            print(f"unchanged {path}")
            return

    with open(path, "w") as f:
        json.dump({"cells": cells, "metadata": nb_meta, "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
        f.write("\n")
    print("wrote", path, len(cells), "cells")
