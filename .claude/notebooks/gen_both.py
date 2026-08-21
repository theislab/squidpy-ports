import json
import os


def lines(src):
    """Split a source block into nbformat's list-of-lines, terminators included."""
    ls = src.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + [ls[-1]]


def md(s):
    """A markdown cell."""
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

TAIL_MD_FIT = """
The objective is not one function across this whole trace. The mixture E step switches on at
iteration 50 (`STalign.py:1233`): before it, the artifact / background / matching weights are
frozen at their initial values, so the energy either side of the dashed line is computed
against different weights and steps discontinuously there. Squidpy's own solver says so --
"the objective changes definition here and its value jumps discontinuously".

So the descent worth reading is the part after the gate. A minimum reported "at iteration 49"
belongs to the pre-gate objective, not to a better alignment, and stopping there would stop
before the mixture model has done anything at all. `tol` already knows this: its warm-up runs
to `50 + 2 + patience` before the improvement test is allowed to fire.

What a flat post-gate tail means here is that the affine and the velocity field have settled;
the panels below are the check that matters, and they are the reason this is read as converged
rather than as a fit that ran away.
"""


def tail(section_title, initial_title):
    """The cells every volume notebook ends with: trace, placement, ontology, panels."""
    return [
        md(TAIL_MD_FIT),
        code("""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Upstream's gate, not a tuning knob: `STalign.py:1233` starts the mixture E step at 50.
MIXTURE_GATE = 50

energies = np.asarray(fit.energies)[: fit.n_iter]
descent = energies[MIXTURE_GATE + 1 :]        # the half of the trace that shares a definition
tail = descent[-len(descent) // 10 :]
print(f'after the gate: {descent[0]:.0f} -> {descent[-1]:.0f}, minimum {descent.min():.0f} '
      f'at iteration {MIXTURE_GATE + 1 + int(descent.argmin())}')
print(f'last tenth: mean {tail.mean():.0f}, spread {np.ptp(tail):.0f} '
      f'({100 * np.ptp(tail) / tail.mean():.1f}% of its mean)')

plt.plot(energies, lw=0.8)
plt.axvline(MIXTURE_GATE, color='0.6', lw=0.8, ls='--')
plt.annotate('mixture E step on', (MIXTURE_GATE, plt.ylim()[1]), fontsize=8,
             xytext=(6, -12), textcoords='offset points', color='0.4')
plt.xlabel('iteration'); plt.ylabel('objective'); plt.grid(alpha=0.3)
"""),
        md("""
## Where each cell lands

`transform` maps `(x, y)` section coordinates to `(x, y, z)` reference coordinates, evaluated at
each point rather than at the nearest raster cell. `sample_volume` then reads the annotation
volume there -- `order=0` because structure ids must not be interpolated.
"""),
        code("""
coords = np.asarray(fit.transform(xy))                 # (N, 3), (x, y, z) in microns
structure_id = sample_volume(labels, fit.ref_axes, coords, order=0).astype(int)

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

section = np.asarray(sdata['section']).squeeze()
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(section, cmap=mpl.cm.Blues)
ax[1].imshow(atlas_at(guess), cmap=mpl.cm.Reds)
ax[2].imshow(atlas_at(fit), cmap=mpl.cm.Reds)
ax[3].imshow(section, cmap=mpl.cm.Blues, alpha=0.9)
ax[3].imshow(atlas_at(fit), cmap=mpl.cm.Reds, alpha=0.3)
for a, t in zip(ax, ('{section_title}', '{initial_title}',
                     'atlas after fitting', 'overlaid'), strict=True):
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


GUESS_TAIL = """
def atlas_at(result):
    plane = np.moveaxis(np.asarray(result.deformation_grid(direction='backward')), 0, -1)[0]
    sampled = sample_volume(atlas, result.ref_axes, plane.reshape(-1, 3)[:, ::-1])
    return sampled.reshape(plane.shape[:2])
"""

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
    if os.path.exists(path):
        current = json.load(open(path))
        if [c["source"] for c in current["cells"]] == [c["source"] for c in cells]:
            print(f"unchanged {path}")
            return

    with open(path, "w") as f:
        json.dump({"cells": cells, "metadata": nb_meta, "nbformat": 4, "nbformat_minor": 5}, f, indent=1)
        f.write("\n")
    print("wrote", path, len(cells), "cells")
