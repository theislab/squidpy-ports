# Notebook generators

The notebooks in `docs/notebooks/squidpy-api/` are generated, then executed on a GPU node, and
the executed copy is what gets committed. These scripts are the source.

Run them from the repository root -- the output paths are repo-relative:

    python .claude/notebooks/gen_obs.py       # the eight point-cloud notebooks
    python .claude/notebooks/gen_planar.py    # merfish-merfish, its landmark-only variant, merfish-visium
    python .claude/notebooks/gen_merfish.py   # merfish-allen3Datlas
    python .claude/notebooks/gen_starmap.py   # starmap-allen3Datlas

`gen_both.py` is imported by the others and writes nothing on its own.

**Running a generator destroys the executed outputs** -- it writes the notebook from scratch. The
order is regenerate, execute, then commit; committing in between leaves 10 kB stubs in git that
look like progress. Execute with:

    ports-clone/.claude/on_held_node.sh          # inside a held allocation, seconds
    sbatch --export=NIL .claude/run_public_api_notebooks.sbatch   # otherwise, queued

One notebook per upstream notebook, named after it, so ours can be read beside
`docs/notebooks/stalign-upstream/`. Solver values are squidpy's defaults except where the data
forces otherwise (`dx`/`blur`, `diffeo_start` for the affine-only pair, `a` for Visium's array
units) -- upstream's own overrides are not automatically right, and `heart-alignment`'s
`niter=1000` was actively wrong.
