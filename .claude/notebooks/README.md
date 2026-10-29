# Notebooks

The notebooks in `docs/notebooks/squidpy-api/` are edited directly, then executed on a GPU node,
and the executed copy is what gets committed. Execute with:

    ports-clone/.claude/on_held_node.sh          # inside a held allocation, seconds
    sbatch --export=NIL .claude/run_public_api_notebooks.sbatch   # otherwise, queued

One notebook per upstream notebook, named after it, so each can be read beside the upstream one
it links to. Solver values are squidpy's defaults except where the data forces otherwise
(`dx`/`blur`, `diffeo_start` for the affine-only pair, `a` for Visium's array units).
