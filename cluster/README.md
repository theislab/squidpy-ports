# Cluster

`run_container.sbatch` is a thin Slurm launcher for the reproduction container — it only schedules
`apptainer run` on a GPU node:

```bash
sbatch cluster/run_container.sbatch <stalign.sif> <notebook.ipynb> <out-dir>
```

- **Reproduction** — portable, any GPU box, no cluster needed: [`../container/README.md`](../container/README.md).
- **Theislab-specific ops** — how to build the image here, the localscratch / fakeroot /
  `--export=NIL` learnings, and the full 17-notebook parity launcher: `.claude/hpc-stalign.md`
  (internal notes, not part of the reproduction contract).
