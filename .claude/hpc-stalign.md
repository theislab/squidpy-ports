# Theislab HPC — building & running the STalign comparison container

Internal ops notes for the Theislab cluster. **Not** the public reproduction path — that is the
portable container in [`../container/`](../container/README.md), runnable on any GPU box with
Apptainer. These are the cluster-specific bits and the learnings from getting it to build and run
here, kept out of the reproduction contract on purpose.

## The two commands

```bash
# Build the image (CPU node; everything churny on /localscratch, final .sif copied to Lustre)
sbatch --export=NIL .claude/build_container.sbatch

# Run one comparison notebook (GPU); thin launcher, just `apptainer run`
sbatch --export=NIL cluster/run_container.sbatch <stalign.sif> <notebook.ipynb> <out-dir>
```

Notebooks: `stalign-xenium-comparison.ipynb`, `stalign-merfish-comparison.ipynb`,
`stalign-visium-affine-comparison.ipynb`.

## Learnings (each cost real time)

- **Submit with `--export=NIL`.** `--export=ALL` triggers Slurm's user-environment retrieval, which
  fails here → `user_env_retrieval_failed_requeued_held` (job stuck, never runs). The scripts
  reconstruct `HOME`/`USER`/`PATH` themselves.
- **Build on `/localscratch`, never Lustre.** Point `APPTAINER_TMPDIR`, `APPTAINER_CACHEDIR`, and the
  `.sif` output at node-local scratch; copy only the finished single `.sif` (~9.8 GB, one big file)
  to Lustre. Lustre hates the many small files a build churns.
- **`apptainer --fakeroot` works without an `/etc/subuid` entry** (Apptainer 1.4.1, `allow setuid`).
  The repeated `libfakeroot ... payload not recognized!` lines are harmless noise, not a failure.
- **Build needs `build-essential`** — `mahotas` (pulled via `squidpy → cp-measure`) has no wheel for
  this Python and compiles from source — **and `LICENSE`** in the def's `%files` (hatchling validates
  `license = { file = "LICENSE" }`).
- **Run with `--nv --writable-tmpfs`**, and set `MPLCONFIGDIR=/tmp/mpl` so matplotlib's cache lands on
  the overlay, not a bound `$HOME` on Lustre.
- **Input data:** the container symlinks the vendored STalign datasets to each notebook's expected
  dir. The names are inconsistent across notebooks — xenium expects `.../stalign-data/xenium`, the
  others `.../stalign-data/<modality>_data`. Wrong name → the notebook tries to download → tmpfs runs
  out of space.
- **Warm uv cache:** a shared per-node cache (`/localscratch/$USER/rapids-singlecell-equivalence/cache/uv`)
  makes re-installs fast; it is tens of thousands of small files, so it must stay off Lustre.

## Resources

- **Build:** `cpu_p` / `cpu_normal`, ~8 CPU / 32 G, ~15–25 min (compiles `mahotas`, pulls the CUDA
  wheels). No GPU needed to build.
- **Run:** `gpu_p`, `--constraint=h100_80gb`, 6 CPU / 90 G. QoS `gpu_priority` (12 h, ≤2 GPU) for
  smoke tests; `gpu_normal` for a full sweep.
- Never compute on the login/submit nodes — `sbatch` onto a compute node.

## Full 17-notebook parity track (separate)

`.claude/run_stalign_comparisons.sbatch` drives `squidpy_ports.stalign.notebook_suite` — it replays
all 17 upstream notebooks and reports per-variable relative L2 (the machine-precision internals
check, e.g. the LDDMM velocity field to ~1e-15). Heavier than the container flow and not folded into
it yet; it still carries the inline uv-cache / node-scratch tricks.

```bash
# whole sweep: 17-task array, one notebook per task, gpu_p/gpu_normal, H100, 6c/90G, 6h/task, %6
sbatch --export=NIL .claude/run_stalign_comparisons.sbatch \
    /lustre/.../squidpy-ports  /lustre/.../squidpy  /lustre/.../out
# a subset (indices into notebook_suite.NOTEBOOKS):
sbatch --array=5,9 --export=NIL .claude/run_stalign_comparisons.sbatch ...
```

One task per notebook (the sweep is ~51k upstream iterations — serial risks one hung notebook
spending the whole allocation). Each task stages both checkouts + the uv env to node-local scratch,
replays its pinned notebook, logs GPU util, and rsyncs PNG/JSON/manifest back to the durable dir.
Positional args work with `--export=NIL`; the script reconstructs only HOME/USER/PATH. The H100
constraint keeps current CUDA wheels off incompatible older GPUs.
