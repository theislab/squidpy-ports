# Running STalign comparisons on the Theislab HPC

Submit [`run_stalign_comparisons.sbatch`](run_stalign_comparisons.sbatch) from a
shared-storage checkout. The login node only submits the job; repository staging,
environment installation, data downloads, compilation, and comparisons all happen
inside the allocated GPU job.

```bash
sbatch cluster/run_stalign_comparisons.sbatch
```

Defaults: a **17-task job array, one notebook per task**, on `gpu_p` / `gpu_normal` with
one H100 80 GB GPU, six CPUs, 90 GB RAM and six hours *per task*, at most six tasks at a
time (`--array=0-16%6`). One task per notebook rather than one job for the suite because
the whole sweep is ~51,000 upstream iterations: run serially that is many hours in a
single allocation, and one hung notebook spends the lot. Each task:

1. stages `squidpy-ports` and a sibling `squidpy` checkout to node-local scratch;
2. puts the uv environment and temporary caches on node-local scratch, with the JAX and
   Torch caches outside the per-task directory. `UV_CACHE_DIR` points at one cache per
   *node* shared with the workspace's other cluster work
   (`/localscratch/$USER/rapids-singlecell-equivalence/cache/uv`, override with
   `STALIGN_UV_CACHE`), so a task landing on a node that has run before installs from
   local disk instead of downloading ~1.5 GB of Torch and CUDA-JAX again. The cache never
   goes on Lustre: it is tens of thousands of small files, and uv locks it, so concurrent
   tasks can share it;
3. creates the environment with upstream PyTorch and CUDA-enabled JAX inside the job;
4. replays the one pinned upstream notebook its `SLURM_ARRAY_TASK_ID` selects -- the index
   resolves against `notebook_suite.NOTEBOOKS`, so the ordering has one definition. Of the
   17, fourteen are 2D LDDMM comparisons, one is the affine landmark comparison, and two
   emit explicit status panels for the not-ported 3D-to-slice workflows;
5. logs GPU utilization throughout; and
6. copies its PNG panel, JSON metrics, manifest, status file, and logs back to shared
   storage.

Every task of one array writes into the same durable directory, so the sweep's evidence
stays together; per-task files carry a `$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID` tag and
per-notebook files are named after the notebook, so nothing collides.

Rerun a subset by overriding the array -- indices are positions in
`notebook_suite.NOTEBOOKS`:

```bash
sbatch --array=5,9 cluster/run_stalign_comparisons.sbatch
```

To run the whole suite in one allocation instead, submit without an array
(`sbatch --array=0 --export=ALL,STALIGN_NOTEBOOK=all ...`, or drop `--array` entirely).

The default durable result directory is `cluster-results/$SLURM_ARRAY_JOB_ID` in the shared
`squidpy-ports` checkout. Override source or output locations at submission time:

```bash
sbatch --export=NIL cluster/run_stalign_comparisons.sbatch \
    /lustre/groups/ml01/workspace/$USER/squidpy-ports \
    /lustre/groups/ml01/workspace/$USER/squidpy \
    /lustre/groups/ml01/workspace/$USER/stalign-comparisons/run-01
```

The positional form works with `--export=NIL`, avoiding Slurm's user-environment
retrieval path. The script reconstructs only `HOME`, `USER`, and the uv path inside the
batch allocation.

The H100 constraint avoids placing current CUDA wheels on an incompatible older GPU.
Override it on the `sbatch` command line only after verifying that the locked Torch and
JAX builds support the target architecture.
