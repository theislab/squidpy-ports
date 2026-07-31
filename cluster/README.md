# Running STalign comparisons on the Theislab HPC

Submit [`run_stalign_comparisons.sbatch`](run_stalign_comparisons.sbatch) from a
shared-storage checkout. The login node only submits the job; repository staging,
environment installation, data downloads, compilation, and comparisons all happen
inside the allocated GPU job.

```bash
sbatch cluster/run_stalign_comparisons.sbatch
```

Defaults: `gpu_p` / `gpu_normal`, one H100 80 GB GPU, six CPUs, 90 GB RAM,
and twelve hours. The job:

1. stages `squidpy-ports` and a sibling `squidpy` checkout to node-local scratch;
2. puts the uv environment and uv, XDG, JAX, Torch, Matplotlib, and temporary caches
   on local scratch;
3. creates the environment with upstream PyTorch and CUDA-enabled JAX inside the job;
4. replays all 17 pinned upstream notebooks sequentially, comparing the 14 2D LDDMM
   workflows and one affine workflow while emitting explicit status panels for the two
   not-yet-ported 3D-to-slice workflows;
5. logs GPU utilization throughout; and
6. copies PNG panels, CSV/JSON metrics, manifests, and logs back to shared storage.

The default durable result directory is `cluster-results/$SLURM_JOB_ID` in the shared
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
