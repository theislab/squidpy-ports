#!/usr/bin/env bash
#SBATCH --job-name=stalign-nb
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=stalign-nb-%j.out
#SBATCH --error=stalign-nb-%j.err

# Execute the public-API notebooks (`docs/notebooks/squidpy-api/`) on a GPU node.
#
# Deliberately NOT the replay harness in `run_stalign_comparisons.sbatch`: these notebooks
# call only squidpy's public API, so there is no upstream pass, no second checkout to stage,
# and nothing to rsync. Both sources stay on Lustre and are read once; everything that
# churns -- the venv, the uv cache, the working directory -- lives on node-local scratch.
#
#   sbatch --export=NIL .claude/run_public_api_notebooks.sbatch [<ports-clone>] [<out-dir>]
#   sbatch --export=NIL .claude/run_public_api_notebooks.sbatch '' '' starmap-allen3Datlas.ipynb
#
# With no third argument every notebook in the directory runs, in name order.
set -euo pipefail

WORKSPACE=/lustre/groups/ml01/workspace/selman.ozleyen/stalign-3d
# A real git clone, so `git fetch && git checkout` picks up new notebooks -- the rsync route
# it replaces once shipped 2.1 GB of caches and killed three tasks by racing a staging tar.
PORTS="${1:-$WORKSPACE/ports-clone}"
OUT="${2:-$WORKSPACE/notebooks-out/$SLURM_JOB_ID}"
(( $# > 2 )) && shift 2 || set --
SELECTION=("$@")

# Tracked at its branch tip, like the ports clone, rather than pinned to a SHA: a pinned rev
# silently goes stale, and this one did -- the notebooks ran for a day against 2bff194f while
# `sigmaR` for the volume case had already been retuned from upstream's 1e8 to 1e6 further up
# the branch. Override with `SQUIDPY_REV=<sha>` to reproduce an older result.
SQUIDPY="$WORKSPACE/squidpy-clone"
SQUIDPY_REV="${SQUIDPY_REV:-origin/feat/experimental-fit-core}"
# The vendored upstream datasets and the Allen atlas. Large files on Lustre, read once each;
# symlinked into the working directory rather than copied, because the notebooks resolve
# their inputs relative to the current directory.
DATA="$WORKSPACE/squidpy-ports/vendor/STalign/docs"
ATLAS="$WORKSPACE/atlas-cache"

RUN_USER="$(id -un)"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
export HOME="$RUN_HOME" USER="$RUN_USER"
export PATH="$RUN_HOME/.local/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

# /localscratch is node-local and much the faster choice, but it is shared with every other
# user of the node and is not cleaned between jobs: supergpu14 was handed over with 112 KB free
# of 131 GB, all of it in other users' unreadable directories, so there was nothing to prune
# and `mkdir` simply failed. Falling back to Lustre is worse for the metadata server -- a venv
# is tens of thousands of small files -- and better than not running at all.
SCRATCH_BASE="/localscratch/$RUN_USER"
if ! mkdir -p "$SCRATCH_BASE" 2>/dev/null ||
   [[ "$(df -BM --output=avail "$SCRATCH_BASE" 2>/dev/null | tail -1 | tr -dc '0-9')" -lt 20000 ]]; then
    SCRATCH_BASE="${STALIGN_SCRATCH_FALLBACK:-/lustre/boost_ai/users/$RUN_USER/stalign-scratch}"
    mkdir -p "$SCRATCH_BASE"
    echo "### /localscratch unusable on $(hostname -s); falling back to $SCRATCH_BASE"
fi

SCRATCH="$SCRATCH_BASE/stalign-nb-$SLURM_JOB_ID"
WORK="$SCRATCH/work"
# One uv cache per scratch base, outside SCRATCH so it survives this job: the torch and CUDA
# wheels are ~2.5 GB and re-downloading them is the whole difference between a 30 s setup and
# a 6 min one.
export UV_CACHE_DIR="$SCRATCH_BASE/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venv"
# Deliberately NOT `UV_LINK_MODE=copy`, which the comparison launcher sets: there the cache and
# the venv could land on different mounts, so copying was the safe choice. Here both are under
# /localscratch, one filesystem, so uv's default hardlink costs nothing -- copy made it write
# 5.1 GB and turned a seconds-long install into two minutes. `UV_COMPILE_BYTECODE` is left off
# for the same reason: it byte-compiles all of torch and jax up front to save import time later,
# and there is no later -- this env is built for one job and deleted with the scratch dir.
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$WORK" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$OUT"
trap 'rm -rf "$SCRATCH"' EXIT

# No MPLBACKEND here, deliberately. `agg` is the right choice for a headless script, and the
# wrong one inside a kernel: it overrides ipykernel's inline backend, which is the thing that
# turns a figure into an `image/png` output. Setting it produced two notebooks with every cell
# green, every number right, and not one of their figures -- 45 kB where the reference is 1.2 MB.
# Measured on the node: agg -> ['text/plain'], unset -> ['text/plain', 'image/png'].
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
# Determinism, the half that has to precede library load. Rank 2 reproduces to ~1e-12
# without it; rank 3 moved cells up to 109 um between two runs of the same code.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# uv builds squidpy in a temp dir that carries no git metadata, so setuptools_scm has
# nothing to read and the build fails rather than guessing a version.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SQUIDPY=1.8.3.dev0

echo "### host=$(hostname) job=$SLURM_JOB_ID started=$(date --iso-8601=seconds)"

# The whole reason the ports source is a clone rather than an rsynced tree: new notebooks
# arrive by fetching them. Done here, in the job, so the log records the exact SHA that ran
# rather than whatever the directory happened to be left at. Pin a rev with
# `PORTS_REV=<sha> sbatch --export=ALL,PORTS_REV` when reproducing an older result.
git -C "$PORTS" fetch --quiet origin
git -C "$PORTS" checkout --quiet --detach "${PORTS_REV:-origin/main}"
git -C "$SQUIDPY" fetch --quiet origin
git -C "$SQUIDPY" checkout --quiet --detach "$SQUIDPY_REV"
echo "### ports=$PORTS @ $(git -C "$PORTS" rev-parse HEAD)"
echo "### squidpy=$SQUIDPY @ $(git -C "$SQUIDPY" rev-parse HEAD) ($SQUIDPY_REV)"
echo "### scratch=$SCRATCH (base $SCRATCH_BASE) out=$OUT"
echo "### uv_cache=$UV_CACHE_DIR size=$(du -sh "$UV_CACHE_DIR" 2>/dev/null | cut -f1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

setup_start=$SECONDS
# The environment is exactly what the committed pyproject.toml + uv.lock resolve to --
# `--frozen` never re-resolves on the node. The two notebook-only additions are pinned here
# instead of in the lock: squidpy is a git checkout at a rev, not a released version.
uv sync --frozen --project "$PORTS" --no-dev
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
    "$SQUIDPY[jax]" "jax[cuda12]" nbclient ipykernel
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
echo "### setup took $((SECONDS - setup_start))s"

"$PY" - <<'PROBE' | tee "$OUT/environment.txt"
from importlib.metadata import version
import jax
for pkg in ("squidpy", "spatialdata", "jax", "jaxlib", "numpy", "scipy", "pandas", "pynrrd"):
    print(f"{pkg}=={version(pkg)}")
print(f"devices={jax.devices()}")
PROBE

# Inputs the notebooks open by relative path. Symlinks, not copies: the Allen volumes alone
# are ~40 MB each and nothing writes to them.
for d in heart_data merfish_data starmap_data visium_data xenium_data; do
    [[ -d "$DATA/$d" ]] && ln -sfn "$DATA/$d" "$WORK/$d"
done
ln -sfn "$ATLAS/ara_nissl_50.nrrd" "$WORK/ara_nissl_50.nrrd"
ln -sfn "$ATLAS/annotation_50.nrrd" "$WORK/annotation_50.nrrd"
ln -sfn "$ATLAS/query.csv" "$WORK/allen_ontology.csv"

NB_DIR="$PORTS/docs/notebooks/squidpy-api"
if [[ ${#SELECTION[@]} -eq 0 ]]; then
    mapfile -t SELECTION < <(cd "$NB_DIR" && ls *.ipynb)
fi
echo "### notebooks: ${SELECTION[*]}"

status=0
for nb in "${SELECTION[@]}"; do
    echo "### === $nb ==="
    # nbclient rather than nbconvert: it is one fewer dependency and it takes the working
    # directory directly, which is the whole point here -- the notebooks resolve their
    # inputs relative to it, and that directory is on node-local scratch.
    if NB_SRC="$NB_DIR/$nb" NB_DST="$OUT/$nb" NB_CWD="$WORK" "$PY" - <<'RUN'; then
import os, time, nbformat
from nbclient import NotebookClient

src, dst, cwd = os.environ["NB_SRC"], os.environ["NB_DST"], os.environ["NB_CWD"]
nb = nbformat.read(src, as_version=4)
start = time.monotonic()
NotebookClient(nb, timeout=None, kernel_name="python3",
               resources={"metadata": {"path": cwd}}).execute()
nbformat.write(nb, dst)
print(f"### ran {len(nb.cells)} cells in {time.monotonic() - start:.1f}s -> {dst}")
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            print("    " + "".join(out.text).strip().replace("\n", "\n    "))
RUN
        echo "### $nb OK"
    else
        echo "### $nb FAILED" >&2
        status=1
    fi
done

echo "### finished=$(date --iso-8601=seconds) exit=$status executed=$OUT"
exit "$status"
