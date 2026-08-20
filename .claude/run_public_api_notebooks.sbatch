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

# Pinned, and installed from the clone rather than fetched: the canonical rev every other
# result in this workspace is measured against. The notebooks touch no comparison-only flag,
# so this behaves as plain 2bff194f.
SQUIDPY="$WORKSPACE/squidpy-clone"
# The vendored upstream datasets and the Allen atlas. Large files on Lustre, read once each;
# symlinked into the working directory rather than copied, because the notebooks resolve
# their inputs relative to the current directory.
DATA="$WORKSPACE/squidpy-ports/vendor/STalign/docs"
ATLAS="$WORKSPACE/atlas-cache"

RUN_USER="$(id -un)"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
export HOME="$RUN_HOME" USER="$RUN_USER"
export PATH="$RUN_HOME/.local/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

SCRATCH="/localscratch/$RUN_USER/stalign-nb-$SLURM_JOB_ID"
WORK="$SCRATCH/work"
# One uv cache per node, outside SCRATCH so it survives this job: the torch and CUDA wheels
# are ~2.5 GB and re-downloading them is the whole difference between a 30 s setup and a
# 6 min one. Node-local, never Lustre -- a uv cache is tens of thousands of small files.
export UV_CACHE_DIR="/localscratch/$RUN_USER/uv-cache"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/.venv"
export UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/mpl" XDG_CACHE_HOME="$SCRATCH/xdg"
mkdir -p "$WORK" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$OUT"
trap 'rm -rf "$SCRATCH"' EXIT

export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=agg
# Determinism, the half that has to precede library load. Rank 2 reproduces to ~1e-12
# without it; rank 3 moved cells up to 109 um between two runs of the same code.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# uv builds squidpy in a temp dir that carries no git metadata, so setuptools_scm has
# nothing to read and the build fails rather than guessing a version.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SQUIDPY=1.8.3.dev0

echo "### host=$(hostname) job=$SLURM_JOB_ID started=$(date --iso-8601=seconds)"
echo "### ports=$PORTS @ $(git -C "$PORTS" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "### squidpy=$SQUIDPY @ $(git -C "$SQUIDPY" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "### scratch=$SCRATCH out=$OUT"
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
