#!/usr/bin/env bash
#SBATCH --job-name=stalign-suite
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
# Per notebook, not per suite. The longest upstream fit is 10,000 iterations; a task that
# overruns this is a hung notebook, and killing it costs only its own evidence.
#SBATCH --time=06:00:00
# One task per notebook, throttled so the sweep does not take every H100 the QoS allows
# (`gpu_normal` permits 16 GPU/user). Override on the command line to rerun a subset,
# e.g. `--array=5,9` after reviewing a divergence.
#SBATCH --array=0-16%6
#SBATCH --nice=0
#SBATCH --output=stalign-compare-%A_%a.out
#SBATCH --error=stalign-compare-%A_%a.err

set -euo pipefail

PORTS_SOURCE="${1:-${PORTS_SOURCE:-$SLURM_SUBMIT_DIR}}"
SQUIDPY_SOURCE="${2:-${SQUIDPY_SOURCE:-$(dirname "$PORTS_SOURCE")/squidpy}}"

# Every task of one array writes into the same durable directory, so the evidence for the
# whole sweep stays together and per-notebook filenames never collide.
SUITE_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_TAG="${SUITE_ID}_${TASK_ID}"
DURABLE_OUTPUT="${3:-${STALIGN_DURABLE_OUTPUT:-$PORTS_SOURCE/cluster-results/$SUITE_ID}}"

# `sbatch --export=NIL` avoids Slurm's user-environment retrieval path, which can
# fail before the batch step starts. Reconstruct only the two user values needed
# by this script on the compute node.
RUN_USER="$(id -un)"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
BASE_PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

LOCAL_BASE="${SLURM_TMPDIR:-/localscratch/$RUN_USER}/squidpy-stalign-comparisons"
JOB_ROOT="$LOCAL_BASE/jobs/$TASK_TAG"
# Deliberately outside JOB_ROOT: the JAX and Torch caches persist on the node between
# tasks of the array rather than being rebuilt per task.
CACHE_ROOT="$LOCAL_BASE/cache"
# One uv cache per *node*, shared with the other cluster work in this workspace instead of
# one cache per project. It is already warm with the Torch and CUDA wheels this environment
# needs, so a task landing on a node that has run before installs from local disk rather
# than downloading ~1.5 GB again. It stays on node-local scratch, never Lustre: a uv cache
# is tens of thousands of small files, which is exactly what the shared filesystem should
# not hold. uv locks its own cache, so concurrent tasks sharing it are safe.
UV_CACHE="${STALIGN_UV_CACHE:-/localscratch/$RUN_USER/rapids-singlecell-equivalence/cache/uv}"
LOCAL_PORTS="$JOB_ROOT/squidpy-ports"
LOCAL_SQUIDPY="$JOB_ROOT/squidpy"
LOCAL_OUTPUT="$JOB_ROOT/results"
UV_ENV="$JOB_ROOT/.venv"

mkdir -p "$JOB_ROOT" "$LOCAL_OUTPUT" "$DURABLE_OUTPUT" "$UV_CACHE"
mkdir -p "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib"
mkdir -p "$CACHE_ROOT/jax" "$CACHE_ROOT/torch" "$CACHE_ROOT/tmp/$TASK_TAG"

export UV_CACHE_DIR="$UV_CACHE"
export UV_PROJECT_ENVIRONMENT="$UV_ENV"
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch"
export TMPDIR="$CACHE_ROOT/tmp/$TASK_TAG"
export STALIGN_COMPARISON_OUTPUT="$LOCAL_OUTPUT"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SETUPTOOLS_SCM_PRETEND_VERSION=1.8.3.dev0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SQUIDPY=1.8.3.dev0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HOME="$RUN_HOME"
export USER="$RUN_USER"
export PATH="$RUN_HOME/.local/bin:$BASE_PATH"

gpu_monitor_pid=""
copy_evidence() {
    status=$?
    if [[ -n "$gpu_monitor_pid" ]]; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
    mkdir -p "$DURABLE_OUTPUT"
    rsync -a "$LOCAL_OUTPUT/" "$DURABLE_OUTPUT/" 2>/dev/null || true
    echo "### exit_status=$status evidence=$DURABLE_OUTPUT finished=$(date --iso-8601=seconds)"
    return "$status"
}
trap copy_evidence EXIT

echo "### host=$(hostname) job=$SLURM_JOB_ID task=$TASK_ID started=$(date --iso-8601=seconds)"
echo "### ports_source=$PORTS_SOURCE"
echo "### squidpy_source=$SQUIDPY_SOURCE"
echo "### job_root=$JOB_ROOT"
echo "### uv_cache=$UV_CACHE size=$(du -sh "$UV_CACHE" 2>/dev/null | cut -f1)"
echo "### durable_output=$DURABLE_OUTPUT"

for source in "$PORTS_SOURCE" "$SQUIDPY_SOURCE"; do
    if [[ ! -d "$source" ]]; then
        echo "Missing source checkout: $source" >&2
        exit 2
    fi
done

# Verify the pinned submodule while its Git metadata is still available on shared
# storage. The scratch copy deliberately excludes all Git metadata.
export SQUIDPY_PORTS_STALIGN_SHA
if SQUIDPY_PORTS_STALIGN_SHA="$(git -C "$PORTS_SOURCE/vendor/STalign" rev-parse HEAD 2>/dev/null)"; then
    :
elif [[ -f "$PORTS_SOURCE/.stalign-upstream-sha" ]]; then
    SQUIDPY_PORTS_STALIGN_SHA="$(<"$PORTS_SOURCE/.stalign-upstream-sha")"
else
    echo "Cannot verify the staged STalign source SHA." >&2
    exit 2
fi
echo "### stalign_sha=$SQUIDPY_PORTS_STALIGN_SHA"

nvidia-smi
# Per task: every task of the array copies into one durable directory.
nvidia-smi dmon -s pucvmet -d 10 > "$LOCAL_OUTPUT/gpu-dmon-$TASK_TAG.log" 2>&1 &
gpu_monitor_pid=$!

rsync -a --delete --exclude=.git --exclude=.venv --exclude=_build --exclude=cluster-results \
    "$PORTS_SOURCE/" "$LOCAL_PORTS/"
rsync -a --delete --exclude=.git --exclude=.venv --exclude=_build \
    "$SQUIDPY_SOURCE/" "$LOCAL_SQUIDPY/"

cd "$LOCAL_PORTS"
command -v uv >/dev/null || {
    echo "uv is not on PATH. Install it in ~/.local/bin before submitting." >&2
    exit 2
}

# All installation and downloads happen inside the allocated compute job. The ports
# environment supplies upstream PyTorch; the staged sibling supplies the JAX port.
uv sync --frozen
uv pip install --python "$UV_ENV/bin/python" --editable "$LOCAL_SQUIDPY[jax]" "jax[cuda12]"

"$UV_ENV/bin/python" - <<'PY' > "$LOCAL_OUTPUT/environment-$TASK_TAG.txt"
from importlib.metadata import version

for package in ("squidpy-ports", "squidpy", "torch", "jax", "jaxlib", "numpy", "scipy"):
    print(f"{package}=={version(package)}")
PY
nvidia-smi -q > "$LOCAL_OUTPUT/nvidia-smi-$TASK_TAG.txt"

# `--index` resolves the array task id against `notebook_suite.NOTEBOOKS`, so the batch
# script never carries a second copy of that ordering. Submitted without an array (or with
# STALIGN_NOTEBOOK=all) this still runs the whole suite in one allocation.
SELECTION=("--index" "$TASK_ID")
if [[ -n "${STALIGN_NOTEBOOK:-}" ]]; then
    SELECTION=("$STALIGN_NOTEBOOK")
elif [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTION=("all")
fi

echo "### running upstream notebook selection: ${SELECTION[*]}"
"$UV_ENV/bin/python" -m squidpy_ports.stalign.notebook_suite \
    "${SELECTION[@]}" --output-dir "$LOCAL_OUTPUT" \
    2>&1 | tee "$LOCAL_OUTPUT/suite-$TASK_TAG.log"

echo "### comparisons completed for selection: ${SELECTION[*]}"
