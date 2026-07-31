#!/usr/bin/env bash
#SBATCH --job-name=stalign-suite
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=stalign-compare-%j.out
#SBATCH --error=stalign-compare-%j.err

set -euo pipefail

PORTS_SOURCE="${1:-${PORTS_SOURCE:-$SLURM_SUBMIT_DIR}}"
SQUIDPY_SOURCE="${2:-${SQUIDPY_SOURCE:-$(dirname "$PORTS_SOURCE")/squidpy}}"
DURABLE_OUTPUT="${3:-${STALIGN_DURABLE_OUTPUT:-$PORTS_SOURCE/cluster-results/$SLURM_JOB_ID}}"

# `sbatch --export=NIL` avoids Slurm's user-environment retrieval path, which can
# fail before the batch step starts. Reconstruct only the two user values needed
# by this script on the compute node.
RUN_USER="$(id -un)"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
BASE_PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

LOCAL_BASE="${SLURM_TMPDIR:-/localscratch/$RUN_USER}/squidpy-stalign-comparisons"
JOB_ROOT="$LOCAL_BASE/jobs/$SLURM_JOB_ID"
CACHE_ROOT="$LOCAL_BASE/cache"
LOCAL_PORTS="$JOB_ROOT/squidpy-ports"
LOCAL_SQUIDPY="$JOB_ROOT/squidpy"
LOCAL_OUTPUT="$JOB_ROOT/results"
UV_ENV="$JOB_ROOT/.venv"

mkdir -p "$JOB_ROOT" "$LOCAL_OUTPUT" "$DURABLE_OUTPUT"
mkdir -p "$CACHE_ROOT/uv" "$CACHE_ROOT/xdg" "$CACHE_ROOT/matplotlib"
mkdir -p "$CACHE_ROOT/jax" "$CACHE_ROOT/torch" "$CACHE_ROOT/tmp/$SLURM_JOB_ID"

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export UV_PROJECT_ENVIRONMENT="$UV_ENV"
export UV_COMPILE_BYTECODE=1
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torch"
export TMPDIR="$CACHE_ROOT/tmp/$SLURM_JOB_ID"
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

echo "### host=$(hostname) job=$SLURM_JOB_ID started=$(date --iso-8601=seconds)"
echo "### ports_source=$PORTS_SOURCE"
echo "### squidpy_source=$SQUIDPY_SOURCE"
echo "### job_root=$JOB_ROOT"
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
nvidia-smi dmon -s pucvmet -d 10 > "$LOCAL_OUTPUT/gpu-dmon.log" 2>&1 &
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

"$UV_ENV/bin/python" - <<'PY' > "$LOCAL_OUTPUT/environment.txt"
from importlib.metadata import version

for package in ("squidpy-ports", "squidpy", "torch", "jax", "jaxlib", "numpy", "scipy"):
    print(f"{package}=={version(package)}")
PY
nvidia-smi -q > "$LOCAL_OUTPUT/nvidia-smi.txt"

echo "### running all upstream notebooks"
"$UV_ENV/bin/python" -m squidpy_ports.stalign.notebook_suite \
    all --output-dir "$LOCAL_OUTPUT" \
    2>&1 | tee "$LOCAL_OUTPUT/suite.log"

echo "### all comparisons completed"
