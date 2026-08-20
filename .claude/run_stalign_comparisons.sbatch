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

# /localscratch is node-local and keeps whatever previous jobs left on it. Each task removes
# its own JOB_ROOT on exit now, but a task killed before that cannot, and the debris is what
# fills the disk: one sweep lost 15 of 17 tasks to `mkdir ... No space left on device` before
# a single log line was written. Prune what earlier *jobs* left behind, matching on the suite
# id -- never on TASK_TAG alone, or concurrent tasks of this same array would delete each
# other's working directories out from under them.
mkdir -p "$LOCAL_BASE/jobs"
for stale in "$LOCAL_BASE"/jobs/*; do
    [[ -d "$stale" ]] || continue
    case "$(basename "$stale")" in
        "${SUITE_ID}_"*) continue ;;
        *) rm -rf "$stale" 2>/dev/null || true ;;
    esac
done
echo "### localscratch: $(df -h /localscratch 2>/dev/null | tail -1 | awk '{print $4" free of "$2}')"

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
# Determinism, the half that has to precede library load -- see
# `notebook_suite._pin_fit_determinism` for the in-process half and for why rank 3
# needs this at all (rank 2 already reproduces to ~1e-12; rank 3 moved cells up to
# 109um and reassigned 15-22% of Allen regions between two runs of the *same* code).
# cuBLAS needs a fixed workspace or its reductions may reorder; XLA needs telling.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SETUPTOOLS_SCM_PRETEND_VERSION=1.8.3.dev0
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SQUIDPY=1.8.3.dev0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HOME="$RUN_HOME"
export USER="$RUN_USER"
export PATH="$RUN_HOME/.local/bin:$BASE_PATH"
# The Allen atlas the two volume-to-section notebooks pull (~40 MB of NRRD plus the ontology
# CSV). Upstream's `download_aba_image_labels` re-fetches unconditionally -- it opens its
# target `'wb'` with no existence check (STalign.py:1885) -- and every replay pass gets a
# fresh working directory, so one comparison would download the atlas four times and the pair
# eight. Kept beside the sources on Lustre rather than under $LOCAL_OUTPUT because it has to
# outlive the job and be shared by both array tasks; it is a handful of large files, so it
# costs the metadata server nothing. Unset, the suite keeps upstream's own download behaviour.
export STALIGN_ATLAS_CACHE="${STALIGN_ATLAS_CACHE:-$(dirname "$PORTS_SOURCE")/atlas-cache}"
mkdir -p "$STALIGN_ATLAS_CACHE"

gpu_monitor_pid=""
copy_evidence() {
    status=$?
    if [[ -n "$gpu_monitor_pid" ]]; then
        kill "$gpu_monitor_pid" 2>/dev/null || true
        wait "$gpu_monitor_pid" 2>/dev/null || true
    fi
    mkdir -p "$DURABLE_OUTPUT"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$LOCAL_OUTPUT/" "$DURABLE_OUTPUT/" 2>/dev/null || true
    else
        cp -a "$LOCAL_OUTPUT/." "$DURABLE_OUTPUT/" 2>/dev/null || true
    fi
    # Only after the evidence is on Lustre: the per-task venv is several GB and nothing else
    # removed it, so five runs in one day filled /localscratch and later array tasks died with
    # "No space left on device" before they could log anything. CACHE_ROOT is deliberately left
    # alone -- it lives outside JOB_ROOT so the JAX and Torch caches survive between tasks.
    rm -rf "$JOB_ROOT" 2>/dev/null || true
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
echo "### tools: rsync=$(command -v rsync || echo -) tar=$(command -v tar || echo -) git=$(command -v git || echo -)"

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

# Same treatment for the squidpy half of the pin. An editable install of a staged working
# tree records no commit in pip's metadata, so without this every manifest would say
# `squidpy_commit: null` -- on pages whose claim is that a result is never separated from
# its code. Resolved here while Git metadata is still on shared storage; the scratch copy
# deliberately excludes it.
export SQUIDPY_COMMIT
if SQUIDPY_COMMIT="$(git -C "$SQUIDPY_SOURCE" rev-parse HEAD 2>/dev/null)"; then
    :
elif [[ -f "$SQUIDPY_SOURCE/.squidpy-commit" ]]; then
    SQUIDPY_COMMIT="$(<"$SQUIDPY_SOURCE/.squidpy-commit")"
else
    SQUIDPY_COMMIT=""
    echo "### warning: squidpy commit unresolved; manifests will record null" >&2
fi
echo "### squidpy_commit=${SQUIDPY_COMMIT:-unresolved}"

# Guarded rather than assumed: a CPU allocation has no `nvidia-smi`, and under `set -e` a
# bare call would abort the launcher before any fit ran. The rank-3 flag jobs run on CPU on
# purpose -- upstream's `LDDMM_3D_to_slice` is bit-identical across processes there and
# wanders on CUDA -- so this launcher has to survive without a GPU.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    # Per task: every task of the array copies into one durable directory.
    nvidia-smi dmon -s pucvmet -d 10 > "$LOCAL_OUTPUT/gpu-dmon-$TASK_TAG.log" 2>&1 &
    gpu_monitor_pid=$!
else
    echo "### no nvidia-smi: running on CPU, no GPU monitoring"
fi

# Not every compute-node image here carries rsync: job 39552781 reached supergpu26, where
# `rsync` is absent although the login node has it at /usr/bin/rsync. tar is in every image
# and takes the same excludes, so staging succeeds either way rather than dying at the last
# step before the real work. Excludes matter for a full checkout; the staged tree already
# omits them.
# The `#SBATCH --output` logs land in the submit directory, which *is* the tree being
# staged, so tar reads a file that is still being appended to and exits non-zero:
# "file changed as we read it". Harmless until two tasks share a node, which is exactly
# what pinning `--nodelist` for a determinism check does -- three tasks died that way.
stage_tree() {
    src="$1"
    dest="$2"
    mkdir -p "$dest"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --exclude=.git --exclude=.venv --exclude=_build \
            --exclude=cluster-results --exclude="stalign-compare-*" "$src/" "$dest/"
    else
        tar -C "$src" --exclude=.git --exclude=.venv --exclude=_build \
            --exclude=cluster-results --exclude="stalign-compare-*" -cf - . | tar -C "$dest" -xf -
    fi
}

stage_tree "$PORTS_SOURCE" "$LOCAL_PORTS"
stage_tree "$SQUIDPY_SOURCE" "$LOCAL_SQUIDPY"

cd "$LOCAL_PORTS"
command -v uv >/dev/null || {
    echo "uv is not on PATH. Install it in ~/.local/bin before submitting." >&2
    exit 2
}

# All installation and downloads happen inside the allocated compute job. The ports
# environment supplies upstream PyTorch; the staged sibling supplies the JAX port.
uv sync --frozen
# The CUDA plugin is the default because the sweep runs on H100s; a CPU allocation sets
# `JAX_INSTALL_SPEC=jax` so it does not pull a GPU backend it cannot use.
uv pip install --python "$UV_ENV/bin/python" --editable "$LOCAL_SQUIDPY[jax]" "${JAX_INSTALL_SPEC:-jax[cuda12]}"

"$UV_ENV/bin/python" - <<'PY' > "$LOCAL_OUTPUT/environment-$TASK_TAG.txt"
from importlib.metadata import version

for package in ("squidpy-ports", "squidpy", "torch", "jax", "jaxlib", "numpy", "scipy"):
    print(f"{package}=={version(package)}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$LOCAL_OUTPUT/nvidia-smi-$TASK_TAG.txt"
fi

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
