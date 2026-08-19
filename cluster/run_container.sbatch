#!/usr/bin/env bash
#SBATCH --job-name=stalign-compare
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=01:30:00
#SBATCH --output=stalign-compare-%j.out
#SBATCH --error=stalign-compare-%j.err
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin"  # submit with --export=NIL; keep the tools reachable

# Thin launcher: schedule the reproduction container on the cluster, nothing more. Everything
# about *how* a result is made -- the pinned env, the squidpy fork, the execution -- lives in
# the container (container/stalign.def), documented in container/README.md so it runs on any
# GPU box, not just here. This file is the only Theislab-specific piece.
#
#   sbatch cluster/run_container.sbatch <stalign.sif> <notebook.ipynb> <output-dir>
SIF="${1:?path to a built stalign.sif}"
NOTEBOOK="${2:?comparison notebook basename, e.g. stalign-xenium-comparison.ipynb}"
OUTPUT_DIR="${3:?durable output dir}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not on PATH (try: module load apptainer)" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
# `STALIGN_NOTEBOOK_DIR` binds a host copy of the comparison notebooks over the ones baked
# into the image. The image is the reproducible artifact and normally supplies them; this
# exists so a notebook fix can be tested against the real environment without a 25-minute
# rebuild first. Leave it unset for a reproduction run.
BINDS=(--bind "$OUTPUT_DIR:/output")
if [[ -n "${STALIGN_NOTEBOOK_DIR:-}" ]]; then
    BINDS+=(--bind "$STALIGN_NOTEBOOK_DIR:/opt/squidpy-ports/docs/notebooks")
    echo "### notebooks bound from $STALIGN_NOTEBOOK_DIR (not the image's own)"
fi
apptainer run --nv --writable-tmpfs "${BINDS[@]}" "$SIF" "$NOTEBOOK"
