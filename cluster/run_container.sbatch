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
apptainer run --nv --writable-tmpfs --bind "$OUTPUT_DIR:/output" "$SIF" "$NOTEBOOK"
