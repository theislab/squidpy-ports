#!/usr/bin/env bash
#SBATCH --job-name=stalign-smoke
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
# One iteration per notebook, so the whole suite is minutes rather than the ~45 GPU-minutes a
# real sweep costs. A task that overruns this is hung, not slow.
#SBATCH --time=00:40:00
# One task per *substitution site*, not per notebook: the parameters are wired once in the
# harness, so a mismatch shows on any notebook that reaches the same interception. Three
# cover all of them -- `LDDMM` (2D, fourteen notebooks), `LDDMM_3D_to_slice`, and
# `L_T_from_points`. Submit with `--array=1,2,3`; override to check others.
#SBATCH --array=1,2,3
#SBATCH --nice=0

# The parameter gate. It answers one question -- do both sides receive the same solver
# parameters -- and answers it in minutes instead of after every fit has converged.
#
# It exists because they once did not: `align_stalign_image` resolves `_IMAGE_DEFAULTS`
# (`a` 20, `niter` 200, `diffeo_start` 100, `epV` 1.0), which are squidpy's own choices for
# images and deliberately not upstream's, and fourteen of the seventeen notebooks pass none
# of those four. A full sweep reported that as a port divergence; one iteration would have
# reported it as a mismatch on the first notebook.
#
# `STALIGN_SMOKE_NITER` caps both passes identically -- see `notebook_suite.smoke_niter`.
set -euo pipefail

PORTS_SOURCE="${1:-${PORTS_SOURCE:-$SLURM_SUBMIT_DIR}}"
SQUIDPY_SOURCE="${2:-${SQUIDPY_SOURCE:-$(dirname "$PORTS_SOURCE")/squidpy}}"
# One directory for the whole gate, not one per array task.
OUTPUT="${3:-$PORTS_SOURCE/cluster-results/smoke-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"

export STALIGN_SMOKE_NITER="${STALIGN_SMOKE_NITER:-1}"
echo "### parameter gate: niter=$STALIGN_SMOKE_NITER over every mapped notebook"

# Everything else -- staging, the localscratch layout, the caches, the atlas -- is the sweep's
# own launcher, reused rather than copied so the two cannot drift.
#
# Resolved from PORTS_SOURCE, not from this script's own path: Slurm runs a batch script
# from a copy in its spool directory, so `dirname "$BASH_SOURCE"` is that spool and the
# launcher is not beside it.
LAUNCHER="$PORTS_SOURCE/.claude/run_stalign_comparisons.sbatch"
[[ -x "$LAUNCHER" || -f "$LAUNCHER" ]] || { echo "Missing launcher: $LAUNCHER" >&2; exit 2; }
exec bash "$LAUNCHER" "$PORTS_SOURCE" "$SQUIDPY_SOURCE" "$OUTPUT"
