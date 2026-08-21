#!/usr/bin/env bash
#SBATCH --job-name=stalign-hold
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
# 12h is the ceiling `gpu_priority` allows -- 1-00:00:00 and above are refused with
# QOSMaxWallDurationPerJobLimit. Resubmit to extend; there is no way to lengthen it in place.
#SBATCH --time=12:00:00
#SBATCH --output=stalign-hold-%j.out
#SBATCH --error=stalign-hold-%j.err

# A node held for the day, so notebook runs are seconds instead of a queue wait.
#
# The GPU partition has been congested enough that a one-notebook rerun costs more time
# queueing than fitting. Everything runs inside this allocation via `srun --overlap`, which
# creates a job *step* rather than asking for another GPU -- and because the step inherits
# this job's SLURM_JOB_ID, the harness lands on the same /localscratch path every time and
# finds its uv cache already warm.
#
# It holds a GPU for up to 12 hours. That is a scarce shared resource: `scancel` it the moment
# the day's work is done, and do not leave it idle overnight.
set -euo pipefail

WORKSPACE=/lustre/groups/ml01/workspace/selman.ozleyen/stalign-3d
RUN_USER="$(id -un)"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
export HOME="$RUN_HOME" USER="$RUN_USER"
export PATH="$RUN_HOME/.local/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"

echo "### host=$(hostname) job=$SLURM_JOB_ID started=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail fast rather than hold a GPU that cannot do the work. /localscratch is shared with every
# other user of the node and is not cleaned between jobs: supergpu14 was handed over with
# 112 KB free of 131 GB, all of it in other users' unreadable directories, so the harness died
# on `mkdir ... No space left on device` with nothing to prune. Without this check that is a
# node held for twelve hours and discovered broken on first use.
FREE_MB="$(df -BM --output=avail /localscratch 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "### localscratch: ${FREE_MB:-?} MB free of $(df -BG --output=size /localscratch 2>/dev/null | tail -1 | tr -d ' ')"
if [[ -z "$FREE_MB" || "$FREE_MB" -lt 20000 ]]; then
    echo "Refusing to hold $(hostname): needs ~20 GB free on /localscratch for the venv and" >&2
    echo "caches, found ${FREE_MB:-0} MB. Resubmit with --exclude=$(hostname -s)." >&2
    exit 1
fi

# Publish the allocation so any session can find and share it.
cat > "$WORKSPACE/held.json" <<JSON
{"job_id": "$SLURM_JOB_ID", "node": "$(hostname)", "workspace": "$WORKSPACE",
 "started": "$(date --iso-8601=seconds)", "time_limit": "12:00:00"}
JSON
echo "### held.json written. Run work with:"
echo "###   $WORKSPACE/ports-clone/.claude/on_held_node.sh [notebook.ipynb ...]"
echo "### Release it with: scancel $SLURM_JOB_ID"

# `sleep infinity` rather than `salloc`: it survives the submitting shell going away.
sleep infinity
