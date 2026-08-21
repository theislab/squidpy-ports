#!/usr/bin/env bash
# Run the notebook harness inside the held allocation instead of queueing a new job.
#
#   ./on_held_node.sh                       # every notebook
#   ./on_held_node.sh xenium-xenium.ipynb   # just these
#
# `--overlap` shares the held allocation; without it srun waits for resources it will never
# get, because the hold job already owns them.
set -euo pipefail
WORKSPACE=/lustre/groups/ml01/workspace/selman.ozleyen/stalign-3d
HELD="$WORKSPACE/held.json"
[[ -f "$HELD" ]] || { echo "No held.json -- is the hold job running? squeue --me" >&2; exit 2; }
JOB=$(python3 -c "import json;print(json.load(open('$HELD'))['job_id'])")
squeue -j "$JOB" -h -o "%T" 2>/dev/null | grep -q RUNNING || {
    echo "Hold job $JOB is not RUNNING. Resubmit .claude/hold_node.sbatch." >&2; exit 2; }
exec srun --jobid="$JOB" --overlap --cpus-per-task=6 \
    bash "$WORKSPACE/ports-clone/.claude/run_public_api_notebooks.sbatch" "" "" "$@"
