#!/usr/bin/env bash
#SBATCH --job-name=imputation-nb
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# Execute the SpaGE and Tangram public-API notebooks on a CPU node. Everything -- the clones, the venv,
# the SpaGE datasets -- lives on node-local scratch; only the executed notebooks are written to OUT.
#
#   sbatch -o $HOME/imputation-nb/%j.log .claude/run_imputation_notebooks.sbatch [<ports-rev>] [<squidpy-rev>]
set -euo pipefail

PORTS_REV="${1:-imputation-notebooks}"
SQUIDPY_REV="${2:-feat/impute-tangram}"
OUT="$HOME/imputation-nb/$SLURM_JOB_ID"
S="/localscratch/$USER/imputation-nb-$SLURM_JOB_ID"
export UV_CACHE_DIR="/localscratch/$USER/uv-cache" UV_PROJECT_ENVIRONMENT="$S/.venv"
export TMPDIR="$S/tmp" MPLCONFIGDIR="$S/mpl" NUMBA_CACHE_DIR="$S/numba"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SQUIDPY=1.8.3.dev0
mkdir -p "$S/work" "$TMPDIR" "$MPLCONFIGDIR" "$UV_CACHE_DIR" "$OUT"
trap 'rm -rf "$S"' EXIT
cd "$S"

git clone -q --recurse-submodules=vendor/SpaGE --recurse-submodules=vendor/Tangram --shallow-submodules -b "$PORTS_REV" https://github.com/selmanozleyen/squidpy-ports.git ports
git clone -q -b "$SQUIDPY_REV" https://github.com/selmanozleyen/squidpy.git squidpy
echo "### ports $(git -C ports rev-parse HEAD) squidpy $(git -C squidpy rev-parse HEAD)"

~/.local/bin/uv sync -q --frozen --project ports --extra spage --extra tangram
~/.local/bin/uv pip install -q --python "$UV_PROJECT_ENVIRONMENT/bin/python" "./squidpy[jax]" loompy nbclient ipykernel
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

# The SpaGE tutorial reads two files of upstream's Zenodo archive from its working directory. They are
# extracted once into DATA, on shared storage: a node's /localscratch can be too full to hold them.
DATA="${SPAGE_DATA:-/ictstr01/boost_ai/users/$USER/spage-data}"
mkdir -p "$DATA"
FILES=(SSp_exons_matrix.csv osmFISH_SScortex_mouse_all_cells.loom)
if [[ ! -f "$DATA/${FILES[1]}" ]]; then
    curl -sSL -o "$DATA/spage.zip" "https://zenodo.org/api/records/3967291/files/SpaGE%20Datasets.zip/content"
    for f in "${FILES[@]}"; do
        unzip -o -j -q "$DATA/spage.zip" "$(unzip -Z1 "$DATA/spage.zip" | grep -m1 "/$f\$\|^$f\$")" -d "$DATA"
    done
    rm "$DATA/spage.zip"
fi
for f in "${FILES[@]}"; do ln -sfn "$DATA/$f" "work/$f"; done
# squidpy's datasets download to `data/` under the working directory; the Tangram ones are 3+ GB
mkdir -p "$DATA/squidpy-datasets" && ln -sfn "$DATA/squidpy-datasets" work/data

status=0
for nb in spage-tutorial tangram-tutorial; do
    if NB_SRC="ports/docs/notebooks/squidpy-api/$nb.ipynb" NB_DST="$OUT/$nb.ipynb" "$PY" - <<'RUN'; then
import os, time, nbformat
from nbclient import NotebookClient

nb = nbformat.read(os.environ["NB_SRC"], as_version=4)
start = time.monotonic()
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "work"}}).execute()
finally:  # a failed notebook is written too, up to the failing cell; without stderr (download progress bars)
    for cell in nb.cells:
        cell["outputs"] = [o for o in cell.get("outputs", []) if o.get("name") != "stderr"]
    nbformat.write(nb, os.environ["NB_DST"])
print(f"### {os.environ['NB_SRC']}: {len(nb.cells)} cells in {time.monotonic() - start:.0f}s")
for cell in nb.cells:
    for out in cell.get("outputs", []):
        if out.output_type == "stream" or "text/plain" in out.get("data", {}):
            print("   ", (out.get("text") or out["data"]["text/plain"]).strip()[:800])
RUN
        echo "### $nb OK"
    else
        echo "### $nb FAILED" >&2
        status=1
    fi
done
exit "$status"
