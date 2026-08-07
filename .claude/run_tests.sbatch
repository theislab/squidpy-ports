#!/usr/bin/env bash
#SBATCH --job-name=stalign-tests
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100_80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=01:00:00
#SBATCH --output=tests-%j.out
#SBATCH --error=tests-%j.err
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin"  # submit with --export=NIL

# Run the squidpy fork's STalign test suites against the same pinned environment the comparison
# figures came from, so the pass/fail table on the docs page and the panels beside it describe one
# environment rather than two. The image has no pytest -- it is a runtime image, not a dev one --
# so it goes into the writable overlay instead of into a 9.8 GB rebuild.
#
#   sbatch --export=NIL .claude/run_tests.sbatch <stalign.sif> <squidpy-checkout> <output-dir>
SIF="${1:?path to a built stalign.sif}"
SQUIDPY="${2:?path to the pinned squidpy fork checkout}"
OUTPUT_DIR="${3:?durable output dir}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not on PATH (try: module load apptainer)" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

apptainer exec --nv --writable-tmpfs \
    --bind "$SQUIDPY:/squidpy" --bind "$OUTPUT_DIR:/output" \
    "$SIF" bash -eux -c '
        export MPLCONFIGDIR=/tmp/mpl HOME=/tmp UV_CACHE_DIR=/tmp/uv
        # The venv is uv-created, so it has no pip. uv writes pytest straight into it, and
        # --writable-tmpfs keeps that in the overlay rather than mutating the image.
        uv pip install --python /opt/squidpy-ports/.venv/bin/python --quiet pytest
        cd /squidpy
        # -m "reference or not reference" overrides squidpy pyproject addopts, which carry
        # -m "not reference" and would otherwise silently DESELECT the entire reference suite --
        # the layer that actually asserts the port against upstream. A run that reports "all
        # passed" while quietly skipping 62 tests is the exact failure this page exists to catch.
        # -p no:cacheprovider: do not scribble .pytest_cache into the bound checkout.
        python -m pytest -p no:cacheprovider -v -rA \
            -m "reference or not reference" \
            --junitxml=/output/stalign-tests.xml \
            tests/experimental/methods/test_stalign_reference.py \
            tests/experimental/tl/test_align.py \
            2>&1 | tee /output/stalign-tests.log
    '
echo "### DONE: $OUTPUT_DIR"
