#!/usr/bin/env bash
#SBATCH --job-name=stalign-d13
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
# Eight sequential replays, each one upstream fit plus one port fit, all on CPU. On an H100
# the pair costs ~130s for `starmap`; on CPU expect tens of minutes, and `merfish` asks for
# 2000 iterations against 800. Sized to finish rather than to be tight -- overrunning this
# means something hung, since nothing here is 12 hours of arithmetic.
#SBATCH --time=12:00:00
# Deliberately NOT an array. Every run has to share one host, one staged tree and one
# environment, because the effect being measured is smaller than what changing hosts can do
# to a rank-3 fit. A single allocation looping the conditions gives that by construction;
# `--array=0-7%1 --nodelist=...` only approximates it.
#SBATCH --nice=0
#SBATCH --output=stalign-d13-%j.out
#SBATCH --error=stalign-d13-%j.err

# Ledger row D13: both `allen3Datlas` notebooks pass a length-3 `muA`/`muB` against a
# single-channel section. Upstream broadcasts and sums over the broadcast axis
# (STalign.py:1554-1555), so its artifact and background widths are effectively
# `sigmaA/sqrt(3)` and `sigmaB/sqrt(3)` while `WM`'s is not; squidpy validates the length, so
# the replay collapses the mean and the two sides fit different mixtures. That is a candidate
# cause of the volume-to-section divergence *independent of D11*, and it has no number.
#
# This job produces the number. Two conditions:
#
#   baseline   flag off -- what every published sweep does today
#   collapsed  flag on  -- upstream is handed the collapsed mean too, so both sides fit the
#                          same mixture and whatever remains is D11 plus noise
#
# The difference between them is D13's contribution and nothing else.
#
# CPU on purpose, and it is the point rather than a concession. Upstream's
# `LDDMM_3D_to_slice` is bit-identical across processes on CPU and wanders on CUDA -- 245um
# of cell displacement and 15-22% of Allen regions between two runs of upstream *alone*,
# determinism pinned, same node. D11's first measurement drowned in exactly that: the region
# effect looked like 49.8% -> 18.1% until a second replicate came back at 42.2%. Running the
# reference where it stands still is route 2 of the open TODO in
# `.claude/stalign-divergence-notes.md`, and it is what turns this from indicative into
# measured -- including for the label and depth metrics, which no amount of GPU replicates
# would have rescued. It also means this job needs no GPU at all, so it queues on `cpu_p`.
#
# Two replicates per condition regardless. On CPU they are expected to be identical, so they
# are a check that the reference really did stand still, not a variance estimate -- and if
# they ever differ, `flag_report` says so in the verdict column instead of averaging over it.
# The port side was already bit-identical across replicates on GPU in the D11 run, and the
# launcher pins OMP_NUM_THREADS to the allocation, so it should be still here too -- but that
# is inference, and the replicates are what check it.
#
# TWO THINGS THIS DOES NOT BUY, both easy to misread off the output:
#
#   1. CPU does not shrink the port-vs-upstream gap. What is measured as reproducible is
#      upstream reproducing *itself* across reruns -- `A` and `v` to the byte between two
#      processes -- not upstream agreeing with squidpy. D11 and D13 are real algorithmic
#      differences and survive the move. Expect `baseline` to land near the published figures
#      (`v` relL2 ~0.93 on starmap), not near zero. What CPU buys is only that the number
#      stops wobbling, so the flag-off/flag-on difference is attributable to the flag.
#   2. `baseline` is not a rerun of `docs/parity/`. Moving *both* sides off the GPU changes
#      the arithmetic order on both, so these numbers will be close to the published ones but
#      not equal. The comparison that means anything is baseline-vs-collapsed *within* this
#      sweep; never one of its cells against the published GPU table.
#
# Usage:
#   sbatch .claude/run_stalign_d13_flag.sbatch [PORTS_SOURCE] [SQUIDPY_SOURCE] [OUTPUT]
# The table is rendered at the end; to re-render it later, from anywhere:
#   python3 src/squidpy_ports/stalign/flag_report.py OUTPUT --output OUTPUT/d13.md
set -euo pipefail

PORTS_SOURCE="${1:-${PORTS_SOURCE:-$SLURM_SUBMIT_DIR}}"
SQUIDPY_SOURCE="${2:-${SQUIDPY_SOURCE:-$(dirname "$PORTS_SOURCE")/squidpy}}"
OUTPUT="${3:-$PORTS_SOURCE/cluster-results/d13-${SLURM_JOB_ID:-manual}}"

# Both volume-to-section notebooks. `xenium-heimage` also fixes both means, but its section is
# a three-channel H&E, so its length-3 mean is correct and there is nothing here to change.
NOTEBOOKS=(
    "starmap-allen3Datlas-alignment.ipynb"
    "merfish-allen3Datlas-alignment.ipynb"
)
CONDITIONS=(baseline collapsed)
REPLICATES=(1 2)

# Everything below -- staging, the localscratch layout, the caches, the atlas, the in-process
# determinism pinning -- is the sweep's own launcher, reused rather than copied so the two
# cannot drift. Resolved from PORTS_SOURCE, not from this script's own path: Slurm runs a
# batch script from a copy in its spool directory, so the launcher is not beside it.
LAUNCHER="$PORTS_SOURCE/.claude/run_stalign_comparisons.sbatch"
[[ -f "$LAUNCHER" ]] || { echo "Missing launcher: $LAUNCHER" >&2; exit 2; }

# No CUDA wheel: see the CPU note above. The launcher reads this instead of hardcoding
# `jax[cuda12]`, and torch's CUDA build imports and runs CPU-only, so it needs no equivalent.
export JAX_INSTALL_SPEC=jax
# The determinism environment the launcher exports is CUDA-specific and inert here; the
# in-process half (`_pin_fit_determinism`) still applies and is what CPU needs.

TOTAL_RUNS=$(( ${#NOTEBOOKS[@]} * ${#CONDITIONS[@]} * ${#REPLICATES[@]} ))

mkdir -p "$OUTPUT"
echo "### d13 flag sweep: host=$(hostname) job=${SLURM_JOB_ID:-manual} output=$OUTPUT"
echo "### $TOTAL_RUNS runs, sequential, one host"

run_index=0
for condition in "${CONDITIONS[@]}"; do
    for replicate in "${REPLICATES[@]}"; do
        run_output="$OUTPUT/${condition}-rep${replicate}"
        mkdir -p "$run_output"
        for notebook in "${NOTEBOOKS[@]}"; do
            run_index=$(( run_index + 1 ))
            case "$condition" in
                baseline) export STALIGN_D13_COLLAPSE_MEANS=0 ;;
                collapsed) export STALIGN_D13_COLLAPSE_MEANS=1 ;;
                *) echo "Unknown condition: $condition" >&2; exit 2 ;;
            esac
            export STALIGN_NOTEBOOK="$notebook"
            # Distinguishes each run's working directory and its TASK_TAG-suffixed logs inside
            # the launcher, which otherwise derives both from the array task id.
            export SLURM_ARRAY_TASK_ID="$run_index"

            echo "### run $run_index/$TOTAL_RUNS: condition=$condition rep=$replicate notebook=$notebook"
            echo "### STALIGN_D13_COLLAPSE_MEANS=$STALIGN_D13_COLLAPSE_MEANS -> $run_output"
            # Not `exec`: this loop owns the allocation. A failed run must not take the rest of
            # the sweep with it -- a sweep missing one cell is still readable for the others,
            # and the launcher writes a status and traceback for whatever failed.
            if bash "$LAUNCHER" "$PORTS_SOURCE" "$SQUIDPY_SOURCE" "$run_output"; then
                echo "### run $run_index: ok"
            else
                echo "### run $run_index: FAILED (exit $?), continuing" >&2
            fi
        done
    done
done

echo "### all runs finished; rendering the table"
# Not from the venv the last run built -- the launcher removes its staged tree on the way out.
# Invoked by path rather than as `-m squidpy_ports.stalign.flag_report`: importing the package
# runs `importlib.metadata.version("squidpy-ports")`, which needs an install. `flag_report`
# imports nothing but the standard library, so as a plain script it needs neither, and the
# table lands even though the environment is gone.
python3 "$PORTS_SOURCE/src/squidpy_ports/stalign/flag_report.py" "$OUTPUT" \
        --output "$OUTPUT/d13-effect-vs-spread.md" \
    || echo "### could not render the table here; run flag_report on $OUTPUT" >&2

echo "### d13 flag sweep complete: $OUTPUT"
