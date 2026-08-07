#!/usr/bin/env bash
#SBATCH --job-name=stalign-build
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=build-%j.out
#SBATCH --error=build-%j.err
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin"

# Build the reproduction container. Everything churny (apptainer tmp, cache, the .sif itself)
# lives on node-local /localscratch; only the finished single .sif is copied to Lustre.
U="$(id -un)"
LS="/localscratch/$U/stalign-build/${SLURM_JOB_ID:-adhoc}"
export APPTAINER_TMPDIR="$LS/tmp"
export APPTAINER_CACHEDIR="$LS/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

PORTS="/lustre/groups/ml01/workspace/$U/stalign-rerun/squidpy-ports"
DURABLE="/lustre/groups/ml01/workspace/$U/stalign-rerun/stalign.sif"

echo "### host=$(hostname) job=${SLURM_JOB_ID:-adhoc} localscratch=$LS"
apptainer --version

echo "### capability probe: tiny --fakeroot build (fails fast if fakeroot is unavailable)"
cat > "$LS/tiny.def" <<'EOF'
Bootstrap: docker
From: ubuntu:22.04
%post
    apt-get update && apt-get install -y --no-install-recommends ca-certificates
    echo TINY_POST_OK
EOF
apptainer build --fakeroot "$LS/tiny.sif" "$LS/tiny.def"
echo "### tiny build OK -- fakeroot works"

echo "### full build from $PORTS"
cd "$PORTS"
apptainer build --fakeroot "$LS/stalign.sif" container/stalign.def
echo "### full build OK; copying single .sif to Lustre"
cp "$LS/stalign.sif" "$DURABLE"
ls -la "$DURABLE"
echo "### DONE: $DURABLE"
