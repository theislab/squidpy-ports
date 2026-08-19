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

# Both take an override, so a rebuild does not have to happen in one hardcoded workspace.
#   sbatch .claude/build_container.sbatch <ports-checkout> <output.sif>
PORTS="${1:-/lustre/groups/ml01/workspace/$U/stalign-rerun/squidpy-ports}"
DURABLE="${2:-/lustre/groups/ml01/workspace/$U/stalign-rerun/stalign.sif}"

echo "### host=$(hostname) job=${SLURM_JOB_ID:-adhoc} localscratch=$LS"
apptainer --version

# Plain `--fakeroot`. This account is not in /etc/subuid, so apptainer uses a root-mapped
# namespace and runs %post under its *bundled* faked, which is linked against GLIBC_2.38. That
# is why the base is ubuntu24.04 (glibc 2.39) and not 22.04 (2.35), where every %post died with
# "version `GLIBC_2.38' not found".
#
# `--ignore-fakeroot-command` is an alternative way past the glibc problem -- it runs %post
# directly in the user namespace -- but it is not needed on a 24.04 base and is left off.
#
# If a build dies at "unable to copy files from host to container fs", read the untruncated
# error before touching the fakeroot flags: that message is also what a *missing source file*
# produces, and the real cause is spelled out several hundred characters in, as
# `cp: cannot stat '<path>': No such file or directory`. A staged checkout that omits `docs/`
# fails exactly this way, and no fakeroot setting will fix it.
BUILD_FLAGS=(--fakeroot)

echo "### capability probe: tiny --fakeroot build (fails fast if fakeroot is unavailable)"
cat > "$LS/tiny.def" <<'EOF'
Bootstrap: docker
From: ubuntu:24.04
%post
    apt-get update && apt-get install -y --no-install-recommends ca-certificates
    echo TINY_POST_OK
EOF
apptainer build "${BUILD_FLAGS[@]}" "$LS/tiny.sif" "$LS/tiny.def"
echo "### tiny build OK -- fakeroot works"

echo "### full build from $PORTS"
cd "$PORTS"
apptainer build "${BUILD_FLAGS[@]}" "$LS/stalign.sif" container/stalign.def
echo "### full build OK; copying single .sif to Lustre"
cp "$LS/stalign.sif" "$DURABLE"
ls -la "$DURABLE"
echo "### DONE: $DURABLE"
