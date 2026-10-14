#!/usr/bin/env bash
# Container side of the remote test lane: the shim runs this as
#   docker exec -i threetears-test bash /testruns/<name>/scripts/remote-test.sh <test.sh args>
# inside the tree scripts/test-remote.sh just rsynced. The dispatcher ships this
# file into that tree on every run (its header says why), so the dispatcher's
# copy runs, not the branch's. It syncs the venv and hands off to the tree's
# ./scripts/test.sh, whose excludes (-m "not integration", the sidecar --ignore)
# apply unchanged; the exit status is pytest's.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "remote-test: ran on $(hostname) as $(id -un), $(nproc) cores"

# The cache sits beside the worktrees on the bind mount, so it survives the
# container and the second run of any worktree is a near no-op sync.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/testruns/.uv-cache}" UV_LINK_MODE=copy
# Never dispatch again from inside the lane.
unset THREETEARS_TEST_REMOTE THREETEARS_TEST_REMOTE_DRY_RUN

# --extra vision mirrors CI's install, so the suite collects the same tests.
# --frozen installs the lock as written and never re-resolves; a uv too old for
# the lock's format exits non-zero with its own error (--quiet keeps errors), and
# set -e stops the run there. The timing line shows a warm sync is a no-op.
started=$SECONDS
uv sync --frozen --extra vision --quiet
echo "remote-test: uv sync ($(uv --version)) took $((SECONDS - started)) s"
exec ./scripts/test.sh "$@"
