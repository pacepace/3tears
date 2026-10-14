#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run tests for a specific package or all packages
# Usage: ./scripts/test.sh [package] [extra pytest args...]
# Examples:
#   ./scripts/test.sh              # all packages
#   ./scripts/test.sh core         # just core
#   ./scripts/test.sh core -v      # core, verbose
#   ./scripts/test.sh -v           # all packages, verbose
#
# THREETEARS_TEST_REMOTE=<host> runs the same invocation on a remote test host
# instead (docs/testing.md, "The remote test lane"). Unset, nothing changes. Set,
# one 5 s probe decides; a host that fails it falls back to local, out loud.
# THREETEARS_TEST_WHERE_FILE, when set, receives where the run went (check-all.sh
# names it in its summary).
ran() {
    if [ -n "${THREETEARS_TEST_WHERE_FILE:-}" ]; then printf '%s\n' "$1" >"$THREETEARS_TEST_WHERE_FILE"; fi
}
if [ -n "${THREETEARS_TEST_REMOTE:-}" ]; then
    lane="$THREETEARS_TEST_REMOTE"
    probe=0
    "$REPO_ROOT/scripts/test-remote.sh" --probe "$lane" "$@" || probe=$?
    if [ "$probe" = 0 ]; then
        ran "on $lane (remote)"
        rc=0
        "$REPO_ROOT/scripts/test-remote.sh" "$lane" "$@" || rc=$?
        echo "tests ran on $lane (remote)"
        exit "$rc"
    elif [ "$probe" = 75 ]; then
        # Another dispatch of this tree held the lane past THREETEARS_LANE_WAIT_S. A
        # local run would quietly take the whole suite onto this box; stop instead.
        echo "remote lane $lane is still busy with this tree: not running" >&2
        exit 75
    elif [ "$probe" = 64 ]; then
        echo "remote lane $lane cannot carry these arguments: running locally" >&2
    else
        echo "remote lane $lane unreachable: running locally" >&2
    fi
fi

PACKAGE=""
EXTRA_ARGS=()

# If first arg exists and is a valid package name (not a flag), use it as package
if [ $# -gt 0 ] && [[ "$1" != -* ]] && [ -d "packages/$1" ]; then
    PACKAGE="$1"
    shift
fi

EXTRA_ARGS=("$@")
ran "locally"

if [ -n "$PACKAGE" ]; then
    uv run pytest "packages/$PACKAGE/tests/" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
else
    # Mirror the CI workflow's invocation (`packages/ tests/` with
    # ``-m "not integration"``). The agent-namespace move
    # (agent-memory -> agent/memory, etc.) means hard-coded per-
    # package paths drift; pointing at ``packages/`` lets pytest's
    # rootdir + collection rules pick up every package's tests.
    uv run pytest packages/ tests/ -m "not integration" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi
