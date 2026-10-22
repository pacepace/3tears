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

PACKAGE=""
EXTRA_ARGS=()

# If first arg exists and is a valid package name (not a flag), use it as package
if [ $# -gt 0 ] && [[ "$1" != -* ]] && [ -d "packages/$1" ]; then
    PACKAGE="$1"
    shift
fi

EXTRA_ARGS=("$@")

# Where testament is installed (see .testament.toml), it runs the tests on its runners instead,
# outside CI and unless TESTAMENT=off. It distributes the tests itself, so no `-n 4`, and its
# manifest command carries `-m "not integration"`, so a package run there skips integration tests.
#
# Two kinds of run stay local, because the pool would quietly change them. A run gated by an
# environment variable (RECORD, THREETEARS_LIVE_*, THREETEARS_EVALS_PAID*) would reach runners that
# do not have it, so its tests would skip -- and a skip reads like a pass. And a run that selects
# with `-m` would meet the manifest's own `-m "not integration"`, where pytest keeps only one.
local_only=""
if env | grep -E '^(RECORD|THREETEARS_LIVE_[A-Z_]*|THREETEARS_EVALS_PAID[A-Z_]*)=' >/dev/null; then
    local_only="an environment gate is set"
fi
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    case "$arg" in
        -m | -m*) local_only="-m selects the tests" ;;
    esac
done
if command -v testament >/dev/null 2>&1 && [ -n "$local_only" ]; then
    echo "test.sh: running locally, not via testament: ${local_only}" >&2
fi
if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" != "off" ] && [ -z "$local_only" ]; then
    echo "test.sh: via testament (integration tests run only locally; TESTAMENT=off for the old path)" >&2
    if [ -n "$PACKAGE" ]; then
        exec testament run -- "packages/$PACKAGE/tests/" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
    fi
    exec testament run -- ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi

if [ -n "$PACKAGE" ]; then
    uv run pytest "packages/$PACKAGE/tests/" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
else
    # Mirror the CI workflow's invocation (`packages/ tests/` with
    # ``-m "not integration"``). The agent-namespace move
    # (agent-memory -> agent/memory, etc.) means hard-coded per-
    # package paths drift; pointing at ``packages/`` lets pytest's
    # rootdir + collection rules pick up every package's tests.
    #
    # four workers, one package per worker (the root conftest marks each test with its package's
    # xdist group). a single-package run above stays serial. pass `-n 0` to run serially, e.g.
    # under a debugger.
    uv run pytest packages/ tests/ -m "not integration" -n 4 --dist loadgroup ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi
