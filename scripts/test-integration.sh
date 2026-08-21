#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the integration tests -- the ones ./scripts/test.sh deliberately excludes.
#
# Usage: ./scripts/test-integration.sh [package] [extra pytest args...]
#
# `test.sh` runs `-m "not integration"` so the fast loop stays fast, which is
# correct for a fast loop and wrong for evidence. Cross-pod behaviour lives
# ENTIRELY in these tests: a publish/subscribe subject mismatch and a wedged
# listener both reached a commit on this repo while every unit test passed,
# because nothing in the recorded evidence had ever executed the tests that
# exercise two pods talking to each other.
#
# These spin real containers (NATS, Postgres) via testcontainers, so they are
# slower and need a working Docker. That is the reason they are a separate
# script rather than folded into `test.sh` -- not a reason to skip them before
# claiming a change works.

PACKAGE=""
EXTRA_ARGS=()

if [ $# -gt 0 ] && [[ "$1" != -* ]] && [ -d "packages/$1" ]; then
    PACKAGE="$1"
    shift
fi

EXTRA_ARGS=("$@")

if [ -n "$PACKAGE" ]; then
    uv run pytest "packages/$PACKAGE" -m integration ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
else
    uv run pytest packages/ tests/ -m integration ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi
