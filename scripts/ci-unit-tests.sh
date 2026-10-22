#!/usr/bin/env bash
# Run the unit + enforcement suite the way CI does: several pytest sessions, not one.
#
#   ./scripts/ci-unit-tests.sh [extra pytest args...]     e.g. -v
#
# WHY SESSIONS. An xdist worker lives for its whole session and its resident memory only grows:
# glibc keeps what a test freed for reuse instead of returning it, so a worker's RSS is the peak of
# everything it has run so far, not what it holds now (tracemalloc shows a few MB of Python objects
# retained across test_campaign_page.py while RSS climbs by hundreds). One session over all 41k
# tests grew four workers past the runner's 16 GB: from 2026-10-21 the `check` job was killed
# around 75% with no pytest summary and no failing test, on any branch. Reproduced in a 4-CPU Linux
# container: workers OOM-killed ("node down: Not properly terminated"), then the run hung.
#
# Each session below starts fresh workers, so no worker carries more than its share of the suite.
# MALLOC_ARENA_MAX=2 caps glibc's per-thread arenas, which on its own cut the climb by about a
# third (7.6 GB against 10.5 GB at the same point) but did not stop it.
#
# Every test still runs exactly once: the sessions partition packages/ and tests/, and a package
# added later lands in the "remaining packages" session without editing this file.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# the heaviest packages each get a session of their own; the rest share one.
SOLO=(packages/evals "packages/core packages/datasources" packages/agent)

remaining=()
for dir in packages/*/; do
    dir="${dir%/}"
    case " ${SOLO[*]} " in
        *" $dir "*) ;;
        *) remaining+=("$dir") ;;
    esac
done

SESSIONS=("${SOLO[@]}" "${remaining[*]}" tests)

failed=()
for session in "${SESSIONS[@]}"; do
    # shellcheck disable=SC2086 -- a session is a space-separated list of directories
    echo "::group::pytest ${session}"
    uv run pytest $session -m "not integration" -n 4 --dist loadgroup "$@"
    status=$?
    echo "::endgroup::"
    # 5 = nothing collected, which a session of only integration tests legitimately is
    if [ "$status" -ne 0 ] && [ "$status" -ne 5 ]; then
        failed+=("${session} (exit ${status})")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED sessions:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
