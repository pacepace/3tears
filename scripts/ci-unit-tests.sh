#!/usr/bin/env bash
# Run the unit + enforcement suite the way CI does: several pytest sessions, not one.
#
#   ./scripts/ci-unit-tests.sh [extra pytest args...]                  e.g. -v
#   ./scripts/ci-unit-tests.sh --shard <name> [extra pytest args...]   one piece, e.g. --shard evals-2 -v
#   ./scripts/ci-unit-tests.sh --loop-timing [extra pytest args...]    only the serial loop_timing session
#
# --loop-timing is not a shard: on the pool route the testament pool runs every other test, and the
# `unit (testament pool)` job then runs this on its own runner (scripts/ci-unit-pool.sh, `-m`).
#
# Without --shard every session below runs, one after another (local use). CI runs the shards as
# parallel jobs instead, so the gate takes as long as its slowest shard rather than their sum:
#   evals-1 .. evals-8   packages/evals, split eight ways by pytest-split (--splits 8 --group N)
#   core                 packages/core and packages/datasources
#   tests                the repo-level tests/
#   rest                 packages/agent, every remaining package, and the serial loop_timing run
# Together the eleven run exactly what the no-selector run does: every test once. pytest-split
# hands each group a disjoint slice of the same collection, so the eight evals groups partition it.
#
# HOW EVALS SPLITS. The root conftest puts a package's tests in one xdist_group, so an evals shard
# runs on ONE worker, serially: its wall clock is the sum of its tests. pytest-split cuts the
# collection, in its own order, into eight runs of about equal recorded duration
# (duration_based_chunks, from .test_durations; ./scripts/update-test-durations.sh refreshes it).
# Contiguous runs, not least_duration's interleaving: a file keeps its tests together, so a costly
# module fixture (test_writer_calibration_dry.py's takes about three minutes) is paid once, not once
# a shard. Without durations it split by count, and alphabetical order put the slow writer and
# reading files in one shard: 56 minutes against 22.
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
cd "$REPO_ROOT" || exit 2

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

SHARDS="evals-1 evals-2 evals-3 evals-4 evals-5 evals-6 evals-7 evals-8 core tests rest"
shard=""
loop_timing_only=""
if [ "${1:-}" = "--loop-timing" ]; then
    loop_timing_only=1
    shift
elif [ "${1:-}" = "--shard" ]; then
    shard="${2:-}"
    case " $SHARDS " in
        *" $shard "*) ;;
        *) echo "unknown shard '${shard}'; one of: ${SHARDS}" >&2; exit 2 ;;
    esac
    shift 2
fi

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

# a session is "<directories>" or "<directories>|<extra pytest args>"; the evals shards add pytest-split's selector
case "$shard" in
    "") SESSIONS=("${SOLO[@]}" "${remaining[*]}" tests) ;;
    evals-*)
        SESSIONS=("packages/evals|--splits 8 --group ${shard#evals-} --splitting-algorithm duration_based_chunks --durations-path .test_durations")
        ;;
    core) SESSIONS=("packages/core packages/datasources") ;;
    tests) SESSIONS=(tests) ;;
    rest) SESSIONS=(packages/agent "${remaining[*]}") ;;
esac

# --loop-timing runs none of the parallel sessions, only the serial run below
[ -n "$loop_timing_only" ] && SESSIONS=()

failed=()
for session in ${SESSIONS+"${SESSIONS[@]}"}; do
    dirs="${session%%|*}"
    split=""
    [ "$dirs" != "$session" ] && split="${session#*|}"
    echo "::group::pytest ${session}"
    # shellcheck disable=SC2086 # dirs is a space-separated list of directories, split a list of arguments
    uv run pytest $dirs -m "not integration and not loop_timing" -n 4 --dist loadgroup $split "$@"
    status=$?
    echo "::endgroup::"
    # 5 = nothing collected, which a session of only integration tests legitimately is
    if [ "$status" -ne 0 ] && [ "$status" -ne 5 ]; then
        failed+=("${session} (exit ${status})")
    fi
done

# the event-loop wall-clock tests, apart and serially: beside busy workers the kernel preempts the thread holding the
# interpreter lock, and a bound on the code becomes a bound on the runner (release run 41231693608: 34.9 ms against
# 25.4). 5 = nothing collected. Part of the `rest` shard, of the full run, and all of --loop-timing.
if [ -z "$shard" ] || [ "$shard" = "rest" ]; then
    echo "::group::pytest loop_timing (serial)"
    uv run pytest packages tests -m "loop_timing and not integration" -n 0 "$@"
    status=$?
    echo "::endgroup::"
    if [ "$status" -ne 0 ] && [ "$status" -ne 5 ]; then
        failed+=("loop_timing (exit ${status})")
    fi
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED sessions:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
