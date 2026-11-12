#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the integration tests -- the ones ./scripts/test.sh deliberately excludes.
#
# Usage: ./scripts/test-integration.sh [--dry-run] [package] [test paths...] [extra pytest args...]
# Examples:
#   ./scripts/test-integration.sh -rs                  # everything: the pool's part, then the laptop's
#   ./scripts/test-integration.sh scheduled-jobs       # one package
#   ./scripts/test-integration.sh --dry-run -rs        # print both commands, run neither
#   ./scripts/test-integration.sh core packages/core/tests/integration/test_process_services.py -v
#
# Test paths come first, before any option, and lie under packages/ or tests/.
#
# `test.sh` runs `-m "not integration"` so the fast loop stays fast, which is
# correct for a fast loop and wrong for evidence. Cross-pod behaviour lives
# ENTIRELY in these tests: a publish/subscribe subject mismatch and a wedged
# listener both reached a commit on this repo while every unit test passed,
# because nothing in the recorded evidence had ever executed the tests that
# exercise two pods talking to each other.
#
# Two routes, while the suite moves to the testament pool one package at a time:
#
#   - the pool: the paths in .testament.toml's [suites.integration] `default_selection` run there,
#     through `./scripts/test.sh --suite integration`, against Postgres, NATS and S3 started as child
#     processes inside the pool's container. test.sh's rules hold: its exit 2 (no verdict) and its
#     refusal of flags that need one local process (-x, --pdb, ...);
#   - the laptop: LOCAL_DIRS, below, run here as they always have, with `-m integration`, against
#     testcontainers (so they need a working Docker). It holds every package not moved yet, and the
#     tests that drive Docker themselves or read a credential (the `docker` and `secrets` markers),
#     which the pool deselects. A suite path inside a local directory is left out of the local run.
#
# tests/enforcement/test_integration_suite_selection.py holds the two lists to every integration test
# in the tree, so a test is never in neither. A package moves by leaving LOCAL_DIRS for the suite.
#
# Where the pool route is not taken -- no `testament` on PATH, CI, or TESTAMENT=off with a stated
# TESTAMENT_OFF_REASON -- everything selected runs here, as before the pool.
LOCAL_DIRS=(
    packages/datasources/tests/integration/test_redshift_driver_live.py
    packages/datasources/tests/integration/test_transaction_redshift_live.py
    packages/nats/tests/integration/test_lame_duck_handover_live.py
    packages/scrape/tests/integration/test_operator_pipe_sidecar.py
    packages/search/tests/test_searxng_live_scoring.py
)

# `--dry-run`, anywhere among the arguments: print the commands each route would run, exactly, and run neither
DRY_RUN=""
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        ARGS+=("$arg")
    fi
done
set -- ${ARGS+"${ARGS[@]}"}
PACKAGE=""
if [ $# -gt 0 ] && [[ "$1" != -* ]] && [ -d "packages/$1" ]; then
    PACKAGE="$1"
    shift
fi
TEST_PATHS=()
while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
    case "$1" in
        packages/* | tests/*) ;;
        *)
            echo "test-integration.sh: '$1' is not a test path under packages/ or tests/ (test paths come before any option)" >&2
            exit 2
            ;;
    esac
    if [ ! -e "${1%%::*}" ]; then
        echo "test-integration.sh: '$1' does not exist" >&2
        exit 2
    fi
    TEST_PATHS+=("${1%/}")
    shift
done
EXTRA_ARGS=("$@")

# the integration suite's paths, from the manifest (the one list of what runs on the pool)
SUITE_PATHS=()
if [ -f .testament.toml ]; then
    if ! command -v python3 >/dev/null 2>&1 || ! python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        echo "test-integration.sh: reading .testament.toml needs python3 3.11 or later (tomllib) on PATH" >&2
        exit 2
    fi
    listed="$(python3 -I -c '
import sys, tomllib
with open(".testament.toml", "rb") as manifest:
    suites = tomllib.load(manifest).get("suites", {})
for path in suites.get("integration", {}).get("default_selection", []):
    if not path.startswith("-"):
        print(path.rstrip("/"))
')"
    while IFS= read -r path; do
        [ -n "$path" ] && SUITE_PATHS+=("$path")
    done <<<"$listed"
fi

# whether $1 is $2 or lies under it
under() {
    [ "$1" = "$2" ] || [[ "$1" == "$2"/* ]]
}
# whether $1 lies in what this run selects: the test paths given, else the package, else everything
selected() {
    local scope
    if [ ${#TEST_PATHS[@]} -gt 0 ]; then
        for scope in "${TEST_PATHS[@]}"; do
            under "$1" "${scope%%::*}" && return 0
        done
        return 1
    fi
    [ -z "$PACKAGE" ] || under "$1" "packages/${PACKAGE%/}"
}
in_suite() {
    local suite_path
    for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
        under "${1%%::*}" "$suite_path" && return 0
    done
    return 1
}

POOL=()
LOCAL=()
for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
    selected "$suite_path" && POOL+=("$suite_path")
done
if [ ${#TEST_PATHS[@]} -gt 0 ]; then
    # a path inside a suite path runs on the pool, alone; any other runs here
    for path in "${TEST_PATHS[@]}"; do
        if ! in_suite "$path"; then
            LOCAL+=("$path")
            continue
        fi
        listed=""
        for pool_path in ${POOL+"${POOL[@]}"}; do
            [ "$pool_path" = "$path" ] && listed=1
        done
        [ -n "$listed" ] || POOL+=("$path")
    done
else
    for local_dir in "${LOCAL_DIRS[@]}"; do
        selected "$local_dir" && LOCAL+=("$local_dir")
    done
    # a package with integration tests in neither list runs here whole, as before the pool
    if [ -n "$PACKAGE" ] && [ ${#POOL[@]} -eq 0 ] && [ ${#LOCAL[@]} -eq 0 ]; then
        LOCAL+=("packages/${PACKAGE%/}")
    fi
fi

pool_route=1
if ! command -v testament >/dev/null 2>&1 || [ -n "${CI:-}" ]; then
    pool_route=""
elif [ "${TESTAMENT:-}" = "off" ]; then
    if [ -z "${TESTAMENT_OFF_REASON:-}" ]; then
        echo "test-integration.sh: TESTAMENT=off needs a reason: drop it to run the suite's part on the pool, or say why with TESTAMENT_OFF_REASON=\"...\"" >&2
        exit 2
    fi
    echo "test-integration.sh: running locally, not via testament: ${TESTAMENT_OFF_REASON}" >&2
    pool_route=""
fi
if [ -z "$pool_route" ] && [ ${#POOL[@]} -gt 0 ]; then
    # Off the pool the suite's paths run here, less the modules marked `process_only` (they check what only
    # process mode has, a cluster of child processes, and testcontainers can't stand in for it): a module named
    # on its own is collected, and tests/require_services.py would deselect it, so it is not named at all. A
    # suite path inside a local directory is run by that directory, not named twice.
    PROCESS_ONLY=()
    listed="$(python3 -I scripts/integration_marks.py --process-only "${POOL[@]}")"
    while IFS= read -r file; do
        [ -n "$file" ] && PROCESS_ONLY+=("$file")
    done <<<"$listed"
    for pool_path in "${POOL[@]}"; do
        skip=""
        for path in ${LOCAL+"${LOCAL[@]}"}; do
            [ "$pool_path" != "$path" ] && under "$pool_path" "$path" && skip=1
        done
        for file in ${PROCESS_ONLY+"${PROCESS_ONLY[@]}"}; do
            [ "$pool_path" = "$file" ] && skip=1
        done
        [ -n "$skip" ] || LOCAL+=("$pool_path")
    done
    POOL=()
fi

# every test_*.py under the given paths that marks a test `integration` (scripts/integration_marks.py, which the
# selection enforcement reads too), or that this machine's python3 cannot parse
integration_files() {
    python3 -I scripts/integration_marks.py "$@"
}
# a local path holding no integration test once the suite's paths are left out (a package root whose only
# integration directory is the suite's) is not run: pytest would collect nothing and exit 5. A named file stays.
MARKED=()
if [ ${#LOCAL[@]} -gt 0 ]; then
    listed="$(integration_files "${LOCAL[@]}")"
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        if [ -n "$pool_route" ] && in_suite "$file"; then
            continue
        fi
        MARKED+=("$file")
    done <<<"$listed"
fi
KEPT=()
for path in ${LOCAL+"${LOCAL[@]}"}; do
    keep=""
    [ -f "${path%%::*}" ] && keep=1
    for file in ${MARKED+"${MARKED[@]}"}; do
        under "$file" "$path" && keep=1
    done
    [ -n "$keep" ] && KEPT+=("$path")
done
LOCAL=(${KEPT+"${KEPT[@]}"})
if [ ${#POOL[@]} -eq 0 ] && [ ${#LOCAL[@]} -eq 0 ]; then
    echo "test-integration.sh: no integration tests in what was selected" >&2
    exit 5
fi

# test.sh keeps a run local while an environment gate (RECORD, THREETEARS_LIVE_*, TAVILY_API_KEY) is set, since
# on the pool those tests would skip. The integration suite deselects `secrets` and ships no environment to
# its runners, so the gates mean nothing to it: the pool call runs without them. The local run keeps them.
GATES=()
while IFS= read -r name; do
    case "$name" in
        RECORD | TAVILY_API_KEY | THREETEARS_LIVE_*) GATES+=(-u "$name") ;;
    esac
done < <(compgen -e)

# the suite's own paths inside a local directory run on the pool, so the local run leaves them out
IGNORES=()
if [ -n "$pool_route" ]; then
    for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
        for path in ${LOCAL+"${LOCAL[@]}"}; do
            if [ "$suite_path" != "$path" ] && under "$suite_path" "$path"; then
                IGNORES+=("--ignore=$suite_path")
                break
            fi
        done
    done
fi

# the two commands, built once, so a dry run prints exactly what a run runs
POOL_CMD=()
if [ ${#POOL[@]} -gt 0 ]; then
    [ ${#GATES[@]} -eq 0 ] || POOL_CMD+=(env "${GATES[@]}")
    POOL_CMD+=(./scripts/test.sh --suite integration "${POOL[@]}" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})
fi
LOCAL_CMD=()
if [ ${#LOCAL[@]} -gt 0 ]; then
    LOCAL_CMD=(uv run pytest "${LOCAL[@]}" -m integration ${IGNORES+"${IGNORES[@]}"} ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})
fi

# a command as a shell would take it back: each word quoted where it needs to be
shell_words() {
    local words
    printf -v words '%q ' "$@"
    printf '%s\n' "${words% }"
}
if [ -n "$DRY_RUN" ]; then
    [ ${#POOL_CMD[@]} -eq 0 ] || echo "on the pool: $(shell_words "${POOL_CMD[@]}")"
    [ ${#LOCAL_CMD[@]} -eq 0 ] || echo "here: $(shell_words "${LOCAL_CMD[@]}")"
    exit 0
fi

pool_status=0
if [ ${#POOL_CMD[@]} -gt 0 ]; then
    "${POOL_CMD[@]}" || pool_status=$?
fi

local_status=0
if [ ${#LOCAL_CMD[@]} -gt 0 ]; then
    echo "test-integration.sh: running here, through testcontainers: ${LOCAL[*]}" >&2
    "${LOCAL_CMD[@]}" || local_status=$?
fi

if [ ${#POOL[@]} -gt 0 ] && [ ${#LOCAL[@]} -gt 0 ]; then
    echo "test-integration.sh: pool exit ${pool_status}, local exit ${local_status}" >&2
fi
if [ "$pool_status" -ne 0 ]; then
    exit "$pool_status"
fi
exit "$local_status"
