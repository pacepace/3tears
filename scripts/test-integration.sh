#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the integration tests -- the ones ./scripts/test.sh deliberately excludes.
#
# Usage: ./scripts/test-integration.sh [--dry-run] [package] [test paths...] [extra pytest args...]
#        ./scripts/test-integration.sh --here [--dry-run] [extra pytest args...]
# Examples:
#   ./scripts/test-integration.sh -rs                  # the integration suite, on the testament pool
#   ./scripts/test-integration.sh scheduled-jobs       # one package's part of it
#   ./scripts/test-integration.sh --dry-run -rs        # print the command, run nothing
#   ./scripts/test-integration.sh --here -rs           # the docker and secrets tests, on this machine
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
# The suite runs on the testament pool: the paths in .testament.toml's [suites.integration]
# `default_selection`, through `./scripts/test.sh --suite integration`, against Postgres, NATS and S3
# started as child processes inside the pool's container. test.sh's rules hold: its exit 2 (no verdict),
# and its refusal of flags that need one local process (-x, --pdb, ...). Nothing runs on this machine.
#
# `--here` runs the rest, on this machine: HERE_FILES, the tests marked `docker` (they drive a Docker
# daemon themselves, which the pool has none of) or `secrets` (they read a credential, which never ships
# to a runner). Only those marks run (`-m "docker or secrets"`). It is needed when a change touches what
# they cover: scrape's sidecar, search, the nats lame-duck handover in Docker, or the credentialed
# packages (datasources' Redshift, models' Claude CLI, search's Tavily); without the credential a
# `secrets` test skips, saying which. tests/enforcement/test_integration_suite_selection.py holds the
# suite and HERE_FILES to every integration test in the tree, and HERE_FILES to every docker and secrets
# file.
#
# When the pool route is not taken -- no `testament` on PATH, CI, or TESTAMENT=off with a stated
# TESTAMENT_OFF_REASON (the pool is down: docs/testing.md, "When the pool is down") -- the suite's paths
# run here instead, through testcontainers, less the modules marked `process_only` (they check the
# pool's child-process services, which testcontainers can't stand in for). That needs Docker and a
# machine with room: one package at a time on a laptop.
HERE_FILES=(
    packages/datasources/tests/integration/test_redshift_driver_live.py
    packages/datasources/tests/integration/test_redshift_fetch_within_live.py
    packages/datasources/tests/integration/test_transaction_redshift_live.py
    packages/models/tests/live/test_claude_cli_structured_output_live.py
    packages/nats/tests/integration/test_lame_duck_handover_live.py
    packages/scrape/tests/integration/test_operator_pipe_sidecar.py
    packages/search/tests/test_embedded_smoke.py
    packages/search/tests/test_searxng_live_scoring.py
)

# a command as a shell would take it back: each word quoted where it needs to be
shell_words() {
    local words
    printf -v words '%q ' "$@"
    printf '%s\n' "${words% }"
}

# `--dry-run` and `--here`, anywhere among the arguments
DRY_RUN=""
HERE=""
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --here) HERE=1 ;;
        *) ARGS+=("$arg") ;;
    esac
done
set -- ${ARGS+"${ARGS[@]}"}

if [ -n "$HERE" ]; then
    HERE_CMD=(uv run pytest "${HERE_FILES[@]}" -m "docker or secrets" "$@")
    if [ -n "$DRY_RUN" ]; then
        echo "here: $(shell_words "${HERE_CMD[@]}")"
        exit 0
    fi
    echo "test-integration.sh --here: the docker and secrets tests, on this machine" >&2
    exec "${HERE_CMD[@]}"
fi

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
if ! command -v python3 >/dev/null 2>&1 || ! python3 -I -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "test-integration.sh: reading .testament.toml needs python3 3.11 or later (tomllib) on PATH" >&2
    exit 2
fi
SUITE_PATHS=()
listed="$(python3 -I -c '
import tomllib
with open(".testament.toml", "rb") as manifest:
    suites = tomllib.load(manifest).get("suites", {})
for path in suites.get("integration", {}).get("default_selection", []):
    if not path.startswith("-"):
        print(path.rstrip("/"))
')"
while IFS= read -r path; do
    [ -n "$path" ] && SUITE_PATHS+=("$path")
done <<<"$listed"

# whether $1 is $2 or lies under it
under() {
    [ "$1" = "$2" ] || [[ "$1" == "$2"/* ]]
}
in_suite() {
    local suite_path
    for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
        under "${1%%::*}" "$suite_path" && return 0
    done
    return 1
}

# what runs: the suite's paths the package or the test paths select, and test paths inside the suite's
SELECTION=()
for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
    if [ ${#TEST_PATHS[@]} -gt 0 ]; then
        for path in "${TEST_PATHS[@]}"; do
            if [ "$suite_path" != "${path%%::*}" ] && under "$suite_path" "${path%%::*}"; then
                SELECTION+=("$suite_path")
            fi
        done
    elif [ -z "$PACKAGE" ] || under "$suite_path" "packages/${PACKAGE%/}"; then
        SELECTION+=("$suite_path")
    fi
done
for path in ${TEST_PATHS+"${TEST_PATHS[@]}"}; do
    if in_suite "$path"; then
        SELECTION+=("$path")
        continue
    fi
    covered=""
    for suite_path in ${SUITE_PATHS+"${SUITE_PATHS[@]}"}; do
        under "$suite_path" "${path%%::*}" && covered=1
    done
    if [ -z "$covered" ]; then
        echo "test-integration.sh: '$path' is not in the integration suite; the docker and secrets tests run with --here" >&2
        exit 2
    fi
done
if [ ${#SELECTION[@]} -eq 0 ]; then
    echo "test-integration.sh: nothing of the integration suite in what was selected; the docker and secrets tests run with --here" >&2
    exit 5
fi

pool_route=1
if ! command -v testament >/dev/null 2>&1 || [ -n "${CI:-}" ]; then
    pool_route=""
elif [ "${TESTAMENT:-}" = "off" ]; then
    if [ -z "${TESTAMENT_OFF_REASON:-}" ]; then
        echo "test-integration.sh: TESTAMENT=off needs a reason: drop it to run the suite on the pool, or say why with TESTAMENT_OFF_REASON=\"...\"" >&2
        exit 2
    fi
    echo "test-integration.sh: running locally, not via testament: ${TESTAMENT_OFF_REASON}" >&2
    pool_route=""
fi

if [ -n "$pool_route" ]; then
    # test.sh keeps a run local while an environment gate (RECORD, THREETEARS_LIVE_*, TAVILY_API_KEY) is set,
    # since on the pool those tests would skip. The integration suite deselects `secrets` and ships no
    # environment to its runners, so the gates mean nothing to it: the pool call runs without them.
    GATES=()
    while IFS= read -r name; do
        case "$name" in
            RECORD | TAVILY_API_KEY | THREETEARS_LIVE_*) GATES+=(-u "$name") ;;
        esac
    done < <(compgen -e)
    CMD=()
    [ ${#GATES[@]} -eq 0 ] || CMD+=(env "${GATES[@]}")
    CMD+=(./scripts/test.sh --suite integration "${SELECTION[@]}" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})
    LABEL="on the pool"
else
    # off the pool: the selection runs here, less the modules marked `process_only`, which check the pool's
    # child-process services (a module named on its own is collected, and tests/require_services.py would
    # deselect it, so it is not named at all)
    PROCESS_ONLY=()
    listed="$(python3 -I scripts/integration_marks.py --process-only "${SELECTION[@]}")"
    while IFS= read -r file; do
        [ -n "$file" ] && PROCESS_ONLY+=("$file")
    done <<<"$listed"
    LOCAL=()
    for path in "${SELECTION[@]}"; do
        skip=""
        for file in ${PROCESS_ONLY+"${PROCESS_ONLY[@]}"}; do
            [ "$path" = "$file" ] && skip=1
        done
        [ -n "$skip" ] || LOCAL+=("$path")
    done
    if [ ${#LOCAL[@]} -eq 0 ]; then
        echo "test-integration.sh: what was selected runs only on the pool's process-mode services" >&2
        exit 5
    fi
    CMD=(uv run pytest "${LOCAL[@]}" -m "integration and not docker and not secrets" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})
    LABEL="here"
fi

if [ -n "$DRY_RUN" ]; then
    echo "${LABEL}: $(shell_words "${CMD[@]}")"
    exit 0
fi
if [ -z "$pool_route" ]; then
    echo "test-integration.sh: the integration suite, here, through testcontainers: ${LOCAL[*]}" >&2
fi
exec "${CMD[@]}"
