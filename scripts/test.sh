#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run tests for a specific package or all packages
# Usage: ./scripts/test.sh [package] [test paths or node ids...] [extra pytest args...]
# Examples:
#   ./scripts/test.sh              # all packages
#   ./scripts/test.sh core         # just core
#   ./scripts/test.sh core -v      # core, verbose
#   ./scripts/test.sh agent/memory # one agent-family package; `agent` runs the whole family
#   ./scripts/test.sh -v           # all packages, verbose
#   ./scripts/test.sh evals packages/evals/tests/test_x.py       # that file only, not all of evals
#   ./scripts/test.sh evals packages/evals/tests/test_x.py::test_y -q   # one test
#
# Test paths (existing files or directories, or `file.py::name` node ids) run instead of the
# package's whole tests/, on the pool or locally alike. A package name with only flags runs the
# whole package.
#
# Runs go to the testament pool where testament is installed (see below). To run locally anyway,
# say why: TESTAMENT=off TESTAMENT_OFF_REASON="debugging under pdb" ./scripts/test.sh core

PACKAGE=""
EXTRA_ARGS=()

# If first arg exists and is a valid package name (not a flag), use it as package
if [ $# -gt 0 ] && [[ "$1" != -* ]] && [ -d "packages/$1" ]; then
    PACKAGE="$1"
    shift
fi

EXTRA_ARGS=("$@")

# the arguments that name tests: existing paths, or `file.py::name` node ids whose file exists.
# The value of an option that takes one (`-c pyproject.toml`, `--ignore dir`) is not a test path.
TEST_PATHS=()
takes_value=""
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    if [ -n "$takes_value" ]; then
        takes_value=""
        continue
    fi
    case "$arg" in
        -k | -m | -p | -c | -o | -W | -n | --ignore | --ignore-glob | --deselect | --rootdir | --confcutdir | \
            --basetemp | --junit-xml | --junitxml | --override-ini | --maxfail | --dist | --tb | --durations | \
            --log-level | --import-mode)
            takes_value=1
            continue
            ;;
        -*) continue ;;
    esac
    if [ -e "$arg" ] || { [[ "$arg" == *::* ]] && [ -f "${arg%%::*}" ]; }; then
        TEST_PATHS+=("$arg")
    fi
done

# a package's tests/; a directory of packages with none of its own (agent) runs all of it
TEST_PATH="packages/$PACKAGE/tests/"
if [ -n "$PACKAGE" ] && [ ! -d "packages/$PACKAGE/tests" ]; then
    TEST_PATH="packages/$PACKAGE/"
fi

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
#
# Anything else goes to the pool. TESTAMENT=off takes it local only with a stated reason
# (TESTAMENT_OFF_REASON), because an unexplained local run is usually a slow run that need not be:
# the usual cause was a single-file run that went to the pool as the whole package.
if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ -z "$local_only" ] && [ "${TESTAMENT:-}" = "off" ]; then
    if [ -z "${TESTAMENT_OFF_REASON:-}" ]; then
        echo "test.sh: TESTAMENT=off needs a reason: drop it to run on the pool (\`./scripts/test.sh <package> <test file>\` runs just that file there), or say why with TESTAMENT_OFF_REASON=\"...\"" >&2
        exit 2
    fi
    echo "test.sh: running locally, not via testament: ${TESTAMENT_OFF_REASON}" >&2
fi
if command -v testament >/dev/null 2>&1 && [ -n "$local_only" ]; then
    echo "test.sh: running locally, not via testament: ${local_only}" >&2
fi
if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" != "off" ] && [ -z "$local_only" ]; then
    echo "test.sh: via testament (integration tests run only locally; TESTAMENT=off with TESTAMENT_OFF_REASON runs it here)" >&2
    if [ -n "$PACKAGE" ] && [ ${#TEST_PATHS[@]} -eq 0 ]; then
        exec testament run -- "$TEST_PATH" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
    fi
    exec testament run -- ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi

if [ ${#TEST_PATHS[@]} -gt 0 ]; then
    # the caller named its tests: run exactly those, serially, as a single-package run does
    uv run pytest ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
elif [ -n "$PACKAGE" ]; then
    uv run pytest "$TEST_PATH" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
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
    if [ -n "$local_only" ]; then
        # the caller selects with -m or sets an environment gate: run exactly what was asked
        uv run pytest packages/ tests/ -m "not integration" -n 4 --dist loadgroup ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
    else
        # the event loop's wall-clock tests run apart and serially, as scripts/ci-unit-tests.sh runs
        # them: beside four busy workers the kernel preempts the thread holding the interpreter lock,
        # and a bound on the code becomes a bound on the machine. a --junit-xml report path gets a
        # second file for the serial run, so the parallel run's report is not overwritten.
        TIMING_ARGS=()
        for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
            case "$arg" in
                --junit-xml=*) TIMING_ARGS+=("${arg%.xml}.loop_timing.xml") ;;
                *) TIMING_ARGS+=("$arg") ;;
            esac
        done
        status=0
        uv run pytest packages/ tests/ -m "not integration and not loop_timing" -n 4 --dist loadgroup \
            ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} || status=$?
        timing=0
        uv run pytest packages/ tests/ -m "loop_timing and not integration" -n 0 \
            ${TIMING_ARGS+"${TIMING_ARGS[@]}"} || timing=$?
        # 5 = nothing collected
        [ "$timing" -eq 5 ] && timing=0
        if [ "$status" -ne 0 ] || [ "$timing" -ne 0 ]; then
            echo "test.sh: parallel run exit ${status}, serial loop_timing run exit ${timing}" >&2
            exit 1
        fi
    fi
fi
