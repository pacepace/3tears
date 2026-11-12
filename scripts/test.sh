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
#   ./scripts/test.sh --not-area evals    # everything but evals, on the pool (areas: .testament.toml)
#
# Test paths (`.py` files, `file.py::name` node ids or directories, under packages/ or tests/) run
# instead of the package's whole tests/, on the pool or locally alike. A package name with only
# flags runs the whole package.
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

# `--area NAME` and `--not-area NAME` (repeatable; see .testament.toml's [suites.default.areas]) are testament's,
# not pytest's: they go to `testament run` ahead of the pytest selection. Only the pool knows the
# areas, so a run that would go local refuses them below rather than running more than was asked.
AREA_ARGS=()
PYTEST_ARGS=()
area_flag=""
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    if [ -n "$area_flag" ]; then
        AREA_ARGS+=("$area_flag" "$arg")
        area_flag=""
        continue
    fi
    case "$arg" in
        --area | --not-area) area_flag="$arg" ;;
        --area=* | --not-area=*) AREA_ARGS+=("$arg") ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done
if [ -n "$area_flag" ]; then
    echo "test.sh: ${area_flag} needs an area name (see [suites.default.areas] in .testament.toml)" >&2
    exit 2
fi
EXTRA_ARGS=(${PYTEST_ARGS+"${PYTEST_ARGS[@]}"})

# the arguments that name tests. One counts only when all three hold, so an option's value is
# not mistaken for a test path and the run widened past the package's tests/ (`--log-file
# out.log`, `--cov packages/core/src`):
#   - it is not an option, nor the value of an option below written apart from it (`-c x`, not
#     `-c=x`). The list need not be complete: an unlisted option's value still has to pass both
#     rules after it, and a log, report or config file does not;
#   - it lies under packages/ or tests/ (relative to the repo root, `./` or absolute);
#   - it is a `.py` file, a `file.py::name` node id whose file exists, or a directory.
is_test_path() {
    local path="${1%%::*}"
    path="${path#"$REPO_ROOT"/}"
    path="${path#./}"
    case "$path" in
        packages/* | tests | tests/*) ;;
        *) return 1 ;;
    esac
    if [[ "$1" == *::* ]] || [[ "$path" == *.py ]]; then
        [[ "$path" == *.py ]] && [ -f "$path" ]
        return
    fi
    [ -d "$path" ]
}
TEST_PATHS=()
takes_value=""
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    if [ -n "$takes_value" ]; then
        takes_value=""
        # `--cov -q`: --cov's value is optional, and an option is never one
        [[ "$arg" == -* ]] || continue
    fi
    case "$arg" in
        -k | -m | -p | -c | -o | -W | -n | --ignore | --ignore-glob | --deselect | --rootdir | --confcutdir | \
            --basetemp | --junit-xml | --junitxml | --junit-prefix | --override-ini | --maxfail | --dist | \
            --tb | --durations | --import-mode | --capture | --color | --timeout | --doctest-glob | \
            --log-level | --log-format | --log-date-format | --log-file | --log-file-level | \
            --log-file-format | --log-cli-level | --log-cli-format | --report-log | --html | --css | \
            --cov | --cov-config | --cov-report | --cov-fail-under)
            takes_value=1
            continue
            ;;
        -*) continue ;;
    esac
    if is_test_path "$arg"; then
        TEST_PATHS+=("$arg")
    fi
done

# a package's tests/; a directory of packages with none of its own (agent) runs all of it
TEST_PATH="packages/$PACKAGE/tests/"
if [ -n "$PACKAGE" ] && [ ! -d "packages/$PACKAGE/tests" ]; then
    TEST_PATH="packages/$PACKAGE/"
fi

# Where testament is installed (see .testament.toml), it runs the tests on its runners instead,
# outside CI and unless TESTAMENT=off. It distributes the tests itself, so no `-n 4`. The run names
# the manifest's unit suite, `--suite default`, so it stays that suite whatever other suites the
# manifest declares, and an `--area` resolves against it. That suite's command carries
# `-m "not integration and not loop_timing and not docker and not secrets"`, so a run there skips
# those. A `-m` given here narrows that expression (testament sends `-m "(<manifest's>) and (<yours>)"`).
#
# A run gated by an environment variable (RECORD, THREETEARS_LIVE_*) stays local, because it
# would reach runners that do not have it, so its tests would skip -- and a skip reads like a pass.
local_only=""
if env | grep -E '^(RECORD|THREETEARS_LIVE_[A-Z_]*)=' >/dev/null; then
    local_only="an environment gate is set"
fi
# a local run's own -m replaces the whole-suite run's `-m "not integration"` (pytest keeps the last)
user_marks=""
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    case "$arg" in
        -m | -m*) user_marks=1 ;;
    esac
done

# Flags that keep a run on this machine: they need one local pytest process (stop at the first
# failure, rerun the last failures, step through, xdist) or this terminal (`-s`, `--pdb`,
# `--trace`). The pool can't honour them: the manifest's `fallback = "pool-only"` refuses the first
# kind, and testament runs the second here whatever the manifest says. A run that names only test
# files (no package-wide or directory selection) runs locally with them; any wider run is refused,
# since that would load this machine with a whole package.
#
# A short-option bundle (`-vxs`, `-xrs`) is read letter by letter up to the first option that takes
# a value (`-r`, `-k`, ...), whose value is the rest of the bundle: `-rx` reports xfails, it is no -x.
bundle_keeps_local() {
    local rest="${1#-}" c
    while [ -n "$rest" ]; do
        c="${rest:0:1}"
        case "$c" in
            x | s | n) return 0 ;;
            v | q | l) ;;
            *) return 1 ;;
        esac
        rest="${rest:1}"
    done
    return 1
}
needs_local_process=""
capture_value=""
for arg in ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}; do
    if [ -n "$capture_value" ]; then
        capture_value=""
        [ "$arg" = "no" ] && needs_local_process="--capture no"
        continue
    fi
    case "$arg" in
        -x | --exitfirst | --maxfail | --maxfail=* | --lf | --last-failed | --ff | --failed-first | \
            --sw | --stepwise | --sw-skip | --stepwise-skip | -n | -n* | --numprocesses | --numprocesses=* | \
            -s | --capture=no | --pdb | --trace)
            needs_local_process="$arg"
            ;;
        --capture) capture_value=1 ;;
        -[!-]*)
            bundle_keeps_local "$arg" && needs_local_process="$arg"
            ;;
    esac
done
only_files=""
if [ ${#TEST_PATHS[@]} -gt 0 ]; then
    only_files=1
    for path in "${TEST_PATHS[@]}"; do
        [[ "${path%%::*}" == *.py ]] || only_files=""
    done
fi
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
if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" != "off" ] && [ -z "$local_only" ] &&
    [ -n "$needs_local_process" ]; then
    if [ -z "$only_files" ]; then
        echo "test.sh: ${needs_local_process} needs one local pytest process or this terminal, so the pool can't take this run, and it is wider than named test files: name the files (\`./scripts/test.sh <package> <test file> ${needs_local_process}\`), drop the flag to run it on the pool, or run it here on purpose with TESTAMENT=off TESTAMENT_OFF_REASON=\"...\"" >&2
        exit 2
    fi
    if [ ${#AREA_ARGS[@]} -gt 0 ]; then
        echo "test.sh: --area and --not-area select on the testament pool, and ${needs_local_process} keeps this run local; name test files instead" >&2
        exit 2
    fi
    echo "test.sh: running locally, not via testament: ${needs_local_process} needs one local pytest process or this terminal, and the run names only test files" >&2
    exec uv run pytest ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"}
fi
if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" != "off" ] && [ -z "$local_only" ]; then
    echo "test.sh: via testament, pool only (integration tests run locally with test-integration.sh, loop_timing tests in CI; if the pool is down: docs/testing.md, \"When the pool is down\")" >&2
    pool_status=0
    if [ -n "$PACKAGE" ] && [ ${#TEST_PATHS[@]} -eq 0 ]; then
        testament run --suite default ${AREA_ARGS+"${AREA_ARGS[@]}"} -- "$TEST_PATH" ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} ||
            pool_status=$?
    else
        testament run --suite default ${AREA_ARGS+"${AREA_ARGS[@]}"} -- ${EXTRA_ARGS+"${EXTRA_ARGS[@]}"} || pool_status=$?
    fi
    # testament exits 2 when it gave no verdict on the tests: the pool, the machine or the manifest
    # stopped it (coordinator_unreachable, no_runner, connection_lost, batches_lost,
    # env_build_failed, machine_not_set_up, bad_manifest). Say so last, where a long run's reader
    # looks, since the banner above has scrolled away.
    if [ "$pool_status" -eq 2 ]; then
        echo "test.sh: testament gave no verdict (exit 2). If the pool, this machine's key or its testament release stopped the run, docs/testing.md, \"When the pool is down\", says how to go on. An agent stops here and tells the person: no package or whole-suite rerun with TESTAMENT=off." >&2
    fi
    exit "$pool_status"
fi

if [ ${#AREA_ARGS[@]} -gt 0 ]; then
    echo "test.sh: --area and --not-area select on the testament pool, and this run is local; name a package or test paths instead (\`--not-area evals\` is roughly \`--ignore=packages/evals\`)" >&2
    exit 2
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
    if [ -n "$local_only" ] || [ -n "$user_marks" ]; then
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
