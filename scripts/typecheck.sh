#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/_tool.sh
source "$REPO_ROOT/scripts/_tool.sh"

# Strict mypy over every package, or one package, as testament's `typecheck` suite.
# Usage: ./scripts/typecheck.sh [package] [extra pytest args...]
# Examples:
#   ./scripts/typecheck.sh                  # every package, then the evals example hosts
#   ./scripts/typecheck.sh evals            # threetears.evals, then the evals example hosts
#   ./scripts/typecheck.sh agent/memory     # threetears.agent.memory
#   ./scripts/typecheck.sh core -v
# The package name is the one test.sh takes (a directory under packages/). Arguments after it go to
# pytest, not mypy: the checks are tests (typecheck/test_mypy.py, which says why) and own their mypy flags.
#
# What is checked is read off the tree (typecheck/targets.py): every package's import roots, so a new
# package needs no entry here. tests/enforcement/test_typecheck_targets.py holds the root pyproject's
# [tool.mypy] `files` and `mypy_path`, which a bare `mypy` reads, to the same set.
#
# Where testament is installed, the run goes to its pool, outside CI and unless TESTAMENT=off: the manifest
# is `fallback = "pool-only"`, so a pool that can't take it fails it rather than running it here. The pool
# serves a tree that already passed from its cache, whoever ran it, so a run here and CI's run of the same
# tree check it once. testament keys that cache on the tree and every release tag in this checkout, so a
# run here serves CI only from a clean committed checkout whose `v*` tags match origin's: this script
# fetches origin's tags first and names any tag that differs from origin's (it moves and deletes none).
#
# CI, and TESTAMENT=off with a stated reason (TESTAMENT_OFF_REASON), run the same tests here.

PACKAGE=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    PACKAGE="${1%/}"
    shift
    if [ ! -d "packages/$PACKAGE/src/threetears" ]; then
        echo "typecheck.sh: no package 'packages/$PACKAGE' with a src/threetears tree" >&2
        exit 2
    fi
fi

SELECTION=(typecheck/)
if [ -n "$PACKAGE" ]; then
    SELECTION=("typecheck/test_mypy.py::test_package_typechecks[${PACKAGE//\//.}]")
    if [ "$PACKAGE" = "evals" ]; then
        SELECTION+=("typecheck/test_mypy.py::test_evals_example_hosts_typecheck")
    fi
fi

# The release tags join testament's cache key, by name and the commit each names: fetch origin's, then name
# every tag that differs from origin's (one only here, one only there, or one at another commit). A tag
# origin moved is refused by the fetch ("would clobber existing tag") and stays different until the person
# moves it here; nothing in this script deletes or moves one.
check_tags() {
    local remote here differ
    git fetch --quiet --tags origin 2>/dev/null || true
    if ! remote="$(git ls-remote --tags --refs origin 'v[0-9]*' 2>/dev/null)"; then
        echo "typecheck.sh: could not read origin's tags; if this checkout's differ, CI checks this tree again" >&2
        return 0
    fi
    remote="$(printf '%s\n' "$remote" | sed -nE 's#^([0-9a-f]+)[[:space:]]+refs/tags/(.+)$#\2 \1#p' | sort)"
    here="$(git for-each-ref --format='%(refname:strip=2) %(objectname)' 'refs/tags/v[0-9]*' | sort)"
    differ="$(comm -3 <(printf '%s\n' "$here") <(printf '%s\n' "$remote") | tr -d '\t' | sed '/^$/d' | cut -d' ' -f1 |
        sort -u | paste -sd ' ' -)"
    if [ -n "$differ" ]; then
        echo "typecheck.sh: tags that differ from origin's: ${differ}. testament's cache key holds every v* tag, so CI will not reuse this run's results until they match (\`git fetch --tags --force origin\` takes origin's)" >&2
    fi
}

if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" = "off" ]; then
    if [ -z "${TESTAMENT_OFF_REASON:-}" ]; then
        echo "typecheck.sh: TESTAMENT=off needs a reason: drop it to run on the pool, or say why with TESTAMENT_OFF_REASON=\"...\"" >&2
        exit 2
    fi
    echo "typecheck.sh: running locally, not via testament: ${TESTAMENT_OFF_REASON}" >&2
fi

if command -v testament >/dev/null 2>&1 && [ -z "${CI:-}" ] && [ "${TESTAMENT:-}" != "off" ]; then
    check_tags
    echo "typecheck.sh: via testament, pool only, suite typecheck (if the pool is down: docs/testing.md, \"When the pool is down\")" >&2
    pool_status=0
    testament run --suite typecheck -- "${SELECTION[@]}" "$@" || pool_status=$?
    # as test.sh: exit 75 (from testament 0.46) and 2 are both "no verdict", this script's exit 2
    if [ "$pool_status" -eq 75 ]; then
        echo "typecheck.sh: testament exit 75: the pool could not serve this run." >&2
        pool_status=2
    fi
    if [ "$pool_status" -eq 2 ]; then
        echo "typecheck.sh: testament gave no verdict (exit 2). If the pool, this machine's key or its testament release stopped the run, docs/testing.md, \"When the pool is down\", says how to go on; TESTAMENT=off TESTAMENT_OFF_REASON=\"...\" runs the typecheck here." >&2
    fi
    exit "$pool_status"
fi

run_tool pytest "${SELECTION[@]}" "$@"
