#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run all checks: lint, typecheck, tests
echo "=== Lint ==="
"$REPO_ROOT/scripts/lint.sh"

echo ""
echo "=== Typecheck ==="
"$REPO_ROOT/scripts/typecheck.sh"

echo ""
echo "=== Tests ==="
# Honours THREETEARS_TEST_REMOTE (docs/testing.md): only this step can leave the
# box. Lint and typecheck above, and the sidecar suite below, always run here.
WHERE_FILE="$(mktemp)"
trap 'rm -f "$WHERE_FILE"' EXIT
THREETEARS_TEST_WHERE_FILE="$WHERE_FILE" "$REPO_ROOT/scripts/test.sh"
TESTS_WHERE="$(cat "$WHERE_FILE")"

# The sidecar has its own venv (nodriver is AGPL and never enters the workspace one), so the
# suite above carries --ignore for it. Ruff still formats its source, and mypy still does not
# check it, which left a window where an autofix could write a syntax error that every gate
# reported as clean. This closes it.
echo ""
echo "=== Tests (nodriver sidecar) ==="
"$REPO_ROOT/scripts/test-sidecar.sh"

echo ""
echo "=== All checks passed (tests ran ${TESTS_WHERE:-locally}; lint, typecheck and sidecar locally) ==="
