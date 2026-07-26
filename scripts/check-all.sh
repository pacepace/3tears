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
"$REPO_ROOT/scripts/test.sh"

# The sidecar has its own venv (nodriver is AGPL and never enters the workspace one), so the
# suite above carries --ignore for it. Ruff still formats its source, and mypy still does not
# check it, which left a window where an autofix could write a syntax error that every gate
# reported as clean. This closes it.
echo ""
echo "=== Tests (nodriver sidecar) ==="
"$REPO_ROOT/scripts/test-sidecar.sh"

echo ""
echo "=== All checks passed ==="
