#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run all checks: lint, typecheck, tests, and the dependency-version suites CI's check job runs
echo "=== Lint ==="
"$REPO_ROOT/scripts/lint.sh"

echo ""
echo "=== Typecheck ==="
"$REPO_ROOT/scripts/typecheck.sh"

echo ""
echo "=== Tests ==="
"$REPO_ROOT/scripts/test.sh"

# CI's check job runs this; it stays local because it overlays mcp 1.x on the locked environment.
echo ""
echo "=== Tests (MCP packages on mcp 1.x) ==="
"$REPO_ROOT/scripts/test-mcp-v1.sh" -q

# CI's check job runs this; it stays local because it pins observe's OpenTelemetry floors in an isolated env.
echo ""
echo "=== Tests (observe at its OpenTelemetry floor) ==="
"$REPO_ROOT/scripts/test-otel-floor.sh" -q

# The sidecar has its own venv (nodriver is AGPL and never enters the workspace one), so the
# workspace suite carries --ignore for it. Ruff still formats its source, and mypy still does not
# check it, which left a window where an autofix could write a syntax error that every gate
# reported as clean. This closes it.
echo ""
echo "=== Tests (nodriver sidecar) ==="
"$REPO_ROOT/scripts/test-sidecar.sh"

echo ""
echo "=== All checks passed ==="
