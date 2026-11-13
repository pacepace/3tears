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
# Exit 2 from test.sh means testament gave no verdict: the pool, this machine's key or its testament
# release stopped the run (the manifest is pool-only). The suites below don't use the pool, so they
# still run, and this gate fails at the end saying the unit suite did not run. A failing test (any
# other status) stops the gate here, as before. docs/testing.md, "When the pool is down".
unit_status=0
"$REPO_ROOT/scripts/test.sh" || unit_status=$?
if [ "$unit_status" -ne 0 ] && [ "$unit_status" -ne 2 ]; then
    exit "$unit_status"
fi

# The integration suite, on the pool only. The same rule: exit 2 is no verdict, reported at the end; any
# other failure stops the gate here. Where the pool route is not taken (no testament, CI, TESTAMENT=off),
# test-integration.sh would run the suite here through testcontainers -- about an hour, or, without Docker,
# every test skipped and the gate green -- so the gate does not run it, and fails at the end saying so. A
# local run stays a choice made with test-integration.sh itself. The docker and secrets tests
# (`test-integration.sh --here`) are not part of the gate: they need Docker or a credential, and a change
# that touches what they cover runs them on purpose (CLAUDE.md, "Scripts").
echo ""
echo "=== Tests (integration suite, on the pool) ==="
integration_status=0
integration_skipped=""
if ! command -v testament >/dev/null 2>&1; then
    integration_skipped="testament is not on PATH"
elif [ -n "${CI:-}" ]; then
    integration_skipped="CI is set"
elif [ "${TESTAMENT:-}" = "off" ]; then
    integration_skipped="TESTAMENT=off"
fi
if [ -n "$integration_skipped" ]; then
    echo "not run: ${integration_skipped}" >&2
else
    "$REPO_ROOT/scripts/test-integration.sh" -rs || integration_status=$?
    if [ "$integration_status" -ne 0 ] && [ "$integration_status" -ne 2 ]; then
        exit "$integration_status"
    fi
fi

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

if [ "$unit_status" -eq 2 ] || [ "$integration_status" -eq 2 ] || [ -n "$integration_skipped" ]; then
    echo ""
    if [ "$unit_status" -eq 2 ]; then
        echo "=== Unit suite NOT run: testament gave no verdict (exit 2). ===" >&2
    fi
    if [ "$integration_status" -eq 2 ]; then
        echo "=== Integration suite NOT run: testament gave no verdict (exit 2). ===" >&2
    fi
    if [ -n "$integration_skipped" ]; then
        echo "=== Integration suite NOT run here: ${integration_skipped}; run ./scripts/test-integration.sh -rs where the pool is reachable, or TESTAMENT=off TESTAMENT_OFF_REASON=\"...\" ./scripts/test-integration.sh <package> -rs on a machine with Docker. ===" >&2
    fi
    echo "=== Every other check passed. docs/testing.md, \"When the pool is down\", says what gates a PR meanwhile. ===" >&2
    exit 2
fi

echo ""
echo "=== All checks passed ==="
