#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the MCP-touching tests against mcp 1.x, while uv.lock pins 2.x.
#
# Why this exists: 3tears 0.75 supports BOTH mcp majors (`mcp>=1.27,<3`), so consumers can move
# to 2.x on their own schedule -- McpServer and the CLI pool pick the SDK's API at runtime. The
# workspace suite only ever runs against the locked 2.x, so without this the 1.x path could break
# with everything green, and a consumer still on 1.x would find out at import.
#
# How: overlay the newest mcp 1.x and the fastmcp 3.x that requires it (fastmcp 4 needs mcp 2) on
# the locked environment, and run the tests of every package that touches the MCP SDK. Remove
# this script, its CI step and the 1.x branches it guards when 3tears drops mcp 1.x -- the owner decides when; no release is set.
#
# Usage: ./scripts/test-mcp-v1.sh [extra pytest args...]

MCP_V1="1.30.0"
FASTMCP_V3="3.4.8"

# the evals tests that touch an MCP SDK; the rest of evals never imports one
EVALS_MCP_TESTS=()
while IFS= read -r test_file; do
    EVALS_MCP_TESTS+=("$test_file")
done < <(grep -rlE 'fastmcp|mcp\.' packages/evals/tests --include='test_*.py' | sort)

exec uv run --frozen \
    --with "mcp==${MCP_V1}" --with "fastmcp==${FASTMCP_V3}" --with "fastmcp-slim==${FASTMCP_V3}" \
    pytest packages/mcp packages/models "${EVALS_MCP_TESTS[@]}" \
    -m "not integration" -n 4 --dist loadgroup "$@"
