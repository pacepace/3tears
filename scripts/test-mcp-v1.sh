#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run the tests of the packages that still support mcp 1.x against it, while uv.lock pins 2.x.
#
# Why this exists: 3tears-models (`mcp>=1.27,<3`, the CLI pool picks the SDK's API at runtime) and
# 3tears-evals' fastmcp extra (`fastmcp>=3.4.2,<5`; fastmcp 3 needs mcp 1) support BOTH mcp
# majors, so their consumers can move to 2.x on their own schedule. The workspace suite only ever
# runs against the locked 2.x, so without this their 1.x path could break with everything green,
# and a consumer still on 1.x would find out at import. 3tears-mcp requires mcp 2.x and is not run
# here.
#
# How: overlay the newest mcp 1.x and the fastmcp 3.x that requires it (fastmcp 4 needs mcp 2) on
# the locked environment, and run the tests of those packages. Remove this script, its CI step and
# the 1.x branches it guards when models and evals drop mcp 1.x too -- the owner decides when.
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
    pytest packages/models "${EVALS_MCP_TESTS[@]}" \
    -m "not integration" -n 4 --dist loadgroup "$@"
