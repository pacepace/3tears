#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/_tool.sh
source "$REPO_ROOT/scripts/_tool.sh"

# Run ruff check and optionally fix
# Usage: ./scripts/lint.sh [--fix]

if [ "${1:-}" = "--fix" ]; then
    run_tool ruff check . --fix
    run_tool ruff format .
else
    run_tool ruff check .
    run_tool ruff format . --check
fi
