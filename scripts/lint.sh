#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run ruff check and optionally fix
# Usage: ./scripts/lint.sh [--fix]

if [ "${1:-}" = "--fix" ]; then
    uv run ruff check . --fix
    uv run ruff format .
else
    uv run ruff check .
    uv run ruff format . --check
fi
