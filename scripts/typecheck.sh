#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Run mypy on all package source directories
uv run mypy packages/core/src/ packages/agent-memory/src/ packages/agent-tools/src/ "$@"
